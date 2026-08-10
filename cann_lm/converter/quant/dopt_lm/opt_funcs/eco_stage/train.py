# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2024. All rights reserved.
import os
import time
import random
import json
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from dopt_lm.common_tools.log import Logger as logger
from dopt_lm.do_opt import (
    generate_config_file, 
    set_quant_state, 
    set_calibrate_state
)


os.environ["WANDB_MODE"]="offline"
device_type = os.getenv("DEVICE", "cuda")
if device_type  == "npu": 
    import torch_npu


def check_is_trainable(name, trainable_keys): 
    for k in trainable_keys: 
        if k in name: 
            return True
    return False


def get_processed_dataset(dataset_files, tokenizer, dataset_config, train_samples=None): 
    cutoff_len = dataset_config["cutoff_len"]
    text_concat = True

    def encode_function(examples): 
        tokened = tokenizer(examples["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=cutoff_len)
        return tokened
    
    def concat_fun(dataset): 
        text = "".join(dataset["train"]["text"])
        result = tokenizer(text, return_tensors=None)
        nsamples = len(result["input_ids"]) // cutoff_len
        data = {
            "input_ids": list(),
        }
        for i in tqdm(range(nsamples)): 
            data["input_ids"].append(result["input_ids"][i * cutoff_len : (i + 1) * cutoff_len])
        if train_samples: 
            data["input_ids"] = data["input_ids"][:train_samples]
        train_dataset = Dataset.from_dict(data)
        return train_dataset
    if not isinstance(dataset_files, list): 
        dataset_files = [dataset_files, ]
    dataset_list = []
    for dataset_file in dataset_files: 
        if dataset_file == "wikitext2": 
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        if dataset_file.endswith(".json") or dataset_file.endswith(".jsonl"): 
            dataset = load_dataset("json", data_files=dataset_file)
        dataset_list.append(dataset["train"])

    dataset = DatasetDict(
        {
            "train": concatenate_datasets(dataset_list),
        }
    )
    if text_concat: 
        return concat_fun(dataset)
    else: 
        dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
        if train_samples: 
            dataset["train"] = dataset["train"].select(range(train_samples))
        train_dataset = dataset["train"].map(encode_function, batched=True)
        return train_dataset
    return None


if device_type == "npu": 
    from torch.npu.amp import GradScaler, autocast
else: 
    from torch.cuda.amp import GradScaler, autocast


def custom_model_generation(model, no_split_module_classes=None): 
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    if no_split_module_classes is None: 
        no_split_module_classes = list()

    skip_keys = getattr(model, "_skip_keys_device_placement", None)

    max_memory = get_balanced_memory(
        model, 
        max_memory=None, 
        no_split_module_classes=no_split_module_classes, 
        dtype="float16", 
        low_zero=False, 
    )
    device_map = infer_auto_device_map(
        model, 
        max_memory=max_memory, 
        no_split_module_classes=no_split_module_classes, 
        dtype="float16", 
    )
    dispatched_model = dispatch_model(model, device_map=device_map, skip_keys=skip_keys)
    return dispatched_model


class DoptTrainerCustomization(torch.nn.Module): 
    def __init__(
        self, 
        model, 
        teacher, 
        kd_config=None, 
        args=None, 
        **argv
    ): 
        super().__init__()
        if kd_config is not None: 
            self.kd_config = kd_config
        else: 
            self.kd_config = dict()
        if "mse" in self.kd_config.get("loss", ""): 
            logger.info("using mse loss")
        self.criterion = torch.nn.MSELoss()
        self.training_args = args
        self.model = model
        self.teacher = teacher
        self.argv = argv

    def train(self): 
        self.training_loop_block()

    def grad_norm(self): 
        total_norm = 0.0
        for param in self.model.parameters(): 
            if param.grad is not None: 
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def training_loop_block(self): 
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
        epochs = self.training_args.num_train_epochs
        params_with_grad = [param for name, param in self.model.named_parameters() if param.requires_grad]
        optimizer = AdamW(
            params_with_grad, 
            lr=self.training_args.learning_rate, 
            weight_decay=self.training_args.weight_decay
        )
        data_train_sets = self.argv.get('train_dataset')
        t_max = epochs * len(data_train_sets) // (self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps)

        lr_scheduler_type = self.training_args.lr_scheduler_type if hasattr(self.training_args, 'lr_scheduler_type') else 'linear'
        if lr_scheduler_type == 'cosine': 
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps= self.training_args.warmup_steps, 
                num_training_steps=t_max
            )
        else: 
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.training_args.warmup_steps, 
                num_training_steps=t_max
            )
        logging_step = int(self.training_args.logging_steps)

        train_dataloader = DataLoader(data_train_sets, batch_size=self.training_args.per_device_train_batch_size, shuffle=True)
        from accelerate import Accelerator
        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(
            self.model
        )
        self.teacher = self.accelerator.prepare(
            self.teacher
        )
        skip_keys = ["past_key_values", "shared_kv_states"]
        if hasattr(self.model, '_hf_hook'): 
            self.model._hf_hook.skip_keys = skip_keys
        if hasattr(self.teacher, '_hf_hook'): 
            self.teacher._hf_hook.skip_keys = skip_keys
        for epoch in range(epochs): 
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}", leave=True) as pbar: 
                for step, batch in enumerate(train_dataloader): 
                    input_ids = torch.stack(batch.get("input_ids"), dim=0).permute(1,0).to(device_type)
                    batch = {
                        "input_ids": input_ids,
                    }
                    with autocast(): 
                        loss = self.compute_loss(self.model, batch)
                    loss = loss / self.training_args.gradient_accumulation_steps
                    loss.backward()
                    if (step + 1) % self.training_args.gradient_accumulation_steps == 0: 
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    if step % logging_step == 0: 
                        pbar.set_postfix_str(f"grad_norm {self.grad_norm()}, Loss {loss.item() * self.training_args.gradient_accumulation_steps:.4f}, lr {optimizer.param_groups[0]['lr']}")
                    pbar.update(1)

    def compute_loss(self, model, inputs, return_outputs=False): 
        teacher = self.teacher
        if teacher is not None: 
            outputs = model(**{**inputs, "labels": None, "output_hidden_states": True})
            logits = outputs["logits"]
            with torch.no_grad(): 
                teacher_outputs = teacher(**{**inputs, "labels": None, "output_hidden_states": True})

            loss_list = list()
            if "mse" in self.kd_config.get("loss", ""): 
                loss_func = torch.nn.MSELoss()
                loss_layers = self.kd_config.get("feature_layers", [-1])
                for layer in loss_layers: 
                    stu_res = outputs.hidden_states[layer].to(torch.float32)
                    tea_res = teacher_outputs.hidden_states[layer].to(torch.float32)
                    loss_list.append(
                        loss_func(
                            stu_res, 
                            tea_res, 
                        )
                    )
            if "kl_div" in self.kd_config.get("loss", ""): 
                loss_list.append(
                    ce_loss(logits, teacher_outputs.logits, topk=self.kd_config.get("topk", -1), 
                    reduction=self.kd_config.get("reduction", "batchmean"))
                )
            if not loss_list: 
                # No explicit loss configured - default to MSE on last hidden state
                loss_func = torch.nn.MSELoss()
                stu_res = outputs.hidden_states[-1].to(torch.float32)
                tea_res = teacher_outputs.hidden_states[-1].to(torch.float32)
                loss_list.append(loss_func(stu_res, tea_res))
            loss = sum(loss_list) / len(loss_list)
            return (loss, outputs) if return_outputs else loss
        return None
    

def ce_loss(student_logits, teacher_logits, topk=-1, reduction="batchmean"): 
    def get_topk_logits(logits, topk=-1): 
        if topk > 0: 
            dim = len(logits.shape) - 1
            vals, ind = torch.topk(logits, topk, dim=dim)
            masked_score = torch.zeros_like(logits)
            masked_score.scatter_(dim, ind, vals)
            del logits
            logits = masked_score
        return logits
    
    teacher_logits = get_topk_logits(teacher_logits, topk)
    model_output_log_prob = F.log_softmax(student_logits, dim=2)
    real_output_soft = F.softmax(teacher_logits, dim=2).to(student_logits.device)
    loss = F.kl_div(model_output_log_prob, real_output_soft, reduction=reduction)
    return loss


def seed_all(seed_value=1234, deterministic=False): 
    logger.info(f"setting seed {seed_value}")
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars

    if device_type == "cuda": 
        if torch.cuda.is_available(): 
            logger.info("CUDA is available")
            torch.use_deterministic_algorithms(deterministic)
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # gpu vars
            torch.backends.cudnn.deterministic = True # needed
    elif device_type == "npu": 
        if torch.npu.is_available(): 
            logger.info("CUDA is available")
            torch.use_deterministic_algorithms(deterministic)
            torch.npu.manual_seed(seed_value)
            torch.npu.manual_seed_all(seed_value) # gpu vars
            torch.backends.cudnn.deterministic = True # needed


def model_define(model_path, torch_dtype=torch.float16): 
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch_dtype, 
        trust_remote_code=True, 
    )
    return model


def generate_config(model_name_or_path, config_path): 
    model = model_define(model_name_or_path)
    generate_config_file(model, config_path)


def prepare_model(
    model, 
    resume_ckpt = None, 
    quant_config = None, 
    config = None, 
    requires_grad=False, 
    trainable_keys=None, 
    refresh=False, 
    device=None
): 
    if isinstance(model, str): 
        fp16_flag = config.get('extra_training_config', dict()).get('fp16', False)
        torch_dtype = torch.float16 if fp16_flag else torch.float32
        model = model_define(model, torch_dtype=torch_dtype)
        if device: 
            model = model.to(device)
    if quant_config: 
        from dopt_lm.do_opt import optimize_model
        model = optimize_model(model, quant_config)

    if resume_ckpt: 
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        load_ckpt = ckpt
        if refresh: 
            optimize_model_ckpt = model.state_dict()
            for layer_key in ckpt.keys(): 
                optimize_model_ckpt[layer_key] = ckpt.get(layer_key)
            load_ckpt = optimize_model_ckpt
            del ckpt
        model.load_state_dict(load_ckpt, strict=False)

    if trainable_keys is None: 
        trainable_keys = list()
    if not isinstance(trainable_keys, (list, tuple)): 
        trainable_keys = [trainable_keys]

    if "default" in trainable_keys: 
        model = model.to(torch.float32)
        for name, p in model.named_parameters(): 
            p.requires_grad = requires_grad
            logger.info(f"setting {name}, requires_grad {p.requires_grad}")
    else: 
        for name, p in model.named_parameters():
            p.requires_grad = False
            if check_is_trainable(name, trainable_keys): 
                p.requires_grad = requires_grad
                if 'norm' in name: 
                    p.data = p.data.to(torch.float32)
                logger.info(f"setting {name}, requires_grad {p.requires_grad}, dtype {p.dtype}")
    return model


def print_trainable_parameters(model): 
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters(): 
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad: 
            trainable_params += num_params
    trainable_params = (trainable_params / total_params) * 100
    logger.info(f"total trainedable params:{trainable_params} %")


def train(
    base_model, 
    config,
    output_dir=None, 
    hf_type_save=False, 
    **argv
): 
    logger.info("loading student model")
    model = prepare_model(
        base_model, 
        config.get("quant_pth"), 
        config.get("config_path"), 
        config, 
        requires_grad=True, 
        trainable_keys=config['kd'].get("trainable_keys", "quant_alpha")
    )
    if config['kd'].get('mode') == 'gptq_finetune': 
        model.load_state_dict(torch.load(os.path.join(output_dir, "trained_quant_weight.pth")))
    if config['kd'].get('mode') == 'resume_finetune': 
        model.load_state_dict(torch.load(os.path.join(output_dir, "trained_quant_weight_fromgptq.pth")))
    model.train()
    no_split_module_classes = config['kd'].get('no_split_module_classes', None)
    model = custom_model_generation(model, no_split_module_classes=no_split_module_classes)
    set_quant_state(model, weight_state=True, input_state=False)
    set_calibrate_state(model, True)
    print_trainable_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0
    )
    logger.info(model)
    val_set_size = 0
    train_file = config["dataset"]["train_files"]
    config["dataset"]["cutoff_len"] = config.get("cutoff_len", 2048)
    val_data = None
    train_data = get_processed_dataset(
        train_file, 
        tokenizer, 
        config["dataset"], 
        config["dataset"].get("ptq_samples", None)
    )

    kd_config = None
    if config.get("kd", None) and config["kd"].get("enable", False): 
        kd_config = config["kd"]
        logger.info("loading teacher model")
        teacher = prepare_model(
            kd_config.get('teacher_model', base_model), 
            kd_config.get('resume_ckpt'), 
            config.get("teacher_config_path"), 
            config=config, 
            requires_grad=False
        )
        teacher.config.use_cache = False
        teacher.eval()
        teacher = custom_model_generation(teacher, no_split_module_classes=no_split_module_classes)

    extra_training_args = config.get("extra_training_config", dict())
    if extra_training_args.get("gradient_checkpointing", False): 
        model.gradient_checkpointing_enable()
        logger.info("gradient checkpointing enabled")
    if "max_steps" in config["dataset"]: 
        extra_training_args["max_steps"] = config["dataset"]["max_steps"]
    trainer = DoptTrainerCustomization(
        model=model, 
        teacher=teacher,
        kd_config=kd_config, 
        train_dataset=train_data, 
        eval_dataset=val_data, 
        args=transformers.TrainingArguments(
            output_dir=output_dir, 
            per_device_train_batch_size=kd_config["micro_batch_size"], 
            per_device_eval_batch_size=1, 
            gradient_accumulation_steps=kd_config["gradient_accumulation_steps"], 
            warmup_steps=kd_config["warmup_steps"], 
            num_train_epochs=kd_config["num_epochs"], 
            learning_rate=kd_config["learning_rate"],
            lr_scheduler_type=kd_config["lr_scheduler_type"], 
            weight_decay=kd_config["weight_decay"], 
            full_determinism=True, 
            data_seed=1234, 
            seed=1234, 
            logging_steps=kd_config.get("logging_step", 10), 
            eval_strategy="steps" if val_set_size > 0 else "no", 
            save_strategy="no", 
            eval_steps=kd_config["eval_step"] if val_set_size > 0 else None, 
            load_best_model_at_end=False, 
            report_to=None, 
            run_name=None, 
            **extra_training_args, 
        ), 
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", 
            padding=False
        ), 
    )
    model.config.use_cache = False
    st = time.time()
    trainer.train()
    if not hf_type_save: 
        ### input and weight quant enable and lora enable
        model.teacher = None
        torch.save(model.state_dict(), os.path.join(output_dir, "trained_quant_weight.pth"))
    else: 
        model.save_pretrained(output_dir)
    logger.info(f"total cost time {time.time() - st}")
    model = None
    del model


def weight_q_params_adapter_into_dopt(dopt_model, weight_q_quantizers, tie_lmhead): 
    for name, module in dopt_model.named_modules(): 
        if hasattr(module, "quant_op"): 
            weight_q_q = weight_q_quantizers.get(name, None)
            if weight_q_q is None: 
                logger.info(f'{name} not find params in PTQ output')
                continue
            module.quant_op.weight_quantizer.s = weight_q_q.group_scale.flatten().to(module.quant_op.weight_quantizer.s.dtype)
            module.quant_op.weight_quantizer.offset = torch.zeros_like(module.quant_op.weight_quantizer.offset)
    if hasattr(dopt_model.model, 'embed_tokens') and hasattr(dopt_model.model.embed_tokens, "quant_op"):
        if not tie_lmhead:
            weight = dopt_model.model.embed_tokens.weight.data
            dopt_model.model.embed_tokens.weight.data.copy_(dopt_model.model.embed_tokens.quant_op.weight_quantizer(weight).to(weight.dtype))
        elif hasattr(dopt_model.lm_head, 'quant_op'):
            dopt_model.model.embed_tokens.quant_op.weight_quantizer = dopt_model.lm_head.quant_op.weight_quantizer
        else:
            weight = dopt_model.model.embed_tokens.weight.data
            dopt_model.model.embed_tokens.weight.data.copy_(dopt_model.model.embed_tokens.quant_op.weight_quantizer(weight).to(weight.dtype))


def weight_quant_pipline(
    model_path, 
    config_yaml, 
    dopt_config=None, 
    output_dir=None, 
    hf_type_save=False, 
    **argv
): 
    with open(config_yaml, "r") as f: 
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config['config_path'] = dopt_config
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.yaml", 'w') as file: 
        yaml.dump(config,file)
    logger.info(f"using {config_yaml}, save to {output_dir}")
    base_model = model_path
    if base_model is None: 
        raise Exception(f"Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'")
    seed_all()

    if "kd" in config and config['kd'].get('enable', False) is True: 
        train(model_path, config, output_dir, hf_type_save, **argv)
    else: 
        quant_method = 'gptq'
        ptq_quant_pipline(model_path, config, output_dir, hf_type_save, quant_method, **argv)
    logger.info(f'weight quant done!!!')


def ptq_quant_pipline(
    base_model, 
    config, 
    output_dir=None, 
    hf_type_save=False, 
    quant_method='gptq', 
    **argv
): 
    train_file = config["dataset"]["train_files"]
    config["dataset"]["cutoff_len"] = config.get("cutoff_len", 2048)
    if quant_method == 'gptq': 
        from dopt_lm.weight_quant_ptq.ptq_algo_heissen.ptq_loop import weight_quant_loop
    st = time.time()
    nsamples = config["dataset"].get("ptq_samples", 256)
    fp16_flag = config.get('extra_training_config', dict()).get('fp16', False)
    torch_dtype = torch.float16 if fp16_flag else torch.float32
    weight_q_quantizer, weight_q_weight = weight_quant_loop(
        base_model, 
        data_set=train_file, 
        nsamples=nsamples, 
        seed=1234, 
        seq_len=config.get("cutoff_len", 2048), 
        quant_config=config.get("config_path"), 
        torch_dtype=torch_dtype
    )
    dopt_model = prepare_model(
        base_model, 
        config.get("quant_pth"), 
        config.get("config_path"), 
        config, 
        requires_grad=True, 
        trainable_keys=config.get("trainable_keys", "lora")
    )
    set_quant_state(dopt_model, weight_state=True, input_state=False)
    set_calibrate_state(dopt_model, True)
    tie_lmhead = getattr(dopt_model.model.embed_tokens, "tie_lmhead", None)
    if tie_lmhead is None: 
        tie_lmhead = getattr(dopt_model.config, "tie_word_embeddings", False)
    if not tie_lmhead: 
        weight_q_weight.pop('model.embed_tokens.weight')
    dopt_model.load_state_dict(weight_q_weight, strict=False)
    weight_q_params_adapter_into_dopt(dopt_model, weight_q_quantizer, tie_lmhead)
    set_calibrate_state(dopt_model, False)
    if not hf_type_save: 
        torch.save(dopt_model.state_dict(), os.path.join(output_dir, "trained_quant_weight.pth"))
    else: 
        dopt_model.save_pretrained(output_dir)
    logger.info(f"total cost time {time.time() - st}")
    dopt_model = None
    weight_q_quantizer = None
    del dopt_model
    del weight_q_quantizer
