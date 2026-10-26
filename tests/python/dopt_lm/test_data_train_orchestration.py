import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from dopt_lm.opt_funcs.eco_stage import train as train_module
from dopt_lm.opt_funcs.weight_quant_ptq.ptq_algo_hessian import datautils, ptq_loop


def test_data_seed_makes_numpy_and_torch_repeatable():
    datautils.set_seed(23)
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)
    datautils.set_seed(23)
    np.testing.assert_allclose(np.random.rand(3), first_numpy)
    torch.testing.assert_close(torch.rand(3), first_torch)


def test_get_pt_data_slices_shuffles_and_combines(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first_values = [torch.tensor([float(i)]) for i in range(6)]
    second_values = [torch.tensor([float(10 + i)]) for i in range(6)]
    torch.save({"input_embed": first_values, "input_mask": list(range(6))}, first)
    torch.save(
        {"input_embed": second_values, "input_position_id": list(range(6))}, second
    )

    combined, validation = datautils.get_pt_data(
        [f"{first}:0.5", f"{second}:0.25"], nsamples=8, seed=7
    )

    assert validation is None
    assert len(combined["input_embed"]) == 6
    values = sorted(int(item.item()) for item in combined["input_embed"])
    assert values[:4] == [0, 1, 2, 3]
    assert values[4:] == [10, 11]


def test_get_loaders_dispatches_pt_and_json(monkeypatch):
    calls = []
    monkeypatch.setattr(
        datautils,
        "get_pt_data",
        lambda *args: calls.append(("pt", args)) or ("pt-result", None),
    )
    monkeypatch.setattr(
        datautils,
        "get_json_data",
        lambda *args: calls.append(("json", args)) or ("json-result", None),
    )

    assert datautils.get_loaders(["sample.pt:1.0"], 3, 4, 5, "model")[0] == "pt-result"
    assert (
        datautils.get_loaders(["sample.json:1.0"], 3, 4, 5, "model")[0] == "json-result"
    )
    assert [kind for kind, _ in calls] == ["pt", "json"]


@pytest.mark.parametrize(
    "token_count,nsamples,seqlen",
    [
        (20, 3, 4),
        (6, 2, 4),
    ],
)
def test_get_json_data_builds_long_and_short_token_chunks(
    monkeypatch, token_count, nsamples, seqlen
):
    import datasets
    import transformers

    tokenizer_calls = []

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None):
            tokenizer_calls.append((text, return_tensors))
            return SimpleNamespace(input_ids=torch.arange(token_count).unsqueeze(0))

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            tokenizer_calls.append((args, kwargs))
            return FakeTokenizer()

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: {"train": [{"text": "alpha"}, {"text": "beta"}]},
    )

    loader, validation = datautils.get_json_data(
        nsamples,
        seed=9,
        seqlen=seqlen,
        model="local-model",
        local_root=["calibration.json:1.0", "unused.json:0.0"],
    )

    assert validation is None
    assert len(loader) == nsamples
    for inputs, targets in loader:
        assert inputs.shape == (1, seqlen)
        assert targets.shape == inputs.shape
        assert torch.all(targets[:, :-1] == -100)
        torch.testing.assert_close(targets[:, -1], inputs[:, -1])
    assert tokenizer_calls


def test_get_processed_dataset_concatenates_sources_and_limits_samples(monkeypatch):
    load_calls = []

    def fake_load_dataset(*args, **kwargs):
        load_calls.append((args, kwargs))
        marker = "wiki" if args[0] == "wikitext" else "json"
        return train_module.DatasetDict(
            {
                "train": train_module.Dataset.from_dict(
                    {"text": [f"{marker}-one", f"{marker}-two"]}
                )
            }
        )

    class Tokenizer:
        def __call__(self, text, return_tensors=None, **kwargs):
            return {"input_ids": list(range(len(text)))}

    monkeypatch.setattr(train_module, "load_dataset", fake_load_dataset)

    result = train_module.get_processed_dataset(
        ["wikitext2", "calibration.jsonl"],
        Tokenizer(),
        {"cutoff_len": 4},
        train_samples=3,
    )

    assert len(result) == 3
    assert all(len(row["input_ids"]) == 4 for row in result)
    assert load_calls[0][0] == ("wikitext", "wikitext-2-raw-v1")
    assert load_calls[1] == (("json",), {"data_files": "calibration.jsonl"})


def test_get_processed_dataset_normalizes_single_filename(monkeypatch):
    monkeypatch.setattr(
        train_module,
        "load_dataset",
        lambda *args, **kwargs: train_module.DatasetDict(
            {"train": train_module.Dataset.from_dict({"text": ["abcdefgh"]})}
        ),
    )

    class Tokenizer:
        def __call__(self, text, return_tensors=None, **kwargs):
            return {"input_ids": list(range(len(text)))}

    result = train_module.get_processed_dataset(
        "one.json", Tokenizer(), {"cutoff_len": 4}
    )
    assert len(result) == 2


def test_custom_model_generation_passes_cpu_dispatch_contract(monkeypatch):
    pytest.importorskip(
        "accelerate"
    )  # 缺 accelerate 时跳过 (非 dopt 核心依赖, 系统环境可选)
    import accelerate
    import accelerate.utils

    calls = []
    model = SimpleNamespace(_skip_keys_device_placement=["past_key_values"])
    dispatched = object()

    monkeypatch.setattr(
        accelerate.utils,
        "get_balanced_memory",
        lambda *args, **kwargs: (
            calls.append(("memory", args, kwargs)) or {"cpu": "1GiB"}
        ),
    )
    monkeypatch.setattr(
        accelerate,
        "infer_auto_device_map",
        lambda *args, **kwargs: calls.append(("map", args, kwargs)) or {"": "cpu"},
    )
    monkeypatch.setattr(
        accelerate,
        "dispatch_model",
        lambda *args, **kwargs: calls.append(("dispatch", args, kwargs)) or dispatched,
    )

    result = train_module.custom_model_generation(model, ["DecoderLayer"])

    assert result is dispatched
    assert calls[0][2]["dtype"] == "float32"
    assert calls[0][2]["no_split_module_classes"] == ["DecoderLayer"]
    assert calls[1][2]["max_memory"] == {"cpu": "1GiB"}
    assert calls[2][2]["device_map"] == {"": "cpu"}
    assert calls[2][2]["skip_keys"] == ["past_key_values"]


@pytest.mark.parametrize(
    "name,keys,expected",
    [
        ("layer.quant_alpha", ["quant_alpha"], True),
        ("layer.weight", ["alpha", "lora"], False),
        ("x", [], False),
    ],
)
def test_check_is_trainable(name, keys, expected):
    assert train_module.check_is_trainable(name, keys) is expected


def test_ce_loss_matches_reference_and_topk_is_finite():
    student = torch.tensor([[[1.0, 0.0, -1.0], [0.5, 1.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]]])
    expected = F.kl_div(
        F.log_softmax(student, dim=2),
        F.softmax(teacher, dim=2),
        reduction="batchmean",
    )

    torch.testing.assert_close(train_module.ce_loss(student, teacher), expected)
    topk = train_module.ce_loss(student, teacher, topk=1, reduction="sum")
    assert topk.ndim == 0
    assert torch.isfinite(topk)


class ModelOutput(dict):
    def __init__(self, logits, hidden_states):
        super().__init__(logits=logits, hidden_states=hidden_states)
        self.logits = logits
        self.hidden_states = hidden_states


class LossModel(torch.nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(float(shift)))

    def forward(self, input_ids, **kwargs):
        base = input_ids.float().unsqueeze(-1)
        hidden = base + self.shift
        logits = torch.cat((hidden, hidden + 1, hidden - 1), dim=-1)
        return ModelOutput(logits, (hidden * 0.5, hidden))


@pytest.mark.parametrize(
    "loss_config",
    [
        {"loss": "mse", "feature_layers": [-1]},
        {"loss": "kl_div"},
        {"loss": "mse+kl_div"},
        {},
    ],
)
def test_trainer_compute_loss_modes_and_return_outputs(loss_config):
    student = LossModel(0.25)
    teacher = LossModel(1.0)
    trainer = train_module.DoptTrainerCustomization(
        student, teacher, kd_config=loss_config
    )
    inputs = {"input_ids": torch.tensor([[1, 2]])}

    loss, outputs = trainer.compute_loss(student, inputs, return_outputs=True)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert isinstance(outputs, ModelOutput)


def test_trainer_without_teacher_returns_none_and_grad_norm():
    model = LossModel(0.0)
    trainer = train_module.DoptTrainerCustomization(model, None)
    assert trainer.compute_loss(model, {"input_ids": torch.tensor([[1]])}) is None
    assert trainer.grad_norm() == 0.0

    (model.shift * 3).backward()
    assert trainer.grad_norm() == pytest.approx(3.0)


def test_seed_all_cpu_is_repeatable():
    train_module.seed_all(31, deterministic=True)
    first = (random.random(), np.random.rand(), torch.rand(1))
    train_module.seed_all(31, deterministic=True)
    second = (random.random(), np.random.rand(), torch.rand(1))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


def test_model_define_returns_plain_causal_model(monkeypatch):
    plain_model = SimpleNamespace(model=SimpleNamespace())
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return plain_model

    monkeypatch.setattr(train_module, "AutoModelForCausalLM", FakeAutoModel)

    result = train_module.model_define("plain-model", torch_dtype=torch.float32)

    assert result is plain_model
    assert calls == [
        (
            ("plain-model",),
            {"torch_dtype": torch.float32, "trust_remote_code": True},
        )
    ]


@pytest.mark.parametrize("tied,use_text_config", [(True, True), (False, False)])
def test_model_define_extracts_language_model_and_head(
    monkeypatch, tied, use_text_config
):
    text_config = SimpleNamespace(tie_word_embeddings=tied)

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(5, 3)
            self.config = text_config

    class LlmShell(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()
            self.lm_head = torch.nn.Linear(3, 5, bias=False)

    language_model = LanguageModel()
    outer = SimpleNamespace(
        model=SimpleNamespace(language_model=language_model),
        config=SimpleNamespace(text_config=text_config if use_text_config else None),
        lm_head=torch.nn.Linear(3, 5, bias=False),
    )
    shell = LlmShell()
    expected_untied = outer.lm_head.weight.detach().clone()
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return outer

        @classmethod
        def from_config(cls, config, **kwargs):
            calls.append((config, kwargs))
            return shell

    monkeypatch.setattr(train_module, "AutoModelForCausalLM", FakeAutoModel)

    result = train_module.model_define("multimodal-model")

    assert result is shell
    assert result.model is language_model
    assert result.training is False
    assert calls == [(text_config, {"trust_remote_code": True})]
    if tied:
        assert result.lm_head.weight is language_model.embed_tokens.weight
    else:
        torch.testing.assert_close(result.lm_head.weight, expected_untied)
        assert result.lm_head.weight is not outer.lm_head.weight


def test_generate_config_delegates_model_and_path(monkeypatch):
    model = object()
    calls = []
    monkeypatch.setattr(
        train_module,
        "model_define",
        lambda path: calls.append(("model", path)) or model,
    )
    monkeypatch.setattr(
        train_module,
        "generate_config_file",
        lambda value, path: calls.append(("config", value, path)),
    )

    train_module.generate_config("model-path", "layers.json")

    assert calls == [
        ("model", "model-path"),
        ("config", model, "layers.json"),
    ]


def test_prepare_model_selects_trainable_parameters_and_uses_optimizer_wrapper(
    tmp_path, monkeypatch
):
    class PrepareToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.norm = torch.nn.LayerNorm(2)

    model = PrepareToy()
    checkpoint = tmp_path / "checkpoint.pt"
    state = {
        key: torch.full_like(value, 2) for key, value in model.state_dict().items()
    }
    torch.save(state, checkpoint)
    optimized = []
    monkeypatch.setattr(
        "dopt_lm.do_opt.optimize_model",
        lambda value, config: optimized.append(config) or value,
    )

    result = train_module.prepare_model(
        model,
        resume_ckpt=str(checkpoint),
        quant_config="quant.json",
        config={},
        requires_grad=True,
        trainable_keys=["norm"],
    )

    assert result is model
    assert optimized == ["quant.json"]
    assert all(
        torch.equal(value, torch.full_like(value, 2))
        for value in model.state_dict().values()
    )
    assert model.norm.weight.requires_grad
    assert model.norm.bias.requires_grad
    assert not model.proj.weight.requires_grad


def test_prepare_model_default_sets_all_parameters_trainable():
    model = torch.nn.Linear(2, 2).to(torch.float16)
    result = train_module.prepare_model(
        model,
        config={},
        requires_grad=True,
        trainable_keys="default",
    )
    assert all(parameter.requires_grad for parameter in result.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in result.parameters())


def test_prepare_model_string_refresh_checkpoint_and_device(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    original_bias = model.bias.detach().clone()
    checkpoint = tmp_path / "partial.pt"
    torch.save({"weight": torch.full_like(model.weight, 7.0)}, checkpoint)
    calls = []

    def fake_model_define(path, torch_dtype):
        calls.append((path, torch_dtype))
        return model

    monkeypatch.setattr(train_module, "model_define", fake_model_define)

    result = train_module.prepare_model(
        "model-path",
        resume_ckpt=str(checkpoint),
        config={"extra_training_config": {"fp16": True}},
        refresh=True,
        device="cpu",
        trainable_keys=None,
    )

    assert result is model
    assert calls == [("model-path", torch.float16)]
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 7.0))
    torch.testing.assert_close(model.bias, original_bias)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_print_trainable_parameters_reports_percentage(monkeypatch):
    model = torch.nn.Linear(2, 2)
    model.weight.requires_grad = True
    model.bias.requires_grad = False
    messages = []
    monkeypatch.setattr(train_module.logger, "info", messages.append)

    train_module.print_trainable_parameters(model)

    assert messages == ["total trainedable params:66.66666666666666 %"]


class AdapterQuantizer(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("s", torch.ones(rows))
        self.register_buffer("offset", torch.ones(rows))
        self.do_calibrate = False

    def forward(self, weight):
        return weight * 0.5


class QuantizedHolder(torch.nn.Module):
    def __init__(self, rows=2, cols=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(rows, cols))
        self.quant_op = torch.nn.Module()
        self.quant_op.weight_quantizer = AdapterQuantizer(rows)


class AdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = QuantizedHolder()
        self.lm_head = QuantizedHolder()
        self.other = QuantizedHolder()


def test_weight_adapter_copies_scales_quantizes_and_ties_embedding():
    model = AdapterModel()
    quantizers = {
        "model.embed_tokens": SimpleNamespace(group_scale=torch.tensor([[2.0], [3.0]])),
        "lm_head": SimpleNamespace(group_scale=torch.tensor([[4.0], [5.0]])),
        "other": SimpleNamespace(group_scale=torch.tensor([[6.0], [7.0]])),
    }

    train_module.weight_q_params_adapter_into_dopt(model, quantizers, tie_lmhead=True)

    assert (
        model.model.embed_tokens.quant_op.weight_quantizer
        is model.lm_head.quant_op.weight_quantizer
    )
    assert model.model.embed_tokens.quant_op.weight_quantizer.do_calibrate is False
    assert model.other.quant_op.weight_quantizer.do_calibrate is False
    assert torch.count_nonzero(model.other.quant_op.weight_quantizer.offset) == 0
    torch.testing.assert_close(
        model.other.weight, torch.full_like(model.other.weight, 0.5)
    )


class TrainingOrchestrationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_enabled = False
        self.saved_pretrained = None

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def save_pretrained(self, output_dir):
        self.saved_pretrained = output_dir


@pytest.mark.parametrize(
    "hf_save,mode,checkpoint_name",
    [
        (False, "gptq_finetune", "trained_quant_weight.pth"),
        (True, "resume_finetune", "trained_quant_weight_fromgptq.pth"),
    ],
)
def test_train_orchestrates_kd_resume_and_save_paths(
    tmp_path, monkeypatch, hf_save, mode, checkpoint_name
):
    student = TrainingOrchestrationModel()
    teacher = TrainingOrchestrationModel()
    prepare_calls = []
    state_calls = []
    trainer_calls = []
    load_calls = []
    save_calls = []

    def fake_prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return student if kwargs["requires_grad"] else teacher

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(padding_side=None, pad_token_id=None)

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_calls.append(("init", kwargs))

        def train(self):
            trainer_calls.append(("train",))

    monkeypatch.setattr(train_module, "prepare_model", fake_prepare)
    monkeypatch.setattr(
        train_module,
        "custom_model_generation",
        lambda model, **kwargs: model,
    )
    monkeypatch.setattr(
        train_module,
        "set_quant_state",
        lambda *args, **kwargs: state_calls.append(("quant", kwargs)),
    )
    monkeypatch.setattr(
        train_module,
        "set_calibrate_state",
        lambda *args: state_calls.append(("calibrate", args[1])),
    )
    monkeypatch.setattr(train_module, "print_trainable_parameters", lambda model: None)
    monkeypatch.setattr(train_module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        train_module,
        "get_processed_dataset",
        lambda *args, **kwargs: [{"input_ids": [1, 2]}],
    )
    monkeypatch.setattr(train_module, "DoptTrainerCustomization", FakeTrainer)
    monkeypatch.setattr(
        train_module.transformers,
        "TrainingArguments",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        train_module.transformers,
        "DataCollatorForSeq2Seq",
        lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, *args, **kwargs: (
            load_calls.append(str(path)) or student.state_dict()
        ),
    )
    monkeypatch.setattr(
        torch,
        "save",
        lambda value, path: save_calls.append((value, str(path))),
    )
    config = {
        "config_path": "layers.json",
        "cutoff_len": 16,
        "dataset": {
            "train_files": ["calibration.json"],
            "ptq_samples": 2,
            "max_steps": 1,
        },
        "extra_training_config": {"gradient_checkpointing": True},
        "kd": {
            "enable": True,
            "mode": mode,
            "trainable_keys": ["quant_alpha"],
            "no_split_module_classes": ["DecoderLayer"],
            "teacher_model": "teacher-path",
            "resume_ckpt": "teacher.pt",
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 0,
            "num_epochs": 1,
            "learning_rate": 1.0e-4,
            "lr_scheduler_type": "linear",
            "weight_decay": 0.0,
            "logging_step": 1,
            "eval_step": 1,
        },
    }

    train_module.train(
        "student-path",
        config,
        output_dir=str(tmp_path),
        hf_type_save=hf_save,
    )

    assert len(prepare_calls) == 2
    assert prepare_calls[0][1]["requires_grad"] is True
    assert prepare_calls[1][1]["requires_grad"] is False
    assert load_calls == [str(tmp_path / checkpoint_name)]
    assert state_calls == [
        ("quant", {"weight_state": True, "input_state": False}),
        ("calibrate", True),
    ]
    assert student.gradient_checkpointing_enabled is True
    assert teacher.training is False
    assert student.config.use_cache is False
    assert trainer_calls[-1] == ("train",)
    training_args = trainer_calls[0][1]["args"]
    assert training_args.max_steps == 1
    assert training_args.gradient_accumulation_steps == 2
    if hf_save:
        assert student.saved_pretrained == str(tmp_path)
        assert save_calls == []
    else:
        assert save_calls[0][1] == str(tmp_path / "trained_quant_weight.pth")


@pytest.mark.parametrize("kd_enabled,expected_route", [(True, "train"), (False, "ptq")])
def test_weight_quant_pipeline_archives_config_cpu_fallback_and_routes(
    tmp_path, monkeypatch, kd_enabled, expected_route
):
    config_path = tmp_path / "input.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"train_files": ["data.json"]},
                "extra_training_config": {"fp16": True},
                "kd": {"enable": kd_enabled},
            }
        )
    )
    output_dir = tmp_path / "output"
    calls = []
    monkeypatch.setattr(train_module, "seed_all", lambda: calls.append("seed"))
    monkeypatch.setattr(
        train_module, "train", lambda *args, **kwargs: calls.append("train")
    )
    monkeypatch.setattr(
        train_module, "ptq_quant_pipline", lambda *args, **kwargs: calls.append("ptq")
    )

    train_module.weight_quant_pipline(
        "model-path",
        str(config_path),
        dopt_config="layers.json",
        output_dir=str(output_dir),
    )

    archived = yaml.safe_load((output_dir / "config.yaml").read_text())
    assert archived["config_path"] == "layers.json"
    assert archived["extra_training_config"]["fp16"] is False
    assert calls == ["seed", expected_route]


def test_weight_quant_pipeline_requires_model_path(tmp_path):
    config_path = tmp_path / "input.yaml"
    config_path.write_text(yaml.safe_dump({"dataset": {}, "kd": {"enable": False}}))
    with pytest.raises(Exception, match="base_model"):
        train_module.weight_quant_pipline(
            None, str(config_path), output_dir=str(tmp_path / "out")
        )


class PipelineModel(torch.nn.Module):
    def __init__(self, tied):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Module()
        self.model.embed_tokens.tie_lmhead = tied
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.loaded = None
        self.saved_pretrained = None

    def load_state_dict(self, state, strict=False):
        self.loaded = (dict(state), strict)
        return SimpleNamespace()

    def save_pretrained(self, path):
        self.saved_pretrained = path


@pytest.mark.parametrize("tied,hf_save", [(True, False), (False, True)])
def test_ptq_pipeline_orchestrates_state_and_save(tmp_path, monkeypatch, tied, hf_save):
    model = PipelineModel(tied)
    weights = {"model.embed_tokens.weight": torch.ones(1), "other": torch.ones(1)}
    quantizers = {"other": object()}
    loop_calls = []
    monkeypatch.setattr(
        ptq_loop,
        "weight_quant_loop",
        lambda *args, **kwargs: (
            loop_calls.append((args, kwargs)) or (quantizers, dict(weights))
        ),
    )
    monkeypatch.setattr(train_module, "prepare_model", lambda *args, **kwargs: model)
    state_calls = []
    monkeypatch.setattr(
        train_module,
        "set_quant_state",
        lambda *args, **kwargs: state_calls.append(("quant", kwargs)),
    )
    monkeypatch.setattr(
        train_module,
        "set_calibrate_state",
        lambda *args: state_calls.append(("calibrate", args[1])),
    )
    adapter_calls = []
    monkeypatch.setattr(
        train_module,
        "weight_q_params_adapter_into_dopt",
        lambda *args: adapter_calls.append(args),
    )
    save_calls = []
    monkeypatch.setattr(torch, "save", lambda *args: save_calls.append(args))
    config = {
        "dataset": {"train_files": ["data.json"], "ptq_samples": 3},
        "cutoff_len": 16,
        "config_path": "layers.json",
        "extra_training_config": {"fp16": False},
    }

    train_module.ptq_quant_pipline(
        "base-model",
        config,
        output_dir=str(tmp_path),
        hf_type_save=hf_save,
    )

    assert loop_calls[0][1]["nsamples"] == 3
    assert loop_calls[0][1]["seq_len"] == 16
    loaded_state, strict = model.loaded
    assert strict is False
    assert ("model.embed_tokens.weight" in loaded_state) is tied
    assert adapter_calls[0][2] is tied
    assert state_calls == [
        ("quant", {"weight_state": True, "input_state": False}),
        ("calibrate", True),
        ("calibrate", False),
    ]
    if hf_save:
        assert model.saved_pretrained == str(tmp_path)
        assert not save_calls
    else:
        assert save_calls


def test_ptq_loop_sort_config_and_defaults():
    names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "other",
    ]
    grouped = ptq_loop.sort_sequent(names)
    assert grouped == [names[:3], [names[3]], names[4:6], [names[6]]]
    config = {
        "layer": {"weight": {"bit": 4}},
        "bad": "value",
        "bad_weight": {"weight": 3},
    }
    assert ptq_loop._get_weight_config(config, "layer") == {"bit": 4}
    assert ptq_loop._get_weight_config(config, "missing") is None
    assert ptq_loop._get_weight_config(config, "bad") is None
    assert ptq_loop._get_weight_config(config, "bad_weight") is None
    args = ptq_loop.QuantArgs(nsamples=9)
    assert args.nsamples == 9
    assert args.sym is True
    assert args.true_sequential is False


class LlamaStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


def test_get_llama_sets_sequence_length_and_moves_layers_to_cpu(monkeypatch):
    model = LlamaStub()
    monkeypatch.setattr(ptq_loop, "model_define", lambda *args, **kwargs: model)

    result = ptq_loop.get_llama("model", 32, torch.float32)

    assert result is model
    assert result.seqlen == 32
    assert next(result.model.layers.parameters()).device.type == "cpu"


def test_weight_quant_loop_wires_dependencies_and_returns_state(monkeypatch):
    model = LlamaStub()
    model.eval = lambda: model
    expected_state = model.state_dict()
    calls = []
    monkeypatch.setattr(
        ptq_loop, "get_llama", lambda *args: calls.append(("model", args)) or model
    )
    monkeypatch.setattr(
        ptq_loop,
        "paser_quant_config",
        lambda *args: calls.append(("config", args)) or {"layer": {}},
    )
    monkeypatch.setattr(
        ptq_loop,
        "get_loaders",
        lambda *args, **kwargs: (
            calls.append(("loader", args, kwargs)) or (["batch"], None)
        ),
    )
    monkeypatch.setattr(
        ptq_loop,
        "llama_sequential",
        lambda *args: calls.append(("sequential", args)) or {"q": "value"},
    )

    quantizers, state = ptq_loop.weight_quant_loop(
        "model-path",
        ["data.json"],
        nsamples=4,
        seed=5,
        seq_len=8,
        quant_config="layers.json",
    )

    assert quantizers == {"q": "value"}
    assert state.keys() == expected_state.keys()
    assert ptq_loop.args.nsamples == 4
    assert ptq_loop.args.seed == 5
    assert ptq_loop.args.quant_config == {"layer": {}}
