# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.

# Stage1 minimal: calibration-data loader for the GPTQ loop.
# Provides get_json_data / get_pt_data / get_loaders.
import os
import random
from collections import defaultdict
import torch
if os.getenv("DEVICE", "cuda") == "npu": 
    import torch_npu  # noqa: F401 - registers the torch.npu backend


def set_seed(seed): 
    import numpy as np
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_json_data(nsamples, seed, seqlen, model, local_root):
    import random
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)

    ratio_all = []
    paths= []
    for item in local_root: 
        path = item.split(":")[0]
        ratio = float(item.split(":")[1]) if ":" in item else 1.0
        ratio_all.append(ratio)
        paths.append(path)
    total_ratio = sum(ratio_all)
    chunks_per_dataset = []
    for ratio in ratio_all: 
        chunks_per_dataset.append(int(nsamples * ratio / total_ratio))

    diff = nsamples - sum(chunks_per_dataset)
    if diff != 0: 
        chunks_per_dataset[0] += diff
    trainloader = []
    for path, target_chunks in zip(paths, chunks_per_dataset): 
        if target_chunks == 0: 
            continue
        trainset = load_dataset("json", data_files={"train": path})
        traindata = [x['text'] for x in trainset['train']]
        random.shuffle(traindata)
        dataset_enc = tokenizer("\n\n".join(traindata), return_tensors='pt')
        input_ids = dataset_enc.input_ids[0]
        total_tokens = input_ids.shape[0]
        if total_tokens > target_chunks * seqlen: 
            max_start = total_tokens - (target_chunks * seqlen) - 1
            start_idx = random.randint(0, max_start)
            sampled_ids = input_ids[start_idx : start_idx + (target_chunks * seqlen)]
            chunks = sampled_ids.view(target_chunks, seqlen)
        else: 
            chunks_list = []
            for _ in range(target_chunks): 
                i = random.randint(0, total_tokens - seqlen - 1)
                chunks_list.append(input_ids[i : i + seqlen])
            chunks = torch.stack(chunks_list)
        for k in range(target_chunks): 
            inp = chunks[k:k+1, :]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    random.shuffle(trainloader)
    return trainloader, None


def get_pt_data(name, nsamples, seed=42): 
    random.seed(seed)
    data_all = list()
    for item in name: 
        path = item.split(":")[0]
        ratio = float(item.split(":")[1])
        slice_len = int(nsamples * ratio)
        data = torch.load(path, map_location='cpu')
        if "input_embed" in data.keys(): 
            data["input_embed"] = data["input_embed"][:slice_len]
        if "input_mask" in data.keys(): 
            data["input_mask"] = data["input_mask"][:slice_len]
        if "input_position_id" in data.keys(): 
            data["input_position_id"] = data["input_position_id"][:slice_len]
        data_all.append(data)

    combined_data = defaultdict(list)
    for d in data_all: 
        value = d.get('input_embed')
        random.shuffle(value)
        combined_data["input_embed"].extend(value)
    return combined_data, None


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
): 
    if name[0].split(":")[0].endswith('pt'): 
        return get_pt_data(name, nsamples, seed)
    return get_json_data(nsamples, seed, seqlen, model, name)

    return None, None
