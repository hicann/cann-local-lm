'''
UltraChat 200K
FineWeb-Edu
ToolBench
BFCLv4
C4
'''
import os
# 1. 必须在 import datasets 之前设置！
os.environ["HF_HUB_DISABLE_XET"] = "1"  # 禁用容易卡住的 Xet 传输
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"  # 将超时时间延长到 120 秒（默认 10 秒太短）
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
from datasets import load_dataset


root = "/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/calibrate_datasets"

def UltraChat_200k(): 
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", 
        cache_dir=root
    )
    def format_text(example): 
        prompt = example.get("prompt", "")
        messages = example.get("messages", [])
        text = prompt + "\n\n"
        for msg in messages: 
            role = msg.get("role", "")
            content = msg.get("content", "")
            text += f"{role}: {content}\n"
        return {"text": text.strip()}
    
    train_ds = ds["train_sft"].shuffle(seed=42).select(range(8192))
    train_ds = train_ds.map(format_text, remove_columns=train_ds.column_names)
    train_ds.to_json(
        f"/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/gptq_calibrate/llm/calibrate_ultrachat.jsonl"
    )

def FineWeb_Edu(): 
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train[:512]", 
        cache_dir=root,
    )
    train_ds = ds.shuffle(seed=42).select(range(min(8192, len(ds))))
    train_ds = train_ds.map(lambda x: {"text": x["text"]}, remove_columns=train_ds.column_names)
    train_ds.to_json(
        f"/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/gptq_calibrate/llm/calibrate_fineweb_edu.jsonl"
    )

def ToolBench():
    ds = load_dataset(
        "tuandunghcmut/toolbench-v1", 
        split="train", 
        cache_dir=root,
    )
    def format_toolbench(example): 
        if "conversations" in example: 
            conv = example["conversations"]
            if isinstance(conv, dict) and "from" in conv and "value" in conv: 
                text = ""
                for role, content in zip(conv["from"], conv["value"]): 
                    text += f"{role}: {content}\n"
                return {"text": text.strip()}
            return {"text": str(example)}
        return {"text": str(example)}
    
    train_ds = ds.shuffle(seed=42).select(range(min(8192,len(ds))))
    train_ds = train_ds.map(format_toolbench, remove_columns=train_ds.column_names)
    train_ds.to_json(
        f"/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/gptq_calibrate/llm/calibrate_toolbench.jsonl"
    )

def BFCLv4(): 
    import json
    bfcl_dir = "/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/calibrate_datasets/bfcl"
    json_files = [
        "BFCL_v3_simple.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_live_simple.json",
        "BFCL_v3_live_multiple.json",
        "BFCL_v3_live_parallel.json",
        "BFCL_v3_live_parallel_multiple.json",
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_composite.json",
        "BFCL_v3_multi_turn_long_context.json",
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v3_multi_turn_miss_param.json",
    ]

    def load_file(file_path): 
        result = []
        with open(file_path) as f: 
            for line in f: 
                result.append(json.loads(line))
        return result
    
    all_data = []
    for fname in json_files: 
        fpath = os.path.join(bfcl_dir, fname)
        if os.path.exists(fpath): 
            all_data.extend(load_file(fpath))

    def format_bfcl(example): 
        question = example.get("question", [])
        function = example.get("function", [])
        if question:
            parts = []
            for turn in question: 
                if isinstance(turn, list): 
                    for msg in turn: 
                        if isinstance(msg, dict) and "content" in msg: 
                            role = msg.get("role", "user")
                            parts.append(f"{role}: {msg['content']}")
            question_text = "\n".join(parts) if parts else ""
            if function: 
                func_text = json.dumps(function, ensure_ascii=False)
                text = f"Question: {question_text}\nFunctions: {func_text}"
            else: 
                text = f"Question: {question_text}"
            return {"text": text.strip()}
        return {"text": str(example)}
    
    import random
    random.seed(42)
    random.shuffle(all_data)
    all_data = all_data[:8192]

    formatted = [format_bfcl(ex) for ex in all_data]
    with open(f"/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/gptq_calibrate/llm/calibrate_bfclv4.jsonl", "w") as f: 
        for item in formatted: 
            f.write(json.dumps(item) + "\n")

def C4(): 
    ds_stream = load_dataset(
        "allenai/c4", 
        name="en", 
        split="train", 
        streaming=True,  # 开启流式加载
    )
    ds = ds_stream.take(8192)
    train_ds = ds.shuffle(seed=42)
    train_ds = train_ds.map(lambda x: {"text": x["text"]}, remove_columns=train_ds.column_names)
    train_ds.to_json(
        f"/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/gptq_calibrate/llm/calibrate_c4.jsonl"
    )

def COIG_CQIA(): 
    ds = load_dataset(
        "json", 
        data_files=f"{root}/COIG-CQIA/COIG-CQIA-full.jsonl", 
        split="train", 
        cache_dir=root, 
    )
    def format_text(example): 
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        text = f"instruction: {instruction}\n"
        if inp: 
            text += f"input: {inp}\n"
        text += f"output: {output}"
        return {"text": text.strip()}
    
    train_ds = ds.shuffle(seed=42).select(range(min(2048, len(ds))))
    train_ds = train_ds.map(format_text, remove_columns=train_ds.column_names)
    train_ds.to_json(
        f"/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/gptq_calibrate/llm/calibrate_coig_cqia.jsonl", 
        force_ascii=False, lines=True, orient="records"
    )

def mC4(): 
    ds_stream = load_dataset(
        "allenai/c4", 
        name="zh", 
        split="train", 
        streaming=True,  # 开启流式加载
    )
    ds = ds_stream.take(8192)
    train_ds = ds.shuffle(seed=42)
    train_ds = train_ds.map(lambda x: {"text": x["text"]}, remove_columns=train_ds.column_names)
    train_ds.to_json(
        f"/srv/workspace/Kirin_AI_Workspace/AIC_I/d00799916/project/eco/2026/model_zoo/gptq_calibrate/llm/calibrate_c4_zh.jsonl", 
        force_ascii=False, lines=True, orient="records"
    )

if __name__ == "__main__": 
    # UltraChat_200k()
    # FineWeb_Edu()
    # ToolBench()
    # BFCLv4()
    # mmC4()
    COIG_CQIA()
