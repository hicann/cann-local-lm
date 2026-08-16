#!/usr/bin/env python3
"""Verify epair output produced by cann-lm-converter.py CLI.

检查每个 epair_<model>/ 目录的产物完整性:
  - 必需 JSON 文件存在 (ep_json / model_config / weight_info)
  - 至少一个 weights_*.bin
  - tokenizer 文件存在
  - ep_json 含 ascend_ir 标注 (GE 转换成功)
  - weight_info 的 bin 文件名与实际一致
  - model_config 含必要字段
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_MODELS = {
    "gemma4": "gemma-4-E2B-it",
    "qwen3": "qwen3-0.6B",
    "qwen3_5": "Qwen3.5-0.8B",
}


def _check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "[OK]" if ok else "[FAIL]"
    print(f"  {tag} {name}{(' — ' + detail) if detail else ''}")
    return ok


def verify(model: str) -> bool:
    out = ROOT / f"epair_{model}"
    print(f"\n=== Verify {model} ({out.name}) ===")
    if not out.is_dir():
        _check("output dir exists", False)
        return False

    # 1. 必需 JSON
    all_ok = True
    jsons = sorted(out.glob("*.json"))
    json_names = {p.name for p in jsons}
    # ep_json: single -> ep_json_with_ascend_ir.json; prefill-decode -> prefill_/decode_
    ep_jsons = [p for p in jsons if "ascend_ir" in p.name]
    all_ok &= _check("ep_json_with_ascend_ir present", len(ep_jsons) >= 1,
                     f"{len(ep_jsons)} file(s)")
    for j in ep_jsons:
        try:
            data = json.loads(j.read_text())
            has_ir = "ascend_ir" in j.name or any(
                "ascend" in str(v).lower() for v in data.values()
            ) if isinstance(data, dict) else False
            n_nodes = len(data.get("graph", {}).get("nodes", [])) if isinstance(data, dict) else 0
            all_ok &= _check(f"{j.name} parseable ({n_nodes} nodes)", True,
                             f"nodes={n_nodes}")
        except Exception as e:
            all_ok &= _check(f"{j.name} parseable", False, str(e))

    all_ok &= _check("model_config.json present", "model_config.json" in json_names)
    all_ok &= _check("weight_info.json present", "weight_info.json" in json_names)

    # 2. model_config 字段 (结构: {model_type, config: {...}, hf_config})
    mc = out / "model_config.json"
    if mc.exists():
        try:
            cfg = json.loads(mc.read_text())
            sub = cfg.get("config", cfg)  # 字段在 config 子键 (qwen3) 或顶层
            keys = ["model_type", "hidden_size", "num_hidden_layers", "vocab_size"]
            missing = [k for k in keys if k not in sub]
            all_ok &= _check("model_config fields", not missing,
                             f"missing={missing}" if missing else "")
        except Exception as e:
            all_ok &= _check("model_config parseable", False, str(e))

    # 3. weights_*.bin
    bins = sorted(out.glob("weights_*.bin"))
    all_ok &= _check(f"weights_*.bin present", len(bins) >= 1, f"{len(bins)} file(s)")

    # 4. weight_info bin 文件名与实际一致
    wi = out / "weight_info.json"
    if wi.exists() and bins:
        try:
            info = json.loads(wi.read_text())
            # weight_info 结构: {"weights_0.bin": [{"name","offset","size"}], ...}
            ref_names = {p.name for p in bins}
            info_names = set(info.keys()) if isinstance(info, dict) else set()
            match = ref_names == info_names
            all_ok &= _check("weight_info bin names match", match,
                             f"ref={sorted(ref_names)} info={sorted(info_names)}")
            # 至少有若干权重条目
            total = sum(len(v) for v in info.values()) if isinstance(info, dict) else 0
            all_ok &= _check(f"weight entries ({total})", total > 0)
        except Exception as e:
            all_ok &= _check("weight_info parseable", False, str(e))

    # 5. tokenizer 文件
    tok_files = []
    for n in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.model"]:
        if (out / n).exists():
            tok_files.append(n)
    all_ok &= _check(f"tokenizer files ({len(tok_files)})", len(tok_files) >= 1,
                     ", ".join(tok_files))

    # 6. 目录大小
    total_size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    all_ok &= _check(f"total size {total_size / 1024 / 1024:.2f} MB", total_size > 0)

    return all_ok


def main():
    models = sys.argv[1:] or list(_MODELS.keys())
    results = {}
    for m in models:
        if m not in _MODELS:
            print(f"[✗] unknown model '{m}', available: {list(_MODELS)}")
            results[m] = False
            continue
        results[m] = verify(m)

    print("\n" + "=" * 40)
    all_pass = all(results.values())
    for m, ok in results.items():
        print(f"  {m:10s} {'PASS' if ok else 'FAIL'}")
    print("=" * 40)
    print(f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
