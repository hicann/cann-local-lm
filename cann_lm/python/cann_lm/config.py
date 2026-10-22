"""Shared testcase.json config loader (package module — pip-installed).

Used by both bbit.py (imports it back) and the cann_lm.generate CLI. Mirrors
tests/bbit/bbit.cpp load_config: JSON camelCase fields (model_path_prefill,
maxGenTokens, topK, stopSeq, ...) → normalized snake_case (prefill_omc_path,
max_tokens, top_k, stop_strings, ...). Weights are the SPLIT fields
model_weight_dir_prefill + model_weight_dir_decode (both required).

Living in the package (not in bbit.py the script) so `pip install` makes it
importable for the cann_lm.generate console-script.
"""
import os
import sys

__all__ = ["load_config", "file_exists", "is_degraded_run", "print_resolved_paths"]


class _CfgErr(Exception):
    pass


def _req(j, name):
    if name not in j:
        print(f'bbit: config missing field "{name}"', file=sys.stderr)
        raise _CfgErr
    return j[name]


def _req_str(j, name):
    v = _req(j, name)
    if not isinstance(v, str):
        print(f'bbit: config field "{name}" must be a string', file=sys.stderr)
        raise _CfgErr
    return v


def _req_int(j, name):
    v = _req(j, name)
    if isinstance(v, bool) or not isinstance(v, int):
        print(f'bbit: config field "{name}" must be an integer', file=sys.stderr)
        raise _CfgErr
    return v


def _req_num(j, name):
    v = _req(j, name)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        print(f'bbit: config field "{name}" must be a number', file=sys.stderr)
        raise _CfgErr
    return v


def _req_str_list(j, name):
    v = _req(j, name)
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        print(f'bbit: config field "{name}" must be an array of strings',
              file=sys.stderr)
        raise _CfgErr
    return v


def load_config(path):
    """Parse the testcase.json contract → normalized Config dict, or None.

    Weights are SPLIT fields model_weight_dir_prefill + model_weight_dir_decode
    (both required), mapped to prefill_weight / decode_weight.
    """
    try:
        with open(path, "rb") as f:
            content = f.read()
    except OSError:
        print(f"bbit: cannot open config: {path}", file=sys.stderr)
        return None
    try:
        import json
        j = json.loads(content)
    except Exception as e:  # noqa: BLE001
        print(f"bbit: config parse error: {e}", file=sys.stderr)
        return None
    try:
        cfg = {
            "tokenizer_type": _req_str(j, "tokenizer_type"),
            "tokenizer_path": _req_str(j, "tokenizer_path"),
            "prefill_omc_path": _req_str(j, "model_path_prefill"),
            "decode_omc_path": _req_str(j, "model_path_decode"),
            "meta_json_path": _req_str(j, "model_meta_config"),
            "prefill_weight": _req_str(j, "model_weight_dir_prefill"),
            "decode_weight": _req_str(j, "model_weight_dir_decode"),
        }
        sentences = _req(j, "sentences")
        if not isinstance(sentences, list) or not sentences:
            print('bbit: config field "sentences" must be a non-empty array',
                  file=sys.stderr)
            return None
        out = []
        for js in sentences:
            out.append({
                "prompt": _req_str(js, "prompt"),
                "max_tokens": _req_int(js, "maxGenTokens"),
                "top_k": _req_int(js, "topK"),
                "top_p": float(_req_num(js, "topP")),
                "temperature": float(_req_num(js, "temperature")),
                "seed": _req_int(js, "seed"),
                "repetition_penalty": float(_req_num(js, "repetitionPenalty")),
                "stop_strings": _req_str_list(js, "stopSeq"),
            })
        cfg["sentences"] = out
        return cfg
    except _CfgErr:
        return None  # reason already printed by the _req_* helper


def file_exists(path):
    """True iff a non-empty path exists on disk (degraded-run branch criterion)."""
    return bool(path) and os.path.isfile(path)


def is_degraded_run(prefill_path, decode_path):
    """Mirrors bbit.cpp's omc_configured && !omc_on_disk:
    omc paths both non-empty but at least one absent on disk → degraded run
    (create must fail-fast)."""
    configured = bool(prefill_path) and bool(decode_path)
    on_disk = file_exists(prefill_path) and file_exists(decode_path)
    return configured and not on_disk


# config field name → resolved path; the 6 paths the C++ load chain consumes.
_PATH_FIELDS = (
    ("tokenizer_path",    "tokenizer_path"),
    ("prefill_omc_path",  "prefill_omc_path"),
    ("decode_omc_path",   "decode_omc_path"),
    ("meta_json_path",    "meta_json_path"),
    ("prefill_weight",    "prefill_weight"),
    ("decode_weight",     "decode_weight"),
)


def print_resolved_paths(cfg, stream=None):
    """Print each resolved config path + its on-disk existence. Returns the
    count of MISSING paths. For diagnosing "I set the paths but the engine
    says meta_json unreadable"-class mismatches between the config and reality.

    Pure: reads the cfg dict + os.path only; no .so, no Engine.
    """
    out = stream if stream is not None else sys.stdout
    tt = cfg.get("tokenizer_type", "")
    print(f"tokenizer_type: {tt!r}", file=out)
    print(f"{'field':<20} {'exists':<12} path", file=out)
    print(f"{'-'*20} {'-'*12} {'-'*40}", file=out)
    missing = 0
    for label, key in _PATH_FIELDS:
        p = cfg.get(key, "")
        if not p:
            status = "(empty)"
        elif os.path.isfile(p):
            status = "file"
        elif os.path.isdir(p):
            status = "dir"
        else:
            status = "MISSING"
            missing += 1
        print(f"{label:<20} {status:<12} {p}", file=out)
    if missing:
        print(f"\n{missing} path(s) MISSING — the engine will reject these "
              f"(INVALID_ARGUMENT). Fix the config or copy the file(s) in place.",
              file=out)
    else:
        print("\nAll paths present — the config matches disk.", file=out)
    return missing
