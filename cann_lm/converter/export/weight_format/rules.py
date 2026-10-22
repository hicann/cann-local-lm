"""Weight format rule resolution: name_pattern + op_type -> format config."""

import fnmatch
import json
from pathlib import Path

_DEFAULT_RULES_PATH = Path(__file__).parent.parent / "weight_format_rules.json"


def load_rules(path: Path | None = None) -> dict:
    """Load shipped weight_format_rules.json (or override path). No cache; caller loads once."""
    rules_path = path if path is not None else _DEFAULT_RULES_PATH
    with open(rules_path) as f:
        return json.load(f)


def resolve_format(name: str, is_quantized: bool, *, rules: dict | None = None) -> dict:
    """Resolve weight format rule by name_pattern + op_type.

    op_type is MatMul (fp16/bf16 Linear) when not quantized, QuantBatchMatmulV4 when
    quantized (state_dict contains weight_scale). Returns the matched rule or default.

    Per-rule ``enable`` field: ``false`` → rule skipped (as if non-existent, continue
    scanning); omitted = ``true``. Replaces the deprecated ``CANN_LM_ENABLE_WEIGHT_TRANS``
    env total switch — control is now per-rule in the config.
    """
    rules = rules if rules is not None else load_rules()
    op_type = "QuantBatchMatmulV4" if is_quantized else "MatMul"
    default = rules.get("default", {"format": "ND"})
    for rule in rules.get("rules", []):
        if rule.get("op_type") != op_type:
            continue
        if not rule.get("enable", True):
            continue
        if fnmatch.fnmatch(name, rule.get("name_pattern", "*")):
            return rule
    return default
