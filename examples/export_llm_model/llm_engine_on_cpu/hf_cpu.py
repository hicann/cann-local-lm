"""Disable transformers' torch_npu probe so HF works on CPU without torch_npu.

Import this module BEFORE `from transformers import ...` in environments where
the `torch_npu` package is installed but its native libs (e.g. libhccl_compat.so)
are missing. transformers' `is_torch_npu_available()` gates on
`importlib.util.find_spec("torch_npu")`, which returns True (package present) and
then fatally `import torch_npu`. Patching it to False short-circuits the probe,
and the lru_cache keeps it False, so no `import torch_npu` ever happens.

Usage::

    import cann_lm.hf_cpu  # noqa: F401  <-- must precede `from transformers import ...`
    from transformers import AutoTokenizer, Qwen3ForCausalLM
"""

from __future__ import annotations


def _disable_torch_npu_probe() -> None:
    import transformers.utils.import_utils as _iu
    import transformers.utils as _u

    _iu.is_torch_npu_available = lambda *args, **kwargs: False
    _u.is_torch_npu_available = _iu.is_torch_npu_available


_disable_torch_npu_probe()

__all__ = []
