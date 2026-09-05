"""ExportWrapper 基类单元测试 — 顶层 cann_lm/models/export_wrapper.py 的基类行为.

基类 forward 依赖子类 _drive_layers (抽象), 故只测 __init__ 属性 / set_quant_embeddings /
_num_layers; 子类 (Gemma4/Qwen3/Qwen3_5 ExportWrapper) 的端到端验证由各模型 test_* 覆盖.
"""
import types

import pytest
import torch
import torch.nn as nn

from cann_lm.models.export_wrapper import ExportWrapper


class _StubInner(nn.Module):
    """模拟 model.model (有 .layers 且作为 submodule 注册, 使 parameters() 非空)."""

    def __init__(self, n: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])


class _StubModel(nn.Module):
    """模拟 model: 有 .config + .model.layers + parameters() (供 _model_dtype)."""

    def __init__(self, n_layers: int = 3, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4)
        self.model = _StubInner(n_layers)  # nn.Module, layers 注册为 submodule
        self.to(dtype)


def test_init_attributes_defaults():
    m = _StubModel(n_layers=5)
    w = ExportWrapper(m, kv_cache_max_len=2048, batch=1)
    assert w.model is m
    assert w.config is m.config
    assert w.kv_cache_max_len == 2048
    assert w.batch == 1
    assert w._model_dtype == torch.float32
    assert w._quant_embeddings == {}


def test_init_custom_kwargs():
    m = _StubModel(n_layers=2, dtype=torch.bfloat16)
    w = ExportWrapper(m, kv_cache_max_len=512, batch=4)
    assert w.kv_cache_max_len == 512
    assert w.batch == 4
    assert w._model_dtype == torch.bfloat16


@pytest.mark.parametrize("n_layers", [1, 3, 8])
def test_num_layers_matches_stub(n_layers):
    w = ExportWrapper(_StubModel(n_layers=n_layers))
    assert w._num_layers() == n_layers


def test_set_quant_embeddings_stores_dict():
    w = ExportWrapper(_StubModel())
    q = {
        "embed_tokens": (torch.randint(-128, 127, (10, 4), dtype=torch.int8), torch.randn(10)),
    }
    w.set_quant_embeddings(q)
    assert set(w._quant_embeddings) == {"embed_tokens"}
    emb_w, scale = w._quant_embeddings["embed_tokens"]
    assert emb_w.dtype == torch.int8
    assert scale.shape == (10,)


def test_set_quant_embeddings_is_copy_not_ref():
    w = ExportWrapper(_StubModel())
    src = {"k": (torch.zeros(2), torch.ones(2))}
    w.set_quant_embeddings(src)
    src["k"] = (torch.ones(2), torch.zeros(2))  # 改原 dict 不应影响 wrapper
    emb_w, _ = w._quant_embeddings["k"]
    assert torch.equal(emb_w, torch.zeros(2))  # set 时已 dict() 复制
