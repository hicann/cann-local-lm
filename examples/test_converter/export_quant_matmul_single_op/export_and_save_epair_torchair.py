"""Export single-op quant_matmul using TORCHAIR'S FULL converter (no monkey-patch).

Injects a fake torch_npu module into sys.modules providing only the dtype scalar
constants (int4=285, float4_e2m1fn_x2=296, ...) that torchair's full quantmatmul
converter reads (line 87 `import torch_npu`). The real torch_npu isn't installed
in the stub venv (only lib-level stub).

Conflict handled: _library.py sets _HAS_TORCH_NPU=True if `import torch_npu`
succeeds -> skips defining npu_quant_matmul op (expects torch_npu to define it).
We force _HAS_TORCH_NPU=False + NPU_LIB so cann_lm defines the op, while the fake
torch_npu remains importable for the converter's dtype constants.

torchair full converter (custom/quantmatmul.py): W4A8 path = int32 int4pack +
Bitcast int32->DT_INT4 (no keep_dim) + transpose_x2=False. is_arch35()=False on
910_9362 -> torchair uses V3 (not V4); export + OM compile to see what happens.
"""

import sys
import types
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXAMPLE_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import torch  # noqa: E402
from torch import nn  # noqa: E402

# ── 2. Import fused_ops FIRST (no fake torch_npu yet) so _library sees no torch_npu
#    -> _HAS_TORCH_NPU=False -> NPU_LIB created -> npu_quant_matmul op defined. ──
from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401,E402  (defines op)

# Define fake npu op schemas torchair's optimization patterns reference (so they
# don't crash with "'npu' has no attribute X"). Graph won't contain these nodes;
# patterns just need the op reference to resolve.
from cann_lm.models.fused_ops._library import NPU_LIB  # noqa: E402
if NPU_LIB is not None:
    for _schema in [
        "_npu_dtype_cast(Tensor self, int dtype, int? input_dtype=None) -> Tensor",
        "npu_dtype_cast(Tensor self, int dtype, int? input_dtype=None) -> Tensor",
        "_npu_dtype_cast_backward(Tensor self, int dtype, int? grad_dtype=None, int? input_dtype=None) -> Tensor",
        "npu_dtype_cast_backward(Tensor self, int dtype, int? grad_dtype=None, int? input_dtype=None) -> Tensor",
    ]:
        try:
            NPU_LIB.define(_schema)
        except Exception:
            pass  # already defined

    @torch.library.register_fake("npu::_npu_dtype_cast")
    def _fake_npu_dtype_cast(self, dtype, input_dtype=None):
        return torch.empty_like(self)

    @torch.library.register_fake("npu::npu_dtype_cast")
    def _fake_npu_dtype_cast_v2(self, dtype, input_dtype=None):
        return torch.empty_like(self)

    @torch.library.register_fake("npu::_npu_dtype_cast_backward")
    def _fake_npu_dtype_cast_bwd(self, dtype, grad_dtype=None, input_dtype=None):
        return torch.empty_like(self)

    @torch.library.register_fake("npu::npu_dtype_cast_backward")
    def _fake_npu_dtype_cast_bwd_v2(self, dtype, grad_dtype=None, input_dtype=None):
        return torch.empty_like(self)

# ── 3. NOW install fake torch_npu (for torchair __init__ + converter dtype constants) ──
import importlib.abc
import importlib.util
import types


class _Stub:
    """Callable + any-attribute stub (for torch_npu.distributed.X etc. not used at export)."""
    def __call__(self, *a, **k): return self
    def __getattr__(self, name): return self
    def __bool__(self): return False
    def __iter__(self): return iter([])


def _make_torch_npu_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__file__ = f"<fake {name}>"  # so inspect.getfile doesn't raise "built-in module"
    m.__path__ = []  # allow submodule imports
    if name == "torch_npu":
        # dtype scalar constants torchair's converter reads (verified on real torch_npu)
        m.int4 = 285
        m.float4_e2m1fn_x2 = 296
        m.float4_e1m2 = 297
        m.float8_e8m0fnu = 293
        m.float8_e5m2 = 291
        m.float8_e4m3fn = 292
        m.hifloat8 = 290
    m.__getattr__ = lambda attr: _Stub()  # PEP 562 module-level __getattr__
    return m


class _TorchNpuFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve `import torch_npu` / `import torch_npu.X` to fake modules (stub env)."""
    def find_spec(self, name, path=None, target=None):
        if name == "torch_npu" or name.startswith("torch_npu."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return _make_torch_npu_module(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _TorchNpuFinder())
sys.modules["torch_npu"] = _make_torch_npu_module("torch_npu")

import torch_delegate  # noqa: E402  (imports torchair -> import torch_npu finds fake)
from cann_lm.converter.export.epair_utils import consolidated_weights_save  # noqa: E402

# DO NOT call cann_lm register_ge_converters (would override torchair's full converter).
import torchair  # noqa: E402
# Apply is_cann_compat monkey-patch (same as cann_lm register_ge_converters) so the
# stub GE doesn't reject ge.Bitcast/ge.DynamicQuant with "OperatorFactory find optype
# X failed". Required for torchair's full int4 converter (uses ge.Bitcast int32->int4).
import torchair._ge_concrete_graph.compat_ir as _compat_ir  # noqa: E402
_compat_ir.is_cann_compat = lambda *a, **k: ""

# Force is_arch35()=True so torchair uses V4 (not V3). 910_9362 isn't matched by
# torchair's is_arch35 (only 910_95/950), but 910_93 HAS V4 prebuilt int8_int4 kernels
# (QuantBatchMatmulV4_ND_ND_int8_int4, tilingKey=24). V3's Bitcast int32->int4 shape
# inference doesn't expand (x2 k stays 1); V4 may handle int4 shape correctly.
import torchair._utils.check_platform as _cp  # noqa: E402
_cp.is_arch35 = lambda: True
import torchair._ge_concrete_graph.ge_converter.custom.quantmatmul as _qm  # noqa: E402
_qm.is_arch35 = _cp.is_arch35  # the converter imported is_arch35 by value; rebind
_has_conv = hasattr(torch.ops.npu.npu_quant_matmul.default, "_ge_converter")
print(f"[info] torchair npu_quant_matmul converter registered: {_has_conv}")


def _pack_int4_int4pack(w_int4_values: torch.Tensor) -> torch.Tensor:
    """Pack [n, k] int4 values (int8 [-8,7]) -> int32 int4pack [k, n//8]
    (8 int4 per int32 along n), pre-transposed for transpose_x2=False.
    Layout: sequential — int4 j -> nibble j (0..7) of int32[j//8]."""
    n, k = w_int4_values.shape
    w = w_int4_values.t().contiguous()  # [k, n]
    assert n % 8 == 0, f"n ({n}) must be divisible by 8 for int4pack"
    w = w.view(k, n // 8, 8)
    nibbles = (w & 0xF).to(torch.int32)
    packed = sum(nibbles[..., j] << (4 * j) for j in range(8))
    return packed.to(torch.int32)  # [k, n//8] int32


class QuantMatmulTorchair(nn.Module):
    def __init__(self, m: int = 4, k: int = 256, n: int = 8, group_size: int = 256):
        super().__init__()
        self.group_size = group_size
        self.n_out = n
        wq = torch.randint(-8, 7, (n, k), dtype=torch.int8)
        self.register_buffer("weight_q", _pack_int4_int4pack(wq))  # [k, n//8] int32
        sc = torch.rand(n, k // group_size, dtype=torch.float32) * 0.01 + 1e-3
        self.register_buffer("weight_scale", sc)
        self.register_buffer("dummy_bias", torch.zeros(n, dtype=torch.float16))

    def forward(self, x1: torch.Tensor, pertoken_scale: torch.Tensor) -> torch.Tensor:
        # torchair full converter: x2_dtype=285 (int4) -> Bitcast int32 int4pack -> DT_INT4
        return torch.ops.npu.npu_quant_matmul(
            x1, self.weight_q, self.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=self.dummy_bias,
            output_dtype=5,        # fp16
            x2_dtype=285,          # torch_npu.int4 -> DT_INT4
            group_sizes=[1, 1, self.group_size],
            transpose_x1=False,
            transpose_x2=False,    # torchair hardcodes False (weight pre-transposed)
        )


ROOT = _EXAMPLE_DIR
torch.manual_seed(43)
m, k, n, gs = 4, 256, 8, 256
net = QuantMatmulTorchair(m, k, n, gs).eval()
x1 = torch.randint(-127, 127, (m, k), dtype=torch.int8)
pertoken = torch.ones(m, dtype=torch.float32) * 0.01
sample_inputs = (x1, pertoken)

exported_program = torch.export.export(net, sample_inputs, strict=False)
epair = torch_delegate.export_to_epair(exported_program)

output_dir = ROOT / "epair_model_torchair"
consolidated_weights_save(epair, output_dir, save_weights=True)

# Post-process: set Bitcast outputDesc shape (torchair W4A8 path doesn't set it /
# Reshape — relies on GE Bitcast InferShape to expand int32[k,n//8] -> int4[k,n],
# which doesn't happen on this CANN). Explicitly set output shape = [k, n//8 * 8].
import json  # noqa: E402
_ep_path = output_dir / "ep_json_with_ascend_ir.json"
_ep = json.loads(_ep_path.read_text())


def _fix_bitcast_shape(node):
    if isinstance(node, dict):
        if node.get("type") == "Bitcast":
            indesc = node.get("inputDesc", [])
            outdesc = node.get("outputDesc", [])
            if indesc and outdesc:
                in_shape = indesc[0].get("shape", {})
                in_dims = in_shape.get("dim", []) if isinstance(in_shape, dict) else []
                if len(in_dims) >= 2 and outdesc[0].get("shape") in (None, "", "?"):
                    # int32 [k, n//8] -> int4 [k, n//8 * 8]  (8 int4 per int32)
                    out_dims = [in_dims[0], str(int(in_dims[1]) * 8)]
                    outdesc[0]["shape"] = {"dim": out_dims}
                    print(f"[fix] Bitcast output shape set to {out_dims}")
        for v in node.values():
            _fix_bitcast_shape(v)
    elif isinstance(node, list):
        for x in node:
            _fix_bitcast_shape(x)


_fix_bitcast_shape(_ep)
_ep_path.write_text(json.dumps(_ep, ensure_ascii=False))

for i, inp in enumerate(sample_inputs):
    (ROOT / f"input_torchair_{i}.bin").write_bytes(inp.cpu().contiguous().numpy().tobytes())
    print(f"Saved input_torchair[{i}] dtype={inp.dtype} shape={list(inp.shape)}")

print(f"\nepair (torchair full converter) saved to: {output_dir}")
print(f"  weight_q int32 int4pack [k, n//8], x2_dtype=285 (int4), transpose_x2=False")
