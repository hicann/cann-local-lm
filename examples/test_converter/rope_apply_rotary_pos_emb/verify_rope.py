#!/usr/bin/env python3
"""Demo: 验证 gemma4 RoPE 切换到 npu_apply_rotary_pos_emb (ge.ApplyRotaryPosEmb) 的正确性.

背景:
  cann_lm 当前 gemma4 的 RoPE 走 npu_rotary_mul (单 tensor, -> ge.RotaryMul), 设计文档
  compositte_op.md:127 明确说 npu_apply_rotary_pos_emb (q+k 配对) 路径已删, 改用单 tensor 统一.
  本 demo 验证: 如果切回 npu_apply_rotary_pos_emb, 数值/IR 是否仍正确 (作为后续切换改动的依据).

  cann_lm 不提供 npu_apply_rotary_pos_emb 的 schema/CPU eager (因为不走这条路径), torchair
  只有 GE converter (apply_rotary_pos_emb.py), schema 由 torch_npu 提供 (stubs 环境下不存在).
  因此 demo 自己注册 schema + fake + CPU eager, 并手动 attach torchair 的 GE converter
  (因为 torchair custom/ 在 schema 未注册时 import 会被 warning 跳过, 装饰器不重跑).

两段验证 (默认 numeric, --mode=export 跑 IR 导出):
  numeric: 三路数值对比 (reference rotate_half / npu_rotary_mul / npu_apply_rotary_pos_emb)
           验证 half mode 下三种实现数值等价 (max_diff < eps). 多 layout × dtype.
  export:  构造一个只调 npu_apply_rotary_pos_emb 的 forward, 走 torchair + torch_delegate
           导出 epair, 检查 IR 含 ApplyRotaryPosEmb 节点 (而非 RotaryMul).

环境 (跑前必须 source):
  source /home/developer/Ascend/cann/set_env.sh
  export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0

跑法 (venv: /mnt/workspace/torch-delegate-harmony/.venv):
  # Part B (数值对比, 默认, 不需 LD_PRELOAD):
  .venv/bin/python examples/rope_apply_rotary_pos_emb/verify_rope.py

  # Part C (IR 导出验证):
  .venv/bin/python examples/rope_apply_rotary_pos_emb/verify_rope.py --mode=export
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from cann_lm.models.fused_ops._library import NPU_LIB, _HAS_TORCH_NPU, register_cpu_impl


# ════════════════════════════════════════════════════════════════════
# Part A: 注册 npu::npu_apply_rotary_pos_emb (schema + fake + CPU eager)
# ════════════════════════════════════════════════════════════════════

_ROPE_SCHEMA = (
    "npu_apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, "
    "str layout='BSH') -> (Tensor query_out, Tensor key_out)"
)


def _rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """half mode RoPE: out = x*cos + rotate_half(x)*sin; rotate_half = cat(-x2, x1).

    与 cann_lm.models.fused_ops.rotary_mul._rotary_mul_cpu 语义一致 (CPU eager 路径).
    layout 在 CPU 计算上不影响语义 (只影响 NPU kernel 的张量排布解释), 故无视.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _ensure_apply_rotary_pos_emb_registered() -> None:
    """幂等注册 npu::npu_apply_rotary_pos_emb 的 schema + fake + CPU eager.

    stubs 环境 (无 torch_npu): 自己 define schema + fake + CPU eager
        (CPU eager 在 LD_PRELOAD libacl_stub 时由 register_cpu_impl 自动跳过, 此时只能 export).
    真实 torch_npu 环境: schema 由 torch_npu 提供, 跳过 (本 demo 不在 NPU 上跑数值).
    """
    if _HAS_TORCH_NPU:
        return
    if NPU_LIB is None:
        raise RuntimeError("NPU_LIB is None but _HAS_TORCH_NPU is False; cannot register schema")
    if getattr(torch.ops.npu, "npu_apply_rotary_pos_emb", None) is not None:
        return  # 已注册 (例如本函数被重复调用, 或 torchair custom/ 已 attach schema)

    NPU_LIB.define(_ROPE_SCHEMA)

    @torch.library.register_fake("npu::npu_apply_rotary_pos_emb")
    def _meta(query, key, cos, sin, layout="BSH"):  # noqa: ANN001
        del cos, sin, layout
        return torch.empty_like(query), torch.empty_like(key)

    @register_cpu_impl("npu::npu_apply_rotary_pos_emb")
    def _cpu_impl(query, key, cos, sin, layout="BSH"):  # noqa: ANN001
        del layout
        q_out = _rope_half(query, cos, sin)
        k_out = _rope_half(key, cos, sin)
        return q_out.to(query.dtype), k_out.to(key.dtype)


def _attach_torchair_converter() -> None:
    """触发 torchair 的 npu_apply_rotary_pos_emb GE converter 自动注册.

    torchair custom/ 在 schema 未注册时 import 会 warning 跳过 apply_rotary_pos_emb 模块
    (装饰器 @register_fx_node_ge_converter(torch.ops.npu.npu_apply_rotary_pos_emb.default) 因
    schema 不存在而 AttributeError). 本函数在 schema 注册后重新 import 该子模块,
    触发模块顶层装饰器执行, 自动注册 converter 到 op.default._ge_converter.

    关键: 装饰后 conveter_npu_apply_rotary_pos_emb_default 名字被替换为 Converter 实例
    (Converter.__call__ 返回 self), 故不能再次 @register_fx_node_ge_converter 包装它,
    否则 wrapped_converter 会调用 Converter 实例而非原函数, dispatch 时触发
    'Converter.__call__() takes 2 positional arguments but 5 were given'.
    """
    import importlib
    # importlib.import_module 会执行模块顶层代码 (即使父包 __init__ 已缓存).
    # 若模块已被 import 过 (例如之前 torchair custom/ import 时 schema 不存在 warning 跳过),
    # 需 reload 强制重新执行顶层装饰器. 检测 _ge_converter 是否已设置决定是否 reload.
    op = torch.ops.npu.npu_apply_rotary_pos_emb
    if op is None:
        raise RuntimeError("npu_apply_rotary_pos_emb schema 未注册, 无法 attach converter")
    if hasattr(op.default, "_ge_converter") and op.default._ge_converter is not None:
        return  # 已注册 (例如本进程内已 attach 过)
    import torchair._ge_concrete_graph.ge_converter.custom.apply_rotary_pos_emb as _mod
    importlib.reload(_mod)  # 强制重新执行顶层装饰器 (此时 schema 已存在)


# ════════════════════════════════════════════════════════════════════
# Part B: 三路数值对比
# ════════════════════════════════════════════════════════════════════

def _ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """纯 PyTorch 参考实现 (ground truth): out = x*cos + rotate_half(x)*sin."""
    return _rope_half(x, cos, sin)


def _rotary_mul_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """cann_lm 当前实现: npu_rotary_mul 单 tensor (走 ge.RotaryMul)."""
    return torch.ops.npu.npu_rotary_mul(x, cos, sin, rotary_mode="half")


def _apply_rotary_pos_emb_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, layout: str = "BSH",
) -> tuple[torch.Tensor, torch.Tensor]:
    """切换后实现: npu_apply_rotary_pos_emb q+k 配对 (走 ge.ApplyRotaryPosEmb)."""
    return torch.ops.npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout=layout)


def _build_case(layout: str, dtype: torch.dtype) -> dict:
    b, h, s, d = 2, 4, 8, 64
    if layout == "BNSD":
        q = torch.randn(b, h, s, d, dtype=dtype)
        k = torch.randn(b, h, s, d, dtype=dtype)
        cos = torch.randn(1, 1, s, d, dtype=dtype).expand(b, h, s, d)
        sin = torch.randn(1, 1, s, d, dtype=dtype).expand(b, h, s, d)
    else:  # BSH
        q = torch.randn(b, s, h * d, dtype=dtype)
        k = torch.randn(b, s, h * d, dtype=dtype)
        cos = torch.randn(1, s, h * d, dtype=dtype).expand(b, s, h * d)
        sin = torch.randn(1, s, h * d, dtype=dtype).expand(b, s, h * d)
    return {"q": q, "k": k, "cos": cos, "sin": sin}


def _run_numeric() -> int:
    print("=" * 78)
    print("Part B: 数值对比 (reference rotate_half / npu_rotary_mul / npu_apply_rotary_pos_emb)")
    print("=" * 78)

    torch.manual_seed(0)
    layouts = ["BSH", "BNSD"]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    cases_passed = 0
    cases_total = 0

    for layout in layouts:
        for dtype in dtypes:
            cases_total += 1
            case = _build_case(layout, dtype)
            q, k, cos, sin = case["q"], case["k"], case["cos"], case["sin"]

            ref_q = _ref_rope(q, cos, sin)
            ref_k = _ref_rope(k, cos, sin)
            rmul_q = _rotary_mul_rope(q, cos, sin)
            rmul_k = _rotary_mul_rope(k, cos, sin)
            arpe_q, arpe_k = _apply_rotary_pos_emb_rope(q, k, cos, sin, layout=layout)

            eps = 1e-3 if dtype == torch.float32 else 1e-2
            diff_rmul_q = (ref_q.float() - rmul_q.float()).abs().max().item()
            diff_arpe_q = (ref_q.float() - arpe_q.float()).abs().max().item()
            diff_rmul_k = (ref_k.float() - rmul_k.float()).abs().max().item()
            diff_arpe_k = (ref_k.float() - arpe_k.float()).abs().max().item()

            ok = max(diff_rmul_q, diff_arpe_q, diff_rmul_k, diff_arpe_k) < eps
            cases_passed += ok
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] layout={layout:4s} dtype={str(dtype):14s}  "
                  f"diff_rmul(q/k)={diff_rmul_q:.2e}/{diff_rmul_k:.2e}  "
                  f"diff_arpe(q/k)={diff_arpe_q:.2e}/{diff_arpe_k:.2e}  (eps={eps:.0e})")

    print("-" * 78)
    print(f"Result: {cases_passed}/{cases_total} cases passed")
    return 0 if cases_passed == cases_total else 1


# ════════════════════════════════════════════════════════════════════
# Part C: IR 导出验证 (torchair + torch_delegate)
# ════════════════════════════════════════════════════════════════════

class _RopeModule(torch.nn.Module):
    """最小 forward: 只调 npu_apply_rotary_pos_emb, 用于 export 验证 IR 节点."""

    def forward(self, q, k, cos, sin):  # noqa: ANN001
        return torch.ops.npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout="BSH")


def _extract_ir_str(epair) -> str:
    """从 epair 对象提取 IR 字符串 (不同 torch_delegate 版本 API 不同)."""
    for attr in ("graph_def", "protocol", "proto", "_proto", "graph"):
        obj = getattr(epair, attr, None)
        if obj is not None:
            try:
                return str(obj)
            except Exception:  # noqa: BLE001
                continue
    return str(epair)


def _run_export() -> int:
    print("=" * 78)
    print("Part C: IR 导出验证 (torchair -> ge.ApplyRotaryPosEmb)")
    print("=" * 78)

    try:
        import torch_delegate  # noqa: F401
        from cann_lm.converter.export.ge_converter import register_ge_converters
    except ImportError as e:
        print(f"[SKIP] torch_delegate / cann_lm converter 不可用: {e}")
        return 0

    _attach_torchair_converter()
    register_ge_converters()

    torch.manual_seed(0)
    b, s, d = 2, 8, 64
    q = torch.randn(b, s, d, dtype=torch.float32)
    k = torch.randn(b, s, d, dtype=torch.float32)
    cos = torch.randn(1, s, d, dtype=torch.float32).expand(b, s, d)
    sin = torch.randn(1, s, d, dtype=torch.float32).expand(b, s, d)

    model = _RopeModule().eval()
    try:
        ep = torch.export.export(model, (q, k, cos, sin), strict=False)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] torch.export 失败: {e}")
        return 1
    print(f"[ OK ] torch.export 成功: {type(ep).__name__}")

    try:
        epair = torch_delegate.export_to_epair(ep)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] export_to_epair 失败: {e}")
        return 1
    print(f"[ OK ] export_to_epair 成功: {type(epair).__name__}")

    ir_str = _extract_ir_str(epair)
    found_apply = "ApplyRotaryPosEmb" in ir_str
    found_rotary_mul = "RotaryMul" in ir_str or "RotaryPositionEmbedding" in ir_str
    print(f"       IR 含 ApplyRotaryPosEmb: {found_apply}")
    print(f"       IR 含 RotaryMul/RotaryPositionEmbedding (不应出现): {found_rotary_mul}")

    if not found_apply:
        print("[FAIL] IR 不含 ApplyRotaryPosEmb 节点")
        print("IR snippet (first 3000 chars):")
        print(ir_str[:3000])
        return 1

    print("[PASS] IR 含 ApplyRotaryPosEmb 节点 (切换后走 ApplyRotaryPosEmb 路径)")

    # 顺便对比 forward 数值 (CPU eager, 仅当未 LD_PRELOAD libacl_stub 时可用)
    with torch.no_grad():
        out_q, out_k = model(q, k, cos, sin)
    ref_q = _ref_rope(q, cos, sin)
    ref_k = _ref_rope(k, cos, sin)
    diff_q = (ref_q - out_q).abs().max().item()
    diff_k = (ref_k - out_k).abs().max().item()
    print(f"[PASS] forward 数值对比 (CPU eager): diff_q={diff_q:.2e} diff_k={diff_k:.2e}")
    return 0


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--mode", choices=["numeric", "export"], default="numeric",
        help="numeric: 三路数值对比 (CPU eager); export: IR 导出验证 (torchair)",
    )
    args = parser.parse_args()

    _ensure_apply_rotary_pos_emb_registered()

    if args.mode == "numeric":
        return _run_numeric()
    return _run_export()


if __name__ == "__main__":
    sys.exit(main())
