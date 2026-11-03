"""对比 hook 方式 vs DumpInterpreter 方式的输出，确保 golden 工程正确性。

同一模型 + 同一输入，两种方式各跑一遍 forward：
  A) hook: register_forward_hook 抓每个 nn.Module 的 input/output
  B) DumpInterpreter: gen_golden_with_dump 的 FX Interpreter 逐 node dump

比对两者的 module-level output（hook 的 output vs DumpInterpreter 对应 node 的 output）。
"""

import os
import sys
import torch
from pathlib import Path
from collections import OrderedDict
from torch.fx import Interpreter

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

MODEL_PATH = os.environ.get(
    "GEMMA4_MODEL_PATH", "/mnt/workspace/CANN-LM_5883/models/unsloth/gemma-4-E4B-it"
)
QUANT_DIR = os.environ.get("DOPT_QUANT_DIR", "/tmp/opencode/dopt_gemma4_full")

from cann_local_lm.converter.loader import load_model
from cann_local_lm.converter.swap import (
    load_dopt_state_dict,
    _read_quant_param_map,
    _swap_linear_from_sd,
    DEFAULT_SKIP,
)
from cann_local_lm.converter.export.quant_swap.materialize import (
    materialize_int4_per_group_symmetric,
)
from cann_local_lm.converter.export.quant_swap.quant_linear import QuantLinear
import cann_local_lm.converter.quant.dopt_lm.quantizer.qmin_max
import cann_local_lm.converter.quant.dopt_lm.quantizer.qeval  # noqa: F401  # 注册量化器
from cann_local_lm.converter.quant.dopt_lm.do_opt import (
    set_quant_state,
    set_calibrate_state,
)


def build_quant_model(hf_path, quant_dir, num_layers=3, seq_len=8):
    """加载模型 + dopt 量化 (QLinear unpacked int8, CPU DumpInterpreter 兼容)。"""
    torch.manual_seed(0)
    artifact = load_model(hf_path, dtype=torch.float16, device="cpu")
    artifact.model.model.layers = artifact.model.model.layers[:num_layers]
    lt = getattr(artifact.model.config, "layer_types", None)
    if lt:
        artifact.model.config.layer_types = lt[:num_layers]

    weight_pth = Path(quant_dir) / "trained_quant_weight.pth"
    sd = load_dopt_state_dict(str(weight_pth))
    param_map = _read_quant_param_map(str(quant_dir))
    _swap_linear_from_sd(artifact.model, sd, skip=DEFAULT_SKIP, param_map=param_map)
    set_quant_state(artifact.model, weight_state=True, input_state=True)
    set_calibrate_state(artifact.model, True)
    artifact.model.load_state_dict(
        torch.load(str(weight_pth), map_location="cpu", weights_only=True), strict=False
    )
    set_calibrate_state(artifact.model, False)
    for name, m in artifact.model.named_modules():
        if hasattr(m, "quant_op") and m.quant_op is not None:
            if "embed" in name or "per_layer_model_projection" in name:
                m.quant_op.set_quant_state(weight=False, input=False)
    # re-materialize QLinear weight 为 unpacked (CPU impl 要 [K,N] 不是 [K//2,N])
    for name, m in artifact.model.named_modules():
        if isinstance(m, QuantLinear):
            wkey, skey = f"{name}.weight", f"{name}.quant_op.weight_quantizer.s"
            if wkey in sd and skey in sd:
                w_fp = sd[wkey].float()
                out_f, in_f = w_fp.shape
                groups = sd[skey].numel() // out_f
                gs = in_f // groups
                scale2d = sd[skey].reshape(out_f, groups)
                wq, wsc = materialize_int4_per_group_symmetric(
                    w_fp, scale2d, gs, signed=True, nbits=4, packed=False
                )
                m.weight_q = torch.nn.Parameter(
                    wq.transpose(-1, -2).contiguous(), requires_grad=False
                )
                m.weight_scale = torch.nn.Parameter(
                    wsc.transpose(-1, -2).contiguous().to(torch.float32),
                    requires_grad=False,
                )
    artifact.model.eval()

    wrapper = artifact.wrapper_cls(
        artifact.model, kv_cache_max_len=2048, batch=1
    ).eval()
    sample_inputs = wrapper.get_sample_inputs(
        seq_len=seq_len, use_embeddings=True, quant_embeddings=False
    )
    wrapper.detach_externalized_lookups()
    return wrapper, sample_inputs, artifact


class HookCollector:
    """register_forward_hook 收集每个 nn.Module 的 output。"""

    def __init__(self):
        self.outputs = OrderedDict()
        self.handles = []

    def attach(self, model):
        for name, m in model.named_modules():
            if name == "":
                continue

            def make_hook(n):
                def hook(_m, _i, o):
                    self.outputs[n] = o

                return hook

            self.handles.append(m.register_forward_hook(make_hook(name)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class DumpInterpreterLite(Interpreter):
    """FX Interpreter, 只收集 named_modules 对应 node 的 output (不存全 node)。"""

    def __init__(self, module):
        super().__init__(module)
        self.node_outputs = {}

    def run_node(self, n):
        result = super().run_node(n)
        self.node_outputs[n.name] = result
        return result


def flatten_tensors(obj):
    out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(flatten_tensors(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_tensors(v))
    return out


def main():
    print("[build] loading gemma4 3-layer + dopt quant ...", flush=True)
    wrapper, sample_inputs, artifact = build_quant_model(
        MODEL_PATH, QUANT_DIR, num_layers=3, seq_len=8
    )

    # === A) Hook 方式 ===
    print("\n[A] hook forward ...", flush=True)
    hook = HookCollector()
    hook.attach(wrapper)
    with torch.no_grad():
        hook_out = wrapper(*sample_inputs)
    hook.detach()
    hook_final = flatten_tensors(hook_out)
    print(
        f"  hooked modules: {len(hook.outputs)}, final output tensors: {len(hook_final)}"
    )

    # === B) DumpInterpreter 方式 ===
    print("\n[B] DumpInterpreter forward ...", flush=True)
    with torch.no_grad():
        ep = torch.export.export(wrapper, sample_inputs, strict=False)
    gm = ep.module()
    interp = DumpInterpreterLite(gm)
    with torch.no_grad():
        interp_out = interp.run(*sample_inputs)
    interp_final = flatten_tensors(interp_out)
    print(
        f"  FX nodes: {len(list(gm.graph.nodes))}, final output tensors: {len(interp_final)}"
    )

    # === 比对最终输出 ===
    print("\n=== 比对最终输出 (hook vs DumpInterpreter) ===", flush=True)
    n = min(len(hook_final), len(interp_final))
    all_pass = True
    for i in range(n):
        h_t = hook_final[i].float().cpu()
        d_t = interp_final[i].float().cpu()
        if h_t.shape != d_t.shape:
            print(
                f"  output[{i}]: SHAPE MISMATCH hook={tuple(h_t.shape)} dump={tuple(d_t.shape)}"
            )
            all_pass = False
            continue
        max_diff = (h_t - d_t).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            h_t.flatten().unsqueeze(0), d_t.flatten().unsqueeze(0)
        ).item()
        ok = max_diff < 1e-4
        all_pass &= ok
        print(
            f"  output[{i}]: shape={tuple(h_t.shape)} maxΔ={max_diff:.2e} cos={cos:.8f} -> {'PASS' if ok else 'FAIL'}"
        )

    # === 比对 QLinear module outputs (hook module vs FX node) ===
    # FX 图里 module call 的 node name 是 torch.export 自动生成的 (如 "_param_const" / "call_module_xxx")
    # 不是完整 module path → 用 hook output 与 DumpInterpreter 最终输出逐算子比对
    # 更直接: 从 DumpInterpreter 的 node_outputs 里找 QuantLinear forward 产生的中间张量
    print(
        "\n=== 比对 QLinear module outputs (hook eager vs FX interpreter) ===",
        flush=True,
    )
    ql_checked = 0
    for name, m in wrapper.named_modules():
        if not isinstance(m, QuantLinear):
            continue
        hook_out_t = hook.outputs.get(name)
        if hook_out_t is None:
            continue
        h_t = flatten_tensors(hook_out_t)
        if not h_t:
            continue
        h_t = h_t[0].float().cpu()
        # 在 DumpInterpreter 的 node_outputs 里找: 与 hook output 形状一致 + cos 最大的那个
        best_match = None
        best_cos = -2.0
        for node_name, node_out in interp.node_outputs.items():
            if not isinstance(node_out, torch.Tensor):
                continue
            d_t = node_out.float().cpu()
            if d_t.shape == h_t.shape:
                cos = torch.nn.functional.cosine_similarity(
                    h_t.flatten().unsqueeze(0), d_t.flatten().unsqueeze(0)
                ).item()
                if cos > best_cos:
                    best_cos = cos
                    best_match = (node_name, d_t)
        if best_match:
            node_name, d_t = best_match
            max_diff = (h_t - d_t).abs().max().item()
            ok = max_diff < 1e-3
            all_pass &= ok
            if ql_checked < 8:
                print(
                    f"  {name}: maxΔ={max_diff:.2e} cos={best_cos:.8f} -> {'PASS' if ok else 'FAIL'} (best FX node: {node_name})"
                )
            ql_checked += 1
    print(
        f"  checked: {ql_checked}/{sum(1 for _, m in wrapper.named_modules() if isinstance(m, QuantLinear))} QLinear modules"
    )

    print(
        f"\n{'PASS' if all_pass else 'FAIL'}: golden 工程一致性验证 (hook vs DumpInterpreter)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
