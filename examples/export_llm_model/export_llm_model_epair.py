#!/usr/bin/env python3
"""Export qwen3-0.6B model to epair format using cann_lm model definitions.

Pipeline: config → model → weights → torch.export → epair → consolidated_weights_save
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from cann_lm.models.fake_ops import register_ge_converters
from cann_lm.converter.export.epair_utils import consolidated_weights_save
import torch_delegate

ROOT = Path(__file__).resolve().parent
HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"

def _remove_assert_tensor_metadata(ep: torch.export.ExportedProgram) -> int:
    """Remove aten._assert_tensor_metadata nodes from the exported graph.

    torch.export 自动插入的 metadata 校验节点（schema: -> ()）没有计算语义，
    但会被下游转换器（torchair / GE）拒绝。在导出阶段提前移除，
    保证 epair 和 reference 路径均不会再遇到该算子。
    """
    graph = ep.graph
    target = torch.ops.aten._assert_tensor_metadata.default
    removed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is target:
            graph.erase_node(node)
            removed += 1
    if removed > 0:
        graph.lint()
        ep.graph_module.recompile()
        print(f"[Strip] 移除了 {removed} 个 _assert_tensor_metadata 节点")
    return removed

def main():
    register_ge_converters()

    print(f"[1/4] Loading config from {HF_PATH}")
    config = Qwen3TextConfig.from_pretrained(HF_PATH)
    print(f"      hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"head_dim={config.head_dim}, vocab={config.vocab_size}")

    print(f"[2/4] Instantiating model (dtype=float16)")
    model = Qwen3ForCausalLM(config)
    model = model.to(dtype=torch.float16).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {total_params:,}")

    print(f"[3/4] Loading weights from {HF_PATH}")
    model = load_qwen3_weights(model, HF_PATH, device="cpu", dtype=torch.float16)

    # Qwen3ForCausalLM.forward 返回 dict（含 loss/logits/past_key_values/output_hidden_states），
    # torch.export 会将 dict 的每个 value 作为单独的 graph output。
    # C++ EpairModelParser 不支持 as_none 类型的 output_spec（如 loss=None, past_key_values=None），
    # 需用 wrapper 只返回 logits tensor，使 ep_json 的 output_specs 只有一个 as_tensor 条目。
    class Qwen3ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            out = self.model(input_ids=input_ids, attention_mask=None,
                             position_ids=None, past_key_values=None,
                             inputs_embeds=None, labels=None,
                             output_hidden_states=False)
            return out["logits"]

    wrapper = Qwen3ExportWrapper(model).eval()

    print(f"[4/4] Exporting to epair")
    batch, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), dtype=torch.long)

    # torch.export 即使 strict=False 也会自动插入 _assert_tensor_metadata 断言节点。
    # 这些节点用于运行时验证中间张量的 shape/dtype/stride 是否与导出时一致，
    # 每层 transformer 约产生 9 个断言点，qwen3-0.6B（28 层）共 262 个。
    # 特点：
    #   - 无输出（只断言不产生张量）
    #   - 无 GE 算子映射（ascend_ir.source 为空）
    #   - 纯 PyTorch 运行时验证，不是计算节点
    #   - GE 编译阶段不需要形状验证，可安全移除
    # 若不移除，C++ PlanFallbackNode 会因 "no outputs" 报错导致 Graph 构建失败。
    exported = torch.export.export(wrapper, (input_ids,), strict=False)
    _remove_assert_tensor_metadata(exported)
    epair = torch_delegate.export_to_epair(exported)

    # export_to_epair 内部会重新生成 ep_json，需再次过滤
    nodes = epair.ep_json["graph_module"]["graph"]["nodes"]
    epair.ep_json["graph_module"]["graph"]["nodes"] = [
        n for n in nodes if "assert" not in n.get("target", "").lower()
    ]

    output_dir = ROOT / "epair_model"
    consolidated_weights_save(epair, output_dir, save_weights=True, max_file_size=512 * 1024 * 1024)

    pt2_path = ROOT / "epair_model.pt2"
    torch_delegate.epair_save(epair, str(pt2_path))
    print(f"Saved: {pt2_path}")

    for i, inp in enumerate((input_ids,)):
        path = ROOT / f"input_{i}.bin"
        inp.cpu().detach().numpy().tofile(path)
        print(f"Saved input[{i}] shape={list(inp.shape)} -> {path}")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
