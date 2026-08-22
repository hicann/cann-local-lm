"""Saver — export model to epair format with tokenizer and config."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import torch

from cann_lm.converter.loader import ModelArtifact
from cann_lm.converter.export.epair_utils import consolidated_weights_save, inline_fused_attn_seq_constants

# 额外分解的 aten 算子白名单：torchair 无内置 GE 映射或仅占位 NotImplementedError，分解成有映射的
# 基元算子（Exp/Log/Add/Select 等）。save_epair / save_epair_prefill_decode 共用，两函数白名单集合
# 保持一致（曾因 prefill-decode 漏 update 这组致 qwen3.5 softplus 未分解 → GE 转换撞占位 NotImplementedError）。
_EXTRA_DECOMPOSITION_OPS: frozenset = frozenset({
    torch.ops.aten.eye.default,
    torch.ops.aten.type_as.default,
    torch.ops.aten.index_copy.default,
    torch.ops.aten.index_put.default,
})


def _extract_host_side_weights(wrapper, use_embeddings: bool) -> list[tuple[str, torch.Tensor]]:
    """Collect embedding-lookup weights to additionally pack as host-side extras.

    torch.export 会 lift 全部参数 (含未用的 embed_tokens), pack_weights 也照常打包到
    weights_*.bin / weight_info.json; 但 host 推理做查表时, 把查表权重额外打包到独立的
    host_weights_info.json + host_weights_*.bin, 便于 host 按名定位、并携带 dtype/shape。
    故这里不做 dedup (即使权重已在图中, 也额外打一份)。仅 use_embeddings=True 时触发。
    """
    if not use_embeddings:
        return []
    candidate_names = wrapper.host_side_weight_names()
    sd = wrapper.state_dict()
    return [(name, sd[name].detach().cpu()) for name in candidate_names if name in sd]


def _assert_externalized_not_lifted(ep, names: list[str]) -> None:
    """断言 detach 生效: 外置查表权重不应出现在 ep.graph_signature 的 lifted 集合里。

    若仍被 lift, 说明 detach 失败 (模型 forward 在纯 embeddings 路径访问了它们, 或 detach 未执行),
    此时 weights_*.bin 会携带冗余查表权重、C++ 建图会为它们建 Data 节点 → 与 OM 不一致。
    """
    lifted: set[str] = set()
    gs = ep.graph_signature
    for mapping in (gs.inputs_to_parameters, gs.inputs_to_buffers, gs.inputs_to_lifted_tensor_constants):
        for _placeholder, logical_name in mapping.items():
            lifted.add(logical_name)
    leaked = [n for n in names if n in lifted]
    if leaked:
        raise RuntimeError(
            f"detach_externalized_lookups 失效: {leaked} 仍被 lift。"
            "纯 embeddings 路径的 forward 不应访问这些查表子模块 (embed_tokens 应被 inputs_embeds "
            "跳过, gemma4 embed_tokens_per_layer 应被实例方法替换跳过)。请检查模型 forward 或 wrapper。"
        )


def save_epair(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = False,
    save_weights: bool = True,
    whitelist_decompositions: set | None = None,
    embed_quant: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    import torch_delegate
    from torch_delegate import EpairConfig
    from cann_lm.converter.export.ge_converter import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    register_ge_converters()

    # 从注册表获取 wrapper 子类，构造统一 I/O wrapper
    if artifact.wrapper_cls is None:
        raise ValueError(
            f"Model '{artifact.model_type}' has no ExportWrapper subclass registered. "
            "Please implement one in cann_lm/converter/export_wrapper.py."
        )
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()

    # quant_embeddings 自动从 embed_quant 推断 (embed_quant 非空 = dopt 产物含 embed scale)
    quant_embeddings = embed_quant is not None
    if quant_embeddings:
        wrapper.set_quant_embeddings(embed_quant)

    # 构造示例输入 (统一 I/O 契约) — quant-embeddings 用 int8+scale gather, 否则用 embed_tokens 算 sample
    sample_inputs = wrapper.get_sample_inputs(
        seq_len=seq_len, use_embeddings=use_embeddings, quant_embeddings=quant_embeddings
    )

    # 抽取 host 查表权重: quant-embeddings 用 int8+scale (从 embed_quant); 否则 float embed_tokens
    if quant_embeddings:
        host_side_weights = [
            (fqn + "_q", int8) for fqn, (int8, _) in embed_quant.items()
        ] + [
            (fqn + "_scale", scale) for fqn, (_, scale) in embed_quant.items()
        ]
    else:
        host_side_weights = _extract_host_side_weights(wrapper, use_embeddings)

    # 纯 embeddings: detach 外置查表子模块, 使 torch.export 不 lift 它们 → weights_*.bin 不含
    # 冗余查表权重 (GE 编译本就剪掉), weights_*.bin ≈ OM。依赖 forward 不访问它们
    # (embed_tokens 被 inputs_embeds 跳过, gemma4 embed_tokens_per_layer 被实例方法替换跳过)。
    if use_embeddings:
        wrapper.detach_externalized_lookups()

    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    # 断言 detach 生效: 外置查表权重不应被 lift (否则 forward 在纯 embeddings 路径访问了它们 → detach 失效)。
    if use_embeddings:
        _assert_externalized_not_lifted(ep, wrapper.host_side_weight_names())

    # 用默认 whitelist (linear/matmul/layer_norm/softmax 等分解), conv1d 走自定义 converter
    config = EpairConfig()
    config.whitelist_decompositions.update(_EXTRA_DECOMPOSITION_OPS)
    epair = torch_delegate.export_to_epair(ep, config)

    # C++ 侧已兜底处理：
    # - _assert_tensor_metadata 节点：PlanNodes 顶部 IsAssertTensorMetadataNode 跳过
    # - as_none 的 constant_input：PreparePlaceholders 里 NodeInputKind::None → continue 跳过
    # 无需 Python 侧 strip ep_json。

    # 释放原始模型内存 (大模型 OOM 防护)
    # epair.ep 持有 ExportedProgram 的独立引用, 删除 model/wrapper/ep 不影响 epair
    del ep, wrapper
    del artifact.model
    import gc; gc.collect()

    consolidated_weights_save(
        epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
        host_side_weights=host_side_weights,
    )

    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir, use_embeddings=use_embeddings,
                      quant_embeddings=quant_embeddings)

def save_epair_prefill_decode(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = False,
    save_weights: bool = True,
    whitelist_decompositions: set | None = None,
    embed_quant: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    """Prefill/decode split export: emit two graphs (prefill + decode) + shared weights.

    Output layout (under output_dir):
    - ``prefill_ep_json_with_ascend_ir.json`` — prefill graph (seq_len 由参数指定)
    - ``decode_ep_json_with_ascend_ir.json`` — decode graph (seq_len=1, reusable)
    - ``weight_info.json`` + ``weights_*.bin`` — shared weights (packed once)
    - ``model_config.json`` + tokenizer files

    At inference time:
    - prefill: pad input_ids to seq_len, mask pad positions,
      output_pos=[prompt_len-1] to select the last real token's logits
    - decode: seq_len=1, reuse decode graph, feed previous-step KV cache

    Does NOT save .pt2 or input_N.bin (those belong in a verify script).
    """
    import torch_delegate
    from torch_delegate import EpairConfig
    from cann_lm.converter.export.ge_converter import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    register_ge_converters()

    if artifact.wrapper_cls is None:
        raise ValueError(
            f"Model '{artifact.model_type}' has no ExportWrapper subclass registered. "
            "Please implement one in cann_lm/converter/export_wrapper.py."
        )
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()

    # quant_embeddings 自动从 embed_quant 推断 (embed_quant 非空 = dopt 产物含 embed scale)
    quant_embeddings = embed_quant is not None
    if quant_embeddings:
        wrapper.set_quant_embeddings(embed_quant)

    # 先算 sample (quant-embeddings 用 int8+scale gather; 否则用 embed_tokens 算), 必须在 detach 前
    prefill_sample = wrapper.get_sample_inputs(
        seq_len=seq_len, use_embeddings=use_embeddings, quant_embeddings=quant_embeddings
    )
    decode_sample = wrapper.get_sample_inputs(
        seq_len=1, use_embeddings=use_embeddings, quant_embeddings=quant_embeddings
    )

    # 抽取 host 查表权重: quant-embeddings 用 int8+scale (从 embed_quant); 否则 float embed_tokens
    if quant_embeddings:
        host_side_weights = [
            (fqn + "_q", int8) for fqn, (int8, _) in embed_quant.items()
        ] + [
            (fqn + "_scale", scale) for fqn, (_, scale) in embed_quant.items()
        ]
    else:
        host_side_weights = _extract_host_side_weights(wrapper, use_embeddings)

    # 纯 embeddings: detach 外置查表子模块 → torch.export 不 lift → weights_*.bin 不含冗余查表权重。
    if use_embeddings:
        wrapper.detach_externalized_lookups()

    # Export prefill_ep (seq_len 由参数指定, shape-specialized)
    prefill_ep = torch.export.export(wrapper, prefill_sample, strict=False)
    if use_embeddings:
        _assert_externalized_not_lifted(prefill_ep, wrapper.host_side_weight_names())
    # 用默认 whitelist (linear/matmul/layer_norm/softmax 等分解), conv1d 走自定义 converter
    if whitelist_decompositions is None:
        whitelist_decompositions = set(EpairConfig().whitelist_decompositions)
    whitelist_decompositions.update(_EXTRA_DECOMPOSITION_OPS)
    config = EpairConfig(whitelist_decompositions=whitelist_decompositions)
    prefill_epair = torch_delegate.export_to_epair(prefill_ep, config)
    del prefill_ep

    # Export decode_ep (seq_len=1, reused for all decode steps)
    decode_ep = torch.export.export(wrapper, decode_sample, strict=False)
    decode_epair = torch_delegate.export_to_epair(decode_ep, config)

    # Release model memory (OOM guard); epair.ep holds independent ExportedProgram ref
    del decode_ep, wrapper
    del artifact.model
    import gc; gc.collect()

    # Write weights + prefill ep_json (save_weights=True, only once) + host-side extras
    consolidated_weights_save(
        prefill_epair, output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
        ep_json_filename="prefill_ep_json_with_ascend_ir.json",
        host_side_weights=host_side_weights,
    )
    # Inline actual_seq_qlen/kvlen as ge.Const (q_len=seq_len) — per-graph const, no
    # tilingDepend/PartitionedCall. Avoids the shared-weights_0.bin name collision.
    inline_fused_attn_seq_constants(
        output_dir / "prefill_ep_json_with_ascend_ir.json", seq_len)

    # Write decode ep_json only (save_weights=False, weights already packed above)
    consolidated_weights_save(
        decode_epair, output_dir,
        save_weights=False,
        ep_json_filename="decode_ep_json_with_ascend_ir.json",
    )
    # decode q_len=1 (seq_len=1, fixed in this mode). Inline as ge.Const so decode
    # resolves its own value (1) instead of prefill's (128) via shared weights_0.bin.
    inline_fused_attn_seq_constants(
        output_dir / "decode_ep_json_with_ascend_ir.json", 1)

    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(artifact, output_dir, use_embeddings=use_embeddings,
                      quant_embeddings=quant_embeddings)


def save_model_config(artifact: ModelArtifact, output_dir: Path, *,
                      use_embeddings: bool = False,
                      quant_embeddings: bool = False) -> None:
    config_obj = artifact.config
    if hasattr(config_obj, "to_dict"):
        config_dict = config_obj.to_dict()
    elif dataclasses.is_dataclass(config_obj):
        config_dict = dataclasses.asdict(config_obj)
    else:
        config_dict = dict(config_obj.__dict__)

    config_data = {
        "model_type": artifact.model_type,
        "config": config_dict,
        "hf_config": artifact.hf_config,
        "use_embeddings": use_embeddings,
        "quant_embeddings": quant_embeddings,
    }
    if artifact.quant_info:
        config_data["quant_info"] = artifact.quant_info
    path = output_dir / "model_config.json"
    with open(path, "w") as f:
        json.dump(config_data, f, indent=2, default=str)
    print(f"Saved: {path}")


def copy_tokenizer_files(files: list[Path], output_dir: Path) -> None:
    for fp in files:
        dest = output_dir / fp.name
        shutil.copy2(fp, dest)
        print(f"Saved: {dest}")
