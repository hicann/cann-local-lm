"""Saver — export model to epair format with tokenizer and config."""

import dataclasses
import json
import shutil
from pathlib import Path

import torch

from cann_lm.converter.loader import ModelArtifact
from cann_lm.converter.export.epair_utils import (
    consolidated_weights_save,
    consolidated_weights_save_pd,
    inline_fused_attn_seq_constants,
    inline_weight_scale_dtype,
)

# Extra aten decomposition whitelist (torchair has no/placeholder GE mapping). Keep in sync across
# save_epair / save_epair_prefill_decode (a miss leaves ops undecomposed -> placeholder NotImplementedError).
_EXTRA_DECOMPOSITION_OPS: frozenset = frozenset(
    {
        torch.ops.aten.eye.default,
        torch.ops.aten.type_as.default,
        torch.ops.aten.index_copy.default,
        torch.ops.aten.index_put.default,
    }
)


def _extract_host_side_weights(
    wrapper, use_embeddings: bool
) -> list[tuple[str, torch.Tensor]]:
    """Additionally pack embedding-lookup weights as host-side extras (no dedup; use_embeddings only)."""
    if not use_embeddings:
        return []
    candidate_names = wrapper.host_side_weight_names()
    sd = wrapper.state_dict()
    return [(name, sd[name].detach().cpu()) for name in candidate_names if name in sd]


def _assert_externalized_not_lifted(ep, names: list[str]) -> None:
    """Assert detach worked: externalized lookup weights must not be lifted (else weights_*.bin carries redundant lookup weights, C++ builds Data nodes -> != OM)."""
    lifted: set[str] = set()
    gs = ep.graph_signature
    for mapping in (
        gs.inputs_to_parameters,
        gs.inputs_to_buffers,
        gs.inputs_to_lifted_tensor_constants,
    ):
        for _placeholder, logical_name in mapping.items():
            lifted.add(logical_name)
    leaked = [n for n in names if n in lifted]
    if leaked:
        raise RuntimeError(
            f"detach_externalized_lookups ineffective: {leaked} still lifted. "
            "The pure-embeddings forward must not touch these lookup submodules (embed_tokens should be "
            "skipped via inputs_embeds, gemma4 embed_tokens_per_layer via instance-method replacement). "
            "Check the model forward or wrapper."
        )


def save_epair(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    seq_len: int = 64,
    use_embeddings: bool = True,
    save_weights: bool = True,
    whitelist_decompositions: set | None = None,
    embed_quant: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    target_soc: str = "Ascend910B",
    allow_hf32: bool = False,
) -> None:
    import torch_delegate
    from torch_delegate import EpairConfig, LiteConfig, lite_export_context
    from cann_lm.converter.export.ge_converters import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lite_config = LiteConfig(target_soc=target_soc, allow_hf32=allow_hf32)
    with lite_export_context(lite_config):
        register_ge_converters()

    if artifact.wrapper_cls is None:
        raise ValueError(
            f"Model '{artifact.model_type}' has no ExportWrapper subclass registered. "
            "Please implement one in cann_lm/models/export_wrapper.py."
        )
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()

    # bool() not `is not None`: dopt's no-embed-quant swap returns {} and must fall back to float embed_tokens (else KeyError).
    quant_embeddings = bool(embed_quant)
    if quant_embeddings:
        wrapper.set_quant_embeddings(embed_quant)

    sample_inputs = wrapper.get_sample_inputs(
        seq_len=seq_len,
        use_embeddings=use_embeddings,
        quant_embeddings=quant_embeddings,
    )

    if quant_embeddings:
        host_side_weights = [
            (fqn + "_q", int8) for fqn, (int8, _) in embed_quant.items()
        ] + [(fqn + "_scale", scale) for fqn, (_, scale) in embed_quant.items()]
    else:
        host_side_weights = _extract_host_side_weights(wrapper, use_embeddings)

    # Pure-embeddings: detach lookup submodules so torch.export doesn't lift them (weights_*.bin ≈ OM; relies on forward not touching them).
    if use_embeddings:
        wrapper.detach_externalized_lookups()

    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    if use_embeddings:
        _assert_externalized_not_lifted(ep, wrapper.host_side_weight_names())

    config = EpairConfig()
    config.whitelist_decompositions.update(_EXTRA_DECOMPOSITION_OPS)
    with lite_export_context(lite_config):
        epair = torch_delegate.export_to_epair(ep, config)

    # C++ skips _assert_tensor_metadata nodes and as_none constant_inputs; no Python-side ep_json strip needed.

    # Release model memory (OOM guard); epair.ep holds an independent ExportedProgram ref.
    del ep, wrapper
    del artifact.model
    import gc

    gc.collect()

    consolidated_weights_save(
        epair,
        output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
        host_side_weights=host_side_weights,
    )

    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(
        artifact,
        output_dir,
        use_embeddings=use_embeddings,
        quant_embeddings=quant_embeddings,
    )


def save_epair_prefill_decode(
    artifact: ModelArtifact,
    output_dir: str | Path,
    *,
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    kv_cache_max_len: int = 2048,
    seq_len: int = 64,
    use_embeddings: bool = True,
    save_weights: bool = True,
    whitelist_decompositions: set | None = None,
    embed_quant: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    target_soc: str = "Ascend910B",
    allow_hf32: bool = False,
) -> None:
    """Prefill/decode split export: emit two graphs (prefill + decode) + shared weights.

    Output layout (under output_dir):
    - ``prefill_ep_json_with_ascend_ir.json`` — prefill graph (seq_len from the arg)
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
    from torch_delegate import EpairConfig, LiteConfig, lite_export_context
    from cann_lm.converter.export.ge_converters import register_ge_converters

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lite_config = LiteConfig(target_soc=target_soc, allow_hf32=allow_hf32)
    with lite_export_context(lite_config):
        register_ge_converters()

    if artifact.wrapper_cls is None:
        raise ValueError(
            f"Model '{artifact.model_type}' has no ExportWrapper subclass registered. "
            "Please implement one in cann_lm/models/export_wrapper.py."
        )
    wrapper = artifact.wrapper_cls(
        artifact.model,
        kv_cache_max_len=kv_cache_max_len,
        batch=1,
    ).eval()

    # bool() not `is not None`: dopt's no-embed-quant swap returns {} and must fall back to float embed_tokens (else KeyError).
    quant_embeddings = bool(embed_quant)
    if quant_embeddings:
        wrapper.set_quant_embeddings(embed_quant)

    prefill_sample = wrapper.get_sample_inputs(
        seq_len=seq_len,
        use_embeddings=use_embeddings,
        quant_embeddings=quant_embeddings,
        prefill_seq=seq_len,
    )
    decode_sample = wrapper.get_sample_inputs(
        seq_len=1,
        use_embeddings=use_embeddings,
        quant_embeddings=quant_embeddings,
        prefill_seq=seq_len,
    )

    if quant_embeddings:
        host_side_weights = [
            (fqn + "_q", int8) for fqn, (int8, _) in embed_quant.items()
        ] + [(fqn + "_scale", scale) for fqn, (_, scale) in embed_quant.items()]
    else:
        host_side_weights = _extract_host_side_weights(wrapper, use_embeddings)

    # Pure-embeddings: detach lookup submodules so torch.export doesn't lift them (weights_*.bin ≈ OM).
    if use_embeddings:
        wrapper.detach_externalized_lookups()

    prefill_ep = torch.export.export(wrapper, prefill_sample, strict=False)
    if use_embeddings:
        _assert_externalized_not_lifted(prefill_ep, wrapper.host_side_weight_names())
    if whitelist_decompositions is None:
        whitelist_decompositions = set(EpairConfig().whitelist_decompositions)
    whitelist_decompositions.update(_EXTRA_DECOMPOSITION_OPS)
    config = EpairConfig(whitelist_decompositions=whitelist_decompositions)
    with lite_export_context(lite_config):
        prefill_epair = torch_delegate.export_to_epair(prefill_ep, config)
    del prefill_ep

    decode_ep = torch.export.export(wrapper, decode_sample, strict=False)
    with lite_export_context(lite_config):
        decode_epair = torch_delegate.export_to_epair(decode_ep, config)

    # Release model memory (OOM guard); epair.ep holds an independent ExportedProgram ref.
    del decode_ep, wrapper
    del artifact.model
    import gc

    gc.collect()

    consolidated_weights_save_pd(
        prefill_epair,
        decode_epair,
        output_dir,
        save_weights=save_weights,
        max_file_size=max_file_size,
        host_side_weights=host_side_weights,
    )

    copy_tokenizer_files(artifact.tokenizer_files, output_dir)
    save_model_config(
        artifact,
        output_dir,
        use_embeddings=use_embeddings,
        quant_embeddings=quant_embeddings,
    )


def save_model_config(
    artifact: ModelArtifact,
    output_dir: Path,
    *,
    use_embeddings: bool = False,
    quant_embeddings: bool = False,
) -> None:
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
