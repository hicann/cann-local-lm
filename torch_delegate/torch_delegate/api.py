# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import json
import dataclasses

import torch.export
from torch.export import ExportedProgram
from torch._export.serde.serialize import serialize

from .core.utils import logger
from .converter.patch_pytorch import patch_pytorch, restore_pytorch
from .epair.schema import EpairContents, EpairManifest
from .epair.config import EpairConfig
from .epair.constants import EPAIR_MANIFEST_FILENAME, EPAIR_ANNOTATED_MODEL_FILENAME
from .epair.writer import EpairWriter
from .lite_context import (
    LiteConversionError,
    LiteExportError,
    active_config,
    check_violations,
    export_lock,
    mark_full_initialized,
    prepare_lite_torchair,
    validate_lite_result,
)


_TORCHAIR_MISSING_MESSAGE = (
    "Full EPAIR export requires TorchAir (shipped with torch_npu), but it is not "
    "importable: {detail}. Install a torch_npu build matching your torch version "
    "(torch_npu>=2.9.0) and CANN, then retry; torch_delegate never installs it "
    "automatically. Use lite_export_context() to export without a CANN runtime."
)


def _torchair_dependency_error(exc: BaseException) -> RuntimeError:
    message = _TORCHAIR_MISSING_MESSAGE.format(detail=exc)
    logger.error(message)
    return RuntimeError(message)


def export_to_epair(
    ep: ExportedProgram, config: EpairConfig = EpairConfig()
) -> EpairContents:
    """将 PyTorch ExportedProgram 转换为含 Ascend IR 注解的 EpairContents。

    该函数执行以下流程：
    1. Monkey-patch PyTorch 的 do_auto_functionalized 为 v2 版本
    2. 使用 TorchAirGEConverter 将 FX 图节点转换为 Ascend GE IR
    3. 通过 EpairWriter 将 ascend_ir 字段注入 EP JSON
    4. 组装 EpairContents（含 manifest 元数据）

    Args:
        ep: PyTorch ExportedProgram，通过 torch.export.export() 获得。
        config: 转换配置，控制算子分解、模式融合等行为。默认使用 EpairConfig()。

    Returns:
        EpairContents 对象，包含：
        - manifest: 转换统计信息（总节点数、已转换节点数等）
        - ep_json: 含 ascend_ir 注解的 EP JSON dict
        - ep: 可执行的 ExportedProgram

    Raises:
        RuntimeError: 当 torchair 转换失败时抛出；完整导出模式下缺少
            torch_npu/torchair 时同样抛出，并提示安装配套版本（不会自动安装）。
        ValueError: 当 ExportedProgram 格式不合法时抛出。

    Example:
        >>> import torch
        >>> from torch_delegate import export_to_epair, epair_save
        >>> model = MyModel()
        >>> ep = torch.export.export(model, (torch.randn(2, 3),))
        >>> contents = export_to_epair(ep)
        >>> print(f"转换: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")
        >>> epair_save(contents, "model.epair")

    Note:
        - 该函数会临时修改 PyTorch 内部函数（do_auto_functionalized），
          调用结束后自动恢复。
        - 转换结果中的 ascend_ir.source 字段标识转换来源：
          "converter" 表示成功转换，空字符串表示 fallback 节点。
    """
    logger.info("Exporting ExportedProgram to epair...")
    with export_lock():
        lite_config = active_config()
        if lite_config is None:
            mark_full_initialized()
        try:
            if lite_config is not None:
                prepare_lite_torchair()
            from .converter.torchair_converter import TorchAirGEConverter
        except ImportError as exc:
            if lite_config is None:
                raise _torchair_dependency_error(exc) from exc
            raise LiteConversionError.from_exception(exc) from exc
        except Exception as exc:
            if lite_config is not None:
                raise LiteConversionError.from_exception(exc) from exc
            raise

        original_func = patch_pytorch()
        logger.debug("PyTorch patched: do_auto_functionalize -> v2")

        try:
            converter = TorchAirGEConverter(config=config)
            try:
                converted_ep = converter.convert_graph(ep)
            except Exception as exc:
                if lite_config is not None and not isinstance(exc, LiteExportError):
                    raise LiteConversionError.from_exception(exc) from exc
                raise
            if lite_config is not None:
                check_violations()
            artifact = serialize(converted_ep)
            for node in converted_ep.graph_module.graph.nodes:
                if node.op == "call_function":
                    torch_fn = node.meta.get("torch_fn")
                    if torch_fn is not None:
                        node.meta["torch_fn"] = torch_fn[:-1]

            ep_json = json.loads(artifact.exported_program)
            writer = EpairWriter(converter)
            ep_json, total_nodes, converted_nodes = writer.write(ep_json)
            if lite_config is not None:
                validate_lite_result(ep_json, total_nodes, converted_nodes)
            logger.info(
                f"Export complete: {converted_nodes}/{total_nodes} nodes converted"
            )
            cann_version = (
                "unknown" if lite_config is not None else writer._get_cann_version()
            )
            manifest = EpairManifest(
                version=1,
                backend_name="ascend_ge",
                cann_version=cann_version,
                total_nodes=total_nodes,
                converted_nodes=converted_nodes,
            )
            contents = EpairContents(
                manifest=manifest, ep_json=ep_json, ep=converted_ep
            )
            logger.info("EpairContents created successfully")
            return contents
        finally:
            restore_pytorch(original_func)
            logger.debug("PyTorch restored to original")


def epair_save(contents: EpairContents, f: str | os.PathLike) -> None:
    """将 EpairContents 序列化为 .epair 文件。

    .epair 文件是 PT2 兼容的 ZIP 归档，包含：
    - models/model.json: 标准 EP JSON
    - data/weights/: 模型权重
    - extra/epair_manifest.json: 转换元数据
    - extra/ep_json_with_ascend_ir.json: 含 ascend_ir 注解的 EP JSON

    Args:
        contents: export_to_epair() 返回的 EpairContents 对象。
        f: 输出文件路径，支持 str 或 os.PathLike。建议以 .epair 为后缀。

    Raises:
        OSError: 当文件写入失败时抛出。
        TypeError: 当 contents 不是 EpairContents 类型时抛出。

    Example:
        >>> from torch_delegate import export_to_epair, epair_save
        >>> contents = export_to_epair(ep)
        >>> epair_save(contents, "model.epair")

    Note:
        .epair 文件可被标准 PT2 工具部分读取（忽略 ascend_ir 字段即可），
        也可通过 epair_load() 完整加载。
    """
    logger.info(f"Saving epair to {f}")
    extra_files = {
        EPAIR_MANIFEST_FILENAME: json.dumps(dataclasses.asdict(contents.manifest)),
        EPAIR_ANNOTATED_MODEL_FILENAME: json.dumps(contents.ep_json),
    }
    torch.export.save(contents.ep, f, extra_files=extra_files)
    logger.info(f"Epair saved successfully to {f}")
