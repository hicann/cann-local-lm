# Manifest字段说明

简体中文 | [English](README_en.md)

`manifest-v1.schema.json` 由Python的类型化Manifest自动生成。Manifest随
`processed_bytes` 写入PTE；OM位于PTE named data，外置权重位于PTD named data。

| 字段 | 作用 |
|---|---|
| `runtime_protocol.name` | 固定为 `AscendBackend`，标识当前backend协议 |
| `runtime_protocol.version` | runtime协议版本，当前为 `1` |
| `runtime_protocol.artifact_abi` | artifact ABI版本，当前为 `1` |
| `method` | ExecuTorch method名 |
| `delegate_index` | method内的delegate序号 |
| `role` | `inference`、`prefill` 或 `decode` |
| `acl_soc_name` | 已验证的设备名，同时作为CANN `ge.socVersion` 和runtime匹配目标 |
| `model.named_data_key` | PTE named data中OM的key |
| `model.sha256` | OM内容摘要 |
| `model.nbytes` | OM逻辑长度 |
| `weight_groups[].weight_group_key` | PTD权重组key |
| `weight_groups[].sha256` | 权重组内容摘要 |
| `weight_groups[].total_nbytes` | 权重组逻辑长度 |
| `file_constant_bindings[].binding_id` | FileConstant绑定ID |
| `file_constant_bindings[].file_name` | OM/GE侧文件名 |
| `file_constant_bindings[].weight_group_key` | 所属权重组 |
| `file_constant_bindings[].offset` | 权重组内偏移 |
| `file_constant_bindings[].nbytes` | 绑定区间长度 |
| `file_constant_bindings[].shape` | 绑定区间逻辑shape |
| `file_constant_bindings[].alias_of` | 可选，共享已有绑定区间 |
| `io_contract.inputs` | 输入tensor binding列表 |
| `io_contract.outputs` | 输出tensor binding列表 |
| `io_contract.*[].et_arg_index` | ExecuTorch参数序号 |
| `io_contract.*[].direction` | `input` 或 `output` |
| `io_contract.*[].ascend_name` | ACL tensor名 |
| `io_contract.*[].dtype` | tensor数据类型 |
| `io_contract.*[].shape` | tensor shape |
| `io_contract.*[].shape_range` | 可选动态shape范围 |
| `io_contract.*[].semantic_role` | `data`、`state_current` 或 `state_next` |
| `state.bindings` | 状态绑定列表，可为空 |
| `state.bindings[].state_id` | 状态ID |
| `state.bindings[].current_input_arg` | 当前状态输入参数序号 |
| `state.bindings[].next_output_arg` | 下一状态输出参数序号 |
| `state.bindings[].dtype` | 状态数据类型 |
| `state.bindings[].shape` | 状态shape |
| `state.bindings[].reset_policy` | `on_prefill`、`explicit` 或 `never` |
| `state.bindings[].reset_owner` | 是否负责重置状态 |
| `state.bindings[].host_visibility` | `device_only` 或 `synchronized` |

状态输入和输出固定使用独立的current/next双buffer，执行成功后提交新状态。
单buffer实验产物中的 `buffer_mode` 字段不再支持，需使用当前代码重新导出。

固定的存储位置、artifact类型、能力声明、对齐参数、存储类型、layout和扩展字段
不再写入Manifest，由producer/runtime使用固定约定处理。

Python字段模型和JSON/ASCD输出位于
`python/executorch_npu/artifacts/manifest.py`。图映射、权重注册和共享状态
检查在 `artifacts/` 的相应构造步骤完成；运行时解析与校验位于
`runtime/manifest.cpp`。编译主路径保留模型对象到最终输出，不反序列化manifest。


## 更新与检查

修改字段后，在仓库根目录运行：

```bash
python scripts/generate_manifest_schema.py
python scripts/generate_manifest_schema.py --check
```

生成文件描述字段类型、取值和结构约束；跨字段及跨delegate的业务约束仍由
artifact builders和C++ runtime校验。测试同时验证生成文件未过期，以及实际
序列化产物（含single/double state）符合JSON Schema。
