# 产物协议

字段级参考模式（schema）见
[`examples/manifest_schema/manifest-v1.schema.json`](../examples/manifest_schema/manifest-v1.schema.json)。
该文件是v1协议参考，不参与构建或运行时校验。Python字段定义位于
`python/executorch_npu/artifacts/manifest.py`，由Pydantic模型管理。
产物构造层在图映射、权重注册和多方法状态汇总时检查对应业务条件；
落盘产物由 `runtime/manifest.cpp` 解析并校验。

`compiler.compile_partition()` 返回OM和权重文件路径。后端在临时目录释放前调用
`artifacts.builder.build_manifest()` 构造 `Manifest` 并注册命名数据。
多方法通过 `check_manifest_set()` 检查委托后端标识和共享状态后，
调用 `Manifest.to_bytes()` 生成JSON UTF-8字节并添加ASCD头，交给
ExecuTorch写入PTE。编译流程不再解码已生成的字节，也不再调用独立的
`validate_manifest()`。原 `encode_manifest()` 调用替换为 `manifest.to_bytes()`；
需要诊断时，可先读取ASCD头，再使用JSON工具检查负载内容。

JSON字段及嵌套保持v1不变，可选字段未设置时省略。JSON键的顺序由模型
字段声明决定，输出保持确定性，但不要求与旧实现逐字节相同。

每个委托后端的 `processed_bytes` 使用以下格式：

```text
ASCD | uint16_le framing_version=1 | uint32_le json_length | UTF-8 JSON
```

v1协议约束：

- `runtime_protocol.version=1`，`artifact_abi=1`。
- `runtime_protocol.name=AscendBackend`。
- OM作为PTE内部命名数据保存，外部权重作为PTD命名数据保存。
- OM和权重组均记录完整的SHA-256摘要及逻辑字节长度。
- `acl_soc_name` 同时表示GE编译目标和运行时必须精确匹配的ACL SoC名称。
- 委托后端由方法名称和序列化后的数字索引共同标识。
- 无状态产物的状态字段固定为 `{"bindings":[]}`。

清单中的权重长度均为逻辑长度。运行时分配设备内存和绑定外部权重时采用固定的
32字节对齐方式，绑定偏移也必须按32字节对齐。

`ascend_name` 保存EPAIR或 `ExportedProgram` 中的张量名称。构建CANN图后，
ACL描述符可能为边界张量提供别名：输入使用 `arg_<index>`，输出使用
`output_<index>_<optimized-op>_<edge>`。运行时进行名称校验时，同一序号的张量既可
使用原始名称，也可使用上述CANN别名；张量数量、方向、序号、数据类型、维数、
形状和字节数仍须严格匹配。

对于有状态委托后端，每个 `state.bindings` 项都将一个 `state_current` 输入映射到一个
`state_next` 输出；如果该委托后端不更新状态，则使用 `next_output_arg=-1`。运行时以
`(device_id, state_namespace, state_id)` 作为状态键。`on_prefill` 只允许有一个重置
所有者；`explicit` 只能通过公开状态API重置；`device_only` 状态不会复制回主机。
