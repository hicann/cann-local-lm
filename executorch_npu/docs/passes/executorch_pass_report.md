# ExecuTorch内置Pass定向调试报告

## 范围与结论

- ExecuTorch：`8dbbd9db19dc36276708d64a991bce91710eb266`
- Python / PyTorch：`3.11.15 / 2.12.0`
- 定向调试：`36/36 PASS`。每项均使用最小图直接执行真实pass，并保存pass前后节点、参数、签名及关键metadata。
- 调试代码：`executorch/tools/pass_debug/trace_builtin_passes.py`。文件头已注明“Codex-written custom debugging code”；未修改正式pass源码。
- 原始快照：`executorch/tools/pass_debug/pass_debug_results.json`。
- FX直观对比：[36项 `print_readable()` Before / After / unified diff](pass_print_readable_diff.md)。为保持主报告凝练，完整图放在可折叠附录中；每项顺序与本报告一致。
- ST对照：仓库现有相关用例补充执行结果为 `22 passed, 1 failed`。失败发生在量化融合已经成功之后：当前PyTorch 2.12中三个 `quantized_decomposed` op缺少out variant，导致后续 `ToOutVarPass` 报错，不影响第19项融合结论。

复现命令：

```bash
cd /Users/a1234/Desktop/executorch
PYTHONPYCACHEPREFIX=/tmp/executorch-pass-pycache \
  /Users/a1234/miniforge3/envs/executorch/bin/python3.11 \
  tools/pass_debug/trace_builtin_passes.py \
  --output tools/pass_debug/pass_debug_results.json \
  --readable-output tools/pass_debug/pass_print_readable_diff.md
```

下文统一使用“触发 / 前→后 / 变化点 / 依据”四项。`ST` 指仓库中的unittest/pytest样例；无现成ST时，以本次定向调试样例为准。

## `remove_unused_parameters_pass` ✅

- 触发：模型同时注册 `used` 和未参与forward的 `unused` Linear。
- 前 → 后：`p_unused_weight/p_unused_bias` placeholder存在 → 两节点消失；`unused.weight/bias` 同时从 `graph_signature`、`state_dict` 删除。
- 变化点：只删 `InputKind.PARAMETER` 且 `len(placeholder.users) == 0` 的参数；计算节点与输出不变。
- 依据：源码 `exir/passes/remove_unused_parameters_pass.py:14`；ST `exir/tests/test_remove_unused_parameters_pass.py:34`。

## `RemoveNonCoreAtenOpGraphAssertsPass` ✅

- 触发：图中插入无用户的 `aten._assert_tensor_metadata.default`。
- 前 → 后：assert节点存在 → 节点被erase，随后DCE清理死节点。
- 变化点：仅处理 `_assert_tensor_metadata`，范围小于第27项。
- 依据：源码 `exir/passes/remove_graph_asserts_pass.py:43`；样例 `trace_builtin_passes.py::trace_remove_non_core_assert`。

## `ReplaceViewOpsWithViewCopyOpsPass` ✅

- 触发：`aten._unsafe_view.default(x, [2, 3])`。
- 前 → 后：`_unsafe_view` → `aten.view_copy.default`；普通view类op同样按schema查找对应 `*_copy`。
- 变化点：target改为无alias的functional op，参数、shape和输出语义不变；HOP保留。
- 依据：PyTorch源码 `torch/_export/passes/replace_view_ops_with_view_copy_ops_pass.py:44`；样例 `trace_replace_view_ops`。

## `ReplaceSymSizeOpPass` ✅

- 触发：FX节点target使用OpOverloadPacket `torch.ops.aten.sym_size`。
- 前 → 后：`aten.sym_size` → `aten.sym_size.int`；同理 `sym_stride → sym_stride.int`、`sym_numel → sym_numel.default`。
- 变化点：只规范target类型，args/users不变，避免后续 `ExportPass` 处理packet失败。
- 依据：源码 `exir/passes/replace_sym_size_op_pass.py:20`；ST组合样例 `exir/tests/test_passes.py:609`。

## `NormalizeTransposePass` ✅

- 触发：`aten.t.default(x)`。
- 前 → 后：`aten.t.default` → `aten.t_copy.default`。
- 变化点：由alias/view语义改为functional copy，输入和tensor metadata保留。
- 依据：源码 `exir/passes/normalize_transpose_pass.py:11`；样例 `trace_normalize_transpose`。

## `ReplaceBrokenOpsWithFunctionalOpsPass` ✅

- 触发：仍带 `op.is_view == True` 的 `aten.permute.default`。
- 前 → 后：`aten.permute.default` → `aten.permute_copy.default`。
- 变化点：依据schema名拼接 `_copy` 并保留overload；任何残留view op均按此规则转换。
- 依据：源码 `exir/passes/replace_broken_ops_with_function_ops_pass.py:13`；ST `exir/tests/test_passes.py:897`。

## `ScalarToTensorPass` ✅

- 触发：`aten.add.Tensor(x, 1)`，schema的第二个参数要求Tensor。
- 前 → 后：args为 `(x, 1)` → 新增tensor `get_attr _tensor_constant0`，args变为 `(x, _tensor_constant0)`。
- 变化点：仅将 `float/int/bool` 转成真实0-D tensor；本次 `PassResult.modified=True`。
- 依据：源码 `exir/passes/scalar_to_tensor_pass.py:13`；ST `exir/tests/test_passes.py:573`。

## `SymToTensorPass` ✅

- 触发：动态维 `x.shape[0] - 1` 作为 `aten.add.Tensor` 的Tensor参数。
- 前 → 后：SymInt直接入参 → 插入 `aten.scalar_tensor.default(sym, dtype=torch.int32)`。
- 变化点：`SymInt/SymFloat/SymBool` 分别映射 `int32/float32/bool`；原符号计算节点保留。
- 依据：源码 `exir/passes/sym_to_tensor_pass.py:17`；ST `exir/tests/test_passes.py:1474`。

## `RemoveNoopPass` ✅

- 触发：覆盖完整范围且shape不变的 `aten.slice_copy.Tensor`。
- 前 → 后：用户从 `slice_copy` 改连原tensor，`slice_copy` 经DCE消失。
- 变化点：同样处理identity `to.dtype/dropout`；若形成同qparams的 `dq → noop → q`，整段可旁路；qparams不同则保留。
- 依据：源码 `exir/passes/remove_noop_pass.py:46`；ST `exir/tests/test_passes.py:302,1946,2041`。

## `PruneEmptyTensorsPass` ✅

- 触发：`aten.cat.default([x, empty(0, 3)], dim=0)`。
- 前 → 后：cat输入列表 `[x, empty]` → `[x]`，空tensor的死链被清理。
- 变化点：只检查 `numel()==0`；若全部为空，cat输出改接 `edge.aten.full.default(shape, 0)`。
- 依据：源码 `exir/passes/prune_empty_tensors_pass.py:20`；ST `exir/tests/test_prune_empty_tensors_pass.py:28`。

## `RemoveToCopyPass` ✅

- 触发：`aten._to_copy.default(x)`，输出与输入的dtype/device/shape/stride全同。
- 前 → 后：用户改连 `x`，`_to_copy` 经DCE消失。
- 变化点：四项属性任一不同即不删除。
- 依据：源码 `exir/passes/remove_noop_pass.py:103`；样例 `trace_remove_to_copy`。

## `RemoveMixedTypeOperators` ✅

- 触发：`int64 x + float32 y`。
- 前 → 后：`add(x:int64, y:float32)` → `_to_copy(x, dtype=float32)` → `add(x':float32, y:float32)`。
- 变化点：显式固化elementwise type promotion；allowlist为add/mul/sub/div/minimum，div是否 `INT_TO_FLOAT` 取决于overload/rounding mode。
- 依据：源码 `exir/passes/remove_mixed_type_operators.py:16`；ST `exir/tests/test_passes.py:196`。

## `OpReplacePass` ✅

- 触发：ATen `relu` OpOverload。
- 前 → 后：`torch.ops.aten.relu.default` → Edge dialect `ops.edge.aten.relu.default`。
- 变化点：target类型由 `OpOverload` 变为 `EdgeOpOverload`；executor symbolic ops和disallow list不转换。
- 依据：源码 `exir/passes/replace_aten_with_edge_pass.py:46`；ST `exir/tests/test_passes.py:1202`。

## `MemoryFormatOpsPass` ✅

- 触发：Edge `aten.clone.default(memory_format=contiguous_format)`。
- 前 → 后：`edge.aten.clone` → `edge.dim_order_ops._clone_dim_order`，kwarg `memory_format` → `dim_order=[0,1,2,3]`。
- 变化点：`_to_copy/empty/clone` 等按映射转换；`preserve_format/None` 从输入tensor继承dim order。
- 依据：源码 `exir/passes/memory_format_ops_pass.py:24`；ST `exir/tests/test_memory_format_ops_pass.py:48`。

## `lift_constant_tensor_pass` ✅

- 触发：第7项生成的 `_tensor_constant0` get_attr。
- 前 → 后：get_attr → `_lifted_tensor_constant0` placeholder；新增 `InputKind.BUFFER` input spec，tensor写入 `state_dict`。
- 变化点：原node metadata拷到placeholder；所有users改连新placeholder；输入顺序放在第一个user input前。
- 依据：源码 `exir/program/_program.py:242`；ST组合样例 `exir/tests/test_passes.py:1583`。

## `convert_constant_dim_order_pass` ✅

- 触发：常量由 `(5,4)` tensor转置得到，shape `(4,5)`、stride `(1,4)`，既非contiguous也非channels-last。
- 前 → 后：stride `(1,4)`、`contiguous=False` → stride `(5,1)`、`contiguous=True`。
- 变化点：同时更新 `constants/state_dict` 与对应placeholder的 `meta['val']`；默认contiguous和channels-last不动。
- 依据：源码 `exir/passes/convert_constant_dim_order_pass.py:46`；ST `exir/tests/test_passes.py:2539`。

## `dead_code_elimination_pass`（ATen→Edge后）✅

- 触发：输出只使用 `relu(x)`，旁路 `neg(x)` 无用户。
- 前 → 后：`placeholder → relu + neg → output(relu)` → `placeholder → relu → output`。
- 变化点：遍历全部GraphModule/子GraphModule，执行 `eliminate_dead_code()` 并recompile。
- 依据：源码 `exir/passes/__init__.py:490`；ST `exir/tests/test_passes.py:1133`。

## `DebugHandleGeneratorPass` ✅

- 触发：`x+x → relu` 的exported graph，计算节点带 `from_node` 溯源信息。
- 前 → 后：add/relu无 `debug_handle` → 分别得到 `1/2`；placeholder/output仍无handle。
- 变化点：同一greatest ancestor的派生节点共享handle，跨子图按BFS处理。
- 依据：源码 `exir/passes/debug_handle_generator_pass.py:21`；ST `exir/tests/test_passes.py:1306`。

## `quant_fusion_and_const_prop_pass` ✅

- 触发：reference quantized `dq(xq) + dq(yq) → q`。
- 前 → 后：两条dequant + `aten.add` + output quant → `quantized_decomposed.add`；输出端仅保留必要dequant，随后执行constant propagation。
- 变化点：融合后的quant op直接接量化输入及输入/输出qparams；无用q/dq/add被DCE。
- 依据：源码 `exir/passes/quant_fusion_pass.py:164`；ST `exir/tests/test_quant_fusion_pass.py:49`。

## `reinplace_pass` ✅

- 触发：mutable buffer `state.index_put_` 被export/Edge functionalize为 `index_put`，且被修改参数后续无不安全用户。
- 前 → 后：functional `edge.aten.index_put.default` → in-place `edge.aten.index_put_.default`。
- 变化点：反向拓扑检查users与graph signature的可变性；安全才换target，不安全则保留functional op。
- 依据：源码 `exir/passes/reinplace.py:269`；ST `exir/tests/test_reinplace_pass.py:44,93`。

## `weights_to_outputs_pass` ✅

- 触发：`_export_forward_backward` 生成 `GRADIENT_TO_PARAMETER(weight)` 输出。
- 前 → 后：weight placeholder仅作输入 → `meta['weight_has_gradient']=True`，并追加到output tuple与output specs，kind暂用 `TOKEN`。
- 变化点：只处理确实存在parameter gradient的训练图；推理图直接返回。
- 依据：源码 `exir/passes/weights_to_outputs_pass.py:13`；ST/样例 `extension/training/examples/XOR/export_model.py:21`。

## `unsafe_remove_auto_functionalized_pass` ✅

- 触发：mutable custom op经decomposition形成 `auto_functionalized_v2 → getitem`。
- 前 → 后：HOP wrapper/getitem → 原始 `codex_pass_debug.mutate.default(x)`；下游add重新使用被修改的base。
- 变化点：设置 `only_clone_these_tensors=[]` 后调用PyTorch decomposition，并移除self-copy与死节点；该pass明确不做安全性检查。
- 依据：PyTorch源码 `torch/export/_remove_auto_functionalized_pass.py:25`；样例 `trace_remove_auto_functionalized`。

## `insert_write_back_for_buffers_pass` ✅

- 触发：buffer `state.add_(1)` 在export后以“新值作为mutation output”表示。
- 前 → 后：仅返回functional新值 → graph epilogue插入 `aten.copy_(state_placeholder, new_value)`，mutation output改为copy_。
- 变化点：更新output node和output signature，使运行时原位写回真实buffer；已reinplace的lineage不重复插copy。
- 依据：源码 `exir/passes/insert_write_back_for_buffers_pass.py:108`；ST `exir/tests/test_passes.py:1900`。

## `SpecPropPass` ✅

- 触发：Edge `aten.add.Tensor(x, y)`。
- 前 → 后：节点只有 `meta['val']/tensor_meta` → tensor节点增加 `meta['spec']=TensorSpec(shape,stride,dtype,...)`。
- 变化点：placeholder默认非const；随后 `update_placeholder_tensor_specs` 会按graph signature将参数、buffer、lifted constant标为const。
- 依据：源码 `exir/passes/spec_prop_pass.py:62`；ST `exir/tests/test_passes.py:637`。

## `PropagateDevicePass` ✅

- 触发：`DeviceAwarePartitioner('cuda:0')` 生成带 `target_device` compile spec的delegate，启用non-CPU memory planning。
- 前 → 后：两个CPU placeholder直连delegate → 各插一个 `_h2d_copy`；delegate/getitem spec变CUDA；输出前插一个 `_d2h_copy` 回CPU。
- 变化点：新增3个copy节点并更新 `TensorSpec.device/device_index`；skip选项可让method input/output保持device tensor。
- 依据：源码 `exir/passes/propagate_device_pass.py:151`；ST `exir/tests/test_propagate_device_pass.py:233`。

## `EdgeToBackendOpsPass` ✅

- 触发：动态shape表达式中的Python `operator.sub`。
- 前 → 后：`operator.sub(sym_size, 1)` → backend dialect `executorch_prim.sub.Scalar`。
- 变化点：先处理Python symbolic op映射；其它已注册 `OpOverload` 再查找对应 `BackendOpOverload`，查不到则保持原target。
- 依据：源码 `exir/passes/replace_edge_with_backend_pass.py:17`；ST `exir/tests/test_passes.py:1008`。

## `RemoveGraphAssertsPass` ✅

- 触发：`aten._assert_async.msg`。
- 前 → 后：assert节点存在 → erase + DCE后消失。
- 变化点：集合包含 `_assert_async.msg/_assert_scalar/sym_constrain_range_for_size/sym_constrain_range/_assert_tensor_metadata`，比第2项覆盖更广。
- 依据：源码 `exir/passes/remove_graph_asserts_pass.py:14`；ST `exir/tests/test_passes.py:1505`。

## `NormalizeViewCopyBasePass` ✅

- 触发：三层 `view_copy(view_copy(view_copy(x)))`。
- 前 → 后：后两层base指向上一个view_copy → 每个view_copy的base都直接指向原始 `x`。
- 变化点：节点数量暂不变，只改args；配合第29项DCE才移除不再使用的中间view_copy。
- 依据：源码 `exir/passes/normalize_view_copy_base_pass.py:24`；ST `exir/tests/test_passes.py:2113`。

## `dead_code_elimination_pass`（pre-memory）✅

- 触发：第28项或其它pass改线后留下无用户节点；定向样例同样使用无用户 `neg`。
- 前 → 后：死节点存在 → 从所有主图/子图删除。
- 变化点：实现与第17项相同，但执行位置不同；这里专门回收pre-memory阶段新产生的dead code。
- 依据：源码 `exir/passes/__init__.py:490`；ST `exir/tests/test_passes.py:1133,2113`。

## `ReplaceViewCopyWithViewPass` ✅

- 触发：非graph output的 `aten.view_copy(x,[2,3])`，且base已有 `TensorSpec`。
- 前 → 后：`view_copy` → `executorch.exir.memory.view`；`meta['spec']` 改为共享base storage的 `_ViewSpec`。
- 变化点：只改非输出view_copy；直接作为graph output的view_copy保留，避免运行时输出指针可变问题。
- 依据：源码 `exir/passes/replace_view_copy_with_view_pass.py:261`；ST `exir/tests/test_passes.py:2162`。

## `ConstraintBasedSymShapeEvalPass` ✅

- 触发：输入/Relu spec为 `shape=[s6,4,s37,s46]`，约束上界分别为 `2,8,8`。
- 前 → 后：shape → `[2,4,8,8]`；stride `4*s37*s46,s37*s46,s46,1` → `[256,64,8,1]`。
- 变化点：原地更新所有 `TensorSpec.shape/stride`；无法得到有限整数上界时明确报错，阻止错误内存规划。
- 依据：源码 `exir/passes/sym_shape_eval_pass.py:257`；ST `exir/tests/test_passes.py:1067`、`exir/tests/test_dynamic_shape_propagation.py:164`。

## `ToOutVarPass` ✅

- 触发：functional `edge.aten.add.Tensor(x,y)` 且已完成SpecProp。
- 前 → 后：新增 `memory.alloc(([2], float32))`；target变 `aten.add.out`，kwargs增加 `out=alloc`。
- 变化点：多输出op为每个out参数建alloc；in-place、HOP、skiplist与已是out variant的节点不重复转换。
- 依据：源码 `exir/passes/__init__.py:328`；ST `exir/tests/test_passes.py:344,357,440`。

## `external_constants_pass` ✅

- 触发：参数placeholder的 `TensorSpec.const=True`，名称不是 `_lifted_tensor*`。
- 前 → 后：无tag → `meta['constant_tag']='_default_external_constant'`。
- 变化点：pass只打tag，不直接写文件；emitter根据tag把非lifted常量放入外部数据文件。
- 依据：源码 `exir/passes/external_constants_pass.py:28`；样例 `trace_external_constants`。

## `external_mutable_weights_pass` ✅

- 触发：训练图parameter同时 `spec.const=True` 且有 `GRADIENT_TO_PARAMETER` 输出。
- 前 → 后：gradient weight无tag → `constant_tag='_default_external_constant'`；普通lifted constant不打tag。
- 变化点：只标记训练中可变权重，判定依赖ExportedProgram graph signature。
- 依据：源码 `exir/passes/external_constants_pass.py:67`；ST/样例 `extension/training/examples/XOR/test/test_export.py:14`。

## `MemoryPlanningPass.run` ✅

- 触发：`add.out → mul.out`，两个alloc及全部tensor spec的 `mem_id/mem_offset/lifetime` 初始为 `None`。
- 前 → 后：示例中 `add/alloc` 得 `(mem_id=1, offset=16, lifetime=[2,5])`，`mul/alloc` 得 `(1,0,[4,6])`；图记录 `non_const_buffer_sizes=[0,48]`。
- 变化点：根据lifetime、大小、对齐及input/output分配策略规划arena；不重叠lifetime可复用offset。
- 依据：源码 `exir/passes/memory_planning_pass.py:238`；ST `exir/tests/test_memory_planning.py:503`。

## `MemoryPlanningPass.run_multimethod` ✅

- 触发：`forward/set_state/get_state` 三个method共享mutable buffer `state`，启用 `share_mutable_buffers=True`。
- 前 → 后：各method中 `state` spec的 `mem_id/mem_offset=None` → 统一 `mem_id=2, mem_offset=0`，每份allocated memory均为16 bytes。
- 变化点：按FQN合并跨method的同一mutable state，单独建立共享arena，并扩充各GraphModule的buffer size表。
- 依据：源码 `exir/passes/memory_planning_pass.py:303`；ST `exir/tests/test_memory_planning.py:1305`。

## 建议的后续调试方式

1. 新增pass时复制调试脚本中的 `@case(index, name)` 结构，保留同一套graph/EP快照格式。
2. 优先直接调用单个pass；只有必须依赖上下游状态时，才用wrapper在真实pipeline入口抓取前后快照（第15项即采用此方式）。
3. 分析graph之外必须同步检查：`graph_signature`、`state_dict/constants`、`node.meta['spec']`、`debug_handle`、`constant_tag`、`mem_id/mem_offset/lifetime`。
4. 对DCE、reinplace、externalization、memory planning一类pass，至少同时保留一个“不应触发”的负例，防止只验证happy path。

## `print_readable()` 图对比附录

完整逐项对比见 [`pass_print_readable_diff.md`](pass_print_readable_diff.md)。其中每个pass都包含：

1. pass前的真实 `GraphModule.print_readable()`；
2. pass后的真实 `GraphModule.print_readable()`；
3. 自动生成的unified diff；
4. `spec/debug_handle/constant_tag/weight_has_gradient/memory planning` 等additional metadata。

若unified diff显示图文本不变，说明该pass只修改 `graph_signature`、`state_dict` 或metadata；应结合本报告的“变化点”和JSON快照判断。
