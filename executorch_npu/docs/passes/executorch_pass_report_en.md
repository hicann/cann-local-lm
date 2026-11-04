# ExecuTorch Built-in Pass Targeted Debugging Report

## Scope and Conclusions

- ExecuTorch: `8dbbd9db19dc36276708d64a991bce91710eb266`
- Python / PyTorch: `3.11.15 / 2.12.0`
- Targeted debugging: `36/36 PASS`. Each item executes the real pass directly on a minimal graph, and saves the nodes, parameters, signatures, and key metadata before and after the pass.
- Debugging code: `executorch/tools/pass_debug/trace_builtin_passes.py`. The file header notes "Codex-written custom debugging code"; the official pass source code was not modified.
- Raw snapshot: `executorch/tools/pass_debug/pass_debug_results.json`.
- FX intuitive comparison: [36 items of `print_readable()` Before / After / unified diff](pass_print_readable_diff_en.md). To keep the main report concise, the complete graphs are placed in a collapsible appendix; the order of each item matches this report.
- ST comparison: supplementary execution of the existing related test cases in the repository yields `22 passed, 1 failed`. The failure occurs after quantization fusion has already succeeded: in the current PyTorch 2.12, three `quantized_decomposed` ops lack an out variant, causing the subsequent `ToOutVarPass` to raise an error, which does not affect the fusion conclusion of item 19.

Reproduction command:

```bash
cd /Users/a1234/Desktop/executorch
PYTHONPYCACHEPREFIX=/tmp/executorch-pass-pycache \
  /Users/a1234/miniforge3/envs/executorch/bin/python3.11 \
  tools/pass_debug/trace_builtin_passes.py \
  --output tools/pass_debug/pass_debug_results.json \
  --readable-output tools/pass_debug/pass_print_readable_diff.md
```

The following text uniformly uses the four items "Trigger / Before→After / Changes / Evidence". `ST` refers to the unittest/pytest samples in the repository; when no existing ST is available, the targeted debugging samples from this run are used as the reference.

## `remove_unused_parameters_pass` ✅

- Trigger: the model registers both a `used` and an `unused` Linear that does not participate in forward.
- Before → After: the `p_unused_weight/p_unused_bias` placeholders exist → both nodes disappear; `unused.weight/bias` are simultaneously removed from `graph_signature` and `state_dict`.
- Changes: only parameters with `InputKind.PARAMETER` and `len(placeholder.users) == 0` are deleted; compute nodes and outputs remain unchanged.
- Evidence: source code `exir/passes/remove_unused_parameters_pass.py:14`; ST `exir/tests/test_remove_unused_parameters_pass.py:34`.

## `RemoveNonCoreAtenOpGraphAssertsPass` ✅

- Trigger: an `aten._assert_tensor_metadata.default` with no users is inserted into the graph.
- Before → After: the assert node exists → the node is erased, and DCE then cleans up the dead nodes.
- Changes: only handles `_assert_tensor_metadata`, with a narrower scope than item 27.
- Evidence: source code `exir/passes/remove_graph_asserts_pass.py:43`; sample `trace_builtin_passes.py::trace_remove_non_core_assert`.

## `ReplaceViewOpsWithViewCopyOpsPass` ✅

- Trigger: `aten._unsafe_view.default(x, [2, 3])`.
- Before → After: `_unsafe_view` → `aten.view_copy.default`; ordinary view-type ops are likewise looked up by schema for the corresponding `*_copy`.
- Changes: the target is changed to an alias-free functional op; parameters, shape, and output semantics remain unchanged; HOPs are preserved.
- Evidence: PyTorch source code `torch/_export/passes/replace_view_ops_with_view_copy_ops_pass.py:44`; sample `trace_replace_view_ops`.

## `ReplaceSymSizeOpPass` ✅

- Trigger: an FX node target uses the OpOverloadPacket `torch.ops.aten.sym_size`.
- Before → After: `aten.sym_size` → `aten.sym_size.int`; similarly `sym_stride → sym_stride.int` and `sym_numel → sym_numel.default`.
- Changes: only normalizes the target type; args/users remain unchanged, avoiding failures when the subsequent `ExportPass` processes the packet.
- Evidence: source code `exir/passes/replace_sym_size_op_pass.py:20`; ST combined sample `exir/tests/test_passes.py:609`.

## `NormalizeTransposePass` ✅

- Trigger: `aten.t.default(x)`.
- Before → After: `aten.t.default` → `aten.t_copy.default`.
- Changes: switches from alias/view semantics to a functional copy; the input and tensor metadata are preserved.
- Evidence: source code `exir/passes/normalize_transpose_pass.py:11`; sample `trace_normalize_transpose`.

## `ReplaceBrokenOpsWithFunctionalOpsPass` ✅

- Trigger: an `aten.permute.default` that still carries `op.is_view == True`.
- Before → After: `aten.permute.default` → `aten.permute_copy.default`.
- Changes: appends `_copy` based on the schema name while preserving the overload; any residual view op is converted by this rule.
- Evidence: source code `exir/passes/replace_broken_ops_with_function_ops_pass.py:13`; ST `exir/tests/test_passes.py:897`.

## `ScalarToTensorPass` ✅

- Trigger: `aten.add.Tensor(x, 1)`, where the second argument of the schema requires a Tensor.
- Before → After: args are `(x, 1)` → a new tensor `get_attr _tensor_constant0` is added, and args become `(x, _tensor_constant0)`.
- Changes: only converts `float/int/bool` into real 0-D tensors; in this run `PassResult.modified=True`.
- Evidence: source code `exir/passes/scalar_to_tensor_pass.py:13`; ST `exir/tests/test_passes.py:573`.

## `SymToTensorPass` ✅

- Trigger: the dynamic dimension `x.shape[0] - 1` used as the Tensor argument of `aten.add.Tensor`.
- Before → After: a SymInt is passed directly as an argument → `aten.scalar_tensor.default(sym, dtype=torch.int32)` is inserted.
- Changes: `SymInt/SymFloat/SymBool` map to `int32/float32/bool` respectively; the original symbolic computation node is preserved.
- Evidence: source code `exir/passes/sym_to_tensor_pass.py:17`; ST `exir/tests/test_passes.py:1474`.

## `RemoveNoopPass` ✅

- Trigger: an `aten.slice_copy.Tensor` that covers the full range and keeps the shape unchanged.
- Before → After: users are rewired from `slice_copy` to the original tensor, and `slice_copy` disappears via DCE.
- Changes: identity `to.dtype/dropout` are handled the same way; if a `dq → noop → q` chain with identical qparams is formed, the whole segment can be bypassed; if qparams differ, it is preserved.
- Evidence: source code `exir/passes/remove_noop_pass.py:46`; ST `exir/tests/test_passes.py:302,1946,2041`.

## `PruneEmptyTensorsPass` ✅

- Trigger: `aten.cat.default([x, empty(0, 3)], dim=0)`.
- Before → After: the cat input list `[x, empty]` → `[x]`, and the dead chain of the empty tensor is cleaned up.
- Changes: only checks `numel()==0`; if all inputs are empty, the cat output is rewired to `edge.aten.full.default(shape, 0)`.
- Evidence: source code `exir/passes/prune_empty_tensors_pass.py:20`; ST `exir/tests/test_prune_empty_tensors_pass.py:28`.

## `RemoveToCopyPass` ✅

- Trigger: `aten._to_copy.default(x)`, where the output has exactly the same dtype/device/shape/stride as the input.
- Before → After: users are rewired to `x`, and `_to_copy` disappears via DCE.
- Changes: if any of the four attributes differs, the node is not deleted.
- Evidence: source code `exir/passes/remove_noop_pass.py:103`; sample `trace_remove_to_copy`.

## `RemoveMixedTypeOperators` ✅

- Trigger: `int64 x + float32 y`.
- Before → After: `add(x:int64, y:float32)` → `_to_copy(x, dtype=float32)` → `add(x':float32, y:float32)`.
- Changes: explicitly materializes elementwise type promotion; the allowlist is add/mul/sub/div/minimum, and whether div uses `INT_TO_FLOAT` depends on the overload/rounding mode.
- Evidence: source code `exir/passes/remove_mixed_type_operators.py:16`; ST `exir/tests/test_passes.py:196`.

## `OpReplacePass` ✅

- Trigger: the ATen `relu` OpOverload.
- Before → After: `torch.ops.aten.relu.default` → Edge dialect `ops.edge.aten.relu.default`.
- Changes: the target type changes from `OpOverload` to `EdgeOpOverload`; executor symbolic ops and the disallow list are not converted.
- Evidence: source code `exir/passes/replace_aten_with_edge_pass.py:46`; ST `exir/tests/test_passes.py:1202`.

## `MemoryFormatOpsPass` ✅

- Trigger: Edge `aten.clone.default(memory_format=contiguous_format)`.
- Before → After: `edge.aten.clone` → `edge.dim_order_ops._clone_dim_order`, with the kwarg `memory_format` → `dim_order=[0,1,2,3]`.
- Changes: `_to_copy/empty/clone` and similar ops are converted according to the mapping; `preserve_format/None` inherits the dim order from the input tensor.
- Evidence: source code `exir/passes/memory_format_ops_pass.py:24`; ST `exir/tests/test_memory_format_ops_pass.py:48`.

## `lift_constant_tensor_pass` ✅

- Trigger: the `_tensor_constant0` get_attr generated by item 7.
- Before → After: get_attr → `_lifted_tensor_constant0` placeholder; a new `InputKind.BUFFER` input spec is added, and the tensor is written into `state_dict`.
- Changes: the original node metadata is copied to the placeholder; all users are rewired to the new placeholder; the input order places it before the first user input.
- Evidence: source code `exir/program/_program.py:242`; ST combined sample `exir/tests/test_passes.py:1583`.

## `convert_constant_dim_order_pass` ✅

- Trigger: a constant obtained by transposing a `(5,4)` tensor, with shape `(4,5)` and stride `(1,4)`, which is neither contiguous nor channels-last.
- Before → After: stride `(1,4)`, `contiguous=False` → stride `(5,1)`, `contiguous=True`.
- Changes: updates both `constants/state_dict` and the corresponding placeholder's `meta['val']`; default contiguous and channels-last constants are left untouched.
- Evidence: source code `exir/passes/convert_constant_dim_order_pass.py:46`; ST `exir/tests/test_passes.py:2539`.

## `dead_code_elimination_pass` (after ATen→Edge) ✅

- Trigger: the output only uses `relu(x)`, while the bypassed `neg(x)` has no users.
- Before → After: `placeholder → relu + neg → output(relu)` → `placeholder → relu → output`.
- Changes: traverses all GraphModules/sub-GraphModules, executes `eliminate_dead_code()`, and recompiles.
- Evidence: source code `exir/passes/__init__.py:490`; ST `exir/tests/test_passes.py:1133`.

## `DebugHandleGeneratorPass` ✅

- Trigger: an exported graph of `x+x → relu`, where compute nodes carry `from_node` provenance information.
- Before → After: add/relu have no `debug_handle` → they receive `1/2` respectively; placeholder/output still have no handle.
- Changes: derived nodes sharing the same greatest ancestor share a handle; cross-subgraph cases are processed by BFS.
- Evidence: source code `exir/passes/debug_handle_generator_pass.py:21`; ST `exir/tests/test_passes.py:1306`.

## `quant_fusion_and_const_prop_pass` ✅

- Trigger: reference quantized `dq(xq) + dq(yq) → q`.
- Before → After: two dequant chains + `aten.add` + output quant → `quantized_decomposed.add`; only the necessary dequant is kept on the output side, followed by constant propagation.
- Changes: the fused quant op directly takes quantized inputs and the input/output qparams; unused q/dq/add nodes are removed by DCE.
- Evidence: source code `exir/passes/quant_fusion_pass.py:164`; ST `exir/tests/test_quant_fusion_pass.py:49`.

## `reinplace_pass` ✅

- Trigger: the mutable buffer `state.index_put_` is functionalized by export/Edge into `index_put`, and the mutated parameter has no unsafe users afterwards.
- Before → After: functional `edge.aten.index_put.default` → in-place `edge.aten.index_put_.default`.
- Changes: checks users and graph signature mutability in reverse topological order; the target is switched only when safe, otherwise the functional op is preserved.
- Evidence: source code `exir/passes/reinplace.py:269`; ST `exir/tests/test_reinplace_pass.py:44,93`.

## `weights_to_outputs_pass` ✅

- Trigger: `_export_forward_backward` generates a `GRADIENT_TO_PARAMETER(weight)` output.
- Before → After: the weight placeholder serves only as an input → `meta['weight_has_gradient']=True`, and it is appended to the output tuple and output specs, temporarily using kind `TOKEN`.
- Changes: only handles training graphs that actually have parameter gradients; inference graphs return directly.
- Evidence: source code `exir/passes/weights_to_outputs_pass.py:13`; ST/sample `extension/training/examples/XOR/export_model.py:21`.

## `unsafe_remove_auto_functionalized_pass` ✅

- Trigger: a mutable custom op forms `auto_functionalized_v2 → getitem` through decomposition.
- Before → After: the HOP wrapper/getitem → the original `codex_pass_debug.mutate.default(x)`; the downstream add reuses the mutated base.
- Changes: calls the PyTorch decomposition after setting `only_clone_these_tensors=[]`, and removes self-copies and dead nodes; this pass explicitly performs no safety checks.
- Evidence: PyTorch source code `torch/export/_remove_auto_functionalized_pass.py:25`; sample `trace_remove_auto_functionalized`.

## `insert_write_back_for_buffers_pass` ✅

- Trigger: the buffer `state.add_(1)` is represented after export as "the new value as a mutation output".
- Before → After: only the functional new value is returned → `aten.copy_(state_placeholder, new_value)` is inserted into the graph epilogue, and the mutation output is changed to copy_.
- Changes: updates the output node and output signature so that the runtime writes back in place to the real buffer; lineages already reinplaced do not get a duplicate copy inserted.
- Evidence: source code `exir/passes/insert_write_back_for_buffers_pass.py:108`; ST `exir/tests/test_passes.py:1900`.

## `SpecPropPass` ✅

- Trigger: Edge `aten.add.Tensor(x, y)`.
- Before → After: nodes only have `meta['val']/tensor_meta` → tensor nodes gain `meta['spec']=TensorSpec(shape,stride,dtype,...)`.
- Changes: placeholders default to non-const; subsequently `update_placeholder_tensor_specs` marks parameters, buffers, and lifted constants as const according to the graph signature.
- Evidence: source code `exir/passes/spec_prop_pass.py:62`; ST `exir/tests/test_passes.py:637`.

## `PropagateDevicePass` ✅

- Trigger: `DeviceAwarePartitioner('cuda:0')` generates a delegate with a `target_device` compile spec, enabling non-CPU memory planning.
- Before → After: two CPU placeholders connect directly to the delegate → one `_h2d_copy` is inserted for each; delegate/getitem specs become CUDA; one `_d2h_copy` is inserted before the output to return to CPU.
- Changes: adds 3 copy nodes and updates `TensorSpec.device/device_index`; the skip option can keep method input/output as device tensors.
- Evidence: source code `exir/passes/propagate_device_pass.py:151`; ST `exir/tests/test_propagate_device_pass.py:233`.

## `EdgeToBackendOpsPass` ✅

- Trigger: the Python `operator.sub` inside a dynamic shape expression.
- Before → After: `operator.sub(sym_size, 1)` → backend dialect `executorch_prim.sub.Scalar`.
- Changes: Python symbolic op mappings are handled first; other registered `OpOverload`s are then looked up for the corresponding `BackendOpOverload`, and the original target is kept if none is found.
- Evidence: source code `exir/passes/replace_edge_with_backend_pass.py:17`; ST `exir/tests/test_passes.py:1008`.

## `RemoveGraphAssertsPass` ✅

- Trigger: `aten._assert_async.msg`.
- Before → After: the assert node exists → it disappears after erase + DCE.
- Changes: the set includes `_assert_async.msg/_assert_scalar/sym_constrain_range_for_size/sym_constrain_range/_assert_tensor_metadata`, covering a broader range than item 2.
- Evidence: source code `exir/passes/remove_graph_asserts_pass.py:14`; ST `exir/tests/test_passes.py:1505`.

## `NormalizeViewCopyBasePass` ✅

- Trigger: three nested layers of `view_copy(view_copy(view_copy(x)))`.
- Before → After: the bases of the latter two layers point to the previous view_copy → the base of every view_copy points directly to the original `x`.
- Changes: the node count stays the same for now; only args are modified; combined with the DCE in item 29, the intermediate view_copy nodes that are no longer used are removed.
- Evidence: source code `exir/passes/normalize_view_copy_base_pass.py:24`; ST `exir/tests/test_passes.py:2113`.

## `dead_code_elimination_pass` (pre-memory) ✅

- Trigger: nodes left without users after item 28 or other passes rewire the graph; the targeted sample likewise uses a `neg` with no users.
- Before → After: dead nodes exist → they are removed from all main graphs/subgraphs.
- Changes: the implementation is the same as item 17, but the execution position differs; here it specifically reclaims dead code newly produced in the pre-memory stage.
- Evidence: source code `exir/passes/__init__.py:490`; ST `exir/tests/test_passes.py:1133,2113`.

## `ReplaceViewCopyWithViewPass` ✅

- Trigger: an `aten.view_copy(x,[2,3])` that is not a graph output, and whose base already has a `TensorSpec`.
- Before → After: `view_copy` → `executorch.exir.memory.view`; `meta['spec']` is changed to a `_ViewSpec` that shares the base storage.
- Changes: only non-output view_copy nodes are modified; a view_copy used directly as a graph output is preserved to avoid mutable runtime output pointer issues.
- Evidence: source code `exir/passes/replace_view_copy_with_view_pass.py:261`; ST `exir/tests/test_passes.py:2162`.

## `ConstraintBasedSymShapeEvalPass` ✅

- Trigger: the input/Relu spec is `shape=[s6,4,s37,s46]`, with constraint upper bounds of `2,8,8` respectively.
- Before → After: shape → `[2,4,8,8]`; stride `4*s37*s46,s37*s46,s46,1` → `[256,64,8,1]`.
- Changes: updates all `TensorSpec.shape/stride` in place; when a finite integer upper bound cannot be obtained, it explicitly raises an error to prevent incorrect memory planning.
- Evidence: source code `exir/passes/sym_shape_eval_pass.py:257`; ST `exir/tests/test_passes.py:1067`, `exir/tests/test_dynamic_shape_propagation.py:164`.

## `ToOutVarPass` ✅

- Trigger: functional `edge.aten.add.Tensor(x,y)` with SpecProp already completed.
- Before → After: a new `memory.alloc(([2], float32))` is added; the target becomes `aten.add.out`, and kwargs gain `out=alloc`.
- Changes: multi-output ops create an alloc for each out parameter; in-place ops, HOPs, skiplist entries, and nodes that are already out variants are not converted again.
- Evidence: source code `exir/passes/__init__.py:328`; ST `exir/tests/test_passes.py:344,357,440`.

## `external_constants_pass` ✅

- Trigger: a parameter placeholder with `TensorSpec.const=True` whose name is not `_lifted_tensor*`.
- Before → After: no tag → `meta['constant_tag']='_default_external_constant'`.
- Changes: the pass only applies the tag and does not write files directly; the emitter places non-lifted constants into external data files based on the tag.
- Evidence: source code `exir/passes/external_constants_pass.py:28`; sample `trace_external_constants`.

## `external_mutable_weights_pass` ✅

- Trigger: a training graph parameter that is both `spec.const=True` and has a `GRADIENT_TO_PARAMETER` output.
- Before → After: the gradient weight has no tag → `constant_tag='_default_external_constant'`; ordinary lifted constants are not tagged.
- Changes: only marks weights that are mutable during training; the decision relies on the ExportedProgram graph signature.
- Evidence: source code `exir/passes/external_constants_pass.py:67`; ST/sample `extension/training/examples/XOR/test/test_export.py:14`.

## `MemoryPlanningPass.run` ✅

- Trigger: `add.out → mul.out`, where the two allocs and all tensor specs have `mem_id/mem_offset/lifetime` initially set to `None`.
- Before → After: in the example, `add/alloc` gets `(mem_id=1, offset=16, lifetime=[2,5])` and `mul/alloc` gets `(1,0,[4,6])`; the graph records `non_const_buffer_sizes=[0,48]`.
- Changes: plans the arena based on lifetime, size, alignment, and input/output allocation strategies; non-overlapping lifetimes can reuse offsets.
- Evidence: source code `exir/passes/memory_planning_pass.py:238`; ST `exir/tests/test_memory_planning.py:503`.

## `MemoryPlanningPass.run_multimethod` ✅

- Trigger: the three methods `forward/set_state/get_state` share the mutable buffer `state`, with `share_mutable_buffers=True` enabled.
- Before → After: the `state` spec in each method has `mem_id/mem_offset=None` → unified to `mem_id=2, mem_offset=0`, with each allocated memory being 16 bytes.
- Changes: merges the same mutable state across methods by FQN, establishes a separate shared arena, and extends the buffer size table of each GraphModule.
- Evidence: source code `exir/passes/memory_planning_pass.py:303`; ST `exir/tests/test_memory_planning.py:1305`.

## Suggested Follow-up Debugging Approaches

1. When adding a new pass, copy the `@case(index, name)` structure from the debugging script and keep the same graph/EP snapshot format.
2. Prefer calling a single pass directly; only when upstream/downstream state is required, use a wrapper to capture before/after snapshots at the real pipeline entry (item 15 uses this approach).
3. Beyond analyzing the graph, the following must be checked in sync: `graph_signature`, `state_dict/constants`, `node.meta['spec']`, `debug_handle`, `constant_tag`, and `mem_id/mem_offset/lifetime`.
4. For passes such as DCE, reinplace, externalization, and memory planning, keep at least one negative case that "should not trigger", to avoid validating only the happy path.

## `print_readable()` Graph Comparison Appendix

The complete item-by-item comparison is in [`pass_print_readable_diff.md`](pass_print_readable_diff_en.md). Each pass there includes:

1. the real `GraphModule.print_readable()` before the pass;
2. the real `GraphModule.print_readable()` after the pass;
3. the automatically generated unified diff;
4. additional metadata such as `spec/debug_handle/constant_tag/weight_has_gradient/memory planning`.

If the unified diff shows the graph text is unchanged, it means the pass only modified `graph_signature`, `state_dict`, or metadata; this should be judged in combination with the "Changes" section of this report and the JSON snapshot.
