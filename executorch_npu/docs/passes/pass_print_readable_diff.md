# ExecuTorch Pass `print_readable()` Before / After

> 本文由 `trace_builtin_passes.py` 自动生成。每段主体均来自真实
> `GraphModule.print_readable(print_output=False, include_stride=True,
> include_device=True, additional_meta=[...])`，不是手写伪图。为展示
> placeholder/output的metadata，原图前附有明确标记的Codex摘要；
> 未加工文本保存在JSON的 `print_readable_raw`。

图结构未变化但pass修改metadata/signature/state_dict时，diff可能为空；
此时以Before/After中的additional metadata和主报告的“变化点”为准。

## `remove_unused_parameters_pass`

关键变化：unused.weight/bias are removed from placeholders, graph signature, and state_dict

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, p_used_weight: "f32[4, 4][4, 1]cpu", p_used_bias: "f32[4][1]cpu", p_unused_weight: "f32[4, 4][4, 1]cpu", p_unused_bias: "f32[4][1]cpu", x: "f32[1, 4][4, 1]cpu"):
        # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        linear: "f32[1, 4][4, 1]cpu" = torch.ops.aten.linear.default(x, p_used_weight, p_used_bias);  x = p_used_weight = p_used_bias = None
        return (linear,)
```

### After · `after.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, p_used_weight: "f32[4, 4][4, 1]cpu", p_used_bias: "f32[4][1]cpu", x: "f32[1, 4][4, 1]cpu"):
        # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        linear: "f32[1, 4][4, 1]cpu" = torch.ops.aten.linear.default(x, p_used_weight, p_used_bias);  x = p_used_weight = p_used_bias = None
        return (linear,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 class GraphModule(torch.nn.Module):
-    def forward(self, p_used_weight: "f32[4, 4][4, 1]cpu", p_used_bias: "f32[4][1]cpu", p_unused_weight: "f32[4, 4][4, 1]cpu", p_unused_bias: "f32[4][1]cpu", x: "f32[1, 4][4, 1]cpu"):
+    def forward(self, p_used_weight: "f32[4, 4][4, 1]cpu", p_used_bias: "f32[4][1]cpu", x: "f32[1, 4][4, 1]cpu"):
         # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
         linear: "f32[1, 4][4, 1]cpu" = torch.ops.aten.linear.default(x, p_used_weight, p_used_bias);  x = p_used_weight = p_used_bias = None
         return (linear,)
```

</details>

## `RemoveNonCoreAtenOpGraphAssertsPass`

关键变化：_assert_tensor_metadata and dead users are erased

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default();  _assert_tensor_metadata_default = None
        return x
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        return x
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,3 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x):
-        # No stacktrace found for following nodes
-        _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default();  _assert_tensor_metadata_default = None
         return x
```

</details>

## `ReplaceViewOpsWithViewCopyOpsPass`

关键变化：_unsafe_view/view becomes functional view_copy

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class <lambda>(torch.nn.Module):
    def forward(self, x_1: "f32[6][1]cpu"):
        # No stacktrace found for following nodes
        _unsafe_view: "f32[2, 3][3, 1]cpu" = torch.ops.aten._unsafe_view.default(x_1, [2, 3]);  x_1 = None
        return _unsafe_view
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x_1: "f32[6][1]cpu"):
        # No stacktrace found for following nodes
        view_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(x_1, [2, 3]);  x_1 = None
        return view_copy
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
-class <lambda>(torch.nn.Module):
+class GraphModule(torch.nn.Module):
     def forward(self, x_1: "f32[6][1]cpu"):
         # No stacktrace found for following nodes
-        _unsafe_view: "f32[2, 3][3, 1]cpu" = torch.ops.aten._unsafe_view.default(x_1, [2, 3]);  x_1 = None
-        return _unsafe_view
+        view_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(x_1, [2, 3]);  x_1 = None
+        return view_copy
```

</details>

## `ReplaceSymSizeOpPass`

关键变化：OpOverloadPacket sym_size becomes sym_size.int

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        sym_size = torch.ops.aten.sym_size(x, 0);  x = None
        return sym_size
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        sym_size = torch.ops.aten.sym_size.int(x, 0);  x = None
        return sym_size
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x):
         # No stacktrace found for following nodes
-        sym_size = torch.ops.aten.sym_size(x, 0);  x = None
+        sym_size = torch.ops.aten.sym_size.int(x, 0);  x = None
         return sym_size
```

</details>

## `NormalizeTransposePass`

关键变化：aten.t.default becomes aten.t_copy.default

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        t: "f32[3, 2][1, 3]cpu" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
        return t
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        t_copy: "f32[3, 2][2, 1]cpu" = torch.ops.aten.t_copy.default(arg0_1);  arg0_1 = None
        return t_copy
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
-class <lambda>(torch.nn.Module):
+class GraphModule(torch.nn.Module):
     def forward(self, arg0_1: "f32[2, 3][3, 1]cpu"):
         # No stacktrace found for following nodes
-        t: "f32[3, 2][1, 3]cpu" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
-        return t
+        t_copy: "f32[3, 2][2, 1]cpu" = torch.ops.aten.t_copy.default(arg0_1);  arg0_1 = None
+        return t_copy
```

</details>

## `ReplaceBrokenOpsWithFunctionalOpsPass`

关键变化：view-like permute becomes permute_copy

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class <lambda>(torch.nn.Module):
    def forward(self, x_1: "f32[2, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        permute: "f32[3, 2][1, 3]cpu" = torch.ops.aten.permute.default(x_1, [1, 0]);  x_1 = None
        return permute
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x_1: "f32[2, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        permute_copy: "f32[3, 2][2, 1]cpu" = torch.ops.aten.permute_copy.default(x_1, [1, 0]);  x_1 = None
        return permute_copy
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
-class <lambda>(torch.nn.Module):
+class GraphModule(torch.nn.Module):
     def forward(self, x_1: "f32[2, 3][3, 1]cpu"):
         # No stacktrace found for following nodes
-        permute: "f32[3, 2][1, 3]cpu" = torch.ops.aten.permute.default(x_1, [1, 0]);  x_1 = None
-        return permute
+        permute_copy: "f32[3, 2][2, 1]cpu" = torch.ops.aten.permute_copy.default(x_1, [1, 0]);  x_1 = None
+        return permute_copy
```

</details>

## `ScalarToTensorPass`

关键变化：Python scalar is materialized as a tensor get_attr

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class <lambda>(torch.nn.Module):
    def forward(self, x_1: "f32[2][1]cpu"):
        # No stacktrace found for following nodes
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        return add
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x_1: "f32[2][1]cpu"):
        # No stacktrace found for following nodes
        _tensor_constant0: "i64[][]cpu" = self._tensor_constant0
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x_1, _tensor_constant0);  x_1 = _tensor_constant0 = None
        return add
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,6 @@
-class <lambda>(torch.nn.Module):
+class GraphModule(torch.nn.Module):
     def forward(self, x_1: "f32[2][1]cpu"):
         # No stacktrace found for following nodes
-        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
+        _tensor_constant0: "i64[][]cpu" = self._tensor_constant0
+        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x_1, _tensor_constant0);  x_1 = _tensor_constant0 = None
         return add
```

</details>

## `SymToTensorPass`

关键变化：SymInt input is wrapped by scalar_tensor(dtype=int32)

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s6, 2][2, 1]cpu"):
        # No stacktrace found for following nodes
        sym_size_int_1: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:347 in forward, code: return torch.ops.aten.add.Tensor(x, x.shape[0] - 1)
        sub: "Sym(s6 - 1)" = sym_size_int_1 - 1;  sym_size_int_1 = None
        add: "f32[s6, 2][2, 1]cpu" = torch.ops.aten.add.Tensor(x, sub);  x = sub = None
        return (add,)
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s6, 2][2, 1]cpu"):
        # No stacktrace found for following nodes
        sym_size: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:347 in forward, code: return torch.ops.aten.add.Tensor(x, x.shape[0] - 1)
        sub: "Sym(s6 - 1)" = sym_size - 1;  sym_size = None
        scalar_tensor: "i32[][]cpu" = torch.ops.aten.scalar_tensor.default(sub, dtype = torch.int32);  sub = None
        add: "f32[s6, 2][2, 1]cpu" = torch.ops.aten.add.Tensor(x, scalar_tensor);  x = scalar_tensor = None
        return (add,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,9 +1,10 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[s6, 2][2, 1]cpu"):
         # No stacktrace found for following nodes
-        sym_size_int_1: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)
+        sym_size: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:347 in forward, code: return torch.ops.aten.add.Tensor(x, x.shape[0] - 1)
-        sub: "Sym(s6 - 1)" = sym_size_int_1 - 1;  sym_size_int_1 = None
-        add: "f32[s6, 2][2, 1]cpu" = torch.ops.aten.add.Tensor(x, sub);  x = sub = None
+        sub: "Sym(s6 - 1)" = sym_size - 1;  sym_size = None
+        scalar_tensor: "i32[][]cpu" = torch.ops.aten.scalar_tensor.default(sub, dtype = torch.int32);  sub = None
+        add: "f32[s6, 2][2, 1]cpu" = torch.ops.aten.add.Tensor(x, scalar_tensor);  x = scalar_tensor = None
         return (add,)
```

</details>

## `RemoveNoopPass`

关键变化：full-range slice_copy is bypassed and DCE removes it

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class <lambda>(torch.nn.Module):
    def forward(self, x_1: "f32[2, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        slice_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten.slice_copy.Tensor(x_1, 0, 0, 9223372036854775807);  x_1 = None
        return slice_copy
```

### After · `after`

```python
class <lambda>(torch.nn.Module):
    def forward(self, x_1: "f32[2, 3][3, 1]cpu"):
        return x_1
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,3 @@
 class <lambda>(torch.nn.Module):
     def forward(self, x_1: "f32[2, 3][3, 1]cpu"):
-        # No stacktrace found for following nodes
-        slice_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten.slice_copy.Tensor(x_1, 0, 0, 9223372036854775807);  x_1 = None
-        return slice_copy
+        return x_1
```

</details>

## `PruneEmptyTensorsPass`

关键变化：empty cat input is removed from the cat argument list

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class <lambda>(torch.nn.Module):
    def forward(self, x_1: "f32[2, 3][3, 1]cpu", empty_1: "f32[0, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        cat: "f32[2, 3][3, 1]cpu" = torch.ops.aten.cat.default([x_1, empty_1]);  x_1 = empty_1 = None
        return cat
```

### After · `after`

```python
class <lambda>(torch.nn.Module):
    def forward(self, x_1: "f32[2, 3][3, 1]cpu", empty_1: "f32[0, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        cat: "f32[2, 3][3, 1]cpu" = torch.ops.aten.cat.default([x_1]);  x_1 = None
        return cat
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 class <lambda>(torch.nn.Module):
     def forward(self, x_1: "f32[2, 3][3, 1]cpu", empty_1: "f32[0, 3][3, 1]cpu"):
         # No stacktrace found for following nodes
-        cat: "f32[2, 3][3, 1]cpu" = torch.ops.aten.cat.default([x_1, empty_1]);  x_1 = empty_1 = None
+        cat: "f32[2, 3][3, 1]cpu" = torch.ops.aten.cat.default([x_1]);  x_1 = None
         return cat
```

</details>

## `RemoveToCopyPass`

关键变化：identity _to_copy is bypassed and eliminated

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 3][3, 1]cpu"):
        # No stacktrace found for following nodes
        _to_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten._to_copy.default(arg0_1);  arg0_1 = None
        return _to_copy
```

### After · `after`

```python
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 3][3, 1]cpu"):
        return arg0_1
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,3 @@
 class <lambda>(torch.nn.Module):
     def forward(self, arg0_1: "f32[2, 3][3, 1]cpu"):
-        # No stacktrace found for following nodes
-        _to_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten._to_copy.default(arg0_1);  arg0_1 = None
-        return _to_copy
+        return arg0_1
```

</details>

## `RemoveMixedTypeOperators`

关键变化：_to_copy promotes int64 input before float32 add

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "i64[2][1]cpu", y: "f32[2][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:395 in forward, code: return torch.ops.aten.add.Tensor(x, y)
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x, y);  x = y = None
        return (add,)
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "i64[2][1]cpu", y: "f32[2][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:395 in forward, code: return torch.ops.aten.add.Tensor(x, y)
        _to_copy: "f32[2][1]cpu" = torch.ops.aten._to_copy.default(x, dtype = torch.float32);  x = None
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(_to_copy, y);  _to_copy = y = None
        return (add,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,6 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x: "i64[2][1]cpu", y: "f32[2][1]cpu"):
         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:395 in forward, code: return torch.ops.aten.add.Tensor(x, y)
-        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x, y);  x = y = None
+        _to_copy: "f32[2][1]cpu" = torch.ops.aten._to_copy.default(x, dtype = torch.float32);  x = None
+        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(_to_copy, y);  _to_copy = y = None
         return (add,)
```

</details>

## `OpReplacePass`

关键变化：ATen OpOverload is replaced by EdgeOpOverload

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, input: "f32[2][1]cpu"):
        input_1 = input

        # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/_dynamo/functional_export.py:218 in forward, code: res = self._export_root(*args, **kwargs)
        relu: "f32[2][1]cpu" = torch.ops.aten.relu.default(input_1);  input_1 = None
        return (relu,)
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, input: "f32[2][1]cpu"):
        input_1 = input

        # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/_dynamo/functional_export.py:218 in forward, code: res = self._export_root(*args, **kwargs)
        aten_relu_default: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_relu_default(input_1);  input_1 = None
        return (aten_relu_default,)
```

### Unified diff

```diff
--- before
+++ after
@@ -3,5 +3,5 @@
         input_1 = input

         # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/_dynamo/functional_export.py:218 in forward, code: res = self._export_root(*args, **kwargs)
-        relu: "f32[2][1]cpu" = torch.ops.aten.relu.default(input_1);  input_1 = None
-        return (relu,)
+        aten_relu_default: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_relu_default(input_1);  input_1 = None
+        return (aten_relu_default,)
```

</details>

## `MemoryFormatOpsPass`

关键变化：edge aten.clone(memory_format) becomes dim_order_ops._clone_dim_order(dim_order=[0,1,2,3])

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[1, 2, 3, 4][24, 12, 4, 1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:423 in forward, code: return torch.ops.aten.clone.default(
        aten_clone_default: "f32[1, 2, 3, 4][24, 12, 4, 1]cpu" = executorch_exir_dialects_edge__ops_aten_clone_default(x, memory_format = torch.contiguous_format);  x = None
        return (aten_clone_default,)
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[1, 2, 3, 4][24, 12, 4, 1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:423 in forward, code: return torch.ops.aten.clone.default(
        dim_order_ops__clone_dim_order_default: "f32[1, 2, 3, 4][24, 12, 4, 1]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default(x, dim_order = [0, 1, 2, 3]);  x = None
        return (dim_order_ops__clone_dim_order_default,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[1, 2, 3, 4][24, 12, 4, 1]cpu"):
         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:423 in forward, code: return torch.ops.aten.clone.default(
-        aten_clone_default: "f32[1, 2, 3, 4][24, 12, 4, 1]cpu" = executorch_exir_dialects_edge__ops_aten_clone_default(x, memory_format = torch.contiguous_format);  x = None
-        return (aten_clone_default,)
+        dim_order_ops__clone_dim_order_default: "f32[1, 2, 3, 4][24, 12, 4, 1]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default(x, dim_order = [0, 1, 2, 3]);  x = None
+        return (dim_order_ops__clone_dim_order_default,)
```

</details>

## `lift_constant_tensor_pass`

关键变化：tensor get_attr becomes a buffer placeholder and is added to signature/state_dict

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:444 in forward, code: return torch.ops.aten.add.Tensor(x, 1)
        _tensor_constant0: "i64[][]cpu" = self._tensor_constant0
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_tensor_constant0, dtype = torch.float32, dim_order = []);  _tensor_constant0 = None
        aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, dim_order_ops__to_dim_order_copy_default);  x = dim_order_ops__to_dim_order_copy_default = None
        return (aten_add_tensor,)
```

### After · `after.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, _lifted_tensor_constant0: "i64[][]cpu", x: "f32[2][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:444 in forward, code: return torch.ops.aten.add.Tensor(x, 1)
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant0, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant0 = None
        aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, dim_order_ops__to_dim_order_copy_default);  x = dim_order_ops__to_dim_order_copy_default = None
        return (aten_add_tensor,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,7 +1,6 @@
 class GraphModule(torch.nn.Module):
-    def forward(self, x: "f32[2][1]cpu"):
+    def forward(self, _lifted_tensor_constant0: "i64[][]cpu", x: "f32[2][1]cpu"):
         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:444 in forward, code: return torch.ops.aten.add.Tensor(x, 1)
-        _tensor_constant0: "i64[][]cpu" = self._tensor_constant0
-        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_tensor_constant0, dtype = torch.float32, dim_order = []);  _tensor_constant0 = None
+        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant0, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant0 = None
         aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, dim_order_ops__to_dim_order_copy_default);  x = dim_order_ops__to_dim_order_copy_default = None
         return (aten_add_tensor,)
```

</details>

## `convert_constant_dim_order_pass`

关键变化：unsupported constant stride/dim order is made contiguous

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, c_constant: "f32[4, 5][1, 4]cpu", x: "f32[4, 5][5, 1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:472 in forward, code: return x + self.constant
        add: "f32[4, 5][5, 1]cpu" = torch.ops.aten.add.Tensor(x, c_constant);  x = c_constant = None
        return (add,)
```

### After · `after.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, c_constant: "f32[4, 5][5, 1]cpu", x: "f32[4, 5][5, 1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:472 in forward, code: return x + self.constant
        add: "f32[4, 5][5, 1]cpu" = torch.ops.aten.add.Tensor(x, c_constant);  x = c_constant = None
        return (add,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 class GraphModule(torch.nn.Module):
-    def forward(self, c_constant: "f32[4, 5][1, 4]cpu", x: "f32[4, 5][5, 1]cpu"):
+    def forward(self, c_constant: "f32[4, 5][5, 1]cpu", x: "f32[4, 5][5, 1]cpu"):
         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:472 in forward, code: return x + self.constant
         add: "f32[4, 5][5, 1]cpu" = torch.ops.aten.add.Tensor(x, c_constant);  x = c_constant = None
         return (add,)
```

</details>

## `dead_code_elimination_pass (ATen→Edge)`

关键变化：unused neg node is eliminated recursively

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        relu_default = torch.ops.aten.relu.default(x)
        neg_default = torch.ops.aten.neg.default(x);  x = neg_default = None
        return relu_default
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        relu_default = torch.ops.aten.relu.default(x);  x = None
        return relu_default
```

### Unified diff

```diff
--- before
+++ after
@@ -1,6 +1,5 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x):
         # No stacktrace found for following nodes
-        relu_default = torch.ops.aten.relu.default(x)
-        neg_default = torch.ops.aten.neg.default(x);  x = neg_default = None
+        relu_default = torch.ops.aten.relu.default(x);  x = None
         return relu_default
```

</details>

## `DebugHandleGeneratorPass`

关键变化：non-input/output nodes receive debug_handle metadata

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:514 in forward, code: return torch.relu(x + x)
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x, x);  x = None
        relu: "f32[2][1]cpu" = torch.ops.aten.relu.default(add);  add = None
        return (relu,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %add: {"debug_handle": 1}
# %relu: {"debug_handle": 2}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu"):
        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:514 in forward, code: return torch.relu(x + x)
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x, x);  x = None

        # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:514 in forward, code: return torch.relu(x + x)
        relu: "f32[2][1]cpu" = torch.ops.aten.relu.default(add);  add = None
        return (relu,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,6 +1,12 @@
+# Codex custom node.meta summary (not part of FX print_readable output)
+# %add: {"debug_handle": 1}
+# %relu: {"debug_handle": 2}
+
 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[2][1]cpu"):
-        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:514 in forward, code: return torch.relu(x + x)
+        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:514 in forward, code: return torch.relu(x + x)
         add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x, x);  x = None
+
+        # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:514 in forward, code: return torch.relu(x + x)
         relu: "f32[2][1]cpu" = torch.ops.aten.relu.default(add);  add = None
         return (relu,)
```

</details>

## `quant_fusion_and_const_prop_pass`

关键变化：dequantize + add + quantize is fused to quantized_decomposed.add, then constants are propagated

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %quantized_decomposed_quantize_per_tensor_default: {"debug_handle": 1}
# %quantized_decomposed_dequantize_per_tensor_default: {"debug_handle": 2}
# %quantized_decomposed_quantize_per_tensor_default_1: {"debug_handle": 3}
# %quantized_decomposed_dequantize_per_tensor_default_1: {"debug_handle": 4}
# %aten_add_tensor: {"debug_handle": 5}
# %quantized_decomposed_quantize_per_tensor_default_2: {"debug_handle": 6}
# %quantized_decomposed_dequantize_per_tensor_default_2: {"debug_handle": 7}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[1, 5][5, 1]cpu", y: "f32[1, 5][5, 1]cpu"):
        # debug_handle: 1 # File: <eval_with_key>.179:5 in forward, code: quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.00974651612341404, 51, 0, 255, torch.uint8);  x = None
        quantized_decomposed_quantize_per_tensor_default: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(x, 0.00974651612341404, 51, 0, 255, torch.uint8);  x = None

        # debug_handle: 2 # File: <eval_with_key>.179:6 in forward, code: dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, 0.00974651612341404, 51, 0, 255, torch.uint8);  quantize_per_tensor_default = None
        quantized_decomposed_dequantize_per_tensor_default: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(quantized_decomposed_quantize_per_tensor_default, 0.00974651612341404, 51, 0, 255, torch.uint8);  quantized_decomposed_quantize_per_tensor_default = None

        # debug_handle: 3 # File: <eval_with_key>.179:7 in forward, code: quantize_per_tensor_default_1 = torch.ops.quantized_decomposed.quantize_per_tensor.default(y, 0.010185849852859974, 193, 0, 255, torch.uint8);  y = None
        quantized_decomposed_quantize_per_tensor_default_1: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(y, 0.010185849852859974, 193, 0, 255, torch.uint8);  y = None

        # debug_handle: 4 # File: <eval_with_key>.179:8 in forward, code: dequantize_per_tensor_default_1 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_1, 0.010185849852859974, 193, 0, 255, torch.uint8);  quantize_per_tensor_default_1 = None
        quantized_decomposed_dequantize_per_tensor_default_1: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(quantized_decomposed_quantize_per_tensor_default_1, 0.010185849852859974, 193, 0, 255, torch.uint8);  quantized_decomposed_quantize_per_tensor_default_1 = None

        # debug_handle: 5 # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/proxy.py:274 in create_node, code: user_stack_summary = CapturedTraceback.extract().summary()
        aten_add_tensor: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(quantized_decomposed_dequantize_per_tensor_default, quantized_decomposed_dequantize_per_tensor_default_1);  quantized_decomposed_dequantize_per_tensor_default = quantized_decomposed_dequantize_per_tensor_default_1 = None

        # debug_handle: 6 # File: <eval_with_key>.179:10 in forward, code: quantize_per_tensor_default_2 = torch.ops.quantized_decomposed.quantize_per_tensor.default(add, 0.011950294487178326, 81, 0, 255, torch.uint8);  add = None
        quantized_decomposed_quantize_per_tensor_default_2: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(aten_add_tensor, 0.011950294487178326, 81, 0, 255, torch.uint8);  aten_add_tensor = None

        # debug_handle: 7 # File: <eval_with_key>.179:11 in forward, code: dequantize_per_tensor_default_2 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_2, 0.011950294487178326, 81, 0, 255, torch.uint8);  quantize_per_tensor_default_2 = None
        quantized_decomposed_dequantize_per_tensor_default_2: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(quantized_decomposed_quantize_per_tensor_default_2, 0.011950294487178326, 81, 0, 255, torch.uint8);  quantized_decomposed_quantize_per_tensor_default_2 = None
        return (quantized_decomposed_dequantize_per_tensor_default_2,)
```

### After · `after.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %quantized_decomposed_quantize_per_tensor_default: {"debug_handle": 1}
# %quantized_decomposed_quantize_per_tensor_default_1: {"debug_handle": 3}
# %quantized_decomposed_dequantize_per_tensor_default_2: {"debug_handle": 7}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[1, 5][5, 1]cpu", y: "f32[1, 5][5, 1]cpu"):
        # debug_handle: 1 # File: <eval_with_key>.179:5 in forward, code: quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.00974651612341404, 51, 0, 255, torch.uint8);  x = None
        quantized_decomposed_quantize_per_tensor_default: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(x, 0.00974651612341404, 51, 0, 255, torch.uint8);  x = None

        # debug_handle: 3 # File: <eval_with_key>.179:7 in forward, code: quantize_per_tensor_default_1 = torch.ops.quantized_decomposed.quantize_per_tensor.default(y, 0.010185849852859974, 193, 0, 255, torch.uint8);  y = None
        quantized_decomposed_quantize_per_tensor_default_1: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(y, 0.010185849852859974, 193, 0, 255, torch.uint8);  y = None

        # No stacktrace found for following nodes
        add_default: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_add_default(quantized_decomposed_quantize_per_tensor_default, 0.00974651612341404, 51, 0, 255, quantized_decomposed_quantize_per_tensor_default_1, 0.010185849852859974, 193, 0, 255, 0.011950294487178326, 81, 0, 255);  quantized_decomposed_quantize_per_tensor_default = quantized_decomposed_quantize_per_tensor_default_1 = None

        # debug_handle: 7 # File: <eval_with_key>.179:11 in forward, code: dequantize_per_tensor_default_2 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_2, 0.011950294487178326, 81, 0, 255, torch.uint8);  quantize_per_tensor_default_2 = None
        quantized_decomposed_dequantize_per_tensor_default_2: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(add_default, 0.011950294487178326, 81, 0, 255, torch.uint8);  add_default = None
        return [quantized_decomposed_dequantize_per_tensor_default_2]
```

### Unified diff

```diff
--- before
+++ after
@@ -1,10 +1,6 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
 # %quantized_decomposed_quantize_per_tensor_default: {"debug_handle": 1}
-# %quantized_decomposed_dequantize_per_tensor_default: {"debug_handle": 2}
 # %quantized_decomposed_quantize_per_tensor_default_1: {"debug_handle": 3}
-# %quantized_decomposed_dequantize_per_tensor_default_1: {"debug_handle": 4}
-# %aten_add_tensor: {"debug_handle": 5}
-# %quantized_decomposed_quantize_per_tensor_default_2: {"debug_handle": 6}
 # %quantized_decomposed_dequantize_per_tensor_default_2: {"debug_handle": 7}

 class GraphModule(torch.nn.Module):
@@ -12,21 +8,12 @@
         # debug_handle: 1 # File: <eval_with_key>.179:5 in forward, code: quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.00974651612341404, 51, 0, 255, torch.uint8);  x = None
         quantized_decomposed_quantize_per_tensor_default: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(x, 0.00974651612341404, 51, 0, 255, torch.uint8);  x = None

-        # debug_handle: 2 # File: <eval_with_key>.179:6 in forward, code: dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, 0.00974651612341404, 51, 0, 255, torch.uint8);  quantize_per_tensor_default = None
-        quantized_decomposed_dequantize_per_tensor_default: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(quantized_decomposed_quantize_per_tensor_default, 0.00974651612341404, 51, 0, 255, torch.uint8);  quantized_decomposed_quantize_per_tensor_default = None
-
         # debug_handle: 3 # File: <eval_with_key>.179:7 in forward, code: quantize_per_tensor_default_1 = torch.ops.quantized_decomposed.quantize_per_tensor.default(y, 0.010185849852859974, 193, 0, 255, torch.uint8);  y = None
         quantized_decomposed_quantize_per_tensor_default_1: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(y, 0.010185849852859974, 193, 0, 255, torch.uint8);  y = None

-        # debug_handle: 4 # File: <eval_with_key>.179:8 in forward, code: dequantize_per_tensor_default_1 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_1, 0.010185849852859974, 193, 0, 255, torch.uint8);  quantize_per_tensor_default_1 = None
-        quantized_decomposed_dequantize_per_tensor_default_1: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(quantized_decomposed_quantize_per_tensor_default_1, 0.010185849852859974, 193, 0, 255, torch.uint8);  quantized_decomposed_quantize_per_tensor_default_1 = None
-
-        # debug_handle: 5 # File: /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/proxy.py:274 in create_node, code: user_stack_summary = CapturedTraceback.extract().summary()
-        aten_add_tensor: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(quantized_decomposed_dequantize_per_tensor_default, quantized_decomposed_dequantize_per_tensor_default_1);  quantized_decomposed_dequantize_per_tensor_default = quantized_decomposed_dequantize_per_tensor_default_1 = None
-
-        # debug_handle: 6 # File: <eval_with_key>.179:10 in forward, code: quantize_per_tensor_default_2 = torch.ops.quantized_decomposed.quantize_per_tensor.default(add, 0.011950294487178326, 81, 0, 255, torch.uint8);  add = None
-        quantized_decomposed_quantize_per_tensor_default_2: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default(aten_add_tensor, 0.011950294487178326, 81, 0, 255, torch.uint8);  aten_add_tensor = None
+        # No stacktrace found for following nodes
+        add_default: "u8[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_add_default(quantized_decomposed_quantize_per_tensor_default, 0.00974651612341404, 51, 0, 255, quantized_decomposed_quantize_per_tensor_default_1, 0.010185849852859974, 193, 0, 255, 0.011950294487178326, 81, 0, 255);  quantized_decomposed_quantize_per_tensor_default = quantized_decomposed_quantize_per_tensor_default_1 = None

         # debug_handle: 7 # File: <eval_with_key>.179:11 in forward, code: dequantize_per_tensor_default_2 = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default_2, 0.011950294487178326, 81, 0, 255, torch.uint8);  quantize_per_tensor_default_2 = None
-        quantized_decomposed_dequantize_per_tensor_default_2: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(quantized_decomposed_quantize_per_tensor_default_2, 0.011950294487178326, 81, 0, 255, torch.uint8);  quantized_decomposed_quantize_per_tensor_default_2 = None
-        return (quantized_decomposed_dequantize_per_tensor_default_2,)
+        quantized_decomposed_dequantize_per_tensor_default_2: "f32[1, 5][5, 1]cpu" = executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default(add_default, 0.011950294487178326, 81, 0, 255, torch.uint8);  add_default = None
+        return [quantized_decomposed_dequantize_per_tensor_default_2]
```

</details>

## `reinplace_pass`

关键变化：safe functional index_put becomes index_put_

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %aten_index_put_default: {"debug_handle": 1}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[5][1]cpu", indices: "i64[1][1]cpu", values: "f32[1][1]cpu"):
        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:565 in forward, code: self.state.index_put_((indices,), values)
        aten_index_put_default: "f32[5][1]cpu" = executorch_exir_dialects_edge__ops_aten_index_put_default(b_state, [indices], values);  b_state = indices = values = None
        return (aten_index_put_default, aten_index_put_default)
```

### After · `after.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[5][1]cpu", indices: "i64[1][1]cpu", values: "f32[1][1]cpu"):
        # No stacktrace found for following nodes
        aten_index_put__default: "f32[5][1]cpu" = executorch_exir_dialects_edge__ops_aten_index_put__default(b_state, [indices], values);  b_state = indices = values = None
        return (aten_index_put__default, aten_index_put__default)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,8 +1,5 @@
-# Codex custom node.meta summary (not part of FX print_readable output)
-# %aten_index_put_default: {"debug_handle": 1}
-
 class GraphModule(torch.nn.Module):
     def forward(self, b_state: "f32[5][1]cpu", indices: "i64[1][1]cpu", values: "f32[1][1]cpu"):
-        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:565 in forward, code: self.state.index_put_((indices,), values)
-        aten_index_put_default: "f32[5][1]cpu" = executorch_exir_dialects_edge__ops_aten_index_put_default(b_state, [indices], values);  b_state = indices = values = None
-        return (aten_index_put_default, aten_index_put_default)
+        # No stacktrace found for following nodes
+        aten_index_put__default: "f32[5][1]cpu" = executorch_exir_dialects_edge__ops_aten_index_put__default(b_state, [indices], values);  b_state = indices = values = None
+        return (aten_index_put__default, aten_index_put__default)
```

</details>

## `weights_to_outputs_pass`

关键变化：gradient-bearing weight gets weight_has_gradient and is appended as TOKEN output

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %aten_mul_tensor: {"debug_handle": 1}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2}
# %aten_sub_tensor: {"debug_handle": 2}
# %aten_pow_tensor_scalar: {"debug_handle": 3}
# %aten_sum_dim_int_list: {"debug_handle": 4}
# %aten_full_like_default: {"debug_handle": 5}
# %aten_expand_copy_default: {"debug_handle": 6}
# %aten_pow_tensor_scalar_1: {"debug_handle": 7}
# %aten_mul_scalar: {"debug_handle": 8}
# %aten_mul_tensor_1: {"debug_handle": 9}
# %aten_mul_tensor_2: {"debug_handle": 10}

class GraphModule(torch.nn.Module):
    def forward(self, p_weight: "f32[2][1]cpu", _lifted_tensor_constant0: "i64[][]cpu", x: "f32[2][1]cpu"):
        # debug_handle: 1 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:6 in forward, code: mul = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight);  p_weight = None

        # debug_handle: 2 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:7 in forward, code: sub = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant0, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant0 = None
        aten_sub_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_sub_Tensor(aten_mul_tensor, dim_order_ops__to_dim_order_copy_default);  aten_mul_tensor = dim_order_ops__to_dim_order_copy_default = None

        # debug_handle: 3 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:8 in forward, code: pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        aten_pow_tensor_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 2)

        # debug_handle: 4 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:9 in forward, code: sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, []);  pow_1 = None
        aten_sum_dim_int_list: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_sum_dim_IntList(aten_pow_tensor_scalar, []);  aten_pow_tensor_scalar = None

        # debug_handle: 5 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:10 in forward, code: full_like = torch.ops.aten.full_like.default(sum_1, 1, pin_memory = False, memory_format = torch.preserve_format)
        aten_full_like_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_full_like_default(aten_sum_dim_int_list, 1, pin_memory = False, memory_format = torch.preserve_format)

        # debug_handle: 6 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:11 in forward, code: expand = torch.ops.aten.expand.default(full_like, [2]);  full_like = None
        aten_expand_copy_default: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_expand_copy_default(aten_full_like_default, [2]);  aten_full_like_default = None

        # debug_handle: 7 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:12 in forward, code: pow_2 = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
        aten_pow_tensor_scalar_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 1.0);  aten_sub_tensor = None

        # debug_handle: 8 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:13 in forward, code: mul_1 = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        aten_mul_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Scalar(aten_pow_tensor_scalar_1, 2.0);  aten_pow_tensor_scalar_1 = None

        # debug_handle: 9 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:14 in forward, code: mul_2 = torch.ops.aten.mul.Tensor(expand, mul_1);  expand = mul_1 = None
        aten_mul_tensor_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_expand_copy_default, aten_mul_scalar);  aten_expand_copy_default = aten_mul_scalar = None

        # debug_handle: 10 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:15 in forward, code: mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_2);  mul_2 = primals_2 = None
        aten_mul_tensor_2: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_mul_tensor_1, x);  aten_mul_tensor_1 = x = None
        return (aten_sum_dim_int_list, aten_mul_tensor_2)
```

### After · `after.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %p_weight: {"weight_has_gradient": true}
# %aten_mul_tensor: {"debug_handle": 1}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2}
# %aten_sub_tensor: {"debug_handle": 2}
# %aten_pow_tensor_scalar: {"debug_handle": 3}
# %aten_sum_dim_int_list: {"debug_handle": 4}
# %aten_full_like_default: {"debug_handle": 5}
# %aten_expand_copy_default: {"debug_handle": 6}
# %aten_pow_tensor_scalar_1: {"debug_handle": 7}
# %aten_mul_scalar: {"debug_handle": 8}
# %aten_mul_tensor_1: {"debug_handle": 9}
# %aten_mul_tensor_2: {"debug_handle": 10}

class GraphModule(torch.nn.Module):
    def forward(self, p_weight: "f32[2][1]cpu", _lifted_tensor_constant0: "i64[][]cpu", x: "f32[2][1]cpu"):
        # debug_handle: 1 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:6 in forward, code: mul = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight)

        # debug_handle: 2 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:7 in forward, code: sub = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant0, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant0 = None
        aten_sub_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_sub_Tensor(aten_mul_tensor, dim_order_ops__to_dim_order_copy_default);  aten_mul_tensor = dim_order_ops__to_dim_order_copy_default = None

        # debug_handle: 3 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:8 in forward, code: pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        aten_pow_tensor_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 2)

        # debug_handle: 4 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:9 in forward, code: sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, []);  pow_1 = None
        aten_sum_dim_int_list: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_sum_dim_IntList(aten_pow_tensor_scalar, []);  aten_pow_tensor_scalar = None

        # debug_handle: 5 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:10 in forward, code: full_like = torch.ops.aten.full_like.default(sum_1, 1, pin_memory = False, memory_format = torch.preserve_format)
        aten_full_like_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_full_like_default(aten_sum_dim_int_list, 1, pin_memory = False, memory_format = torch.preserve_format)

        # debug_handle: 6 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:11 in forward, code: expand = torch.ops.aten.expand.default(full_like, [2]);  full_like = None
        aten_expand_copy_default: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_expand_copy_default(aten_full_like_default, [2]);  aten_full_like_default = None

        # debug_handle: 7 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:12 in forward, code: pow_2 = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
        aten_pow_tensor_scalar_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 1.0);  aten_sub_tensor = None

        # debug_handle: 8 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:13 in forward, code: mul_1 = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        aten_mul_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Scalar(aten_pow_tensor_scalar_1, 2.0);  aten_pow_tensor_scalar_1 = None

        # debug_handle: 9 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:14 in forward, code: mul_2 = torch.ops.aten.mul.Tensor(expand, mul_1);  expand = mul_1 = None
        aten_mul_tensor_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_expand_copy_default, aten_mul_scalar);  aten_expand_copy_default = aten_mul_scalar = None

        # debug_handle: 10 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:15 in forward, code: mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_2);  mul_2 = primals_2 = None
        aten_mul_tensor_2: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_mul_tensor_1, x);  aten_mul_tensor_1 = x = None
        return (aten_sum_dim_int_list, aten_mul_tensor_2, p_weight)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,4 +1,5 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
+# %p_weight: {"weight_has_gradient": true}
 # %aten_mul_tensor: {"debug_handle": 1}
 # %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2}
 # %aten_sub_tensor: {"debug_handle": 2}
@@ -14,7 +15,7 @@
 class GraphModule(torch.nn.Module):
     def forward(self, p_weight: "f32[2][1]cpu", _lifted_tensor_constant0: "i64[][]cpu", x: "f32[2][1]cpu"):
         # debug_handle: 1 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:6 in forward, code: mul = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
-        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight);  p_weight = None
+        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight)

         # debug_handle: 2 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:7 in forward, code: sub = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
         dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant0, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant0 = None
@@ -43,4 +44,4 @@

         # debug_handle: 10 # File: <eval_with_key>.413 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:15 in forward, code: mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_2);  mul_2 = primals_2 = None
         aten_mul_tensor_2: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_mul_tensor_1, x);  aten_mul_tensor_1 = x = None
-        return (aten_sum_dim_int_list, aten_mul_tensor_2)
+        return (aten_sum_dim_int_list, aten_mul_tensor_2, p_weight)
```

</details>

## `unsafe_remove_auto_functionalized_pass`

关键变化：auto_functionalized_v2/getitem wrapper is decomposed back to the original mutating op

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:629 in forward, code: custom(x)
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.codex_pass_debug.mutate.default, _x_base_index = 0, _all_bases = [x]);  x = None
        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:630 in forward, code: return x + 1
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, 1)
        return (getitem_1, add)
```

### After · `after.graph`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:629 in forward, code: custom(x)
        mutate_default = torch.ops.codex_pass_debug.mutate.default(x);  mutate_default = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:630 in forward, code: return x + 1
        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x, 1)
        return (x, add)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,9 +1,8 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[2][1]cpu"):
         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:629 in forward, code: custom(x)
-        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.codex_pass_debug.mutate.default, _x_base_index = 0, _all_bases = [x]);  x = None
-        getitem_1: "f32[2][1]cpu" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
+        mutate_default = torch.ops.codex_pass_debug.mutate.default(x);  mutate_default = None

         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:630 in forward, code: return x + 1
-        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(getitem_1, 1)
-        return (getitem_1, add)
+        add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x, 1)
+        return (x, add)
```

</details>

## `insert_write_back_for_buffers_pass`

关键变化：copy_ is inserted in the epilogue to write mutated output back to buffer

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %aten_add_tensor: {"debug_handle": 1}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2}
# %aten_add_tensor_1: {"debug_handle": 2}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[1][1]cpu", _lifted_tensor_constant1: "i64[][]cpu", x: "f32[1][1]cpu"):
        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:650 in forward, code: y = x + self.state
        aten_add_tensor: "f32[1][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, b_state);  x = None

        # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:651 in forward, code: self.state.add_(1)
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant1, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant1 = None
        aten_add_tensor_1: "f32[1][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(b_state, dim_order_ops__to_dim_order_copy_default);  b_state = dim_order_ops__to_dim_order_copy_default = None
        return (aten_add_tensor_1, aten_add_tensor)
```

### After · `after.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %aten_add_tensor: {"debug_handle": 1}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2}
# %aten_add_tensor_1: {"debug_handle": 2}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[1][1]cpu", _lifted_tensor_constant1: "i64[][]cpu", x: "f32[1][1]cpu"):
        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:650 in forward, code: y = x + self.state
        aten_add_tensor: "f32[1][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, b_state);  x = None

        # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:651 in forward, code: self.state.add_(1)
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant1, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant1 = None
        aten_add_tensor_1: "f32[1][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(b_state, dim_order_ops__to_dim_order_copy_default);  dim_order_ops__to_dim_order_copy_default = None

        # No stacktrace found for following nodes
        copy__default = torch.ops.aten.copy_.default(b_state, aten_add_tensor_1);  b_state = aten_add_tensor_1 = None
        return (copy__default, aten_add_tensor)
```

### Unified diff

```diff
--- before
+++ after
@@ -10,5 +10,8 @@

         # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:651 in forward, code: self.state.add_(1)
         dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant1, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant1 = None
-        aten_add_tensor_1: "f32[1][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(b_state, dim_order_ops__to_dim_order_copy_default);  b_state = dim_order_ops__to_dim_order_copy_default = None
-        return (aten_add_tensor_1, aten_add_tensor)
+        aten_add_tensor_1: "f32[1][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(b_state, dim_order_ops__to_dim_order_copy_default);  dim_order_ops__to_dim_order_copy_default = None
+
+        # No stacktrace found for following nodes
+        copy__default = torch.ops.aten.copy_.default(b_state, aten_add_tensor_1);  b_state = aten_add_tensor_1 = None
+        return (copy__default, aten_add_tensor)
```

</details>

## `SpecPropPass`

关键变化：TensorSpec is attached to tensor-producing nodes

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %aten_add_tensor: {"debug_handle": 1}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu", y: "f32[2][1]cpu"):
        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:667 in forward, code: return x + y
        aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, y);  x = y = None
        return (aten_add_tensor,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %y: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu", y: "f32[2][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:667 in forward, code: return x + y
        aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, y);  x = y = None
        return (aten_add_tensor,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,8 +1,11 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %aten_add_tensor: {"debug_handle": 1}
+# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %y: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[2][1]cpu", y: "f32[2][1]cpu"):
-        # debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:667 in forward, code: return x + y
+        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:667 in forward, code: return x + y
         aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, y);  x = y = None
         return (aten_add_tensor,)
```

</details>

## `PropagateDevicePass`

关键变化：two H2D copies and one D2H copy are inserted; delegate/getitem specs become CUDA

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %a: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %b: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %lowered_module_0: {"spec": null}
# %executorch_call_delegate: {"debug_handle": 2, "spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}
# %getitem: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, a: "f32[2][1]cpu", b: "f32[2][1]cpu"):
        # spec: None # No stacktrace found for following nodes
        lowered_module_0 = self.lowered_module_0

        # spec: (TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0),), debug_handle: 2 # No stacktrace found for following nodes
        executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, a, b);  lowered_module_0 = a = b = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        getitem: "f32[2][1]cpu" = executorch_call_delegate[0];  executorch_call_delegate = None
        return (getitem,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %a: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %b: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %lowered_module_0: {"spec": null}
# %_h2d_copy_default: {"spec": {"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %_h2d_copy_default_1: {"spec": {"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %executorch_call_delegate: {"debug_handle": 2, "spec": [{"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}
# %getitem: {"spec": {"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %_d2h_copy_default: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, a: "f32[2][1]cpu", b: "f32[2][1]cpu"):
        # spec: None # No stacktrace found for following nodes
        lowered_module_0 = self.lowered_module_0

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CUDA:0) # No stacktrace found for following nodes
        _h2d_copy_default: "f32[2][1]cpu" = torch.ops.et_copy._h2d_copy.default(a);  a = None
        _h2d_copy_default_1: "f32[2][1]cpu" = torch.ops.et_copy._h2d_copy.default(b);  b = None

        # spec: (TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CUDA:0),), debug_handle: 2 # No stacktrace found for following nodes
        executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, _h2d_copy_default, _h2d_copy_default_1);  lowered_module_0 = _h2d_copy_default = _h2d_copy_default_1 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CUDA:0) # No stacktrace found for following nodes
        getitem: "f32[2][1]cpu" = executorch_call_delegate[0];  executorch_call_delegate = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        _d2h_copy_default: "f32[2][1]cpu" = torch.ops.et_copy._d2h_copy.default(getitem);  getitem = None
        return (_d2h_copy_default,)
```

### Unified diff

```diff
--- before
+++ after
@@ -2,18 +2,28 @@
 # %a: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
 # %b: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
 # %lowered_module_0: {"spec": null}
-# %executorch_call_delegate: {"debug_handle": 2, "spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}
-# %getitem: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
-# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}
+# %_h2d_copy_default: {"spec": {"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %_h2d_copy_default_1: {"spec": {"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %executorch_call_delegate: {"debug_handle": 2, "spec": [{"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}
+# %getitem: {"spec": {"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %_d2h_copy_default: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %output: {"spec": [{"const": false, "device": "CUDA", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

 class GraphModule(torch.nn.Module):
     def forward(self, a: "f32[2][1]cpu", b: "f32[2][1]cpu"):
         # spec: None # No stacktrace found for following nodes
         lowered_module_0 = self.lowered_module_0

-        # spec: (TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0),), debug_handle: 2 # No stacktrace found for following nodes
-        executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, a, b);  lowered_module_0 = a = b = None
+        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CUDA:0) # No stacktrace found for following nodes
+        _h2d_copy_default: "f32[2][1]cpu" = torch.ops.et_copy._h2d_copy.default(a);  a = None
+        _h2d_copy_default_1: "f32[2][1]cpu" = torch.ops.et_copy._h2d_copy.default(b);  b = None
+
+        # spec: (TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CUDA:0),), debug_handle: 2 # No stacktrace found for following nodes
+        executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, _h2d_copy_default, _h2d_copy_default_1);  lowered_module_0 = _h2d_copy_default = _h2d_copy_default_1 = None
+
+        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CUDA:0) # No stacktrace found for following nodes
+        getitem: "f32[2][1]cpu" = executorch_call_delegate[0];  executorch_call_delegate = None

         # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
-        getitem: "f32[2][1]cpu" = executorch_call_delegate[0];  executorch_call_delegate = None
-        return (getitem,)
+        _d2h_copy_default: "f32[2][1]cpu" = torch.ops.et_copy._d2h_copy.default(getitem);  getitem = None
+        return (_d2h_copy_default,)
```

</details>

## `EdgeToBackendOpsPass`

关键变化：Python symbolic sub becomes executorch_prim.sub.Scalar

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %sym_size: {"debug_handle": 1}
# %sub: {"debug_handle": 2}
# %scalar_tensor: {"debug_handle": 3}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 3}
# %aten_add_tensor: {"debug_handle": 3}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s6, 2][2, 1]cpu"):
        # debug_handle: 1 # No stacktrace found for following nodes
        sym_size: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

        # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:699 in forward, code: return torch.add(x, x.shape[0] - 1)
        sub: "Sym(s6 - 1)" = sym_size - 1;  sym_size = None

        # debug_handle: 3 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:699 in forward, code: return torch.add(x, x.shape[0] - 1)
        scalar_tensor: "i32[][]cpu" = torch.ops.aten.scalar_tensor.default(sub, dtype = torch.int32);  sub = None
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(scalar_tensor, dtype = torch.float32, dim_order = []);  scalar_tensor = None
        aten_add_tensor: "f32[s6, 2][2, 1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, dim_order_ops__to_dim_order_copy_default);  x = dim_order_ops__to_dim_order_copy_default = None
        return (aten_add_tensor,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %sym_size: {"debug_handle": 1}
# %sub: {"debug_handle": 2}
# %scalar_tensor: {"debug_handle": 3}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 3}
# %aten_add_tensor: {"debug_handle": 3}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s6, 2][2, 1]cpu"):
        # debug_handle: 1 # No stacktrace found for following nodes
        sym_size: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

        # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:699 in forward, code: return torch.add(x, x.shape[0] - 1)
        sub: "Sym(s6 - 1)" = executorch_exir_dialects_backend__ops_executorch_prim_sub_Scalar(sym_size, 1);  sym_size = None

        # debug_handle: 3 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:699 in forward, code: return torch.add(x, x.shape[0] - 1)
        scalar_tensor: "i32[][]cpu" = torch.ops.aten.scalar_tensor.default(sub, dtype = torch.int32);  sub = None
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(scalar_tensor, dtype = torch.float32, dim_order = []);  scalar_tensor = None
        aten_add_tensor: "f32[s6, 2][2, 1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, dim_order_ops__to_dim_order_copy_default);  x = dim_order_ops__to_dim_order_copy_default = None
        return (aten_add_tensor,)
```

### Unified diff

```diff
--- before
+++ after
@@ -11,7 +11,7 @@
         sym_size: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

         # debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:699 in forward, code: return torch.add(x, x.shape[0] - 1)
-        sub: "Sym(s6 - 1)" = sym_size - 1;  sym_size = None
+        sub: "Sym(s6 - 1)" = executorch_exir_dialects_backend__ops_executorch_prim_sub_Scalar(sym_size, 1);  sym_size = None

         # debug_handle: 3 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:699 in forward, code: return torch.add(x, x.shape[0] - 1)
         scalar_tensor: "i32[][]cpu" = torch.ops.aten.scalar_tensor.default(sub, dtype = torch.int32);  sub = None
```

</details>

## `RemoveGraphAssertsPass`

关键变化：runtime-unsupported assert node is erased and DCE runs

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        _assert_async_msg = torch.ops.aten._assert_async.msg();  _assert_async_msg = None
        return x
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        return x
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,3 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x):
-        # No stacktrace found for following nodes
-        _assert_async_msg = torch.ops.aten._assert_async.msg();  _assert_async_msg = None
         return x
```

</details>

## `NormalizeViewCopyBasePass`

关键变化：every view_copy points directly to the first non-view base

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[6][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:727 in forward, code: x = torch.ops.aten.view_copy.default(x, [6, 1])
        view_copy: "f32[6, 1][1, 1]cpu" = torch.ops.aten.view_copy.default(x, [6, 1]);  x = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:728 in forward, code: x = torch.ops.aten.view_copy.default(x, [2, 3])
        view_copy_1: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(view_copy, [2, 3]);  view_copy = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:729 in forward, code: return torch.ops.aten.view_copy.default(x, [3, 2])
        view_copy_2: "f32[3, 2][2, 1]cpu" = torch.ops.aten.view_copy.default(view_copy_1, [3, 2]);  view_copy_1 = None
        return (view_copy_2,)
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[6][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:727 in forward, code: x = torch.ops.aten.view_copy.default(x, [6, 1])
        view_copy: "f32[6, 1][1, 1]cpu" = torch.ops.aten.view_copy.default(x, [6, 1]);  view_copy = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:728 in forward, code: x = torch.ops.aten.view_copy.default(x, [2, 3])
        view_copy_1: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(x, [2, 3]);  view_copy_1 = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:729 in forward, code: return torch.ops.aten.view_copy.default(x, [3, 2])
        view_copy_2: "f32[3, 2][2, 1]cpu" = torch.ops.aten.view_copy.default(x, [3, 2]);  x = None
        return (view_copy_2,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,11 +1,11 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[6][1]cpu"):
         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:727 in forward, code: x = torch.ops.aten.view_copy.default(x, [6, 1])
-        view_copy: "f32[6, 1][1, 1]cpu" = torch.ops.aten.view_copy.default(x, [6, 1]);  x = None
+        view_copy: "f32[6, 1][1, 1]cpu" = torch.ops.aten.view_copy.default(x, [6, 1]);  view_copy = None

         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:728 in forward, code: x = torch.ops.aten.view_copy.default(x, [2, 3])
-        view_copy_1: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(view_copy, [2, 3]);  view_copy = None
+        view_copy_1: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(x, [2, 3]);  view_copy_1 = None

         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:729 in forward, code: return torch.ops.aten.view_copy.default(x, [3, 2])
-        view_copy_2: "f32[3, 2][2, 1]cpu" = torch.ops.aten.view_copy.default(view_copy_1, [3, 2]);  view_copy_1 = None
+        view_copy_2: "f32[3, 2][2, 1]cpu" = torch.ops.aten.view_copy.default(x, [3, 2]);  x = None
         return (view_copy_2,)
```

</details>

## `dead_code_elimination_pass (pre-memory)`

关键变化：second pipeline DCE removes newly-dead nodes before view lowering

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        relu_default = torch.ops.aten.relu.default(x)
        neg_default = torch.ops.aten.neg.default(x);  x = neg_default = None
        return relu_default
```

### After · `after`

```python
class GraphModule(torch.nn.Module):
    def forward(self, x):
        # No stacktrace found for following nodes
        relu_default = torch.ops.aten.relu.default(x);  x = None
        return relu_default
```

### Unified diff

```diff
--- before
+++ after
@@ -1,6 +1,5 @@
 class GraphModule(torch.nn.Module):
     def forward(self, x):
         # No stacktrace found for following nodes
-        relu_default = torch.ops.aten.relu.default(x)
-        neg_default = torch.ops.aten.neg.default(x);  x = neg_default = None
+        relu_default = torch.ops.aten.relu.default(x);  x = None
         return relu_default
```

</details>

## `ReplaceViewCopyWithViewPass`

关键变化：non-output view_copy becomes memory.view with a storage-sharing _ViewSpec

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["6"], "stride": ["1"]}}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[6][1]cpu"):
        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:752 in forward, code: viewed = torch.ops.aten.view_copy.default(x, [2, 3])
        view_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(x, [2, 3]);  x = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:753 in forward, code: return viewed + 1
        add: "f32[2, 3][3, 1]cpu" = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
        return (add,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["6"], "stride": ["1"]}}
# %view_copy: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "3"], "stride": ["3", "1"]}}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[6][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]), layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:752 in forward, code: viewed = torch.ops.aten.view_copy.default(x, [2, 3])
        view_copy: "f32[2, 3][3, 1]cpu" = executorch_exir_memory_view(x, [2, 3]);  x = None

        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:753 in forward, code: return viewed + 1
        add: "f32[2, 3][3, 1]cpu" = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
        return (add,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,10 +1,11 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
 # %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["6"], "stride": ["1"]}}
+# %view_copy: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "3"], "stride": ["3", "1"]}}

 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[6][1]cpu"):
-        # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:752 in forward, code: viewed = torch.ops.aten.view_copy.default(x, [2, 3])
-        view_copy: "f32[2, 3][3, 1]cpu" = torch.ops.aten.view_copy.default(x, [2, 3]);  x = None
+        # spec: TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]), layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:752 in forward, code: viewed = torch.ops.aten.view_copy.default(x, [2, 3])
+        view_copy: "f32[2, 3][3, 1]cpu" = executorch_exir_memory_view(x, [2, 3]);  x = None

         # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:753 in forward, code: return viewed + 1
         add: "f32[2, 3][3, 1]cpu" = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
```

</details>

## `ConstraintBasedSymShapeEvalPass`

关键变化：symbolic shape/stride expressions become bounded concrete integers

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["s6", "4", "s37", "s46"], "stride": ["4*s37*s46", "s37*s46", "s46", "1"]}}
# %aten_relu_default: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["s6", "4", "s37", "s46"], "stride": ["4*s37*s46", "s37*s46", "s46", "1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["s6", "4", "s37", "s46"], "stride": ["4*s37*s46", "s37*s46", "s46", "1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s6, 4, s37, s46][4*s37*s46, s37*s46, s46, 1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[s6, 4, s37, s46], layout=torch.strided, is_sparse=False, shape_dynamism=1, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:768 in forward, code: return torch.relu(x)
        aten_relu_default: "f32[s6, 4, s37, s46][4*s37*s46, s37*s46, s46, 1]cpu" = executorch_exir_dialects_edge__ops_aten_relu_default(x);  x = None
        return (aten_relu_default,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "4", "8", "8"], "stride": ["256", "64", "8", "1"]}}
# %aten_relu_default: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "4", "8", "8"], "stride": ["256", "64", "8", "1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "4", "8", "8"], "stride": ["256", "64", "8", "1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[s6, 4, s37, s46][4*s37*s46, s37*s46, s46, 1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2, 4, 8, 8], layout=torch.strided, is_sparse=False, shape_dynamism=1, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:768 in forward, code: return torch.relu(x)
        aten_relu_default: "f32[s6, 4, s37, s46][4*s37*s46, s37*s46, s46, 1]cpu" = executorch_exir_dialects_edge__ops_aten_relu_default(x);  x = None
        return (aten_relu_default,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,10 +1,10 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["s6", "4", "s37", "s46"], "stride": ["4*s37*s46", "s37*s46", "s46", "1"]}}
-# %aten_relu_default: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["s6", "4", "s37", "s46"], "stride": ["4*s37*s46", "s37*s46", "s46", "1"]}}
-# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["s6", "4", "s37", "s46"], "stride": ["4*s37*s46", "s37*s46", "s46", "1"]}]}
+# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "4", "8", "8"], "stride": ["256", "64", "8", "1"]}}
+# %aten_relu_default: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "4", "8", "8"], "stride": ["256", "64", "8", "1"]}}
+# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2", "4", "8", "8"], "stride": ["256", "64", "8", "1"]}]}

 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[s6, 4, s37, s46][4*s37*s46, s37*s46, s46, 1]cpu"):
-        # spec: TensorSpec(dtype=torch.float32, shape=[s6, 4, s37, s46], layout=torch.strided, is_sparse=False, shape_dynamism=1, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:768 in forward, code: return torch.relu(x)
+        # spec: TensorSpec(dtype=torch.float32, shape=[2, 4, 8, 8], layout=torch.strided, is_sparse=False, shape_dynamism=1, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:768 in forward, code: return torch.relu(x)
         aten_relu_default: "f32[s6, 4, s37, s46][4*s37*s46, s37*s46, s46, 1]cpu" = executorch_exir_dialects_edge__ops_aten_relu_default(x);  x = None
         return (aten_relu_default,)
```

</details>

## `ToOutVarPass`

关键变化：functional add becomes add.out and receives a memory.alloc out argument

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %y: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu", y: "f32[2][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:792 in forward, code: return x + y
        aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, y);  x = y = None
        return (aten_add_tensor,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %y: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2][1]cpu", y: "f32[2][1]cpu"):
        # No stacktrace found for following nodes
        alloc: "f32[2][1]cpu" = executorch_exir_memory_alloc(((2,), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:792 in forward, code: return x + y
        aten_add_tensor: "f32[2][1]cpu" = torch.ops.aten.add.out(x, y, out = alloc);  x = y = alloc = None
        return (aten_add_tensor,)
```

### Unified diff

```diff
--- before
+++ after
@@ -6,6 +6,9 @@

 class GraphModule(torch.nn.Module):
     def forward(self, x: "f32[2][1]cpu", y: "f32[2][1]cpu"):
+        # No stacktrace found for following nodes
+        alloc: "f32[2][1]cpu" = executorch_exir_memory_alloc(((2,), torch.float32))
+
         # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:792 in forward, code: return x + y
-        aten_add_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_add_Tensor(x, y);  x = y = None
+        aten_add_tensor: "f32[2][1]cpu" = torch.ops.aten.add.out(x, y, out = alloc);  x = y = alloc = None
         return (aten_add_tensor,)
```

</details>

## `external_constants_pass`

关键变化：const parameter placeholder receives the external constant tag

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %p_weight: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, p_weight: "f32[2][1]cpu", x: "f32[2][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:811 in forward, code: return x * self.weight
        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight);  x = p_weight = None
        return (aten_mul_tensor,)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %p_weight: {"constant_tag": "_default_external_constant", "spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, p_weight: "f32[2][1]cpu", x: "f32[2][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:811 in forward, code: return x * self.weight
        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight);  x = p_weight = None
        return (aten_mul_tensor,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %p_weight: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
+# %p_weight: {"constant_tag": "_default_external_constant", "spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
 # %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
 # %aten_mul_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
 # %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}
```

</details>

## `external_mutable_weights_pass`

关键变化：gradient-bearing const weight receives the external constant tag

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %p_weight: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}, "weight_has_gradient": true}
# %_lifted_tensor_constant0: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.int64", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %aten_sub_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_pow_tensor_scalar: {"debug_handle": 3, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_sum_dim_int_list: {"debug_handle": 4, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %aten_full_like_default: {"debug_handle": 5, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %aten_expand_copy_default: {"debug_handle": 6, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_pow_tensor_scalar_1: {"debug_handle": 7, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_scalar: {"debug_handle": 8, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor_1: {"debug_handle": 9, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor_2: {"debug_handle": 10, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}, {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}, {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, p_weight: "f32[2][1]cpu", _lifted_tensor_constant0: "i64[][]cpu", x: "f32[2][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 1 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:6 in forward, code: mul = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight)

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:7 in forward, code: sub = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant0, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant0 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 2 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:7 in forward, code: sub = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        aten_sub_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_sub_Tensor(aten_mul_tensor, dim_order_ops__to_dim_order_copy_default);  aten_mul_tensor = dim_order_ops__to_dim_order_copy_default = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 3 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:8 in forward, code: pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        aten_pow_tensor_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 2)

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 4 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:9 in forward, code: sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, []);  pow_1 = None
        aten_sum_dim_int_list: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_sum_dim_IntList(aten_pow_tensor_scalar, []);  aten_pow_tensor_scalar = None

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 5 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:10 in forward, code: full_like = torch.ops.aten.full_like.default(sum_1, 1, pin_memory = False, memory_format = torch.preserve_format)
        aten_full_like_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_full_like_default(aten_sum_dim_int_list, 1, pin_memory = False, memory_format = torch.preserve_format)

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 6 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:11 in forward, code: expand = torch.ops.aten.expand.default(full_like, [2]);  full_like = None
        aten_expand_copy_default: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_expand_copy_default(aten_full_like_default, [2]);  aten_full_like_default = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 7 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:12 in forward, code: pow_2 = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
        aten_pow_tensor_scalar_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 1.0);  aten_sub_tensor = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 8 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:13 in forward, code: mul_1 = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        aten_mul_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Scalar(aten_pow_tensor_scalar_1, 2.0);  aten_pow_tensor_scalar_1 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 9 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:14 in forward, code: mul_2 = torch.ops.aten.mul.Tensor(expand, mul_1);  expand = mul_1 = None
        aten_mul_tensor_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_expand_copy_default, aten_mul_scalar);  aten_expand_copy_default = aten_mul_scalar = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 10 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:15 in forward, code: mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_2);  mul_2 = primals_2 = None
        aten_mul_tensor_2: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_mul_tensor_1, x);  aten_mul_tensor_1 = x = None
        return (aten_sum_dim_int_list, aten_mul_tensor_2, p_weight)
```

### After · `after`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %p_weight: {"constant_tag": "_default_external_constant", "spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}, "weight_has_gradient": true}
# %_lifted_tensor_constant0: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.int64", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %aten_sub_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_pow_tensor_scalar: {"debug_handle": 3, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_sum_dim_int_list: {"debug_handle": 4, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %aten_full_like_default: {"debug_handle": 5, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %aten_expand_copy_default: {"debug_handle": 6, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_pow_tensor_scalar_1: {"debug_handle": 7, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_scalar: {"debug_handle": 8, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor_1: {"debug_handle": 9, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %aten_mul_tensor_2: {"debug_handle": 10, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}, {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}, {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, p_weight: "f32[2][1]cpu", _lifted_tensor_constant0: "i64[][]cpu", x: "f32[2][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 1 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:6 in forward, code: mul = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
        aten_mul_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(x, p_weight)

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:7 in forward, code: sub = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default(_lifted_tensor_constant0, dtype = torch.float32, dim_order = []);  _lifted_tensor_constant0 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 2 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:7 in forward, code: sub = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        aten_sub_tensor: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_sub_Tensor(aten_mul_tensor, dim_order_ops__to_dim_order_copy_default);  aten_mul_tensor = dim_order_ops__to_dim_order_copy_default = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 3 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:8 in forward, code: pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        aten_pow_tensor_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 2)

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 4 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:9 in forward, code: sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, []);  pow_1 = None
        aten_sum_dim_int_list: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_sum_dim_IntList(aten_pow_tensor_scalar, []);  aten_pow_tensor_scalar = None

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 5 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:10 in forward, code: full_like = torch.ops.aten.full_like.default(sum_1, 1, pin_memory = False, memory_format = torch.preserve_format)
        aten_full_like_default: "f32[][]cpu" = executorch_exir_dialects_edge__ops_aten_full_like_default(aten_sum_dim_int_list, 1, pin_memory = False, memory_format = torch.preserve_format)

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 6 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:11 in forward, code: expand = torch.ops.aten.expand.default(full_like, [2]);  full_like = None
        aten_expand_copy_default: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_expand_copy_default(aten_full_like_default, [2]);  aten_full_like_default = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 7 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:12 in forward, code: pow_2 = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
        aten_pow_tensor_scalar_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar(aten_sub_tensor, 1.0);  aten_sub_tensor = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 8 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:13 in forward, code: mul_1 = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        aten_mul_scalar: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Scalar(aten_pow_tensor_scalar_1, 2.0);  aten_pow_tensor_scalar_1 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 9 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:14 in forward, code: mul_2 = torch.ops.aten.mul.Tensor(expand, mul_1);  expand = mul_1 = None
        aten_mul_tensor_1: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_expand_copy_default, aten_mul_scalar);  aten_expand_copy_default = aten_mul_scalar = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=True, device=CPU:0), debug_handle: 10 # File: <eval_with_key>.919 from /Users/a1234/miniforge3/envs/executorch/lib/python3.11/site-packages/torch/fx/experimental/proxy_tensor.py:1720 in wrapped:15 in forward, code: mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_2);  mul_2 = primals_2 = None
        aten_mul_tensor_2: "f32[2][1]cpu" = executorch_exir_dialects_edge__ops_aten_mul_Tensor(aten_mul_tensor_1, x);  aten_mul_tensor_1 = x = None
        return (aten_sum_dim_int_list, aten_mul_tensor_2, p_weight)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %p_weight: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}, "weight_has_gradient": true}
+# %p_weight: {"constant_tag": "_default_external_constant", "spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}, "weight_has_gradient": true}
 # %_lifted_tensor_constant0: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.int64", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
 # %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
 # %aten_mul_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["2"], "stride": ["1"]}}
```

</details>

## `MemoryPlanningPass.run`

关键变化：TensorSpecs receive lifetime, mem_id, and mem_offset; arena sizes are recorded

<details>
<summary>展开print_readable对比</summary>

### Before · `before`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %a: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
# %b: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, a: "f32[4][1]cpu", b: "f32[4][1]cpu"):
        # No stacktrace found for following nodes
        alloc: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:838 in forward, code: return (a + b) * a
        aten_add_tensor: "f32[4][1]cpu" = torch.ops.aten.add.out(a, b, out = alloc);  b = alloc = None

        # No stacktrace found for following nodes
        alloc_1: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:838 in forward, code: return (a + b) * a
        aten_mul_tensor: "f32[4][1]cpu" = torch.ops.aten.mul.out(aten_add_tensor, a, out = alloc_1);  aten_add_tensor = a = alloc_1 = None
        return (aten_mul_tensor,)
```

### After · `after.graph`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %a: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 5], "mem_id": 1, "mem_offset": 32, "shape": ["4"], "stride": ["1"]}}
# %b: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 3], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}}
# %alloc: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [2, 5], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [2, 5], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
# %alloc_1: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [4, 6], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [4, 6], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [4, 6], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, a: "f32[4][1]cpu", b: "f32[4][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:838 in forward, code: return (a + b) * a
        aten_add_tensor: "f32[4][1]cpu" = torch.ops.aten.add.out(a, b, out = alloc);  b = alloc = None

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc_1: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:838 in forward, code: return (a + b) * a
        aten_mul_tensor: "f32[4][1]cpu" = torch.ops.aten.mul.out(aten_add_tensor, a, out = alloc_1);  aten_add_tensor = a = alloc_1 = None
        return (aten_mul_tensor,)
```

### Unified diff

```diff
--- before
+++ after
@@ -1,19 +1,21 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %a: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
-# %b: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
-# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
-# %aten_mul_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}}
-# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [null, null], "mem_id": null, "mem_offset": null, "shape": ["4"], "stride": ["1"]}]}
+# %a: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 5], "mem_id": 1, "mem_offset": 32, "shape": ["4"], "stride": ["1"]}}
+# %b: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 3], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}}
+# %alloc: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [2, 5], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
+# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [2, 5], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
+# %alloc_1: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [4, 6], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}}
+# %aten_mul_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [4, 6], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}}
+# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [4, 6], "mem_id": 1, "mem_offset": 0, "shape": ["4"], "stride": ["1"]}]}

 class GraphModule(torch.nn.Module):
     def forward(self, a: "f32[4][1]cpu", b: "f32[4][1]cpu"):
-        # No stacktrace found for following nodes
+        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
         alloc: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

         # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:838 in forward, code: return (a + b) * a
         aten_add_tensor: "f32[4][1]cpu" = torch.ops.aten.add.out(a, b, out = alloc);  b = alloc = None

-        # No stacktrace found for following nodes
+        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
         alloc_1: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

         # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:838 in forward, code: return (a + b) * a
```

</details>

## `MemoryPlanningPass.run_multimethod`

关键变化：same mutable state across methods is assigned shared arena mem_id=2 and identical offset

<details>
<summary>展开print_readable对比</summary>

### Before · `before.graphs[0]`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %_lifted_tensor_constant1: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.int64", "lifetime": [0, 7], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 4], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
# %alloc: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 10], "mem_id": 1, "mem_offset": 32, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 10], "mem_id": 1, "mem_offset": 32, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %aten_view_copy_default: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 10], "mem_id": 1, "mem_offset": 32, "shape": ["4"], "stride": ["1"]}}
# %alloc_1: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [6, 9], "mem_id": 1, "mem_offset": 0, "shape": [], "stride": []}}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [6, 9], "mem_id": 1, "mem_offset": 0, "shape": [], "stride": []}}
# %alloc_2: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
# %copy_: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [11, 11], "mem_id": 1, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}, {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[2, 2][2, 1]cpu", _lifted_tensor_constant1: "i64[][]cpu", x: "f32[1][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2, 2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc: "f32[2, 2][2, 1]cpu" = executorch_exir_memory_alloc(((2, 2), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[2, 2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        aten_add_tensor: "f32[2, 2][2, 1]cpu" = torch.ops.aten.add.out(b_state, x, out = alloc);  x = alloc = None

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        aten_view_copy_default: "f32[4][1]cpu" = executorch_exir_memory_view(aten_add_tensor, [-1])

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc_1: "f32[][]cpu" = executorch_exir_memory_alloc(((), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = torch.ops.dim_order_ops._to_dim_order_copy.out(_lifted_tensor_constant1, dim_order = [], out = alloc_1);  _lifted_tensor_constant1 = alloc_1 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc_2: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        aten_mul_tensor: "f32[4][1]cpu" = torch.ops.aten.mul.out(aten_view_copy_default, dim_order_ops__to_dim_order_copy_default, out = alloc_2);  aten_view_copy_default = dim_order_ops__to_dim_order_copy_default = alloc_2 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2, 2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        copy_: "f32[2, 2][2, 1]cpu" = torch.ops.aten.copy_.default(b_state, aten_add_tensor);  b_state = aten_add_tensor = None
        return (copy_, aten_mul_tensor)
```

### Before · `before.graphs[1]`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 3], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 2], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
# %copy_: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 2], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 3], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}, null]}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[2, 2][2, 1]cpu", state: "f32[1][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[1], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        copy_: "f32[1][1]cpu" = torch.ops.aten.copy_.default(state, state);  state = None
        return (copy_, None)
```

### Before · `before.graphs[2]`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %b_state: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %output: {"spec": [{"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[2, 2][2, 1]cpu"):
        return (b_state,)
```

### After · `after.graphs[0]`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %_lifted_tensor_constant1: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.int64", "lifetime": [0, 7], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
# %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 4], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
# %alloc: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 10], "mem_id": 1, "mem_offset": 32, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %aten_add_tensor: {"debug_handle": 1, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 10], "mem_id": 1, "mem_offset": 32, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %aten_view_copy_default: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 10], "mem_id": 1, "mem_offset": 32, "shape": ["4"], "stride": ["1"]}}
# %alloc_1: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [6, 9], "mem_id": 1, "mem_offset": 0, "shape": [], "stride": []}}
# %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [6, 9], "mem_id": 1, "mem_offset": 0, "shape": [], "stride": []}}
# %alloc_2: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
# %aten_mul_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
# %copy_: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [11, 11], "mem_id": 1, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}, {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[2, 2][2, 1]cpu", _lifted_tensor_constant1: "i64[][]cpu", x: "f32[1][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[2, 2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc: "f32[2, 2][2, 1]cpu" = executorch_exir_memory_alloc(((2, 2), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[2, 2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 1 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        aten_add_tensor: "f32[2, 2][2, 1]cpu" = torch.ops.aten.add.out(b_state, x, out = alloc);  x = alloc = None

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        aten_view_copy_default: "f32[4][1]cpu" = executorch_exir_memory_view(aten_add_tensor, [-1])

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc_1: "f32[][]cpu" = executorch_exir_memory_alloc(((), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        dim_order_ops__to_dim_order_copy_default: "f32[][]cpu" = torch.ops.dim_order_ops._to_dim_order_copy.out(_lifted_tensor_constant1, dim_order = [], out = alloc_1);  _lifted_tensor_constant1 = alloc_1 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        alloc_2: "f32[4][1]cpu" = executorch_exir_memory_alloc(((4,), torch.float32))

        # spec: TensorSpec(dtype=torch.float32, shape=[4], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0), debug_handle: 2 # File: /Users/a1234/Desktop/executorch/tools/pass_debug/trace_builtin_passes.py:859 in forward, code: return self.state.add_(x).view(-1) * 2
        aten_mul_tensor: "f32[4][1]cpu" = torch.ops.aten.mul.out(aten_view_copy_default, dim_order_ops__to_dim_order_copy_default, out = alloc_2);  aten_view_copy_default = dim_order_ops__to_dim_order_copy_default = alloc_2 = None

        # spec: TensorSpec(dtype=torch.float32, shape=[2, 2], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        copy_: "f32[2, 2][2, 1]cpu" = torch.ops.aten.copy_.default(b_state, aten_add_tensor);  b_state = aten_add_tensor = None
        return (copy_, aten_mul_tensor)
```

### After · `after.graphs[1]`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 3], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 2], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
# %copy_: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 2], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
# %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 3], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}, null]}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[2, 2][2, 1]cpu", state: "f32[1][1]cpu"):
        # spec: TensorSpec(dtype=torch.float32, shape=[1], layout=torch.strided, is_sparse=False, shape_dynamism=0, const=False, requires_grad=False, device=CPU:0) # No stacktrace found for following nodes
        copy_: "f32[1][1]cpu" = torch.ops.aten.copy_.default(state, state);  state = None
        return (copy_, None)
```

### After · `after.graphs[2]`

```python
# Codex custom node.meta summary (not part of FX print_readable output)
# %b_state: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
# %output: {"spec": [{"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}]}

class GraphModule(torch.nn.Module):
    def forward(self, b_state: "f32[2, 2][2, 1]cpu"):
        return (b_state,)
```

### Unified diff · graph 1

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
+# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
 # %_lifted_tensor_constant1: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.int64", "lifetime": [0, 7], "mem_id": null, "mem_offset": null, "shape": [], "stride": []}}
 # %x: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 4], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
 # %alloc: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 10], "mem_id": 1, "mem_offset": 32, "shape": ["2", "2"], "stride": ["2", "1"]}}
@@ -9,7 +9,7 @@
 # %dim_order_ops__to_dim_order_copy_default: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [6, 9], "mem_id": 1, "mem_offset": 0, "shape": [], "stride": []}}
 # %alloc_2: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
 # %aten_mul_tensor: {"debug_handle": 2, "spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}}
-# %copy_: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
+# %copy_: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 11], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
 # %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [11, 11], "mem_id": 1, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}, {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [8, 11], "mem_id": 1, "mem_offset": 16, "shape": ["4"], "stride": ["1"]}]}

 class GraphModule(torch.nn.Module):
```

### Unified diff · graph 2

```diff
--- before
+++ after
@@ -1,5 +1,5 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 3], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
+# %b_state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 3], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
 # %state: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 2], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
 # %copy_: {"spec": {"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 2], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}}
 # %output: {"spec": [{"const": false, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [3, 3], "mem_id": 1, "mem_offset": 0, "shape": ["1"], "stride": ["1"]}, null]}
```

### Unified diff · graph 3

```diff
--- before
+++ after
@@ -1,6 +1,6 @@
 # Codex custom node.meta summary (not part of FX print_readable output)
-# %b_state: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}}
-# %output: {"spec": [{"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": null, "mem_offset": null, "shape": ["2", "2"], "stride": ["2", "1"]}]}
+# %b_state: {"spec": {"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}}
+# %output: {"spec": [{"const": true, "device": "CPU", "device_index": 0, "dtype": "torch.float32", "lifetime": [0, 1], "mem_id": 2, "mem_offset": 0, "shape": ["2", "2"], "stride": ["2", "1"]}]}

 class GraphModule(torch.nn.Module):
     def forward(self, b_state: "f32[2, 2][2, 1]cpu"):
```

</details>
