# host_cpu_mixed_case：CPU/NPU 混合调度（HostCpu 后端）端到端示例

本示例验证 TorchDelegate 的 CPU/NPU 混合调度能力：没有 NPU 执行路径、只能在
CPU 上执行的算子（自定义 namespace 仅注册 CPU 实现），与 NPU 算子同处一张
`ge::Graph`，由 GE 统一调度执行。

## 模型形态

| 模型 | 结构 | 覆盖场景 |
|------|------|---------|
| `model_mixed` | mm(NPU) → `hcpu_ops::feature_scale`(CPU) → relu(NPU) | CPU 段夹在 NPU 段中间 |
| `model_pure_cpu` | `feature_scale`(CPU) → `feature_merge`(CPU) | 纯 CPU 段 |
| `model_non_adjacent` | `feature_scale`(CPU) → mm(NPU) → relu(NPU) → `feature_merge`(CPU, 共享输入) | 非相邻 CPU 段 + 共享输入 |
| `model_recommender` | `feature_gather`(CPU embedding 查表) → 3 层 MLP(NPU Task Tower) → `score_rank`(CPU 排序打分) | 推荐网络三段式架构近似 |

`hcpu_ops::feature_scale` / `hcpu_ops::feature_merge` 只注册 CPU kernel（无
PrivateUse1 实现），导出期 Device 归属自动判定将其判为 `device="cpu"` 并固化进
epair；加载期 FallbackCodegen 生成注册到 GE HostCpu 后端（`REG_OP_BACKEND` +
`ge::OpBackend::kHostCPU`）的自定义算子，`Execute` 回调 PyTorch Dispatcher 的
CPU 实现。

> 已知规避：`model_recommender` 的索引输入以 float32 传入、CPU kernel 内转回
> int64 查表。GE runtime 对 HostCpu 算子的 Data（用户运行时输入）做
> device→host staging 时按 4 字节/元素计算拷贝长度，DT_INT64 输入会被截断；
> 构图侧已加 fail-fast 检测（`FallbackNodePlanner::BuildInputDefs`）。GE 修复
> 后可还原为 int64 输入。

## 运行

```bash
bash run.sh
```

四个阶段：

1. **导出与归属校验**：导出四个模型，断言 manifest v2（`version=2`、
   `cpu_nodes` 计数）与 `ascend_ir.device` 注入正确；
2. **epair 内容校验**：解包 `.epair` 检查 manifest 与 device 字段；
3. **C++ 加载执行**：`host_cpu_runner` 依次加载三个 epair，
   `EpairModelLoader` 构造期完成 HostCpu codegen（g++ 编译 + dlopen 注册），
   `CompileAndLoad` 后一次 `RunModelWithStreamAsync` 执行混合图（需要 NPU 设备）；
   runner 先 `dlopen libcustom_ops.so`（C++ 侧 `TORCH_LIBRARY` 注册，与
   `custom_ops.py` 的 schema 一致）——C++ 部署进程与 Python 导出进程不共享
   Dispatcher 注册表；
4. **精度对比**：delegate 输出与 eager CPU 基线逐元素对比。

无 NPU 设备时阶段 3/4 自动跳过（仅验证导出与归属判定）。

## 关键观测点

- 归属报告：导出结束时打印完整归属清单（NPU/CPU 双侧，CPU 侧按 namespace 分级，
  aten→CPU 标记为可疑 WARNING，自定义 namespace→CPU 为预期 INFO）；
- manifest v2：`cpu_nodes` 统计 CPU 节点数；
- 混合图执行：对外仍是一次 `RunGraphWithStreamAsync`，host↔device 搬运由 GE 在
  区段边界统一插入，CPU 区段内数据驻留 host 内存零搬运流转。
