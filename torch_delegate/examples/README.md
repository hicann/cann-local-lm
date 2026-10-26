# TorchDelegate 示例

本目录提供 TorchDelegate 的端到端验证示例，覆盖模型导出、C++ 接入、Fallback 机制和精度对比等场景。

## 示例概览

| 示例目录 | 说明 |
|---------|------|
| [`export_and_load_example`](./export_and_load_example/) | 最小 C++ 接入示例：加载 `.epair`、构建 GE Graph、执行推理并与 eager 精度对比 |
| [`inplace_input`](./inplace_input/) | Inplace 输入操作行为一致性验证：对比 `torch.export` / `AOTI` / `torch.compile` / `torch_delegate epair` 四种路径 |
| [`sym_int_case`](./sym_int_case/) | Fallback 机制对动态/静态非 Tensor 输入的支持验证（`int scale`、动态 `batch_size`） |
| [`sym_pack_optimization_case`](./sym_pack_optimization_case/) | Sym Pack 优化验证：对比 torchair `optimize_sym_pack` 消除冗余 Pack 与 epair 导出 + C++ 加载推理两条路径 |
| [`assert_tensor_metadata_demo`](./assert_tensor_metadata_demo/) | 演示 `torch.export` 自动插入的 `aten._assert_tensor_metadata` 节点的移除方法，并导出 epair 在 C++ 侧加载执行 |
| [`accuracy_compare`](./accuracy_compare/) | 端到端精度对比（Reference vs Delegate），使用 BeamSearch 模型验证 NPU 上的数值一致性 |
| [`inplace_return_case`](./inplace_return_case/) | Inplace 输入 + Tensor 返回值组合场景的 Fallback codegen 正确性验证 |
| [`fallback_inplace_optional_case`](./fallback_inplace_optional_case/) | Fallback codegen 对多 Inplace 参数和多可选参数的正确性验证 |
| [`fallback_dynamic_optional_case`](./fallback_dynamic_optional_case/) | Fallback codegen 对动态输入 (Tensor[]) 和可选输入 (Tensor?) 的正确性验证 |
| [`run_model_multi_instance`](./run_model_multi_instance/) | 多进程并发示例：一进程一实例（绕开 GE 同步 `RunGraph` 进程级锁），worker 内生产者-消费者线程并行调用 `RunModel`，度量 host 并发下发与 device 多流并行 |
| [`common`](./common/) | 公共工具脚本（`build_libtorch_npu.sh`，供各示例复用） |

## 环境准备

运行示例前，首先需要准备 Python 虚拟环境：

```bash
# 创建虚拟环境并安装所有依赖（torch、torch_npu、triton、triton-ascend 等）
cd examples
bash prepare_env.sh
```

该脚本会：
1. 创建 `venv_test` 虚拟环境
2. 安装 `requirements.txt` 中的 Python 依赖
3. 安装 `torch==2.9.0`（CPU 版本）、`triton==3.5.0`、`triton-ascend==3.2.1`
4. 从 Daily 构建下载并安装 `torch_npu`

## 批量运行

使用 `run_all.sh` 可一键执行所有示例：

```bash
bash run_all.sh
```

前置条件：
- 已设置 `ASCEND_HOME_PATH` 环境变量（`source CANN set_env.sh`）
- 已安装 torch_delegate whl（`TORCH_DELEGATE_ROOT` 由 `run_all.sh` 自动从已安装包推导，无需 source set_env.sh）

执行过程：
1. 创建 `venv_test` 虚拟环境（如已存在则跳过）
2. 检查环境变量
3. 激活虚拟环境
4. 依次执行每个已启用的示例
5. 汇总 PASS/FAIL/SKIP 结果并输出到 `case_result.txt`

可通过修改 `run_all.sh` 中的 `CASES` 数组控制各用例的启用/禁用状态：

```bash
# enabled  = 执行
# disabled = 跳过
# ignored  = 忽略
CASES=(
  "export_and_load_example|run.sh|enabled"
  "inplace_input|run.sh|enabled"
  "sym_int_case|run.sh|enabled"
  "sym_pack_optimization_case|run.sh|enabled"
  "assert_tensor_metadata_demo|run.sh|enabled"
  "accuracy_compare|run.sh|ignored"
  "fallback_inplace_optional_case|run.sh|enabled"
  "fallback_dynamic_optional_case|run.sh|enabled"
  "run_model_multi_instance|run.sh|enabled"
)
```
