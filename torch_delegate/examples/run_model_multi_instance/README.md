# run_model_multi_instance

简体中文 | [English](README_en.md)

多进程并发端到端示例：调用 `td::EpairModelLoader::RunModel`（同步接口），
验证**多实例并发执行**——父进程拉起多个worker进程（一进程一实例），
每个worker内部是"生产者线程+消费线程"，并行调用各自实例的 `RunModel`，
观测host侧多进程并发下发与device侧多流并行。

## 为什么是多进程（一进程一实例）

当前部署的 `RunModel` 封装GE同步 `RunGraph`：`InnerSession::RunGraph` 抢
**进程级static mutex**（`try_to_lock`，锁持有至图完成），同进程多线程并发调用
会被 `GE_SESS_ALREADY_RUNNING` 顶回并串行化。一进程一实例（各自独立的
`GeSession`/ACL上下文/Device流）可绕开进程内共享锁，恢复device侧多流并发。

## 模型（深而宽）

`export_matmul_epair.py` 生成 `depth` 层堆叠、每层 `branches` 条**独立** MatMul+ReLU
分支的网络（默认24层 × 8分支 ≈ 600个算子：每层8 MatMul + 9 ReLU + 8 Add；
权重跨层复用，显存占用可控）：

- 深层数+多算子：单次 `RunModel` 的Device执行时间达到毫秒级，并发效果可稳定观测；
- 图内独立分支：每层多条互不依赖的MatMul+ReLU分支，算子规模大、结构规则。

## 并发度量口径

两阶段（`CLOCK_MONOTONIC`，跨进程时间可直接比较）：

1. **SOLO**：每个实例单独执行1个任务（串行基线，父进程逐个放行），得到单任务耗时 `t_solo`；
2. **RUN**：全部实例并发执行剩余任务。

| 指标 | 含义 | 判读 |
|------|------|------|
| `host_max_inflight` | `RunModel` 调用区间的扫描线最大重叠数（同时进入接口的进程数） | ≥2即host侧并发下发；≈workers为满载 |
| `device_parallelism` | `并行任务数 × t_solo / wall`（`RunModel` 阻塞至图完成，wall反映device吞吐） | ≈workers表示device多流并行充分；≈1表示被串行化 |
| `avg concurrency` | `Σ任务耗时 / wall`（参考值） | 数量级参考 |

## 前置条件

1. 已完成 `torch_delegate` whl的构建与安装（`pip install <torch_delegate whl>`），
   `TORCH_DELEGATE_ROOT` 会自动从已安装的whl推导，无需 `source set_env.sh`
2. 已 `source <cann_path>/cann/set_env.sh`
3. 生成模型与输入：
   ```bash
   python3 export_matmul_epair.py --dim 1024 --depth 24 --branches 8
   ```
   生成 `matmul_model.epair` 与 `input_0.bin`（`dim × dim` 的float32）

## 一键运行

```bash
bash run.sh                       # 默认：8 进程（实例），48 任务

# 可覆盖参数：
INSTANCES=8 TASKS=48 DIM=1024 DEPTH=24 BRANCHES=8 bash run.sh
MIN_PARALLEL_RATIO=0.5 bash run.sh         # 并发门禁比例
TD_MULTI_TIMEOUT_S=900 bash run.sh         # 栅栏超时（秒）
```

`run.sh` 校验：全部任务成功、checksum一致，**并且**并发门禁：
`host_max_inflight >= 2`、`device_parallelism >= INSTANCES * MIN_PARALLEL_RATIO`。

## 手动构建与运行

```bash
cmake -S . -B build \
    -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
    -DCANN_ROOT="${ASCEND_HOME_PATH}" \
    -DTORCH_PKG_DIR="$(python3 -c 'import os,torch;print(os.path.dirname(torch.__file__))')"
cmake --build build -j

# <epair> [workers=8] [tasks=48] [dim=1024]，要求 tasks >= 2*workers
./build/run_model_multi_instance matmul_model.epair 8 48 1024
```

## device侧多流的来源

**跨进程多流**：每进程独立ACL上下文与流，各实例的执行stream相互独立，
多进程并发调用 `RunModel` 时各实例的算子在不同stream上并行执行。

> 注：GE图内自动多流（`ge.autoMultistreamParallelMode`）为实验特性，本示例
> 暂不开启，device侧并行完全来自跨进程多流。

## 用msprof直接观察多流并行（可选）

```bash
msprof --output=./prof_multi_instance \
  --application="./build/run_model_multi_instance matmul_model.epair 8 48 1024"
```

查看timeline：各进程算子应落在不同stream上且执行区间重叠。

> 注：msprof的设备侧算子采集按device独占注册，多进程形态下op_summary通常只
> 含其中一个worker的算子（可用应用侧指标 `device_parallelism` 判断整体并行度）。

## 预期结果

- 全部 `tasks` 个任务执行成功，输出checksum一致；
- `host_max_inflight >= 2` 且 `device_parallelism` 接近 `workers`，device侧算子多流并行。

> [!NOTE]说明
> `RunModel` 返回即代表图执行完成、输出（Host侧）可直接读取；每个实例只被
> 其所属worker进程调用，进程之间无GE共享锁。
