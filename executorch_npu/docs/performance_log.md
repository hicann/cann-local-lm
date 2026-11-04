# Gemma4 / Qwen3.5轻量性能与数值回归记录

固定容量5120，batch=1，BF16；输入128/512/2048 tokens，每次输出32 tokens。
轨迹取初始PTE首次贪心结果，后续eager/PTE均喂入相同前缀；输出不因EOS提前停止。
各用例预热1次、测量3次，报告中位数；模型加载、文件读取、日志及logits快照不计时。
TTFT包含请求状态初始化、mask、搬运、prefill和选词；总延迟为移除快照开销后的各步耗时之和。
TPOT=(总延迟−TTFT)/31；吞吐=32/总延迟秒。原始wall_with_capture_ms保留供审计。
这些是固定轨迹的推理端到端指标，不包含tokenization，也不是服务并发吞吐。
余弦比较完整词表最终logits；step=0为prefill。逐层检查和MMLU不属于本快速集。
初始eager使用与PTE相同的模型代码、静态容量和有状态decode；PTE标准export。
每次命令默认顺序执行两个模型；请保持NPU独占。数值WARN用于人工检查，不自动覆盖基线。

## 2026-09-30T03:25:31.834669+00:00 — baseline — gemma4

结果目录：`/mnt/workspace/executorch-npu-eval/light_baseline_20260930_v2/gemma4`；配置/提交/产物指纹见 `metadata.json`。

| 用例 | 后端 | 对eager最低余弦 | 对初始PTE最低余弦 | 对PTE top-1一致率 | TTFT ms | TPOT ms | 总延迟ms | tokens/s | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99988709 | 1.00000000 | 100.00% | 107.21 | 30.60 | 1053.48 | 30.375 | BASELINE |
| short (128) | eager | 1.00000000 | 0.99988709 | 100.00% | 213.99 | 200.50 | 6429.48 | 4.977 | BASELINE |
| medium (512) | pte | 0.99994653 | 1.00000000 | 100.00% | 116.16 | 30.35 | 1056.88 | 30.278 | BASELINE |
| medium (512) | eager | 1.00000000 | 0.99994653 | 96.88% | 211.38 | 194.80 | 6248.66 | 5.121 | BASELINE |
| long (2048) | pte | 0.99997342 | 1.00000000 | 100.00% | 289.70 | 31.29 | 1259.04 | 25.416 | BASELINE |
| long (2048) | eager | 1.00000000 | 0.99997342 | 100.00% | 247.81 | 192.62 | 6220.21 | 5.145 | BASELINE |

## 2026-09-30T03:41:14.565549+00:00 — baseline — qwen3_5

结果目录：`/mnt/workspace/executorch-npu-eval/light_baseline_20260930_v2/qwen3_5`；配置/提交/产物指纹见 `metadata.json`。

| 用例 | 后端 | 对eager最低余弦 | 对初始PTE最低余弦 | 对PTE top-1一致率 | TTFT ms | TPOT ms | 总延迟ms | tokens/s | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99985068 | 1.00000000 | 100.00% | 1058.08 | 12.40 | 1442.69 | 22.181 | BASELINE |
| short (128) | eager | 1.00000000 | 0.99985068 | 96.88% | 1731.52 | 126.86 | 5664.11 | 5.650 | BASELINE |
| medium (512) | pte | 0.99957259 | 1.00000000 | 100.00% | 1696.56 | 12.28 | 2076.90 | 15.408 | BASELINE |
| medium (512) | eager | 1.00000000 | 0.99957259 | 100.00% | 1612.48 | 127.80 | 5574.33 | 5.741 | BASELINE |
| long (2048) | pte | 0.99985404 | 1.00000000 | 100.00% | 4265.02 | 12.87 | 4668.01 | 6.855 | BASELINE |
| long (2048) | eager | 1.00000000 | 0.99985404 | 100.00% | 1676.02 | 126.31 | 5591.71 | 5.723 | BASELINE |

## 2026-09-30T03:44:00.448941+00:00 — 基线复跑（无优化） — gemma4

结果目录：`/mnt/workspace/executorch-npu-eval/light_replay_20260930/gemma4`；配置/提交/产物指纹见 `metadata.json`。

| 用例 | 后端 | 对eager最低余弦 | 对初始PTE最低余弦 | 对PTE top-1一致率 | TTFT ms | TPOT ms | 总延迟ms | tokens/s | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99988709 | 1.00000000 | 100.00% | 80.34 | 30.73 | 1035.09 | 30.915 | PASS |
| medium (512) | pte | 0.99994653 | 1.00000000 | 100.00% | 118.54 | 30.81 | 1075.32 | 29.759 | PASS |
| long (2048) | pte | 0.99997342 | 1.00000000 | 100.00% | 286.13 | 32.89 | 1307.89 | 24.467 | PASS |


## 验证与当前波动范围

已通过runner构建、4项比较逻辑测试，以及两个模型各3个用例、每用例3次测量的实际验证。
Gemma4单次加载连续运行与分别加载的三个用例，轨迹及完整logits完全一致，验证了用例间状态隔离。
Gemma4无优化复跑的模型/PTE/PTD/runner指纹与评测源码指纹均未改变，三个用例全部PASS，logits与初始PTE完全一致。

相同代码、相同产物下，本次总延迟波动如下（不是优化收益）：

- short：总延迟变化 -1.75%。
- medium：总延迟变化 +1.74%。
- long：总延迟变化 +3.88%。

短输入TTFT从107.21 ms到80.34 ms，单次运行之间存在明显波动。以上是一次复跑的观察范围，不是统计置信区间；同量级的小收益应复测。表中PASS/WARN仅表示数值回归判定，不表示性能改善。

初始eager与PTE并非逐token完全相等：Gemma4 medium和Qwen3.5 short各有1/32步top-1不同；保留这些初始差异，后续主要检查相对初始PTE是否新增漂移。

Qwen3.5历史单次PTE加载约13.5分钟，本次同样观察到较长加载等待。评测已采用每模型一次加载运行三个用例；加载时间不混入TTFT/TPOT/总延迟/输出吞吐。
## 2026-09-30T07:51:11.824141+00:00 — GDN阶段一：融合prefill与BF16 state — qwen3_5

结果目录：`/mnt/workspace/executorch-npu-eval/qwen_gdn_stage01_20260930/regression/qwen3_5`；配置/提交/产物指纹见 `metadata.json`。

| 用例 | 后端 | 对eager最低余弦 | 对初始PTE最低余弦 | 对PTE top-1一致率 | TTFT ms | TPOT ms | 总延迟ms | tokens/s | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99984300 | 0.99984394 | 100.00% | 84.42 | 12.05 | 459.50 | 69.641 | WARN |
| medium (512) | pte | 0.99960461 | 0.99979134 | 100.00% | 60.51 | 11.60 | 420.24 | 76.147 | WARN |
| long (2048) | pte | 0.99974154 | 0.99975077 | 100.00% | 131.77 | 11.70 | 494.37 | 64.729 | WARN |


## GDN阶段一：导出成本与误差分解

环境：executorch_npu；加载CANN 9.1.0与 `/mnt/workspace/libtorch_delegate/set_env.sh`。融合开关默认关闭，原始基线不覆盖。

下列导出耗时对照历史记录，未重新执行旧版完整导出，不是同一次受控A/B：

| 项目 | 历史参考s | 本次s |
|---|---:|---:|
| PT2 prefill捕获 | 263.61 | 20.04 |
| PT2 prefill保存 | 158.05 | 14.85 |
| PT2总计 | 450.29 | 63.05 |
| PTE总计 | 13863.54 | 666.64 |

Prefill新图统计：`{"aten.matmul.default": 13, "npu.npu_chunk_gated_delta_rule.default": 18}`。

数值分解（各用例最低余弦）：

| 用例 | PTE vs融合eager | 融合eager vs仅BF16 state | 融合eager vs初始eager |
|---|---:|---:|---:|
| short | 0.99984300 | 0.99983443 | 0.99979635 |
| medium | 0.99967152 | 0.99981586 | 0.99981846 |
| long | 0.99965519 | 0.99982986 | 0.99978250 |

现有回归判定：**WARN**。保持原阈值；WARN表示存在数值漂移，不能据此声明精度无变化。单个动态GDN PTE的8个长度与原生算子逐元素一致，证明转换路径本身已通过该定向检查。

完整产物、原始logits和诊断记录：`/mnt/workspace/executorch-npu-eval/qwen_gdn_stage01_20260930`。
## 2026-09-30T09:08:24.783011+00:00 — Recipes阶段2/3/4组合 — qwen3_5

结果目录：`/mnt/workspace/executorch-npu-eval/qwen_recipe234_20260930/regression/qwen3_5`；配置/提交/产物指纹见 `metadata.json`。

| 用例 | 后端 | 对eager最低余弦 | 对初始PTE最低余弦 | 对PTE top-1一致率 | TTFT ms | TPOT ms | 总延迟ms | tokens/s | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99984938 | 0.99984077 | 100.00% | 49.86 | 9.71 | 350.84 | 91.210 | WARN |
| medium (512) | pte | 0.99950715 | 0.99968010 | 100.00% | 60.43 | 9.99 | 371.50 | 86.138 | WARN |
| long (2048) | pte | 0.99963177 | 0.99964774 | 100.00% | 112.88 | 9.80 | 416.58 | 76.816 | WARN |


## Recipes阶段2/3/4组合：相对阶段一

采用recipes-infer 92d9e1f的FA、QKVZ/BA与QKV/门控投影融合、RMSNorm/Add-RMSNorm、recurrent GDN方案；适配现有连续KV Cache，容量5120、BF16 state。默认参考路径和阶段一产物均保留。

| 用例 | TTFT加速比 | TPOT加速比 | 总延迟加速比 | 对阶段一最低余弦 | top-1一致率 |
|---|---:|---:|---:|---:|---:|
| short | 1.693 | 1.241 | 1.310 | 0.99986479 | 100.00% |
| medium | 1.001 | 1.161 | 1.131 | 0.99960874 | 100.00% |
| long | 1.167 | 1.194 | 1.187 | 0.99974587 | 100.00% |

PTE vs优化后eager的对比：

- short：最低余弦0.99986483，top-1一致率100.00%。
- medium：最低余弦0.99971009，top-1一致率100.00%。
- long：最低余弦0.99973002，top-1一致率100.00%。

对初始基线的判定为 **WARN**；不放宽原阈值。产物和原始结果：`/mnt/workspace/executorch-npu-eval/qwen_recipe234_20260930`。
## 2026-09-30T14:44:35.957546+00:00 — Gemma4 recipes融合适配 — gemma4

结果目录：`/mnt/workspace/executorch-npu-eval/gemma_recipe_20260930/regression/gemma4`；配置/提交/产物指纹见 `metadata.json`。

| 用例 | 后端 | 对eager最低余弦 | 对初始PTE最低余弦 | 对PTE top-1一致率 | TTFT ms | TPOT ms | 总延迟ms | tokens/s | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99997014 | 0.99991453 | 100.00% | 96.39 | 31.27 | 1065.90 | 30.022 | WARN |
| medium (512) | pte | 0.99994846 | 0.99997083 | 100.00% | 105.15 | 29.52 | 1020.16 | 31.368 | WARN |
| long (2048) | pte | 0.99997441 | 0.99997545 | 100.00% | 210.76 | 32.71 | 1224.77 | 26.127 | WARN |


## Gemma4 E2B recipes融合适配

参考cann-recipes-infer 92d9e1f Gemma4报告：FA v2（连续BNSD、显式窗口掩码、scale=1）、RMSNorm/AddRMSNorm、RoPE查表和按两半配对的部分旋转。保留共享KV、GELU tanh、PLE及最终softcap。容量5120；原始基线不变。PT2捕获后直接传入编译器，未序列化PT2文件，捕获时间不能与包含序列化的历史时间直接比较。

- short PTE vs优化后eager：最低余弦0.99993527，top-1一致率100.00%。
- medium PTE vs优化后eager：最低余弦0.99996993，top-1一致率100.00%。
- long PTE vs优化后eager：最低余弦0.99997646，top-1一致率100.00%。

初始基线判定 **WARN**，不放宽阈值。结果 `/mnt/workspace/executorch-npu-eval/gemma_recipe_20260930`；大体积PTE/PTD存于 `/tmp/gemma_recipe_20260930/pte`，artifacts/pte为软链接，尚未备份到持久卷。

## 2026-10-10 — vLLM Ascend单卡对比

单卡910B3；BF16；TP=1；输入128/512/2048、输出32；预热1次、测量3次取各指标中位数。vLLM 0.30.0+empty / vllm-ascend 0.19.1rc2.dev2667+g7fca7cae5，固定镜像manifest c030d15d93a7d9b3c24347637929b0defe07dade7042704b5fdf65f4d378d36e。

以下为FULL_DECODE_ONLY图模式，已确认ACL graph replay生效。TTFT/总延迟为离线引擎请求端到端时间，含调度/IPC，不含模型加载、分词、文本解码和HTTP。TPOT=(总延迟−TTFT)/31；输出吞吐=32/总延迟秒。

| 模型 | 输入tokens | TTFT ms | TPOT ms | 总延迟ms | 输出tokens/s |
|---|---:|---:|---:|---:|---:|
| gemma4 | 128 | 163.91 | 19.71 | 775.06 | 41.287 |
| gemma4 | 512 | 173.54 | 20.33 | 809.61 | 39.525 |
| gemma4 | 2048 | 157.64 | 18.51 | 732.89 | 43.663 |
| qwen3_5 | 128 | 178.04 | 56.36 | 1925.26 | 16.621 |
| qwen3_5 | 512 | 168.46 | 56.22 | 1912.85 | 16.729 |
| qwen3_5 | 2048 | 175.39 | 56.55 | 1920.79 | 16.660 |

并发有限批次输出吞吐（各请求512输入/32输出）：Gemma4并发4/8为138.14/274.75 tokens/s；Qwen3.5为58.80/116.41 tokens/s。不是HTTP持续负载最大吞吐。

Gemma4需两项兼容性修复：BF16视觉初始化quant_method属性保护、共享KV首次prefill分页缓存路径。单请求与并发共10项attention定向对照输出逐元素一致；保留模型权重与共享KV结构。Gemma4 long末token及并发输出存在差异，未做完整数值回归/MMLU，不声明精度无变化。Qwen eager/graph三用例及并发输出一致。

历史PTE为固定轨迹，本次vLLM自由贪心生成；输入token IDs与历史基线完全相同，但不是严格同轨迹A/B。历史recipes判定仍为WARN；数据保留，不覆盖。完整eager/graph对照、配置、补丁、逐次原始数据与复现脚本：`/mnt/workspace/executorch-npu-eval/vllm_compare_20261010/REPORT.md`。


## 2026-10-12 — msprof采集

使用保留的Gemma4 recipes与Qwen3.5 recipes234完整PTE/PTD，SHA256与原始优化记录一致。容量5120、BF16、batch=1；输入128/512/2048、输出32，预热1次、采集1次，冻结轨迹，两个模型串行执行。

两模型三组结果与timeline/算子CSV均已导出；采集包含加载和预热，不将带profiling的耗时写回性能基线。Qwen正常退出。Gemma三组结果完成后出现堆内存退出错误，msprof返回码仍为0；独立长用例补采状态：FAILED，异常原因尚未定位。

命令、指纹、数值对照和中文说明：`/mnt/workspace/executorch-npu-eval/msprof_20261012`；完整原始PROF及导出文件已压缩保存：`/mnt/workspace/executorch-npu-eval/msprof_20261012/profiles.tar.gz`。

补充对照：同一Gemma独立长用例不带msprof时正常退出（exit=0），带msprof的原始采集及独立补采均在完整结果写入后出现退出错误。两模型六组logits与保留的优化PTE结果逐元素一致；异常根因尚未定位。


### 2026-10-12 — Qwen cache拷贝源码与编译路径复核

已核对原始完整PT2、实际TorchDelegate GE转换、当前torch_npu原生定向profiling和已有PTE trace。原生scatter_update_只有Scatter；PTE函数式Scatter引入12个KV TensorMove。Linear显式copy_在图中消除，但decode仍有36个recurrent-state Transpose；functional GDN的18个IR TensorMove未单独下发，不能按IR计数替代任务计数。尾部60 MiB D2D的地址级归因仍待确认。

证据与重新分析报告：/mnt/workspace/executorch-npu-eval/msprof_20261012/cache_reanalysis/evidence。模型/PTE未修改。


### 2026-10-12：自定义GE Scatter cache原位更新pass（子图原型）
- skill：ge-fusion-pass-skill，GitCode master e7a4876f72724382aab1d659413d054205f42dd2；case：/mnt/workspace/executorch-npu-eval/ge_scatter_reinplace_20261012。
- C++ graph pass在BeforeInferShape将显式许可的Data→TensorMove→Scatter短路为Data→Scatter。私有OPP加载，默认无许可不替换；旧cache仍被其他路径读取时跳过。
- 实际PT2→PTE→NPU：BF16 [1,2,5120,256] cache，baseline/custom共16轮，完整cache与读取结果逐位一致；skill三组图检查、八组输出对照通过。
- 最后PreRunAfterBuild图：prefill/decode的TensorMove均由1降至0，旧cache观察负例保持1；没有测设备任务数或整网提速。
- 现场离线BuildModel不支持ge.exec.inputReuseMemIndexes；原型用实验进程CANN_ET_SCATTER_REINPLACE_INPUTS显式声明可改写输入。
- 完整Qwen/Gemma重编译、整网精度、TTFT/TPOT/吞吐、独立控制边/子图故障注入均未运行；不改变既有整网基线结论。验证报告见case根validation-report.md。


### 2026-10-13：Qwen已有profiling的TransData/切片/Cast/Transpose分析
数据仍为2026-10-12完整PTE采集，正式轮prefill4/decode95；没有重新导出或修改模型。
- TransData：prefill38个/1.60358ms，decode36个/0.28552ms。decode全部为18层卷积的NCHW↔NC1HWC0 FLOAT转换；prefill另外2个aclnnMm TransData共0.91244ms，shape/format为N/A，具体转换对象待查。
- StridedSlice：prefill257个/3.78862ms，decode244个/0.63294ms；涵盖投影split/chunk、partial RoPE、卷积裁剪与state窗口。prefill末4列state提取18个共0.97742ms。
- Cast：prefill269个/2.19472ms，decode208个/0.32176ms。decode97个FLOAT→BF16标量Cast均来自lifted常量，应查常量折叠；不能直接删除卷积、归一化、门控等数值敏感转换。
- Transpose：prefill79个/1.26728ms，decode36个/0.34558ms。decode全部为18层GDN state前后交换最后两轴，建议研究统一state存储布局。
累计任务时间不是可直接节省的端到端延迟。后续重点是causal-conv完整链融合、GDN state布局、常量Cast、prefill尾切片及LMhead相关转换。
证据目录：/mnt/workspace/executorch-npu-eval/msprof_20261012/analysis_iterations_3_63，完整GE溯源在cache_reanalysis。


## 2026-10-13 — Qwen3.5 GDN state布局验证启动

以nohup串行运行布局测试→完整PT2导出→从已保存PT2编译PTE→旧/新PTE精度性能对比→当前eager验证→旧/新msprof采集→报告。仅验证BF16 state [B,H,V,K] 布局，PTE导出进程禁用标量Cast折叠；保留并复测recipes234完整基线。batch=1、cache=5120、输入128/512/2048、输出32、预热1次、正式测量3次，固定输入及token轨迹；profiling单独采集。重点统计实际设备GDN state Transpose、按shape分组的TensorMove和异步D2D拷贝。

任务与日志：/mnt/workspace/executorch-npu-eval/gdn_state_layout_20261013；大产物：/tmp/gdn_state_layout_20261013。源码快照和SHA256已冻结，任一步失败自动停止。结果待完成后追加。


## 2026-10-13 — Qwen3.5 GDN state布局验证中断

错误：`RuntimeError('layout_tests exit=1; see logs/layout_tests.log')`。证据目录：`/mnt/workspace/executorch-npu-eval/gdn_state_layout_20261013`。此前基线与历史WARN保留。


## 2026-10-13 — Qwen3.5 GDN布局与标量Cast折叠联合验证启动

按用户要求启用当前已有的标量Cast折叠，同时验证BF16 recurrent state [B,H,V,K] 布局。前一轮在导出前因state I/O测试失败停止（6通过、1失败），未产生PT2/PTE。已复现PyTorch功能化将跨dtype copy_隐式转换消除的问题，恢复显式BF16转换以保持导出图与缓存合约一致；BF16正常路径转换为无操作。之前失败日志完整保留。

nohup串行：GDN与Cast单测→PT2保存→PTE编译→旧/新精度性能验证→当前eager验证→旧/新msprof→报告。保留recipes234完整基线并重新测量；cache5120、batch1、输入128/512/2048、输出32、预热1、正式3次。重点记录Transpose、TensorMove、D2D及Cast变化和实际标量折叠数量。性能为两项修改的联合收益，不能单独归因。

任务目录：/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013；大产物：/tmp/gdn_state_cast_20261013；结果待追加。


## 2026-10-13 — Qwen3.5 GDN state布局验证中断

错误：`RuntimeError('pte_export exit=1; see logs/pte_export.log')`。证据目录：`/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013`。此前基线与历史WARN保留。


### Qwen3.5联合验证：PTE阶段续跑

36项GDN/Cast测试通过；完整PT2导出65.03秒，prefill.pt2与decode.pt2均已保存。首次PTE在TorchDelegate导入前停止；prefill分区已记录实际折叠97个标量Cast。后台脚本现保留set_env.sh设置的PYTHONPATH并前置源码快照，TorchDelegate及冻结源码导入均已验证。复用既有PT2，以nohup从PTE阶段续跑及执行后续精度、性能和profiling。失败现场保留于/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/attempt_01_environment_failure。


## 2026-10-13 — Qwen3.5 profiling统计更正

初版报告按外层Model ID过滤，漏掉decode静态GE嵌套Model ID=1的主体任务。初版“没有新增TensorMove”结论已撤回；初版数据与脚本保留在/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/reports_before_nested_model_fix。不改变原始采集和数值/性能结果，按互不重叠的实际迭代时间窗重新统计全部设备任务，完整校验每窗18个GDN与6个full-attention算子。下方更正报告为准。


## 2026-10-13 — Qwen3.5 GDN state布局与标量Cast折叠验证

统一BF16 recurrent state为 `[B,H,V,K]`，同时启用当前标量Cast折叠实现。固定PT2→PTE流程；旧recipes234 PTE完整保留并重新测量。

batch=1、cache=5120、输入128/512/2048、输出32；冻结输入和teacher-forcing轨迹；性能预热1次、测量3次取中位数；profiling另行串行采集旧/新产物，各预热1次、测量1次。

| 输入 | TTFT旧→新ms | TPOT旧→新ms | 总延迟 旧→新ms | 输出吞吐 旧→新token/s | 最低余弦（对当前PTE） | 精度 |
|---|---|---|---|---|---|---|
| short | 62.828→54.401 | 10.750→9.703 | 394.457→353.567 | 81.124→90.506 | 1.0000000000 | PASS |
| medium | 67.580→64.791 | 10.453→10.463 | 393.589→389.133 | 81.303→82.234 | 1.0000000000 | PASS |
| long | 113.503→117.367 | 9.883→10.210 | 419.161→433.876 | 76.343→73.754 | 1.0000000000 | PASS |

GDN state Transpose全部消失：True；新增TensorMove：True；新增D2D拷贝：False。

各编译分区实际标量Cast折叠数量：[97, 97]。两项优化同时开启，性能收益不能单独归因于布局修改。

| 正式迭代 | state Transpose旧→新 | 全部Transpose旧→新 | 全部Cast旧→新 | TensorMove旧→新 | 新增搬运 |
|---|---|---|---|---|---|
| short_prefill | 18→0 | 79→61 | 269→269 | 12→12 | [] |
| short_decode | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |
| medium_prefill | 18→0 | 79→61 | 269→269 | 12→12 | [] |
| medium_decode_first | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |
| medium_decode | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |
| long_prefill | 18→0 | 79→61 | 269→269 | 12→12 | [] |
| long_decode | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |

同时记录初始eager对照、当前eager对照、旧PTE复测稳定性、逐次精度与性能范围。历史recipes234对初始eager的WARN不被覆盖。profiling按实际设备任务、shape、拷贝方向和大小统计；不能仅凭IR的TensorMove节点数认定实际发生了搬运。decode的静态GE任务使用嵌套Model ID=1，与step_trace外层ID不同，已纳入同一迭代时间窗；每个时间窗校验18个GDN和6个full-attention算子且没有迭代重叠。

完整证据：`/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013`；报告：`/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/reports/comparison.json`。
旧trace：`/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/profiles/current_baseline/raw/PROF_000001_20261013152541501_03519402OCPKNDCB/mindstudio_profiler_output/msprof_20261013152630.json`；新trace：`/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/profiles/native_gdn_layout/raw/PROF_000001_20261013152708619_03519829RFLHBMNJ/mindstudio_profiler_output/msprof_20261013152755.json`。

### decode新增18个state TensorMove的来源分析

来源是当前TorchAir的 `_ge_concrete_graph/ge_converter/custom/recurrent_gated_delta_rule.py:64`：functional converter先执行 `state_copy = ge.TensorMove(state)`，再把副本交给会更新state的 `ge.RecurrentGatedDeltaRule`。保存的torch_npu C++ 源码也对应 `state.clone()` 后调用原生recurrent GDN；每个linear-attention层一份，共18份，每份BF16 [1,16,128,128]，512 KiB。

新旧PT2图JSON已直接核对。旧版state输入链：cache→transpose→to(BF16)→contiguous→functional GDN，输出再转置回cache布局；新版：cache→to(BF16)→functional GDN。新版cache本身已经是连续BF16，[B,H,V,K]，同dtype转换与contiguous不再生成独立state临时存储。

旧版完整GE审计IR中也已经存在这18个TensorMove：首层为Transpose_115→Identity_1（clone的GE映射）→TensorMove_1→RecurrentGatedDeltaRule。旧profiling实际仅下发Transpose→GDN→Transpose；新版实际为TensorMove→GDN。因此不是模型新增18个copy_，而是已有functional输入保护副本现在独立下发。旧版转置生成可独占的临时buffer，新版直接读持久cache图输入，解释了两者的复制消除条件差异。负责旧版复制消除/内存复用的具体GE pass尚未确认，不将其归因于某个已证实的reinplace pass。

本次新增的state TensorMove位于GDN之前；cache.update_recurrent_state的copy_位于GDN之后，且旧版也存在。相同dtype/shape的功能化aten.copy在当前converter中可以直接返回src，不能仅凭Python出现copy_就把这18个任务归到缓存回写。运行时state输入使用current，输出使用next，Commit交换两者；删除converter保护副本或改用原位GDN需要同时验证导出功能化、输入别名/生存期及state I/O绑定，不能直接把TensorMove换成Identity。

512输入的正式decode迭代95：旧36个state Transpose累计336.60 us；新18个state TensorMove累计190.46 us，相关state搬运任务数36→18，累计任务时长约减少146.14 us（43.4%）。12个K/V大TensorMove保持不变。新增TensorMove不等于state搬运净增加，也不能直接用它解释此前2048用例端到端回退3.5%。


## 2026-10-13 — GDN / Scatter共用GE pass子图与持久state验证

同一C++ pass保留Scatter input0规则，新增RecurrentGatedDeltaRule state input4规则；独占可修改缓存输入才消除前置TensorMove，GDN输出0/1保持。真实GE有8个有效输入，兼容10 anchor中两个尾部空optional输入。

CANN9.1/910B3隔离构建通过。GDN baseline/custom共16组完整输出及state逐位一致，旧state观察者负例正确跳过；真实stateful runner共4次prefill+7步decode全部logits逐位一致；Scatter回归16组全部逐位一致。两个profile的输出也逐位一致，未见应用退出异常。

单层session正式decode迭代9：state TensorMove baseline 1个/7.58us，custom 0个；两侧全部捕获范围的512KiB D2D均为0，复制未转移到输出DMA。未修改运行时current/next实现。

这是单层原型验证，不代表完整Qwen已消除18个state拷贝或已有整网加速收益；尚未接入产品编译路径或重编译完整Qwen/Gemma。外部地址别名、失败回滚及独立控制边/子图故障注入未覆盖，G3 DEGRADED/NOT_RUN。源码、验证与逐步骤矩阵：/mnt/workspace/executorch-npu-eval/ge_cache_reinplace_20261013。


## 2026-10-13 — Qwen GDN GE pass整网编译验证启动

按用户授权以nohup串行运行decode GE输入审计→完整PTE编译→无pass/有pass精度性能复测→两侧msprof→报告。复用/tmp/gdn_state_cast_20261013/pt2完整prefill/decode PT2与冻结源码；对照当前布局统一+Cast折叠PTE，仅decode的18个owned recurrent state输入允许原位化，prefill/Scatter不变。cache5120、输入128/512/2048、输出32、batch1、预热1次、正式测量3次；固定输入和轨迹；同源eager参考复用上轮结果。重点核对18个state TensorMove及全部采集范围的512KiB D2D。

任务目录：/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013；大产物：/tmp/qwen_ge_cache_reinplace_20261013。原基线和PT2保留，结果待追加。


## 2026-10-13 — Qwen3.5完整模型GDN GE pass验证

基线与优化侧均使用BF16 `[B,H,V,K]` state布局和相同标量Cast折叠。复用保存的完整PT2编译PTE，仅decode的18个recurrent state输入允许原位化；prefill与Scatter保持基线行为。

batch=1、cache=5120、输入128/512/2048、输出32；冻结输入和teacher-forcing轨迹；性能预热1次、测量3次取中位数；profiling另行串行采集旧/新产物，各预热1次、测量1次。

| 输入 | TTFT旧→新ms | TPOT旧→新ms | 总延迟 旧→新ms | 输出吞吐 旧→新token/s | 最低余弦（对当前PTE） | 精度 |
|---|---|---|---|---|---|---|
| short | 54.817→49.361 | 9.830→9.438 | 359.536→341.946 | 89.004→93.582 | 1.0000000000 | PASS |
| medium | 58.530→59.488 | 9.538→9.790 | 353.362→362.030 | 90.559→88.390 | 1.0000000000 | PASS |
| long | 111.569→109.721 | 9.413→9.441 | 403.361→403.582 | 79.333→79.290 | 1.0000000000 | PASS |

GDN state Transpose全部消失：True；新增TensorMove：False；新增D2D拷贝：True。

各编译分区标量Cast折叠数量：[97, 97]，本轮保持该优化。

正式decode state TensorMove计数：{'short_decode': {'baseline': 18, 'optimized': 0}, 'medium_decode': {'baseline': 18, 'optimized': 0}, 'long_decode': {'baseline': 18, 'optimized': 0}}；18个是否全部消除：True。

全部采集范围新增512KiB state D2D事件数：3348。

| 正式迭代 | state Transpose旧→新 | 全部Transpose旧→新 | 全部Cast旧→新 | TensorMove旧→新 | 新增搬运 |
|---|---|---|---|---|---|
| short_prefill | 0→0 | 61→61 | 269→269 | 12→12 | [] |
| short_decode | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |
| medium_prefill | 0→0 | 61→61 | 269→269 | 12→12 | [] |
| medium_decode_first | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |
| medium_decode | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |
| long_prefill | 0→0 | 61→61 | 269→269 | 12→12 | [] |
| long_decode | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |

同时记录初始eager对照、当前eager对照、旧PTE复测稳定性、逐次精度与性能范围。历史recipes234对初始eager的WARN不被覆盖。profiling按实际设备任务、shape、拷贝方向和大小统计；不能仅凭IR的TensorMove节点数认定实际发生了搬运。decode的静态GE任务使用嵌套Model ID=1，与step_trace外层ID不同，已纳入同一迭代时间窗；每个时间窗校验18个GDN和6个full-attention算子且没有迭代重叠。

完整证据：`/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013`；报告：`/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013/reports/comparison.json`。
旧trace：`/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013/profiles/without_pass/raw/PROF_000001_20261013192758665_03559383PCRQKRFN/mindstudio_profiler_output/msprof_20261013192844.json`；新trace：`/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013/profiles/with_pass/raw/PROF_000001_20261013192919780_03559803MHHIPBKP/mindstudio_profiler_output/msprof_20261013193006.json`。


## 2026-10-13 — Qwen prefill/decode单buffer state验证

对照：相同GDN GE pass、BF16布局及Cast折叠；双buffer → 单buffer。复用已验证PT2。
输入128/512/2048，输出32，性能预热1次、测量3次取中位数；profiling单独采集。

| 用例 | TTFT ms | TPOT ms | 总延迟ms | 输出token/s | 精度 |
|---|---|---|---|---|---|
| short | 43.568→52.061 | 9.417→9.936 | 335.054→362.436 | 95.507→88.291 | PASS |
| medium | 52.667→54.492 | 9.999→9.922 | 362.027→363.250 | 88.391→88.094 | PASS |
| long | 112.051→109.705 | 10.290→9.750 | 432.258→410.988 | 74.030→77.861 | PASS |

全部采集中的512 KiB state D2D：3348 → 0。
验收通过：True。完整证据：`/mnt/workspace/executorch-npu-eval/state_single_buffer_20261013/full`。

按shape和大小归属state搬运；逐窗口核对18个GDN与6个Full Attention算子。


## 2026-10-13 — Qwen单buffer性能交替复测

复用模型与冻结runner，4对交替顺序会话，每例预热3次、测量7次；每版每例共28次。用例顺序也轮换。
所有测量请求logits均与保留基线逐位一致。百分比为四对会话各自中位数变化的中位数。

| 输入 | TTFT ms双→单 | TPOT ms双→单 | 总延迟ms双→单 | 吞吐token/s双→单 | 配对总延迟变化 | 四对变化范围 |
|---|---|---|---|---|---|---|
| short | 45.574→46.516 | 9.539→9.137 | 343.871→330.519 | 93.059→96.817 | -1.81% | -5.97% ~ +4.43% |
| medium | 53.590→51.385 | 9.462→9.280 | 344.060→341.865 | 93.007→93.605 | -3.89% | -7.52% ~ +5.62% |
| long | 108.275→106.947 | 9.581→9.540 | 405.064→402.435 | 79.000→79.517 | -1.88% | -5.64% ~ +4.41% |

原始逐次测量、配对结果及环境快照：`/tmp/qwen_perf_recheck_20261013`。本轮不改源码、不重新导出、不将profiling耗时作为性能结果。

分析结论：原短输入总延迟 +8.17% 的回退没有稳定复现。三种输入长度均为前三对会话更快、第四对更慢，存在明显会话间波动；四对数据不足以宣称稳定加速，也不能精确归因到CPU调度或设备频率。建议保留已验证正确且消除state搬运的单buffer实现，将性能记为小幅改善趋势。

已有profiling的所选short decode迭代：算子累计耗时5195.66→5193.04 μs，异步拷贝累计耗时170.64→20.78 μs。这支持搬运确实减少，但这些数值来自单独profiling，不能直接替代无profiling的请求延迟。

本轮第一次复测因 /mnt/workspace写满而中断，已整体保留到 /tmp/qwen_perf_recheck_20261013/attempt1_disk_full，不纳入上述统计。有效复测退出码为0，共168次测量，全部logits与保留基线逐位一致。


## 2026-10-14 — Qwen单buffer性能交替复测

复用模型与冻结runner，4对交替顺序会话，每例预热3次、测量7次；每版每例共28次。用例顺序也轮换。
所有测量请求logits均与保留基线逐位一致。百分比为四对会话各自中位数变化的中位数。

| 输入 | TTFT ms双→单 | TPOT ms双→单 | 总延迟ms双→单 | 吞吐token/s双→单 | 配对总延迟变化 | 四对变化范围 |
|---|---|---|---|---|---|---|
| short | 44.154→44.847 | 9.349→9.234 | 335.071→333.423 | 95.502→95.976 | -1.39% | -2.65% ~ -0.50% |
| medium | 51.613→51.346 | 9.458→9.209 | 348.101→338.816 | 91.927→94.447 | -1.70% | -5.37% ~ +0.40% |
| long | 109.328→107.804 | 10.256→9.633 | 427.685→405.793 | 74.883→78.874 | -2.83% | -7.53% ~ +0.40% |

原始逐次测量、配对结果及环境快照：`/tmp/qwen_perf_recheck_20261014`。本轮不改源码、不重新导出、不将profiling耗时作为性能结果。


## 2026-10-16 — 两模型最新代码全量评测已启动

源码 `f37cb9b56d603e6b2b72da0c05310770698b2339` 加启动时工作区修改，快照：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/source`。后台PID `3934960`。
串行执行重新构建、PT2→PTE、全量5-shot MMLU和固定轨迹性能测试。当前尚无本轮完成结果。
任务状态与日志：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/status.json`、`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/run.log`；每模型完成后追加结果。


## 2026-10-16 — 最新代码MMLU与性能 — gemma4

源码快照基于 `f37cb9b56d603e6b2b72da0c05310770698b2339` 及启动时全部未提交修改；来源与SHA256：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/source_manifest.json`。
由该快照重新构建runner，重新导出PT2→PTE；按当前仓库默认编译入口执行，未额外注入实验性GE pass。
BF16、batch=1、cache=5120。全量MMLU：57科、14,042题、5-shot，复用历史token输入，不截断、不抽样。

| MMLU | 本轮 | 历史标准PTE | 变化 |
|---|---:|---:|---:|
| 正确数 | 6853/14042 | 6877/14042 | -24 |
| 准确率 | 48.8036% | 48.9745% | -0.1709 pp |

答案一致率：97.7354%；旧对新错112，旧错新对88。
逐题分数、预测与分科结果：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/gemma4/mmlu/summary.jsonl`、`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/gemma4/mmlu/summary.json`。

性能为独立无profiling的固定轨迹测试；输入128/512/2048、输出32，预热3次、测量7次。模型加载与logits诊断快照不计入四项指标。

| 输入 | TTFT ms | TPOT ms | 总延迟ms | 输出token/s | 对初始PTE最低余弦 | top-1一致率 | 数值判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| short (128) | 89.763 | 32.824 | 1106.729 | 28.914 | 0.99991453 | 100.00% | WARN |
| medium (512) | 88.830 | 29.350 | 1000.238 | 31.992 | 0.99997083 | 100.00% | WARN |
| long (2048) | 200.617 | 33.801 | 1247.631 | 25.649 | 0.99997545 | 100.00% | WARN |

完整源码、环境、PT2/PTE指纹及各阶段日志：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/gemma4`；共享runner指纹见 `artifacts.json`。
历史数值基线与WARN均保留，不放宽阈值。全量评分墙钟时间包含加载，不作为上述固定轨迹推理延迟。


## 2026-10-16 — qwen3_5最新代码评测中断

错误：`CalledProcessError(1, ['/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/runner', 'session', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/pte/model.pte', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/pte/ascend.weights.package.ge_aot.ptd', 'prefill', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/inputs/short.i64', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/inputs/short.positions.i64', 'decode', '128', '1,1', '32', '0', 'light_eval', '--max-seq-len', '5120', '--eval-output', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/pte', '--warmup', '3', '--repeats', '7', '--causal-mask', '--eval-manifest', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/pte/cases.tsv'])`。日志：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/logs`。未将部分评分标记为完成；继续评测另一模型。


## 2026-10-16 — Qwen3.5全量MMLU已独立续跑

原任务decode性能检查失败；评分使用已通过冒烟的prefill。续跑PID `3974189`，复用并校验本轮PTE/PTD与runner，不重做导出。当前尚无完整Qwen分数。


## 2026-10-16 — Qwen3.5全量MMLU续跑结果

复用本轮最新代码快照及已完成的PT2/PTE、runner；未重新导出，未额外注入GE pass。
全量57科、14,042题、5-shot、不截断。评分走prefill；decode性能问题单独保留。

MMLU：**41.2263%（5789/14042）**。
相对历史标准PTE：+0.1139 pp；答案一致率98.5472%。
旧对新错54；旧错新对70。

逐题与分科结果：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/mmlu/summary.jsonl`、`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/mmlu/summary.json`。
性能评测仍失败：`decode logits must be valid float32 or bfloat16 with vocabulary last`；没有有效完整性能结果，不能报出四项指标。
原失败日志：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/pte/runner.log`。
本节仅将完整MMLU标记完成，不将decode性能问题标记通过。


## 2026-10-16 — Qwen decode NaN定位对照已启动

已复现：prefill的248,320个logits均有限；第一步decode的248,320个logits全为NaN，shape/dtype正常。新旧产物/运行时交叉组合均被ABI检查拒绝，不能据此归因。
隔离对照仅将decode的18个recurrent state改为double，复用本轮PT2；其余模型计算、权重、编译入口不变。不改仓库源码或覆盖已评分PTE。PID `3990154`，目录：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`。尚无对照结论。


## 2026-10-16 — Qwen decode NaN定位对照已启动

已复现：prefill的248,320个logits均有限；第一步decode的248,320个logits全为NaN，shape/dtype正常。新旧产物/运行时交叉组合均被ABI检查拒绝，不能据此归因。
隔离对照仅将decode的18个recurrent state改为double，复用本轮PT2；其余模型计算、权重、编译入口不变。不改仓库源码或覆盖已评分PTE。PID `3995653`，目录：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`。尚无对照结论。


## 2026-10-16 — Qwen decode NaN定位对照已启动

已复现：prefill的248,320个logits均有限；第一步decode的248,320个logits全为NaN，shape/dtype正常。新旧产物/运行时交叉组合均被ABI检查拒绝，不能据此归因。
隔离对照仅将decode的18个recurrent state改为double，复用本轮PT2；其余模型计算、权重、编译入口不变。不改仓库源码或覆盖已评分PTE。PID `4003672`，目录：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`。尚无对照结论。


## 2026-10-16 — Qwen decode recurrent state双buffer对照结果

此为隔离诊断对照，不替代原最新代码single-buffer失败结果；原全量MMLU仍来自原产物。

| 输入 | TTFT ms | TPOT ms | 总延迟ms | token/s | 数值判定 |
|---|---:|---:|---:|---:|---|
| short | 43.061 | 9.577 | 341.013 | 93.838 | WARN |
| medium | 49.163 | 9.571 | 346.272 | 92.413 | WARN |
| long | 104.704 | 9.174 | 389.291 | 82.201 | WARN |

仅改变decode recurrent state的buffer mode。完整证据：`/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`。
若本对照通过，只说明问题与该绑定路径相关，尚不能定位到具体CANN/运行时操作。


## 2026-10-19 — Qwen single buffer判定缺陷确认

同一份10月16日PTE/PTD、同一诊断运行时，single第一decode全NaN；仅在加载后把18个decode recurrent state的运行时绑定改为double，即得到有限输出。没有重编译OM，排除了重编译与新旧ABI差异作为这项对照的解释。
正在验证正式编译器修复：保留TensorMove的functional GDN不宣告边界同址，回退double；真实直接引用和未读取旧state的路径保留single。证据：`/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019`。


## 2026-10-19 — Qwen recurrent state单buffer判定修复与双buffer回退

根因定位到编译器过宽的alias声明：GDN state输出引用TensorMove的结果，而不是原Data输入；原判定把这两块存储视为同一边界，允许同址绑定。相同PTE的运行时单变量对照已复现NaN并用double消除。
正式修复：该保护拷贝路径回退double；真实直接引用及未读旧state的single路径保留。没有新增模型选项或修改CANN/TorchAir，全量评分历史保留。

修复前回归用例2项失败、1项通过；修复后状态规划/manifest/backend/compiler-context测试通过。正常默认导出得到18个decode recurrent state的double契约；3例 × 7次 × 32 token全部有限，logits与已验证双buffer对照逐位一致。初始数值阈值仍为WARN，不放宽。

| 输入 | TTFT ms | TPOT ms | 总延迟ms | token/s | 数值判定 |
|---|---:|---:|---:|---:|---|
| short | 44.917 | 9.331 | 335.820 | 95.289 | WARN |
| medium | 50.786 | 9.514 | 344.986 | 92.757 | WARN |
| long | 105.044 | 9.307 | 393.499 | 81.322 | WARN |

已应用补丁：`/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019/fix.diff`；新产物：`/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019/pte`；完整证据：`/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019`。
本轮未重新运行全量MMLU；此前分数来自原10月16日产物，不能标成新产物重测结果。


## 2026-10-19 — GDN state原位化正式集成验证

默认流水线现为GE降图 → 基于device-only state契约的保守原位化 → 实际拓扑buffer规划 → 离线编译。不依赖外部OPP pass注入、手写Data索引或模型recipe开关；保留上一轮严格alias修复，未命中继续double。
范围限定为8输入、2输出、静态BF16 [1,H,128,128] 的RecurrentGatedDeltaRule。保护拷贝及state必须无额外读者与控制边，输出必须对应同一state契约；改图失败中止导出。
正常Qwen导出实际改写18处Data→TensorMove→GDN为Data→GDN，并在随后规划出18个decode recurrent single。连续decode、跨请求重置覆盖3例×7次×32 token，输出全部有限且与安全double对照逐位一致。初始baseline与阈值不变，原数值判定仍为WARN；本轮没有重跑全量MMLU。

| 输入 | 新single TPOT ms | 同日double TPOT ms | TPOT降低 | 新TTFT ms | 新token/s |
|---|---:|---:|---:|---:|---:|
| short | 9.731 | 12.067 | 19.35% | 47.049 | 91.676 |
| medium | 10.719 | 10.659 | -0.56% | 58.926 | 81.454 |
| long | 9.626 | 11.821 | 18.57% | 110.039 | 78.330 |

表中收益为同日顺序运行的实测，不代表已排除频率/负载噪声。图改写证据不单独等同于运行期任务消除，未采集msprof时不宣称最终TensorMove任务数。
新产物：`/mnt/workspace/executorch-npu-eval/gdn_reinplace_integration_20261019/pte`；完整证据与补丁：`/mnt/workspace/executorch-npu-eval/gdn_reinplace_integration_20261019`；安全双buffer产物保留：`/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019/pte`。


## 2026-10-19 — 按用户要求撤回single buffer / reinplace方案

当前代码统一恢复current/next双buffer，移除Python GE原位改写、自动single规划、manifest buffer_mode字段及运行时同址绑定分支。保留原有模型适配、融合、状态契约和评测工具。此前single buffer / reinplace结果仅作为历史实验记录，不再代表当前默认实现。旧实验PTE中的buffer_mode字段会被拒绝，需重新导出并使用本轮重建的运行时。回退前备份与补丁：`/mnt/workspace/executorch-npu-eval/double_buffer_rollback_20261019`；正在执行构建与回归验证。


## 2026-10-19 — 双buffer回退验证完成

single buffer / reinplace代码及专属测试已撤回，schema已同步。状态输入固定绑定current、输出固定绑定next，成功后交换；只读state不交换。C++ / Python回归通过，实际NPU探针验证输入输出地址不同、成功交换、失败不提交、只读不交换以及reset/release。

基于当前回退代码重新构建运行时并正常导出Qwen：18个decode GDN保留TensorMove，所有状态契约不再含buffer_mode。3类输入 × 7次 × 32 token全部有限，logits与已验证的安全double对照逐位一致。原数值WARN与阈值保留。本轮未重新导出Gemma，也未重跑全量MMLU。

当前运行时：`/mnt/workspace/executorch_npu/build/release/executorch_npu_stateful_runner`；新Qwen PTE：`/mnt/workspace/executorch-npu-eval/double_buffer_rollback_20261019/pte`；证据与回退补丁：`/mnt/workspace/executorch-npu-eval/double_buffer_rollback_20261019`。


## 2026-10-19 — 当前双buffer两模型全量评测已启动

源码 `f37cb9b56d603e6b2b72da0c05310770698b2339` 加启动时工作区修改，快照：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/source`。后台PID `12041`。
冻结当前双buffer代码并重新构建runner；Gemma重新导出PTE，Qwen复用经源码一致性验证的双buffer产物。两模型均重新跑全量57科 / 14,042题 / 5-shot MMLU和固定轨迹性能测试。当前尚无本轮完成结果。
任务状态与日志：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/status.json`、`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/run.log`；每模型完成后追加结果。


## 2026-10-19 — 当前双buffer全量MMLU与性能 — gemma4

源码快照基于 `f37cb9b56d603e6b2b72da0c05310770698b2339` 及启动时全部未提交修改；来源与SHA256：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/source_manifest.json`。
由该快照重新构建runner；固定current/next双buffer，成功后交换，无buffer_mode / reinplace。
Gemma：核对模型/导出代码及PT2指纹后，使用当前默认编译器重新导出PTE。
BF16、batch=1、cache=5120。全量MMLU：57科、14,042题、5-shot，复用历史token输入，不截断、不抽样。

| MMLU | 本轮 | 历史标准PTE | 变化 |
|---|---:|---:|---:|
| 正确数 | 6853/14042 | 6877/14042 | -24 |
| 准确率 | 48.8036% | 48.9745% | -0.1709 pp |

答案一致率：97.7354%；旧对新错112，旧错新对88。
逐题分数、预测与分科结果：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/gemma4/mmlu/summary.jsonl`、`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/gemma4/mmlu/summary.json`。

性能为独立无profiling的固定轨迹测试；输入128/512/2048、输出32，预热3次、测量7次。模型加载与logits诊断快照不计入四项指标。

| 输入 | TTFT ms | TPOT ms | 总延迟ms | 输出token/s | 对初始PTE最低余弦 | top-1一致率 | 数值判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| short (128) | 88.816 | 30.943 | 1049.607 | 30.488 | 0.99991453 | 100.00% | WARN |
| medium (512) | 98.056 | 29.320 | 1006.293 | 31.800 | 0.99997083 | 100.00% | WARN |
| long (2048) | 215.141 | 32.893 | 1232.509 | 25.963 | 0.99997545 | 100.00% | WARN |

完整源码、环境、PT2/PTE指纹及各阶段日志：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/gemma4`；共享runner指纹见 `artifacts.json`。
历史数值基线与WARN均保留，不放宽阈值。全量评分墙钟时间包含加载，不作为上述固定轨迹推理延迟。


## 2026-10-19 — 当前双buffer全量MMLU与性能 — qwen3_5

源码快照基于 `f37cb9b56d603e6b2b72da0c05310770698b2339` 及启动时全部未提交修改；来源与SHA256：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/source_manifest.json`。
由该快照重新构建runner；固定current/next双buffer，成功后交换，无buffer_mode / reinplace。
Qwen：源码逐项核对与已通过回归的双buffer产物一致，复用该PTE，本轮重新评分及测量性能。
BF16、batch=1、cache=5120。全量MMLU：57科、14,042题、5-shot，复用历史token输入，不截断、不抽样。

| MMLU | 本轮 | 历史标准PTE | 变化 |
|---|---:|---:|---:|
| 正确数 | 5789/14042 | 5773/14042 | +16 |
| 准确率 | 41.2263% | 41.1124% | +0.1139 pp |

答案一致率：98.5472%；旧对新错54，旧错新对70。
逐题分数、预测与分科结果：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/qwen3_5/mmlu/summary.jsonl`、`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/qwen3_5/mmlu/summary.json`。

性能为独立无profiling的固定轨迹测试；输入128/512/2048、输出32，预热3次、测量7次。模型加载与logits诊断快照不计入四项指标。

| 输入 | TTFT ms | TPOT ms | 总延迟ms | 输出token/s | 对初始PTE最低余弦 | top-1一致率 | 数值判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| short (128) | 50.673 | 9.609 | 348.122 | 91.922 | 0.99984077 | 100.00% | WARN |
| medium (512) | 54.051 | 9.403 | 345.717 | 92.561 | 0.99968010 | 100.00% | WARN |
| long (2048) | 113.402 | 9.827 | 418.499 | 76.464 | 0.99964774 | 100.00% | WARN |

完整源码、环境、PT2/PTE指纹及各阶段日志：`/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/qwen3_5`；共享runner指纹见 `artifacts.json`。
历史数值基线与WARN均保留，不放宽阈值。全量评分墙钟时间包含加载，不作为上述固定轨迹推理延迟。
