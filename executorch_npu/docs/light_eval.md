# 轻量端到端回归

目标：用三个固定文本输入及32个输出步骤，快速对比优化前后的最终logits和推理延迟。不是MMLU替代品。

## 用法

在 `executorch_npu` 环境中直接传入模型、PTE和runner路径，不在仓库保存某次运行的JSON配置。下面以Gemma为例，Qwen改为`--model qwen3_5`并指定对应路径。

```bash
bash examples/scripts/run_light_eval.sh baseline \
  --model gemma4 --model-path /mnt/workspace/models/gemma-4-E2B-it \
  --pte-dir /path/to/baseline/pte --output /path/to/light_baseline --stage baseline

bash examples/scripts/run_light_eval.sh run \
  --model gemma4 --model-path /mnt/workspace/models/gemma-4-E2B-it \
  --pte-dir /path/to/candidate/pte --baseline /path/to/light_baseline \
  --output /path/to/light_stage_01 --stage 'stage_01: 优化内容'
```

可传`--runner`、`--capacity`、`--output-tokens`、`--warmup`、`--repeats`和`--device-id`。运行配置及元数据写入结果目录。`--config`仅用于可选的外部配置/旧结果重放。后续run只执行候选PTE，不重新运行eager，默认追加`docs/performance_log.md`；完整基线存在时禁止覆盖。

模型相关优化需要使用现有导出入口重新生成PTE；评测不会自动导出。runtime优化需要先重建runner。

## 固定条件

- Cache容量5120、batch=1、BF16，输入长度128/512/2048，输出32步。
- 每个用例第一次预热以初始PTE贪心生成32个token并冻结；随后eager / PTE均使用相同历史token做teacher forcing。EOS不截断。
- 每个模型只加载一次，三个用例共享常驻模型，各预热1次、测量3次。加载时间不进入四项指标。Qwen当前产物的历史加载耗时约13.5分钟，是整轮评测的主要等待来源。
- eager使用仓库现有ExportWrapper + 自定义静态Cache，逐步执行；不是旧MMLU的use_cache=False，也不是独立于该仓库实现的正确性oracle。
- 两模型串行运行，测量时保持NPU独占，避免同时导出/编译或其他算力任务。

## 指标边界

计时从请求状态初始化开始，包含mask构造、H2D/D2H、prefill、decode及每步greedy选词，到最后一个token可用结束。tokenization和输入文件读取提前完成。PTE runtime同步复制输出到host，eager `.cpu()`后进行选词。

每步计时结束后才复制logits快照并重新启动下一段计时。总延迟是各段耗时之和，排除诊断复制时间；同时保留 `wall_with_capture_ms` 原始墙钟值。计时是固定轨迹下的模型推理端到端耗时，不是包含请求排队、tokenization、网络、文本渲染的服务延迟。快照仍可能扰动CPU缓存，所有阶段使用同样方式测量。

- TTFT：首个输出可用耗时。
- TPOT：后续31步耗时和 / 31。
- 总延迟：32步测量区间之和。
- 输出吞吐：32 / 总延迟秒。

记录三次各自的四项指标，表中分别取中位数，因此四项中位数之间可能存在微小公式差异。

## 数值比较

保存每次每步最终完整词表logits（FP32文件，每行一个步骤，step=0是prefill），计算与初始eager/PTE的逐步余弦相似度。报告最差一次运行的平均/最低余弦、最差步骤、top-1一致率，原始JSON保留逐步结果。检查非有限数和零范数。输入、容量、权重/tokenizer变化时拒绝不兼容比较。

初始记录不假定eager/PTE完全相等。后续WARN容差为 `max(1e-5, 5 * 初始PTE重复余弦误差)`；若相对初始PTE越界、比初始PTE相对eager的最低余弦再下降超过容差，或top-1一致率低于基线自身重复一致率，则告警。这个阈值是数值漂移检查的初始规则，不代表任务准确率保证；模型运行错误、缺失结果、非有限输出则直接失败并不写成功阶段报告。

## 结果

每模型目录包含 `metadata.json`（提交、diff、软件版本、设备、权重/PTE/runner SHA256）、`config.json`、冻结输入、每后端每用例的三次原始logits与计时、`summary.json`。基线始终保持不变；优化阶段使用新目录。运行命令和完整runner日志也保留。
