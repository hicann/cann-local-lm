# 新手指南 — 读懂 Sampler SPEC

> 配套文档：`SPEC.md`（Sampler 设计规格书，v0.10，**Sampler-only**）。本文是它的"白话注解版"——面向**完全没接触过大模型推理框架**的读者。
> 读法建议：先读本文建立直觉，再回去看 SPEC.md，会有"原来如此"的感觉。
> 维护规则：SPEC.md 是契约真源，本文是讲解。两者冲突时以 SPEC.md 为准；SPEC 改了本文也要跟着改。
> **v0.10 边界**：本文只讲 **Sampler（Layer 3 采样组件，C++ 实现，静态方法）**。runner / backend / registry / CLI / ModelExecutor 归他人，**本文不讲解它们**，只在"Sampler 与它们的边界"处提一句。配套验收是 `cann_lm/engine/core/sample/test/sampler_ut.cc`（100 条断言全 GREEN）。

---

## 0. 这份指南能帮你什么

你拿到一个任务：做一个 **Sampler**——给它模型算出的分数（logits），它按你的采样旋钮选出下一个字，并交出这张选择背后的概率分布。

但你连这些词都是第一次见。这份指南分三步带你：

1. **建立直觉**：一次"大模型推理"到底发生了什么、Sampler 在哪一步（第 1-2 章）
2. **名词扫盲**：Sampler 相关的每个专业词都讲清楚（第 3-4 章）
3. **逐节解读**：SPEC 每一节为什么这么设计（第 5-9 章）

---

## 1. 先建立直觉：大模型推理在干什么

### 1.1 一个类比：超级补全器

把大模型想象成一个**极其聪明的"下一字预测器"**。你给它一段话，它一个字一个字地猜"下一个最可能是什么字"，猜完再猜下一个，循环往复，直到它觉得"说完了"或者达到你限定的长度。

- 你给的起始段话 = **prompt**（提示词）
- 它一个一个往外蹦的字 = **生成的 token**
- "觉得说完了"的信号 = 遇到 **eos**（end of sentence，句末标记）
- 你限定的最大蹦字数 = **max_new_tokens**

整个"你给 prompt → 模型一个字一个字往外蹦 → 拼成一段完整文本"的过程，就叫**推理（inference）**或**生成（generation）**。

### 1.2 一次生成一步的微观过程

假设 prompt = "你好"。每往前蹦一个字，内部其实是这样转的：

```
第1步：喂 "你好"     → 模型前向算出"下一字的分数(logits)" → 挑出 "，" → 拼成 "你好，"
第2步：只喂 "，"(新字) → 算 logits → 挑 "世" → "你好，世"
第3步：只喂 "世"     → 算 logits → 挑 "界" → "你好，世界"
   ...直到挑到 eos 或到 max_new_tokens...
```

注意第 2 步起**只喂上一个新字**，不是把整段重新喂——因为前面算过的中间结果被**缓存**下来了（这叫 KV cache，见 §3.5）。这是推理快的关键。

**Sampler 就是上面箭头里"挑出哪个字"那一步**：拿到这一步的 logits，按你的旋钮挑一个 token 返回。模型前向（算 logits）、循环（一步一步蹦）、停止判断（到没到上限/eos）、最后 decode（数字翻回字）——这些**都不归 Sampler**，归上层。

### 1.3 "Sampler" 是什么

在 LLM 圈，**sample** 主要指**采样**：从概率分布里"抽"一个字（而不是总选概率最高的那个），让输出有随机性、更像人话。

我们这个组件叫 **Sampler**，就是那个"采样器"：给分数、按旋钮、还一个字 + 一张概率表。它不循环、不判停、不前向——调一次、还一次。

---

## 2. Sampler 在整条推理里的位置

你（或上层 runner）跑一次推理，背后是这样一条线；**加粗**的是 Sampler 的地盘，其余归他人：

```
prompt
   │
   ▼  (encode/tokenize — 他人)
input_ids
   │
   ▼  ┌──────────────── 循环（上层 runner 驱动，他人）────────────────┐
   │  │ ① 模型/executor 前向（产出 logits，已含 softcap）— 他人        │
   │  │ ② 【Sampler】Sampler::Sample(logits, generated_ids, cfg) → token + probs│ ◀ 你负责
   │  │ ③ token 拼回序列；上层 Context 判停（eos/到上限）— 他人        │
   │  └─────────────────────────────────────────────────────────────┘
   │  （probs 留给上层做投机采样 — 他人）
   ▼  (decode — 他人)
文本 + 统计
```

记住：**Sampler 只管"② 这一步挑哪个字 + 出概率分布"**。① 算分数、③ 判停、循环、encode/decode、计时——都是别人的活，Sampler 不碰。SPEC §6/§7 把这条边界写得很死。

---

## 3. 名词扫盲（Sampler 相关）

> 这一章是"词典"。每个词给出：是什么、在 SPEC 哪出现、为什么需要它。

### 3.1 token / logits / softmax / probs

- **token**：大模型不认汉字，只认数字。把话切成小段、每段给个编号，这些编号就是 token。
- **logits**：模型每一步给词表里**每个 token**打的"分数"，**未归一化**——可以是任意大小、有正有负。还看不出概率。这是 Sampler 的主输入。
- **softmax / 归一化**：把 logits 转成"加起来 = 1"的概率分布。比如三个字 logits `[2, 1, 0]` → 概率约 `[0.67, 0.24, 0.09]`。
- **概率分布（probs）**：每个字被选中的可能性，所有可能性加起来 = 1。Sampler 的副产出就是这个分布（见 §9.5 投机采样为什么要它）。

### 3.2 采样旋钮（temperature / top_k / top_p / repetition_penalty / do_sample / seed）

模型每步算出 logits。怎么从里面选字，由这些旋钮控制（SPEC §8.1 `SamplerConfig`）：

| 旋钮 | 作用 | 类比 |
|---|---|---|
| **do_sample** | True=从分布里抽样（有随机性）；False=永远选最高（贪心） | 抽签 vs 状元 |
| **temperature** | >1 更随机发散；<1 更保守；=1 不变。越低越"稳" | 调"放飞"程度 |
| **top_k** | 只在概率最高的 k 个里选，其余砍掉 | 只在前 k 名里抽 |
| **top_p**（核采样） | 只在累计概率达到 p 的那几个里选 | 只在"够大众"里抽 |
| **repetition_penalty** | 对**已经生成过的字**降权，避免反复说同一字；>1 降权、1.0 关闭 | "刚说过的话别重复" |
| **seed** | 给定则每次采样可复现（同输入+同 seed→同结果）；None 走默认随机 | 随机种子 |

这六个旋钮就是 `SamplerConfig` 的全部字段。**不含 `max_new_tokens`**——"到没到上限"是停止判断，归上层 Context，不是采样决策。

### 3.3 generated_ids（Repetition Penalty 要看历史）

`repetition_penalty` 这一步需要知道"到目前生成过哪些 token"才能对它们降权。所以 Sampler 每次调用要额外接收 **`generated_ids`**：到当前为止已生成的 token 序列（首步为空）。

**关键**：这个序列**由调用方（上层 runner/Context）按次传入**，Sampler 用完就忘——**Sampler 自己不跨调用存任何东西**（实例不攒历史）。所以"需要 generated_ids"和"实例不跨调用存状态"不矛盾。SPEC §7 把这点写得很清楚。

### 3.4 multinomial / argmax

- **argmax**：永远选分数/概率最高的那个（贪心）。`do_sample=False` 时用这个。
- **multinomial**：按概率分布"抽签"——概率高的字更容易被抽中，但低概率的也有机会。`do_sample=True` 时用这个。

### 3.5 不归 Sampler 的词（认个脸就行）

这些词你会在 SPEC/代码里撞见，但**不是 Sampler 的职责**，简单知道它们存在即可：

- **tokenizer / encode / decode**：文字↔token id 的翻译官。Sampler 只收 logits、只还 token id，不碰翻译。
- **special tokens（bos/eos/pad）**：控制符标记。Sampler 不管它们（停止判断归 Context）。
- **KV cache**：模型前向缓存历史中间结果、让后续步只喂一个字。是模型内部的事，Sampler 不碰。
- **softcapping**：把注意力/logits 分数"压"一下防过大（gemma4 在 forward 的 lm_head 后做，`modeling:549-552`）。**Sampler 吃到的 logits 已被 softcap 过，不重复做。**
- **safetensors / 权重加载**：训练好的模型存在文件里，加载是 backend 的事，与 Sampler 无关。
- **backend / runner / registry / CLI**：上层模块，归他人。Sampler 只暴露一个 `Sampler::Sample()`，被它们调用。

---

## 4. 两模型后处理实测（对 Sampler 设计的帮助）

SPEC §5 列了本机验证过的事实，其中两条直接帮到 Sampler：

1. **gemma4 的 `model.generate()`（`modeling_gemma4.py:572-627`）没有 repetition_penalty** —— Sampler 为 gemma4 路径**新增**了它原本没有的能力。这也是为什么验收 AC7 写的是"rep=1.0 时等价 + >1.0 时降权生效"，而不是"与 gemma4 完全一致"（因为 gemma4 压根没这功能）。
2. **softcapping 在 gemma4 `forward` 的 lm_head 内部做**（`modeling:549-552`：`logits/softcap → tanh → *softcap`）—— Sampler 吃到的 logits 已含 softcap，**不重复做**。SPEC §7 职责边界专门注明。

> （qwen3.5 用 transformers GenerationMixin、import 策略、两脚本计时不一致等，都是 runner/backend 的事，已出范围，这里不展开。）

---

## 5. SPEC 逐节解读（设计思路）

把 SPEC 各节用大白话串一遍。括号里是 SPEC 章节号。

### 5.1 目的 §1 / 范围 §2 / 非目标 §3
- **目的**：Sampler 从 logits 筛选最终 token + 返回概率分布；支持 Temperature/Top-K/Top-P/Repetition Penalty。
- **范围**：做 `SamplerConfig`/`SamplerOutput` struct + `class Sampler`（**静态方法**，无实例）+ 6 步纯函数流水线 + C++ 可运行测试。**不含** runner/backend/registry/CLI/ModelExecutor 的实现——只约定 Sampler 与它们的接口边界。
- **非目标**：不管停止/前向/截断 decode 计时/beam search；不做投机采样（只使能）；不重构既有 backend generate。

**思路**：先把"采样决策"这块独立、可测、可复现立住，其余都划到别人那。

### 5.2 实测前置事实 §5
见第 4 章。**思路**：每一步扎根在"本机验证过的事实"上，不靠假设。

### 5.3 分层定位 §6 + 职责边界 §7
Sampler = Layer 3。logits 从 Layer 2 模型/executor 来，token + probs 回 Layer 4。职责边界一张表：做什么（采样决策）、额外产出（probs）、不管停止/前向/截断 decode 计时/不持有序列。一句话：**实例不跨调用存状态**的"给 logits + 已生成序列、还 token + 分布"的纯决策器。

### 5.4 Sampler 设计 §8（核心）
- **§8.1 对外暴露**：只露一个 `Sampler::Sample(logits, generated_ids, cfg)`（静态）+ 两个 struct。`SamplerConfig` 六旋钮（含 rep penalty、seed）。`SamplerOutput` = token + probs。
- **§8.2 内部 6 步流水线**：① rep_penalty → ② temperature → ③ top_k → ④ top_p → ⑤ normalize → ⑥ pick。每步纯函数、可单测（AC9）。**①必须在 ③④ 前**（rep 与 top_k/top_p 不可交换：rep 改变 top-k 成员/截断集）；①与 ②（temperature）可交换（顺序无影响），放最前是与 HF 一致的约定。
- **§8.3 明确外移**：`no_repeat_ngram`/`min_length` 等仍外移上层（形态多样，会让签名膨胀）；**rep penalty 例外**进 Sampler（形态单一、用户直觉上属采样旋钮）。
- **§8.4 C++ 实现状态**：`sampler.h`/`sampler.cc` **已实现**（6 步静态方法 + `generated_ids` + rep penalty），`test/sampler_ut.cc` 100 条断言全 GREEN。生产版可换 CANN ACL Tensor（接口形态不变）。
- **§8.5 投机采样扩展点**：Sampler 不实现投机，只通过返回 probs 使能它；`sample_with_draft` 预留不首版实现。

### 5.5 调用示例 §9 / 验收 §10 / 落地 §11 / 风险 §12
- **§9 调用示例**：`Sampler::Sample(logits, generated_ids, SamplerConfig{...})` → `out.next_token_id` / `out.probs`。
- **§10 验收**：AC7（可复现 + rep 等价/降权）、AC8（probs 归一化）、AC9（纯函数可单测）。**可运行**：`test/sampler_ut.cc` 覆盖各输入参数组合。
- **§11 落地清单**：`cann_lm/engine/core/sample/sampler.h` + `sampler.cc` + `test/sampler_ut.cc`。
- **§12 风险**：R1 rep penalty 镜像 HF（与 GenerationMixin 路径未对测，属他人）、R2 全 vocab probs 显存。

---

## 6. 关键设计决策的"为什么"（问答式）

**Q1 为什么 Repetition Penalty 进了 Sampler（而不是像别的序列 processor 一样外移）？**
因为它形态单一（对"已出现 token"统一降权）、用户直觉上就属"采样旋钮"（用户明确要求 Temperature/Top-K/Top-P/Repetition Penalty 并列）。别的（`no_repeat_ngram`/`min_length`）形态多样、塞进会让公开 `Sampler::Sample()` 签名随 processor 数膨胀，仍外移上层。代价：Sampler 单次调用需 `generated_ids`，但实例仍不跨调用存状态。见 SPEC §8.3。

**Q2 为什么 rep penalty 排在 temperature/top_k/top_p 之前？**
因为顺序对结果有影响——但**仅当后面跟 top_k/top_p 时**：rep 会改变哪些 token 进入 top-k / 核采样截断集，先选后惩会让被屏蔽 token 逃脱惩罚、改变结果。注意：rep 与 temperature 其实**可交换**（都是逐元素标量乘），所以"rep 在 temperature 前"对结果无影响；真正不可交换的是 rep 对 top_k/top_p。放最前是与 HuggingFace LogitsProcessor 一致的约定。SPEC §8.2 "组合"段讲了。

**Q3 为什么 Sampler 要返回概率分布（probs），不只返回 token？**
因为上层可能做**投机采样**加速：用小模型先猜、大模型验证，验证时要拿大模型的概率分布做"接受/拒绝"。Sampler 不实现投机，但通过返回 probs **使能**它。见 §9.5 / SPEC §8.5。

**Q4 为什么"实例不跨调用存状态"却又要 generated_ids？**
不矛盾。实例**不跨调用保存**任何东西（不攒历史序列/计数/generator）；但**单次调用**需要 generated_ids 来做 rep penalty——这个序列是调用方**按次传进来**的，Sampler 用完就忘。这样既可复现、又不让 Sampler变成有状态大杂烩。见 SPEC §7。

**Q5 为什么 softcapping 不进 Sampler？**
它在模型 `forward` 的 lm_head 内部做（gemma4 `modeling:549-552`），Sampler 只调一次 `sample`、拿到的 logits 已被 softcap 过，碰不到也不该重复。见 SPEC §5/§7。

---

## 7. 一图流：Sampler 内部

```
你给的输入                              Sampler 内部（6 步纯函数）        输出
─────────────────────────────────────────────────────────────────────────────
logits ─────────────┐                ① ApplyRepetitionPenalty         ───▶ next_token_id
[1, vocab]           │   ①在②③④前      ② ApplyTemperature              (选中的字, int)
(模型/executor 前向产出)│                  ③ ApplyTopK
generated_ids ──────┤  (rep penalty用)  ④ ApplyTopP                  ───▶ probs
[seq] (到目前生成)    │                  ⑤ Normalize (softmax, 和≈1)       [1, vocab]
SamplerConfig ──────┘  ⑥ Pick (multinomial/argmax)                     (归一化分布)
·temperature/do_sample
·top_k/top_p
·repetition_penalty
·seed

落地文件：cann_lm/engine/core/sample/sampler.h/.cc    验收：cann_lm/engine/core/sample/test/sampler_ut.cc
```

---

## 8. 新手学习路径建议

1. **读 SPEC §7/§8**：先看懂职责边界 + 6 步流水线。
2. **读 `test/sampler_ut.cc`**：每个测试就是一条"某输入→某期望"的契约。期望值用测试内独立 softmax 直算，不依赖实现——读懂它就懂了 Sampler 该怎么动。
3. **实现 `sampler.cc`**：把 6 个静态纯函数 + `Sampler::Sample` 写到让 `test/sampler_ut.cc` 全 GREEN。先写纯函数（AC9，好测），再串 `Sample`。
4. **跑 `g++ -std=c++17 sampler.cc test/sampler_ut.cc -o test/sampler_ut && ./test/sampler_ut`**：全 PASS 即 AC7/AC8/AC9 达成。
5. **回头看 §8.4**：C++ 已实现（`sampler.cc`，GREEN）；生产版可换 CANN ACL Tensor（接口形态不变）。

---

## 9. Sampler 内部解剖（对外暴露 vs 内部能力）

> 对应 SPEC §8.1/§8.2/§8.3。一句话原则：**对外只露一个 `Sampler::Sample(logits, generated_ids, cfg)`，内部分成一条可单测的纯函数流水线**。实例不跨调用存状态，但单次调用需要 `generated_ids`（供 rep penalty）。

### 9.1 对外暴露的（别人能调的）

Sampler 对外只露三样东西——一个类（静态方法，无实例）、一个静态方法、两个 struct：

| 暴露项 | 形式 | 干什么 |
|--------|------|--------|
| `Sampler`（静态方法，无实例） | 类 | 直接 `Sampler::Sample(...)` 调用，无需实例化/继承 |
| `Sampler::Sample(logits, generated_ids, cfg)` | **唯一公开方法** | 给 logits + 已生成序列 + 配置，还 token + 分布 |
| `SamplerConfig` | 入参 dataclass | 采样旋钮 |
| `SamplerOutput` | 返回 dataclass | token + 概率分布 |

**入参 `SamplerConfig` 装的是"采样旋钮"**：`temperature`、`do_sample`、`top_p`、`top_k`、`repetition_penalty`、`seed`。⚠ 它**不含** `max_new_tokens`——停止计数归上层 Context。`generated_ids` 单独作方法入参传（不进 SamplerConfig，因为它每步都在变，是序列不是旋钮）。

**返回 `SamplerOutput`**：`next_token_id`（选中的字）+ `probs`（归一化概率分布）。

> 记住：**对外就是"给 logits + 已生成序列 + 旋钮 → 拿 token 和分布"**，仅此一个入口。Sampler 内部怎么实现的，调用方不该关心。

### 9.2 内部能力（私有的 6 步流水线）

`Sampler::Sample` 内部把"采样"拆成 6 个**静态纯函数步骤**，按顺序串起来。每步都是纯函数（输入→输出、不记东西），能单独单测。

```
 logits + generated_ids
   │
   ▼ ① Sampler::ApplyRepetitionPenalty(logits, generated_ids, p)  对已生成字降权（p>1；镜像 HF）
   │
   ▼ ② Sampler::ApplyTemperature(logits, t)     logits/温度（越小越稳，→0 退贪心）
   │
   ▼ ③ Sampler::ApplyTopK(logits, k)            只留最高 k 个，其余 -inf
   │
   ▼ ④ Sampler::ApplyTopP(logits, p)            核采样：累计概率达 p 后截尾
   │
   ▼ ⑤ Sampler::Normalize(logits) → probs        softmax 归一化（和≈1）
   │
   ▼ ⑥ Sampler::Pick(probs, do_sample, seed)     do_sample=True→multinomial 抽；False→argmax
   │
   ▼
 next_token_id + probs  →  组装成 SamplerOutput 返回
```

| 步 | 内部方法 | 干什么 | 失败保护 |
|----|----------|--------|---------|
| ① | `ApplyRepetitionPenalty` | 对已生成过的字降权（`score<0` 时 `*p`、`≥0` 时 `/p`） | p/None/1.0 或 generated_ids 空 → 跳过 |
| ② | `ApplyTemperature` | logits 除以温度 | t≤0 钳到极小正值 |
| ③ | `ApplyTopK` | 只留最高 k 个 | k=None 或 ≥词表 → 跳过 |
| ④ | `ApplyTopP` | 累计概率达 p 截尾 | p=None → 跳过 |
| ⑤ | `Normalize` | softmax 归一化 | 减最大值保数值稳定 |
| ⑥ | `Pick` | 抽样 or 贪心选字 | seed 给定则 per-call 建 seeded Generator |

**① 必须在 ③④ 之前**：rep penalty 与 top_k/top_p **不可交换**——rep 会改变哪些 token 进入 top-k / 核采样截断集，先选后惩会让被屏蔽 token 逃脱惩罚、改变结果（`test_rep_before_topk_order` 验证）。①与 ②（temperature）则**可交换**（均为逐元素标量乘，顺序无影响），放最前只是与 HF LogitsProcessor 一致的约定。贪心路径（`do_sample=False`）下 ③④ 仍可应用（只影响 probs 不改 argmax），但 ① 会改变 argmax 结果——这正是 rep penalty 的目的（避免反复输出同一 token）。

**为什么拆成纯函数？** 为了**可单测、可复用、可替换**（AC9）。比如你想换个更激进的采样器，只换内部步骤组合，对外那个 `Sampler::Sample()` 不变——调用方无感。

### 9.3 明确不进 Sampler 的（外移到别处）

这是边界纪律，避免 Sampler 变成"什么都记一点"的大杂烩：

| 能力 | 为什么不进 | 归谁 |
|------|-----------|------|
| `no_repeat_ngram` / `min_length` 等 | 依赖序列结构/位置且形态各异，塞进会让公开 `Sampler::Sample()` 签名随 processor 数膨胀 | 上层：调 Sampler 前对 logits 预处理 |
| 停止判断（eos/到上限） | 是停止决策非采样决策 | 上层 Context |
| beam search | 多分支编排非单步采样 | 上层（可多次调 `sample`） |
| 前向算 logits | 模型计算 | Layer 2 模型/executor |

> **`repetition_penalty` 为何例外**：它同样依赖序列历史，但形态单一（对"已出现 token"统一降权）、用户直觉上就属"采样旋钮"，故作为一等字段进 `SamplerConfig`，序列由调用方经 `generated_ids` 传入、Sampler 内部步骤①完成。其余序列相关 processor 仍外移——它们的多样性与 Sampler"公开契约稳定"的目标冲突。

**一句话**：Sampler 只做"给定 logits + 已生成序列、按旋钮选一个字 + 出分布"；实例**不跨调用存任何状态**（序列由调用方按次传入）。需要**多分支**（beam）或**非统一形态序列 processor**的，不在它内部。

### 9.4 可复现与"实例不跨调用存状态"怎么并存？

采样（`multinomial`）本身有随机性。要可复现（AC7：同 logits+同 generated_ids+同 seed → 同结果），又不能让 Sampler"跨调用存状态"。解法：

- `SamplerConfig.seed` 给定 → 每次 `sample` 调用**临时**建一个 seeded 随机发生器，用完即弃。
- `seed=None` → 走默认全局 RNG（每次随机）。

这样 Sampler 实例**不跨调用保存**任何东西（实例无状态），但单次调用内可复现。两全。

### 9.5 投机采样为什么需要概率分布

这是上层的一种加速技术，**Sampler 不实现它，但要"使能"它**。原理（白话）：

1. **draft 模型**（小、快）先猜几个字（候选 token）。
2. **target 模型**（大、准）算这些位置的真实分布。
3. 对比：draft 猜得"够准"就**接受**（省掉 target 逐步生成的时间）；猜得离谱就**拒绝**并按 target 分布重采。
4. 这个"接受/拒绝"要用 target 的**概率分布**做 `min(1, q_target/q_draft)` 计算。

所以 Sampler **必须把概率分布（probs）返回出来**，上层投机采样才有料可算。这就是 SPEC §8.5 说的"Sampler 通过返回分布使能投机采样"。

> 新手记住到这层就够：**投机采样是上层的加速戏法，Sampler 给它提供"概率分布"这味原料；Sampler 自己不演这出戏。**

### 9.6 C++ 侧长一样

C++ 实现的 `Sampler`（`cann_lm/engine/core/sample/sampler.h`）：对外一个 `Sampler::Sample(logits, generated_ids, cfg)`（静态），内部 6 个静态纯函数镜像那 6 步（含 `ApplyRepetitionPenalty`）。对接关系：Layer 2 模型/executor 产出 logits → Sampler 采样 → 上层投机采样消费 probs / Context 判停。

---

## 10. 术语速查表（Sampler 相关）

| 术语 | 一句话 | SPEC 章节 |
|---|---|---|
| token | 模型读的数字编号 | §3.1 / §4 |
| logits | 模型对词表每个字打的未归一化分数（已含 softcap） | §3.1 / §8.1 |
| 概率分布 probs | 加起来=1 的每个字被选中的可能性 | §3.1 / §8.1 |
| softmax/归一化 | logits → 概率分布的转换 | §3.1 / §8.2 |
| temperature | 调放飞程度；越低越稳，→0 退贪心 | §3.2 / §8.2 |
| top_k | 只留概率最高 k 个 | §3.2 / §8.2 |
| top_p | 核采样，累计概率达 p 截尾 | §3.2 / §8.2 |
| repetition_penalty | 对已生成字降权，避免重复；v0.5 进 Sampler（gemma4 原本无此能力） | §3.2 / §8.2 |
| do_sample | True 抽样 / False 贪心 | §3.2 / §8.2 |
| multinomial / argmax | 按概率抽签 / 永远选最高（抽样 vs 贪心） | §3.4 / §8.2 |
| generated_ids | 到目前已生成的字，经 Sampler 入参传给 rep penalty；Sampler 不跨调用存 | §3.3 / §7 |
| seed / Generator | 随机种子；Sampler per-call 临时建，保可复现 | §3.2 / §8.1 / §9.4 |
| Sampler | Layer 3 采样器：logits + generated_ids → token + probs | §1 / §8 |
| SamplerConfig | 入参 dataclass，六采样旋钮 | §8.1 |
| SamplerOutput | 返回 dataclass：next_token_id + probs | §8.1 |
| Context | 上层组件，管停止判断（eos+计数）；不归 Sampler | §7 |
| 投机采样 | 上层加速技术：draft 猜 + target 分布验证；Sampler 只供 probs | §9.5 / §8.5 |
| 分层架构 | Sampler 在 Layer 3；logits 从 Layer 2 来，token+probs 回 Layer 4 | §6 |
| 纯函数 | 输入→输出、无副作用、不记状态；Sampler 内部步骤的设计要求（AC9） | §8.2 |
| softcapping | 注意力/logits 软截断，模型 forward 内部做（gemma4 `modeling:549-552`），Sampler 不重复 | §5 / §7 |

> 表里没有的词（tokenizer、KV cache、eos/pad、safetensors、backend、runner、registry、CLI、ModelExecutor 等）**不归 Sampler**——它们存在、在别处讲，本文不展开。

---

*本文为 SPEC.md（v0.10，Sampler-only）的新手注解。SPEC 变更时本文同步更新；冲突以 SPEC.md 为准。*
