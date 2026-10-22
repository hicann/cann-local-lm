# CANN-LM Sampler — 设计规格书 (SPEC)

> 📖 **新手请先读 [`GUIDE.md`](./GUIDE.md)**：从零讲起的概念扫盲 + 本 SPEC 逐节白话解读。
> 状态：**草案 (Draft)**。
> 维护者：本 SPEC 只负责 **Sampler（Layer 3 采样组件）**。runner / backend / registry / CLI / ModelExecutor 等模块归他人，本 SPEC **只约定 Sampler 与它们的接口边界，不设计它们**。
> 配套验收：`cann_lm/engine/core/sample/test/sampler_ut.cc`（可运行 C++ 测试套件，100 条断言，覆盖各输入参数→输出）。

---

## 1. 目的 (Purpose)

**Sampler 负责从模型/executor 输出的 Logits 中筛选最终的 Token，并返回采样用的概率分布。**

支持配置（即"采样旋钮"）：**Temperature、Top-K、Top-P、Repetition Penalty**（外加 `do_sample` 贪心/采样开关、`seed` 可复现）。

Sampler 是一个**实例不跨调用存状态**的决策盒子：给它 logits + 到目前生成过的 token + 旋钮，它还一个 token + 一张概率表。不循环、不判停、不前向。

## 2. 范围 (Scope)

**包含 (In scope)**
- `SamplerConfig` / `SamplerOutput` struct、`class Sampler`（静态方法，无实例/OOP）。
- 6 步纯函数采样流水线（`ApplyRepetitionPenalty → ApplyTemperature → ApplyTopK → ApplyTopP → Normalize → Pick`）。
- 可运行的 C++ 验收测试套件（`test/sampler_ut.cc`，各输入参数组合→期望输出）。
- C++ 实现（`sampler.h`/`sampler.cc`，无外部依赖，已 GREEN；生产版可换 CANN ACL Tensor，接口形态不变）。

**不包含 (Out of scope — 归他人)**
- runner 编排、backend 适配（gemma4/qwen3.5）、registry 注册表、CLI、ModelExecutor 的 `forward_step` 实现——本 SPEC 只约定 Sampler 的**输入来自哪、输出去哪**，不设计这些模块。
- 投机采样（speculative sampling）的逻辑实现——Sampler 只通过返回概率分布**使能**它。
- 重构既有 backend 的 `model.generate()`——Sampler 独立存在，是否被某 backend 调用由该 backend 的维护者决定。

## 3. 非目标 (Non-goals)
- 不管停止判断（eos / max_new_tokens 计数）——归上层 Context。
- 不管前向计算（产出 logits）——归模型/executor。
- 不管截断 prompt / decode / 计时——归上层 runner。
- 不做多分支编排（beam search）——归上层（可多次调 `sample`）。
- 不追求性能最优；首版以"可测、可复现、契约清晰"为第一目标。

## 4. 术语 (Glossary)

| 术语 | 含义 |
|---|---|
| logits | 模型单步前向给词表每个 token 打的**未归一化**分数（有正有负）。Sampler 的主输入。 |
| generated_ids | 到当前为止已生成的 token 序列（1d long Tensor），供 Repetition Penalty 用；首步为空。**由调用方按次传入，Sampler 不持有**。 |
| token / next_token_id | Sampler 选出的下一个 token 编号。 |
| probs | 归一化概率分布（沿 vocab 维和≈1）；Sampler 的副产出，供上层投机采样做 rejection。 |
| softmax / 归一化 | logits → 概率分布（和≈1）的转换。 |
| Repetition Penalty | 对已生成过的 token 降权，避免反复输出同一 token；>1 降权、1.0 关闭。 |
| seed / Generator | 随机种子；给定则 Sampler per-call 建临时 seeded Generator，保证可复现。 |

## 5. 实测前置事实 (Verified facts)

> 方案成立的基础，已在本机验证：

1. gemma4 的 `model.generate()`（`cann_lm/models/gemma4/modeling_gemma4.py:572-627`）**没有 repetition_penalty** —— Sampler 为 gemma4 路径**新增**了此能力。
2. softcapping 在 gemma4 `forward` 的 `lm_head` 之后内部完成（`modeling_gemma4.py:549-552`：`logits/softcap → tanh → *softcap`）—— Sampler 吃到的 logits **已含 softcap**，不重复做。

## 6. 分层定位 (Layered architecture)

Sampler 处于 6 层架构的 **Layer 3（核心层）**：

| 层 | 名称 | 和 Sampler 的关系 |
|----|------|------------------|
| 4 | Pipeline | 上游（导演）：传 `SamplerConfig` + `generated_ids`、驱动循环、收 token 拼序列并判停；下游：消费 `probs` 做投机采样 |
| 3 | 核心层 | **Sampler 本体**（logits → token + 分布） |
| 2 | Backend | 上游：模型/executor 单步前向产出 logits（已含 softcap） |

> 即：logits 从 Layer 2 来，token + probs 回 Layer 4。Sampler 全程只管"这一步选哪个字"，别的不沾。

## 7. 职责边界

| 维度 | 决断 | 依据 |
|------|------|------|
| 做什么 | **纯采样决策**：logits → next token id（含 rep penalty，需 `generated_ids`） | 用户定义 |
| 额外产出 | **返回概率分布**（`probs`） | 供 Layer 4 投机采样 rejection |
| **不管停止** | eos/max_new_tokens 计数归上层 Context | 采到 eos 只如实返回 eos id，由 Context 判停 |
| **不管前向** | logits 由模型/executor 产出 | 不重复模型计算 |
| **不管截断/decode/计时** | 归上层 runner | — |
| **不持有序列** | `generated_ids` 由调用方按次传入，Sampler 不跨调用存 | 实例可持 `SamplerConfig`，但不存生成序列/计数 |

> 即：Sampler 是一个**实例不跨调用存状态**的"给 logits + 已生成序列、还 token + 分布"的纯决策器；rep penalty 所需的已生成序列由调用方按次传入，不由 Sampler 持有。

## 8. Sampler 组件设计

> 设计原则：**对外只暴露一个 `Sampler::Sample(SamplerInputs, SamplerConfig)`，内部分成一条可单测的纯函数流水线**。实例不跨调用存状态、不持有生成序列、不判停；rep penalty 所需的已生成序列由调用方按次（经 `SamplerInputs`）传入。

### 8.1 对外暴露的接口（public contract）

> **v0.7：实现语言改为 C++**，且因 Sampler 无状态，对外是**静态方法**（C++ 风格，不实例化、不要 OOP 多态基类）。Python 原型（`sampler.py`/`test_sampler.py`）已废弃删除，C++（`sampler.h`/`sampler.cc`/`test_sampler.cc`）取代之。

Sampler 对外只露：四个 struct + 一个静态方法（+ 6 个静态 helper，公开以便 AC9 单测）。

| 暴露项 | 形式 | 职责 |
|--------|------|------|
| `SamplerConfig` | struct（旋钮入参） | 采样旋钮（温度/top_k/top_p/rep/seed） |
| `SamplerInputs` | struct（数据入参） | 数据打包：logits + generated_ids + precomputed_topk（v0.9） |
| `SamplerOutput` | struct（返回） | 选中 token + 概率分布 |
| `Sampler::Sample(SamplerInputs, SamplerConfig)` | **静态方法**（唯一入口） | 给数据 + 旋钮，还 token + 分布 |
| `Sampler::ApplyRepetitionPenalty / ApplyTemperature / ApplyTopK / ApplyPrecomputedTopK / ApplyTopP / Normalize / Pick` | 静态方法 | 内部纯函数，公开以便独立单测（AC9） |

```cpp
// cann_lm/engine/core/sample/sampler.h
namespace cann_lm::sampler {
struct SamplerConfig {
    float temperature = 1.0f;
    bool do_sample = false;
    std::optional<float> top_p;
    std::optional<int> top_k;
    std::optional<float> repetition_penalty;  // >1 降权、1.0 关闭、<1 升权
    std::optional<uint64_t> seed;             // 给定则 per-call 建 seeded RNG（可复现）
};
// NPU 预算 top-k（数据输入，非旋钮）
struct PrecomputedTopK {
    std::vector<int32_t> token_ids;  // NPU 已算出的 top-k token 下标集合
};
// 数据输入打包（旋钮在 SamplerConfig，数据在此——分离）。logits/generated_ids 按值持有；
// 调用方可 std::move 进来，Sample 按值取 in 后 std::move(in.logits) 走零拷贝路径。
struct SamplerInputs {
    std::vector<float> logits;
    std::vector<int32_t> generated_ids;
    std::optional<PrecomputedTopK> precomputed_topk;
};
struct SamplerOutput {
    int32_t next_token_id = 0;  // token id 用 int32_t（词表 <2^31 够用，省内存）
    std::vector<float> probs;   // [vocab]，和≈1
};
class Sampler {
public:
    // 无状态：静态方法，无需实例化。数据打包进 SamplerInputs（按值取，可 move in.logits），旋钮在 cfg。
    // logits 已含 softcap；generated_ids 首步可为空 {}；precomputed_topk 给了则覆盖 cfg.top_k。
    static SamplerOutput Sample(SamplerInputs in, const SamplerConfig& cfg);
    static std::vector<float> ApplyRepetitionPenalty(std::vector<float>, const std::vector<int32_t>&, std::optional<float>);
    static std::vector<float> ApplyTemperature(std::vector<float>, float);
    static std::vector<float> ApplyTopK(std::vector<float>, std::optional<int>);
    static std::vector<float> ApplyPrecomputedTopK(std::vector<float>, const std::optional<PrecomputedTopK>&);  // 复用 NPU 预算 top-k
    static std::vector<float> ApplyTopP(std::vector<float>, std::optional<float>);
    static std::vector<float> Normalize(const std::vector<float>&);
    static int32_t Pick(const std::vector<float>&, bool, std::optional<uint64_t>);
};
}
```

- **无状态 → 静态**：`Sample` 是 `static`，调用 `cann_lm::sampler::Sampler::Sample(...)`，**不实例化、不要 ABC/子类**。`generated_ids` 由调用方按次传入（Repetition Penalty 要看"到目前生成了哪些 token"，见 §8.2 步骤①），Sampler 不跨调用持有。
- **batch=1**：`logits`/`probs` 为扁平 `std::vector<float>`（vocab 维），非 `[batch][vocab]`（首版 batch=1，见 Q1）。生产版可改取 CANN ACL Tensor，接口形态不变。

**契约**：
- `probs` 必须归一化（沿 vocab 维和≈1），供 Layer 4 投机采样做 `min(1, q_target/q_draft)` rejection（AC8）。
- 默认返回全 vocab 分布；若实测显存吃紧，可加 `return_top_k` 选项返回截断 + 索引（预留）。
- 同 `logits` + 同 `generated_ids` + 同 `cfg` + 同 `seed` → 同 `next_token_id`（可复现，AC7）。
- 原型用 host `std::vector<float>`（无 device 概念）；生产版取 CANN ACL Tensor 时运算跟随其 device，Sampler 不搬数据。

**设计理路：为什么用 `SamplerConfig` 打包传参（而非每参数一套 set/get）**
- Sampler **无状态**（§7/§8.1 写死"实例不跨调用存状态"）。set/get 会把配置变成可变成员 → 重新引入状态，与"静态方法、不实例化"的设计直接冲突。
- **可复现（AC7）**：cfg 作为参数，每次调用的配置一目了然、可日志可回放；set/get 把配置藏在成员里，要复现得 dump 所有 get()，且易状态残留/遗忘。
- **线程安全**：`Sample(SamplerInputs, const cfg&)` 无共享可变状态，多线程随便调；set/get 共享实例则竞态，需加锁或每线程一实例。
- **维护面**：A = 1 个 Sample + 2 个 struct（SamplerInputs/SamplerConfig）；B = 1 个 Sample + 12 个 set/get + 6 个成员，加字段要三处同步（成员/set/get）。
- **B 唯一优势"配一次调多次"在 A 里同样可得**：`const SamplerConfig preset{...}` 建一次，多次 `Sample(SamplerInputs{...}, preset)`——不可变、显式、无状态残留。
- **B 才合理的场景**：对象本就有状态且长期存活（如持有模型/KV-cache），或需要 setter 强校验。Sampler 不属此类（纯决策函数、无状态），故 B 的优势用不上、劣势全占。
- 结论：**采用 config struct 打包传参**。需要"链式构造"语法（`cfg.temperature(0.7).top_k(40)`）只是 A 的构造语法糖（返回 SamplerConfig），不引入状态，可加但不急。

### 8.2 内部能力（private pipeline）

`Sampler::Sample` 内部按序调用下列**静态纯函数**步骤。每步都是纯函数（输入→输出、无副作用），可独立单测（AC9）。

| 顺序 | 内部方法 | 输入 → 输出 | 作用 | 失败保护 |
|------|----------|-----------|------|---------|
| ① | `Sampler::ApplyRepetitionPenalty(logits, generated_ids, p)` | logits → penalized logits | 对**已生成过的 token** 降权：`score = gather(logits, ids)`；`score<0` 时 `*p`（更负）、`≥0` 时 `/p`（更小），再 `scatter` 回。`p>1` 降权、`1.0` 不变。**镜像 HF `RepetitionPenaltyLogitsProcessor`** | `p`/`None`/`1.0` 或 `generated_ids` 空 → 跳过（原样返回 logits） |
| ② | `Sampler::ApplyTemperature(logits, t)` | logits → scaled logits | `logits / t`；`t` 极小→退贪心 | `t ≤ 0` 钳到极小正值，避免除零 |
| ③ | `Sampler::ApplyTopK(logits, k)` / `Sampler::ApplyPrecomputedTopK(logits, precomputed)` | logits → masked logits | 只留最高 k 个，其余 `-inf`。**二选一**：给了 `precomputed_topk` 则 `ApplyPrecomputedTopK`（复用 NPU 已算的 top-k 集合，不在集合里→`-inf`，覆盖 `cfg.top_k`）；否则 `ApplyTopK`（自己排序找阈值） | `k`/`None` 或 ≥vocab → 跳过；`precomputed` 空/nullopt → no-op |
| ④ | `Sampler::ApplyTopP(logits, p)` | logits → masked logits | 核采样：累计概率达 p 后截断尾部 | `p`/`None` → 跳过；保留最高概率项 |
| ⑤ | `Sampler::Normalize(logits)` | masked logits → probs | `softmax` 归一化得概率分布（和≈1） | 数值稳定：减最大值 |
| ⑥ | `Sampler::Pick(probs, do_sample, seed)` | probs → token_id | `do_sample=True`→`multinomial`（`seed` 给定则 per-call 建 seeded Generator）；`False`→`argmax` | — |

**组合**：`Sampler::Sample()` = ①→②→③→④→⑤→⑥，返回 `SamplerOutput(next_token_id, probs)`。**顺序要求**：①必须在 ③④ **之前**——rep penalty 与 top_k/top_p **不可交换**：rep 会改变哪些 token 进入 top-k / 核采样截断集，先选后惩会让被屏蔽的 token 逃脱惩罚、改变结果（`test_rep_before_topk_order` 验证）。①与 ②（temperature）则**可交换**（均为逐元素标量乘，顺序对结果无影响），放最前只是与 HF LogitsProcessor 一致的约定。贪心路径（`do_sample=False`）下 ③/④ 仍可应用（影响 `probs` 但不改变 argmax 结果）；①（rep penalty）在贪心下**会**改变 argmax 结果——这正是 rep penalty 的目的（避免反复输出同一 token）。

> 设计取舍：把每步做成纯函数，是为了让采样逻辑**可单测、可复用、可替换**。未来若要加 `TopKTopPMinMaxSampler` 等变体，只换内部步骤组合，公开 `Sampler::Sample()` 契约不变。

### 8.3 明确不进 Sampler 内部（外移到别处）

下列能力**不属于 Sampler**，由上层负责，避免 Sampler 变成有状态"大杂烩"：

| 能力 | 为什么不进 | 归谁 |
|------|-------------------|------|
| `no_repeat_ngram` / `min_length` / `no_bad_words` 等 | **依赖序列结构/位置**且形态各异，塞进 Sampler 会让公开 `Sampler::Sample()` 签名随 processor 数膨胀 | 上层：在调 Sampler 前对 logits 预处理 |
| 停止判断（eos 命中 / `max_new_tokens` 计数） | 是停止决策，非采样决策 | 上层 Context |
| beam search | 是多分支编排，非单步采样 | 上层编排（可多次调 `sample`） |
| 前向计算（产出 logits） | 模型计算 | Layer 2 模型/executor |

> **`repetition_penalty` 为何例外**：它同样依赖序列历史，但形态单一（对"已出现 token"统一降权）、用户直觉上就属"采样旋钮"，故作为一等字段进 `SamplerConfig`，由调用方传入 `generated_ids`、Sampler 内部步骤①完成。其余序列相关 processor 仍外移——它们的多样性与 Sampler"公开契约稳定"的目标冲突。

**边界一句话**：Sampler 只做"给定 logits + 已生成序列、按旋钮选一个字 + 出分布"；实例**不跨调用存任何状态**（序列由调用方按次传入）。需要**多分支**（beam）或**非统一形态序列 processor**的事不在它内部。

### 8.4 C++ 实现状态 + 生产版变体

**已实现**：`sampler.h`/`sampler.cc`（接口见 §8.1，**静态方法** + `std::vector<float>`，无外部依赖）；`test/sampler_ut.cc` 100 条断言全 GREEN。无 OOP 实例/ABC——`Sample` 与 6 个 helper 全是 `static`。

**生产版变体**（待 CANN 引擎阶段）：把 `std::vector<float>` 换成 CANN ACL `Tensor`/`Span<float>`，**接口形态不变**（仍静态、同签名形状）：

```cpp
// 生产版示意（ACL Tensor 入参）；签名形状与 §8.1 原型一致
class Sampler {
public:
    static SamplerOutput Sample(const Tensor& logits /*末位 [vocab]，已含 softcap*/,
                               const Span<int64_t>& generated_ids,
                               const SamplerConfig& cfg);
    static Tensor ApplyRepetitionPenalty(Tensor, const Span<int64_t>&, std::optional<float>);
    static Tensor ApplyTemperature(Tensor, float);
    static Tensor ApplyTopK(Tensor, std::optional<int>);
    static Tensor ApplyTopP(Tensor, std::optional<float>);
    static Tensor Normalize(const Tensor&);           // softmax（减最大值）
    static int64_t Pick(const Tensor& probs, bool do_sample, std::optional<uint64_t> seed);
};
```

### 8.5 投机采样扩展点

上层 Pipeline 做投机采样：draft 模型产出候选 token → target 模型产出 target_logits → `Sampler::Sample(target_logits, gen_ids, cfg)` 得 target 分布 → rejection sampling 接受/拒绝候选。

**Sampler 不实现投机逻辑**（那是上层职责），只通过**返回概率分布**使能它。未来若需"带 draft 的 rejection 采样"原生支持，可在 Sampler 加 `SampleWithDraft(logits, draft_tokens, gen_ids, cfg)` 变体——**预留，不首版实现**。

## 9. 调用示例（上层如何用 Sampler）

> 本节只示意调用形态；runner/backend/Context 的实现不在本 SPEC 范围。

```cpp
#include "sampler.h"
using namespace cann_lm::sampler;

// 数据打包进 SamplerInputs（logits: [vocab]，由模型/executor 单步前向产出，已含 softcap；
//   generated_ids: 到目前生成的 token，首步为空 {}；precomputed_topk: NPU 预算 top-k，可选）
SamplerInputs in{.logits = std::move(logits), .generated_ids = std::move(generated_ids)};
SamplerOutput out = Sampler::Sample(
    std::move(in),
    SamplerConfig{
        .temperature = 0.7f,
        .do_sample = true,
        .top_p = 0.9f,
        .top_k = 40,
        .repetition_penalty = 1.1f,
        .seed = 0ull,
    });
int32_t next_token = out.next_token_id;   // 喂回模型继续下一步前向；由上层 Context 判停
std::vector<float> probs = out.probs;     // [vocab]，供上层投机采样做 rejection
```

## 10. 验收标准 (Acceptance criteria — 可运行：`test/sampler_ut.cc`)

> AC1-AC6（runner/backend/CLI 相关）已随范围收窄移出本 SPEC。下列为 Sampler 验收，均由 `test/sampler_ut.cc` 覆盖（各输入参数组合→期望输出，期望值用测试内独立 softmax 直算，不依赖被测实现）。编译运行：`g++ -std=c++17 -O2 sampler.cc test/sampler_ut.cc -o test/sampler_ut && ./test/sampler_ut`。

| # | 场景 | 预期 |
|---|---|---|
| AC7 | `Sampler::Sample(logits, generated_ids, cfg)` 对同一 logits + 同一 generated_ids + 固定 seed 采样，结果确定可复现；且 `repetition_penalty=1.0` 时与 gemma4 现有 `model.generate` 内联采样（top_k/top_p/multinomial）等价；`repetition_penalty>1.0` 时对已生成 token 降权生效（gemma4 原本无此能力，属 sample 新增） | 采样逻辑可复现、rep penalty 正确 |
| AC8 | `Sample` 返回的 `probs` 归一化（和≈1）、大小同 logits、非负；可被上层投机采样消费 | 概率分布契约成立 |
| AC9 | 各 `ApplyRepetitionPenalty`/`ApplyTemperature`/`ApplyTopK`/`ApplyTopP`/`Normalize`/`Pick` 为纯函数，可脱离 `Sample` 单独单测且结果确定 | 内部流水线可单测、可替换 |

`test/sampler_ut.cc` 另覆盖：temperature 不改 argmax + t≤0 钳制、top_k 屏蔽非 top-2、top_p 极小只留 argmax、rep penalty argmax 翻转（正/负 logit 两式）+ rep<1 升权 + rep 在 top_k 前顺序、seed 可复现、组合（全旋钮）无 nan/inf。当前 100 条断言全 GREEN。

## 11. 落地清单 (Deliverables)

```
cann_lm/engine/core/sample/
  sampler.h             # C++ 接口：SamplerConfig/SamplerOutput struct + class Sampler（静态方法）— §8
  sampler.cc            # 6 步流水线实现（std::vector<float>，无外部依赖）— §8.2
  test/sampler_ut.cc       # 可运行验收（100 条断言，自定义 runner，不依赖 gtest）— §10
  SPEC.md               # 本文件
  GUIDE.md              # 新手注解
```

> 编译运行：`g++ -std=c++17 -O2 -Wall cann_lm/engine/core/sample/sampler.cc cann_lm/engine/core/sample/test/sampler_ut.cc -o test/sampler_ut && ./test/sampler_ut` → 100 passed, 0 failed（GREEN）。
> Sampler 落位 `cann_lm/engine/core/sample/`。无 Python 依赖；生产版可改取 CANN ACL Tensor（接口形态不变）。runner/backend/registry/CLI 等他人模块若同落 `sample/` 子包，各自文件、互不干涉。

## 12. 风险与开放问题

| # | 项 | 风险/问题 | 处置 |
|---|---|---|---|
| R1 | rep penalty 镜像 HF `RepetitionPenaltyLogitsProcessor` | 与 transformers GenerationMixin 路径行为应一致但**未对测**（对测属他人 backend 验证） | 标注；本 SPEC 保证公式与 HF 一致，单测覆盖正/负 logit两式 |
| R2 | 默认返回全 vocab `probs`（供投机采样） | 全 vocab 显存开销大 | 默认全 vocab（Q4 决断）；若实测吃紧，加 `return_top_k` 选项返回截断 + 索引 |
| Q1 | 是否需要 batch>1 | 首版 batch=1 | 非 goal，后续 |
| Q2 | `repetition_penalty` 等序列相关 processor 归属 | 已决断：`repetition_penalty` 进 Sampler（形态单一），其余（`no_repeat_ngram`/`min_length`）外移上层 | 见 §8.3 |

## 13. 变更记录

| 日期 | 版本 | 摘要 |
|---|---|---|
| 2026-07-27 | 0.1 | 初稿，多方案并存草案 |
| 2026-07-28 | 0.2 | READ 验证两模型后处理兼容；采纳 `gen_kwargs` 透传口、`@register(model_types=)` 自描述注册表、`input_len` 语义明确化、ByteTokenizer.decode 忽略 skip_special_tokens、softcapping 由 forward 内部做、runner 统一计时 |
| 2026-07-29 | 0.3 | 引入 6 层架构定位；新增 Layer 3 Sampler 设计——Python 原型 + C++ 契约并行、采样 + 返回概率分布、不管停止；新增 R5/Q4/AC7/AC8 |
| 2026-07-29 | 0.4 | Sampler 内部解剖：对外只露 `Sampler::Sample()`+两 dataclass，内部分 5 步纯函数流水线；移除 `SamplerConfig.extra`（序列相关 processor 外移，Q5 决断）；新增 `seed` 字段保可复现；C++ 加 private helper 镜像；新增 AC9/Q5 |
| 2026-07-30 | 0.5 | `repetition_penalty` 进 Sampler（反转 0.4 Q5）、引入 ModelExecutor、§16.6 混合架构（gemma4 阶段2 / qwen3.5 阶段1 fallback）+ 全阶段2 参考方案、AC7 改写、C++ 契约同步 generated_ids+rep penalty |
| 2026-08-03 | 0.6 | **范围收窄到 Sampler-only**：删除 runner/backend/registry/CLI/ModelExecutor 的设计（§6 架构、§7 接口契约、§8 backend 规格、§9 CLI、§10 扩展、AC1-AC6 移出，归他人）；保留 §15/§16 重编为 §7/§8；Sampler 落位 `cann_lm/engine/core/sample/sampler.py`；新增 §9 调用示例、§10 可运行验收（指向 `test_sampler.py`）；R3/R4/R5（他人 backend 相关）移出，保留 R1/R2 + Q1/Q2 |
| 2026-08-05 | 0.7 | **实现语言转 C++**：删 Python 原型（`sampler.py`/`test_sampler.py`），新增 `sampler.h`/`sampler.cc`/`test_sampler.cc`（无外部依赖，`std::vector<float>` + `std::mt19937_64`，自定义测试 runner 不依赖 gtest）。**`Sample` 改静态方法**（Sampler 无状态，C++ 风格，不实例化、去 OOP ABC/`DefaultSampler`）；§8.1/§8.2/§8.4/§9/§10/§11 同步 C++ 化。73 条断言全 GREEN。生产版可换 CANN ACL Tensor（接口形态不变）。 |
| 2026-08-12 | 0.8 | **借鉴他项 sample 入参**两处：(1) **NPU 预算 top-k 钩子**——新增 `PrecomputedTopK` struct + `Sampler::ApplyPrecomputedTopK` + `Sample` 第 4 参 `std::optional<PrecomputedTopK>`；给了则复用、跳过 `ApplyTopK`（覆盖 `cfg.top_k`），NPU 已算 top-k 时省 host 重算。host 可测（不依赖设备），加 M 组 7 条断言。(2) **token id 改 `int32_t`**（词表<2^31 够用，省内存）——`generated_ids`/`next_token_id`/`Pick` 返回/`argmax` 全 `int32_t`。§8.1 接口块/§8.2 表同步。94 条断言全 GREEN。另：§8.1 加"为什么用 config struct 而非 set/get"设计理路。 |
| 2026-08-12 | 0.9 | **数据打包成 `SamplerInputs`**（借鉴他项 #3）：`logits`+`generated_ids`+`precomputed_topk` 三个数据入参拢进一个 struct，旋钮仍单独 `SamplerConfig`（数据/旋钮分离）。`Sample(SamplerInputs in, const SamplerConfig& cfg)` **按值取 in**→`std::move(in.logits)` 走零拷贝路径。选 vector 字段（非 span）：span 不能拥有/内联字面量初始化不了/有悬空风险；vector+按值+move 既零拷贝又测试友好、C++17 不升。§8.1 接口块/§9 调用示例同步；加 N 组 5 条断言（直接构造 SamplerInputs + move 语义验证）。99 条断言全 GREEN。 |
| 2026-08-19 | 0.10 | **UT 收口**：测试文件迁至 `test/` 子目录（与源码分置）、重命名 `test_sampler.cc`→`test/sampler_ut.cc`（带 `_ut` 后缀，文件名表义）；`#include "../sampler.h"`；`scripts/build_sample.sh` + `.gitignore`（二进制 `test/sampler_ut`）+ 文档路径同步。补 M6 边角（precomputed 含越界 id 应被静默忽略）。100 条断言全 GREEN。 |