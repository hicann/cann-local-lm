# UT 指南 — 读懂 Sampler 的单元测试（新手向）

> 配套：`test/sampler_ut.cc`（真实测试代码）、`SPEC.md §10`（验收标准）、`GUIDE.md`（Sampler 概念）。
> 读法：本文不讲"采样器是什么"（那看 GUIDE），讲**它的测试是怎么设计的、为什么可信、典型用例怎么想出来的**。不假设你懂测试框架。
> 运行：`bash scripts/build_sample.sh` → `100 passed, 0 failed`（详见末节）。

---

## 一、UT 整体是怎么设计的

### 1. 测的是谁、不测谁
- **测**：`sampler.h`/`sampler.cc` 里那个 `Sampler`——6 步采样流水线（重复惩罚→温度→Top-K→Top-P→归一化→挑字）+ 对外入口 `Sample`。
- **不测**：模型前向、生成循环、停止判断、解码、计时——那些归别人（SPEC §7 边界）。UT 只管"给 logits + 旋钮，采样器挑对字、出对概率"。

### 2. 结构：分组 + 单函数单主题
29 个测试函数、100 条断言，按主题分 12 组（A-N）：

| 组 | 主题 | 测什么 |
|---|---|---|
| A | 形状&归一化 | probs 形状对、和≈1、非负、id 合法 |
| B | 贪心=argmax | do_sample=false 时挑 argmax、probs==softmax |
| C | temperature | 温度不改 argmax、probs==softmax(logits/t)、t≤0 钳制 |
| D | top_k | 屏蔽非 top-k、k=1 argmax≈1、None/≥vocab no-op |
| E | top_p | 极小 p 只留 argmax、None no-op |
| F | repetition_penalty（7 子项） | noop / argmax 翻转 / 负 logit / <1 升权 / 顺序 / +temp / 重复 gen |
| G | 可复现（2 子项） | 同 seed 同结果、rep+采样 可复现 |
| H | 纯函数单测（6 子项） | 6 个 `Apply*`/`Normalize`/`Pick` 逐个独立测 |
| K | 组合 | 全旋钮齐开不出 nan/inf |
| L | 边角（6 子项） | 10 万词表、全 -inf、rep 多 token、temp 极大、top_p 性质、multinomial 统计 |
| M | 预算 top-k（8 断言） | NPU 预算 top-k 复用、等价性、优先级、越界 id 忽略 |
| N | SamplerInputs 直接（5 断言） | 真实接口直调、move 语义（零拷贝） |

**每个函数只测一个主题**（如 B 全是贪心、D 全是 top_k）——挂了能一眼定位是哪类功能坏。函数内部多条 `check()` 断言细化（如 C 有 7 条覆盖 t=0.5/1/0/-1/1e-8）。

### 3. 不用框架，自包含
- **不依赖 gtest/catch2**：测试自己写了个极简 runner（`check(name, cond)` 计 pass/fail，`main()` 跑全部 + 打印 `100 passed, 0 failed`）。
- **单 TU**：`sampler.cc` + `test/sampler_ut.cc` 一编一跑，无外部库。好处：任意有 g++ 的机器直接跑，无需装框架、无需 CMake。
- **退出码**：全过=0、有挂=非0——CI 里直接拿来判定。

### 4. 测试先行（test-first）
测试**先于/同期**写，作为"标准答案"——实现要来对上测试，不是反过来。所以测试本身定义了"什么算对"。

---

## 二、可靠性怎么保证（核心）

测试值不值得信，取决于它**会不会和实现一起犯错**。我们用四道防线防这个：

### 防线 1：独立 oracle（最重要）
"期望值"**不靠被测代码算**，而是测试文件里**自己写一套独立的小工具**算：

| 测试内工具 | 干什么 | 为什么独立 |
|---|---|---|
| `T_softmax` | 用最直白的循环算 softmax（减最大值→exp→归一） | 和实现的 `softmax_stable` 是**两段独立代码**；实现就算 softmax 写错，测试这套不会跟着错 |
| `T_argmax_local` | 扫一遍找最大值下标 | 独立于实现的 `argmax` |
| `sum` / `approx` / `no_bad` | 算和、比近似、查 nan/inf | 通用小工具，不含采样逻辑 |

→ 比如 B2 断言 `probs == T_softmax(logits)`：实现的 probs 要等于**测试自己算的** softmax。实现若把 softmax 写歪，测试这套不歪，挂——抓出来。

### 防线 2：手算精确值（关键行为）
对**关键行为**不靠 oracle、直接**手算**期望：
- F4 重复惩罚翻转：logits=[2.0, 1.9]，token0 生成过、rep=2.0 → token0 变 2.0/2.0=**1.0**，token1 不变=1.9 → argmax 从 0 翻到 **1**。这条"翻转到 1"是手推出来的，不依赖任何代码。
- D3 top_k=2：阈值=第 2 大值=2.0，<2.0 的置 -inf → 结果 [3, -inf, 2] → probs==softmax([3,-inf,2])。

→ 这些用例把"实现到底该输出什么"钉死，连 oracle 都不靠，最硬。

### 防线 3：性质断言（怕脆的地方改用性质）
有些场景算精确值又难又脆（如采样随机性、超大词表），就**断言性质**而非精确值：
- A2 `probs 和≈1`、A3 `非负`——不卡具体数值，卡"必须是合法概率分布"。
- L1 10 万词表：只验"不崩、和≈1、size 对、id 合法"——不卡每个值。
- G3 不同 seed 结果可不同——不卡具体选哪个，卡"真的有随机性"。
- L14 multinomial 强偏分布 token0 命中>60%——不卡精确次数，卡统计合理范围。

→ 性质断言对实现细节不敏感（不脆），但仍抓"行为对不对"。

### 防线 4：确定性 + 隔离
- **确定性**：采样用固定 `seed`（`mt19937_64`）→ 同输入同 seed 必同结果（AC7 可复现）。没有"跑十次偶尔挂"的 flaky。
- **隔离**：每个测试自己建配置/数据，纯函数、无共享可变状态——**测试之间互不依赖**，换顺序跑也一样。
- **数值稳健**：测试的 `sum()` 用 **double** 累加（不用 float）——10 万项 float 累加会丢精度误判；double 避免测试自己算错。编译开 `-Werror`（零警告）。

### 防线 5：直测真实接口
- N 组**直接** `Sampler::Sample(SamplerInputs{...}, cfg)`（5 条）——不走任何包装，验真实对外接口。
- 其余用 `S(...)` 包装（把 4 参打包成 `SamplerInputs` 再调真 `Sample`）——`S` 只是少打字，内部仍调真实接口；它的存在和映射在注释里写明。

→ 所以 100 条断言全在测**真实 `Sample`**，不是测一个假壳。

---

## 三、典型用例的设计逻辑（几个代表）

### 用例 1：B 贪心 = 最简基线
**想**：先验最简单的路径——`do_sample=false`（不抽样）时，采样器应该**永远挑分数最高的字**，且概率分布就是输入的 softmax。
**断言**：`next==argmax`、`probs==softmax(logits)`。
**价值**：这是"采样器的 1+1=2"——连这都不过，后面全免谈。

### 用例 2：F4 重复惩罚翻转 = 验"惩罚真起作用"
**想**：rep penalty 的目的是**改变**输出（避免重复）。如果设了 rep 但 argmax 不变，那 rep 没生效。所以造一个"argmax 恰好是已生成过的字"的场景，看 rep 是否把它**拉下马**。
**构造**：logits=[2.0, 1.9]，argmax=token0，但 token0 在 generated_ids 里。rep=2.0。
**手算期望**：token0: 2.0/2=1.0；token1: 1.9 不变。→ 1.9 > 1.0，argmax 翻成 1。
**断言**：`next==1`（翻了）+ `probs==softmax([1.0, 1.9])`。
**价值**：钉死"rep 改变 argmax"这个关键行为，且正分用 `/=p`、负分用 `*=p` 两式都各测一遍（F4 正、F6 负）。

### 用例 3：D top_k 屏蔽 = 验"-inf 踢人"
**想**：top_k 要"只留最高 k 个、其余概率 0"。怎么验"踢干净"？看被踢的是不是真 0。
**构造**：logits=[3,1,2]，k=2 → 留 {3,2}、踢 1。
**断言**：被踢的 idx1 `probs[1]≈0`（精确 0，因 -inf 经 softmax→0）+ `probs==softmax([3,-inf,2])`。
**价值**：验了 `-inf → softmax → 0` 这条工程链（不是拍脑袋说"踢了"，而是数学上验概率真为 0）。

### 用例 4：G 同 seed 同结果 = 验可复现
**想**：AC7 要求"同输入+同 seed→同结果"。造两次同 seed 调用，看是否相等。
**断言**：两次 `next_token_id` 相等。
**价值**：钉死可复现契约（生产/调试要靠它回放）。同时 G3 用"近均匀分布+多 seed 至少两个不同"反向验"真有随机性"（不是写死）。

### 用例 5：M2 等价性 = 验"NPU 复用路径"对
**想**：NPU 预算 top-k（precomputed）是"复用设备算好的"，必须和"自己算 top-k"结果**一致**，否则复用就错了。
**构造**：同一组 logits，一路走 `ApplyTopK(top_k=2)`、一路走 `ApplyPrecomputedTopK(真实 top-k 集合)`。
**断言**：两路 `probs` 和 `next` 相同。
**价值**：保证"设备给的 top-k"和"host 自己算的"等价——这是 NPU 加速钩子能用的前提。

### 用例 6：L1 10 万词表 = 验健壮性
**想**：大词表会不会崩？概率和还准吗？
**构造**：vocab=100000，一个 argmax。
**断言**：size 对、next==argmax、和≈1（用 double sum 放宽 tol 到 1e-3，因 100k float 累加固有误差）。
**价值**：压力测试——顺带抓出过"float 累加丢精度"的真 bug（后改 impl 用 double 累加、测试 sum 也改 double）。

### 用例 7：N3 move 语义 = 验零拷贝路径
**想**：`Sample(SamplerInputs in, cfg)` 按值取 in、可 `std::move(in.logits)`。验 move 真发生了（原 vector 应空）。
**构造**：`SamplerInputs{.logits=std::move(Lm)}`，调完看 `Lm.empty()`。
**断言**：`Lm.empty()` + 结果仍正确。
**价值**：验"零拷贝路径"不是嘴上说说——move 真生效、且不影响正确性。

---

## 四、怎么跑、看什么

**一条命令**（仓库根目录）：
```bash
bash scripts/build_sample.sh
```
输出（成功样例）：
```
==> 编译 sampler.cc + test/sampler_ut.cc
==> 运行 test/sampler_ut
--- A 形状&归一化 ---
  [PASS] A1 probs size==4
  [PASS] A2 probs sum≈1
  ...
--- B 贪心=argmax ---
  [PASS] B1 greedy==argmax
  ...
Sampler tests: 100 passed, 0 failed
```
- 全 `[PASS]` + `100 passed, 0 failed` = 通过。
- 出 `[FAIL]` 会指名哪条 + 退出码非 0。
- **环境**：任意 Linux + g++≥7（x86/ARM 都行），**无需 NPU/Python/CANN**——纯 host C++。
- 要 CI 自动跑：把这条命令塞流水线，靠退出码（0=过）判定。

---

## 五、一句话总结

这套 UT 的设计哲学：**测试当"标准答案"先写，期望值用独立小工具或手算（不和实现共享逻辑），采样用固定 seed 保证可复现，难卡精确值的地方改卡性质——四道防线让"测试和实现一起犯错"的盲区尽可能小**。所以 100 条全绿时，可以信"采样器按 SPEC 干活了"。

---

*本文为 `test/sampler_ut.cc` 的新手向设计注解。测试代码以真实文件为准；改测试时本文同步更新。*
