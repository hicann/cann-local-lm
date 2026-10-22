# sampler.cc 逐行解析（新手向）

> 配套：`sampler.cc`（真实代码）、`SPEC.md §8`（设计）、`GUIDE.md §9`（白话概念）。
> 读法：每个方法先贴代码（带行号，对得上真实文件），再逐行解释 +「为什么这么写」。不假设你懂 C++ 高级特性，遇到 `vector`/`optional`/`move`/`lambda` 等会顺带讲。
> v0.8 更新：token id 改 `int32_t`（词表<2^31 够用、省内存）；新增 `ApplyPrecomputedTopK`（③ 的另一条路：复用 NPU 预算 top-k，跳过 ApplyTopK 重算）。
> v0.9 更新：数据打包进 `SamplerInputs`（logits+generated_ids+precomputed_topk），`Sample(SamplerInputs in, const SamplerConfig& cfg)` 按值取 in→`std::move(in.logits)` 零拷贝。行号以真实 `sampler.cc` 为准。

---

## 先建立大局观

`sampler.cc` 干一条 6 步流水线：

```
① 重复惩罚 → ② 温度 → ③ Top-K → ④ Top-P → ⑤ 归一化 → ⑥ 挑字
```

- 前 5 步都在**改 logits**（那个"给每个字的分数"数组）。
- 第 6 步从最终概率里**挑一个字**。
- `Sample()`（最底下）把 6 步串起来，是唯一对外入口。

核心数据：`logits` = `std::vector<float>`，一个浮点数组，每个元素是"词表里第 i 个字"的分数。最终挑出的字 = 数组某个下标。

---

## 文件头 + 基础设施（行 1-13）

```cpp
1   // cann_lm/engine/core/sample/sampler.cc — Layer 3 Sampler 实现（C++ 原型）
2   //
3   // 6 步纯函数流水线（镜像 HuggingFace 采样顺序）。期望行为由 test/sampler_ut.cc 校验。
4   #include "sampler.h"
5
6   #include <algorithm>
7   #include <cmath>
8   #include <random>
9   #include <set>
10
11  namespace cann_lm::sampler {
12
13  namespace {
```

- **行 4** `#include "sampler.h"`：引入本组件自己的头文件（接口声明在那）。双引号 = 先在当前目录找。
- **行 6-9** 标准库：
  - `<algorithm>` → `std::sort`、`std::max`、`std::fill` 等通用算法。
  - `<cmath>` → `std::exp`、`std::isfinite`、`INFINITY` 等数学函数。
  - `<random>` → `std::mt19937_64`（随机数发生器）。
  - `<set>` → `std::set`（去重容器，rep penalty 用）。
- **行 11** `namespace cann_lm::sampler { ... }`：把所有东西包进 `cann_lm::sampler` 命名空间，避免跟别人同名冲突。外面要用就得写 `cann_lm::sampler::Sampler::Sample(...)`。
- **行 13** `namespace { ... }`：**匿名命名空间**。里面的函数（`softmax_stable`、`argmax`）是本 `.cc` 文件**私有的**——别的文件链接不到、用不了。相当于"实现细节，别对外开放"。到行 45 的 `}` 结束。

---

## 辅助函数①：`softmax_stable`（行 15-35）

把 logits 转成"加起来=1"的概率分布。是第 ⑤ 步的本体，也被 Top-P 内部用。

```cpp
15  // 数值稳定 softmax（减最大值；double 累加避免大词表下 float 累加丢小项）。处理 -inf → 0。
16  std::vector<float> softmax_stable(const std::vector<float>& logits) {
17      float maxv = -INFINITY;
18      for (float v : logits) maxv = std::max(maxv, v);
19      if (!std::isfinite(maxv)) maxv = 0.0f;  // 全 -inf 兜底（不应发生）
20      std::vector<float> e(logits.size());
21      double sum = 0.0;
22      for (size_t i = 0; i < logits.size(); ++i) {
23          double d = (double)logits[i] - (double)maxv;
24          double ei = std::isfinite(d) ? std::exp(d) : 0.0;  // -inf → 0
25          e[i] = (float)ei;
26          sum += ei;
27      }
28      if (sum <= 0.0) {
29          // 退化（理论上不会，除非全 -inf）：均匀
30          std::fill(e.begin(), e.end(), 1.0f / (float)e.size());
31          return e;
32      }
33      for (float& v : e) v = (float)((double)v / sum);
34      return e;
35  }
```

- **行 16** 签名：`const std::vector<float>& logits`——`&` 是引用，**不拷贝**整个数组（省内存/时间）；`const` 表示只读、不改它。返回**新的** `vector<float>`（概率）。
- **行 17-18** 先找出 logits 里的**最大值** `maxv`。`for (float v : logits)` 是范围 for 循环，遍历每个元素。`-INFINITY` 是"负无穷"当起点，保证任何值都比它大。
- **行 19** `if (!std::isfinite(maxv))`：如果最大值是 -inf（说明**所有**元素都是 -inf，输入坏了），把 maxv 钳到 0 兜底。
- **行 20** 建结果数组 `e`，大小同 logits。
- **行 21** `double sum = 0.0`：用 **double（双精度）**累加。**为什么不用 float？** 词表大（如 10 万）时，float 累加会在和增长后"丢掉小项"（精度不够），导致概率和不准；double 精度高得多。
- **行 23** `d = logits[i] - maxv`：**减最大值**——softmax 的数值稳定技巧。直接 `exp(大数)` 会溢出成 inf；减掉最大值后最大项变 0，`exp(0)=1`，其余为负，`exp(负)`在 0~1 之间，不会溢出。
- **行 24** `ei = exp(d)`：算 e 的 d 次方。`isfinite(d) ? ... : 0.0`——三元运算符，如果 d 是 -inf（被屏蔽的字）就给 0。`-inf - maxv` 还是 -inf，`exp(-inf)=0`，这里显式写成 0 避免依赖隐式行为。
- **行 25** 存进 `e[i]`（转回 float 存）。
- **行 26** 累加进 `sum`。
- **行 28-32** 兜底：如果 `sum <= 0`（全 -inf 时所有 ei=0），退化为**均匀分布**（每个字等概率 `1/n`），避免除零。
- **行 33** 每个元素除以 `sum` 归一化——现在 `e` 加起来≈1，就是概率分布。`(float)((double)v/sum)` 在 double 里除完再转 float，更准。
- **行 34** 返回概率数组。

> 🌟 **关键概念**：softmax = "把任意分数变成概率（和=1）"。减最大值是防溢出的工程技巧，跟数学结果等价。

---

## 辅助函数②：`argmax`（行 37-43）

返回"概率最大的那个字的下标"——贪心挑字用。

```cpp
37  int32_t argmax(const std::vector<float>& v) {
38      int32_t best = 0;
39      for (size_t i = 1; i < v.size(); ++i) {
40          if (v[i] > v[best]) best = (int32_t)i;
41      }
42      return best;
43  }
```

- **行 38** `best = 0`：先假定第 0 个字最大。
- **行 39-40** 从第 1 个往后扫，遇到比当前最大的还大的，就更新 `best` 为它的下标。
- **行 42** 返回最大那个的下标。
- 平局时（多个一样大）返回**第一个**——因为只有"严格大于"才更新。后面 Top-K/Top-P 的 tie 行为也一致。

---

## 行 45：匿名命名空间结束

```cpp
45  }  // namespace
```

`softmax_stable` 和 `argmax` 的私有性到此为止。下面的都是公开的（头文件里声明过）。

---

## ① `Sampler::ApplyRepetitionPenalty`（行 47-61）

**重复惩罚**：对"已经生成过的字"降权，避免模型反复说同一个字。

```cpp
47  std::vector<float> Sampler::ApplyRepetitionPenalty(std::vector<float> logits,
48                                                     const std::vector<int32_t>& generated_ids,
49                                                     std::optional<float> penalty) {
50      // ① 对已生成 token 降权：score<0 → *=p；score>=0 → /=p。镜像 HF
51      // RepetitionPenaltyLogitsProcessor（从原始 logits 取分再写回，重复 token 等价单次）。
52      if (!penalty || *penalty == 1.0f || generated_ids.empty()) return logits;
53      float p = *penalty;
54      std::set<int32_t> seen(generated_ids.begin(), generated_ids.end());
55      for (int32_t id : seen) {
56          if (id < 0 || (size_t)id >= logits.size()) continue;  // 越界保护
57          float s = logits[id];  // 原始分（id 唯一，未被写过）
58          logits[id] = (s < 0.0f) ? (s * p) : (s / p);
59      }
60      return logits;
61  }
```

- **行 47** 参数 `std::vector<float> logits`——**按值传**（没有 `&`）。意思：调用时**拷贝一份**进来。所以我们改这份拷贝，不影响调用方原数组。每个 Apply* 都这样：输入按值、返回改后的副本——**纯函数**（不改外部状态）。
- **行 48** `generated_ids`：到目前生成过的字下标序列（`int32_t` 整数）。`const&` 只读引用。
- **行 49** `std::optional<float> penalty`：`optional` = "可能没有值"。penalty 可以是 `1.5`，也可以是 `nullopt`（= 没设、不惩罚）。
- **行 52** 早退：如果 `!penalty`（没设）或 `==1.0`（=1 等于不惩罚）或生成序列空（没生成过任何字），直接原样返回 logits，啥都不做。`*penalty` 是"取出 optional 里的值"。
- **行 53** 取出惩罚系数 `p`。
- **行 54** `std::set seen(...)`：把 generated_ids 塞进 set，**自动去重**。因为 [0,0,0] 和 [0] 惩罚效果应一样（同一个字只罚一次）。
- **行 55** 遍历每个**唯一**的已生成字 id。
- **行 56** 越界保护：id 不能为负、不能超词表大小。`continue` = 跳过这个。
- **行 57** `s = logits[id]`：**先读出原始分**。注意：因为 set 里每个 id 唯一，读的时候这个位置还没被改过，拿的是原始值。
- **行 58** 核心：`s < 0 ? s*p : s/p`。
  - 正分（s≥0）：`s / p`——除以 p。p>1 时分数变小 → 概率降。
  - 负分（s<0）：`s * p`——乘以 p。负数乘 p>1 更负 → 概率也降。
  - 两种情况都降权，但对正负分用不同运算，是因为除法和乘法对正负数的效果方向不同。这**镜像 HuggingFace 的实现**（保证行为一致）。
- **行 60** 返回改好的 logits 拷贝。

> 🌟 **为什么"先读原始分再写"重要**：如果顺序处理 [0,0]，先改 token0 成新值、再读 token0 又基于新值改一次，会"罚两次"。这里 set 去重 + 读原始值，保证重复字只罚一次。

---

## ② `Sampler::ApplyTemperature`（行 63-68）

**温度**：把所有分数除以 t，控制随机性。

```cpp
63  std::vector<float> Sampler::ApplyTemperature(std::vector<float> logits, float temperature) {
64      // ② logits / temperature；t<=0 钳到极小正值避免除零（退化为贪心）。
65      float t = (temperature <= 0.0f) ? 1e-8f : temperature;
66      for (float& v : logits) v /= t;
67      return logits;
68  }
```

- **行 65** 钳制：`t<=0` 时用 `1e-8`（极小正数）。因为后面要除以 t，t=0 会除零。t→0 时分数被放得极大，softmax 后最高分的字概率≈1，等价于"永远选最高"（贪心）——这是温度的极端行为。
- **行 66** `for (float& v : logits) v /= t`——`float&` 是**引用**，直接改原元素。每个分数除以 t。`t<1` 放大分差（更确定）、`t>1` 缩小分差（更随机）。
- **行 67** 返回。

> 🌟 **为什么除以 t 不改 argmax**：所有分数都除以同一个正数 t，"谁最大"的关系不变。所以贪心挑字时温度不影响结果；它只影响**概率分布**（给抽样用）和相对集中度。

---

## ③ `Sampler::ApplyTopK`（行 70-84）

**Top-K**：只留分数最高的 k 个，其余置 -inf（=概率 0）。

```cpp
70  std::vector<float> Sampler::ApplyTopK(std::vector<float> logits, std::optional<int> top_k) {
71      // ③ 只留概率最高 k 个，其余 -inf。k/None 或 >=vocab 或 <=0 → 跳过。
72      if (!top_k) return logits;
73      int k = *top_k;
74      int n = (int)logits.size();
75      if (k <= 0 || k >= n) return logits;
76      // 找第 k 大的阈值：降序排序后取 sorted[k-1]
77      std::vector<float> sorted = logits;
78      std::sort(sorted.begin(), sorted.end(), std::greater<float>());
79      float threshold = sorted[k - 1];
80      for (float& v : logits) {
81          if (v < threshold) v = -INFINITY;
82      }
83      return logits;
84  }
```

- **行 72** 没设 top_k → 跳过。
- **行 73-74** 取 k 值、词表大小 n。
- **行 75** k≤0 或 k≥n（要留的不少于总词数）→ 跳过（没意义）。
- **行 77** 复制一份 logits 到 `sorted`（不动原 logits）。
- **行 78** `std::sort(..., std::greater<float>())`——**降序**排（`greater` = 从大到小）。默认 sort 是升序。
- **行 79** `threshold = sorted[k-1]`——排好序后第 k 个（下标 k-1）就是"第 k 大的值"。它就是分界线：≥它的留下，<它的置 -inf。
  - 例：logits=[3,1,2]，k=2。排序后 [3,2,1]，`sorted[1]=2` 是阈值。3≥2 留、2≥2 留、1<2 → -inf。结果 [3,-inf,2]。
- **行 80-82** 遍历原 logits，**严格小于阈值**的置 `-INFINITY`。`-inf` 经 softmax 后变概率 0（等于"踢出候选"）。
- **行 83** 返回。

> 🌟 **为什么用 -inf 不用 0**：0 还是会有概率（softmax 给它正概率）。-inf 才能彻底踢掉。而且后面 softmax 减最大值后，-inf - maxv = -inf，exp(-inf)=0，干净。

---

## ③' `Sampler::ApplyPrecomputedTopK`（v0.8 新增）

**复用 NPU 预算的 top-k**：当 NPU 已经在设备上算好 top-k（生产流水线常见——大词表 host 排序贵，NPU 顺手算完把结果递过来），host 采样器直接复用、跳过自己的 ApplyTopK 排序。

```cpp
std::vector<float> Sampler::ApplyPrecomputedTopK(std::vector<float> logits,
                                                 const std::optional<PrecomputedTopK>& precomputed) {
    // ③' 复用 NPU 预算的 top-k 集合：不在集合里的置 -inf（设备已算过 top-k 的加速路径）。
    // 没给/空 → no-op（原样返回）。
    if (!precomputed || precomputed->token_ids.empty()) return logits;
    std::set<int32_t> keep(precomputed->token_ids.begin(), precomputed->token_ids.end());
    for (size_t i = 0; i < logits.size(); ++i) {
        if (keep.find((int32_t)i) == keep.end()) logits[i] = -INFINITY;
    }
    return logits;
}
```

- 参数 `precomputed`：`std::optional<PrecomputedTopK>`，"可能没有"。`PrecomputedTopK` 就一个字段 `token_ids`（NPU 算出的 top-k 下标集合）。
- **早退**：没给（nullopt）或空集合 → 原样返回（等于没这一步）。
- `std::set keep(...)`：把 token_ids 塞进 set 便于 O(log n) 查"在不在"。
- 遍历每个位置 i：**不在 keep 里**的置 -inf（跟 ApplyTopK 用 -inf 屏蔽同一手法）。
- 注意 `keep.find((int32_t)i)`——i 是 size_t，转 int32_t 去查 set<int32_t>。

> 🌟 **ApplyTopK vs ApplyPrecomputedTopK**：前者自己排序找阈值（host 算），后者直接用别人（NPU）给的结果集。**两者在 `Sample` 里二选一**：给了 precomputed 就用它、跳过 ApplyTopK（覆盖 `cfg.top_k`）；没给就走 ApplyTopK。等价性由测试 M 组验证（precomputed=真实 top-k 集合 ≡ ApplyTopK 结果）。
>
> 🌟 **为什么 precomputed 作 Sample 第 4 参、不塞进 SamplerConfig**：它是"数据输入"（NPU 算的结果，可能很大），不是"旋钮"。旋钮（温度/top_k/...）才进 SamplerConfig；数据单列第 4 参，符合 §8.1"数据与旋钮分离"理路。

---

## ④ `Sampler::ApplyTopP`（行 86-113）

**核采样（Top-P）**：按概率从高到低累加，累计达到 p 后面就截掉。比 Top-K 更聪明——它留"够大众"的字，数量不固定。

```cpp
86  std::vector<float> Sampler::ApplyTopP(std::vector<float> logits, std::optional<float> top_p) {
87      // ④ 核采样：累计概率达 p 后截尾；保留最高概率项。p/None → 跳过。
88      if (!top_p) return logits;
89      float p = *top_p;
90      int n = (int)logits.size();
91      // 按降序排索引
92      std::vector<int> idx(n);
93      for (int i = 0; i < n; ++i) idx[i] = i;
94      std::sort(idx.begin(), idx.end(),
95                [&](int a, int b) { return logits[a] > logits[b]; });
96      // 排序后的 softmax + 累计
97      std::vector<float> sorted_logits(n);
98      for (int i = 0; i < n; ++i) sorted_logits[i] = logits[idx[i]];
99      std::vector<float> probs = softmax_stable(sorted_logits);
100     std::vector<char> remove(n, 0);
101     float cum = 0.0f;
102     for (int i = 0; i < n; ++i) {
103         cum += probs[i];
104         if (cum > p) remove[i] = 1;
105     }
106     // shift 保留首个（最高概率项必留）
107     for (int i = n - 1; i >= 1; --i) remove[i] = remove[i - 1];
108     remove[0] = 0;
109     for (int i = 0; i < n; ++i) {
110         if (remove[i]) logits[idx[i]] = -INFINITY;
111     }
112     return logits;
113 }
```

- **行 88-90** 早退/取值，同 Top-K。
- **行 92-93** 建 `idx` = [0,1,2,...,n-1]（下标序列）。我们要**排序下标**而非 logits 本身——这样排完还能回查原位置。
- **行 94-95** `std::sort` 用 **lambda**（`[&](int a,int b){...}`）做比较器：按 `logits[a] > logits[b]` 排，即"对应的 logits 从大到小"。`[&]` = 捕获外部变量（logits）按引用。排完后 `idx[0]` 是 logits 最大那个的下标，依此类推。
- **行 97-98** 按排好的下标顺序，把 logits 值拷进 `sorted_logits`（从大到小排列）。
- **行 99** 对排好序的 logits 算概率 `probs`（也是从大到小）。
- **行 100** `remove` 数组（char 当 bool 用，省内存），标记每个位置要不要踢掉。
- **行 101-105** 累计概率 `cum`；一旦 `cum > p`，从这位置起标记要踢。意思是"前面这些已经凑够 p 的概率了，后面的小角色都踢掉"。
  - 例：probs=[0.67,0.24,0.09]，p=0.9。cum 走到 0.67(<0.9)、0.91(>0.9) → 第 2 位置起 remove=1。
- **行 107-108** **shift（右移一位）保留首个**：`remove[i]=remove[i-1]`，从后往前移；再把 `remove[0]=0`。
  - **为什么**：累计到刚超过 p 的那一项本身不该被踢（它正好补够 p），只有它**之后**的才踢。原始 remove 标的是"超过 p 的位置起"，shift 一下就变成"超过 p 的下一位置起"，保证最高概率项（第一个）一定留下。
- **行 109-111** 对被标记 remove 的位置，把**原 logits** 对应下标 `idx[i]` 置 -inf。注意是写回 `logits[idx[i]]`（原位置），不是排序后的位置。
- **行 112** 返回。

> 🌟 这是最绕的一步。抓住两点：① 排序的是**下标**不是值，方便回写原位置；② shift 保证"最高概率项必留"，且"刚好补够 p 的那项也留，只踢更后的"。

---

## ⑤ `Sampler::Normalize`（行 115-118）

```cpp
115 std::vector<float> Sampler::Normalize(const std::vector<float>& logits) {
116     // ⑤ softmax 归一化（数值稳定）。
117     return softmax_stable(logits);
118 }
```

- 直接调上面的 `softmax_stable`，把 logits 转成概率分布。单独包一层是为了对外暴露名字清晰（"归一化"），且让流水线 6 步名字整齐。
- 注意这里 `const&`（只读引用），`softmax_stable` 内部自己建新数组返回，不改输入。

---

## ⑥ `Sampler::Pick`（行 120-137）

从概率分布里**挑一个字**。两种挑法：贪心 or 抽样。

```cpp
120 int32_t Sampler::Pick(const std::vector<float>& probs, bool do_sample, std::optional<uint64_t> seed) {
121     // ⑥ do_sample=True→按概率 multinomial（inverse-CDF）；False→argmax。
122     if (!do_sample) return argmax(probs);
123     std::mt19937_64 gen;
124     if (seed) {
125         gen.seed(*seed);
126     } else {
127         gen.seed(std::random_device{}());
128     }
129     std::uniform_real_distribution<float> uni(0.0f, 1.0f);
130     float u = uni(gen);
131     float c = 0.0f;
132     for (size_t i = 0; i < probs.size(); ++i) {
133         c += probs[i];
134         if (u < c) return (int32_t)i;
135     }
136     return (int32_t)(probs.size() - 1);  // 浮点兜底
137 }
```

- **行 120** `probs` 是上一步 Normalize 出来的概率分布（和≈1）。
- **行 122** 贪心模式（`do_sample=false`）：直接 argmax，永远选概率最大的。可复现、确定。
- **行 123** 抽样模式：建随机数发生器 `mt19937_64`（梅森旋转，质量好）。
- **行 124-128** 种子：给了 `seed` 就用它（→ 同输入同种子**可复现**）；没给就用 `random_device`（真随机）。这是 AC7"可复现"的关键。
- **行 129-130** `uniform_real_distribution<float> uni(0,1)`：均匀分布，产生 [0,1) 随机数 `u`。`uni(gen)` = 用 gen 生成一个。
- **行 131-135** **逆 CDF 抽样**（inverse-CDF multinomial）：
  - `c` 累计概率，从头往后走。
  - 一旦 `u < c`，就返回当前位置 i。
  - 直觉：把 [0,1) 区间按概率切成段（概率大的段长），扔一个随机点 u，落在哪段就选哪个字。概率大的字段长、更容易被选中，但小字也有机会——这就是"采样"vs"贪心"的区别。
- **行 136** 兜底：浮点累计误差导致循环走完没命中（理论上 u<1=总概率=最后 cum，但浮点可能差一丁点），返回最后一个字，保证总有返回。

> 🌟 **为什么逆 CDF = multinomial**：multinomial（多项分布抽样）的等价实现就是"生成 [0,1) 均匀随机数、沿累计概率找到对应槽"。数学上等价，实现简单。

---

## 总装：`Sampler::Sample`（v0.9：SamplerInputs 打包）

唯一对外入口。把 6 步串起来。**v0.9 起数据打包进 `SamplerInputs`**（logits + generated_ids + precomputed_topk），旋钮在 `cfg`。

```cpp
139 SamplerOutput Sampler::Sample(SamplerInputs in, const SamplerConfig& cfg) {
140     // ①→②→③(precomputed 或 ApplyTopK)→④→⑤→⑥；in 按值传入，可 move in.logits 走零拷贝
141     std::vector<float> x = ApplyRepetitionPenalty(std::move(in.logits), in.generated_ids, cfg.repetition_penalty);
142     x = ApplyTemperature(std::move(x), cfg.temperature);
143     // ③ TopK：precomputed 在场则复用（覆盖 cfg.top_k，NPU 已算过），否则自己算
144     if (in.precomputed_topk && !in.precomputed_topk->token_ids.empty()) {
145         x = ApplyPrecomputedTopK(std::move(x), in.precomputed_topk);
146     } else {
147         x = ApplyTopK(std::move(x), cfg.top_k);
148     }
149     x = ApplyTopP(std::move(x), cfg.top_p);
150     std::vector<float> probs = Normalize(x);
151     int32_t id = Pick(probs, cfg.do_sample, cfg.seed);
152     return SamplerOutput{id, probs};
153 }
```

- **行 139** 签名：`SamplerInputs in`（**按值取**——数据打包：logits + generated_ids + precomputed_topk）、`cfg`（旋钮：温度/Top-K/Top-P/rep/seed）。返回 `SamplerOutput`（= token id + probs）。按值取 `in` 是为了能 `std::move(in.logits)` 走零拷贝。
- **行 141** 第①步：rep penalty。`std::move(in.logits)`——把 in.logits 的内容**搬**进 ApplyRepetitionPenalty（按值参），不拷贝（零拷贝路径）。`in.generated_ids` 只读引用传（rep penalty 不改它）。
- **行 142-148** 第②③步：②温度；③ TopK **二选一**——`in.precomputed_topk` 给了且非空就 `ApplyPrecomputedTopK`（复用 NPU 结果、覆盖 `cfg.top_k`），否则 `ApplyTopK`（自己排序）。`std::move(x)` 把 x 搬给下一步，省内存分配。
- **行 149** 第④步 ApplyTopP：核采样截尾。
- **行 150** 第⑤步 Normalize：把最终 logits 转成概率 `probs`。
- **行 151** 第⑥步 Pick：从 probs 挑一个字，得 `id`。
- **行 152** 打包返回：`SamplerOutput{id, probs}`。

> 🌟 **顺序为什么是这样**：①rep 必须在 ③④top_k/top_p 前——rep 会改变哪些字进 top-k/截断集，先选后惩会让被屏蔽的字逃过惩罚（SPEC §8.2 + `test_rep_before_topk_order` 验证过）。①和②temperature 其实可交换（都是标量乘），放最前是跟 HuggingFace 一致的约定。贪心模式下 ③④不影响挑哪个字（只影响 probs），但 ①rep 会改变 argmax——这正是 rep 的目的（避免重复）。

---

## 附：新手常踩的 C++ 点（本文件涉及）

| 点 | 出现在 | 一句话 |
|---|---|---|
| `const T&` | 几乎所有参数 | 只读引用，不拷贝、不改原数据 |
| `T`（按值） | Apply* 的 logits 参数 | 拷贝一份进来，改的是拷贝 → 纯函数 |
| `std::optional<T>` | penalty/top_k/top_p/seed | "可能没有值"，用 `*x` 取值、`!x` 判断有没有 |
| `std::move(x)` | Sample 里串接 | 把内容搬走而非拷贝，省内存（搬完 x 别再用） |
| `for (T& v : vec)` | 多处 | 引用遍历，能改原元素；`for (T v : vec)` 是拷贝遍历 |
| lambda `[&](int a,int b){...}` | Top-P 排序 | 临时比较函数，`[&]` 捕获外部按引用 |
| 匿名 `namespace {}` | 行 13-45 | 文件私有，外部链接不到 |
| `-INFINITY` | Top-K/Top-P/softmax | 负无穷，经 softmax→0，用于"踢掉候选" |

---

*本文为 `sampler.cc` 的新手逐行注解。代码以 `sampler.cc` 真实文件为准；改代码时本文同步更新。*
