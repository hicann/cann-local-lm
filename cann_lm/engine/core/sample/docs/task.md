# task.md — Sample（Sampler）后续实现规划

> 范围：仅 **Sampler（Layer 3 采样组件，C++）**。runner / backend / registry / CLI / ModelExecutor 归他人，本文件不规划它们，只在"集成边界"处提一句。
> 现状（2026-08-07）：Sampler C++ **原型已实现并 GREEN**——`sampler.h`/`sampler.cc`（6 步静态方法流水线，`std::vector<float>`，无外部依赖）+ `test/sampler_ut.cc`（100 条断言，自定义 runner，不依赖 gtest）。SPEC v0.10 / GUIDE v0.10 已对齐。
> 编译运行：`bash scripts/build_sample.sh` → 100 passed, 0 failed（P3.1 后）。
> 环境：CANN 9.0.0-beta.2 工具链已装（`/home/developer/Ascend/cann/set_env.sh`，`libascendcl.so` 可链接），**但 `aclrtGetDeviceCount`=0——本机无可用 NPU 设备**。

## 执行状态
- ✅ **P0.1** 构建 `scripts/build_sample.sh`（`-Wall -Wextra -Werror -Wno-missing-field-initializers`，GREEN 可复现）
- ✅ **P0.3** 登记进 harness `test_commands`（并修复 toml 中被 linter 损坏的 `test_commands` 键——原为 `/ =`，已恢复为合法 key）
- ✅ **P3.1** 边角测试（大词表/全 -inf/rep 多 token/temp 极大/top_p 性质/multinomial 统计）→ 87 GREEN；顺带加固：impl softmax 改 double 累加（100k 词表精度 1e-3→1e-8）、test `sum()` 改 double
- ✅ **P3.2** `-Werror` 清洁、删未用 `T_argmax`（注：边角测试后重新引入 `T_argmax_local` 并已使用）
- ⛔ **P1 阻塞**：本机无 NPU 设备（devices=0），真实 ACL Tensor adapter 无法验证。且采样本身是 host 侧计算——`vector<float>` 可能就是生产接口（device↔host 编组归他人）。待设备就绪或确认需要 device 侧采样再实现。
- ⏳ P2（SampleWithDraft / batch>1 / return_top_k）：SPEC 标注"按需/预留/首版 batch=1"，无引擎驱动前属投机性扩展，暂缓。

---

## 优先级图

```
P0 构建/CI 收口 ──┐
                  ├─→ P3 加固（边角测试 + -Werror）
P1 生产版 ACL Tensor 适配 ──┘
P2 扩展（投机采样 / batch>1 / return_top_k）─ 独立，可并行
```

---

## P0 — 构建/CI 收口（无依赖，小）

> 目的：让"73 GREEN"可一键复现、可被持续验证，而不是每次手敲 g++。

- [ ] **P0.1** 加构建脚本 `scripts/build_sample.sh`：source CANN set_env（兼容无 CANN 时跳过）→ `g++ -std=c++17 -O2 -Wall sampler.cc test/sampler_ut.cc -o test/sampler_ut` → 跑 `./test/sampler_ut`，退出码反映通过/失败。
  - 验收：`bash scripts/build_sample.sh` 在任意 cwd 跑通，输出 `100 passed, 0 failed`，非零退出码当失败。
  - 涉及：`scripts/build_sample.sh`（新建）。
- [ ] **P0.2**（可选）`cann_lm/engine/core/sample/CMakeLists.txt`：`add_executable(test_sampler sampler.cc test/sampler_ut.cc)` + `enable_testing()` + `add_test(...)`。CANN env 已带 cmake。
  - 验收：`cmake -S . -B build && cmake --build build && ctest --test-dir build` 通过。
  - 取舍：脚本更简、无依赖；CMake 更标准、易接 CANN target。先脚本，CMake 留作 P1 起需要 CANN 链接时一并上。
- [ ] **P0.3** 把 `bash scripts/build_sample.sh` 登记进 harness `test_commands`（`.kirinharness/tunables/edit-test-nudge-config.toml`），让 edit-test-gate 自动识别为测试证据。
  - 验收：hook 将该命令运行记为有效测试。

---

## P1 — 生产版 ACL Tensor 适配（依赖 CANN，已就绪，中）

> 目的：把原型的 `std::vector<float>` 换成 CANN ACL Tensor，对接 §8.4 生产版契约（**接口形态不变，仍静态**）。先正确性、后性能。

- [ ] **P1.1** 选型：生产版放 `sampler_acl.h`/`sampler_acl.cc`（与原型并行，复用算法），还是把 `Sampler` 模板化（template over Tensor trait）。
  - 倾向：先 **adapter 式**——`SamplerAcl` 接 ACL Tensor，内部把 logits 拉到 host vector → 跑原型 6 步 → probs 写回 device。正确性优先，性能后置。
  - 决断点：ACL Tensor 的 host 访问 API（`aclrtMemcpy` / Tensor accessor）。
- [ ] **P1.2** 实现 `sampler_acl.h`/`sampler_acl.cc`：`class SamplerAcl { static SamplerOutput Sample(const acl::Tensor&, const Span<int64_t>&, const SamplerConfig&); ... }`，6 步复用 `sampler.cc` 的纯函数（抽公共 impl）。
  - 验收：ACL Tensor 路径产出与 host 原型一致（同输入→同 token + probs）。
  - 依赖：`source /home/developer/Ascend/cann/set_env.sh`；链接 `-lacl` 等。
- [ ] **P1.3** `test_sampler_acl.cc`：用 ACL Tensor（或 mock Tensor）喂同组用例，断言与 host 原型结果一致。
  - 验收：新增测试 GREEN；device/host 一致性成立。
- [ ] **P1.4**（性能，后置）把 6 步算子逐步迁到 device 侧（ACL op / 自定义算子），减少 host↔device 拷贝。仅在 P1.2/3 正确性立住后做。
  - 验收：大词表下吞吐优于 adapter 式；数值仍一致。

### P1.5（备忘，未决）host 侧 SIMD 优化（ARM NEON）

> 来源：他项 sample 用 `__ARM_NEON` 宏给 argmax 加并行比较（4 元素/指令）。本机 aarch64 已验 NEON 可用（`__ARM_NEON=1`、`<arm_neon.h>` 编译运行 OK，NEON 是 aarch64 基线、无需额外 flag）。**暂不实现，仅记录待 benchmark 后定夺。**

- **候选点**：
  - `argmax`——自动向量化差（带下标归约有数据依赖），手写 NEON（`vld1q_f32`/`vmaxq_f32`/`vmaxvq_f32`）能 4 路并行找最大值；但只在贪心模式调、且只占流水线一小段，**收益窄**。
  - `normalize`/softmax（`exp`+`sum`）——计算更密集、O(vocab)，**可能更值钱**（`vexpq`/`vpaddq`）。
  - `ApplyTemperature`/`ApplyRepetitionPenalty` 等逐元素循环——g++ `-O2` 已能自动向量化（无下标依赖），**不用手写**。
- **实现要点（若做）**：宏守卫 `#if defined(__ARM_NEON)||defined(__ARM_NEON__)` + `#include <arm_neon.h>` + NEON 路径，`#else` 标量回退（保 x86/ARMv7 可移植）；两遍法（并行找最大值→标量扫首下标，保持"平局取首个"语义）；尾部 `n%4` 标量；行为不变（测试仍过）。
- **决策门槛**：进 P1.4 阶段时先写 benchmark（10 万词表 ×1000 次 Sample，profile 各步耗时）→ 确认 argmax/normalize 占比再决定 SIMD 哪个；**无 benchmark 不优化**。属 P1.4 子项或独立 host-SIMD pass。

---

## P2 — 扩展（SPEC 预留项，独立可并行，中）

- [ ] **P2.1 投机采样** `SampleWithDraft(logits, draft_tokens, generated_ids, cfg)`（§8.5）：target 分布 + draft 候选 → rejection sampling 接受/拒绝。
  - 验收：draft 命中时接受、偏离时拒绝并按 target 重采；用例 GREEN。
  - 依赖：`Sampler::Sample` 已返回 `probs`（已就绪）。
- [ ] **P2.2 batch>1**（Q1）：logits/probs 从 `[vocab]` 推广到 `[batch][vocab]`，6 步沿 batch 维广播。
  - 验收：batch=2 用例 GREEN；batch=1 行为不变（回归）。
- [ ] **P2.3 `return_top_k` 截断分布选项**（R2/Q4）：显存吃紧时返回截断分布 + 索引而非全 vocab。
  - 验收：`return_top_k=k` 时 probs 仅 k 项 + indices；投机采样 rejection 仍可用。

---

## P3 — 加固（边角 + 工程化，小-中）

- [ ] **P3.1** 边角测试：超大词表（10万+，验显存/性能不崩）、全 -inf logits、rep penalty 大量 generated token、temperature 极端、top_p 浮点末位（H9 类，系统化）、multinomial 多 seed 分布统计。
  - 验收：新增边角用例 GREEN，无 nan/inf/越界。
- [ ] **P3.2** `-Werror` / 静态分析清洁：`g++ -std=c++17 -O2 -Wall -Wextra -Werror` 干净；删 `test/sampler_ut.cc` 里未用的 `T_argmax`。
  - 验收：零 warning 零 error 编译通过。
- [ ] **P3.3** 文档回写：每完成一项，更新 SPEC §10（验收清单加新 AC）、§13 changelog、GUIDE 对应章节。
  - 验收：SPEC/GUIDE/test 三处一致。

---

## 集成边界（归他人，不在本规划内，仅记录接口契约）

Sampler 只暴露 `Sampler::Sample(logits, generated_ids, cfg) -> {next_token_id, probs}`（静态）。上层（他人）负责：
- 生成循环：`forward_step → logits → Sampler::Sample → token → 拼序列 → Context 判停`。
- 投机采样：消费 `probs` 做 rejection（若 P2.1 未由 Sampler 提供，则上层用 `Sample` 自行实现）。
- 序列 processor（`no_repeat_ngram`/`min_length`）：调 Sampler 前对 logits 预处理（SPEC §8.3）。

本规划不实现上述循环/编排/processor——等他人模块就绪后做**对接联调**（届时可作为独立 task）。

---

## 建议执行顺序

1. **P0.1**（脚本）+ **P3.2**（`-Werror` 清洁，顺手）——先让"GREEN 可复现、零 warning"。
2. **P3.1**（边角测试）——原型阶段把数值边界夯实，避免 P1 生产版踩同样的坑。
3. **P1.1→P1.2→P1.3**（生产版 ACL Tensor，adapter 式起步）——对接真实硬件。
4. **P2.1/2.2/2.3**（扩展）——按需排期，可并行。
5. **P1.4**（device 侧性能）——最后优化。
6. **P1.5**（host SIMD，备忘未决）——进 P1.4 时一并 benchmark 定夺；先看 normalize/softmax 是否比 argmax 更值钱，无 benchmark 不做。

> 阻塞点：P1 依赖 CANN（已就绪）；P2.1 依赖 `probs` 已返回（已就绪）；其余无外部阻塞。
