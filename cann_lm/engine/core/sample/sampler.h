// cann_lm/engine/core/sample/sampler.h — Layer 3 Sampler（C++ 原型）
//
// 设计：Sampler 无状态 → 对外是**静态方法**（C++ 风格），不实例化、不要 OOP 多态基类。
// 数据用 STL 容器（logits/probs = std::vector<float>，batch=1；token id = int32_t，词表<2^31 够用），
// 无外部依赖、单 TU 可编译可测。生产版可改取 CANN ACL Tensor（见 SPEC §8.4），接口形态不变。
//
// 契约见 docs/SPEC.md（v0.8）§8。6 步流水线：
//   ① ApplyRepetitionPenalty → ② ApplyTemperature → ③ TopK → ④ ApplyTopP → ⑤ Normalize → ⑥ Pick
//   其中 ③ TopK = 若给了 NPU 预算 top-k（precomputed_topk）则 ApplyPrecomputedTopK 复用、
//   否则 ApplyTopK 自己算。两者互斥：precomputed 在场则覆盖 cfg.top_k。
// ①必须在 ③④ 之前（rep 与 top_k/top_p 不可交换：rep 改变 top-k 成员/截断集）；
// ①与 ②（temperature）可交换（均为标量乘，顺序无影响），放最前是 HF 约定。
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace cann_lm::sampler {

// 采样旋钮（不含 max_new_tokens——停止计数归上层 Context）
struct SamplerConfig {
    float temperature = 1.0f;
    bool do_sample = false;
    std::optional<float> top_p;
    std::optional<int> top_k;
    std::optional<float> repetition_penalty; // >1 降权、1.0 关闭、<1 升权
    std::optional<uint64_t> seed;            // 给定则 per-call 建 seeded RNG（可复现）
};

// NPU 预算的 top-k 结果（数据输入，非旋钮）。
// 给了就复用、跳过 ApplyTopK 重算（设备已算过 top-k 的加速路径）。token_ids 即"留下的 top-k token 下标集合"。
struct PrecomputedTopK {
    std::vector<int32_t> token_ids;
};

// 数据输入打包（旋钮在 SamplerConfig，数据在此——分离，见 §8.1 理路）。
// logits/generated_ids 按值持有；调用方可 std::move 进来，Sample 按值取 in 后可 std::move(in.logits) 走零拷贝路径。
struct SamplerInputs {
    std::vector<float> logits;
    std::vector<int32_t> generated_ids;
    std::optional<PrecomputedTopK> precomputed_topk;
};

// 返回：选中的 token + 归一化概率分布（和≈1）
struct SamplerOutput {
    int32_t next_token_id = 0;
    std::vector<float> probs; // [vocab]
};

// 无状态采样器：全部静态，无需实例化。
class Sampler {
public:
    // 对外唯一入口。数据打包进 SamplerInputs，旋钮在 cfg。按值取 in（pipeline 可 move in.logits）。
    // logits 已含 softcap（模型 forward 内部做，Sampler 不重复）；generated_ids 首步可为空 {}。
    static SamplerOutput Sample(SamplerInputs in, const SamplerConfig& cfg);

    // 内部 6 步纯函数（公开以便 AC9 独立单测）。均返回新的 logits/probs，不改输入。
    static std::vector<float> ApplyRepetitionPenalty(std::vector<float> logits,
                                                     const std::vector<int32_t>& generated_ids,
                                                     std::optional<float> penalty);
    static std::vector<float> ApplyTemperature(std::vector<float> logits, float temperature);
    static std::vector<float> ApplyTopK(std::vector<float> logits, std::optional<int> top_k);
    static std::vector<float> ApplyPrecomputedTopK(std::vector<float> logits,
                                                   const std::optional<PrecomputedTopK>& precomputed);
    static std::vector<float> ApplyTopP(std::vector<float> logits, std::optional<float> top_p);
    static std::vector<float> Normalize(const std::vector<float>& logits);
    static int32_t Pick(const std::vector<float>& probs, bool do_sample, std::optional<uint64_t> seed);
};

} // namespace cann_lm::sampler
