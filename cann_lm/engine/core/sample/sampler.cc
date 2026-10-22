// cann_lm/engine/core/sample/sampler.cc — Layer 3 Sampler 实现（C++ 原型）
//
// 6 步纯函数流水线（镜像 HuggingFace 采样顺序）。期望行为由 test/sampler_ut.cc 校验。
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

namespace cann_lm::sampler {

namespace {

// 数值稳定 softmax（减最大值；double 累加避免大词表下 float 累加丢小项）。处理 -inf → 0。
std::vector<float> softmax_stable(const std::vector<float>& logits) {
    float maxv = -INFINITY;
    for (float v : logits) maxv = std::max(maxv, v);
    if (!std::isfinite(maxv)) maxv = 0.0f;  // 全 -inf 兜底（不应发生）
    std::vector<float> e(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        double d = (double)logits[i] - (double)maxv;
        double ei = std::isfinite(d) ? std::exp(d) : 0.0;  // -inf → 0
        e[i] = (float)ei;
        sum += ei;
    }
    if (sum <= 0.0) {
        // 退化（理论上不会，除非全 -inf）：均匀
        std::fill(e.begin(), e.end(), 1.0f / (float)e.size());
        return e;
    }
    for (float& v : e) v = (float)((double)v / sum);
    return e;
}

int32_t argmax(const std::vector<float>& v) {
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = (int32_t)i;
    }
    return best;
}

}  // namespace

std::vector<float> Sampler::ApplyRepetitionPenalty(std::vector<float> logits,
                                                   const std::vector<int32_t>& generated_ids,
                                                   std::optional<float> penalty) {
    // ① 对已生成 token 降权：score<0 → *=p；score>=0 → /=p。镜像 HF
    // RepetitionPenaltyLogitsProcessor（从原始 logits 取分再写回，重复 token 等价单次）。
    if (!penalty || *penalty == 1.0f || generated_ids.empty()) return logits;
    float p = *penalty;
    std::set<int32_t> seen(generated_ids.begin(), generated_ids.end());
    for (int32_t id : seen) {
        if (id < 0 || (size_t)id >= logits.size()) continue;  // 越界保护
        float s = logits[id];  // 原始分（id 唯一，未被写过）
        logits[id] = (s < 0.0f) ? (s * p) : (s / p);
    }
    return logits;
}

std::vector<float> Sampler::ApplyTemperature(std::vector<float> logits, float temperature) {
    // ② logits / temperature；t<=0 钳到极小正值避免除零（退化为贪心）。
    float t = (temperature <= 0.0f) ? 1e-8f : temperature;
    for (float& v : logits) v /= t;
    return logits;
}

std::vector<float> Sampler::ApplyTopK(std::vector<float> logits, std::optional<int> top_k) {
    // ③ 只留概率最高 k 个，其余 -inf。k/None 或 >=vocab 或 <=0 → 跳过。
    if (!top_k) return logits;
    int k = *top_k;
    int n = (int)logits.size();
    if (k <= 0 || k >= n) return logits;
    // 找第 k 大的阈值：降序排序后取 sorted[k-1]
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    float threshold = sorted[k - 1];
    for (float& v : logits) {
        if (v < threshold) v = -INFINITY;
    }
    return logits;
}

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

std::vector<float> Sampler::ApplyTopP(std::vector<float> logits, std::optional<float> top_p) {
    // ④ 核采样：累计概率达 p 后截尾；保留最高概率项。p/None → 跳过。
    if (!top_p) return logits;
    float p = *top_p;
    int n = (int)logits.size();
    // 按降序排索引
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });
    // 排序后的 softmax + 累计
    std::vector<float> sorted_logits(n);
    for (int i = 0; i < n; ++i) sorted_logits[i] = logits[idx[i]];
    std::vector<float> probs = softmax_stable(sorted_logits);
    std::vector<char> remove(n, 0);
    float cum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cum += probs[i];
        if (cum > p) remove[i] = 1;
    }
    // shift 保留首个（最高概率项必留）
    for (int i = n - 1; i >= 1; --i) remove[i] = remove[i - 1];
    remove[0] = 0;
    for (int i = 0; i < n; ++i) {
        if (remove[i]) logits[idx[i]] = -INFINITY;
    }
    return logits;
}

std::vector<float> Sampler::Normalize(const std::vector<float>& logits) {
    // ⑤ softmax 归一化（数值稳定）。
    return softmax_stable(logits);
}

int32_t Sampler::Pick(const std::vector<float>& probs, bool do_sample, std::optional<uint64_t> seed) {
    // ⑥ do_sample=True→按概率 multinomial（inverse-CDF）；False→argmax。
    if (!do_sample) return argmax(probs);
    std::mt19937_64 gen;
    if (seed) {
        gen.seed(*seed);
    } else {
        gen.seed(std::random_device{}());
    }
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float u = uni(gen);
    float c = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        c += probs[i];
        if (u < c) return (int32_t)i;
    }
    return (int32_t)(probs.size() - 1);  // 浮点兜底
}

SamplerOutput Sampler::Sample(SamplerInputs in, const SamplerConfig& cfg) {
    // ①→②→③(precomputed 或 ApplyTopK)→④→⑤→⑥；in 按值传入，可 move in.logits 走零拷贝
    std::vector<float> x = ApplyRepetitionPenalty(std::move(in.logits), in.generated_ids, cfg.repetition_penalty);
    x = ApplyTemperature(std::move(x), cfg.temperature);
    // ③ TopK：precomputed 在场则复用（覆盖 cfg.top_k，NPU 已算过），否则自己算
    if (in.precomputed_topk && !in.precomputed_topk->token_ids.empty()) {
        x = ApplyPrecomputedTopK(std::move(x), in.precomputed_topk);
    } else {
        x = ApplyTopK(std::move(x), cfg.top_k);
    }
    x = ApplyTopP(std::move(x), cfg.top_p);
    std::vector<float> probs = Normalize(x);
    int32_t id = Pick(probs, cfg.do_sample, cfg.seed);
    return SamplerOutput{id, probs};
}

}  // namespace cann_lm::sampler
