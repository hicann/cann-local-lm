// cann_local_lm/engine/core/component/sampler/sampler.cpp — L3 Sampler 实现
//
// 算法层：6 步纯函数流水线，期望行为见 tests/ut/ut_sampler.cpp；引擎接入层：
// ITensorBuffer 行提取 + SamplingParams → SamplerConfig 映射（规则见 sampler.h 头注释）。
#include "core/component/sampler/sampler.h"

#include "utils/data_type_utils.h" // HalfBitsToFloat（FLOAT16 logits → float32）

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

namespace cann_local_lm::engine {

namespace {

// 数值稳定 softmax（减最大值；double 累加避免大词表下 float 累加丢小项）。处理 -inf → 0。
std::vector<float> softmax_stable(const std::vector<float>& logits)
{
    float maxv = -INFINITY;
    for (float v : logits)
        maxv = std::max(maxv, v);
    if (!std::isfinite(maxv))
        maxv = 0.0f; // 全 -inf 兜底（不应发生）
    std::vector<float> e(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        double d = (double)logits[i] - (double)maxv;
        double ei = std::isfinite(d) ? std::exp(d) : 0.0; // -inf → 0
        e[i] = (float)ei;
        sum += ei;
    }
    if (sum <= 0.0) {
        // 退化（理论上不会，除非全 -inf）：均匀
        std::fill(e.begin(), e.end(), 1.0f / (float)e.size());
        return e;
    }
    for (float& v : e)
        v = (float)((double)v / sum);
    return e;
}

int32_t argmax(const std::vector<float>& v)
{
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best])
            best = (int32_t)i;
    }
    return best;
}

} // namespace

// ────────────── 算法层 ──────────────

std::vector<float> Sampler::ApplyRepetitionPenalty(std::vector<float> logits, const std::vector<int32_t>& generated_ids,
                                                   std::optional<float> penalty)
{
    // ① 对已生成 token 降权：score<0 → *=p；score>=0 → /=p。镜像 HF
    // RepetitionPenaltyLogitsProcessor（从原始 logits 取分再写回，重复 token 等价单次）。
    if (!penalty || *penalty == 1.0f || generated_ids.empty())
        return logits;
    float p = *penalty;
    std::set<int32_t> seen(generated_ids.begin(), generated_ids.end());
    for (int32_t id : seen) {
        if (id < 0 || (size_t)id >= logits.size())
            continue;         // 越界保护
        float s = logits[id]; // 原始分（id 唯一，未被写过）
        logits[id] = (s < 0.0f) ? (s * p) : (s / p);
    }
    return logits;
}

std::vector<float> Sampler::ApplyTemperature(std::vector<float> logits, float temperature)
{
    // ② logits / temperature；t<=0 钳到极小正值避免除零（退化为贪心）。
    float t = (temperature <= 0.0f) ? 1e-8f : temperature;
    for (float& v : logits)
        v /= t;
    return logits;
}

std::vector<float> Sampler::ApplyTopK(std::vector<float> logits, std::optional<int> top_k)
{
    // ③ 只留概率最高 k 个，其余 -inf。k/None 或 >=vocab 或 <=0 → 跳过。
    if (!top_k)
        return logits;
    int k = *top_k;
    int n = (int)logits.size();
    if (k <= 0 || k >= n)
        return logits;
    // 找第 k 大的阈值：降序排序后取 sorted[k-1]
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    float threshold = sorted[k - 1];
    for (float& v : logits) {
        if (v < threshold)
            v = -INFINITY;
    }
    return logits;
}

std::vector<float> Sampler::ApplyPrecomputedTopK(std::vector<float> logits,
                                                 const std::optional<PrecomputedTopK>& precomputed)
{
    // ③' 复用 NPU 预算的 top-k 集合：不在集合里的置 -inf（设备已算过 top-k 的加速路径）。
    // 没给/空 → no-op（原样返回）。
    if (!precomputed || precomputed->token_ids.empty())
        return logits;
    std::set<int32_t> keep(precomputed->token_ids.begin(), precomputed->token_ids.end());
    for (size_t i = 0; i < logits.size(); ++i) {
        if (keep.find((int32_t)i) == keep.end())
            logits[i] = -INFINITY;
    }
    return logits;
}

std::vector<float> Sampler::ApplyTopP(std::vector<float> logits, std::optional<float> top_p)
{
    // ④ 核采样：累计概率达 p 后截尾；保留最高概率项。p/None → 跳过。
    if (!top_p)
        return logits;
    float p = *top_p;
    int n = (int)logits.size();
    // 按降序排索引
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    // 排序后的 softmax + 累计
    std::vector<float> sorted_logits(n);
    for (int i = 0; i < n; ++i)
        sorted_logits[i] = logits[idx[i]];
    std::vector<float> probs = softmax_stable(sorted_logits);
    std::vector<char> remove(n, 0);
    float cum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cum += probs[i];
        if (cum > p)
            remove[i] = 1;
    }
    // shift 保留首个（最高概率项必留）
    for (int i = n - 1; i >= 1; --i)
        remove[i] = remove[i - 1];
    remove[0] = 0;
    for (int i = 0; i < n; ++i) {
        if (remove[i])
            logits[idx[i]] = -INFINITY;
    }
    return logits;
}

std::vector<float> Sampler::Normalize(const std::vector<float>& logits)
{
    // ⑤ softmax 归一化（数值稳定）。
    return softmax_stable(logits);
}

int32_t Sampler::Pick(const std::vector<float>& probs, bool do_sample, std::optional<uint64_t> seed)
{
    // ⑥ do_sample=True→按概率 multinomial（inverse-CDF）；False→argmax。
    if (!do_sample)
        return argmax(probs);
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
        if (u < c)
            return (int32_t)i;
    }
    return (int32_t)(probs.size() - 1); // 浮点兜底
}

SamplerOutput Sampler::Sample(SamplerInputs in, const SamplerConfig& cfg)
{
    // ①→②→③(precomputed 或 ApplyTopK)→④→⑤→⑥；in 按值传入，可 move in.logits 走零拷贝
    std::vector<float> x = ApplyRepetitionPenalty(std::move(in.logits), in.generated_ids, cfg.repetition_penalty);
    const bool zero_temperature = cfg.temperature <= 0.0f;
    if (!zero_temperature)
        x = ApplyTemperature(std::move(x), cfg.temperature);
    // ③ TopK：precomputed 在场则复用（覆盖 cfg.top_k，NPU 已算过），否则自己算
    if (in.precomputed_topk && !in.precomputed_topk->token_ids.empty()) {
        x = ApplyPrecomputedTopK(std::move(x), in.precomputed_topk);
    } else {
        x = ApplyTopK(std::move(x), cfg.top_k);
    }
    if (!zero_temperature)
        x = ApplyTopP(std::move(x), cfg.top_p);
    // 贪心免归一化快路径（R1，perf_optimize.md §4.1）：softmax 单调保序 ⇒
    // argmax(x) ≡ argmax(Normalize(x))；tie-break 语义逐字保留（argmax 首个
    // 最大值下标，:40-46 不动）。probs 惰性产出——引擎接入层只消费
    // next_token_id（:244），probs 无消费者，免 262K exp + double 累加 +
    // 262K 概率分配。正温度 do_sample=true 路径保留 softmax/multinomial。
    if (!cfg.do_sample || zero_temperature) {
        return SamplerOutput{argmax(x), std::vector<float>()};
    }
    std::vector<float> probs = Normalize(x);
    int32_t id = Pick(probs, cfg.do_sample, cfg.seed);
    return SamplerOutput{id, std::move(probs)};
}

// ────────────── 引擎接入层 ──────────────

Status Sampler::Sample(ITensorBuffer& batch_logits, int slot, const SamplingParams& params,
                       const std::vector<TokenId>& generated_ids, TokenId* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    if (slot < 0)
        return Status::INVALID_ARGUMENT;

    // logits 缺失（pipeline 已保证非空才调，此为防御；契约与 TC203 一致）
    const void* data = batch_logits.GetData();
    if (!data)
        return Status::FAILURE;

    // dtype 支持 FLOAT32 与 FLOAT16（模型输出常为 fp16，此处展开为 float 再进算法层）
    const DataType dtype = batch_logits.GetTensorDesc().dataType;
    const size_t elem_bytes = (dtype == DataType::FLOAT16) ? sizeof(uint16_t) : sizeof(float);
    if (dtype != DataType::FLOAT32 && dtype != DataType::FLOAT16)
        return Status::INVALID_ARGUMENT;
    const size_t bytes = batch_logits.GetSize();
    if (bytes == 0 || bytes % elem_bytes != 0)
        return Status::INVALID_ARGUMENT;
    const size_t elems = bytes / elem_bytes;

    // 行宽推导：优先 desc.dims（row-major，最后一维 = vocab，前缀积 = 行容量）；
    // dims 缺失（host 回退缓冲常见）时仅 slot==0 可整块视作单行。
    const TensorDesc& desc = batch_logits.GetTensorDesc();
    size_t vocab = 0;
    size_t rows = 0;
    if (!desc.dims.empty()) {
        if (desc.dims.back() <= 0)
            return Status::INVALID_ARGUMENT;
        int64_t r = 1;
        for (size_t i = 0; i + 1 < desc.dims.size(); ++i) {
            if (desc.dims[i] <= 0)
                return Status::INVALID_ARGUMENT;
            if (r > INT64_MAX / desc.dims[i])
                return Status::INVALID_ARGUMENT; // 前缀积溢出守卫
            r *= desc.dims[i];
        }
        vocab = static_cast<size_t>(desc.dims.back());
        rows = static_cast<size_t>(r);
    } else {
        if (slot > 0)
            return Status::FAILURE; // 无 dims 无法安全切行
        vocab = elems;
        rows = 1;
    }
    if (vocab > elems || rows > elems / vocab)
        return Status::INVALID_ARGUMENT; // desc 与缓冲不符（除法式防溢出）
    if (static_cast<size_t>(slot) >= rows)
        return Status::INVALID_ARGUMENT; // slot 越界

    // The tensor row may be padded beyond the model vocabulary. Keep its
    // physical width for addressing, but never sample a padding token.
    const size_t candidates = vocab_size_ == 0 ? vocab : vocab_size_;
    if (candidates > vocab)
        return Status::INVALID_ARGUMENT;

    // SamplingParams → SamplerConfig（映射规则见 sampler.h 头注释）。前移到行
    // 提取之前：贪心 + 无 top-k/p 时整行向量物化被免掉（R1 配套）。
    // 默认参数 → 贪心；任一采样旋钮非默认 → 概率采样（temp<=0 显式选择贪心）。
    // repetition_penalty / seed：经 SamplingParams 对外（F-SAMPLE），
    // 关闭值 = 1.0 及一切非法值（<=0/NaN——C ABI 零初始化得 0.0，须与 top_k/top_p
    // 同按"越界视为关闭"处理，否则 p=0 除零污染分布）；precomputed_topk 仍不对外。
    SamplerConfig cfg;
    cfg.temperature = params.temperature;
    if (params.top_k > 0)
        cfg.top_k = params.top_k;
    if (params.top_p > 0.0f && params.top_p < 1.0f)
        cfg.top_p = params.top_p;
    if (params.repetition_penalty > 0.0f && params.repetition_penalty != 1.0f) {
        cfg.repetition_penalty = params.repetition_penalty;
    }
    if (params.seed != 0)
        cfg.seed = params.seed;
    cfg.do_sample = params.temperature != 1.0f || cfg.top_k.has_value() || cfg.top_p.has_value();

    // ── R1 配套：行免中转贪心（§4.1）──在源 dtype 位型上直接搜第一个最大值
    // 下标（fp16 逐元素 HalfBitsToFloat 比较，先转后比，与 fp32 行同 tie-break），
    // 免 262K float 物化 + 拷贝往返。走此短路的前提 = 贪心（!do_sample 已蕴含
    // temperature==1.0f，见上 :257 映射）+ 无重复惩罚（无已生成上下文可降权）；
    // 其余（do_sample / rep penalty）进 6 步算法层全量路径——softmax/
    // multinomial 行为不变。fp16 逐位比较不可行（符号位/指数位序 ≠ 数值序），
    // 故转换开销保留在扫描内联里，与向量物化版相比仍免一次整行分配。
    if (!cfg.do_sample && !cfg.repetition_penalty.has_value()) {
        const size_t slot_base = static_cast<size_t>(slot) * vocab;
        int32_t best = 0;
        if (dtype == DataType::FLOAT16) {
            const uint16_t* row = static_cast<const uint16_t*>(data) + slot_base;
            float bestv = HalfBitsToFloat(row[0]);
            for (size_t i = 1; i < candidates; ++i) {
                const float v = HalfBitsToFloat(row[i]);
                if (v > bestv) {
                    bestv = v;
                    best = static_cast<int32_t>(i);
                }
            }
        } else {
            const float* row = static_cast<const float*>(data) + slot_base;
            float bestv = row[0];
            for (size_t i = 1; i < candidates; ++i) {
                if (row[i] > bestv) {
                    bestv = row[i];
                    best = static_cast<int32_t>(i);
                }
            }
        }
        *out = static_cast<TokenId>(best);
        return Status::SUCCESS;
    }

    // 取 slot 行（拷贝进算法层 vector；后续如需零拷贝 Span/Tensor 变体，
    // 待真实通路接入后按需替换）
    SamplerInputs in;
    if (dtype == DataType::FLOAT16) {
        const uint16_t* row = static_cast<const uint16_t*>(data) + static_cast<size_t>(slot) * vocab;
        in.logits.resize(candidates);
        for (size_t i = 0; i < candidates; ++i)
            in.logits[i] = HalfBitsToFloat(row[i]);
    } else {
        const float* row = static_cast<const float*>(data) + static_cast<size_t>(slot) * vocab;
        in.logits.assign(row, row + candidates);
    }
    // generated_ids 剪枝（R1 配套）：默认零拷贝——算法层入参面是
    // const vector<int32_t>&（ApplyRepetitionPenalty），仅重复惩罚真正启用
    // 且非空时才按值物化（贪心/无惩罚轮次免上下文整拷贝）。
    if (cfg.repetition_penalty.has_value() && !generated_ids.empty()) {
        in.generated_ids.assign(generated_ids.begin(), generated_ids.end());
    }

    SamplerOutput res = Sample(std::move(in), cfg);
    *out = static_cast<TokenId>(res.next_token_id); // 算法层返回 [0, vocab) 下标
    return Status::SUCCESS;
}

} // namespace cann_local_lm::engine
