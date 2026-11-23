// cann_local_lm/engine/core/component/sampler/sampler.h — L3 Sampler 组件
//
// 算法本体（6 步流水线，无状态静态纯函数，接口形态不变）：
// ①ApplyRepetitionPenalty → ②ApplyTemperature → ③TopK（precomputed 在场则复用，
// 覆盖 cfg.top_k）→ ④ApplyTopP → ⑤Normalize → ⑥Pick。①必须在 ③④ 之前（rep 改变
// top-k 截断集）；①②可交换（均为标量乘）。期望行为锚定：tests/ut/ut_sampler.cpp。
// R1 贪心快路径（perf_optimize.md §4.1）：do_sample=false 时 ⑤⑥ 被 argmax(x)
// 短路（softmax 单调保序，tie-break 同首个最大值下标），probs 空返回（惰性
// 产出——接入层只消费 next_token_id）；概率采样路径 ⑤⑥ 原样。
//
// 引擎接入层（成员 Sample，PipelineExecutor 后处理调用）与参考实现的差异，
// 对齐 docs/pipeline+engine_executor/SPEC.md §4.1.5（F-SAMPLE）：
//   - 输入是 ITensorBuffer（batched logits，按 slot 取行），非 std::vector<float>；
//   - 旋钮来自对外 SamplingParams（temperature/top_k/top_p/seed/repetition_penalty）。
//     rep penalty 读**本请求已生成** token（generated_ids 入参，
//     RunLoop 传 req->token_ids；不含 prompt/跨轮历史）；seed≠0 时 per-call 建 seeded
//     RNG（可复现，仅 do_sample 时有意义）。precomputed_topk 仍不对外（无来源通道）；
//   - do_sample 无对外开关，映射规则：temperature!=1 或 top_k>0 或 top_p∈(0,1) 任一
//     成立 → 概率采样，否则贪心；越界旋钮值（top_k<=0、top_p∉(0,1)）视为关闭且不计入
//     判定；temperature<=0 显式选择贪心（不进行温度缩放或概率采样）；repetition_penalty==1
//     视为关闭（贪心下仍生效——对齐 HF processors 语义）。
//   - prefill chunk（chunk_len>1）的 logits 行语义待真实后端落地时核对：自回归续写应
//     取 chunk 末位置而非 slot 0 对应首位置，中间 chunk 不应采样——当前 host 前向桩
//     不可达（外审 M-1 登记，届时在 RunLoop/接入层二选一处理）。
//
// 已知限制：logits 支持 FLOAT32 与 FLOAT16（fp16 行内展开为 float 再进算法层）。
#pragma once

#include "core/tensor_manager.h" // ITensorBuffer / TensorDesc / DataType（ModelConfig 已迁 model_executor.h，D-06）
#include "core/component/model_executor/model_executor.h" // ModelConfig（Configure 注入 vocab_size）
#include "infra/status.h"
#include "api/cpp/types.h" // SamplingParams / TokenId

#include <cstdint>
#include <optional>
#include <vector>

namespace cann_local_lm::engine {

// ────────────── 算法层类型 ──────────────

// 采样旋钮（不含 max_new_tokens——停止计数归上层 SamplingParams/Context）
struct SamplerConfig {
    float temperature = 1.0f;
    bool do_sample = false;
    std::optional<float> top_p;
    std::optional<int> top_k;
    std::optional<float> repetition_penalty; // >1 降权、1.0 关闭、<1 升权
    std::optional<uint64_t> seed;            // 给定则 per-call 建 seeded RNG（可复现）
};

// NPU 预算的 top-k 结果（数据输入，非旋钮）。给了就复用、跳过 ApplyTopK 重算
// （设备已算过 top-k 的加速路径）。token_ids 即"留下的 top-k token 下标集合"。
struct PrecomputedTopK {
    std::vector<int32_t> token_ids;
};

// 数据输入打包（旋钮在 SamplerConfig，数据在此——分离）。logits/generated_ids 按值
// 持有；调用方可 std::move 进来，Sample 按值取 in 后可 std::move(in.logits) 零拷贝。
struct SamplerInputs {
    std::vector<float> logits;
    std::vector<int32_t> generated_ids;
    std::optional<PrecomputedTopK> precomputed_topk;
};

// 返回：选中的 token + 归一化概率分布（和≈1；R1 贪心快路径下 probs 惰性
// 不物化——空向量，需分布者走 do_sample）
struct SamplerOutput {
    int32_t next_token_id = 0;
    std::vector<float> probs; // [vocab]；贪心快路径为空
};

// Sampler：算法层（静态纯函数，无需实例化）+ 引擎接入层（成员方法，pipeline 按值持有）。
class Sampler {
public:
    Sampler() = default;
    explicit Sampler(size_t vocab_size) : vocab_size_(vocab_size) {}

    /// 从 ModelConfig 注入 vocab_size（pipeline Load 期调用）。
    void Configure(const ModelConfig& cfg) { vocab_size_ = cfg.vocab_size; }

    // ── 引擎接入层：batched logits 按 slot 取行 → 映射 SamplingParams → 算法层。──
    // generated_ids：本请求已生成 token（rep penalty 输入；首步为空）。不含 prompt 历史。
    // 成功时 *out 为采样 token。失败码：
    //   INVALID_ARGUMENT —— !out / slot<0 / dtype 非 FLOAT32/FLOAT16 / 字节非元素对齐 /
    //                       size==0 / dims 异常 / slot 越界；
    //   FAILURE          —— GetData()==nullptr（logits 缺失）/ dims 缺失且 slot>0
    //                       （无法安全推导行宽）。
    Status Sample(ITensorBuffer& batch_logits, int slot, const SamplingParams& params,
                  const std::vector<TokenId>& generated_ids, TokenId* out);

    // ── 算法层（对外唯一入口 + 6 步纯函数；公开以供独立单测。均不改输入语义：
    //    返回新 vector，入参按值 move）──
    // logits 已含 softcap（模型 forward 内部做，Sampler 不重复）；generated_ids 首步可为空。
    static SamplerOutput Sample(SamplerInputs in, const SamplerConfig& cfg);
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

private:
    size_t vocab_size_ = 0; // ModelConfig 注入（Configure，pipeline Load 期）
};

} // namespace cann_local_lm::engine
