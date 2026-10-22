// ═══════════════════════════════════════════════════════════════════════════
// 对外 API 头文件（Public API）。用户可直接使用本文件声明的类型与别名。
// 本文件为对外类型层：TokenId / Prompt / SamplingParams / Result / 状态码 / buffer 与
// 模型构件 / 流式回调。Status 完整定义见 status.h。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "tensor/nd_tensor_buffer.h"  // ITensorBuffer（v0.12：替代旧 Tensor）

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cann_lm::engine {

// ────────────────── 标量与别名（对外 API）──────────────────

/// token id。底层词表大小决定宽度，默认 uint32_t；如需更大词表可在编译期切 uint64_t。对外 API。
using TokenId = uint32_t;

/// 模态标签。多模态 embedding 的模态标注，仅用于标识，不参与占位符匹配
/// （匹配按数组顺序，见需求 #1）。对外 API。
enum class ModalityTag {
    Image,
    Audio,
    Video,
    // 可扩展（不列 Unknown，未识别模态在加载期返回 Status::FAILURE 拒绝）
};

/// 底层提供的张量类型。v0.12：使用 backend ITensorBuffer。
/// Engine 层不感知其内部结构，只持有 shared_ptr 引用。
using ITensorBuffer = hiai::INDTensorBuffer;

/// 只读 buffer 视图（C++17 无 std::span，自研轻量）。对外 API。
struct Buffer {
    const void* data = nullptr;   ///< 数据指针
    size_t      size = 0;          ///< 字节数
};

/// 多模态 embedding 条目。对外 API。
struct ModalityEmbedding {
    ModalityTag tag;                            ///< 模态标注
    Buffer embedding;  ///< 非所有权数据视图，调用方须保证数据在 Prompt 使用期间存活
};

/// 输入 Prompt。单段文本 + 零或多个多模态 embedding（需求 #1）。对外 API。
struct Prompt {
    std::string text;                            ///< 单段文本，不含多轮 chat
    std::vector<ModalityEmbedding> embeddings;   ///< 可空，纯文本时为空
};

// ────────────────── 采样参数（对外 API）──────────────────

/// 采样旋钮（每请求级）。**只承载 sample 模块需要的参数**（与 core/sample 的
/// SamplerConfig 对齐）；终止语义（max_tokens / stop 集）归 RequestOptions，见下。
/// 对外 API。
struct SamplingParams {
    float    temperature        = 1.0f;   ///< 1.0 = 不缩放；0 经钳制退化为贪心（对齐 HF）
    int      top_k              = 0;      ///< 0 = 不启用 top-k
    float    top_p              = 1.0f;   ///< 1.0 = 不启用 top-p
    uint64_t seed               = 0;      ///< 0 = 不播种（每次随机）；非 0 = 可复现采样（仅概率采样时有意义）
    float    repetition_penalty = 1.0f;   ///< 1.0 = 关闭；>1 对已生成 token 降权（仅本请求生成部分）
};

/// 请求级终止参数（对外 API）。与采样旋钮分离；**长度未超限时，推理停止仅由
/// 本结构的 stop_token_ids / stop_strings 控制**（与 tokenizer 配置的 eos 无关）。
/// 停止字面量依赖 detokenize 不剥离 special token（skip_special_tokens 已从推理链路移除）。
struct RequestOptions {
    int                         max_tokens = 0;   ///< 0/负 = 默认上限 1024
    std::vector<TokenId>       stop_token_ids;   ///< 命中（含生成 token）→ StopToken 终止
    std::vector<std::string>   stop_strings;     ///< 命中文本尾部 → StopString 终止（截去停止子串）
};

// ────────────────── 状态码（forward declare，完整定义见 status.h）──────────────────

struct Status;

// ────────────────── 终止原因（对外 API）──────────────────

/// 终止原因（Result 的完成语义）。对外 API。
enum class FinishReason {
    MaxTokens,    ///< 达到 max_tokens（RequestOptions）
    StopToken,    ///< 命中 RequestOptions.stop_token_ids
    StopString,   ///< 命中 RequestOptions.stop_strings
    Canceled,     ///< 用户主动取消（需求 #2 c）
    Error,        ///< 推理过程出错（配合返回 Status::FAILURE，Result 可能含部分 token）
};

// ────────────────── 结果（对外 API）──────────────────

/// 推理结果。token id 序列 + 明文 + 终止原因。对外 API。
/// 错误信息由调用方通过返回的 Status 获取（不在 Result 内）。
struct Result {
    std::vector<TokenId> token_ids;                      ///< 生成的 token id 序列
    std::string          text;                           ///< 与 token_ids 对应的明文（pipeline 控制 detokenize，需求 #8）
    FinishReason         finish_reason = FinishReason::MaxTokens;  ///< 终止原因
};

// ────────────────── buffer 与模型构件（对外 API）──────────────────

/// 路径/buffer 双态源（图位与 meta_json 共用承载语义；纯字段，无方法——D-15）。
/// 取值规则（C-02 契约）：data 非空 → 用 buffer（path 忽略 + WARN）；
/// data 空且 path 非空 → 用路径；均空 → 未提供。对外 API。
struct ArtifactSource {
    std::string path;   ///< 空 = 未提供路径
    Buffer      data;   ///< 空 = 未提供 buffer
};

/// 单一模型构件（对外 API；SR20260903001 合并原 ModelArtifact/ModelArtifactPath）
/// prefill/decode 双图位收拢，D-10）。图与权重分离语义维持。
/// 权重路径 prefill/decode 对称拆分（docs/seperate_pd_weights.md §4.1）：
/// 共用 = 两字段填同一路径；空 = 图权重不分离（逐图独立判定，
/// 一空一非空 = 混合形态）。
struct ModelArtifact {
    ArtifactSource prefill_omc;  ///< prefill 图位
    ArtifactSource decode_omc;   ///< decode 图位
    std::string    prefill_weight;  ///< prefill 权重目录（空 = 图权重不分离）
    std::string    decode_weight;  ///< decode 权重目录（空 = 图权重不分离）
    ArtifactSource meta_json;    ///< 元信息+配置 json（构件级单份）
};

// ────────────────── 流式回调（对外 API）──────────────────

/// 流式回调。每步 Decode 产出 token 后触发（detokenize 后、stop 检测前，见需求 #8 / #3）。对外 API。
/// 回调在 pipeline 内部线程触发，用户回调内不应阻塞或调用 Engine 同步 API。
using StreamCallback = std::function<void(TokenId, std::string)>;

}  // namespace cann_lm::engine
