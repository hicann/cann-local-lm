// ═══════════════════════════════════════════════════════════════════════════
// CANN-LM Engine C API — Stable ABI boundary for cross-language binding.
//
// ⚠ ABI CHANGELOG（pre-release，布局尚无兼容承诺）：
//   2026-09-07 — SamplingParams 重排（max_tokens/stop_token_ids/stop_strings 移出，
//               新增 seed/repetition_penalty，字段顺序变更）；新增 RequestOptions
//               结构；ExecutorGenerate/GenerateAsync 各增 options 参数。
//               外部绑定（Python/Rust 等）须按本版头文件重新生成。
//
// This header is the sole C entry point to the engine. It wraps the C++ API
// (EngineExecutor / Context / RequestHandle) behind extern "C" functions with
// opaque handles, providing a compiler-independent ABI.
//
// Design constraints:
//   - No C++ standard library headers included (only <stddef.h> / <stdint.h>)
//   - No exceptions cross the boundary (all C++ → C calls are try/catch-guarded)
//   - Memory: Result.token_ids and Result.text are library-allocated (malloc);
//     caller frees via CANN_LMEngine_ResultDestroy().
//   - Naming: CANN_LMEngine_<Type><Verb> (e.g. CANN_LMEngine_ExecutorCreate)
//   - Status codes are int values 1:1 with C++ Status constants
//
// See docs/c_api/ for SPEC and DESIGN.
// ═══════════════════════════════════════════════════════════════════════════
#ifndef CANN_LM_ENGINE_C_API_H
#define CANN_LM_ENGINE_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ────────────────────────── Status Codes ──────────────────────────
// int values are 1:1 with cann_lm::engine::Status constants (infra/status.h).

typedef int CANN_LMEngine_Status;
#define CANN_LM_ENGINE_SUCCESS            0
#define CANN_LM_ENGINE_FAILURE            1
#define CANN_LM_ENGINE_INVALID_ARGUMENT    2
#define CANN_LM_ENGINE_INVALID_STATE       3
#define CANN_LM_ENGINE_NOT_FOUND           4
#define CANN_LM_ENGINE_CANCELED            5

// ────────────────────────── Scalar Types ──────────────────────────

typedef uint32_t CANN_LMEngine_TokenId;
typedef uint64_t CANN_LMEngine_ContextId;

// ────────────────────────── Enums ──────────────────────────
// Values match C++ enum class counterparts exactly (see types.h / request_handle.h).

typedef enum {
    CANN_LM_ENGINE_FINISH_MAX_TOKENS  = 0,
    CANN_LM_ENGINE_FINISH_STOP_TOKEN  = 1,
    CANN_LM_ENGINE_FINISH_STOP_STRING = 2,
    CANN_LM_ENGINE_FINISH_CANCELED    = 3,
    CANN_LM_ENGINE_FINISH_ERROR       = 4
} CANN_LMEngine_FinishReason;

typedef enum {
    CANN_LM_ENGINE_REQUEST_QUEUED      = 0,
    CANN_LM_ENGINE_REQUEST_PREFILLING  = 1,
    CANN_LM_ENGINE_REQUEST_DECODING     = 2,
    CANN_LM_ENGINE_REQUEST_COMPLETED   = 3,
    CANN_LM_ENGINE_REQUEST_CANCELED    = 4,
    CANN_LM_ENGINE_REQUEST_FAILED       = 5
} CANN_LMEngine_RequestStatus;

// ────────────────────────── Opaque Handles ──────────────────────────
// Callers receive these from Create functions and must pass them to Destroy.

typedef struct CANN_LMEngine_Executor CANN_LMEngine_Executor;
typedef struct CANN_LMEngine_Request  CANN_LMEngine_Request;

// ────────────────────────── POD Structs ──────────────────────────

typedef struct {
    const void* data;
    size_t      size;
} CANN_LMEngine_Buffer;

/// SR20260903001 合并构件：prefill/decode 双图位 + prefill/decode 双权重路径
/// （权重拆分：docs/seperate_pd_weights.md §4.1）+ meta_json，
/// 双态字段路径（const char*，NULL = 未提供）/buffer（data==NULL = 未提供），
/// 取值 buffer 优先（路径被忽略）。
typedef struct {
    const char*            prefill_omc_path;
    CANN_LMEngine_Buffer   prefill_omc_buf;
    const char*            decode_omc_path;
    CANN_LMEngine_Buffer   decode_omc_buf;
    const char*            prefill_weight;    ///< prefill 权重路径（NULL → ""；图与权重分离）
    const char*            decode_weight;     ///< decode 权重路径（NULL → ""；图与权重分离）
    const char*            meta_json_path;
    CANN_LMEngine_Buffer   meta_json_buf;
} CANN_LMEngine_ModelArtifact;

typedef struct {
    int                    default_chunk_size;
    int                    max_concurrent_prefill;
    int                    max_batch_size_override;
    /// tokenizer 算法："BPE" | "Qwen" | "SPM"；NULL/空 = 默认 BPE。
    /// 未知值记 WARN 后回退 BPE（选择权在此，取代通路上原硬编码 BPE）。
    const char*            tokenizer_type;
    /// tokenizer.json 词表产物路径；NULL/空 = 不经 options 供给
    /// （回退 meta_json tokenizer_config.tokenizer_file，见 input_config.md §4）。
    /// 非空时优先于 meta_json；指向不可加载文件 → tokenizer 置空（Generate 快速失败）。
    const char*            tokenizer_path;
} CANN_LMEngine_EngineOptions;

/// 采样旋钮（镜像 C++ SamplingParams）：只承载 sample 模块需要的参数；
/// 终止语义（max_tokens / stop 集）归 CANN_LMEngine_RequestOptions。
typedef struct {
    float    temperature;           ///< 1.0 = 不缩放；0 退化为贪心
    int      top_k;                 ///< 0 = 不启用
    float    top_p;                 ///< 1.0 = 不启用
    uint64_t seed;                  ///< 0 = 不播种（每次随机）；非 0 = 可复现采样
    float    repetition_penalty;    ///< 1.0 = 关闭；>1 对已生成 token 降权
} CANN_LMEngine_SamplingParams;

/// 请求级终止参数（镜像 C++ RequestOptions）。长度未超限时，停止仅由
/// stop_token_ids / stop_strings 控制（与 tokenizer 配置的 eos 无关）。
typedef struct {
    int                            max_tokens;         ///< 0/负 = 默认上限 1024
    const CANN_LMEngine_TokenId*   stop_token_ids;     ///< NULL = 无
    size_t                         stop_token_ids_count;
    const char* const*             stop_strings;       ///< NULL = 无
    size_t                         stop_strings_count;
} CANN_LMEngine_RequestOptions;

typedef struct {
    CANN_LMEngine_TokenId*      token_ids;
    size_t                      token_ids_count;
    char*                       text;
    CANN_LMEngine_FinishReason finish_reason;
} CANN_LMEngine_Result;

// ────────────────────────── Stream Callback ──────────────────────────
// Called from the pipeline internal thread after each token is decoded.
// |text| and |text_len| are valid only for the duration of the callback call.
// Do not block or call Engine APIs from within the callback.

typedef void (*CANN_LMEngine_StreamCallback)(
    CANN_LMEngine_TokenId  token,
    const char*            text,
    size_t                 text_len,
    void*                  user_data
);

// ────────────────────────── Executor Lifecycle ──────────────────────────

/// 单一构造入口（SR20260903001：合并原 ExecutorCreate + ExecutorCreateFromBuffer）。
/// 模型图位（omc path/buffer）已配置而加载失败 → 构造内终止初始化并完成卸载，
/// 返回失败码且 *out 保持 NULL（2026-09-22）；图位全空（无模型形态）构造成功，
/// 生命周期 API 可用、推理报错。
CANN_LMEngine_Status CANN_LMEngine_ExecutorCreate(
    const CANN_LMEngine_ModelArtifact*   artifact,
    const CANN_LMEngine_EngineOptions*   options,
    CANN_LMEngine_Executor**             out);

void CANN_LMEngine_ExecutorDestroy(CANN_LMEngine_Executor* executor);

// ────────────────────────── Context Management ──────────────────────────

CANN_LMEngine_Status CANN_LMEngine_ContextCreate(
    CANN_LMEngine_Executor*    executor,
    CANN_LMEngine_ContextId*   out);

CANN_LMEngine_Status CANN_LMEngine_ContextDestroy(
    CANN_LMEngine_Executor*   executor,
    CANN_LMEngine_ContextId   context_id);

CANN_LMEngine_Status CANN_LMEngine_ContextCancel(
    CANN_LMEngine_Executor*   executor,
    CANN_LMEngine_ContextId   context_id);

// ────────────────────────── Inference ──────────────────────────

CANN_LMEngine_Status CANN_LMEngine_ExecutorGenerate(
    CANN_LMEngine_Executor*               executor,
    const char*                           prompt_text,
    CANN_LMEngine_ContextId              context_id,
    const CANN_LMEngine_SamplingParams*   params,
    const CANN_LMEngine_RequestOptions*   options,   ///< NULL = 全默认
    CANN_LMEngine_Result*                 out);

CANN_LMEngine_Status CANN_LMEngine_ExecutorGenerateAsync(
    CANN_LMEngine_Executor*               executor,
    const char*                           prompt_text,
    CANN_LMEngine_ContextId              context_id,
    const CANN_LMEngine_SamplingParams*   params,
    const CANN_LMEngine_RequestOptions*   options,   ///< NULL = 全默认
    CANN_LMEngine_Request**               out,
    CANN_LMEngine_StreamCallback          callback,
    void*                                 user_data);

// ────────────────────────── Request Handle ──────────────────────────

CANN_LMEngine_RequestStatus CANN_LMEngine_RequestGetStatus(
    const CANN_LMEngine_Request* request);

int CANN_LMEngine_RequestIsDone(
    const CANN_LMEngine_Request* request);

CANN_LMEngine_Status CANN_LMEngine_RequestCancel(
    CANN_LMEngine_Request* request);

CANN_LMEngine_Status CANN_LMEngine_RequestWait(
    CANN_LMEngine_Request*  request,
    CANN_LMEngine_Result*  out);

CANN_LMEngine_Status CANN_LMEngine_RequestTryGetResult(
    const CANN_LMEngine_Request*  request,
    CANN_LMEngine_Result*         out);

void CANN_LMEngine_RequestDestroy(CANN_LMEngine_Request* request);

// ────────────────────────── Result Memory Management ──────────────────────────
// Frees token_ids and text pointers inside |result| (set to NULL after free).
// Calling on a zero-initialized or already-destroyed Result is a safe no-op.

void CANN_LMEngine_ResultDestroy(CANN_LMEngine_Result* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CANN_LM_ENGINE_C_API_H
