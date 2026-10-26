// ═══════════════════════════════════════════════════════════════════════════
// CANN-LM Engine C API implementation.
//
// Wraps the C++ API (EngineExecutor / Context / RequestHandle) behind
// extern "C" functions with opaque handles. All C++ → C boundary crossings
// are try/catch-guarded; exceptions never propagate across the C ABI.
//
// Memory convention: Result.token_ids and Result.text are malloc'd by the
// library and freed by the caller via CANN_LMEngine_ResultDestroy().
// ═══════════════════════════════════════════════════════════════════════════
#include "api/c/c_api.h"

#include "api/cpp/engine_executor.h"
#include "api/cpp/request_handle.h"
#include "api/cpp/types.h"
#include "core/context.h"
#include "infra/log.h"
#include "infra/status.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ────────────────────────── Opaque Handle Definitions ──────────────────────────

struct CANN_LMEngine_Executor {
    std::unique_ptr<cann_lm::engine::EngineExecutor> ee;
    std::unordered_map<uint64_t, cann_lm::engine::Context> contexts;
};

struct CANN_LMEngine_Request {
    cann_lm::engine::RequestHandle handle;
};

// ────────────────────────── Helpers ──────────────────────────

static int status_to_c(cann_lm::engine::Status s) { return s.code; }

/// C ABI 异常兜底：记录异常后统一转 FAILURE（异常不跨边界，细节进日志 C11）。
static int log_exception(const char* func, const char* what)
{
    CANN_LLM_LOGE("c_api.exception: %s threw (%s)", func, what);
    return CANN_LM_ENGINE_FAILURE;
}

/// C-07：合并 C 构件 → C++ ModelArtifact（双态字段一一映射；NULL → 空）
static cann_lm::engine::ModelArtifact to_cpp_artifact(const CANN_LMEngine_ModelArtifact* a)
{
    cann_lm::engine::ModelArtifact art;
    art.prefill_omc.path = a->prefill_omc_path ? a->prefill_omc_path : "";
    art.prefill_omc.data = {a->prefill_omc_buf.data, a->prefill_omc_buf.size};
    art.decode_omc.path = a->decode_omc_path ? a->decode_omc_path : "";
    art.decode_omc.data = {a->decode_omc_buf.data, a->decode_omc_buf.size};
    art.prefill_weight = a->prefill_weight ? a->prefill_weight : "";
    art.decode_weight = a->decode_weight ? a->decode_weight : "";
    art.meta_json.path = a->meta_json_path ? a->meta_json_path : "";
    art.meta_json.data = {a->meta_json_buf.data, a->meta_json_buf.size};
    return art;
}

static cann_lm::engine::EngineOptions to_cpp_options(const CANN_LMEngine_EngineOptions* opts)
{
    cann_lm::engine::EngineOptions o;
    if (opts) {
        o.default_chunk_size = opts->default_chunk_size;
        o.max_concurrent_prefill = opts->max_concurrent_prefill;
        o.max_batch_size_override = opts->max_batch_size_override;
        o.tokenizer_type = opts->tokenizer_type ? opts->tokenizer_type : "";
        o.tokenizer_path = opts->tokenizer_path ? opts->tokenizer_path : "";
    }
    return o;
}

static cann_lm::engine::SamplingParams to_cpp_params(const CANN_LMEngine_SamplingParams* p)
{
    cann_lm::engine::SamplingParams s;
    if (p) {
        s.temperature = p->temperature;
        s.top_k = p->top_k;
        s.top_p = p->top_p;
        s.seed = p->seed;
        s.repetition_penalty = p->repetition_penalty;
    }
    return s;
}

/// C-07：请求级终止参数 → C++ RequestOptions（NULL = 全默认）
static cann_lm::engine::RequestOptions to_cpp_request_options(const CANN_LMEngine_RequestOptions* o)
{
    cann_lm::engine::RequestOptions ro;
    if (o) {
        ro.max_tokens = o->max_tokens;
        if (o->stop_token_ids && o->stop_token_ids_count > 0) {
            ro.stop_token_ids.assign(o->stop_token_ids, o->stop_token_ids + o->stop_token_ids_count);
        }
        if (o->stop_strings && o->stop_strings_count > 0) {
            for (size_t i = 0; i < o->stop_strings_count; i++) {
                if (o->stop_strings[i]) {
                    ro.stop_strings.emplace_back(o->stop_strings[i]);
                }
            }
        }
    }
    return ro;
}

static cann_lm::engine::StreamCallback wrap_callback(CANN_LMEngine_StreamCallback cb, void* user_data)
{
    if (!cb) {
        return {};
    }
    return [cb, user_data](cann_lm::engine::TokenId token, const std::string& text) {
        cb(static_cast<CANN_LMEngine_TokenId>(token), text.c_str(), text.size(), user_data);
    };
}

static void fill_c_result(const cann_lm::engine::Result& cpp, CANN_LMEngine_Result* out)
{
    out->token_ids_count = cpp.token_ids.size();
    if (out->token_ids_count > 0) {
        out->token_ids = static_cast<CANN_LMEngine_TokenId*>(
            std::malloc(out->token_ids_count * sizeof(CANN_LMEngine_TokenId)));
        if (out->token_ids) {
            std::memcpy(out->token_ids, cpp.token_ids.data(), out->token_ids_count * sizeof(CANN_LMEngine_TokenId));
        }
    } else {
        out->token_ids = nullptr;
    }

    out->text = static_cast<char*>(std::malloc(cpp.text.size() + 1));
    if (out->text) {
        std::memcpy(out->text, cpp.text.data(), cpp.text.size());
        out->text[cpp.text.size()] = '\0';
    }

    out->finish_reason = static_cast<CANN_LMEngine_FinishReason>(cpp.finish_reason);
}

// ────────────────────────── Executor Lifecycle ──────────────────────────

extern "C" CANN_LMEngine_Status CANN_LMEngine_ExecutorCreate(const CANN_LMEngine_ModelArtifact* artifact,
                                                             const CANN_LMEngine_EngineOptions* options,
                                                             CANN_LMEngine_Executor** out)
{
    if (!artifact || !out)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    *out = nullptr;
    try {
        auto art = to_cpp_artifact(artifact);
        auto opts = to_cpp_options(options);
        auto e = std::make_unique<CANN_LMEngine_Executor>();
        e->ee = std::make_unique<cann_lm::engine::EngineExecutor>(art, opts);
        // 加载失败即终止（2026-09-22）：模型图位已配置而加载失败（含权重
        // InitWeights，如设备端 NPU 配额耗尽）→ 构造内已终止初始化并完成
        // 卸载 → 此处不交出半初始化实例（*out 保持 null），失败码原样上抛。
        const cann_lm::engine::Status init = e->ee->InitStatus();
        if (!init.ok()) {
            CANN_LLM_LOGE("c_api.create: engine init failed (st=%d), executor not created", init.code);
            return status_to_c(init);
        }
        *out = e.release();
        return CANN_LM_ENGINE_SUCCESS;
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

extern "C" void CANN_LMEngine_ExecutorDestroy(CANN_LMEngine_Executor* executor) { delete executor; }

// ────────────────────────── Context Management ──────────────────────────

extern "C" CANN_LMEngine_Status CANN_LMEngine_ContextCreate(CANN_LMEngine_Executor* executor,
                                                            CANN_LMEngine_ContextId* out)
{
    if (!executor || !out)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    *out = 0;
    try {
        cann_lm::engine::Context ctx;
        auto s = executor->ee->CreateContext(&ctx);
        if (!s.ok())
            return status_to_c(s);
        auto id = ctx.id();
        executor->contexts[id.value] = std::move(ctx);
        *out = id.value;
        return CANN_LM_ENGINE_SUCCESS;
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

extern "C" CANN_LMEngine_Status CANN_LMEngine_ContextDestroy(CANN_LMEngine_Executor* executor,
                                                             CANN_LMEngine_ContextId context_id)
{
    if (!executor)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    try {
        cann_lm::engine::ContextId cid{context_id};
        auto s = executor->ee->DestroyContext(cid);
        executor->contexts.erase(static_cast<uint64_t>(context_id));
        return status_to_c(s);
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

extern "C" CANN_LMEngine_Status CANN_LMEngine_ContextCancel(CANN_LMEngine_Executor* executor,
                                                            CANN_LMEngine_ContextId context_id)
{
    if (!executor)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    try {
        cann_lm::engine::ContextId cid{context_id};
        auto s = executor->ee->CancelContext(cid);
        return status_to_c(s);
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

// ────────────────────────── Inference ──────────────────────────

extern "C" CANN_LMEngine_Status CANN_LMEngine_ExecutorGenerate(
    CANN_LMEngine_Executor* executor, const char* prompt_text, CANN_LMEngine_ContextId context_id,
    const CANN_LMEngine_SamplingParams* params, const CANN_LMEngine_RequestOptions* options, CANN_LMEngine_Result* out)
{
    if (!executor || !prompt_text || !out)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    *out = CANN_LMEngine_Result{};
    try {
        auto it = executor->contexts.find(static_cast<uint64_t>(context_id));
        if (it == executor->contexts.end())
            return CANN_LM_ENGINE_NOT_FOUND;
        cann_lm::engine::Prompt prompt;
        prompt.text = prompt_text;
        auto sp = to_cpp_params(params);
        auto ro = to_cpp_request_options(options);
        cann_lm::engine::Result cpp_result;
        auto s = executor->ee->Generate(prompt, it->second, sp, ro, &cpp_result);
        if (s.ok() || s.code == cann_lm::engine::Status::FAILURE) {
            fill_c_result(cpp_result, out);
        }
        return status_to_c(s);
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

extern "C" CANN_LMEngine_Status CANN_LMEngine_ExecutorGenerateAsync(
    CANN_LMEngine_Executor* executor, const char* prompt_text, CANN_LMEngine_ContextId context_id,
    const CANN_LMEngine_SamplingParams* params, const CANN_LMEngine_RequestOptions* options,
    CANN_LMEngine_Request** out, CANN_LMEngine_StreamCallback callback, void* user_data)
{
    if (!executor || !prompt_text || !out)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    *out = nullptr;
    try {
        auto it = executor->contexts.find(static_cast<uint64_t>(context_id));
        if (it == executor->contexts.end())
            return CANN_LM_ENGINE_NOT_FOUND;
        cann_lm::engine::Prompt prompt;
        prompt.text = prompt_text;
        auto sp = to_cpp_params(params);
        auto ro = to_cpp_request_options(options);
        auto cb = wrap_callback(callback, user_data);
        cann_lm::engine::RequestHandle cpp_handle;
        auto s = executor->ee->GenerateAsync(prompt, it->second, sp, ro, &cpp_handle, cb);
        if (s.ok()) {
            auto req = std::make_unique<CANN_LMEngine_Request>();
            req->handle = std::move(cpp_handle);
            *out = req.release();
        }
        return status_to_c(s);
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

// ────────────────────────── Request Handle ──────────────────────────

extern "C" CANN_LMEngine_RequestStatus CANN_LMEngine_RequestGetStatus(const CANN_LMEngine_Request* request)
{
    if (!request)
        return CANN_LM_ENGINE_REQUEST_QUEUED;
    return static_cast<CANN_LMEngine_RequestStatus>(request->handle.GetStatus());
}

extern "C" int CANN_LMEngine_RequestIsDone(const CANN_LMEngine_Request* request)
{
    if (!request)
        return 0;
    return request->handle.IsDone() ? 1 : 0;
}

extern "C" CANN_LMEngine_Status CANN_LMEngine_RequestCancel(CANN_LMEngine_Request* request)
{
    if (!request)
        return CANN_LM_ENGINE_SUCCESS;
    return status_to_c(request->handle.Cancel());
}

extern "C" CANN_LMEngine_Status CANN_LMEngine_RequestWait(CANN_LMEngine_Request* request, CANN_LMEngine_Result* out)
{
    if (!request || !out)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    *out = CANN_LMEngine_Result{};
    try {
        cann_lm::engine::Result cpp_result;
        auto s = request->handle.Wait(&cpp_result);
        if (s.ok() || s.code == cann_lm::engine::Status::FAILURE) {
            fill_c_result(cpp_result, out);
        }
        return status_to_c(s);
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

extern "C" CANN_LMEngine_Status CANN_LMEngine_RequestTryGetResult(const CANN_LMEngine_Request* request,
                                                                  CANN_LMEngine_Result* out)
{
    if (!request || !out)
        return CANN_LM_ENGINE_INVALID_ARGUMENT;
    *out = CANN_LMEngine_Result{};
    try {
        cann_lm::engine::Result cpp_result;
        auto s = request->handle.TryGetResult(&cpp_result);
        if (s.ok() || s.code == cann_lm::engine::Status::FAILURE) {
            fill_c_result(cpp_result, out);
        }
        return status_to_c(s);
    } catch (const std::exception& e) {
        return log_exception(__func__, e.what());
    } catch (...) {
        return log_exception(__func__, "unknown exception");
    }
}

extern "C" void CANN_LMEngine_RequestDestroy(CANN_LMEngine_Request* request) { delete request; }

// ────────────────────────── Result Memory Management ──────────────────────────

extern "C" void CANN_LMEngine_ResultDestroy(CANN_LMEngine_Result* result)
{
    if (!result)
        return;
    std::free(result->token_ids);
    result->token_ids = nullptr;
    result->token_ids_count = 0;
    std::free(result->text);
    result->text = nullptr;
    result->finish_reason = CANN_LM_ENGINE_FINISH_MAX_TOKENS;
}
