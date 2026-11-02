// bbit — Black-Box Integration Test for CANN-LM Engine C API.
//
// A bbit is a real user of the engine: it links against the engine library
// and calls every C ABI function in sequence. No mock backends, no dlopen.
//
// In stub-era (current): Generate returns FAILURE — no backend linked.
//   Executor/Context lifecycle still succeeds; Generate/Async return FAILURE.
// In real deployment: the engine library carries a real backend; full flow
//   succeeds end-to-end (tokenize → prefill → decode → detokenize → result).
//
// 2026-09-22 load-fail-fast contract: model omc paths configured but load
//   fails (incl. weight InitWeights, e.g. NPU quota exhausted on device)
//   → engine aborts init, unloads, and CANN_LMEngine_ExecutorCreate returns
//   the error WITHOUT handing out an executor (no half-initialized engine).
//   bbit adapts: model files present on this machine → create must succeed
//   (any failure is a hard regression); model files absent (host stub-era
//   with device-style paths) → degraded run: create must fail with a null
//   executor and the flow stops after verifying that contract.
//
// bbit passes (exit 0) in both environments as long as:
//   - NULL inputs are rejected (C ABI contract)
//   - Executor/Context lifecycle functions succeed
//   - Generate returns SUCCESS, FAILURE, or INVALID_STATE (not crash):
//     SUCCESS = real backend full chain; FAILURE = inference failed (stub-era
//     norm); INVALID_STATE = tokenizer explicitly supplied but unloadable
//     (fast-fail, TC_107 semantics — e.g. device-only vocab path on host)
//   - If GenerateAsync succeeds, the full request lifecycle works
//
// Flow mode is config-driven (nlohmann/json, template: tests/bbit/testcase.json):
//   - model_path_prefill / model_path_decode / model_meta_config /
//     model_weight_dir_prefill / model_weight_dir_decode
//     → CANN_LMEngine_ModelArtifact paths（权重路径 prefill/decode 拆分：
//       docs/separate_pd_weights.md §4.3；共用 = 两键同值）
//   - tokenizer_type / tokenizer_path → CANN_LMEngine_EngineOptions
//     (algorithm selection + vocab path; options wins over meta_json
//      tokenizer_config.tokenizer_file)
//   - sentences[] sampler knobs: topK / topP / temperature / seed /
//     repetitionPenalty → CANN_LMEngine_SamplingParams
//   - sentences[] termination: maxGenTokens / stopSeq →
//     CANN_LMEngine_RequestOptions (request-level stop sets; independent of
//      tokenizer eos; stop literals match because detokenize keeps specials)
//   - one sync Generate per sentence; the first sentence also drives one
//     GenerateAsync to cover the Request lifecycle (its stream callback
//     prints each decoded token to the terminal as it arrives)
//
// Usage:
//   bbit --check            — NULL defense tests (C ABI boundary, no config)
//   bbit --config <path>    — full API flow driven by JSON config
//   bbit                    — same, default-searching ./testcase.json then
//                             testcase.json next to this executable
//
// Exit codes: 0 = all checks passed; 1 = a check failed; 2 = config/usage error.
#include "api/c/c_api.h"

#include <nlohmann/json.hpp>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

static int g_total = 0;
static int g_pass = 0;

static void check(bool ok, const char* msg)
{
    g_total++;
    if (ok) {
        g_pass++;
    } else {
        fprintf(stderr, "FAIL: %s\n", msg);
    }
}

/// 文件/目录在场判定（降级跑分支判据：模型 omc 是否在本机存在）。
static bool file_exists(const std::string& path)
{
    struct stat st;
    return !path.empty() && stat(path.c_str(), &st) == 0;
}

// Generate/GenerateAsync 合法返回集（见文件头注释）：SUCCESS（真实后端全链路）/
// FAILURE（推理失败，stub 时代常态）/ INVALID_STATE（tokenizer 显式供给但不可加载
// 的快速失败——host 上配置了设备路径的词表时即此态）。
static void check_generate_status(CANN_LMEngine_Status actual, const char* msg)
{
    g_total++;
    if (actual == CANN_LM_ENGINE_SUCCESS || actual == CANN_LM_ENGINE_FAILURE ||
        actual == CANN_LM_ENGINE_INVALID_STATE) {
        g_pass++;
    } else {
        fprintf(stderr, "FAIL: %s (got %d, expected 0/1/3)\n", msg, actual);
    }
}

// ── Config (tests/bbit/testcase.json) ────────────────────────────────────────

struct SentenceCase {
    std::string prompt;
    int max_tokens = 0; // maxGenTokens
    int top_k = 0;      // topK
    float top_p = 1.0f; // topP
    float temperature = 1.0f;
    std::vector<std::string> stop_strings; // stopSeq[]
    // Sampler knobs (→ CANN_LMEngine_SamplingParams)
    long long seed = 0;              // seed：0 = 不播种（每次随机）
    float repetition_penalty = 1.0f; // repetitionPenalty：1.0 = 关闭
};

struct Config {
    std::string tokenizer_type;      // → EngineOptions.tokenizer_type（算法选择）
    std::string tokenizer_path;      // → EngineOptions.tokenizer_path（词表路径）
    std::string prefill_omc_path;    // model_path_prefill
    std::string decode_omc_path;     // model_path_decode
    std::string meta_json_path;      // model_meta_config
    std::string prefill_weight_path; // model_weight_dir_prefill
    std::string decode_weight_path;  // model_weight_dir_decode
    std::vector<SentenceCase> sentences;
};

// ── Config loading ───────────────────────────────────────────────────────────

namespace {
// Error marker for config-field problems: the helper already printed the
// reason to stderr; load_config catches this to unwind the parse.
struct ConfigError {};
} // namespace

static const nlohmann::json& req_field(const nlohmann::json& j, const char* name)
{
    const auto it = j.find(name);
    if (it == j.end()) {
        fprintf(stderr, "bbit: config missing field \"%s\"\n", name);
        throw ConfigError{};
    }
    return *it;
}

static std::string req_string(const nlohmann::json& j, const char* name)
{
    const auto& v = req_field(j, name);
    if (!v.is_string()) {
        fprintf(stderr, "bbit: config field \"%s\" must be a string\n", name);
        throw ConfigError{};
    }
    return v.get<std::string>();
}

static long long req_int(const nlohmann::json& j, const char* name)
{
    const auto& v = req_field(j, name);
    if (!v.is_number_integer()) {
        fprintf(stderr, "bbit: config field \"%s\" must be an integer\n", name);
        throw ConfigError{};
    }
    return v.get<long long>();
}

// req_int narrowed to int with a range check (silent wraparound otherwise).
static int req_int32(const nlohmann::json& j, const char* name)
{
    const long long v = req_int(j, name);
    if (v < INT_MIN || v > INT_MAX) {
        fprintf(stderr, "bbit: config field \"%s\" out of int range\n", name);
        throw ConfigError{};
    }
    return static_cast<int>(v);
}

static double req_number(const nlohmann::json& j, const char* name)
{
    const auto& v = req_field(j, name);
    if (!v.is_number()) {
        fprintf(stderr, "bbit: config field \"%s\" must be a number\n", name);
        throw ConfigError{};
    }
    return v.get<double>();
}

// Parses the testcase template contract. All template fields are required;
// a missing or wrongly-typed field prints the field name and returns false.
static bool load_config(const std::string& path, Config* out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "bbit: cannot open config: %s\n", path.c_str());
        return false;
    }
    std::string content;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
        content.append(buf, n);
    }
    const bool read_err = ferror(f);
    fclose(f);
    if (read_err) {
        fprintf(stderr, "bbit: cannot read config: %s\n", path.c_str());
        return false;
    }

    try {
        const auto j = nlohmann::json::parse(content);
        out->tokenizer_type = req_string(j, "tokenizer_type");
        out->tokenizer_path = req_string(j, "tokenizer_path");
        out->prefill_omc_path = req_string(j, "model_path_prefill");
        out->decode_omc_path = req_string(j, "model_path_decode");
        out->meta_json_path = req_string(j, "model_meta_config");
        out->prefill_weight_path = req_string(j, "model_weight_dir_prefill");
        out->decode_weight_path = req_string(j, "model_weight_dir_decode");

        const auto& sentences = req_field(j, "sentences");
        if (!sentences.is_array() || sentences.empty()) {
            fprintf(stderr, "bbit: config field \"sentences\" must be a non-empty array\n");
            throw ConfigError{};
        }
        for (const auto& js : sentences) {
            SentenceCase s;
            s.prompt = req_string(js, "prompt");
            s.max_tokens = req_int32(js, "maxGenTokens");
            s.top_k = req_int32(js, "topK");
            s.top_p = static_cast<float>(req_number(js, "topP"));
            s.temperature = static_cast<float>(req_number(js, "temperature"));
            s.seed = req_int(js, "seed");
            s.repetition_penalty = static_cast<float>(req_number(js, "repetitionPenalty"));
            const auto& stop = req_field(js, "stopSeq");
            if (!stop.is_array()) {
                fprintf(stderr, "bbit: config field \"stopSeq\" must be an array\n");
                throw ConfigError{};
            }
            for (const auto& ss : stop) {
                if (!ss.is_string()) {
                    fprintf(stderr, "bbit: config field \"stopSeq\" entries must be strings\n");
                    throw ConfigError{};
                }
                s.stop_strings.push_back(ss.get<std::string>());
            }
            out->sentences.push_back(std::move(s));
        }
        return true;
    } catch (const ConfigError&) {
        return false; // reason already printed by the req_* helper
    } catch (const std::exception& e) {
        fprintf(stderr, "bbit: config parse error: %s\n", e.what());
        return false;
    }
}

// ── Config → C API parameters ────────────────────────────────────────────────

// Sentence → C sampling params (sampler knobs only; termination is in
// make_request_options below).
static CANN_LMEngine_SamplingParams make_params(const SentenceCase& s)
{
    CANN_LMEngine_SamplingParams sp = {};
    sp.temperature = s.temperature;
    sp.top_k = s.top_k;
    sp.top_p = s.top_p;
    sp.seed = static_cast<uint64_t>(s.seed);
    sp.repetition_penalty = s.repetition_penalty;
    return sp;
}

// Sentence → C request-level termination options. stop_cstrs keeps the
// SentenceCase-owned strings alive; the C ABI copies during the call.
static CANN_LMEngine_RequestOptions make_request_options(const SentenceCase& s, std::vector<const char*>* stop_cstrs)
{
    stop_cstrs->clear();
    for (const auto& ss : s.stop_strings) {
        stop_cstrs->push_back(ss.c_str());
    }
    CANN_LMEngine_RequestOptions ro = {};
    ro.max_tokens = s.max_tokens;
    ro.stop_token_ids = nullptr;
    ro.stop_token_ids_count = 0;
    ro.stop_strings = stop_cstrs->empty() ? nullptr : stop_cstrs->data();
    ro.stop_strings_count = stop_cstrs->size();
    return ro;
}

// ── Config file location ─────────────────────────────────────────────────────

static bool file_readable(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

// Default search when no --config is given: ./testcase.json, then
// testcase.json next to the executable. Empty string = not found.
static std::string default_config_path(const char* argv0)
{
    if (file_readable("testcase.json")) {
        return "testcase.json";
    }
    if (const char* slash = strrchr(argv0, '/')) {
        std::string side(argv0, static_cast<size_t>(slash - argv0));
        side += "/testcase.json";
        if (file_readable(side.c_str())) {
            return side;
        }
    }
    return "";
}

static void print_usage(const char* prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --check          NULL defense tests (no config needed)\n"
            "  %s --config <path>  full API flow driven by JSON config\n"
            "  %s                  same as --config; default-searches\n"
            "                     ./testcase.json then <prog dir>/testcase.json\n",
            prog, prog, prog);
}

// ── --check: NULL defense (C ABI contract) ────────────────────────────────────

static int run_check()
{
    // Executor-level: NULL inputs → INVALID_ARGUMENT
    check(CANN_LMEngine_ExecutorCreate(nullptr, nullptr, nullptr) == CANN_LM_ENGINE_INVALID_ARGUMENT,
          "ExecutorCreate NULL → INVALID_ARGUMENT");
    check(
        CANN_LMEngine_ExecutorGenerate(nullptr, nullptr, nullptr, nullptr, nullptr) == CANN_LM_ENGINE_INVALID_ARGUMENT,
        "ExecutorGenerate NULL → INVALID_ARGUMENT");
    check(CANN_LMEngine_ContextCreate(nullptr, nullptr) == CANN_LM_ENGINE_INVALID_ARGUMENT,
          "ContextCreate NULL → INVALID_ARGUMENT");
    check(CANN_LMEngine_ContextDestroy(nullptr, 0) == CANN_LM_ENGINE_INVALID_ARGUMENT,
          "ContextDestroy NULL → INVALID_ARGUMENT");
    check(CANN_LMEngine_ContextCancel(nullptr, 0) == CANN_LM_ENGINE_INVALID_ARGUMENT,
          "ContextCancel NULL → INVALID_ARGUMENT");

    // Request-level: NULL inputs → safe defaults (not INVALID_ARGUMENT)
    check(CANN_LMEngine_RequestGetStatus(nullptr) == CANN_LM_ENGINE_REQUEST_QUEUED, "RequestGetStatus NULL → QUEUED");
    check(CANN_LMEngine_RequestIsDone(nullptr) == 0, "RequestIsDone NULL → 0");
    check(CANN_LMEngine_RequestCancel(nullptr) == CANN_LM_ENGINE_SUCCESS, "RequestCancel NULL → SUCCESS");
    check(CANN_LMEngine_RequestWait(nullptr, nullptr) == CANN_LM_ENGINE_INVALID_ARGUMENT,
          "RequestWait NULL → INVALID_ARGUMENT");
    check(CANN_LMEngine_RequestTryGetResult(nullptr, nullptr) == CANN_LM_ENGINE_INVALID_ARGUMENT,
          "RequestTryGetResult NULL → INVALID_ARGUMENT");

    // NULL-safe no-ops (no crash = pass)
    CANN_LMEngine_ExecutorDestroy(nullptr);
    g_total++;
    g_pass++;
    CANN_LMEngine_RequestDestroy(nullptr);
    g_total++;
    g_pass++;
    CANN_LMEngine_ResultDestroy(nullptr);
    g_total++;
    g_pass++;

    printf("bbit --check: %d/%d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}

// ── Stream callback (GenerateAsync) ──────────────────────────────────────────

// 流式回调（c_api.h Stream Callback 契约）：pipeline 内部线程每解码出一个
// token 调一次，|text|/|text_len| 仅在回调期间有效；契约禁止在回调内阻塞或
// 调 Engine API —— 这里只做 fwrite + fflush，把增量文本实时打到终端。
static void stream_print_callback(CANN_LMEngine_TokenId token, const char* text, size_t text_len, void* user_data)
{
    (void)token;
    (void)user_data;
    if (text != nullptr && text_len > 0) {
        fwrite(text, 1, text_len, stdout);
        fflush(stdout);
    }
}

// ── Full flow: call every C ABI function in sequence ──────────────────────────
// Config-driven: artifact paths and per-sentence sampling params come from the
// JSON testcase file. Requires cfg.sentences to be non-empty (load_config
// enforces this).
static int run_flow(const Config& cfg)
{
    printf("bbit: using config: prefill=%s decode=%s "
           "prefill_weight=%s decode_weight=%s meta=%s (%zu sentence(s))\n",
           cfg.prefill_omc_path.c_str(), cfg.decode_omc_path.c_str(), cfg.prefill_weight_path.c_str(),
           cfg.decode_weight_path.c_str(), cfg.meta_json_path.c_str(), cfg.sentences.size());
    printf("bbit: tokenizer via EngineOptions: type=%s path=%s\n", cfg.tokenizer_type.c_str(),
           cfg.tokenizer_path.c_str());

    // 1. ExecutorCreate — 2026-09-22 契约：图位已配置而加载失败（含权重
    //    InitWeights，如设备端 NPU 配额耗尽）→ 引擎终止初始化、完成卸载、
    //    create 直接报错且不交出实例（fail-fast，不再放半初始化引擎出来）。
    //    本机区分两种期望：omc 文件在场（真实部署）→ create 必须成功并走
    //    全流程；文件不在场（host 桩期跑设备路径）→ create 必须报错且
    //    实例为 null，验证报错契约后降级收尾。
    CANN_LMEngine_ModelArtifact artifact = {};
    artifact.prefill_omc_path = cfg.prefill_omc_path.c_str();
    artifact.decode_omc_path = cfg.decode_omc_path.c_str();
    artifact.prefill_weight = cfg.prefill_weight_path.c_str();
    artifact.decode_weight = cfg.decode_weight_path.c_str();
    artifact.meta_json_path = cfg.meta_json_path.c_str();
    // tokenizer_type/tokenizer_path 经 EngineOptions 传入（算法选择 + 词表路径；
    // 其余 EngineOptions 字段模板暂无对应项，保持默认值）
    CANN_LMEngine_EngineOptions opts = {1024, 1, 0, cfg.tokenizer_type.c_str(), cfg.tokenizer_path.c_str()};
    CANN_LMEngine_Executor* executor = nullptr;
    const CANN_LMEngine_Status create_st = CANN_LMEngine_ExecutorCreate(&artifact, &opts, &executor);
    const bool omc_configured = !cfg.prefill_omc_path.empty() && !cfg.decode_omc_path.empty();
    const bool omc_on_disk = file_exists(cfg.prefill_omc_path) && file_exists(cfg.decode_omc_path);
    if (omc_configured && !omc_on_disk) {
        printf("bbit: model omc absent on this machine (%s / %s) — "
               "degraded run: engine aborts init, ExecutorCreate reports error\n",
               cfg.prefill_omc_path.c_str(), cfg.decode_omc_path.c_str());
        check(create_st != CANN_LM_ENGINE_SUCCESS, "ExecutorCreate (omc absent → engine init must fail)");
        check(executor == nullptr, "failed ExecutorCreate → null executor (no half-initialized engine)");
        CANN_LMEngine_ContextId no_cid = 0;
        check(CANN_LMEngine_ContextCreate(executor, &no_cid) == CANN_LM_ENGINE_INVALID_ARGUMENT,
              "ContextCreate(NULL executor) rejected");
        printf("bbit flow (degraded, no model): %d/%d passed\n", g_pass, g_total);
        return (g_pass == g_total) ? 0 : 1;
    }
    check(create_st == CANN_LM_ENGINE_SUCCESS, "ExecutorCreate");
    check(executor != nullptr, "executor non-null");
    if (!executor) {
        printf("bbit: aborted (no executor)\n");
        return 1;
    }

    // 2. ContextCreate
    CANN_LMEngine_ContextId cid = 0;
    check(CANN_LMEngine_ContextCreate(executor, &cid) == CANN_LM_ENGINE_SUCCESS, "ContextCreate");
    check(cid != 0, "context id non-zero");

    // 2b. (ADR-0002) SetContext — activate the context before Generate.
    //     Generate without active context must be rejected (S2):
    //     empty stub check first, then activate.
    {
        CANN_LMEngine_Result pre = {};
        check(CANN_LMEngine_ExecutorGenerate(executor, "hi", nullptr, nullptr, &pre) == CANN_LM_ENGINE_INVALID_STATE,
              "Generate before SetContext → INVALID_STATE (S2)");
    }
    check(CANN_LMEngine_ExecutorSetContext(executor, 999) == CANN_LM_ENGINE_NOT_FOUND,
          "SetContext unregistered id → NOT_FOUND");
    check(CANN_LMEngine_ExecutorSetContext(executor, cid) == CANN_LM_ENGINE_SUCCESS, "SetContext on created context");

    // 3. Generate (sync) — one per sentence; SUCCESS (real backend), FAILURE
    //    (inference failed), or INVALID_STATE (tokenizer supplied but unloadable)
    for (size_t i = 0; i < cfg.sentences.size(); ++i) {
        const SentenceCase& s = cfg.sentences[i];
        std::vector<const char*> stop_cstrs;
        const CANN_LMEngine_SamplingParams sp = make_params(s);
        const CANN_LMEngine_RequestOptions ro = make_request_options(s, &stop_cstrs);
        CANN_LMEngine_Result result = {};
        const CANN_LMEngine_Status st = CANN_LMEngine_ExecutorGenerate(executor, s.prompt.c_str(), &sp, &ro, &result);
        printf("bbit: sentence[%zu] prompt=\"%s\" -> status %d\n", i, s.prompt.c_str(), st);
        check_generate_status(st, "Generate sync (SUCCESS/FAILURE/INVALID_STATE)");
        if (st == CANN_LM_ENGINE_SUCCESS) {
            check(result.token_ids_count > 0, "Generate sync produced tokens");
            check(result.text != nullptr, "Generate sync produced text");
            if (result.text != nullptr) {
                printf("bbit: sentence[%zu] generated: %s\n", i, result.text);
            }
        }
        CANN_LMEngine_ResultDestroy(&result);
    }

    // 4. GenerateAsync — first sentence; 流式回调把每个解码出的 token 直接
    //    打到终端（见 stream_print_callback），并走完下方 Request 生命周期。
    const SentenceCase& s0 = cfg.sentences.front();
    std::vector<const char*> stop_cstrs0;
    const CANN_LMEngine_SamplingParams sp0 = make_params(s0);
    const CANN_LMEngine_RequestOptions ro0 = make_request_options(s0, &stop_cstrs0);
    CANN_LMEngine_Request* request = nullptr;
    printf("bbit: sentence[0] async stream:\n");
    CANN_LMEngine_Status st = CANN_LMEngine_ExecutorGenerateAsync(executor, s0.prompt.c_str(), &sp0, &ro0, &request,
                                                                  stream_print_callback, nullptr);
    printf("bbit: async submit -> status %d\n", st);
    check_generate_status(st, "GenerateAsync (SUCCESS/FAILURE/INVALID_STATE)");

    // 5. Request lifecycle (only if async succeeded)
    if (st == CANN_LM_ENGINE_SUCCESS && request) {
        // RequestWait (blocks until terminal; the callback above has been
        // streaming tokens to the terminal — close that line with a newline)
        CANN_LMEngine_Result result2 = {};
        check(CANN_LMEngine_RequestWait(request, &result2) == CANN_LM_ENGINE_SUCCESS, "RequestWait");
        printf("\n");
        if (result2.token_ids_count > 0) {
            check(result2.text != nullptr, "RequestWait produced text");
        }
        CANN_LMEngine_ResultDestroy(&result2);

        // RequestGetStatus / RequestIsDone (after completion)
        check(CANN_LMEngine_RequestGetStatus(request) == CANN_LM_ENGINE_REQUEST_COMPLETED,
              "RequestGetStatus → COMPLETED");
        check(CANN_LMEngine_RequestIsDone(request) == 1, "RequestIsDone → 1");

        // RequestTryGetResult (on completed request)
        CANN_LMEngine_Result result3 = {};
        check(CANN_LMEngine_RequestTryGetResult(request, &result3) == CANN_LM_ENGINE_SUCCESS,
              "RequestTryGetResult on completed");
        CANN_LMEngine_ResultDestroy(&result3);

        // RequestCancel (idempotent on completed request)
        check(CANN_LMEngine_RequestCancel(request) == CANN_LM_ENGINE_SUCCESS, "RequestCancel (idempotent)");
    } else {
        // Stub era / submission failure: request is null and the Request
        // lifecycle block above was NOT exercised this run (printed so a
        // "N/N passed" line is never mistaken for async coverage).
        printf("bbit: async submit failed — Request lifecycle NOT exercised this run\n");
        check(request == nullptr, "GenerateAsync failure → null request");
    }

    // 6. RequestDestroy (NULL-safe no-op when submit failed)
    CANN_LMEngine_RequestDestroy(request);
    g_total++;
    g_pass++;

    // 7. ContextCancel (on context with no active generate)
    check(CANN_LMEngine_ContextCancel(executor, cid) == CANN_LM_ENGINE_SUCCESS, "ContextCancel (no active generate)");

    // 8. ContextDestroy
    check(CANN_LMEngine_ContextDestroy(executor, cid) == CANN_LM_ENGINE_SUCCESS, "ContextDestroy");

    // 9. ContextDestroy (idempotent)
    check(CANN_LMEngine_ContextDestroy(executor, cid) == CANN_LM_ENGINE_SUCCESS, "ContextDestroy (idempotent)");

    // 10. ExecutorDestroy
    CANN_LMEngine_ExecutorDestroy(executor);
    g_total++;
    g_pass++;

    printf("bbit flow: %d/%d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc >= 2 && strcmp(argv[1], "--check") == 0) {
        return run_check();
    }

    // Flow mode: --config <path>, else default-search testcase.json.
    // An explicitly given path (even an empty string) is used as-is — only a
    // missing --config falls back to the default search.
    std::string config_path;
    bool config_given = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--config") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "bbit: missing value for --config\n");
                print_usage(argv[0]);
                return 2;
            }
            config_path = argv[++i];
            config_given = true;
        } else {
            fprintf(stderr, "bbit: unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }
    if (!config_given) {
        config_path = default_config_path(argv[0]);
        if (config_path.empty()) {
            fprintf(stderr, "bbit: config not found (looked for ./testcase.json "
                            "and testcase.json next to the executable)\n");
            print_usage(argv[0]);
            return 2;
        }
    }

    Config cfg;
    if (!load_config(config_path, &cfg)) {
        return 2;
    }
    return run_flow(cfg);
}
