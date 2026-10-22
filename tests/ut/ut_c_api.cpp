#include "api/c/c_api.h"

#include <gtest/gtest.h>

namespace {

// 无模型部署形态构件（2026-09-22 契约第 2 点——双图位全空分支）：三路径 NULL +
// 全空 buffer = 图位未配置 → 构造跳过模型加载，ExecutorCreate SUCCESS、
// 生命周期 API 可用（推理报 not-ready）。与 c_api_smoke.c 第 3 节 empty_artifact
// 同口径。原此处为 dummy.* 假路径内联构件，严格加载门控下 create 必失败，故清空
//（生命周期 TC 断言语义全部保持不变）。
CANN_LMEngine_ModelArtifact MakeEmptyArtifact() {
    CANN_LMEngine_ModelArtifact a{nullptr, {nullptr, 0}, nullptr, {nullptr, 0},
                                  nullptr, nullptr, nullptr, {nullptr, 0}};
    return a;
}

TEST(CApiTest, ContextCancelReturnsNotFoundForNonexistentContext) {
    CANN_LMEngine_ModelArtifact p = MakeEmptyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    ASSERT_EQ(CANN_LMEngine_ExecutorCreate(&p, &opts, &exec),
              CANN_LM_ENGINE_SUCCESS);

    auto s = CANN_LMEngine_ContextCancel(exec, 999);
    EXPECT_EQ(s, CANN_LM_ENGINE_NOT_FOUND);

    auto ds = CANN_LMEngine_ContextDestroy(exec, 999);
    EXPECT_EQ(ds, CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_ExecutorDestroy(exec);
}

TEST(CApiTest, NullExecutorReturnsInvalidArgument) {
    CANN_LMEngine_ContextId id;
    EXPECT_EQ(CANN_LMEngine_ContextCreate(nullptr, &id),
              CANN_LM_ENGINE_INVALID_ARGUMENT);
    EXPECT_EQ(CANN_LMEngine_ContextCancel(nullptr, 0),
              CANN_LM_ENGINE_INVALID_ARGUMENT);
    EXPECT_EQ(CANN_LMEngine_ContextDestroy(nullptr, 0),
              CANN_LM_ENGINE_INVALID_ARGUMENT);
}

TEST(CApiTest, RequestIsDoneReturnsFalseForNull) {
    EXPECT_EQ(CANN_LMEngine_RequestIsDone(nullptr), 0);
}

TEST(CApiTest, RequestCancelReturnsSuccessForNull) {
    EXPECT_EQ(CANN_LMEngine_RequestCancel(nullptr),
              CANN_LM_ENGINE_SUCCESS);
}

TEST(CApiTest, RequestGetStatusReturnsQueuedForNull) {
    EXPECT_EQ(CANN_LMEngine_RequestGetStatus(nullptr),
              CANN_LM_ENGINE_REQUEST_QUEUED);
}

TEST(CApiTest, GenerateOnNonexistentContextReturnsNotFound) {
    CANN_LMEngine_ModelArtifact p = MakeEmptyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    ASSERT_EQ(CANN_LMEngine_ExecutorCreate(&p, &opts, &exec),
              CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_Result result{};
    auto s = CANN_LMEngine_ExecutorGenerate(exec, "hello", 999, nullptr, nullptr, &result);
    EXPECT_EQ(s, CANN_LM_ENGINE_NOT_FOUND);

    CANN_LMEngine_ExecutorDestroy(exec);
}

TEST(CApiTest, ContextCreateSucceeds) {
    CANN_LMEngine_ModelArtifact p = MakeEmptyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    ASSERT_EQ(CANN_LMEngine_ExecutorCreate(&p, &opts, &exec),
              CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_ContextId id = 0;
    auto s = CANN_LMEngine_ContextCreate(exec, &id);
    EXPECT_EQ(s, CANN_LM_ENGINE_SUCCESS);
    EXPECT_NE(id, 0u);

    CANN_LMEngine_ExecutorDestroy(exec);
}

TEST(CApiTest, ContextCancelSucceedsOnCreatedContext) {
    CANN_LMEngine_ModelArtifact p = MakeEmptyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    ASSERT_EQ(CANN_LMEngine_ExecutorCreate(&p, &opts, &exec),
              CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_ContextId id = 0;
    ASSERT_EQ(CANN_LMEngine_ContextCreate(exec, &id), CANN_LM_ENGINE_SUCCESS);
    auto s = CANN_LMEngine_ContextCancel(exec, id);
    EXPECT_EQ(s, CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_ExecutorDestroy(exec);
}

TEST(CApiTest, GenerateReturnsFailureOnCreatedContextStubEra) {
    // Context exists but tokenizer is stub → Generate returns FAILURE
    CANN_LMEngine_ModelArtifact p = MakeEmptyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    ASSERT_EQ(CANN_LMEngine_ExecutorCreate(&p, &opts, &exec),
              CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_ContextId id = 0;
    ASSERT_EQ(CANN_LMEngine_ContextCreate(exec, &id), CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_SamplingParams params{};
    CANN_LMEngine_Result result{};
    auto s = CANN_LMEngine_ExecutorGenerate(exec, "hello", id, &params, nullptr, &result);
    EXPECT_NE(s, CANN_LM_ENGINE_SUCCESS);
    CANN_LMEngine_ResultDestroy(&result);

    CANN_LMEngine_ExecutorDestroy(exec);
}

// ═══════════════════════════════════════════════════════════════════════════
// UT-TC-CAPI-FB-* : 合并后单入口 buffer 态覆盖（NEW-12..NEW-16 收敛为 artifact/out
// 防御；2026-09-22 增补 TC_104 严格加载门控契约面）
// 关联 docs/ut/SPEC.md §4.3 / docs/ut/DESIGN.md §9
// ═══════════════════════════════════════════════════════════════════════════

namespace {
// buffer 态构件：omc/meta buffer 字段在位但零长度（size=0 = 未提供，C-02 双态）
// → 无模型形态：create SUCCESS、生命周期 API 可用。2026-09-22 严格加载门控下
// 非空 omc buffer 必经严格加载，buffer 态构造成功路径改由零长度承载；
// 双权重路径同步清空（weight 单独非空不触发门控，但与 c_api_smoke.c
// empty_artifact 口径保持一致）。严格加载失败面由 MakeMissingModelArtifact 承载。
CANN_LMEngine_ModelArtifact MakeDummyArtifact() {
    static char buf[8] = {0};
    CANN_LMEngine_Buffer b{buf, 0};   // 零长度 buffer：data 在位、size=0 → 未提供
    CANN_LMEngine_ModelArtifact ma{nullptr, b, nullptr, b, nullptr, nullptr, nullptr, b};
    return ma;
}

// 非空伪 omc buffer（零填充非法图内容）：图位已配置 ⇒ 2026-09-22 严格加载 →
// ModelExecutor::Load 失败 → 构造内终止初始化并卸载双执行器 → ExecutorCreate
// 返回失败码、*out 保持 NULL（不交出半初始化实例）。
// buffer 态镜像 c_api_smoke.c 第 2 节 missing_artifact。
CANN_LMEngine_ModelArtifact MakeMissingModelArtifact() {
    static char buf[8] = {0};
    CANN_LMEngine_Buffer fake{buf, sizeof(buf)};
    CANN_LMEngine_ModelArtifact ma{nullptr, fake, nullptr, fake, nullptr, nullptr,
                                   nullptr, {nullptr, 0}};
    return ma;
}
}  // namespace

// UT-TC-CAPI-FB-001 (NEW-12): buffer 态构件（零长度 buffer = 未提供 → 无模型形态）
// 经单入口构造成功（与 path 态空构件同行为）
TEST(CApiTest, CreateFromBufferSucceedsStubEra_TC001) {
    CANN_LMEngine_ModelArtifact ma = MakeDummyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    auto s = CANN_LMEngine_ExecutorCreate(&ma, &opts, &exec);
    EXPECT_EQ(s, CANN_LM_ENGINE_SUCCESS);
    EXPECT_NE(exec, nullptr);
    CANN_LMEngine_ExecutorDestroy(exec);
}

// UT-TC-CAPI-FB-006 (2026-09-22 严格加载门控): 非空伪 omc buffer → 图位已配置 ⇒
// 严格加载 → Load 失败 → 构造中止初始化并卸载 → create 返回失败码且 *out 保持
// NULL（不交出半初始化实例）。与 C++ 侧 ut_engine_executor 的
// LoadFailureAbortsInitAndReports 分层覆盖（C ABI 不交出面在其上）。
TEST(CApiTest, CreateFromInvalidBufferAbortsInit_TC104) {
    CANN_LMEngine_ModelArtifact ma = MakeMissingModelArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    auto s = CANN_LMEngine_ExecutorCreate(&ma, &opts, &exec);
    EXPECT_NE(s, CANN_LM_ENGINE_SUCCESS);
    EXPECT_EQ(exec, nullptr);   // 失败不交出：*out 保持 NULL
}

// UT-TC-CAPI-FB-002/003 (NEW-13/14 收敛): artifact=nullptr → INVALID_ARGUMENT
TEST(CApiTest, CreateNullArtifactReturnsInvalidArgument_TC101) {
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    auto s = CANN_LMEngine_ExecutorCreate(nullptr, &opts, &exec);
    EXPECT_EQ(s, CANN_LM_ENGINE_INVALID_ARGUMENT);
    EXPECT_EQ(exec, nullptr);
}

// UT-TC-CAPI-FB-004 (NEW-15): out=nullptr → INVALID_ARGUMENT
TEST(CApiTest, CreateNullOutReturnsInvalidArgument_TC103) {
    CANN_LMEngine_ModelArtifact ma = MakeDummyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    auto s = CANN_LMEngine_ExecutorCreate(&ma, &opts, nullptr);
    EXPECT_EQ(s, CANN_LM_ENGINE_INVALID_ARGUMENT);
}

// UT-TC-CAPI-FB-005 (NEW-16): buffer 态构造的 Executor 支持后续 ContextCreate
// 验证 init_pipeline 后 pipeline_ 真就绪（不只是 ctor 不抛异常）。
TEST(CApiTest, CreateFromBufferSupportsContextCreate_TC002) {
    CANN_LMEngine_ModelArtifact ma = MakeDummyArtifact();
    CANN_LMEngine_EngineOptions opts{};
    CANN_LMEngine_Executor* exec = nullptr;
    ASSERT_EQ(CANN_LMEngine_ExecutorCreate(&ma, &opts, &exec),
              CANN_LM_ENGINE_SUCCESS);

    CANN_LMEngine_ContextId id = 0;
    auto s = CANN_LMEngine_ContextCreate(exec, &id);
    EXPECT_EQ(s, CANN_LM_ENGINE_SUCCESS);
    EXPECT_NE(id, 0u);

    EXPECT_EQ(CANN_LMEngine_ContextDestroy(exec, id), CANN_LM_ENGINE_SUCCESS);
    CANN_LMEngine_ExecutorDestroy(exec);
}

}  // namespace
