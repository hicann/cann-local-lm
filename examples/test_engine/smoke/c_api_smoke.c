// ═══════════════════════════════════════════════════════════════════════════
// C API smoke test — compiled with a C compiler (not C++).
// Verifies that the C header compiles, links, and runs correctly from pure C.
// ═══════════════════════════════════════════════════════════════════════════
#include "api/c/c_api.h"

#include <assert.h>
#include <stdio.h>

int main(void)
{
    CANN_LMEngine_Status s;

    // 1. NULL-parameter guards
    s = CANN_LMEngine_ExecutorCreate(NULL, NULL, NULL);
    assert(s == CANN_LM_ENGINE_INVALID_ARGUMENT);

    s = CANN_LMEngine_ExecutorGenerate(NULL, "hi", NULL, NULL, NULL);
    assert(s == CANN_LM_ENGINE_INVALID_ARGUMENT);

    // 2. Model paths configured but load fails (files absent) → create aborts:
    //    engine terminates init, unloads, and returns the error without
    //    handing out a half-initialized executor (2026-09-22 contract).
    CANN_LMEngine_ModelArtifact missing_artifact = {"/tmp/fake_prefill.omc",
                                                    {NULL, 0},
                                                    "/tmp/fake_decode.omc",
                                                    {NULL, 0},
                                                    "/tmp/fake_prefill.weight",
                                                    "/tmp/fake_decode.weight",
                                                    NULL,
                                                    {NULL, 0}};
    CANN_LMEngine_Executor* exec = NULL;
    s = CANN_LMEngine_ExecutorCreate(&missing_artifact, NULL, &exec);
    assert(s != CANN_LM_ENGINE_SUCCESS);
    assert(exec == NULL);

    // 3. No-model artifact (all slots empty, stub-era form) → create succeeds;
    //    lifecycle API usable, inference reports not-ready.
    CANN_LMEngine_ModelArtifact empty_artifact = {NULL, {NULL, 0}, NULL, {NULL, 0}, NULL, NULL, NULL, {NULL, 0}};
    s = CANN_LMEngine_ExecutorCreate(&empty_artifact, NULL, &exec);
    assert(s == CANN_LM_ENGINE_SUCCESS);
    assert(exec != NULL);

    // 4. ContextCreate succeeds (pipeline creates real context)
    CANN_LMEngine_ContextId ctx_id = 0;
    s = CANN_LMEngine_ContextCreate(exec, &ctx_id);
    assert(s == CANN_LM_ENGINE_SUCCESS);
    assert(ctx_id != 0);

    // 5. (ADR-0002) Generate with no active context -> INVALID_STATE (S2);
    //    SetContext unregistered id -> NOT_FOUND (S3); activate created id -> SUCCESS
    CANN_LMEngine_Result result = {NULL, 0, NULL, CANN_LM_ENGINE_FINISH_MAX_TOKENS};
    s = CANN_LMEngine_ExecutorGenerate(exec, "hello", NULL, NULL, &result);
    assert(s == CANN_LM_ENGINE_INVALID_STATE);
    s = CANN_LMEngine_ExecutorSetContext(exec, 999);
    assert(s == CANN_LM_ENGINE_NOT_FOUND);
    s = CANN_LMEngine_ExecutorSetContext(exec, ctx_id);
    assert(s == CANN_LM_ENGINE_SUCCESS);

    // 5b. Generate with active context on no-model executor -> error (not SUCCESS)
    s = CANN_LMEngine_ExecutorGenerate(exec, "hello", NULL, NULL, &result);
    assert(s != CANN_LM_ENGINE_SUCCESS);
    CANN_LMEngine_ResultDestroy(&result);

    // 6. ContextCancel on created context -> SUCCESS
    s = CANN_LMEngine_ContextCancel(exec, ctx_id);
    assert(s == CANN_LM_ENGINE_SUCCESS);

    // 7. RequestGetStatus on NULL -> QUEUED (safe default)
    CANN_LMEngine_RequestStatus rs = CANN_LMEngine_RequestGetStatus(NULL);
    assert(rs == CANN_LM_ENGINE_REQUEST_QUEUED);
    (void)rs;

    // 8. RequestIsDone on NULL -> 0 (false)
    assert(CANN_LMEngine_RequestIsDone(NULL) == 0);

    // 9. RequestCancel on NULL -> SUCCESS (idempotent)
    s = CANN_LMEngine_RequestCancel(NULL);
    assert(s == CANN_LM_ENGINE_SUCCESS);

    // 10. ResultDestroy on already-clean Result (safe no-op)
    CANN_LMEngine_ResultDestroy(&result);
    assert(result.token_ids == NULL);
    assert(result.token_ids_count == 0);
    assert(result.text == NULL);

    // 11. ContextDestroy on created context -> SUCCESS (idempotent)
    s = CANN_LMEngine_ContextDestroy(exec, ctx_id);
    assert(s == CANN_LM_ENGINE_SUCCESS);

    // 12. ContextDestroy on non-existent context -> SUCCESS (idempotent)
    s = CANN_LMEngine_ContextDestroy(exec, 999);
    assert(s == CANN_LM_ENGINE_SUCCESS);

    // 13. ExecutorDestroy (no crash)
    CANN_LMEngine_ExecutorDestroy(exec);

    // 14. ExecutorDestroy on NULL (safe no-op)
    CANN_LMEngine_ExecutorDestroy(NULL);

    printf("c_api_smoke: all assertions passed\n");
    (void)s;
    return 0;
}
