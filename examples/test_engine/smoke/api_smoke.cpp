#include "core/context.h"
#include "api/cpp/engine_executor.h"
#include "api/cpp/request_handle.h"
#include "infra/status.h"
#include "api/cpp/types.h"

int main()
{
    cann_lm::engine::EngineOptions opts;
    cann_lm::engine::ModelArtifact artifact{}; // SR20260903001：单一构件（双图位）
    cann_lm::engine::EngineExecutor exec(artifact, opts);

    // CreateContext now succeeds (pipeline creates real context)
    cann_lm::engine::Context ctx;
    cann_lm::engine::Status s = exec.CreateContext(&ctx);
    if (!s.ok()) {
        return 1;
    }
    if (!static_cast<bool>(ctx)) {
        return 2;
    }

    // Generate on valid context fails (stub tokenizer → PreparePrefill FAILURE)
    cann_lm::engine::Prompt p;
    p.text = "hi";
    cann_lm::engine::SamplingParams sp;
    cann_lm::engine::RequestHandle h;
    s = exec.GenerateAsync(p, ctx, sp, cann_lm::engine::RequestOptions{}, &h);
    if (s.ok()) {
        return 3;
    }

    // Cancel on empty handle → SUCCESS (idempotent)
    cann_lm::engine::Status cs = h.Cancel();
    if (!cs.ok()) {
        return 4;
    }

    // ContextCancel on created context → SUCCESS
    cann_lm::engine::ContextId cid = ctx.id();
    s = exec.CancelContext(cid);
    if (!s.ok()) {
        return 5;
    }

    // DestroyContext → SUCCESS (idempotent)
    s = exec.DestroyContext(cid);
    if (!s.ok()) {
        return 6;
    }

    return 0;
}
