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

    // (ADR-0002) GenerateAsync with no active context -> INVALID_STATE (S2)
    cann_lm::engine::Prompt p;
    p.text = "hi";
    cann_lm::engine::SamplingParams sp;
    cann_lm::engine::RequestHandle h;
    s = exec.GenerateAsync(p, sp, cann_lm::engine::RequestOptions{}, &h);
    if (s.code != cann_lm::engine::Status::INVALID_STATE) {
        return 3;
    }

    // SetContext unregistered id -> NOT_FOUND (S3); activate created -> SUCCESS
    s = exec.SetContext(cann_lm::engine::ContextId{999});
    if (s.code != cann_lm::engine::Status::NOT_FOUND) {
        return 4;
    }
    s = exec.SetContext(ctx.id());
    if (!s.ok()) {
        return 5;
    }

    // GenerateAsync with active context on no-model executor -> error
    s = exec.GenerateAsync(p, sp, cann_lm::engine::RequestOptions{}, &h);
    if (s.ok()) {
        return 6;
    }

    // Cancel on empty handle → SUCCESS (idempotent)
    cann_lm::engine::Status cs = h.Cancel();
    if (!cs.ok()) {
        return 7;
    }

    // ContextCancel on created (active) context → SUCCESS
    cann_lm::engine::ContextId cid = ctx.id();
    s = exec.CancelContext(cid);
    if (!s.ok()) {
        return 8;
    }

    // DestroyContext → SUCCESS (idempotent)
    s = exec.DestroyContext(cid);
    if (!s.ok()) {
        return 9;
    }

    return 0;
}
