// tests/stub_ndk/probe_run_failure.cpp
// H2: Run/RunAsync failure cleanup + empty/mismatched input edge cases.
#include "probe_common.h"
using namespace cann_llm_engine;

static int test_sync_fail() {
    probe::ManagerEnv e;
    if (!probe::setup(e)) return 1;
    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    probe::makeTensors(*e.built, e.type, inputs, outputs);
    probe::setEnv("STUB_RUN_FAIL", "1");
    Status st = e.mm->Run(inputs, outputs);
    probe::clearEnv("STUB_RUN_FAIL");
    fprintf(stderr, "[probe] Run(STUB_RUN_FAIL) -> %d\n", static_cast<int>(st));
    if (st == Status::SUCCESS) { fprintf(stderr, "FAIL: Run succeeded despite STUB_RUN_FAIL\n"); return 1; }
    e.mm->DeInit();
    return 0;  // no crash = datasets cleaned (Run destroys unconditionally)
}

static int test_async_fail() {
    probe::ManagerEnv e;
    if (!probe::setup(e)) return 1;
    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    probe::makeTensors(*e.built, e.type, inputs, outputs);
    // async path routes through RunSync (shim RunModelAsync wraps sync); fail RunSync there.
    probe::setEnv("STUB_RUN_FAIL", "1");
    Status st = e.mm->RunAsync(probe::makeCtx(7), inputs, outputs, 0);
    probe::clearEnv("STUB_RUN_FAIL");
    fprintf(stderr, "[probe] RunAsync(STUB_RUN_FAIL) -> %d fired=%d err=%d\n",
            static_cast<int>(st), e.listener->runFired, static_cast<int>(e.listener->runErr));
    if (!e.listener->runFired) { fprintf(stderr, "FAIL: OnRunDone not fired on async failure (datasets leak?)\n"); return 1; }
    if (e.listener->runErr == Status::SUCCESS) { fprintf(stderr, "FAIL: OnRunDone err=SUCCESS despite failure\n"); return 1; }
    e.mm->DeInit();
    return 0;  // reaching here = no crash (datasets cleaned in callback)
}

static int test_empty_inputs() {
    probe::ManagerEnv e;
    if (!probe::setup(e)) return 1;
    std::vector<std::shared_ptr<ITensorBuffer>> emptyIn, emptyOut;
    Status st = e.mm->Run(emptyIn, emptyOut);
    fprintf(stderr, "[probe] Run(empty) -> %d\n", static_cast<int>(st));
    if (st == Status::SUCCESS) { fprintf(stderr, "FAIL: Run accepted empty inputs\n"); return 1; }
    e.mm->DeInit();
    return 0;  // graceful reject = no crash
}

static int test_mismatched_inputs() {
    probe::ManagerEnv e;
    if (!probe::setup(e)) return 1;
    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    probe::makeTensors(*e.built, e.type, inputs, outputs);
    inputs.resize(1);  // stub expects 4; give 1
    Status st = e.mm->Run(inputs, outputs);
    fprintf(stderr, "[probe] Run(1 input, expected 4) -> %d\n", static_cast<int>(st));
    // graceful: either backend rejects (FAILED) or shim errors; must NOT crash
    e.mm->DeInit();
    return 0;  // no crash = pass
}

int main() {
    int rc = 0;
    fprintf(stderr, "=== test_sync_fail ===\n");      rc |= test_sync_fail();
    fprintf(stderr, "=== test_async_fail ===\n");     rc |= test_async_fail();
    fprintf(stderr, "=== test_empty_inputs ===\n");   rc |= test_empty_inputs();
    fprintf(stderr, "=== test_mismatched_inputs ===\n"); rc |= test_mismatched_inputs();
    if (rc == 0) printf("probe_run_failure: ALL PASS\n");
    else fprintf(stderr, "probe_run_failure: FAIL (see above)\n");
    return rc;
}
