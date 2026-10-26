// tests/stub_ndk/probe_tensor_alloc_fail.cpp
// H6: STUB_TENSOR_ALLOC_FAIL -> nullptr tensor; Load/Run must not crash.
#include "probe_common.h"
using namespace cann_llm_engine;

int main()
{
    // setup with flag UNSET (so Init's own tensor allocs succeed)
    probe::ManagerEnv e;
    if (!probe::setup(e))
        return 1;
    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    probe::makeTensors(*e.built, e.type, inputs, outputs);

    // now flip the flag ON + Run -> shim's RunModel creates output tensors (nullptr) -> graceful?
    probe::setEnv("STUB_TENSOR_ALLOC_FAIL", "1");
    Status st = e.mm->Run(inputs, outputs);
    probe::clearEnv("STUB_TENSOR_ALLOC_FAIL");
    fprintf(stderr, "[probe] Run(STUB_TENSOR_ALLOC_FAIL) -> %d\n", static_cast<int>(st));
    // Reaching here without crash = pass. Run may return FAILED (nullptr tensor) — that's graceful.
    e.mm->DeInit();
    printf("probe_tensor_alloc_fail: PASS (nullptr tensor path, no crash)\n");
    return 0;
}
