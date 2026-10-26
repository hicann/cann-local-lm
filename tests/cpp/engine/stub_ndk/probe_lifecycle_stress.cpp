// tests/stub_ndk/probe_lifecycle_stress.cpp
// H5: repeated Load/Run/DeInit cycles -> no leak/double-free (CannRuntime refcount + aclmdlUnload).
#include "probe_common.h"
using namespace cann_llm_engine;

int main()
{
    const int N = 5; // cycles
    const int K = 3; // Runs per cycle
    for (int c = 0; c < N; ++c) {
        probe::ManagerEnv e;
        if (!probe::setup(e)) {
            fprintf(stderr, "FAIL: setup failed on cycle %d\n", c);
            return 1;
        }
        std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
        probe::makeTensors(*e.built, e.type, inputs, outputs);
        for (int r = 0; r < K; ++r) {
            Context ctx = probe::makeCtx(static_cast<uint64_t>(c * 100 + r));
            Status st = e.mm->RunAsync(ctx, inputs, outputs, 0);
            if (st != Status::SUCCESS) {
                fprintf(stderr, "FAIL: Run[%d,%d] -> %d\n", c, r, static_cast<int>(st));
                return 1;
            }
        }
        e.mm->DeInit();
        fprintf(stderr, "[probe] cycle %d: %d Runs + DeInit OK\n", c, K);
    }
    printf("probe_lifecycle_stress: PASS (%d cycles x %d Runs, no crash)\n", N, K);
    return 0;
}
