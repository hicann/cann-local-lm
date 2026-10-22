// tests/stub_ndk/probe_concurrent_runasync.cpp
// H4: 8 concurrent RunAsync -> no race/UAF/leak (AsyncRunCtx per-call; listener shared).
#include "probe_common.h"
#include <thread>
#include <vector>
#include <atomic>
using namespace cann_llm_engine;

int main() {
    probe::ManagerEnv e;
    if (!probe::setup(e)) return 1;
    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    probe::makeTensors(*e.built, e.type, inputs, outputs);

    const int N = 8;
    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&, i]() {
            // each thread needs its own input/output tensors (datasets are per-call)
            std::vector<std::shared_ptr<ITensorBuffer>> tin, tout;
            probe::makeTensors(*e.built, e.type, tin, tout);
            Status st = e.mm->RunAsync(probe::makeCtx(static_cast<uint64_t>(100 + i)), tin, tout, 0);
            if (st == Status::SUCCESS) successCount.fetch_add(1);
        });
    }
    for (auto& t : threads) t.join();

    fprintf(stderr, "[probe] concurrent: success=%d/%d listener.runFired=%d\n",
            successCount.load(), N, static_cast<int>(e.listener->runFired));
    if (successCount.load() != N) {
        fprintf(stderr, "FAIL: only %d/%d RunAsync succeeded\n", successCount.load(), N);
        return 1;
    }
    e.mm->DeInit();
    printf("probe_concurrent_runasync: PASS (8 concurrent, no crash/UAF)\n");
    return 0;
}
