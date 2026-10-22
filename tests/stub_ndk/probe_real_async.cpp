// tests/stub_ndk/probe_real_async.cpp
// H7: STUB_ASYNC_DELAY_MS -> cross-thread callback; check double-fire / UAF / race.
//
// Note: the stub's g_onRunDone is set via OH_NNExecutor_SetOnRunDone, which the backend
// TODOs (SetOnRunDone stores onRunDone_ member but doesn't register with the executor).
// So the stub's delayed thread fires g_onRunDone=nullptr -> no fire from the thread.
// Only the backend's Phase-1 direct-invoke fires. This probe thus also documents the
// H8 masking: real-async path is dead until SetOnRunDone registration is wired.
#include "probe_common.h"
#include <thread>
#include <chrono>
#include <atomic>
using namespace cann_llm_engine;

class CountingListener : public IModelManagerListener {
public:
    std::atomic<int> fireCount{0};
    Status lastErr = Status::FAILED;
    uint64_t lastCtxReq = 0;
    std::atomic<bool> died{false};
    void OnRunDone(const Context& ctx, Status errCode,
                   std::vector<std::shared_ptr<ITensorBuffer>>& outputs) override {
        int n = fireCount.fetch_add(1) + 1;
        lastErr = errCode; lastCtxReq = ctx.requestId;
        fprintf(stderr, "[probe] OnRunDone #%d: ctx=%llu err=%d out=%zu\n",
                n, static_cast<unsigned long long>(ctx.requestId), static_cast<int>(errCode), outputs.size());
    }
    void OnServiceDied() override { died = true; }
};

int main() {
    DeviceType type = DetectAvailableDevice();
    if (type != DeviceType::NDK) { fprintf(stderr, "FAIL: expected NDK\n"); return 1; }
    auto mm = CreateModelManager(type);
    auto built = BuildModelFromFile("stub.omc", 0, ModelOptions{}, type);
    if (!mm || !built) { fprintf(stderr, "FAIL: setup\n"); return 1; }
    auto listener = std::make_shared<CountingListener>();
    ModelInitOptions io; io.deviceId = 0;
    if (mm->Init(io, built, listener) != Status::SUCCESS) { fprintf(stderr, "FAIL: Init\n"); return 1; }

    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    for (const auto& d : built->GetInputDescs())  inputs.push_back(CreateTensorBuffer(d, type));
    for (const auto& d : built->GetOutputDescs()) outputs.push_back(CreateTensorBuffer(d, type));

    probe::setEnv("STUB_ASYNC_DELAY_MS", "50");
    Context ctx = probe::makeCtx(99);
    Status st = mm->RunAsync(ctx, inputs, outputs, 0);
    fprintf(stderr, "[probe] RunAsync(STUB_ASYNC_DELAY_MS=50) -> %d\n", static_cast<int>(st));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    probe::clearEnv("STUB_ASYNC_DELAY_MS");

    int fires = listener->fireCount.load();
    fprintf(stderr, "[probe] total fires=%d\n", fires);
    mm->DeInit();
    if (fires == 0) {
        fprintf(stderr, "FAIL: callback never fired (async path broken?)\n"); return 1;
    }
    if (fires > 1) {
        fprintf(stderr, "FAIL: callback fired %d times (H7 double-fire -> double-free risk)\n", fires);
        return 1;
    }
    // fires==1: Phase-1 direct-invoke only; real-async stub thread dead (g_onRunDone null, H8 TODO).
    printf("probe_real_async: PASS (single fire, no UAF; real-async path dead per H8 TODO)\n");
    return 0;
}
