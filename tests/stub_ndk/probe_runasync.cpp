/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file tests/stub_ndk/probe_runasync.cpp
 * @brief Probe:验证 B2+B3 async 修复(RunAsync dataset 生命周期 + OnRunDone 回调)。
 *
 * 直调 ModelManagerImpl::RunAsync(bbit 走 sync Run 不触达),用 TestListener 捕获
 * OnRunDone,验证:B3 回调触发 + ctx 非竞态(requestId 回传对)+ outputs 非空;
 * B2 不 UAF 崩(datasets 在回调销毁,非 RunAsync 提前销毁)。
 * 链 cann_backend(真 backend)+ 需 stub .so 旁 exe(NDKProxy dlopen)。
 */
#include "backend_base/factory.h"               // CreateModelManager/BuildModelFromFile/CreateTensorBuffer/DetectAvailableDevice
#include "backend_base/listener.h"             // IModelManagerListener
#include "backend_base/types.h"                 // Context/ModelInitOptions
#include "device_base/cann_device_types.h"      // Status/DeviceType/TensorDesc/ModelOptions

#include <cstdio>
#include <memory>
#include <vector>

using namespace cann_llm_engine;

class TestListener : public IModelManagerListener {
public:
    bool      fired    = false;
    Status    err      = Status::FAILED;
    uint64_t  ctxReqId = 0;
    size_t    outCount = 0;
    void OnRunDone(const Context& ctx, Status errCode,
                   std::vector<std::shared_ptr<ITensorBuffer>>& outputs) override {
        fired = true;
        err = errCode;
        ctxReqId = ctx.requestId;
        outCount = outputs.size();
        fprintf(stderr, "[probe] OnRunDone fired: ctx.requestId=%llu err=%d outCount=%zu\n",
                static_cast<unsigned long long>(ctx.requestId), static_cast<int>(errCode), outputs.size());
    }
};

int main() {
    DeviceType type = DetectAvailableDevice();
    fprintf(stderr, "[probe] DetectAvailableDevice -> %d\n", static_cast<int>(type));
    if (type != DeviceType::NDK) { fprintf(stderr, "FAIL: expected NDK (stub .so present?)\n"); return 1; }

    auto mm = CreateModelManager(type);
    if (!mm) { fprintf(stderr, "FAIL: CreateModelManager null\n"); return 1; }

    auto built = BuildModelFromFile("stub.omc", 0, ModelOptions{}, type);
    if (!built) { fprintf(stderr, "FAIL: BuildModelFromFile null (stub .so not loaded?)\n"); return 1; }

    auto listener = std::make_shared<TestListener>();
    ModelInitOptions io;
    io.deviceId = 0;
    if (mm->Init(io, built, listener) != Status::SUCCESS) {
        fprintf(stderr, "FAIL: Init failed\n"); return 1;
    }
    fprintf(stderr, "[probe] Init OK (listener registered -> SetOnRunDone called -> fixed lambda wired)\n");

    auto inDescs  = built->GetInputDescs();
    auto outDescs = built->GetOutputDescs();
    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    for (const auto& d : inDescs)  inputs.push_back(CreateTensorBuffer(d, type));
    for (const auto& d : outDescs) outputs.push_back(CreateTensorBuffer(d, type));
    fprintf(stderr, "[probe] tensors: %zu in / %zu out\n", inputs.size(), outputs.size());

    Context ctx;
    ctx.userData  = reinterpret_cast<void*>(0x1234);
    ctx.requestId = 42;
    Status st = mm->RunAsync(ctx, inputs, outputs, 0);
    fprintf(stderr, "[probe] RunAsync -> %d\n", static_cast<int>(st));
    if (st != Status::SUCCESS) { fprintf(stderr, "FAIL: RunAsync not SUCCESS\n"); return 1; }

    // B3:回调触发(ExecuteAsync sync 后直接 invoke onRunDone_)
    if (!listener->fired) { fprintf(stderr, "FAIL: OnRunDone not fired (B3 callback未接通)\n"); return 1; }
    if (listener->err != Status::SUCCESS) {
        fprintf(stderr, "FAIL: OnRunDone err=%d != SUCCESS\n", static_cast<int>(listener->err)); return 1; }
    // B3:ctx 非竞态(requestId 经 AsyncRunCtx→userData 回传,非 racy currentCtx_ 成员)
    if (listener->ctxReqId != 42) {
        fprintf(stderr, "FAIL: ctx.requestId=%llu != 42 (B3 ctx 竞态? currentCtx_ 仍用?)\n",
                static_cast<unsigned long long>(listener->ctxReqId)); return 1; }
    // B3:outputs 非空(经 runCtx->outputs 直传 listener,非原空 vector)
    if (listener->outCount != outDescs.size()) {
        fprintf(stderr, "FAIL: outputs count=%zu != %zu (B3 空 outputs?)\n",
                listener->outCount, outDescs.size()); return 1; }
    // B2:不 UAF 崩(datasets 在回调销毁,非 RunAsync 提前销毁)。到此未崩即过。
    fprintf(stderr, "[probe] B2(no UAF crash) + B3(fired, ctx=%llu, outputs=%zu) OK\n",
            static_cast<unsigned long long>(listener->ctxReqId), listener->outCount);

    mm->DeInit();
    printf("probe_runasync: ALL PASS\n");
    return 0;
}
