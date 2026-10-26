// tests/stub_ndk/probe_common.h
// Shared setup/teardown + listener + env helpers for backend bug-sweep probes (Round 2).
//
// Provides:
//   probe::TestListener  — captures OnRunDone (fired/err/ctxReqId/outCount) + OnServiceDied.
//   probe::setEnv/clearEnv — portable env-var set/unset (_putenv_s on Windows, setenv else).
//   probe::ManagerEnv    — holds mm/built/listener/type from setup().
//   probe::setup(e, modelPath="stub.omc") — full Load: Detect→CreateModelManager→Build→Init.
//   probe::makeTensors(built, type, in, out) — create input/output ITensorBuffers from descs.
//   probe::makeCtx(reqId=42) — Context with userData + requestId.
#pragma once
#include "backend_base/factory.h"  // DetectAvailableDevice/CreateModelManager/BuildModelFromFile/CreateTensorBuffer
#include "backend_base/listener.h" // IModelManagerListener
#include "backend_base/types.h"    // Context/ModelInitOptions
#include "device_base/cann_device_types.h" // Status/DeviceType/ModelOptions/TensorDesc

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace probe {

class TestListener : public cann_llm_engine::IModelManagerListener {
public:
    bool runFired = false;
    cann_llm_engine::Status runErr = cann_llm_engine::Status::FAILED;
    uint64_t ctxReqId = 0;
    size_t outCount = 0;
    bool diedFired = false;
    void OnRunDone(const cann_llm_engine::Context& ctx, cann_llm_engine::Status errCode,
                   std::vector<std::shared_ptr<cann_llm_engine::ITensorBuffer>>& outputs) override
    {
        runFired = true;
        runErr = errCode;
        ctxReqId = ctx.requestId;
        outCount = outputs.size();
        fprintf(stderr, "[probe] OnRunDone: ctx=%llu err=%d out=%zu\n", static_cast<unsigned long long>(ctx.requestId),
                static_cast<int>(errCode), outputs.size());
    }
    void OnServiceDied() override
    {
        diedFired = true;
        fprintf(stderr, "[probe] OnServiceDied fired\n");
    }
};

inline void setEnv(const char* k, const char* v)
{
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
inline void clearEnv(const char* k) { setEnv(k, ""); }

struct ManagerEnv {
    std::shared_ptr<cann_llm_engine::IModelManager> mm;
    std::shared_ptr<cann_llm_engine::IBuiltModel> built;
    std::shared_ptr<TestListener> listener;
    cann_llm_engine::DeviceType type = cann_llm_engine::DeviceType::NONE;
};

// Full Load setup: Detect → CreateModelManager → BuildModelFromFile → Init(listener).
// Returns false (with stderr) on any failure. modelPath default "stub.omc".
inline bool setup(ManagerEnv& e, const char* modelPath = "stub.omc")
{
    using namespace cann_llm_engine;
    e.type = DetectAvailableDevice();
    if (e.type != DeviceType::NDK) {
        fprintf(stderr, "FAIL: DetectAvailableDevice=%d != NDK (stub .so present?)\n", static_cast<int>(e.type));
        return false;
    }
    e.mm = CreateModelManager(e.type);
    if (!e.mm) {
        fprintf(stderr, "FAIL: CreateModelManager null\n");
        return false;
    }
    e.built = BuildModelFromFile(modelPath, 0, ModelOptions{}, e.type);
    if (!e.built) {
        fprintf(stderr, "FAIL: BuildModelFromFile null (stub mode or .so?)\n");
        return false;
    }
    e.listener = std::make_shared<TestListener>();
    ModelInitOptions io;
    io.deviceId = 0;
    if (e.mm->Init(io, e.built, e.listener) != Status::SUCCESS) {
        fprintf(stderr, "FAIL: Init\n");
        return false;
    }
    return true;
}

inline void makeTensors(const cann_llm_engine::IBuiltModel& built, cann_llm_engine::DeviceType type,
                        std::vector<std::shared_ptr<cann_llm_engine::ITensorBuffer>>& inputs,
                        std::vector<std::shared_ptr<cann_llm_engine::ITensorBuffer>>& outputs)
{
    for (const auto& d : built.GetInputDescs())
        inputs.push_back(CreateTensorBuffer(d, type));
    for (const auto& d : built.GetOutputDescs())
        outputs.push_back(CreateTensorBuffer(d, type));
}

inline cann_llm_engine::Context makeCtx(uint64_t reqId = 42)
{
    cann_llm_engine::Context ctx;
    ctx.userData = reinterpret_cast<void*>(0x1234);
    ctx.requestId = reqId;
    return ctx;
}

} // namespace probe
