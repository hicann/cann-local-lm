// tests/stub_ndk/probe_load_failure.cpp
// H1: STUB_COMPILE_FAIL -> BuildModelFromFile null -> Load fails cleanly (no crash).
//     Init(null built) should reject (INVALID_PARAMETER per model_manager_impl.h:53), not crash.
#include "probe_common.h"
using namespace cann_llm_engine;

int main()
{
    probe::setEnv("STUB_COMPILE_FAIL", "1");
    DeviceType type = DetectAvailableDevice();
    fprintf(stderr, "[probe] DetectAvailableDevice -> %d\n", static_cast<int>(type));
    if (type != DeviceType::NDK) {
        fprintf(stderr, "FAIL: expected NDK\n");
        return 1;
    }

    // BuildModelFromFile should return null (stub compile failed)
    auto built = BuildModelFromFile("stub.omc", 0, ModelOptions{}, type);
    if (built != nullptr) {
        fprintf(stderr, "FAIL: BuildModelFromFile returned non-null despite STUB_COMPILE_FAIL\n");
        return 1;
    }
    fprintf(stderr, "[probe] BuildModelFromFile -> null (stub compile fail honored) OK\n");

    // CreateModelManager + Init with null built should fail cleanly (not crash)
    auto mm = CreateModelManager(type);
    if (!mm) {
        fprintf(stderr, "FAIL: CreateModelManager null\n");
        return 1;
    }
    auto listener = std::make_shared<probe::TestListener>();
    ModelInitOptions io;
    io.deviceId = 0;
    Status st = mm->Init(io, built, listener);
    fprintf(stderr, "[probe] Init(null built) -> %d\n", static_cast<int>(st));
    // Init should reject null built (INVALID_PARAMETER per model_manager_impl.h:53), not crash
    if (st == Status::SUCCESS) {
        fprintf(stderr, "FAIL: Init accepted null built (should reject)\n");
        return 1;
    }
    mm->DeInit();
    probe::clearEnv("STUB_COMPILE_FAIL");
    printf("probe_load_failure: PASS (Load-failure path clean, no crash)\n");
    return 0;
}
