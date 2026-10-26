// tests/stub_ndk/probe_service_died.cpp
// H8: STUB_FIRE_SERVICE_DIED -> should fire OnServiceDied; confirms SetOnServiceDied TODO gap.
#include "probe_common.h"
using namespace cann_llm_engine;

int main()
{
    probe::ManagerEnv e;
    if (!probe::setup(e))
        return 1;
    std::vector<std::shared_ptr<ITensorBuffer>> inputs, outputs;
    probe::makeTensors(*e.built, e.type, inputs, outputs);

    probe::setEnv("STUB_FIRE_SERVICE_DIED", "1");
    Status st = e.mm->Run(inputs, outputs); // stub fires g_onServiceDied here (if registered)
    probe::clearEnv("STUB_FIRE_SERVICE_DIED");
    fprintf(stderr, "[probe] Run(STUB_FIRE_SERVICE_DIED) -> %d, listener.diedFired=%d\n", static_cast<int>(st),
            e.listener->diedFired);
    e.mm->DeInit();

    if (e.listener->diedFired) {
        printf("probe_service_died: PASS (service-died callback fired through to listener)\n");
    } else {
        // H8 confirmed: SetOnServiceDied TODO -> registration gap -> callback never reaches listener.
        printf("probe_service_died: FINDING (TODO confirmed) - SetOnServiceDied not registered "
               "via OH_NNExecutor_SetOnServiceDied; callback never fires. (expected per spec H8)\n");
    }
    return 0; // exit 0 either way: documents the gap, doesn't assert a fix
}
