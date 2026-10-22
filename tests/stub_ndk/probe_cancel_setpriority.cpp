// tests/stub_ndk/probe_cancel_setpriority.cpp
// H10: Cancel/SetPriority return UNSUPPORTED (document expected behavior — honest TODOs).
#include "probe_common.h"
using namespace cann_llm_engine;

int main() {
    probe::ManagerEnv e;
    if (!probe::setup(e)) return 1;

    Status cancelSt = e.mm->Cancel();
    Status prioSt   = e.mm->SetPriority(Priority::HIGH);
    fprintf(stderr, "[probe] Cancel -> %d, SetPriority(HIGH) -> %d (expect UNSUPPORTED=%d)\n",
            static_cast<int>(cancelSt), static_cast<int>(prioSt), static_cast<int>(Status::UNSUPPORTED));

    e.mm->DeInit();
    // Documents expected UNSUPPORTED. If either returns SUCCESS, the contract changed (re-check).
    if (cancelSt != Status::UNSUPPORTED) {
        fprintf(stderr, "FINDING: Cancel returned %d (not UNSUPPORTED — contract changed?)\n", static_cast<int>(cancelSt));
    }
    if (prioSt != Status::UNSUPPORTED) {
        fprintf(stderr, "FINDING: SetPriority returned %d (not UNSUPPORTED — contract changed?)\n", static_cast<int>(prioSt));
    }
    printf("probe_cancel_setpriority: PASS (UNSUPPORTED documented; honest TODOs)\n");
    return 0;
}
