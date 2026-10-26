// tests/stub_ndk/probe_getoutputshape.cpp
// H9: GetOutputShape code path (aclmdl*Desc) — dims correct, desc destroyed, bad index graceful.
// IModelManager doesn't expose GetOutputShape (it's on IDeviceModelManager); this probe calls the
// shim's aclmdl* path directly (NdkDeviceModelManager::GetOutputShape is a thin wrapper over these)
// with the modelId from mm->GetModelID().
#include "probe_common.h"
#include "acl/acl_mdl.h" // shim ACL model API (aclmdlCreateDesc/GetDesc/GetCurOutputDims/DestroyDesc)
using namespace cann_llm_engine;

int main()
{
    probe::ManagerEnv e;
    if (!probe::setup(e))
        return 1;

    uint32_t modelId = e.mm->GetModelID();
    fprintf(stderr, "[probe] modelId=%u\n", modelId);
    if (modelId == 0) {
        fprintf(stderr, "FAIL: GetModelID=0 (not loaded?)\n");
        return 1;
    }

    // H9(a): index 0 -> dims {1,349}
    aclmdlDesc* desc = aclmdlCreateDesc();
    if (!desc) {
        fprintf(stderr, "FAIL: aclmdlCreateDesc null\n");
        return 1;
    }
    aclError ret = aclmdlGetDesc(desc, modelId);
    fprintf(stderr, "[probe] aclmdlGetDesc -> %d\n", static_cast<int>(ret));
    if (ret != ACL_SUCCESS) {
        aclmdlDestroyDesc(desc);
        fprintf(stderr, "FAIL: aclmdlGetDesc\n");
        return 1;
    }

    aclmdlIODims dims{};
    ret = aclmdlGetCurOutputDims(desc, 0, &dims); // index 0
    fprintf(stderr, "[probe] GetCurOutputDims(0) -> %d dimCount=%zu\n", static_cast<int>(ret), dims.dimCount);
    aclmdlDestroyDesc(desc); // H9: must destroy (leak check via no-crash + repeat)
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "FAIL: GetCurOutputDims(0)\n");
        return 1;
    }
    // stub canned output shape = {1,349}
    if (dims.dimCount < 2 || dims.dims[0] != 1 || dims.dims[1] != 349) {
        fprintf(stderr, "FAIL: dims mismatch (got dimCount=%zu", dims.dimCount);
        for (size_t i = 0; i < dims.dimCount; ++i)
            fprintf(stderr, " %lld", static_cast<long long>(dims.dims[i]));
        fprintf(stderr, ", expected {1,349})\n");
        return 1;
    }

    // H9(b): bad index -> graceful (not crash)
    aclmdlDesc* desc2 = aclmdlCreateDesc();
    aclmdlGetDesc(desc2, modelId);
    aclmdlIODims dims2{};
    ret = aclmdlGetCurOutputDims(desc2, 999, &dims2); // bad index
    aclmdlDestroyDesc(desc2);
    fprintf(stderr, "[probe] GetCurOutputDims(999) -> %d (expect error, no crash)\n", static_cast<int>(ret));
    if (ret == ACL_SUCCESS) {
        fprintf(stderr, "WARN: bad index returned SUCCESS\n");
    }

    e.mm->DeInit();
    printf("probe_getoutputshape: PASS (dims {1,349}, bad index graceful, no crash)\n");
    return 0;
}
