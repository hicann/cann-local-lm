/**
 * Copyright (c) 2025. (C3 fill — was an empty placeholder "install package to override".)
 *
 * ACL model API surface — declares the aclmdl* functions + types that the vendored
 * acl_base_rt.h/acl_rt.h do NOT cover. The shim (hiai_ascendcl) DEFINES these in
 * acl_mdl.cpp / acl_data.cpp; the backend (ndk/) CALLS them via this header.
 *
 * NOTE on linkage: real ACL's acl_mdl.h declares these `extern "C"`. The shim's
 * definitions (in .cpp, no extern "C" keyword) inherit C linkage from these
 * declarations, so symbol linkage matches between shim (provider) and backend
 * (consumer). If a real ACL SDK overrides this file, the extern "C" declarations
 * are identical and the shim's inherited C linkage still matches.
 */
#ifndef ACL_ACL_MDL_H
#define ACL_ACL_MDL_H

#include "acl_base.h"   // aclError, aclDataType, aclFormat, aclDataBuffer, aclrtStream
#include <stddef.h>
#include <stdint.h>
#include <vector>       // C2 follow-on: aclmdlConfigHandle::inputTensorShapes

#ifdef __cplusplus
extern "C" {
#endif

/* ===== aclmdl I/O dims (C3: real ACL defines these in acl_mdl.h. The shim's
 * acl_data.h previously defined them — moved here so both shim + backend see one
 * definition. Layout matches real ACL for ABI compat.) ===== */
#ifndef ACL_MAX_DIM_CNT
#define ACL_MAX_DIM_CNT 8
#endif

struct aclmdlIODims {
    size_t dimCount;
    int64_t dims[ACL_MAX_DIM_CNT];
};

struct aclmdlIODimsRange {
    size_t rangeCount;
    int64_t range[ACL_MAX_DIM_CNT][2];
};

/* ===== Opaque ACL model types (full structs defined in the shim: aclmdlDesc +
 * aclmdlConfigHandle in acl_manager.h [global scope, C3], aclmdlDataset in
 * acl_data.h). Consumers treat them as opaque. ===== */
typedef struct aclmdlDesc aclmdlDesc;
typedef struct aclmdlDataset aclmdlDataset;
// C2: aclmdlConfigHandle is NON-opaque (full struct here, PUBLIC) so the backend can
// access members directly (shim's simplification; real ACL keeps it opaque + uses
// aclmdlSetConfigOpt — a future migration if real ACL is installed). Mirrors
// cann_llm_engine::ModelOptions — backend fills it, aclmdlFinishLoad reads + applies.
// Fields default to -1/false/nullptr so FinishLoad only applies the ones the backend set.
// C2 follow-on: inputTensorShapes — named input shapes for NDK compile-time fixed shapes.
struct aclmdlInputShape {
    const char* name;
    const int64_t* dims;
    size_t dimCount;
};
struct aclmdlConfigHandle {
    std::vector<aclmdlInputShape> inputTensorShapes;  // C2 follow-on: HIAI_NDK_HiAIOptions_SetInputTensorShapes
    size_t mdlLoadType = 0;
    const char* loadPath = nullptr;
    const void* mdlAddr = nullptr;
    size_t mdlSize = 0;
    int32_t priority = -1;              // OH_NN_Priority
    int32_t prefMode = -1;              // OH_NN_PerformanceMode
    bool enableFloat16 = false;         // OH_NNCompilation_EnableFloat16
    const char* cachePath = nullptr;    // OH_NNCompilation_SetCache
    uint32_t cacheVersion = 0;          // OH_NNCompilation_SetCache
    int32_t bandMode = -1;              // HiAI_BandMode            (C2 follow-on)
    int32_t tuningStrategy = -1;        // HiAI_TuningStrategy       (C2 follow-on)
    int32_t tuningMode = -1;            // HiAI_TuningMode           (C2 follow-on)
    const char* tuningCacheDir = nullptr;  // (C2 follow-on)
    int32_t fallbackMode = -1;          // HiAI_FallbackMode         (C2 follow-on)
    int32_t deviceMemoryReusePlan = -1; // HiAI_DeviceMemoryReusePlan(C2 follow-on)
    int32_t formatMode = -1;            // HiAI_FormatMode           (C2 follow-on)
    bool dynamicShapeStatus = false;    // HiAI_DynamicShapeStatus   (C2 follow-on)
    size_t dynamicShapeMaxCache = 0;    //                            (C2 follow-on)
    int32_t dynamicShapeCacheMode = -1; // HiAI_DynamicShapeCacheMode(C2 follow-on)
    const char* quantConfigData = nullptr;  // (C2 follow-on)
    const char* omOutputDir = nullptr;  // HiAI_OmType               (C2 follow-on)
    int32_t omType = -1;  // (C2 follow-on)
    bool enableDump = false;            // ACL-only, NDK ignores
    const char* dumpConfigPath = nullptr;
};
typedef void* aclmdlModel;

/* ===== Config handle ===== */
aclmdlConfigHandle* aclmdlCreateConfigHandle(void);
aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle* handle);

/* ===== Model load / unload ===== */
aclError aclmdlUnload(uint32_t modelId);
/* ===== Cache-based load (compiled model restoration, no compilation) ===== */
aclError aclmdlLoadFromCacheFile(const char* cacheFilePath, uint32_t* modelId);
aclError aclmdlLoadFromCacheBuffer(const void* buffer, size_t size, uint32_t* modelId);

/* ===== Model desc (IO query) ===== */
aclmdlDesc* aclmdlCreateDesc(void);
aclError aclmdlDestroyDesc(aclmdlDesc* modelDesc);
aclError aclmdlGetDesc(aclmdlDesc* modelDesc, uint32_t modelId);
size_t aclmdlGetNumInputs(aclmdlDesc* modelDesc);
size_t aclmdlGetNumOutputs(aclmdlDesc* modelDesc);
size_t aclmdlGetInputSizeByIndex(aclmdlDesc* modelDesc, size_t index);
size_t aclmdlGetOutputSizeByIndex(aclmdlDesc* modelDesc, size_t index);
const char* aclmdlGetInputNameByIndex(const aclmdlDesc* modelDesc, size_t index);
const char* aclmdlGetOutputNameByIndex(const aclmdlDesc* modelDesc, size_t index);
aclFormat aclmdlGetInputFormat(const aclmdlDesc* modelDesc, size_t index);
aclFormat aclmdlGetOutputFormat(const aclmdlDesc* modelDesc, size_t index);
aclDataType aclmdlGetInputDataType(const aclmdlDesc* modelDesc, size_t index);
aclDataType aclmdlGetOutputDataType(const aclmdlDesc* modelDesc, size_t index);
aclError aclmdlGetInputDims(const aclmdlDesc* modelDesc, size_t index, aclmdlIODims* dims);
aclError aclmdlGetOutputDims(const aclmdlDesc* modelDesc, size_t index, aclmdlIODims* dims);
aclError aclmdlGetCurOutputDims(const aclmdlDesc* modelDesc, size_t index, aclmdlIODims* dims);
aclError aclmdlGetInputDimsRange(const aclmdlDesc* modelDesc, size_t index, aclmdlIODimsRange* dimsRange);

/* ===== Dataset (IO container) ===== */
aclmdlDataset* aclmdlCreateDataset(void);
aclError aclmdlDestroyDataset(const aclmdlDataset* dataset);
aclError aclmdlAddDatasetBuffer(aclmdlDataset* dataset, aclDataBuffer* dataBuffer);
aclDataBuffer* aclmdlGetDatasetBuffer(const aclmdlDataset* dataset, size_t index);
size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset* dataset);

/* ===== Execute ===== */
aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset* input, aclmdlDataset* output);
aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset* input, aclmdlDataset* output, aclrtStream stream);

/* ===== Dynamic shape ===== */
aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset* dataset, size_t index, const aclmdlIODims* dims);

#ifdef __cplusplus
}
#endif

#endif  // ACL_ACL_MDL_H
