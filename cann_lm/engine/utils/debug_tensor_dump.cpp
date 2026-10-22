// ═══════════════════════════════════════════════════════════════════════════
// 【临时调试代码 — 验证完成后删除】DebugTensorDump 实现。
// 详见 debug_tensor_dump.h 的说明。
// ═══════════════════════════════════════════════════════════════════════════
#include "utils/debug_tensor_dump.h"

#include "infra/log.h"
#include "utils/data_type_utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace cann_lm::engine {

namespace {

constexpr const char* kDumpEnv    = "CANN_LM_DEBUG_DUMP_TENSORS";
constexpr const char* kDumpKvEnv  = "CANN_LM_DEBUG_DUMP_KV";
constexpr const char* kDumpDirEnv = "CANN_LM_DEBUG_DUMP_DIR";
constexpr const char* kDefaultDir = "/data/local/tmp/dump/";

/// 非空 env 取值（返回 nullptr / 空串均视为未设置）。
const char* EnvVal(const char* name) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

/// 目录路径：取 env，缺省 kDefaultDir；确保以 '/' 结尾。
std::string DumpDir() {
    const char* v = EnvVal(kDumpDirEnv);
    std::string dir = (v != nullptr) ? v : kDefaultDir;
    if (dir.empty() || dir.back() != '/') dir += '/';
    return dir;
}

/// 确保目录存在（不存在则 mkdir；已存在不报错）。失败仅告警，由后续 fopen 兜底。
void EnsureDir(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) return;
    if (mkdir(dir.c_str(), 0775) != 0) {
        CANN_LLM_LOGW("debug.dump: mkdir '%s' failed, will try write anyway", dir.c_str());
    }
}

std::string DimsStr(const std::vector<int32_t>& dims) {
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(dims[i]);
    }
    s += "]";
    return s;
}

/// 工位名（空名回退 unnamed）。
std::string TensorName(const TensorDesc& desc) {
    return desc.tensorName.empty() ? std::string("unnamed") : desc.tensorName;
}

/// KV 池工位查找结果。
struct KvSlotRef {
    bool found = false;
    bool is_v  = false;   // false = K
    int  layer = -1;
};

/// 按 GetData() 指针在池中查找：命中即为 KV 工位（输出侧 KV 工位名未知，只能按指针识别）。
/// 同时校验 dims/dtype，避免指针巧合误判。
KvSlotRef FindKvSlot(const std::shared_ptr<ITensorBuffer>& t,
                     const std::shared_ptr<KvCachePool>& kv_pool) {
    KvSlotRef ref;
    if (!kv_pool || !t || !t->GetData()) return ref;
    void* addr = t->GetData();
    const auto& desc = t->GetTensorDesc();
    const int layers = kv_pool->num_hidden_layers();
    for (int layer = 0; layer < layers; ++layer) {
        for (int kv = 0; kv < 2; ++kv) {
            auto pool_t = kv == 0 ? kv_pool->KTensor(layer) : kv_pool->VTensor(layer);
            if (!pool_t || pool_t->GetData() != addr) continue;
            const auto& pdesc = pool_t->GetTensorDesc();
            if (pdesc.dims == desc.dims && pdesc.dataType == desc.dataType) {
                ref.found = true;
                ref.is_v  = (kv == 1);
                ref.layer = layer;
                return ref;
            }
        }
    }
    return ref;
}

/// 写出单个工位。失败只记日志，返回 void（调试件不阻断推理）。
void DumpOne(const std::shared_ptr<ITensorBuffer>& t, const std::string& dir,
             const char* side, const char* mode, std::size_t step, std::size_t idx,
             const KvSlotRef& kv) {
    if (!t) return;
    if (!t->GetData()) {
        CANN_LLM_LOGW("debug.dump: side=%s idx=%zu data null, skipped", side, idx);
        return;
    }
    const auto& desc = t->GetTensorDesc();
    const std::size_t elem = DataTypeElemSize(desc.dataType);
    if (elem == 0) {
        CANN_LLM_LOGW("debug.dump: side=%s idx=%zu name=%s unknown dtype, skipped", side, idx,
                      TensorName(desc).c_str());
        return;
    }
    const std::size_t bytes = t->GetSize();
    if (bytes == 0) {
        CANN_LLM_LOGW("debug.dump: side=%s idx=%zu name=%s size=0, skipped", side, idx,
                      TensorName(desc).c_str());
        return;
    }

    // 文件名：KV 工位带 kv_k{layer}/kv_v{layer} 前缀，便于与普通工位区分。
    std::string tail;
    if (kv.found) {
        tail = std::string(kv.is_v ? "kv_v" : "kv_k") + std::to_string(kv.layer);
    } else {
        tail = std::to_string(idx) + "_" + TensorName(desc);
    }
    const std::string path = dir + side + "_" + mode + "_s" + std::to_string(step) + "_" + tail +
                             ".bin";

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        CANN_LLM_LOGE("debug.dump: open '%s' failed", path.c_str());
        return;
    }
    const std::size_t written = std::fwrite(t->GetData(), 1, bytes, fp);
    std::fclose(fp);
    if (written != bytes) {
        CANN_LLM_LOGE("debug.dump: short write '%s' (%zu/%zu)", path.c_str(), written, bytes);
        return;
    }

    CANN_LLM_LOGD(
        "debug.dump: side=%s idx=%zu name=%s shape=%s dtype=%s elems=%zu bytes=%zu -> %s",
        side, idx, TensorName(desc).c_str(), DimsStr(desc.dims).c_str(),
        DataTypeName(desc.dataType), bytes / elem, bytes, path.c_str());
}

}  // namespace

bool DebugDumpEnabled() {
    const char* v = EnvVal(kDumpEnv);
    // 取值 "0" 视为关闭，其余非空视为开启（与 CANN_LM_DEBUG_* 其余开关同风格）。
    return v != nullptr && !(v[0] == '0' && v[1] == '\0');
}

void DebugDumpTensors(const std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                      const char* side, const char* mode, std::size_t step,
                      const std::shared_ptr<KvCachePool>& kv_pool) {
    if (!DebugDumpEnabled()) return;
    if (side == nullptr || mode == nullptr) return;

    const char* kv_env = EnvVal(kDumpKvEnv);
    const bool dump_kv = kv_env != nullptr && !(kv_env[0] == '0' && kv_env[1] == '\0');

    const std::string dir = DumpDir();
    EnsureDir(dir);

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (!t) continue;
        const KvSlotRef kv = FindKvSlot(t, kv_pool);
        if (kv.found && !dump_kv) continue;   // KV 工位未开开关 → 跳过（每步数 GB）
        DumpOne(t, dir, side, mode, step, i, kv);
    }
}

}  // namespace cann_lm::engine