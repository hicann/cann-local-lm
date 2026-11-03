// ═══════════════════════════════════════════════════════════════════════════
// DebugTensorDump 实现（env 开关默认关）。详见 debug_tensor_dump.h 的说明。
// ═══════════════════════════════════════════════════════════════════════════
#include "utils/debug_tensor_dump.h"

#include "infra/log.h"
#include "utils/data_type_utils.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace cann_local_lm::engine {

namespace {

constexpr const char* kDumpEnv = "CANN_LM_DEBUG_DUMP_TENSORS";
constexpr const char* kDumpKvEnv = "CANN_LM_DEBUG_DUMP_KV";
constexpr const char* kDumpDirEnv = "CANN_LM_DEBUG_DUMP_DIR";
constexpr const char* kDefaultDir = "/data/local/tmp/dump/";

/// 非空 env 取值（返回 nullptr / 空串均视为未设置）。
const char* EnvVal(const char* name)
{
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

/// 目录路径：取 env，缺省 kDefaultDir；确保以 '/' 结尾。
std::string DumpDir()
{
    const char* v = EnvVal(kDumpDirEnv);
    std::string dir = (v != nullptr) ? v : kDefaultDir;
    if (dir.empty() || dir.back() != '/')
        dir += '/';
    return dir;
}

/// 确保目录存在（不存在则 mkdir；已存在不报错）。失败仅告警，由后续 fopen 兜底。
void EnsureDir(const std::string& dir)
{
    struct stat st;
    if (stat(dir.c_str(), &st) == 0)
        return;
    if (mkdir(dir.c_str(), 0775) != 0) {
        CANN_LLM_LOGW("debug.dump: mkdir '%s' failed, will try write anyway", dir.c_str());
    }
}

std::string DimsStr(const std::vector<int32_t>& dims)
{
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i)
            s += ",";
        s += std::to_string(dims[i]);
    }
    s += "]";
    return s;
}

/// 工位名（空名回退 unnamed）。
std::string TensorName(const TensorDesc& desc)
{
    return desc.tensorName.empty() ? std::string("unnamed") : desc.tensorName;
}

/// KV 池工位查找结果。
struct KvSlotRef {
    bool found = false;
    bool is_v = false; // false = K
    int layer = -1;
};

/// 按 GetData() 指针在池中查找：命中即为 KV 工位（输出侧 KV 工位名未知，只能按指针识别）。
/// 同时校验 dims/dtype，避免指针巧合误判。
KvSlotRef FindKvSlot(const std::shared_ptr<ITensorBuffer>& t, const std::shared_ptr<KvCachePool>& kv_pool)
{
    KvSlotRef ref;
    if (!kv_pool || !t || !t->GetData())
        return ref;
    void* addr = t->GetData();
    const auto& desc = t->GetTensorDesc();
    const int layers = kv_pool->num_hidden_layers();
    for (int layer = 0; layer < layers; ++layer) {
        for (int kv = 0; kv < 2; ++kv) {
            auto pool_t = kv == 0 ? kv_pool->KTensor(layer) : kv_pool->VTensor(layer);
            if (!pool_t || pool_t->GetData() != addr)
                continue;
            const auto& pdesc = pool_t->GetTensorDesc();
            if (pdesc.dims == desc.dims && pdesc.dataType == desc.dataType) {
                ref.found = true;
                ref.is_v = (kv == 1);
                ref.layer = layer;
                return ref;
            }
        }
    }
    return ref;
}

/// 写出单个工位。失败只记日志，返回 void（调试件不阻断推理）。
/// 文件名 {mode}_{side}_{轮次}_{name}.bin；同轮同名工位（输出侧名称常空）由调用方
/// 追加序号消歧。
void DumpOne(const std::shared_ptr<ITensorBuffer>& t, const std::string& dir, const char* side, const char* mode,
             std::size_t step, const std::string& tail)
{
    if (!t)
        return;
    if (!t->GetData()) {
        CANN_LLM_LOGW("debug.dump: side=%s name=%s data null, skipped", side, tail.c_str());
        return;
    }
    const auto& desc = t->GetTensorDesc();
    const std::size_t elem = DataTypeElemSize(desc.dataType);
    if (elem == 0) {
        CANN_LLM_LOGW("debug.dump: side=%s name=%s unknown dtype, skipped", side, tail.c_str());
        return;
    }
    const std::size_t bytes = t->GetSize();
    if (bytes == 0) {
        CANN_LLM_LOGW("debug.dump: side=%s name=%s size=0, skipped", side, tail.c_str());
        return;
    }

    const std::string path = dir + mode + "_" + side + "_" + std::to_string(step) + "_" + tail + ".bin";

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

    CANN_LLM_LOGD("debug.dump: side=%s name=%s shape=%s dtype=%s elems=%zu bytes=%zu -> %s", side,
                  TensorName(desc).c_str(), DimsStr(desc.dims).c_str(), DataTypeName(desc.dataType), bytes / elem,
                  bytes, path.c_str());
}

} // namespace

bool DebugDumpEnabled()
{
    const char* v = EnvVal(kDumpEnv);
    // 取值 "0" 视为关闭，其余非空视为开启（与 CANN_LM_DEBUG_* 其余开关同风格）。
    return v != nullptr && !(v[0] == '0' && v[1] == '\0');
}

void DebugDumpTensors(const std::vector<std::shared_ptr<ITensorBuffer>>& tensors, const char* side, const char* mode,
                      std::size_t step, const std::shared_ptr<KvCachePool>& kv_pool)
{
    if (!DebugDumpEnabled())
        return;
    if (side == nullptr || mode == nullptr)
        return;

    const char* kv_env = EnvVal(kDumpKvEnv);
    const bool dump_kv = kv_env != nullptr && !(kv_env[0] == '0' && kv_env[1] == '\0');

    const std::string dir = DumpDir();
    EnsureDir(dir);

    // 同侧同名去重：输出侧名称常空（NDK 输出 desc 无名），多个输出会全落
    // "unnamed"——同名冲突时追加 _2/_3… 后缀防覆盖（含 KV 工位 kv_k*/kv_v*）。
    std::map<std::string, int> used;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (!t)
            continue;
        const KvSlotRef kv = FindKvSlot(t, kv_pool);
        if (kv.found && !dump_kv)
            continue; // KV 工位未开开关 → 跳过（每步数 GB）
        std::string name;
        if (kv.found) {
            name = std::string(kv.is_v ? "kv_v" : "kv_k") + std::to_string(kv.layer);
        } else {
            name = TensorName(t->GetTensorDesc());
        }
        std::string tail = name;
        if (++used[name] > 1) {
            tail += "_" + std::to_string(used[name]);
        }
        DumpOne(t, dir, side, mode, step, tail);
    }
}

void DebugDumpTokenizerIds(const char* mode, const std::vector<int32_t>& ids, const char* name)
{
    // tokenizer 结果 dump（golden 对齐）。
    // naive/env 语义与 DebugDumpTensors 完全同体例：
    //   - 总闸 CANN_LM_DEBUG_DUMP_TENSORS 未开 → 立即返回，零文件零日志；
    //   - 目录/建目录逻辑与工位 dump 共用（DumpDir/EnsureDir）；
    //   - 任何失败只打日志，不改推理通路。
    // 文件名 {mode}_tokenizer_{name}.bin：mode 区分 prefill/decode 调用点，
    // name 区分同一轮的多次 encode（如 prefill 首次 vs PromptCache 增量轮）。
    // 内容 = int32 裸数组（行优先、无文件头，np.fromfile(dtype=int32) 直读），
    // 与 golden 侧 Python tokenizer.encode 结果（tolist 后 np.array(…, int32)
    // tofile）逐值可比——对齐 embeddings 前 N 行的查表源 id。
    // 注意写到 int32：Preprocessor 的 ids 是 int（tokenizer 面），TokenId
    // （engine 面）值域同为 [0, vocab)，窄转无损；decode 路径 token id 由
    // Scheduler 从历史取，不经 Preprocessor，本 dump 只覆盖 prefill encode。
    if (!DebugDumpEnabled())
        return;
    if (mode == nullptr || name == nullptr)
        return;
    if (ids.empty())
        return; // 空 prompt encode 结果无可比性，不产出空文件

    const std::string dir = DumpDir();
    EnsureDir(dir);

    const std::string path = dir + mode + "_tokenizer_" + name + ".bin";
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        CANN_LLM_LOGE("debug.dump: open '%s' failed", path.c_str());
        return;
    }
    const std::size_t bytes = ids.size() * sizeof(int32_t);
    const std::size_t written = std::fwrite(ids.data(), 1, bytes, fp);
    std::fclose(fp);
    if (written != bytes) {
        CANN_LLM_LOGE("debug.dump: short write '%s' (%zu/%zu)", path.c_str(), written, bytes);
        return;
    }
    CANN_LLM_LOGD("debug.dump: tokenizer mode=%s name=%s count=%zu bytes=%zu -> %s", mode, name, ids.size(), bytes,
                  path.c_str());
}

} // namespace cann_local_lm::engine
