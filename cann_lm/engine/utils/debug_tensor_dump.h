// ═══════════════════════════════════════════════════════════════════════════
// 【临时调试代码 — 验证完成后删除】
//
// 三个开关，默认关闭（env 未设置 = 零行为变化）：
//
//   CANN_LM_DEBUG_DUMP_TENSORS=1     推理前后 dump 全部「非 KV」工位
//   CANN_LM_DEBUG_DUMP_KV=1          额外 dump KV 工位（须与主开关同时开）
//   CANN_LM_DEBUG_DUMP_DIR=<path>    dump 根目录（默认 /data/local/tmp/dump/）
//
// 用途：把送进模型（mm->Run 之前）和模型吐出（之后）的每个 tensor 落盘，与参考实现
// （framework model_executor_for_generation.cpp 的 DumpInputsToBins / DumpKvCacheToBins）
// 的 dump 逐 tensor 比对，定位「输入不对」还是「模型输出不对」。
//
// ── 文件格式（裸 dump，无文件头，行优先，可直接 numpy fromfile）──────────────
//   文件名：{side}_{mode}_s{step}_{idx}_{name}.bin
//     side = in | out      （输入工位 / 输出工位）
//     mode = prefill | decode（由 ForwardOptions.chunk_len 判别）
//     step = 本步步进序号（仅用于避免同名覆盖，不参与正确性）
//     idx  = 工位下标（重名工位靠它区分，不覆盖）
//     name = TensorDesc.tensorName（空则 unnamed）
//   内容：GetData() 起 GetSize() 字节原样写出。
//
// ── KV 工位单列开关（重要）───────────────────────────────────────────────────
// executor 把 KV 池 tensor 同时绑进输入工位和输出工位（om 原地更新 KV，
// model_executor.cpp 按 dims+dtype 识别输出侧 KV 工位）。若默认全 dump，
// decode 每步要写出 2×num_hidden_layers 个整层 KV tensor（可达数 GB/步）。
// 故 KV 工位需显式 CANN_LM_DEBUG_DUMP_KV=1 才 dump。
// KV 工位识别不依赖 tensorName（输出侧名称未知），而是按 GetData() 指针与
// KvCachePool 各层 K/V tensor 比对；命中后文件名带 kv_k{layer} / kv_v{layer}。
//
// 任何数据不可用（空指针 / 未知 dtype / 写失败）只打日志跳过，不失败、不改变推理通路。
// ═══════════════════════════════════════════════════════════════════════════
#ifndef CANN_LM_ENGINE_UTILS_DEBUG_TENSOR_DUMP_H
#define CANN_LM_ENGINE_UTILS_DEBUG_TENSOR_DUMP_H

#include "core/tensor_manager.h"                    // ITensorBuffer
#include "core/component/kv_cache_pool/kv_cache_pool.h"  // KvCachePool（KV 工位识别）

#include <cstddef>
#include <memory>
#include <vector>

namespace cann_lm::engine {

/// CANN_LM_DEBUG_DUMP_TENSORS 是否开启（非空且非 "0" → true）。
bool DebugDumpEnabled();

/// dump 一组工位。未开启时立即返回（no-op）。
///   tensors : 工位列表（输入侧或输出侧）
///   side    : "in" / "out"（文件名用）
///   mode    : "prefill" / "decode"（文件名用）
///   step    : 本步步进序号（文件名用，不参与正确性）
///   kv_pool : KV 池（识别 KV 工位用；可为 null = 无法识别，全部按普通工位处理）
/// 不抛异常、不返回错误：单个工位失败只打日志跳过（调试件不阻断推理）。
void DebugDumpTensors(const std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                      const char* side, const char* mode, std::size_t step,
                      const std::shared_ptr<KvCachePool>& kv_pool);

}  // namespace cann_lm::engine

#endif  // CANN_LM_ENGINE_UTILS_DEBUG_TENSOR_DUMP_H