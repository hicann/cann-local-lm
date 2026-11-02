// ═══════════════════════════════════════════════════════════════════════════
// 耗时观测工具
//
// 计时基准：steady_clock（单调、不受系统时间跳变影响），微秒承载。
//
// ── 打点协议 ────────────────────────────────────────────────────────────────
//   INFO  = 整体阶段：模型图/权重/tokenizer 加载、prefill/decode 整体完成；
//   DEBUG = 每轮：每次 chunk prefill forward / 每次 decode batch forward。
//   日志键统一 perf.* 前缀，elapsed 单位 ms（%g/%.3f，三位小数）。
//   典型形态：NowUs() 取起点 → 调用点持起点成员/局部变量 →
//   终点 UsToMs(NowUs() - start) 随 CANN_LLM_LOGI/CANN_LLM_LOGD 输出。
//
//   起点哨兵约定：起点成员以 0 = 未起步（steady_clock epoch 不产 0），
//   失败/取消路径不补写（无观测意义）。
//
// 无状态纯函数，header-only（同 common_utils.h 体例；不新增 .cpp 即无需
// 动两份引擎源码清单）。
// ═══════════════════════════════════════════════════════════════════════════
#ifndef CANN_LM_ENGINE_UTILS_PERF_TIMER_H
#define CANN_LM_ENGINE_UTILS_PERF_TIMER_H

#include <chrono>
#include <cstdint>

namespace cann_lm::engine {

/// steady_clock 当前时刻（微秒，单调不受系统时间跳变影响）。
inline std::int64_t NowUs()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// 微秒差 → 毫秒（double，随 %.3f 打三位小数）。
inline double UsToMs(std::int64_t us) { return static_cast<double>(us) / 1000.0; }

} // namespace cann_lm::engine

#endif // CANN_LM_ENGINE_UTILS_PERF_TIMER_H
