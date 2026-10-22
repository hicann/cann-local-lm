// ═══════════════════════════════════════════════════════════════════════════
// 对外 API 头文件（Public API）。用户可直接调用本文件声明的类型与方法。
// 任何被标记 `对外 API` 的方法构成稳定的对外接口表面；其余为生命周期或内部细节。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

namespace cann_lm::engine {

/// 统一状态码（C2，非异常）。贯穿全部 public API。对外 API。
/// 一组 `static constexpr` 常量；详细失败原因经日志（C11）输出，Status 仅给粗粒度码。
struct Status {
    int code;
    static constexpr int SUCCESS           = 0;   ///< 成功
    static constexpr int FAILURE           = 1;   ///< 通用失败（加载 / 运行时内部错误）
    static constexpr int INVALID_ARGUMENT  = 2;   ///< 参数非法
    static constexpr int INVALID_STATE     = 3;   ///< 状态冲突（如同一 Context 并发调用）
    static constexpr int NOT_FOUND         = 4;   ///< Context / 文件 / 请求不存在
    static constexpr int CANCELED          = 5;   ///< 已取消

    constexpr Status() : code(SUCCESS) {}
    constexpr Status(int c) : code(c) {}

    // ────────────────── 访问器（对外 API） ──────────────────

    constexpr bool ok() const noexcept          { return code == SUCCESS; }       ///< 是否成功（code == SUCCESS）
    constexpr bool IsCanceled() const noexcept  { return code == CANCELED; }       ///< 是否被取消
    constexpr explicit operator bool() const noexcept { return ok(); }             ///< if (status) 等价于 ok()
};

}  // namespace cann_lm::engine
