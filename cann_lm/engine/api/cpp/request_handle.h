// ═══════════════════════════════════════════════════════════════════════════
// 对外 API 头文件（Public API）。用户可直接调用本文件声明的类型与方法。
// 任何被标记 `对外 API` 的方法构成稳定的对外接口表面；其余为生命周期或内部细节。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "infra/status.h"
#include "api/cpp/types.h"

#include <memory>

namespace cann_lm::engine {

class RequestImpl;  // internal, forward declare（不对用户暴露）
class AutoregressivePipelineExecutor;  // internal, forward declare（friend 访问）

/// 请求状态枚举（对外 API）。GetStatus 返回值。
enum class RequestStatus {
    Queued,        ///< 已入队，等待 Prefill
    Prefilling,    ///< Prefill 进行中
    Decoding,      ///< 已加入 decode 池，Decode 进行中
    Completed,     ///< 完成（Result 可取）
    Canceled,      ///< 已取消
    Failed,        ///< 失败（配合返回 Status::FAILURE，Result 可能含部分 token）
};

/// 请求句柄。轻量、move-only（持有请求槽所有权）。对外 API。
/// 流式回调在 GenerateAsync 传入，Handle 不持有回调。
class RequestHandle {
public:
    // ────────────────── 生命周期（对外 API） ──────────────────
    RequestHandle() noexcept;                                  ///< 空句柄
    RequestHandle(const RequestHandle&)            = delete;    ///< 不可拷贝（持请求槽所有权）
    RequestHandle& operator=(const RequestHandle&) = delete;
    RequestHandle(RequestHandle&&) noexcept;                    ///< 可移动
    RequestHandle& operator=(RequestHandle&&) noexcept;
    ~RequestHandle();

    // ────────────────── 状态查询（对外 API，可并发调用） ──────────────────

    /// 查询请求状态（需求 #3）。空句柄返回 RequestStatus::Queued。
    RequestStatus GetStatus() const noexcept;

    /// 是否终态（Completed / Canceled / Failed）。空句柄返回 false。
    bool IsDone() const noexcept;

    /// 是否有效句柄（impl_ 非空）。空句柄为 false。
    explicit operator bool() const noexcept;

    // ────────────────── 控制（对外 API） ──────────────────

    /// 单请求 cancel（需求 #3）。标记 canceled，当前 iteration 后退出池，部分 token 丢弃。
    /// 已终态再调返回 SUCCESS（幂等）。空句柄返回 SUCCESS。
    Status Cancel();

    /// 阻塞等待完整 Result。终态语义（见 04 §6）：
    /// - Completed → Status::SUCCESS，*out 填完整 token 序列 + text；
    /// - Canceled  → Status::CANCELED，*out 不含部分（已生成 token 丢弃，不返回给用户）；
    /// - Failed    → Status::FAILURE，*out 含已生成部分（便于排查/续接）。
    Status Wait(Result* out);

    /// 非阻塞取结果。终态时填 *out：
    /// - Completed → Status::SUCCESS（完整）；
    /// - Canceled  → Status::CANCELED（*out 空，丢弃）；
    /// - Failed    → Status::FAILURE（*out 含部分）；
    /// - 未终态   → Status::INVALID_STATE。
    Status TryGetResult(Result* out) const;

private:
    friend class EngineExecutor;
    friend class AutoregressivePipelineExecutor;
    std::shared_ptr<RequestImpl> impl_;   // RequestImpl internal, forward declared
    explicit RequestHandle(std::shared_ptr<RequestImpl>);
};

}  // namespace cann_lm::engine
