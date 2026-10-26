// ═══════════════════════════════════════════════════════════════════════════
// 对外 API 头文件（Public API）。用户可直接调用本文件声明的类型与方法。
// 任何被标记 `对外 API` 的方法构成稳定的对外接口表面；其余为生命周期或内部细节。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <cstdint>
#include <memory>

namespace cann_lm::engine {

class ContextImpl;                    // internal, forward declare（不对用户暴露）
class AutoregressivePipelineExecutor; // internal, forward declare（friend 访问）

/// Context 标识。用于多会话切换、状态查询、Context 级 cancel（需求 #5 / #3）。对外 API。
struct ContextId {
    uint64_t value = 0;
    bool operator==(const ContextId&) const noexcept; ///< 相等比较
    bool operator!=(const ContextId&) const noexcept; ///< 不等比较
    bool operator<(const ContextId&) const noexcept;  ///< 用于有序容器
    explicit operator bool() const noexcept;          ///< value != 0（有效标识）
};

/// 上下文句柄。轻量、可复制（内部状态共享）。对外 API。
/// 实体由 pipeline 拥有；用户持句柄读 id / 校验有效。
class Context {
public:
    // ────────────────── 生命周期（对外 API） ──────────────────
    Context() noexcept;               ///< 空句柄
    Context(const Context&) noexcept; ///< 浅拷贝，共享内部状态
    Context& operator=(const Context&) noexcept;
    Context(Context&&) noexcept; ///< 可移动
    Context& operator=(Context&&) noexcept;
    ~Context();

    // ────────────────── 访问器（对外 API） ──────────────────

    /// 取标识。空句柄返回默认 ContextId{value=0}。
    ContextId id() const noexcept;

    /// 是否有效句柄（impl_ 非空）。
    explicit operator bool() const noexcept;

private:
    friend class EngineExecutor;
    friend class AutoregressivePipelineExecutor;
    std::shared_ptr<ContextImpl> impl_; // 可为空
    explicit Context(std::shared_ptr<ContextImpl>);

    /// 内部工厂：按 id 创建 ContextImpl 并包装为 Context。
    /// 仅供 pipeline 调用（friend 访问）。
    static Context CreateWithId(ContextId id);

    // TODO
    // KvCacheHandle
};

} // namespace cann_lm::engine
