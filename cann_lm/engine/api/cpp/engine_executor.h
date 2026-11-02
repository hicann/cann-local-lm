// ═══════════════════════════════════════════════════════════════════════════
// 对外 API 头文件（Public API）。用户可直接调用本文件声明的类型与方法。
// 任何被标记 `对外 API` 的方法构成稳定的对外接口表面；其余为生命周期或内部细节。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "core/context.h"
#include "api/cpp/request_handle.h"
#include "infra/status.h"
#include "api/cpp/types.h"

#include <memory>
#include <string>

namespace cann_lm::engine {

/// 聚合构造期可调参数（草案字段集，其余字段待 DESIGN-AUDIT 补全）。对外 API。
struct EngineOptions {
    int default_chunk_size = 1024;   ///< 默认切块大小（token 数），可被 ConfigureContext 覆盖
    int max_concurrent_prefill = 1;  ///< 并发 prefill 上限（v0.3 Continuous Batching 下默认 1）
    int max_batch_size_override = 0; ///< 0 = 用模型元信息 max_batch_size；>0 = 覆盖
    /// tokenizer 算法："BPE"|"Qwen"|"SPM"；空 = 默认 BPE，未知值 WARN 后回退 BPE。
    /// 算法选择权在此（C ABI 同名字段传入），取代 pipeline 内原硬编码 BPE。
    std::string tokenizer_type;
    /// tokenizer.json 词表产物路径；空 = 不经 options 供给（回退 meta_json
    /// tokenizer_config.tokenizer_file）。非空时优先；不可加载 → tokenizer 置空。
    std::string tokenizer_path;
};

/// ConfigureContext 参数集占位（草案，字段待 DESIGN-AUDIT 列）。对外 API。
struct ContextOptions {};

class PipelineExecutorBase; // PIMPL impl 基类（自回归等派生），forward declare（内部类型，不对用户暴露）

/// 引擎执行器。库的主入口对象（对外 API）。PIMPL 封装，独占所有权（不可复制、可移动）。
class EngineExecutor {
public:
    // ────────────────── 生命周期（对外 API） ──────────────────

    /// 构造（SR20260903001 单一构件：prefill/decode 双图位 + prefill/decode 双权重
    /// 路径（docs/separate_pd_weights.md §4.1）+ meta_json，
    /// 各双态字段路径/buffer 二选一，buffer 优先）。构造内完成 配置解析（内联步骤，
    /// 失败降级桩默认+WARN，不阻断）→ 组件装配 → 模型加载 → worker 启动。
    /// 模型加载失败（图位已配置）时构造内终止初始化并完成卸载，不抛异常——
    /// 经 InitStatus() 上报；此时本对象仅剩 NULL 防御语义，
    /// 全部 API 返回 Status::FAILURE。
    EngineExecutor(const ModelArtifact& artifact, const EngineOptions& options = {});

    /// 构造期初始化结果。SUCCESS = 模型加载与装配完成；
    /// 非 SUCCESS = 加载失败（已终止初始化并卸载，实例不可用于推理）。
    Status InitStatus() const;

    ~EngineExecutor();
    EngineExecutor(const EngineExecutor&) = delete; ///< 不可复制
    EngineExecutor& operator=(const EngineExecutor&) = delete;
    EngineExecutor(EngineExecutor&&) noexcept; ///< 可移动
    EngineExecutor& operator=(EngineExecutor&&) noexcept;

    // ────────────────── Context 管理（对外 API） ──────────────────

    /// 创建 Context。成功时 *out 接收有效句柄（返回 Status::SUCCESS）；
    /// 失败时 *out 保持原状（不写入），返回 Status::FAILURE / INVALID_STATE。
    Status CreateContext(Context* out);

    /// 销毁 Context。释放对应 KV 句柄。
    /// 已销毁的 ContextId 再调返回 SUCCESS（幂等）。
    /// 该 Context 有在飞请求 → Status::INVALID_STATE（先 CancelContext）；
    /// 销毁活动 Context（无在飞）→ 允许，绑定清空回无活动态，不隐式
    /// fallback 到其他存活 Context（ADR-0002 S6/S7）。
    Status DestroyContext(ContextId);

    /// 显式激活 Context 为活动上下文（ADR-0002 Session/SetContext）。
    /// 未注册 id → NOT_FOUND；有在飞请求（Executor 全局）→ INVALID_STATE；
    /// 目标即当前活动 → 幂等 SUCCESS。切换在锁内原子完成：模板池转发
    /// 登记 + 双执行器 KV 工位刷新 + assembler 记账面同步——此后所有
    /// Generate 归属该 Context 直至另行切换/销毁。
    Status SetContext(ContextId);

    /// 配置 Context（运行期调参）。字段集待 DESIGN-AUDIT 列，当前 ContextOptions 为空占位。
    Status ConfigureContext(ContextId, const ContextOptions&);

    /// Context 级 cancel（需求 #3）。取消该 Context 上所有在飞请求。
    /// 可与该 Context 上的请求并发调用（线程安全，见 04 §7）。
    Status CancelContext(ContextId);

    // ────────────────── 推理（对外 API） ──────────────────

    /// 同步生成。阻塞当前线程直到完成，*out 接收完整 Result。
    /// 请求归属活动 Context：无活动 Context（未 SetContext / 已销毁活动）
    /// → Status::INVALID_STATE（S2）；活动 Context 已有在飞请求 →
    /// INVALID_STATE。终止语义归 req_opts；采样旋钮归 sp。
    Status Generate(const Prompt&, const SamplingParams&, const RequestOptions&, Result* out);

    /// 异步生成。立即返回，*out 接收 RequestHandle。
    /// 请求归属活动 Context（同上 S2 口径）。
    /// @param callback 可选流式回调（默认空 = 非流式）；在 pipeline 内部线程触发，
    ///                 回调内不应阻塞或调用 Engine 同步 API。
    /// decode 池满时入队等待（不返回失败），见 04 §6。
    Status GenerateAsync(const Prompt&, const SamplingParams&, const RequestOptions&, RequestHandle* out,
                         StreamCallback callback = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cann_lm::engine
