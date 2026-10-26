#include "api/cpp/engine_executor.h"

#include "pipeline/pipeline_executor.h"
#include "infra/log.h"

#include <utility>

namespace cann_lm::engine {

class EngineExecutor::Impl {
public:
    Impl(ModelArtifact artifact, EngineOptions options) : artifact_(std::move(artifact)), options_(options)
    {
        init_pipeline();
    }

    /// 构造期初始化结果（init_pipeline 落笔；pipeline 级加载失败码原样透传）
    Status init_status_;

    Status CreateContext(Context* out)
    {
        if (!pipeline_)
            return Status::FAILURE;
        return pipeline_->CreateContext(out);
    }
    Status DestroyContext(ContextId cid)
    {
        if (!pipeline_)
            return Status::SUCCESS;
        return pipeline_->DestroyContext(cid);
    }
    Status ConfigureContext(ContextId cid, const ContextOptions& opts)
    {
        if (!pipeline_)
            return Status::FAILURE;
        return pipeline_->ConfigureContext(cid, opts);
    }
    Status CancelContext(ContextId cid)
    {
        if (!pipeline_)
            return Status::FAILURE;
        return pipeline_->CancelContext(cid);
    }

    Status Generate(const Prompt& p, const Context& c, const SamplingParams& s, const RequestOptions& ro, Result* out)
    {
        if (!pipeline_)
            return Status::FAILURE;
        return pipeline_->Generate(p, c, s, ro, out);
    }
    Status GenerateAsync(const Prompt& p, const Context& c, const SamplingParams& s, const RequestOptions& ro,
                         RequestHandle* out, StreamCallback cb)
    {
        if (!pipeline_)
            return Status::FAILURE;
        return pipeline_->GenerateAsync(p, c, s, ro, out, cb);
    }

private:
    ModelArtifact artifact_{};
    EngineOptions options_{};
    std::unique_ptr<PipelineExecutorBase> pipeline_;

    void init_pipeline()
    {
        // D-08：组件创建与 meta_json 解析均下沉 pipeline 构造；L1 仅转发。
        Status st = PipelineExecutorFactory::create(PipelineType::Autoregressive, artifact_, options_, &pipeline_);
        if (!st.ok()) {
            CANN_LLM_LOGE("engine.init: pipeline create failed (st=%d)", st.code);
            init_status_ = st; // 加载失败即终止（2026-09-22）：记失败码供 InitStatus 上抛
        }
    }
};

EngineExecutor::EngineExecutor(const ModelArtifact& artifact, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(artifact, options))
{}

Status EngineExecutor::InitStatus() const { return impl_->init_status_; }

EngineExecutor::~EngineExecutor() = default;
EngineExecutor::EngineExecutor(EngineExecutor&&) noexcept = default;
EngineExecutor& EngineExecutor::operator=(EngineExecutor&&) noexcept = default;

Status EngineExecutor::CreateContext(Context* out) { return impl_->CreateContext(out); }
Status EngineExecutor::DestroyContext(ContextId cid) { return impl_->DestroyContext(cid); }
Status EngineExecutor::ConfigureContext(ContextId cid, const ContextOptions& o)
{
    return impl_->ConfigureContext(cid, o);
}
Status EngineExecutor::CancelContext(ContextId cid) { return impl_->CancelContext(cid); }

Status EngineExecutor::Generate(const Prompt& p, const Context& c, const SamplingParams& s, const RequestOptions& ro,
                                Result* out)
{
    return impl_->Generate(p, c, s, ro, out);
}
Status EngineExecutor::GenerateAsync(const Prompt& p, const Context& c, const SamplingParams& s,
                                     const RequestOptions& ro, RequestHandle* out, StreamCallback cb)
{
    return impl_->GenerateAsync(p, c, s, ro, out, cb);
}

} // namespace cann_lm::engine
