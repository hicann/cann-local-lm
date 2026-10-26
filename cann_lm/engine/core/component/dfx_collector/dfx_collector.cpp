// DfxCollector 实现。DESIGN M-04。
#include "core/component/dfx_collector/dfx_collector.h"

namespace cann_lm::engine {

DfxCollector::DfxCollector(DfxConfig cfg, std::shared_ptr<KvCachePool> pool)
    : cfg_(std::move(cfg)), pool_(std::move(pool))
{}

Status DfxCollector::OnForwardStart(const std::vector<std::shared_ptr<ITensorBuffer>>& /*inputs*/,
                                    const std::vector<SlotId>& /*slots*/, bool dump)
{
    if (!cfg_.io_dump || !dump)
        return Status::SUCCESS;
    return Status::SUCCESS;
}

Status DfxCollector::OnForwardEnd(const std::shared_ptr<ITensorBuffer>& /*logits*/, double /*elapsed_us*/)
{
    if (cfg_.timing)
        forward_count_.fetch_add(1, std::memory_order_relaxed);
    return Status::SUCCESS;
}

Status DfxCollector::PoolSnapshot(std::vector<SlotInfo>* out) const
{
    if (!pool_ || !out)
        return Status::FAILURE;
    return pool_->Snapshot(out);
}

Status DfxCollector::FlushTimings(void* out) const
{
    if (out)
        *static_cast<int*>(out) = forward_count_.load(std::memory_order_relaxed);
    return Status::SUCCESS;
}

} // namespace cann_lm::engine
