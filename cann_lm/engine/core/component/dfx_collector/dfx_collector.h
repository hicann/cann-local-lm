// DfxCollector（DESIGN M-04）。
#pragma once

#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/tensor_manager.h"   // ITensorBuffer
#include "infra/status.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <vector>

namespace cann_lm::engine {

struct DfxConfig {
    bool io_dump = false;
    bool timing  = true;
    std::filesystem::path dump_dir;
};

class DfxCollector {
public:
    DfxCollector(DfxConfig cfg, std::shared_ptr<KvCachePool> pool);

    Status OnForwardStart(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                          const std::vector<SlotId>& slots, bool dump);
    Status OnForwardEnd(const std::shared_ptr<ITensorBuffer>& logits, double elapsed_us);
    Status PoolSnapshot(std::vector<SlotInfo>* out) const;
    Status FlushTimings(void* out) const;

    const DfxConfig& config() const noexcept { return cfg_; }

private:
    DfxConfig                       cfg_;
    std::shared_ptr<KvCachePool>    pool_;
    mutable std::atomic<int>        forward_count_{0};
};

}  // namespace cann_lm::engine
