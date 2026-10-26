// TEST-WRITE(GREEN)：维测收集器单元测试（v0.13：hiai NDTensor 适配）。
#include "core/component/dfx_collector/dfx_collector.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/tensor_manager.h"

#include <gtest/gtest.h>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

namespace cann_lm::engine {
namespace {

std::shared_ptr<ITensorBuffer> MakeTensor()
{
    TensorDesc d;
    d.dims = {1};
    d.dataType = DataType::FLOAT32;
    return hiai::CreateNDTensorBuffer(d);
}

std::shared_ptr<KvCachePool> MakePool() { return std::make_shared<KvCachePool>(4, 16, 2, 4, 8); }

int CountFiles(const std::filesystem::path& dir)
{
    int n = 0;
    if (std::filesystem::exists(dir))
        for (auto& _ : std::filesystem::directory_iterator(dir)) {
            (void)_;
            ++n;
        }
    return n;
}

TEST(DfxCollectorTest, DumpOffZeroOverhead_TC205)
{
    auto dump_dir = std::filesystem::temp_directory_path() / "dfx_test_205";
    std::filesystem::remove_all(dump_dir);
    std::filesystem::create_directories(dump_dir);

    DfxConfig cfg{/*.io_dump=*/false, /*.timing=*/false, /*.dump_dir=*/dump_dir};
    auto pool = MakePool();
    DfxCollector dfx(cfg, pool);

    std::vector<std::shared_ptr<ITensorBuffer>> inputs{MakeTensor(), MakeTensor()};
    std::vector<SlotId> slots{0, 1};
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(dfx.OnForwardStart(inputs, slots, /*dump=*/false).ok());
        EXPECT_TRUE(dfx.OnForwardEnd(MakeTensor(), 10.0).ok());
    }
    EXPECT_EQ(CountFiles(dump_dir), 0);
    int counter = -1;
    EXPECT_TRUE(dfx.FlushTimings(&counter).ok());
    EXPECT_EQ(counter, 0);
    std::filesystem::remove_all(dump_dir);
}

TEST(DfxCollectorTest, PoolSnapshotContentMatchesAllocState)
{
    DfxConfig cfg;
    auto pool = MakePool();
    DfxCollector dfx(cfg, pool);
    SlotId a = -1, b = -1;
    ASSERT_TRUE(pool->AllocKvSlot(&a).ok());
    ASSERT_TRUE(pool->AllocKvSlot(&b).ok());
    ASSERT_TRUE(pool->Advance(a, 5).ok());
    std::vector<SlotInfo> snap;
    ASSERT_TRUE(dfx.PoolSnapshot(&snap).ok());
    ASSERT_EQ(snap.size(), 4u);
    EXPECT_TRUE(snap[a].active);
    EXPECT_EQ(snap[a].total_written, 5u);
    EXPECT_TRUE(snap[b].active);
    EXPECT_EQ(snap[b].total_written, 0u);
}

TEST(DfxCollectorTest, DumpFailureDoesNotBlockInference)
{
    DfxConfig cfg{/*.io_dump=*/true, /*.timing=*/true,
                  /*.dump_dir=*/"/nonexistent-dir-xyz/"};
    auto pool = MakePool();
    DfxCollector dfx(cfg, pool);
    std::vector<std::shared_ptr<ITensorBuffer>> inputs{MakeTensor()};
    std::vector<SlotId> slots{0};
    EXPECT_TRUE(dfx.OnForwardStart(inputs, slots, /*dump=*/true).ok());
    EXPECT_TRUE(dfx.OnForwardEnd(MakeTensor(), 5.0).ok());
}

} // namespace
} // namespace cann_lm::engine
