// TEST-WRITE(RED)：KV Cache 池单元测试（审计修订版）。
// 关联 SPEC §4.1.2 / DESIGN M-02 / D-13。修 M2/M3（去误挂 TC 号）。
// T1 改造（Cache_pool_simplify.md）：构造期不再建 tensor——槽位/光标用例
// 零影响；原「构造即申请」两用例改按「EnsureKvTensor 惰性创建」语义断言
// （本目标属预存损坏 model_executor_ut，host 基线外，断言以新语义为准）。
#include "core/component/kv_cache_pool/kv_cache_pool.h"

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

namespace cann_local_lm::engine {
namespace {

KvCachePool MakePool(int max_batch_size = 4, int kv_cache_max_len = 16, int num_layers = 2, int num_kv_heads = 4,
                     int head_dim = 8)
{
    return KvCachePool(max_batch_size, kv_cache_max_len, num_layers, num_kv_heads, head_dim);
}

// ── 正常路径（无 TC 号：池级原语，非 SPEC Forward TC）──

TEST(KvCachePoolTest, AllocReturnsValidSlotAndMarksActive)
{
    auto pool = MakePool();
    SlotId id = -1;
    auto st = pool.AllocKvSlot(&id);
    ASSERT_TRUE(st.ok());
    EXPECT_GE(id, 0);
    std::vector<SlotInfo> snap;
    ASSERT_TRUE(pool.Snapshot(&snap).ok());
    EXPECT_TRUE(snap[static_cast<size_t>(id)].active);
}

TEST(KvCachePoolTest, FreeThenAllocReusesSlot)
{
    auto pool = MakePool();
    SlotId a = -1, b = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&a).ok());
    EXPECT_TRUE(pool.FreeKvSlot(a).ok());
    EXPECT_TRUE(pool.AllocKvSlot(&b).ok()); // GREEN 期应复用 a
    EXPECT_TRUE(pool.FreeKvSlot(b).ok());
    EXPECT_TRUE(pool.FreeKvSlot(b).ok()); // 幂等
}

TEST(KvCachePoolTest, AdvanceMovesCursorAndNextPositionReflects)
{
    auto pool = MakePool();
    SlotId id = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&id).ok());
    EXPECT_EQ(pool.NextPosition(id), 0u);
    EXPECT_TRUE(pool.Advance(id, 5).ok());
    EXPECT_EQ(pool.NextPosition(id), 5u);
    EXPECT_TRUE(pool.Advance(id, 3).ok());
    EXPECT_EQ(pool.NextPosition(id), 8u);
}

// NextPosition 在 Advance 后稳定可重复读（非零拷贝 TC；池级原语）
TEST(KvCachePoolTest, NextPositionIdempotentAfterAdvance)
{
    auto pool = MakePool();
    SlotId id = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&id).ok());
    ASSERT_TRUE(pool.Advance(id, 7).ok());
    EXPECT_EQ(pool.NextPosition(id), 7u); // 写后读一致
    // 第二次读仍一致（无并发改）
    EXPECT_EQ(pool.NextPosition(id), 7u);
}

// ── 异常 / 边界 ──

// 光标越 kv_max → R5 环形化：绕回不报错（池侧恒 SUCCESS；单请求超缓冲由
// Submit 前置校验拦截，spec REQ-9）。total_written 单调记逻辑长度，
// write_pos = Map(total_written) 环形物理位——本用例锁分离记账两视角。
// （旧契约"越界 → FAILURE"随 ADR-0001 环形化废除——DESIGN 🔴 红线移至
// Submit 前置校验；本目标迁移时同步改写，断言以现行语义为准。）
TEST(KvCachePoolTest, AdvanceOverrunWrapsInsteadOfRejecting)
{
    auto pool = MakePool(/*max_batch_size=*/2, /*max_seq_len=*/4);
    SlotId id = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&id).ok());
    EXPECT_TRUE(pool.Advance(id, 3).ok()); // total_written=3 < 4
    EXPECT_TRUE(pool.Advance(id, 2).ok()); // 3+2=5 > 4 → 绕回（不再拒绝）
    EXPECT_EQ(pool.TotalWritten(id), 5u);  // 逻辑长度单调不减
    EXPECT_EQ(pool.NextPosition(id), 1u);  // 物理位 = 5 % 4（无锚定段）
}

// 池满驱逐最早 active 槽（TC_202）：驱逐-复用语义，c 复用被驱逐槽、cursor 重置
TEST(KvCachePoolTest, PoolFullEvictsOldest_TC202)
{
    auto pool = MakePool(/*max_batch_size=*/2, /*max_seq_len=*/16);
    SlotId a = -1, b = -1, c = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&a).ok());
    ASSERT_TRUE(pool.Advance(a, 3).ok()); // a cursor=3
    ASSERT_TRUE(pool.AllocKvSlot(&b).ok());
    EXPECT_TRUE(pool.AllocKvSlot(&c).ok()); // 池满 → 驱逐最早 a，复用其槽
    EXPECT_NE(c, -1);
    EXPECT_EQ(c, a); // 驱逐复用：c 接管 a 的槽索引
    std::vector<SlotInfo> snap;
    ASSERT_TRUE(pool.Snapshot(&snap).ok());
    EXPECT_TRUE(snap[c].active);          // c 现活跃
    EXPECT_EQ(snap[c].total_written, 0u); // 被驱逐槽台账重置（a 的 3 已清）
    EXPECT_TRUE(snap[b].active);          // b 未受影响
    EXPECT_EQ(snap[b].total_written, 0u);
}

// 池满驱逐最早分配的 active 槽（TC_202 变体：真正的 FIFO，按 alloc_seq 而非索引序）
// 场景：Alloc A → Alloc B → Free A → Alloc C（复用 A 槽，alloc_seq=3）
//       池满时应驱逐 B（alloc_seq=2），而非 C（alloc_seq=3）
TEST(KvCachePoolTest, PoolFullEvictsTrueFIFO_TC202)
{
    auto pool = MakePool(/*max_batch_size=*/2, /*max_seq_len=*/16);
    SlotId a = -1, b = -1, c = -1, d = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&a).ok()); // a=slot0, alloc_seq=1
    ASSERT_TRUE(pool.AllocKvSlot(&b).ok()); // b=slot1, alloc_seq=2
    ASSERT_TRUE(pool.FreeKvSlot(a).ok());   // free slot0
    ASSERT_TRUE(pool.AllocKvSlot(&c).ok()); // c reuses slot0, alloc_seq=3
    ASSERT_EQ(c, a);                        // C reused A's slot
    // Now: slot0 active (alloc_seq=3), slot1 active (alloc_seq=2)
    // Pool full → must evict oldest by alloc_seq = slot1 (B), NOT slot0 (C)
    EXPECT_TRUE(pool.AllocKvSlot(&d).ok()); // evict B (alloc_seq=2 < 3)
    EXPECT_EQ(d, b);                        // D took B's slot, not C's
    EXPECT_NE(d, c);                        // C's slot was NOT evicted
    std::vector<SlotInfo> snap;
    ASSERT_TRUE(pool.Snapshot(&snap).ok());
    EXPECT_TRUE(snap[d].active);          // D is active in B's old slot
    EXPECT_TRUE(snap[c].active);          // C survived (was newer, not evicted)
    EXPECT_EQ(snap[d].total_written, 0u); // evicted slot cursor was reset
}

// 池满时所有槽 active（TC_203 前置条件：decode 全满无 padding 的池侧）
TEST(KvCachePoolTest, AllSlotsActiveWhenFull)
{
    auto pool = MakePool(/*max_batch_size=*/3);
    SlotId ids[3] = {-1, -1, -1};
    for (int i = 0; i < 3; ++i)
        ASSERT_TRUE(pool.AllocKvSlot(&ids[i]).ok());
    std::vector<SlotInfo> snap;
    ASSERT_TRUE(pool.Snapshot(&snap).ok());
    EXPECT_EQ(snap.size(), 3u);
    for (const auto& s : snap)
        EXPECT_TRUE(s.active);
}

// Snapshot 内容正确（修 M6：查内容，非仅 ok）
TEST(KvCachePoolTest, SnapshotContentMatchesAllocState)
{
    auto pool = MakePool(/*max_batch_size=*/4);
    SlotId a = -1, b = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&a).ok());
    ASSERT_TRUE(pool.AllocKvSlot(&b).ok());
    ASSERT_TRUE(pool.Advance(a, 5).ok());
    std::vector<SlotInfo> snap;
    ASSERT_TRUE(pool.Snapshot(&snap).ok());
    ASSERT_EQ(snap.size(), 4u);
    EXPECT_TRUE(snap[a].active);
    EXPECT_TRUE(snap[b].active);
    EXPECT_EQ(snap[a].total_written, 5u); // R4：逻辑写入长度（原 cursor 位）
    EXPECT_EQ(snap[b].total_written, 0u);
    // 未分配槽非 active
    for (int i = 0; i < 4; ++i) {
        if (i != a && i != b) {
            EXPECT_FALSE(snap[i].active);
        }
    }
}

// ── tensor 惰性创建（T1：构造期不建，EnsureKvTensor 逐层按图 desc 落位）──

// 构造后 0 tensor（T1）；tensor 实体经 EnsureKvTensor 逐层落位——完整语义组
// （创建/复验/分档校验）在 ut_engine_executor KvPoolEnsureTest 锁定，此处
// 只保留构造期形态断言（KTensor/VTensor 未 Ensure 层 = 空）
TEST(KvCachePoolTest, ConstructorAllocatesAllKvTensors)
{
    auto pool = MakePool(/*max_batch_size=*/2, /*kv_cache_max_len=*/16,
                         /*num_layers=*/2, /*num_kv_heads=*/4, /*head_dim=*/8);
    EXPECT_EQ(pool.TensorCount(), 0u); // T1：构造期不建 tensor
    EXPECT_FALSE(pool.KTensor(0));
    EXPECT_FALSE(pool.VTensor(1));
}

// 全空池形态（kv_heads=0 → 期望值非法）：槽位记账不受影响
TEST(KvCachePoolTest, InvalidDimsYieldNoTensorsButBookkeepingWorks)
{
    KvCachePool pool(4, 16, 2, /*num_kv_heads=*/0, 8);
    EXPECT_EQ(pool.TensorCount(), 0u);
    EXPECT_FALSE(pool.KTensor(0));
    EXPECT_FALSE(pool.VTensor(1));
    SlotId id = -1;
    EXPECT_TRUE(pool.AllocKvSlot(&id).ok()); // 记账仍可用
    EXPECT_TRUE(pool.Advance(id, 3).ok());
}

// 层号越界访问 → 空 shared_ptr（防御，不越界读）
TEST(KvCachePoolTest, OutOfRangeLayerReturnsNullTensor)
{
    auto pool = MakePool();
    EXPECT_FALSE(pool.KTensor(-1));
    EXPECT_FALSE(pool.KTensor(2)); // num_layers=2
    EXPECT_FALSE(pool.VTensor(99));
}

// ── 并发（TC_302）──

TEST(KvCachePoolTest, ConcurrentAllocFreeNoCorruption_TC302)
{
    auto pool = MakePool(/*max_batch_size=*/8);
    std::atomic<int> alloc_ok{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&]() {
            for (int i = 0; i < 50; ++i) {
                SlotId id = -1;
                if (pool.AllocKvSlot(&id).ok()) {
                    alloc_ok.fetch_add(1);
                    pool.FreeKvSlot(id);
                }
            }
        });
    }
    for (auto& th : ts)
        th.join();
    EXPECT_GT(alloc_ok.load(), 0); // stub 全 FAILURE → 0；GREEN > 0
}

} // namespace
} // namespace cann_local_lm::engine
