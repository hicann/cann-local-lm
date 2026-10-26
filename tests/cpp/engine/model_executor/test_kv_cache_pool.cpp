// TEST-WRITE(RED)：KV Cache 池单元测试（审计修订版）。
// 关联 SPEC §4.1.2 / DESIGN M-02 / D-13。修 M2/M3（去误挂 TC 号）。
#include "core/component/kv_cache_pool/kv_cache_pool.h"

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

namespace cann_lm::engine {
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

// R5/ADR-0001 环形化：Advance 越 kv_max 绕回不报错（原"越界即 FAILURE"质量
// 红线作废，容量硬约束前移到 Submit 前置校验）。total_written 保逻辑绝对值，
// write_pos 物理绕回 kv_max。
TEST(KvCachePoolTest, AdvanceWrapsAroundOnOverrun)
{
    auto pool = MakePool(/*max_batch_size=*/2, /*kv_cache_max_len=*/4);
    SlotId id = -1;
    ASSERT_TRUE(pool.AllocKvSlot(&id).ok());
    EXPECT_TRUE(pool.Advance(id, 3).ok()); // total_written=3 (< kv_max=4)
    EXPECT_EQ(pool.TotalWritten(id), 3u);
    EXPECT_EQ(pool.NextPosition(id), 3u);  // write_pos=3
    EXPECT_TRUE(pool.Advance(id, 2).ok()); // total_written=5 (>4) → 绕回不报错
    EXPECT_EQ(pool.TotalWritten(id), 5u);  // 逻辑绝对值累加
    EXPECT_EQ(pool.NextPosition(id), 1u);  // write_pos=MapSlot(5)=1 (5%4=1)
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
    EXPECT_EQ(snap[c].total_written, 0u); // 被驱逐槽 cursor 重置（a 的 3 已清）
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
    EXPECT_EQ(snap[a].total_written, 5u);
    EXPECT_EQ(snap[b].total_written, 0u);
    // 未分配槽非 active
    for (int i = 0; i < 4; ++i) {
        if (i != a && i != b) {
            EXPECT_FALSE(snap[i].active);
        }
    }
}

// ── EnsureKvTensor 惰性创建全部 KV tensor（R1/T1：ctor 仅记账，tensor 由首次
//    executor Load 期 EnsureKvTensor 按 om 图 desc 直建）──

// EnsureKvTensor 建 2×num_hidden_layers 个 tensor：计数/层序/dims/dtype/size/命名
// （batch=2 ≠ kv_heads=4：dims 各维取可区分值，钉死图 desc→tensor dims 转发——
//  池 expect_dims_ HF 序 [batch, kv_len, kv_heads, dim] 与图 desc dims 逐位比对，
//  R3 失配致命；本例图 desc 全对齐 → 直建，tensor dims = 图 desc dims）
TEST(KvCachePoolTest, ConstructorAllocatesAllKvTensors)
{
    auto pool = MakePool(/*max_batch_size=*/2, /*kv_cache_max_len=*/16,
                         /*num_layers=*/2, /*num_kv_heads=*/4, /*head_dim=*/8);
    // ctor 仅武装（记账），未建 tensor（lazy-create R1/T1）
    EXPECT_EQ(pool.TensorCount(), 0u);
    // 模拟 executor Load：按 om 图工位 desc（HF 序 = 池 expect_dims_）EnsureKvTensor
    TensorDesc desc;
    desc.dims = {2, 4, 16, 8}; // HF 序 [batch, heads, kv_len, dim]
    desc.dataType = DataType::FLOAT16;
    for (int layer = 0; layer < 2; ++layer) {
        ASSERT_TRUE(pool.EnsureKvTensor(layer, /*is_value=*/false, desc).ok());
        ASSERT_TRUE(pool.EnsureKvTensor(layer, /*is_value=*/true, desc).ok());
    }
    EXPECT_EQ(pool.TensorCount(), 4u); // 2 层 × (K+V)
    const std::vector<int32_t> expect_dims = {2, 4, 16, 8};
    const size_t expect_size = 2u * 4u * 16u * 8u * 2u; // × fp16 字节
    for (int layer = 0; layer < 2; ++layer) {
        auto k = pool.KTensor(layer);
        auto v = pool.VTensor(layer);
        ASSERT_TRUE(k);
        ASSERT_TRUE(v);
        EXPECT_EQ(k->GetTensorDesc().dims, expect_dims);
        EXPECT_EQ(v->GetTensorDesc().dims, expect_dims);
        EXPECT_EQ(k->GetTensorDesc().dataType, DataType::FLOAT16);
        EXPECT_EQ(k->GetSize(), expect_size);
        EXPECT_EQ(v->GetSize(), expect_size);
        EXPECT_EQ(k->GetTensorDesc().tensorName, "kv_cache_k_" + std::to_string(layer));
        EXPECT_EQ(v->GetTensorDesc().tensorName, "kv_cache_v_" + std::to_string(layer));
    }
    // K/V tensor 各自独立（非共享句柄）
    EXPECT_NE(pool.KTensor(0).get(), pool.VTensor(0).get());
    EXPECT_NE(pool.KTensor(0).get(), pool.KTensor(1).get());
}

// 维度非法（kv_heads=0）→ tensor 无产出，但槽位记账不受影响
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
} // namespace cann_lm::engine
