// PromptCache 环形 KV 池 + 缓存台账 + 增量切块 UT（PromptCache 改造 S2-S5）。
// 关联 docs/prompt_cache_spec.md（REQ-3/REQ-4/REQ-5/REQ-6/REQ-9）、
// docs/prompt_cache_design.md §3.2/§4.1/§4.2/§4.3。
// 纯 host 逻辑用例（无图、无 NDK/stub so 依赖——池构造期不建 tensor / 台账
// 自持 / SliceChunks 纯切片），全部无条件运行。
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "pipeline/history_prompt_manager.h"
#include "pipeline/preprocessor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace cann_local_lm::engine {
namespace {

// 池构造便捷版：2 层稠密（层型与本组用例无关，MapSlot 只看 kv_max/init）
KvCachePool MakePool(int max_batch_size, int kv_max) { return KvCachePool(max_batch_size, kv_max, 2, 4, 8); }

// ═══════════════════════════════════════════════════════════════════════════
// REQ-3 MapSlot 锚定式环形映射（spec 验收表数值例）
// ═══════════════════════════════════════════════════════════════════════════

// kv_max=8、init_token_len=4：anchor=4、ring=4——[0,8) 恒等直通；位 8/12 绕回
// 槽 4、位 13→槽 5、位 16 第二圈→槽 4；锚定段 [0,4) 不参与绕回。
TEST(KvPoolMapSlotTest, AnchoredRingReq3Table)
{
    KvCachePool pool = MakePool(2, 8);
    pool.SetInitTokenLen(4);
    for (std::size_t p = 0; p < 8; ++p) {
        EXPECT_EQ(pool.MapSlot(p), p) << "pos=" << p << "（< kv_max 恒等）";
    }
    EXPECT_EQ(pool.MapSlot(8), 4u);  // 4+(8-4)%4=4
    EXPECT_EQ(pool.MapSlot(12), 4u); // 4+(12-4)%4=4（首圈绕回 + 第二圈同位）
    EXPECT_EQ(pool.MapSlot(13), 5u);
    EXPECT_EQ(pool.MapSlot(16), 4u); // 第二圈起点
    EXPECT_EQ(pool.init_token_len(), 4u);
}

// anchor=0（未设 init_token_len）：全缓冲圆环，纯模绕回。
TEST(KvPoolMapSlotTest, AnchorZeroFullRing)
{
    KvCachePool pool = MakePool(2, 8);
    EXPECT_EQ(pool.init_token_len(), 0u);
    EXPECT_EQ(pool.MapSlot(8), 0u);
    EXPECT_EQ(pool.MapSlot(15), 7u);
    EXPECT_EQ(pool.MapSlot(16), 0u);
}

// init_token_len >= kv_max 退化保护：几何不成立 → anchor 视为 0 纯模绕回。
TEST(KvPoolMapSlotTest, AnchorGeKvMaxDegradesToMod)
{
    KvCachePool pool = MakePool(2, 8);
    pool.SetInitTokenLen(8);         // == kv_max
    EXPECT_EQ(pool.MapSlot(12), 4u); // 12%8（anchor 退化 0）
    pool.SetInitTokenLen(10);        // > kv_max
    EXPECT_EQ(pool.MapSlot(12), 4u);
    EXPECT_EQ(pool.MapSlot(8), 0u);
}

// ── 锚定段构造期供给（2026-10-22 静态化，meta_json model_config.init_token_len）──

// 构造期合法锚定（0 < v < kv_max）：init_token_len 即时生效——与上方 Set 手动
// 路径（AnchoredRingReq3Table）区分，此处锁"构造参数供给"链路本身。
TEST(KvPoolCtorAnchorTest, ConstructorAnchorsInitTokenLen)
{
    // kv_max=8, init=4 → anchor=4, ring=4
    KvCachePool pool(2, 8, 2, 4, 8, {}, /*sliding_window=*/0,
                     /*init_token_len=*/4);
    EXPECT_EQ(pool.init_token_len(), 4u);
    EXPECT_EQ(pool.MapSlot(8), 4u);  // 4+(8-4)%4=4（首圈绕回）
    EXPECT_EQ(pool.MapSlot(13), 5u); // 4+(13-4)%4=5
    EXPECT_EQ(pool.MapSlot(3), 3u);  // [0,init) 恒等直通
    // Advance/NextPosition 全链经 MapSlot 消费构造锚定
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.Advance(slot, 12).code, Status::SUCCESS);
    EXPECT_EQ(pool.NextPosition(slot), 4u); // MapSlot(12)=4
}

// 构造期缺省/0：无锚定段（全缓冲圆环）——现状行为零变化的迁移锚点。
TEST(KvPoolCtorAnchorTest, ConstructorDefaultZeroKeepsFullRing)
{
    KvCachePool pool = MakePool(2, 8); // 既有便捷直构（init_token_len 缺省 0）
    EXPECT_EQ(pool.init_token_len(), 0u);
    EXPECT_EQ(pool.MapSlot(12), 4u); // 纯模 12%8
    EXPECT_EQ(pool.MapSlot(8), 0u);
}

// 构造期越界（≥ kv_max / 负值）：WARN 降 0（防御深度——解析层段降级已拒，
// 正常不可达；池不静默采信残值）。
TEST(KvPoolCtorAnchorTest, ConstructorOutOfRangeInitDegradesToZero)
{
    {
        KvCachePool pool(2, 8, 2, 4, 8, {}, 0, /*init_token_len=*/8); // == kv_max
        EXPECT_EQ(pool.init_token_len(), 0u);
        EXPECT_EQ(pool.MapSlot(12), 4u); // 纯模
    }
    {
        KvCachePool pool(2, 8, 2, 4, 8, {}, 0, /*init_token_len=*/100);
        EXPECT_EQ(pool.init_token_len(), 0u);
    }
    {
        KvCachePool pool(2, 8, 2, 4, 8, {}, 0, /*init_token_len=*/-1);
        EXPECT_EQ(pool.init_token_len(), 0u);
    }
}

// miss 重建恢复配置基线（R13 修订 2026-10-22）：configured > 0 时
// ResetLedgerOnSlot 恢复配置值而非归零。
TEST(KvPoolCtorAnchorTest, ResetLedgerRestoresConfiguredAnchor)
{
    KvCachePool pool(2, 8, 2, 4, 8, {}, 0, /*init_token_len=*/4);
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.Advance(slot, 12).code, Status::SUCCESS);
    pool.ResetLedgerOnSlot(slot);
    EXPECT_EQ(pool.init_token_len(), 4u) << "R13 修订：miss 恢复配置基线";
    EXPECT_EQ(pool.MapSlot(12), 4u);
}

// kv_max=0 塌零防御：MapSlot 恒 0（Assemble* 侧拦 FAILURE，池侧记账不拦）。
TEST(KvPoolMapSlotTest, ZeroKvMaxMapsEverythingToZero)
{
    KvCachePool pool = MakePool(2, 0);
    EXPECT_EQ(pool.MapSlot(0), 0u);
    EXPECT_EQ(pool.MapSlot(123), 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Advance 环形推进（spec REQ-3：越 kv_max 绕回不报错——原红线作废）
// ═══════════════════════════════════════════════════════════════════════════

// 连续 Advance 跨过 kv_max：恒 SUCCESS、TotalWritten 单调累加、
// NextPosition 恒 == MapSlot(TotalWritten)（write_pos 重映射一致性）。
TEST(KvPoolAdvanceTest, WrapAroundKvMaxMonotonicAndRemapped)
{
    KvCachePool pool = MakePool(2, 8);
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);

    // 跨 kv_max 多段推进全 SUCCESS（含恰好落在 kv_max 的步）+ 单调 + 重映射不变量
    const std::size_t steps[] = {5, 3, 2, 4, 100};
    std::size_t expect_total = 0;
    for (std::size_t n : steps) {
        ASSERT_EQ(pool.Advance(slot, n).code, Status::SUCCESS) << "n=" << n;
        expect_total += n;
        EXPECT_EQ(pool.TotalWritten(slot), expect_total) << "n=" << n;
        EXPECT_EQ(pool.NextPosition(slot), pool.MapSlot(expect_total))
            << "n=" << n << "（write_pos == MapSlot(total_written)）";
    }
    EXPECT_EQ(pool.TotalWritten(slot), 114u); // 5+3+2+4+100
    EXPECT_EQ(pool.NextPosition(slot), 2u);   // MapSlot(114)（anchor=0：114%8=2）
}

// 非 active 槽 Advance 拒绝（环形化不改变 active 守卫）。
TEST(KvPoolAdvanceTest, InactiveSlotRejected)
{
    KvCachePool pool = MakePool(2, 8);
    // 未分配槽：记账未激活
    EXPECT_EQ(pool.Advance(0, 1).code, Status::FAILURE);
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.FreeKvSlot(slot).code, Status::SUCCESS);
    EXPECT_EQ(pool.Advance(slot, 1).code, Status::FAILURE) << "free 后 inactive";
}

// ═══════════════════════════════════════════════════════════════════════════
// ResumeSlot（REQ-4 命中行：total_written 回退到复用边界 + active 置位）
// ═══════════════════════════════════════════════════════════════════════════

// 命中复用回退：Advance 10 → Resume 6 → total/access 一致 + active 保持；
// 前进 target（> 当前）→ INVALID_ARGUMENT（防御拒绝，不经 Advance）。
TEST(KvPoolResumeSlotTest, RollbackOnlyAndActiveSet)
{
    KvCachePool pool = MakePool(2, 16);
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.Advance(slot, 10).code, Status::SUCCESS);

    ASSERT_EQ(pool.ResumeSlot(slot, 6).code, Status::SUCCESS);
    EXPECT_EQ(pool.TotalWritten(slot), 6u);
    EXPECT_EQ(pool.NextPosition(slot), 6u); // MapSlot(6)=6（无绕回）
    // active 置位可观测：inactive 槽 Advance 可成功
    ASSERT_EQ(pool.Advance(slot, 1).code, Status::SUCCESS);

    // 前进拒绝：target > 当前 total_written
    EXPECT_EQ(pool.ResumeSlot(slot, 100).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.TotalWritten(slot), 7u) << "拒绝时记账不动";

    // free 后 ResumeSlot 重新激活（active 置位语义）
    ASSERT_EQ(pool.FreeKvSlot(slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.ResumeSlot(slot, 2).code, Status::SUCCESS);
    EXPECT_EQ(pool.TotalWritten(slot), 2u);
    EXPECT_EQ(pool.Advance(slot, 1).code, Status::SUCCESS) << "Resume 后 active 可推进";
}

// ═══════════════════════════════════════════════════════════════════════════
// RollbackTo（R10 失败回退：只许回退 + write_pos 重映射）
// ═══════════════════════════════════════════════════════════════════════════

// 绕回后的回退：write_pos 按回退点重映射（非简单减法）；前进拒绝。
TEST(KvPoolRollbackTest, RemapsWritePosAndRefusesForward)
{
    KvCachePool pool = MakePool(2, 8);
    pool.SetInitTokenLen(4);
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.Advance(slot, 12).code, Status::SUCCESS);
    EXPECT_EQ(pool.NextPosition(slot), 4u); // MapSlot(12)=4

    // 回退到 9：write_pos = MapSlot(9) = 4+(9-4)%4 = 5（重映射验证）
    ASSERT_EQ(pool.RollbackTo(slot, 9).code, Status::SUCCESS);
    EXPECT_EQ(pool.TotalWritten(slot), 9u);
    EXPECT_EQ(pool.NextPosition(slot), 5u);

    // 回退到 3（< kv_max 恒等区）：write_pos = 3
    ASSERT_EQ(pool.RollbackTo(slot, 3).code, Status::SUCCESS);
    EXPECT_EQ(pool.NextPosition(slot), 3u);

    // 同值回退（幂等边界）+ 前进拒绝（日志 kv_pool.rollback: refuse forward move）
    ASSERT_EQ(pool.RollbackTo(slot, 3).code, Status::SUCCESS);
    EXPECT_EQ(pool.RollbackTo(slot, 4).code, Status::INVALID_ARGUMENT);
    EXPECT_EQ(pool.TotalWritten(slot), 3u) << "拒绝时记账不动";
}

// ═══════════════════════════════════════════════════════════════════════════
// ResetLedgerOnSlot（R13"一切重建"：记账归零 + init_token_len 归零）
// ═══════════════════════════════════════════════════════════════════════════

// miss 重建：Set init 5 + Advance 后 Reset → total/write_pos 归零且
// init_token_len()==0（构造未配置锚定 ⇒ 基线 0，reset 回基线——2026-10-22
// 静态化后语义；configured>0 的恢复路径见 test_kv_cache_pool
// ResetLedgerRestoresConfiguredAnchor）。
TEST(KvPoolResetLedgerTest, ClearsBookkeepingAndInitTokenLen)
{
    KvCachePool pool = MakePool(2, 16);
    pool.SetInitTokenLen(5);
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.Advance(slot, 9).code, Status::SUCCESS);
    ASSERT_EQ(pool.init_token_len(), 5u);

    pool.ResetLedgerOnSlot(slot);
    EXPECT_EQ(pool.TotalWritten(slot), 0u);
    EXPECT_EQ(pool.NextPosition(slot), 0u);
    EXPECT_EQ(pool.init_token_len(), 0u) << "R13：init_token_len 随 miss 重建归零";
    // 归零后 MapSlot 恢复纯模语义（anchor=0）
    EXPECT_EQ(pool.MapSlot(12), 12u);
}

// 越界 slot id 防御：ResetLedgerOnSlot 不越权写（noexcept，无副作用可观测）。
TEST(KvPoolResetLedgerTest, OutOfRangeSlotIsNoop)
{
    KvCachePool pool = MakePool(2, 8);
    pool.SetInitTokenLen(3);
    pool.ResetLedgerOnSlot(-1);
    pool.ResetLedgerOnSlot(99);
    EXPECT_EQ(pool.init_token_len(), 3u) << "越界 slot 不动 init_token_len";
}

// ═══════════════════════════════════════════════════════════════════════════
// MatchCachedTokens 规则表（design §3.2 / spec REQ-4 逐行）
// ═══════════════════════════════════════════════════════════════════════════

// 规则表·禁用行（R3）：SetEnabled(false) 后比对恒 0、台账不动、stats 填 0。
TEST(HistoryMatchTest, DisabledReturnsZeroWithoutSideEffect)
{
    HistoryPromptManager m;
    m.SetEnabled(false);
    m.AppendDecodedToken(7);
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({7, 7, 7}, 16, &st), 0u);
    EXPECT_EQ(m.Size(), 1u) << "禁用行走账不动";
    EXPECT_EQ(st.history_len, 1u);
    EXPECT_EQ(st.lcp, 0u);
    EXPECT_FALSE(st.wrapped);
}

// 规则表·空台账行（首轮）：miss 且无账可动，stats 以比对前长度 0 填。
TEST(HistoryMatchTest, EmptyLedgerReturnsZero)
{
    HistoryPromptManager m;
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3}, 16, &st), 0u);
    EXPECT_EQ(m.Size(), 0u);
    EXPECT_EQ(st.history_len, 0u);
    EXPECT_EQ(st.lcp, 0u);
    EXPECT_FALSE(st.wrapped);
}

// 规则表·首 token 失配行（无绕回 LCP==0）：miss 且台账清空。
TEST(HistoryMatchTest, FirstTokenMismatchClearsLedger)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u})
        m.AppendDecodedToken(t);
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({9, 2, 3}, 16, &st), 0u);
    EXPECT_EQ(m.Size(), 0u) << "miss 清账";
    EXPECT_EQ(st.lcp, 0u);
    EXPECT_EQ(st.history_len, 3u);
    EXPECT_FALSE(st.wrapped);
}

// 规则表·普通命中行（LCP > 0 且输入长于台账）：reusable = LCP，台账截到 LCP。
TEST(HistoryMatchTest, PrefixHitReturnsLcpAndTruncates)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u})
        m.AppendDecodedToken(t); // 台账长 4
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 9, 8}, 16, &st), 3u);
    EXPECT_EQ(m.Size(), 3u) << "台账截到 LCP";
    EXPECT_EQ(st.history_len, 4u);
    EXPECT_EQ(st.lcp, 3u);
    EXPECT_FALSE(st.wrapped);
}

// 规则表·前缀重合行（LCP == 输入长 < 台账长，两轮重发同句形态；DTS202610220945
// Fix A 一般化）：reusable = LCP − 1（末 token 强制重算——增量段为空时末位
// logits 只能由重算 prefill 产出），台账截到 LCP−1。
TEST(HistoryMatchTest, InputPrefixOfHistoryReturnsInputLen)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u, 5u})
        m.AppendDecodedToken(t);
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3}, 16, &st), 2u) << "LCP-1（末 token 重算）";
    EXPECT_EQ(st.lcp, 3u);
    EXPECT_EQ(st.history_len, 5u);
    EXPECT_EQ(m.Size(), 2u) << "台账截到 LCP-1";
    EXPECT_FALSE(st.wrapped);
}

// 前缀重合（数值例 2，DTS202610220945 §6.1）：台账 {1,2,3,4}、输入 {1,2,3}
// （LCP=3=输入长 < 台账 4）→ reusable=LCP−1=2，台账截到 2，lcp 可观测 3。
TEST(HistoryMatchTest, PrefixOverlapRecomputesLastToken)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u})
        m.AppendDecodedToken(t);
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3}, 16, &st), 2u) << "LCP-1";
    EXPECT_EQ(m.Size(), 2u);
    EXPECT_EQ(st.lcp, 3u);
    EXPECT_EQ(st.history_len, 4u);
}

// 前缀重合 + 输入长 1：退化 reusable = 0 = 清账（等价 miss，池侧由
// GenerateAsync miss 收口分支 ResetLedgerOnSlot 处理）；lcp=1 可观测。
TEST(HistoryMatchTest, SingleTokenInputPrefixDegeneratesZero)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u})
        m.AppendDecodedToken(t);
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1}, 16, &st), 0u);
    EXPECT_EQ(m.Size(), 0u) << "边界长 1 退化清账";
    EXPECT_EQ(st.lcp, 1u);
}

// 回归锚：输入超台账且台账整段命中（LCP == 台账长 < 输入长，延续轮保守口径）
// → 减一行为不受 Fix A 一般化扰动（现行即减一）。
TEST(HistoryMatchTest, InputLongerThanLedgerFullOverlapKeepsMinusOne)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u})
        m.AppendDecodedToken(t);
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 4, 9, 8}, 16, &st), 3u);
    EXPECT_EQ(m.Size(), 3u);
    EXPECT_EQ(st.lcp, 4u);
    EXPECT_EQ(st.history_len, 4u);
}

// 规则表·全命中行（REQ-5/R7：LCP == 台账长，input == history）：
// reusable = LCP - 1（末 token 强制重算，保证至少 1 个真实 prefill 块）。
TEST(HistoryMatchTest, FullMatchRecomputesLastToken)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u})
        m.AppendDecodedToken(t);
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 4}, 16, &st), 3u) << "L-1";
    EXPECT_EQ(m.Size(), 3u) << "台账截断生效：Size == reusable";
    EXPECT_EQ(st.lcp, 4u);
}

// 全命中特例（台账长 1）：reusable = 0 = 等价全量重算（REQ-5 特例）。
TEST(HistoryMatchTest, FullMatchSingleTokenYieldsZero)
{
    HistoryPromptManager m;
    m.AppendDecodedToken(42);
    EXPECT_EQ(m.MatchCachedTokens({42}, 16), 0u);
}

// 规则表·绕回未全覆盖行（R6）：台账长 > kv_max（wrapped）且 LCP < 台账长
// → miss 清账（窗口内发散同判 miss）。
TEST(HistoryMatchTest, WrappedPartialCoverIsMiss)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u, 5u})
        m.AppendDecodedToken(t); // 长 5 > kv_max=4
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 9}, 4, &st), 0u) << "绕回+未全覆盖";
    EXPECT_EQ(m.Size(), 0u) << "清账重建";
    EXPECT_TRUE(st.wrapped);
    EXPECT_EQ(st.lcp, 3u);
}

// 绕回 + 完全覆盖（LCP == 台账长）：命中整段，reusable = 台账长 - 1（R7），
// wrapped 标志 true；台账截到台账长-1。
TEST(HistoryMatchTest, WrappedFullCoverHitsMinusOne)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u, 5u})
        m.AppendDecodedToken(t); // 长 5 > kv_max=4
    HistoryPromptManager::MatchStats st;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 4, 5, 6}, 4, &st), 4u) << "5-1";
    EXPECT_EQ(m.Size(), 4u);
    EXPECT_TRUE(st.wrapped);
    EXPECT_EQ(st.lcp, 5u);
    EXPECT_EQ(st.history_len, 5u);
}

// U6（方案 §5）：R6 绕回后台账规则——台账长 36 > kv_max=16（多圈绕回，
// 与 U5 单轮绕回记账全链同数值布局）。部分覆盖（LCP < 台账长）→ miss
// 清账；完全覆盖（LCP == 台账长）→ reusable = 36-1 = 35（R7 减一）。
TEST(HistoryMatchTest, WrappedLedger36PartialMissFullCoverHits)
{
    std::vector<TokenId> ledger(36);
    for (std::size_t i = 0; i < ledger.size(); ++i)
        ledger[i] = static_cast<TokenId>(100 + i);
    { // 部分覆盖：末位发散（LCP=35 < 36）→ R6 miss 清账
        HistoryPromptManager m;
        for (TokenId t : ledger)
            m.AppendDecodedToken(t);
        HistoryPromptManager::MatchStats st;
        std::vector<TokenId> ids = ledger;
        ids.pop_back(); // 长 35，真前缀
        EXPECT_EQ(m.MatchCachedTokens(ids, 16, &st), 0u) << "绕回+部分覆盖 miss";
        EXPECT_EQ(m.Size(), 0u) << "清账重建";
        EXPECT_TRUE(st.wrapped);
        EXPECT_EQ(st.lcp, 35u);
    }
    { // 完全覆盖：重发全同序列 + 1 新 token → R7 reusable = 35
        HistoryPromptManager m;
        for (TokenId t : ledger)
            m.AppendDecodedToken(t);
        HistoryPromptManager::MatchStats st;
        std::vector<TokenId> ids = ledger;
        ids.push_back(200);
        EXPECT_EQ(m.MatchCachedTokens(ids, 16, &st), 35u) << "绕回+全覆盖 36-1";
        EXPECT_EQ(m.Size(), 35u) << "台账截到 reusable";
        EXPECT_TRUE(st.wrapped);
        EXPECT_EQ(st.lcp, 36u);
        EXPECT_EQ(st.history_len, 36u);
    }
}

// L > kv_max 的 wrapped 观测面（绕回可由单轮超长输入直接触发——2026-10-23
// REQ-9 预算闸措辞刷新：输入上界归 max_io_tokens、与 kv_max 解耦）：
// reusable = 台账长-1（R7 规则值），wrapped 标志记录。
// 说明：输入上界由 GenerateAsync 提交期 max_io_tokens 预算闸把关（REQ-9，
// 与 KV 缓冲容量解耦），Match 面承诺不减 1 规则 + wrapped 可观测（本例
// 台账长 6、kv_max=4 → 全覆盖 reusable = 5 > kv_max-1——环形语义下
// reusable 供 pool ResumeSlot 回退 total_written，物理 KV 落位由 MapSlot
// 环形映射承担；此处直接断言 wrapped 标志 + reusable == 台账长-1 规则值）。
TEST(HistoryMatchTest, WrappedFlagObservableWhenLedgerExceedsKvMax)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u, 5u, 6u})
        m.AppendDecodedToken(t); // 长 6
    HistoryPromptManager::MatchStats st;
    // 覆盖全台账：reusable = 6-1 = 5（R7 规则值）；wrapped=true
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 4, 5, 6}, 4, &st), 5u);
    EXPECT_TRUE(st.wrapped) << "台账长 6 > kv_max 4";
    EXPECT_EQ(st.history_len, 6u);
}

// ═════════════════════════════════════════════════════════════════════════════
// 滑窗收窄门（sliding_kv_seq_dim_rebind D6/D10）：keep_cap = min(C, kv_max)
// ——新契滑窗层环容量 C = W+P（SlidingLayerWidth）、full 层环恒 kv_max；旧契
// （sw_hd=0）恒 kv_max（b688221 的 w 收窄作废 D8）。台账 > keep_cap 即
// wrapped。台账本身无层型信息，直接以 keep_cap 传参锁"门收窄后 wrapped 判定"
// 语义（pipeline 侧取值口径的台账面镜像：新契传 min(C, kv_max)、旧契传 kv_max）。
// ═════════════════════════════════════════════════════════════════════════════

// D10 正面：kv_max=16、C=4 ⇒ keep_cap=4。台账长 5（>C、≤kv_max——旧门
// （判 kv_max）会放行的区间）部分命中 → 新门 wrapped=true 触发 R6 ⇒
// miss 清账（滑窗层窗尾已落被覆写圈，复用即静默读旧代）。
TEST(HistoryMatchTest, SlidingNarrowGateLedgerInOldGapPartialIsMiss)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u, 5u})
        m.AppendDecodedToken(t); // 长 5 ∈ (C=4, kv_max=16]
    HistoryPromptManager::MatchStats st;
    const std::size_t keep_cap = 4; // min(C=4, kv_max=16)
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 9}, keep_cap, &st), 0u) << "台账 > C：部分命中必须 miss";
    EXPECT_EQ(m.Size(), 0u) << "清账重建";
    EXPECT_TRUE(st.wrapped) << "滑窗层小环先绕回";
    EXPECT_EQ(st.lcp, 3u);
}

// D10 同账全命中：LCP == 台账长 5 > keep_cap=4 ⇒ wrapped 但 R6 全覆盖支路
// 仍复用（reusable = 5-1 = 4）——全命中窗口尾 = 台账末重推前存活圈，不受
// 窗尾回落影响（§2.6 全命中支路论证）。
TEST(HistoryMatchTest, SlidingNarrowGateFullCoverStillHits)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u, 5u})
        m.AppendDecodedToken(t); // 长 5 > cap 4
    HistoryPromptManager::MatchStats st;
    const std::size_t keep_cap = 4;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 4, 5, 6}, keep_cap, &st), 4u) << "5-1";
    EXPECT_EQ(m.Size(), 4u);
    EXPECT_TRUE(st.wrapped);
    EXPECT_EQ(st.lcp, 5u);
    EXPECT_EQ(st.history_len, 5u);
}

// D10 边界台账长 ≤ C：台账长 4 == keep_cap=4（滑窗层环恰未绕回、
// kv 环更未绕回）⇒ wrapped=false，部分命中照常复用 LCP（现状语义——门未触发）。
TEST(HistoryMatchTest, SlidingGateLedgerWithinWindowReusesAsBefore)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u})
        m.AppendDecodedToken(t); // 长 4 == C
    HistoryPromptManager::MatchStats st;
    const std::size_t keep_cap = 4;
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 9, 8}, keep_cap, &st), 3u) << "环内现状复用";
    EXPECT_EQ(m.Size(), 3u);
    EXPECT_FALSE(st.wrapped) << "台账长 == C：恰未绕回";
}

// D11 退化护栏（旧契，sw_hd=0）⇒ keep_cap = kv_max，台账 ∈ (0, kv_max]
// 全程现状逐值——sliding_kv_seq_dim_rebind D8：b688221 的 w 收窄作废，
// 回 pre-b688221 恒定 kv_max 门（分派只看配置 flag D9，不做宽度比较）。
TEST(HistoryMatchTest, DegradedModelGateIsKvMaxAsBefore)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u, 4u, 5u})
        m.AppendDecodedToken(t); // 长 5
    HistoryPromptManager::MatchStats st;
    const std::size_t keep_cap = 16; // 旧契（sw_hd=0）→ 恒 kv_max
    EXPECT_EQ(m.MatchCachedTokens({1, 2, 3, 9}, keep_cap, &st), 3u) << "现状普通命中";
    EXPECT_FALSE(st.wrapped);
    EXPECT_EQ(m.Size(), 3u);
}

// 台账不变量面（REQ-2）：Append/Size/Snapshot 一致 + Restore/Clear 全量语义。
TEST(HistoryMatchTest, LedgerAppendSizeSnapshotRestoreClear)
{
    HistoryPromptManager m;
    for (TokenId t : {1u, 2u, 3u})
        m.AppendDecodedToken(t);
    EXPECT_EQ(m.Size(), 3u);
    EXPECT_EQ(m.SnapshotTokens(), (std::vector<TokenId>{1, 2, 3}));

    auto snap = m.SnapshotTokens(); // R10 快照
    m.Clear();
    EXPECT_EQ(m.Size(), 0u);
    m.Restore(snap); // 失败重灌
    EXPECT_EQ(m.Size(), 3u);
    EXPECT_EQ(m.SnapshotTokens(), (std::vector<TokenId>{1, 2, 3}));
}

// ═══════════════════════════════════════════════════════════════════════════
// Preprocessor::SliceChunks 增量切块（spec REQ-6 验收数值例）
// ═══════════════════════════════════════════════════════════════════════════

// 数值例：ids 长 7、reusable=2、chunk=4 → 两块 [2,6) 与 [6,7)
// （首块满档、尾块自然收短；无过算、无对齐）。
TEST(SliceChunksTest, NumericExampleLen7Reuse2Chunk4)
{
    Preprocessor prep(/*tokenizer=*/nullptr, /*text_embedding=*/nullptr,
                      /*default_chunk_size=*/4);
    const std::vector<TokenId> ids{10, 11, 12, 13, 14, 15, 16};
    std::vector<std::vector<TokenId>> chunks;
    ASSERT_EQ(prep.SliceChunks(ids, /*reusable=*/2, &chunks).code, Status::SUCCESS);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0], (std::vector<TokenId>{12, 13, 14, 15})) << "首块 [2,6)";
    EXPECT_EQ(chunks[1], (std::vector<TokenId>{16})) << "尾块 [6,7)";
}

// reusable == ids.size()（全命中极端）：空块链 SUCCESS（上游 Match 全命中
// 已减 1 保证至少 1 真实块，此处防御语义）。
TEST(SliceChunksTest, FullReuseYieldsEmptyChunkChain)
{
    Preprocessor prep(nullptr, nullptr, 4);
    const std::vector<TokenId> ids{1, 2, 3};
    std::vector<std::vector<TokenId>> chunks{{9}}; // 预污染验证 clear 语义
    ASSERT_EQ(prep.SliceChunks(ids, /*reusable=*/3, &chunks).code, Status::SUCCESS);
    EXPECT_TRUE(chunks.empty());
}

// reusable > ids.size()：调用时序 bug 防御 → INVALID_ARGUMENT。
TEST(SliceChunksTest, ReuseGreaterThanLenRejected)
{
    Preprocessor prep(nullptr, nullptr, 4);
    const std::vector<TokenId> ids{1, 2, 3};
    std::vector<std::vector<TokenId>> chunks;
    EXPECT_EQ(prep.SliceChunks(ids, /*reusable=*/4, &chunks).code, Status::INVALID_ARGUMENT);
    EXPECT_TRUE(chunks.empty()) << "拒绝零半写";
}

// ids 为空（zero-token prompt 历史行为）：空块链 SUCCESS。
TEST(SliceChunksTest, EmptyIdsYieldsEmptyChunkChain)
{
    Preprocessor prep(nullptr, nullptr, 4);
    std::vector<std::vector<TokenId>> chunks;
    ASSERT_EQ(prep.SliceChunks({}, /*reusable=*/0, &chunks).code, Status::SUCCESS);
    EXPECT_TRUE(chunks.empty());
}

// reusable=0 全量切块 + 恰满整除：两整块无尾块（基线行为不回归）。
TEST(SliceChunksTest, ZeroReuseCutsEverythingAligned)
{
    Preprocessor prep(nullptr, nullptr, 3);
    const std::vector<TokenId> ids{1, 2, 3, 4, 5, 6};
    std::vector<std::vector<TokenId>> chunks;
    ASSERT_EQ(prep.SliceChunks(ids, /*reusable=*/0, &chunks).code, Status::SUCCESS);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0], (std::vector<TokenId>{1, 2, 3}));
    EXPECT_EQ(chunks[1], (std::vector<TokenId>{4, 5, 6}));
}

// default_chunk_size ≤ 0 回退链：chunk = ids.size() 单块兜底（1024 回退位在
// 构造器内已是本值，此处构造传 0 验证不崩溃且切全量单块）。
TEST(SliceChunksTest, NonPositiveChunkSizeFallsBackToSingleBlock)
{
    Preprocessor prep(nullptr, nullptr, 0);
    const std::vector<TokenId> ids{1, 2, 3, 4, 5, 6, 7};
    std::vector<std::vector<TokenId>> chunks;
    ASSERT_EQ(prep.SliceChunks(ids, 2, &chunks).code, Status::SUCCESS);
    EXPECT_EQ(chunks.size(), 1u); // chunk_size 兜底为全量（构造器内置回退）
    EXPECT_EQ(chunks[0], (std::vector<TokenId>{3, 4, 5, 6, 7})) << "仍从 reusable=2 起切";
}

// out_chunks 空指针防御：INVALID_ARGUMENT（宁拒勿错）。
TEST(SliceChunksTest, NullOutRejected)
{
    Preprocessor prep(nullptr, nullptr, 4);
    EXPECT_EQ(prep.SliceChunks({1, 2}, 0, nullptr).code, Status::INVALID_ARGUMENT);
}

// ═══════════════════════════════════════════════════════════════════════════
// AllocKvSlot 分配/驱逐路径（design §4.2：池满 FIFO 驱逐最早 active 槽，U-4）
// ═══════════════════════════════════════════════════════════════════════════

// 池满驱逐：容量 2 全占后第三次 Alloc → 驱逐 alloc_seq 最小的槽 0 并复用；
// 被逐槽记账归零、新槽归零起记账（驱逐重置语义，kv_cache_pool.cpp:236-258）。
TEST(KvPoolAllocEvictTest, FullPoolEvictsEarliestActive)
{
    KvCachePool pool = MakePool(2, 8);
    SlotId s0 = -1, s1 = -1, s2 = -1;
    ASSERT_EQ(pool.AllocKvSlot(&s0).code, Status::SUCCESS);
    ASSERT_EQ(pool.Advance(s0, 5).code, Status::SUCCESS) << "s0 先跑远（制造记账差）";
    ASSERT_EQ(pool.AllocKvSlot(&s1).code, Status::SUCCESS);
    // 池满 → 驱逐 s0（最早分配，alloc_seq 最小）
    ASSERT_EQ(pool.AllocKvSlot(&s2).code, Status::SUCCESS);
    EXPECT_EQ(s2, 0) << "FIFO 驱逐复用槽 0";
    EXPECT_EQ(pool.TotalWritten(s2), 0u) << "驱逐重置：归零起记账";
    EXPECT_EQ(pool.NextPosition(s2), 0u);
    EXPECT_EQ(pool.TotalWritten(s1), 0u) << "未逐槽记账不动";
}

// max_batch_size<=0 塌零池：无槽可分配 → FAILURE（kv_cache_pool.cpp:260）。
TEST(KvPoolAllocEvictTest, ZeroSlotPoolAllocFails)
{
    KvCachePool pool = MakePool(0, 8);
    SlotId slot = 1;
    EXPECT_EQ(pool.AllocKvSlot(&slot).code, Status::FAILURE);
    EXPECT_EQ(slot, 1) << "失败不写出参";
}

// Free 越界幂等防御：负/超界 id 返回 SUCCESS 不崩（kv_cache_pool.cpp:265）。
TEST(KvPoolAllocEvictTest, FreeOutOfRangeIdempotent)
{
    KvCachePool pool = MakePool(2, 8);
    EXPECT_EQ(pool.FreeKvSlot(-1).code, Status::SUCCESS);
    EXPECT_EQ(pool.FreeKvSlot(99).code, Status::SUCCESS);
    SlotId slot = -1;
    EXPECT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS) << "防御不破坏池";
}

// ═══════════════════════════════════════════════════════════════════════════
// 池只读观测面（Snapshot / TensorCount / NumKvLayers / KvLayerIds / K/V）
// ═══════════════════════════════════════════════════════════════════════════

// Snapshot dfx 快照：槽位状态/记账回读一致；out=nullptr 防御（:359-364）。
TEST(KvPoolInspectTest, SnapshotReflectsLedgerAndNullGuard)
{
    KvCachePool pool = MakePool(2, 8);
    SlotId slot = -1;
    ASSERT_EQ(pool.AllocKvSlot(&slot).code, Status::SUCCESS);
    ASSERT_EQ(pool.Advance(slot, 3).code, Status::SUCCESS);
    std::vector<SlotInfo> snap;
    ASSERT_EQ(pool.Snapshot(&snap).code, Status::SUCCESS);
    ASSERT_EQ(snap.size(), 2u);
    EXPECT_EQ(snap[0].id, 0);
    EXPECT_TRUE(snap[0].active);
    EXPECT_EQ(snap[0].total_written, 3u);
    EXPECT_EQ(snap[0].write_pos, 3u);
    EXPECT_FALSE(snap[1].active) << "未分配槽 inactive";
    EXPECT_EQ(pool.Snapshot(nullptr).code, Status::INVALID_ARGUMENT);
}

// 构造期观测：稠密池 TensorCount=0（T1 惰性建）、NumKvLayers=层数、
// 层表恒等；Ensure 前的 KTensor/VTensor 空 + 越界层空（:366-398）。
TEST(KvPoolInspectTest, DensePoolConstructorInspectAndTensorGuards)
{
    KvCachePool pool = MakePool(2, 8); // 2 层稠密
    EXPECT_EQ(pool.TensorCount(), 0u) << "构造期不建 tensor（T1）";
    EXPECT_EQ(pool.NumKvLayers(), 2);
    EXPECT_EQ(pool.KvLayerIds(), (std::vector<int>{0, 1}));
    EXPECT_EQ(pool.KTensor(0), nullptr) << "Ensure 前 K 空";
    EXPECT_EQ(pool.VTensor(1), nullptr) << "Ensure 前 V 空";
    EXPECT_EQ(pool.KTensor(-1), nullptr) << "越界层防御";
    EXPECT_EQ(pool.KTensor(99), nullptr);
    EXPECT_EQ(pool.max_batch_size(), 2);
    EXPECT_EQ(pool.kv_cache_max_len(), 8);
    EXPECT_EQ(pool.num_hidden_layers(), 2);
}

// 稀疏池构造（Qwen3.5 稀疏化，support_qwen3.5.md §4.3）：linear_attention
// 层不建池 → NumKvLayers=2、层表跳层 {0,2}；长度不符 WARN 回退稠密 → 3。
TEST(KvPoolInspectTest, SparseLayerTypesLinearSkipped)
{
    KvCachePool sparse(2, 8, 3, 4, 8, {"full_attention", "linear_attention", "sliding_attention"});
    EXPECT_EQ(sparse.NumKvLayers(), 2) << "linear 层不建池";
    EXPECT_EQ(sparse.KvLayerIds(), (std::vector<int>{0, 2}));
    EXPECT_EQ(sparse.TensorCount(), 0u);

    // 长度不符（2 != 3）→ WARN 回退稠密（防御，不静默稀疏）
    KvCachePool fallback(2, 8, 3, 4, 8, {"full_attention", "sliding_attention"});
    EXPECT_EQ(fallback.NumKvLayers(), 3) << "长度不符回退稠密";
    EXPECT_EQ(fallback.KvLayerIds(), (std::vector<int>{0, 1, 2}));
}

// ═══════════════════════════════════════════════════════════════════════════
// HistoryPromptManager 锚定段/开关访问器（design §3.3 / design §3.5）
// ═══════════════════════════════════════════════════════════════════════════

// 锚定段记账（design §3.3）：Set/InitTokenLen 直通；管理面 Enabled 开关
// 缺省开（R11 默认启用）、SetEnabled 双向（R3 门控数据面已由
// DisabledReturnsZeroWithoutSideEffect 覆盖，此处覆盖管理访问器本体）。
TEST(HistoryAnchorTest, InitTokenLenAndEnabledAccessors)
{
    HistoryPromptManager m;
    EXPECT_TRUE(m.Enabled()) << "R11：默认启用";
    EXPECT_EQ(m.InitTokenLen(), 0u) << "锚定段缺省 0";
    m.SetInitTokenLen(7);
    EXPECT_EQ(m.InitTokenLen(), 7u);
    m.SetEnabled(false);
    EXPECT_FALSE(m.Enabled());
    m.SetEnabled(true);
    EXPECT_TRUE(m.Enabled());

    // 工位绑定缺省（2026-10-21 修复回归锚点）：构造时未绑定任何工位 → 0
    EXPECT_EQ(m.KvSlotId(), 0) << "kv_slot_id_ 缺省 0";
}

// ═══════════════════════════════════════════════════════════════════════════
// 工位绑定访问器 SetKvSlotId/KvSlotId（2026-10-21 命中工位直用修复：
// history_prompt_manager.h "工位绑定" 区——pipeline RunLoop 记账点回写
// req->kv_slot_actual、下轮命中 GenerateAsync 按此值 ResumeSlot+预绑定，
// 详见 pipeline_executor.cpp:1517-1522 / :1338）
// ═══════════════════════════════════════════════════════════════════════════

// TC_901：set/get 往返——pipeline reliant on "写后读同值" 契约；
// 且 Clear（miss/失败回退/DestroyContext 路径）只清台账 token 不动工位
// 绑定记录（kv_slot_id_ 与 tokens_ 独立字段，scheduler.cpp 回退链上
// ResetLedgerOnSlot/FreeKvSlot 均不触本字段）。
TEST(HistoryAnchorTest, KvSlotIdSetGetRoundTripAndClearKeepsBinding_TC901)
{
    HistoryPromptManager m;
    // 非默认值写入-读回（0 是缺省值，用非 0 排除"未写读缺省"伪绿）
    m.SetKvSlotId(3);
    EXPECT_EQ(m.KvSlotId(), 3) << "set/get 往返";
    // 覆写最近一次写值（工位可能被换绑）
    m.SetKvSlotId(1);
    EXPECT_EQ(m.KvSlotId(), 1) << "重复 set 覆写";

    // Clear 只动 tokens_（台账），不动工位绑定
    m.AppendDecodedToken(11);
    m.AppendDecodedToken(12);
    ASSERT_EQ(m.Size(), 2u);
    m.Clear();
    EXPECT_EQ(m.Size(), 0u) << "台账已清";
    EXPECT_EQ(m.KvSlotId(), 1) << "Clear 不清工位绑定记录";
}

// 边界值：负工位（防御面——宿主 pipeline 只回写 ≥0 值（scheduler.cpp
// 回写点门禁 req->kv_slot_actual >= 0），但栈台账须能承载任意 int 不
// 丢失/钳制，供调试观测）。
TEST(HistoryAnchorTest, KvSlotIdCarriesArbitraryIntUnclamped_TC902)
{
    HistoryPromptManager m;
    m.SetKvSlotId(-1);
    EXPECT_EQ(m.KvSlotId(), -1) << "负值透传不钳制";
    m.SetKvSlotId(0);
    EXPECT_EQ(m.KvSlotId(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Preprocessor::PreparePrefill / PrepareDecodeStep（tokenizer 直连 + fast path）
// ═══════════════════════════════════════════════════════════════════════════

/// 测试桩 tokenizer：encode 按空格切词映射静态 id；encode 失败可注入。
class StubTokenizer final : public TokenizerBase {
public:
    bool fail_encode = false;
    Status encode(std::string_view, const EncodeOptions&, std::vector<int>* out_ids) override
    {
        if (fail_encode)
            return Status::FAILURE;
        *out_ids = {11, 12, 13, 14, 15}; // 词表无须真实——id 序即断言面
        return Status::SUCCESS;
    }
    Status decode(const std::vector<int>&, const DecodeOptions&, std::string*) override { return Status::SUCCESS; }
    size_t vocab_size() const override { return 99; }
    std::optional<int> special_token_id(std::string_view) const override { return std::nullopt; }
};

// 正常链路：encode → out_ids 回传全量 + 增量切块 [reusable, len)。
// ids={11..15}、reusable=2、chunk=2 → out_ids 全量 5 元、块 [{13,14},{15}]。
TEST(PreparePrefillTest, EncodesAndSlicesIncrementally)
{
    StubTokenizer tok;
    Preprocessor prep(&tok, nullptr, /*default_chunk_size=*/2);
    Prompt prompt;
    prompt.text = "hello prompt cache";
    std::vector<std::vector<TokenId>> chunks;
    std::vector<TokenId> out_ids;
    ASSERT_EQ(prep.PreparePrefill(prompt, &chunks, /*reusable=*/2, &out_ids).code, Status::SUCCESS);
    EXPECT_EQ(out_ids, (std::vector<TokenId>{11, 12, 13, 14, 15})) << "out_ids 回传 encode 全量";
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0], (std::vector<TokenId>{13, 14})) << "首块 [2,4)";
    EXPECT_EQ(chunks[1], (std::vector<TokenId>{15})) << "尾块 [4,5)";
}

// tokenizer 为空：INVALID_STATE 降级（桩时代链路口径，preprocessor.cpp:25-27）。
TEST(PreparePrefillTest, NullTokenizerRejectedInvalidState)
{
    Preprocessor prep(nullptr, nullptr, 4);
    Prompt prompt;
    std::vector<std::vector<TokenId>> chunks;
    EXPECT_EQ(prep.PreparePrefill(prompt, &chunks).code, Status::INVALID_STATE);
    EXPECT_TRUE(chunks.empty());
}

// encode 失败上抛：时段化失败不切块（preprocessor.cpp:34-37）。
TEST(PreparePrefillTest, EncodeFailurePropagates)
{
    StubTokenizer tok;
    tok.fail_encode = true;
    Preprocessor prep(&tok, nullptr, 4);
    Prompt prompt;
    std::vector<std::vector<TokenId>> chunks{{9}}; // 预污染验证 clear
    EXPECT_EQ(prep.PreparePrefill(prompt, &chunks).code, Status::FAILURE);
    EXPECT_TRUE(chunks.empty()) << "失败路径先 clear 不半写";
}

// out_chunks=nullptr / reusable>len：防御 INVALID_ARGUMENT（:21,55-59）。
TEST(PreparePrefillTest, NullOutAndOversizeReuseRejected)
{
    StubTokenizer tok;
    Preprocessor prep(&tok, nullptr, 4);
    Prompt prompt;
    std::vector<TokenId> out_ids;
    EXPECT_EQ(prep.PreparePrefill(prompt, nullptr, 0, &out_ids).code, Status::INVALID_ARGUMENT);
    std::vector<std::vector<TokenId>> chunks;
    EXPECT_EQ(prep.PreparePrefill(prompt, &chunks, /*reusable=*/99).code, Status::INVALID_ARGUMENT)
        << "reusable > ids 长";
    EXPECT_TRUE(chunks.empty());
}

// PrepareDecodeStep fast path：单 token 透传；nullptr 防御（:98-101）。
TEST(PreparePrefillTest, DecodeStepSingleTokenAndNullGuard)
{
    Preprocessor prep(nullptr, nullptr, 4);
    std::vector<TokenId> input{1, 2}; // 预污染验证整写
    ASSERT_EQ(prep.PrepareDecodeStep(42, &input).code, Status::SUCCESS);
    EXPECT_EQ(input, (std::vector<TokenId>{42}));
    EXPECT_EQ(prep.PrepareDecodeStep(42, nullptr).code, Status::INVALID_ARGUMENT);
}

// ═══════════════════════════════════════════════════════════════════════════
// EnsureKvTensor 防御与建池路径（R1/R2/R3 把关；kv_cache_pool.cpp:101-219）
// host 无 NDK/stub so 时创建类用例跳过（守卫模式同 test_input_assembler.cpp
// 的 UT_NEED_BACKEND —— 池构造期不建 tensor，触及 CreateKv 才需 backend）
// ═══════════════════════════════════════════════════════════════════════════

/// host 可用性探测：CreateTensor 走真后端（NDKProxy 搜 stub .so），无 .so → 失败。
bool HostBackendAvailable()
{
    TensorDesc d;
    d.dims = {1};
    d.dataType = DataType::INT32;
    std::shared_ptr<ITensorBuffer> t;
    return TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &t).ok();
}

#define PC_UT_NEED_BACKEND()                                            \
    do {                                                                \
        if (!HostBackendAvailable())                                    \
            GTEST_SKIP() << "host 无 NDK/stub so：CreateTensor 不可用"; \
    } while (0)

/// HF 序图 KV 工位 desc 便捷构造（池参 MakePool 口径 [batch=2, heads=4, kv_len, dim=8]）
TensorDesc MakeGraphKvDesc(const std::vector<int32_t>& dims)
{
    TensorDesc d;
    d.dims = dims;
    d.dataType = DataType::FLOAT16;
    d.format = Format::ND;
    return d;
}

// desc 防御（无 backend 依赖）：layer<0 / dims 空 / RESERVED dtype / 逐维≤0
// → INVALID_ARGUMENT 先于层表与具建（:103-108）。
TEST(PromptCacheEnsureTest, BadDescRejectedBeforeLayerLookup)
{
    KvCachePool pool = MakePool(2, 16);
    EXPECT_EQ(pool.EnsureKvTensor(-1, false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::INVALID_ARGUMENT)
        << "layer<0";
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({})).code, Status::INVALID_ARGUMENT) << "dims 空";
    TensorDesc reserved = MakeGraphKvDesc({2, 4, 16, 8});
    reserved.dataType = DataType::RESERVED;
    EXPECT_EQ(pool.EnsureKvTensor(0, false, reserved).code, Status::INVALID_ARGUMENT) << "RESERVED dtype";
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 0, 16, 8})).code, Status::INVALID_ARGUMENT) << "零维";
}

// 非 KV 池层（linear 层稀疏池）/ 越界层 → FAILURE 资源防御（:112-117）。
TEST(PromptCacheEnsureTest, NonPoolLayerReturnsFailure)
{
    KvCachePool sparse(2, 16, 3, 4, 8, {"full_attention", "linear_attention", "full_attention"});
    EXPECT_EQ(sparse.EnsureKvTensor(1, false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::FAILURE)
        << "linear 层不在池内";
    KvCachePool pool = MakePool(2, 16);
    EXPECT_EQ(pool.EnsureKvTensor(99, false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::FAILURE) << "越界层";
}

// kv_len/batch 维失配致命：INVALID_ARGUMENT 终止 Load（2026-10-15 裁定，
// RingPromptCache 化后容量权威 = meta kv_cache_max_len，:142-160）。
TEST(PromptCacheEnsureTest, KvLenAndBatchMismatchFatal)
{
    KvCachePool pool = MakePool(2, 16); // expect HF [batch=2, heads=4, len=16, dim=8]
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 4, 8, 8})).code, Status::INVALID_ARGUMENT)
        << "kv_len 8<16 失配";
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({1, 4, 16, 8})).code, Status::INVALID_ARGUMENT)
        << "batch 1<2 失配";
    EXPECT_EQ(pool.TensorCount(), 0u) << "失配不落位";
}

// 非 4D desc：跳过 HF 期望比对照常直建（每池告警去重一次；:125-137）。
TEST(PromptCacheEnsureTest, Non4dDescSkipsCompareAndCreates)
{
    PC_UT_NEED_BACKEND();
    KvCachePool pool = MakePool(2, 16);
    ASSERT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 4, 16})).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 1u) << "3D desc 照建";
    auto t = pool.KTensor(0);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->GetTensorDesc().tensorName, "kv_cache_k_0") << "命名约定落位";
    EXPECT_EQ(t->GetTensorDesc().dims, (std::vector<int32_t>{2, 4, 16})) << "dims 直通图（未按 HF 期望补维）";
    EXPECT_EQ(t->GetTensorDesc().dataType, DataType::FLOAT16);
    EXPECT_EQ(t->GetSize(), 2u * 4 * 16 * 2) << "2B×2×4×16（按 3D desc 派生）";
}

// gemma4 滑窗层末维期望口径（sliding_kv_seq_dim_rebind D2；2026-10-27 改写——
// SP-5②：旧版按"期望末维 = sliding_window"断言，b688221 废口径（D8）下该
// 期望因末维失配仅 WARN 恒过、断言在错误前提上通过）。现行口径：
//   · sw_hd=0（旧契，本例尾参缺省）：sliding 层期望末维 = head_dim（与
//     full 层同形）——末维失配即图异形，仍只是 WARN 放行（om 权威口径不变）；
//   · 期望生效的直接观测：合法旧契期望 {2,4,16,8} 命中 → 同 desc 重建复验
//     逐值全等（R2）——期望面测试见 NewContractLedgerPerLayerExpectDims。
TEST(PromptCacheEnsureTest, SlidingLayerEndDimExpectationWarnsOnly)
{
    PC_UT_NEED_BACKEND();
    // 旧契 typed 池：sliding 层期望 = full 同形 {2,4,16,8}（sw_hd 缺省 0）
    KvCachePool pool(2, 16, 2, 4, 8, {"full_attention", "sliding_attention"},
                     /*sliding_window=*/4);
    EXPECT_EQ(pool.NumKvLayers(), 2) << "两建池层";
    // layer 1（sliding）期望末维 = head_dim = 8 → 命中直建（无告警面）
    ASSERT_EQ(pool.EnsureKvTensor(1, false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::SUCCESS)
        << "旧契期望末维=head_dim，图命中";
    EXPECT_EQ(pool.TensorCount(), 1u);
    // 末维 4（b688221 过渡图口径形）≠ 期望 8 → WARN 放行仍 SUCCESS——
    // 断言面锁"告警不终止"（om 权威，:164-178 裁定不变），且图 dims 原样
    ASSERT_EQ(pool.EnsureKvTensor(1, true, MakeGraphKvDesc({2, 4, 16, 4})).code, Status::SUCCESS)
        << "末维失配仅告警（:164-178）";
    EXPECT_EQ(pool.VTensor(1)->GetTensorDesc().dims, (std::vector<int32_t>{2, 4, 16, 4}));
    EXPECT_EQ(pool.TensorCount(), 2u) << "告警不拦建";
}

// 直建 + 二次 Ensure 复验一致 SUCCESS / desc 变化拒绝（R1/R2；:182-218）。
// 创建即清零口径（2026-10-15 裁定）：全 buffer 字节为 0。
TEST(PromptCacheEnsureTest, EnsureCreateClearThenRevalidate)
{
    PC_UT_NEED_BACKEND();
    KvCachePool pool = MakePool(2, 16);
    ASSERT_EQ(pool.EnsureKvTensor(0, /*is_value=*/false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 1u);
    auto k = pool.KTensor(0);
    ASSERT_NE(k, nullptr);
    EXPECT_EQ(k->GetTensorDesc().tensorName, "kv_cache_k_0");
    EXPECT_EQ(k->GetSize(), 2u * 4 * 16 * 8 * 2) << "2B×2×4×16×8";
    // 创建即清零
    const auto* bytes = static_cast<const uint8_t*>(k->GetData());
    std::size_t non_zero = 0;
    for (std::size_t i = 0; i < k->GetSize(); ++i)
        non_zero += bytes[i] != 0;
    EXPECT_EQ(non_zero, 0u) << "创建即全 buffer 清零";

    // desc 一致 → 次图复用 SUCCESS 且不重建
    ASSERT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::SUCCESS);
    EXPECT_EQ(pool.TensorCount(), 1u) << "复用不重建";

    // dtype 变化 → 复验拒绝
    TensorDesc fp32 = MakeGraphKvDesc({2, 4, 16, 8});
    fp32.dataType = DataType::FLOAT32;
    EXPECT_EQ(pool.EnsureKvTensor(0, false, fp32).code, Status::INVALID_ARGUMENT);
    // dims 变化 → 复验拒绝
    EXPECT_EQ(pool.EnsureKvTensor(0, false, MakeGraphKvDesc({2, 4, 16, 7})).code, Status::INVALID_ARGUMENT)
        << "末维变化（heads/末维失配告警不拦，但槽已建走 R2 复验）";
    EXPECT_EQ(pool.TensorCount(), 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 新契 typed 池期望——Load 期 desc 账本路径（EnsureKvDesc/CheckKvDescCommon；
// 审查 SP-2/SP-3 补 §5 U2/U6 缺口；tensor 建形路径的镜像面在
// ut_engine_executor TC_613/TC_614，此处锁账本面）。
// 池小形状：b=2 / kv_max=16 / heads=4 / head_dim=8；W=4 / P=8 ⇒ C=12、
// sw_hd=64 ⇒ 新契 sliding 层期望 {2,4,12,64}、full 层期望 {2,4,16,8}；
// 旧契（尾参缺省）sliding 层期望与 full 层同形 {2,4,16,8}。
// ═══════════════════════════════════════════════════════════════════════════

// 新契双层池：sliding 层账本记 C=W+P×sw_hd 形 desc；旧形图 desc（kv_max
// ×head_dim）在 sliding 层 fatal（kv_len 维失配——D7 fail-loud），在 full
// 层照记（full 不受新契影响）。记账面 = R3 期望比对，fatal 即 Load 终止点。
TEST(PromptCacheEnsureTest, NewContractLedgerPerLayerExpectDims)
{
    KvCachePool pool(2, 16, 2, 4, 8, {"sliding_attention", "full_attention"},
                     /*sliding_window=*/4, /*init_token_len=*/0,
                     /*prefill_max=*/8, /*sliding_window_head_dim=*/64);
    // layer 0（sliding，新契）：期望 {2,4,12,64} → 命中记账 + 账本回读原样
    ASSERT_EQ(pool.EnsureKvDesc(0, false, MakeGraphKvDesc({2, 4, 12, 64})).code, Status::SUCCESS);
    auto k0 = pool.KvDesc(0, false);
    ASSERT_TRUE(k0.has_value());
    EXPECT_EQ(k0->dims, (std::vector<int32_t>{2, 4, 12, 64}));
    // layer 0（sliding）配旧形图 {2,4,16,8} → kv_len 16≠12 fatal（不记账）
    EXPECT_EQ(pool.EnsureKvDesc(0, true, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::INVALID_ARGUMENT)
        << "新契×旧形：seq 轴失配致命";
    EXPECT_FALSE(pool.KvDesc(0, true).has_value()) << "fatal 不落账本";
    // layer 1（full）：期望 {2,4,16,8} 不受新契影响 → 旧形照记
    ASSERT_EQ(pool.EnsureKvDesc(1, false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::SUCCESS);
    ASSERT_TRUE(pool.KvDesc(1, false).has_value());
    EXPECT_EQ(pool.KvDesc(1, false)->dims, (std::vector<int32_t>{2, 4, 16, 8}));
    // 账本完整性：V0 记账被拒 ⇒ 半账（DescLedgerComplete=false）
    EXPECT_FALSE(pool.DescLedgerComplete()) << "V0 未记 → 账本不完整";
}

// 兼容矩阵反向 fatal（D7/U6）：旧契池（尾参缺省——sliding 层期望与 full
// 同形 kv_max）配新形图 desc {2,4,12,64} → kv_len 12≠16 致命拒记；末维
// sw_hd=64 形态同样被 seq 轴失配先拦（致命维优先于告警维）。
TEST(PromptCacheEnsureTest, OldContractPoolRejectsNewFormGraphDesc)
{
    KvCachePool pool(2, 16, 2, 4, 8, {"sliding_attention", "full_attention"},
                     /*sliding_window=*/4); // init/prefill_max/sw_hd 全缺省（旧契）
    EXPECT_EQ(pool.EnsureKvDesc(0, false, MakeGraphKvDesc({2, 4, 12, 64})).code, Status::INVALID_ARGUMENT)
        << "旧契池×新形图：kv_len 12≠16";
    EXPECT_EQ(pool.EnsureKvDesc(0, false, MakeGraphKvDesc({2, 4, 12, 8})).code, Status::INVALID_ARGUMENT)
        << "新形 seq 轴 + 旧末维同样 fatal";
    // full 层期望不变（{2,4,16,8} 命中）
    ASSERT_EQ(pool.EnsureKvDesc(1, false, MakeGraphKvDesc({2, 4, 16, 8})).code, Status::SUCCESS);
    EXPECT_FALSE(pool.DescLedgerComplete()) << "半账（K0/V0 未记）";
}

} // namespace
} // namespace cann_local_lm::engine
