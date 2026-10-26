// cann_lm/engine/core/sample/test_sampler.cc — Sampler C++ 验收（port 自 test_sampler.py）
//
// 期望值用本文件内的 softmax/argmax 直算（独立于被测实现，避免共享 bug）。
// 编译运行（从 sample/ 目录）：
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers sampler.cc test/sampler_ut.cc -o
//   test/sampler_ut && ./test/sampler_ut
// 或一键：bash scripts/build_sample.sh
#include "../sampler.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

using namespace cann_lm::sampler;

static int g_pass = 0, g_fail = 0;
static void check(const char* name, bool cond, const char* detail = "")
{
    if (cond) {
        ++g_pass;
        printf("  [PASS] %s\n", name);
    } else {
        ++g_fail;
        printf("  [FAIL] %s  %s\n", name, detail);
    }
}

// ---- 测试本地 oracle（独立于 sampler.cc）----
static std::vector<float> T_softmax(const std::vector<float>& v)
{
    float m = -INFINITY;
    for (float x : v)
        m = std::max(m, x);
    std::vector<float> e(v.size());
    float s = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        e[i] = std::exp(v[i] - m);
        s += e[i];
    }
    for (float& x : e)
        x /= s;
    return e;
}
static bool approx(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}
static bool no_bad(const std::vector<float>& v)
{
    for (float x : v)
        if (std::isnan(x) || std::isinf(x))
            return false;
    return true;
}
static double sum(const std::vector<float>& v)
{
    double s = 0;
    for (float x : v)
        s += x;
    return s;
}
static int T_argmax_local(const std::vector<float>& v)
{
    int b = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[b])
            b = (int)i;
    return b;
}

static SamplerConfig Cfg() { return {}; } // 默认：do_sample=false, temp=1.0, 无 k/p/rep

// 测试便利：旧式 4 参打包成 SamplerInputs 再调 Sample（logits/gen 按值取，move 进 SamplerInputs 走零拷贝）。
// 真实接口是 Sampler::Sample(SamplerInputs, cfg)；S 只是把 4 参打包，便于测试书写（见 N 组直接构造）。
static SamplerOutput S(std::vector<float> logits, std::vector<int32_t> gen, SamplerConfig cfg,
                       std::optional<PrecomputedTopK> pre = std::nullopt)
{
    return Sampler::Sample(
        SamplerInputs{.logits = std::move(logits), .generated_ids = std::move(gen), .precomputed_topk = std::move(pre)},
        cfg);
}

// ───────── A. 形状 & 归一化 ─────────
static void test_shape_norm()
{
    auto out = S({1, 2, 3, 4}, {}, Cfg());
    check("A1 probs size==4", out.probs.size() == 4);
    check("A2 probs sum≈1", std::fabs(sum(out.probs) - 1.0f) < 1e-5f);
    bool nn = true;
    for (float x : out.probs)
        if (x < 0)
            nn = false;
    check("A3 probs 非负", nn);
    check("A4 id 合法", out.next_token_id >= 0 && out.next_token_id < 4);
}
// ───────── B. 贪心=argmax ─────────
static void test_greedy()
{
    std::vector<float> L{1, 5, 2};
    auto out = S(L, {}, Cfg());
    check("B1 greedy==argmax", out.next_token_id == 1);
    check("B2 probs==softmax", approx(out.probs, T_softmax(L)));
}
// ───────── C. temperature ─────────
static void test_temperature()
{
    std::vector<float> L{1, 5, 2};
    auto o5 = S(L, {}, {.temperature = 0.5f});
    check("C1 温度不改 argmax", o5.next_token_id == 1);
    std::vector<float> Ld;
    for (float x : L)
        Ld.push_back(x / 0.5f);
    check("C2 probs==softmax(/t)", approx(o5.probs, T_softmax(Ld)));
    auto o1 = S(L, {}, {.temperature = 1.0f});
    check("C3 t=1==softmax", approx(o1.probs, T_softmax(L)));
    check("C4 t<1 更尖锐", o5.probs[1] > o1.probs[1]);
    auto o0 = S(L, {}, {.temperature = 0.0f});
    check("C5 t=0 钳制无 nan 且 argmax", no_bad(o0.probs) && o0.next_token_id == 1);
    auto on = S(L, {}, {.temperature = -1.0f});
    check("C6 t<0 钳制无 nan 且 argmax", no_bad(on.probs) && on.next_token_id == 1);
    auto oe = S(L, {}, {.temperature = 1e-8f});
    check("C7 t→0 argmax 概率≈1", std::fabs(oe.probs[1] - 1.0f) < 1e-3f);
}
// ───────── D. top_k ─────────
static void test_top_k()
{
    std::vector<float> L{3, 1, 2};
    auto o2 = S(L, {}, {.top_k = 2});
    check("D1 top_k=2 屏蔽非 top-2", std::fabs(o2.probs[1]) < 1e-8f);
    check("D2 top_k=2 仍 argmax", o2.next_token_id == 0);
    check("D3 probs==softmax(masked)", approx(o2.probs, T_softmax({3.0f, -INFINITY, 2.0f})));
    auto o1 = S(L, {}, {.top_k = 1});
    check("D4 top_k=1 argmax 概率≈1", std::fabs(o1.probs[0] - 1.0f) < 1e-5f);
    check("D5 top_k=1 next==argmax", o1.next_token_id == 0);
    check("D6 top_k=None no-op", approx(S(L, {}, Cfg()).probs, T_softmax(L)));
    check("D7 top_k>=vocab no-op", approx(S(L, {}, {.top_k = 100}).probs, T_softmax(L)));
}
// ───────── E. top_p ─────────
static void test_top_p()
{
    std::vector<float> L{1, 2, 3};
    auto ot = S(L, {}, {.top_p = 0.01f});
    check("E1 top_p 极小 argmax 概率≈1", std::fabs(ot.probs[2] - 1.0f) < 1e-5f);
    check("E2 top_p 极小 next==argmax", ot.next_token_id == 2);
    check("E3 top_p 极小 其余≈0", ot.probs[0] < 1e-8f && ot.probs[1] < 1e-8f);
    auto of = S(L, {}, {.top_p = 1.0f});
    check("E4 top_p=1.0 argmax 保留+无 nan", of.next_token_id == 2 && no_bad(of.probs));
    check("E5 top_p=None no-op", approx(S(L, {}, Cfg()).probs, T_softmax(L)));
}
// ───────── F. repetition_penalty ─────────
static void test_rep_noop()
{
    std::vector<float> L{2.0f, 1.9f};
    std::vector<int32_t> g{0};
    auto op = S(L, g, {.repetition_penalty = 1.0f});
    auto on = S(L, g, {.repetition_penalty = std::nullopt});
    auto oe = S(L, {}, {.repetition_penalty = 2.0f});
    check("F1 rep=1.0==无惩罚", op.next_token_id == on.next_token_id && approx(op.probs, T_softmax(L)));
    check("F2 空序列==无惩罚", oe.next_token_id == on.next_token_id && approx(oe.probs, T_softmax(L)));
    check("F3 argmax 未翻转(rep 关闭)", on.next_token_id == 0);
}
static void test_rep_flip()
{
    auto out = S({2.0f, 1.9f}, {0}, {.repetition_penalty = 2.0f});
    // token0 2.0/2=1.0; token1 1.9 不变 -> [1.0,1.9] -> argmax 翻转 0->1
    check("F4 rep 翻转 argmax 0->1", out.next_token_id == 1);
    check("F5 rep probs==softmax(惩罚后)", approx(out.probs, T_softmax({1.0f, 1.9f})));
}
static void test_rep_negative()
{
    auto out = S({-2.0f, -1.0f}, {0}, {.repetition_penalty = 2.0f});
    // token0 -2*2=-4.0; token1 -1.0 不变 -> [-4,-1]
    check("F6 负 logit rep probs==softmax", approx(out.probs, T_softmax({-4.0f, -1.0f})));
    auto base = T_softmax({-2.0f, -1.0f});
    check("F7 负 logit rep 降低概率", out.probs[0] < base[0]);
}
static void test_rep_boost()
{
    auto out = S({2.0f, 1.9f}, {0}, {.repetition_penalty = 0.5f});
    // token0 2.0/0.5=4.0(升权); token1 1.9 -> [4.0,1.9]
    check("F8 rep<1 probs==softmax(升权后)", approx(out.probs, T_softmax({4.0f, 1.9f})));
    auto base = T_softmax({2.0f, 1.9f});
    check("F9 rep<1 概率升高", out.probs[0] > base[0]);
    check("F10 rep<1 argmax 仍 0", out.next_token_id == 0);
    auto on = S({-2.0f, -1.0f}, {0}, {.repetition_penalty = 0.5f});
    // token0 -2*0.5=-1.0(升权,更接近0); token1 -1.0 -> [-1,-1]
    check("F11 rep<1 负logit probs==softmax", approx(on.probs, T_softmax({-1.0f, -1.0f})));
    auto bn = T_softmax({-2.0f, -1.0f});
    check("F12 rep<1 负logit 概率升高", on.probs[0] > bn[0]);
}
static void test_rep_order()
{
    // rep 必须在 top_k 前：rep 改变 top-k 成员
    auto out = S({3.0f, 2.5f, 2.0f}, {0}, {.top_k = 2, .repetition_penalty = 2.0f});
    // ① token0 3/2=1.5 -> [1.5,2.5,2.0]; ③ top_k=2 留 {2.5,2.0}(idx1,2), idx0->-inf
    check("F13 rep 在 top_k 前: probs==softmax", approx(out.probs, T_softmax({-INFINITY, 2.5f, 2.0f})));
    check("F14 rep 在 top_k 前: next==1", out.next_token_id == 1);
}
static void test_rep_plus_temp()
{
    auto out = S({2.0f, 1.9f}, {0}, {.temperature = 0.5f, .repetition_penalty = 2.0f});
    // ① token0 2/2=1.0, token1 1.9 -> [1.0,1.9]; ② /0.5 -> [2.0, 3.8]
    check("F15 rep+temp probs==softmax", approx(out.probs, T_softmax({2.0f, 3.8f})));
    check("F16 rep+temp next==1", out.next_token_id == 1);
}
static void test_rep_dup_gen()
{
    std::vector<float> L{2.0f, 1.9f};
    auto single = S(L, {0}, {.repetition_penalty = 2.0f});
    auto dup = S(L, {0, 0, 0}, {.repetition_penalty = 2.0f});
    check("F17 重复 gen 等价单次", dup.next_token_id == single.next_token_id && approx(dup.probs, single.probs));
    auto multi = S({3.0f, 2.0f, 1.0f}, {0, 1}, {.repetition_penalty = 2.0f});
    // token0 3/2=1.5, token1 2/2=1.0, token2 1.0(不在gen) -> [1.5,1.0,1.0]
    check("F18 多 gen 各降权一次 probs", approx(multi.probs, T_softmax({1.5f, 1.0f, 1.0f})));
    check("F19 多 gen next==0", multi.next_token_id == 0);
}
// ───────── G. 可复现 ─────────
static void test_determinism()
{
    std::vector<float> L{1, 2, 3, 4};
    SamplerConfig cfg{.do_sample = true, .seed = 42};
    auto o1 = S(L, {}, cfg);
    auto o2 = S(L, {}, cfg);
    check("G1 同 seed 同 token", o1.next_token_id == o2.next_token_id);
    check("G2 合法 id", o1.next_token_id >= 0 && o1.next_token_id < 4);
    std::vector<float> uni{0.0f, 0.1f, 0.2f, 0.3f};
    std::set<int32_t> res;
    for (uint64_t s = 0; s < 8; ++s)
        res.insert(S(uni, {}, {.do_sample = true, .seed = s}).next_token_id);
    check("G3 不同 seed 结果可不同", res.size() >= 2);
    auto o = S(L, {}, {.do_sample = true}); // seed=None
    check("G4 seed=None 可跑", o.next_token_id >= 0 && o.next_token_id < 4);
}
static void test_rep_sample_reproducible()
{
    std::vector<float> L{1.0f, 5.0f, 2.0f, 0.5f};
    std::vector<int32_t> g{1};
    SamplerConfig cfg{.temperature = 1.0f, .do_sample = true, .repetition_penalty = 1.5f, .seed = 7};
    auto o1 = S(L, g, cfg);
    auto o2 = S(L, g, cfg);
    check("G5 rep+采样 同 seed 可复现", o1.next_token_id == o2.next_token_id);
    check("G6 rep+采样 合法 id", o1.next_token_id >= 0 && o1.next_token_id < 4);
    auto on = S(L, g, {.temperature = 1.0f, .do_sample = true, .seed = 7});
    check("G7 rep 降低已生成 token probs", o1.probs[1] < on.probs[1]);
}
// ───────── H. 纯函数单测 ─────────
static void test_pure_temperature()
{
    std::vector<float> x{2, 1, 0};
    check("H1 t=1 no-op", approx(Sampler::ApplyTemperature(x, 1.0f), x));
    std::vector<float> r;
    for (float v : x)
        r.push_back(v / 0.5f);
    check("H2 t=0.5 翻倍", approx(Sampler::ApplyTemperature(x, 0.5f), r));
    check("H3 t<=0 无 inf/nan", no_bad(Sampler::ApplyTemperature(x, 0.0f)));
}
static void test_pure_top_k()
{
    std::vector<float> x{3, 1, 2};
    auto r = Sampler::ApplyTopK(x, 2);
    check("H4 top_k 屏蔽最低项", r[1] == -INFINITY && r[0] == 3 && r[2] == 2);
    check("H5 top_k None no-op", approx(Sampler::ApplyTopK(x, std::nullopt), x));
    check("H6 top_k>=vocab no-op", approx(Sampler::ApplyTopK(x, 100), x));
}
static void test_pure_top_p()
{
    std::vector<float> x{1, 2, 3};
    auto r = Sampler::ApplyTopP(x, 0.01f);
    check("H7 top_p 极小仅留最大项", r[2] == 3 && r[0] == -INFINITY && r[1] == -INFINITY);
    check("H8 top_p None no-op", approx(Sampler::ApplyTopP(x, std::nullopt), x));
    auto r1 = Sampler::ApplyTopP(x, 1.0f);
    check("H9 top_p=1.0 无 nan 且 argmax 保留", no_bad(r1) && r1[2] == 3);
}
static void test_pure_normalize()
{
    std::vector<float> x{1, 2, 3};
    auto r = Sampler::Normalize(x);
    check("H10 normalize==softmax", approx(r, T_softmax(x)));
    check("H11 和≈1", std::fabs(sum(r) - 1.0f) < 1e-6f);
}
static void test_pure_pick()
{
    std::vector<float> p{0.1f, 0.0f, 0.9f};
    check("H12 pick 贪心==argmax", Sampler::Pick(p, false, std::nullopt) == 2);
    auto g1 = Sampler::Pick(p, true, 0);
    auto g2 = Sampler::Pick(p, true, 0);
    check("H13 pick seeded 可复现", g1 == g2);
    check("H14 pick 合法 id", g1 >= 0 && g1 < 3);
}
static void test_pure_rep()
{
    std::vector<float> x{2.0f, 1.9f};
    std::vector<int32_t> g{0};
    auto r = Sampler::ApplyRepetitionPenalty(x, g, 2.0f);
    check("H15 rep 正 logit /=p", std::fabs(r[0] - 1.0f) < 1e-6f && std::fabs(r[1] - 1.9f) < 1e-6f);
    auto rn = Sampler::ApplyRepetitionPenalty({-2.0f, -1.0f}, g, 2.0f);
    check("H16 rep 负 logit *=p", std::fabs(rn[0] - (-4.0f)) < 1e-6f && std::fabs(rn[1] - (-1.0f)) < 1e-6f);
    check("H17 rep None no-op", approx(Sampler::ApplyRepetitionPenalty(x, g, std::nullopt), x));
    check("H18 rep 1.0 no-op", approx(Sampler::ApplyRepetitionPenalty(x, g, 1.0f), x));
    check("H19 rep 空序列 no-op", approx(Sampler::ApplyRepetitionPenalty(x, {}, 2.0f), x));
}
// ───────── K. 组合 ─────────
static void test_composition()
{
    auto out = S({1.0f, 5.0f, 2.0f, 0.5f}, {1},
                 {.temperature = 0.8f, .do_sample = false, .top_p = 0.9f, .top_k = 3, .repetition_penalty = 1.3f});
    check("K1 组合：合法 id", out.next_token_id >= 0 && out.next_token_id < 4);
    check("K2 组合：probs 和≈1", std::fabs(sum(out.probs) - 1.0f) < 1e-5f);
    check("K3 组合：无 nan/inf", no_bad(out.probs));
}

// ───────── L. 边角（P3.1）─────────
static void test_edge_large_vocab()
{
    // 10 万词表：argmax 在中间，验不崩、和≈1、size 对、id 对。
    std::vector<float> L(100000, 0.0f);
    L[50000] = 5.0f; // 唯一 argmax
    auto out = S(L, {}, Cfg());
    check("L1 大词表 size 对", out.probs.size() == 100000);
    check("L2 大词表 next==argmax", out.next_token_id == 50000);
    check("L3 大词表 和≈1", std::fabs(sum(out.probs) - 1.0f) < 1e-3f); // 100k float32 累积误差放宽
    check("L4 大词表 无 nan/inf", no_bad(out.probs));
}
static void test_edge_all_neg_inf()
{
    // 全 -inf logits：Normalize 应退化兜底（均匀），不产 nan
    std::vector<float> L{-INFINITY, -INFINITY, -INFINITY};
    auto out = S(L, {}, Cfg());
    check("L5 全 -inf 无 nan/inf", no_bad(out.probs));
    check("L6 全 -inf 和≈1", std::fabs(sum(out.probs) - 1.0f) < 1e-5f);
    bool approx_uni = true;
    for (float x : out.probs)
        if (std::fabs(x - (1.0f / 3.0f)) > 1e-5f)
            approx_uni = false;
    check("L7 全 -inf 退化均匀", approx_uni);
}
static void test_edge_rep_many_tokens()
{
    // 多个不同 generated token 各降权一次
    auto out = S({4.0f, 3.0f, 2.0f, 1.0f}, {0, 1, 2}, {.repetition_penalty = 2.0f});
    // 4/2=2, 3/2=1.5, 2/2=1, 1 不变 -> [2, 1.5, 1, 1]
    check("L8 多 token rep probs==softmax", approx(out.probs, T_softmax({2.0f, 1.5f, 1.0f, 1.0f})));
    check("L9 多 token rep next==0", out.next_token_id == 0);
}
static void test_edge_temperature_large()
{
    // t 极大 → 近均匀；argmax 不变；max 概率 < 0.5
    auto out = S({1.0f, 5.0f, 2.0f}, {}, {.temperature = 100.0f});
    check("L10 t 极大 next==argmax", out.next_token_id == 1);
    check("L11 t 极大 无 nan/inf", no_bad(out.probs));
    float mx = 0;
    for (float x : out.probs)
        mx = std::max(mx, x);
    check("L12 t 极大 近均匀(max<0.5)", mx < 0.5f);
}
static void test_edge_top_p_property()
{
    // 多个 p：argmax token 必保留(prob>0)、和≈1、无 nan
    std::vector<float> L{1, 2, 3, 4, 5};
    bool ok = true;
    for (float p : {0.001f, 0.5f, 0.9f, 0.99f, 1.0f}) {
        auto out = S(L, {}, {.top_p = p});
        int am = T_argmax_local(L);
        if (!(out.probs[am] > 0.0f))
            ok = false;
        if (std::fabs(sum(out.probs) - 1.0f) > 1e-5f)
            ok = false;
        if (!no_bad(out.probs))
            ok = false;
    }
    check("L13 top_p 各 p: argmax 保留+和≈1+无 nan", ok);
}
static void test_edge_multinomial_stats()
{
    // 强偏分布：token0 概率≈0.9996，200 seed 下 token0 应被选中 >60%
    std::vector<float> L{10.0f, 1.0f, 1.0f, 1.0f};
    int hit0 = 0;
    for (uint64_t s = 0; s < 200; ++s) {
        auto out = S(L, {}, {.do_sample = true, .seed = s});
        if (out.next_token_id == 0)
            ++hit0;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "hit0=%d", hit0);
    check("L14 multinomial 强偏分布 token0 命中>60%", hit0 > 120, buf);
}

// ───────── M. 预算 top-k（NPU 加速钩子）─────────
static void test_precomputed_topk()
{
    // M1 单元：不在集合里的置 -inf
    auto r = Sampler::ApplyPrecomputedTopK({3.0f, 1.0f, 2.0f}, PrecomputedTopK{.token_ids = {0, 2}});
    check("M1 ApplyPrecomputedTopK 屏蔽非集合", r[1] == -INFINITY && r[0] == 3.0f && r[2] == 2.0f, "");
    // M1b nullopt no-op
    check("M1b precomputed nullopt no-op",
          approx(Sampler::ApplyPrecomputedTopK({3.0f, 1.0f, 2.0f}, std::nullopt), {3.0f, 1.0f, 2.0f}));
    // M1c 空集 no-op
    check("M1c precomputed 空集 no-op",
          approx(Sampler::ApplyPrecomputedTopK({3.0f, 1.0f, 2.0f}, PrecomputedTopK{}), {3.0f, 1.0f, 2.0f}));

    // M2 等价：precomputed=真实 top-k 集合 ≡ ApplyTopK(top_k) 结果
    std::vector<float> L{5.0f, 1.0f, 4.0f, 3.0f, 2.0f}; // top-2 = idx{0,2}（值 5、4）
    auto by_topk = S(L, {}, {.top_k = 2});
    auto by_pre = S(L, {}, Cfg(), PrecomputedTopK{.token_ids = {0, 2}});
    check("M2 precomputed 等价 ApplyTopK",
          approx(by_pre.probs, by_topk.probs) && by_pre.next_token_id == by_topk.next_token_id);

    // M3 优先级：precomputed 在场 + cfg.top_k 也设 → precomputed 覆盖
    auto both = S(L, {}, {.top_k = 1}, PrecomputedTopK{.token_ids = {0, 2}});
    // precomputed 赢 → 留 {0,2}（idx2 概率>0）；若 top_k=1 赢则只留 idx0（idx2=0）
    check("M3 precomputed 覆盖 cfg.top_k", both.probs[2] > 0.0f);

    // M4 精确值：precomputed={0,2} 无 cfg.top_k → softmax([5,-inf,4,-inf,-inf])
    auto out = S(L, {}, Cfg(), PrecomputedTopK{.token_ids = {0, 2}});
    check("M4 precomputed probs 精确", approx(out.probs, T_softmax({5.0f, -INFINITY, 4.0f, -INFINITY, -INFINITY})));
    check("M5 precomputed next==argmax", out.next_token_id == 0);

    // M6 precomputed 含越界 id（>=vocab 或负值）应被静默忽略、不崩、结果同不含
    std::vector<float> Lv{5.0f, 1.0f, 4.0f, 3.0f, 2.0f}; // vocab=5
    auto with_bad = S(Lv, {}, Cfg(), PrecomputedTopK{.token_ids = {0, 2, 999, -1}});
    auto without_bad = S(Lv, {}, Cfg(), PrecomputedTopK{.token_ids = {0, 2}});
    check("M6 precomputed 越界 id 被忽略",
          approx(with_bad.probs, without_bad.probs) && with_bad.next_token_id == without_bad.next_token_id);
}

// ───────── N. SamplerInputs 直接构造（真实接口，不借 S 包装）─────────
static void test_sampler_inputs_direct()
{
    // N1/N2: 直接构造 SamplerInputs，默认字段（generated_ids 空、precomputed nullopt）
    std::vector<float> L{3.0f, 1.0f, 2.0f};
    auto out = Sampler::Sample(SamplerInputs{.logits = L}, Cfg());
    check("N1 直接 SamplerInputs greedy==argmax", out.next_token_id == 0);
    check("N2 probs==softmax", approx(out.probs, T_softmax(L)));
    // N3/N4: move logits（零拷贝路径）——move 进 SamplerInputs 后原 vector 应空
    std::vector<float> Lm{5.0f, 1.0f, 2.0f};
    auto outm = Sampler::Sample(SamplerInputs{.logits = std::move(Lm)}, Cfg());
    check("N3 move logits 后原 vector 空", Lm.empty());
    check("N4 move 路径结果正确", outm.next_token_id == 0 && approx(outm.probs, T_softmax({5.0f, 1.0f, 2.0f})));
    // N5: 直接带 precomputed_topk
    std::vector<float> Lp{5.0f, 1.0f, 4.0f, 3.0f, 2.0f};
    auto outp = Sampler::Sample(SamplerInputs{.logits = Lp, .precomputed_topk = PrecomputedTopK{.token_ids = {0, 2}}},
                                Cfg());
    check("N5 直接 precomputed probs", approx(outp.probs, T_softmax({5.0f, -INFINITY, 4.0f, -INFINITY, -INFINITY})));
}

int main()
{
    struct T {
        const char* name;
        void (*fn)();
    };
    T tests[] = {
        {"A 形状&归一化", test_shape_norm},
        {"B 贪心=argmax", test_greedy},
        {"C temperature", test_temperature},
        {"D top_k", test_top_k},
        {"E top_p", test_top_p},
        {"F rep noop", test_rep_noop},
        {"F rep 翻转", test_rep_flip},
        {"F rep 负logit", test_rep_negative},
        {"F rep <1升权", test_rep_boost},
        {"F rep 在top_k前顺序", test_rep_order},
        {"F rep+temp 组合", test_rep_plus_temp},
        {"F rep 重复/多gen", test_rep_dup_gen},
        {"G 可复现", test_determinism},
        {"G rep+采样可复现", test_rep_sample_reproducible},
        {"H 纯temp", test_pure_temperature},
        {"H 纯top_k", test_pure_top_k},
        {"H 纯top_p", test_pure_top_p},
        {"H 纯normalize", test_pure_normalize},
        {"H 纯pick", test_pure_pick},
        {"H 纯rep", test_pure_rep},
        {"K 组合", test_composition},
        {"L 大词表", test_edge_large_vocab},
        {"L 全-inf", test_edge_all_neg_inf},
        {"L rep多token", test_edge_rep_many_tokens},
        {"L temp极大", test_edge_temperature_large},
        {"L top_p性质", test_edge_top_p_property},
        {"L multinomial统计", test_edge_multinomial_stats},
        {"M 预算topk", test_precomputed_topk},
        {"N SamplerInputs直接", test_sampler_inputs_direct},
    };
    for (auto& t : tests) {
        printf("--- %s ---\n", t.name);
        t.fn();
    }
    printf("\nSampler tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
