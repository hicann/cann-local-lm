# 量化过程验证方法论 + 现状

本文说明"如何验证整条量化过程"，以及 `dopt_op_benchmark/` 当前在这个体系里处于什么位置、还缺什么。配套的 `README.md` 是本目录的用法说明；本文是"为什么这么验、验到什么程度"的方法论与路线图。

---

## 一、量化在干什么

把原始 fp16 浮点模型（精度高、体积大、慢）压成 int4（每个权重只用 4 比特，小、快、但有误差），像把高清照片描成像素画。整条流水线 5 道工序：

```
原始 fp16 模型 + 校准数据
   ↓ ① GPTQ (dopt_lm)          算出每个权重该取哪个 int4 值 → trained_quant_weight.pth
   ↓ ② materialize (quant_swap) 把 int4 压成硬件认识的格式 (pack_int4 / transdata_scale)
   ↓ ③ QuantLinear / cann_lm::quant_matmul   组装成 W4A8 量化 matmul 算子
   ↓ ④ export → OM (ge.QuantBatchMatmulV4)   编译成 NPU 能跑的 OM
   ↓ ⑤ NPU 端到端                            上机器跑出文字 / PPL
```

每道工序都可能出错。**验证量化过程 = 一道道查过去，每道用合适的办法查。**

---

## 二、先分清两个根本不同的问题（所有困惑的根）

问"量化对不对"，其实在问两件不同的事：

| 问题 | 通俗说法 | 性质 |
|---|---|---|
| **A. 精度** | "描完的像素画跟原图像不像？" | 量化后模型还准不准 |
| **B. 算子正确性** | "描像素的工序有没有出错？" | 打包/算子实现有没有 bug |

这俩**用的标准不一样，混在一起就乱套**：

- 测精度时，拿**原图（fp16）**来比是合法的——你测的就是"掉了多少画质"。
- 测算子正确性时，**不能拿"另一台描像素机器描出来的"当标准**——万一两台机器同一个毛病呢？得拿一把**跟机器无关的尺子（数学定义）**。

> 即：测精度时 fp16 是对的参照；测算子正确性时 fp16（甚至 dopt 的仿真结果）都不能当参照。

---

## 三、问题 A（精度）怎么验

拿**量化后模型**和**原始 fp16 模型**，喂同样一批输入，比输出：

- **PPL（困惑度）**：模型对下个词预测有多准，越低越好。量化质量的金标准。
- **top-1 命中率**：两模型预测的下一个词一样不一样。
- **accuracy**：下游任务（MMLU/CEval）正确率。

这里 **fp16 是合法标准**（测的是相对原图的损失）。注意：几条 prompt 的 top-1 只是**冒烟**，统计上无意义；正式要上百上千条样本算 PPL。

---

## 四、问题 B（算子正确性）怎么验（难点）

### 4.1 直觉上会犯的错

"拿 dopt 仿真结果当标准，去对 converter 的算子"——**不行**，两个原因：

1. **语义对不上**：dopt 仿真只量化权重（W4A16），激活还是浮点；converter 是权重+激活都量化（W4A8）。算的东西不一样，对不齐。
2. **可能同源**：dopt 和 converter 在一个仓里，可能共享同一处规格级误解（比如都以为 int4 范围 [-7,7] 其实该 [-8,7]）。一致不代表都对。

> 就像拿工厂 A 的尺子去验工厂 B 的尺子——两把尺子同一个模具（都错），验出来"一致"也白搭。

### 4.2 正确做法：用"跟实现无关的标准"——数学定义

int4 量化说白了一句公式：

```
q = 四舍五入(权重 / scale)，限制在 [-7, 7]      # 量化
反量化 = q * scale
输出 = 输入 @ (q * scale)                       # matmul
```

这公式**不归任何代码所有**，是"定义"。用最朴素的 torch 照着写一个极简 reference（每步一行，肉眼可审），它就是"按定义的真值"。

### 4.3 但"独立实现"还不够，要钉到"手算夹具"

两个独立实现可能共享同一处规格级误解。所以最底层要有**完全不依赖代码**的真值——**手算**。造极小输入，用纸笔算出答案，断言 reference 对上：

```
权重 w = [2.0, -3.0, 1.5, 0.7]，scale = 1.0，一组
  → q = round([2,-3,1.5,0.7]/1.0) = [2, -3, 2, 1]   (1.5→2, 0.7→1)
  → 反量化 = q*scale = [2,-3,2,1]
输入 x = [1, 2, -1, 0.5]
  → 输出 = 1×2 + 2×(-3) + (-1)×2 + 0.5×1 = -5.5
```

**-5.5 是人用计算器算出来的，不卷入任何量化代码。** 这叫"手算夹具"，是整个验证可信度的**根**。

> 就像验一台计算器先按 "2+2=4"——这结果你背下来的，不来自任何计算器。

### 4.4 手算夹具只证明"reference 符合规格"，还要 NPU 校准

因为"规格本身可能写错"（比如规格写"四舍五入"，真硬件用"银行家舍入"）。这时手算夹具照样全过（reference 忠实执行了规格），却跟真硬件对不上。

所以再做一次 **NPU 校准**：同组输入分别跑 reference 和**NPU 真内核** `ge.QuantBatchMatmulV4`，对一下。一致才证明"规格忠实描述了硬件"。

> 一次性校准（reference ↔ 真硬件），之后 reference 就能在 CPU 上便宜地当标准，不用每次上 NPU。

### 4.5 对齐之后才验算子

reference 钉好（手算夹具 + NPU 校准两道保险），它就是可信标准。converter 的算子（`materialize` 打包、`cann_lm::quant_matmul`）都是**被测对象**，拿 reference 对：
- 打包对不对 → bit 级（pack 再 unpack 能否还原）
- 量化网格对不对 → `materialize` 算出的 q 跟 reference 的 q（round(w/s)）逐个对，该 bit-exact
- matmul 输出对不对 → 同权重同输入，输出跟 reference 比，在容差内

---

## 五、整条链五道关卡，各验各的

| 关卡 | 验什么 | 用什么标准 | 怎么验 |
|---|---|---|---|
| ① GPTQ 产出权重 | 精度掉多少 + 产物合法 + 权重网格对 | fp16 基线（精度）+ schema（契约）+ 数学（网格）| PPL、`verify_output.py`、查 q 离散且 offset=0 |
| ② materialize 打包 | q 跟数学定义一致吗；pack 能还原吗 | Tier 0 数学定义（手算钉）| bit-exact 对比 |
| ③ quant_matmul 算子 | 输出跟 W4A8 数学定义一致吗 | Tier 0 W4A8 reference | 数值容差对比 |
| ④ OM 编译 | OM 内核跟 reference 一致吗 | Tier 0 + NPU 内核（校准）| 跨 env，同输入对输出 |
| ⑤ NPU 端到端 | 部署模型准不准、出字合不合理 | fp16 基线 + 语义 | PPL、看出字 |

**每关用对应标准，别串。** 精度用 fp16，算子用数学定义。

---

## 六、标准权威层级

```
Tier 0  数学定义 (纯 torch 极简 reference, 手算微夹具钉 bedrock)  ← 算子正确性的标准
Tier 1  dopt fakequant (W4A16, GPTQ 意图)                        ← 权重网格交叉校验 + 量化精度(vs fp16)
Tier 2  NPU 真内核 ge.QuantBatchMatmulV4                          ← 部署真值, 校准 Tier 0 (规格↔现实)
e2e     端到端模型质量                                            ← 意图兜底
```

**三角互证**：手算夹具（数学）× NPU 内核（运行）× 端到端质量（意图）一致 = 强证据（非绝对）；任一不一致 = 定位到层去查。

**残留风险**：手算夹具不可避免编码了"规格选择"（int4 对称、clamp ±7…），这些是对内核行为的建模，可能建错。NPU 校准专门抓"规格 vs 现实"偏差，不可省。Tier 0 永远是"高可信代理"而非"绝对真值"，可信度 = 三层锚定的吻合度。

---

## 七、为什么不能图省事只用一个标准

- **全用 fp16**：算子对不上（W4A16 vs W4A8 语义不同）。
- **全用 dopt 仿真**：不覆盖激活量化 + 可能同源 bug。
- **全用 cann_lm CPU impl**：它自己是被测对象，可能错。
- **全用 NPU 内核**：慢、要跨 env、CI 跑不起。

所以**分层**：数学定义（便宜、bit 级、手算钉）做日常标准；NPU 内核做一次性校准（钉"规格↔现实"）；fp16 只用在"测精度损失"处。

---

## 信任分级：PASS ≠ 正确

`-8` 漏检的教训：**PASS 可能只是"循环一致"而非"正确"**。判据 = 标准是否独立于被测 + spec 选择是否锚到独立源（手算算术 / dopt 代码 / NPU 内核）。每个检查按此分三级：

| 等级 | 含义 | PASS 证明 |
|---|---|---|
| ✅ **真正确** | 标准独立于被测 + spec 锚到独立源 | 真的对 |
| ⚠️ **仅一致** | 标准与被测共享 spec | 实现没 bug，**不证明 spec 对** |
| ∅ **非正确性** | 标准 = 被测自身 | 自洽/可复现，**不是 correctness** |

**本标杆各检查的信任账**：

| 检查 | 等级 | PASS 证明了什么 | 不证明什么 |
|---|---|---|---|
| Part2 `pack_int4` round-trip | ✅ 真正确 | nibble 编码 bit-exact（与量化 spec 无关） | — |
| Part3 `transdata_scale` round-trip | ✅ 真正确 | scale 存储 bit-exact（与量化 spec 无关） | — |
| Part5 dopt -8 回归守卫 | ✅ 真正确（独立交叉） | materialize 保住 dopt -8（P1 修复后 0 改动/3层 11.5万 -8） | — |
| Tier0 W3/A1/M1 公式夹具 | ✅ 真正确 | round-half-to-even / dynamic_quant(absmax/127) / matmul 公式对（锚代码+算术） | clamp bound |
| ③A `dynamic_quant` CPU vs ref | ⚠️ 仅一致 + ✅ wiring | dynamic_quant 的 shape/scale/clamp wiring 对（bit-exact） | absmax/127 spec（同源） |
| ③B `quant_matmul` CPU(unpacked) vs ref | ⚠️ 仅一致 + ✅ wiring | matmul 的 转置/scale展开/pertoken/bias/dtype wiring 对（maxRelΔ~1e-6） | dequant+matmul spec（同源） |
| ③C `QuantLinear.forward`(packed) eager | ✅ 独立交叉（finding） | surface: packed[K//2,N] vs CPU impl 要 unpacked[K,N] → eager 断裂 | —（已是 finding） |
| Part1/Part4 `materialize==Tier0` | ⚠️ 仅一致 | materialize 的 round/scale展开/clamp **机制**没 bug | `[-7,7]` 是对的 spec（clamp 同源） |
| Tier0 W2 clamp 夹具 | ⚠️ 仅一致 | Tier 0 把 clamp `[-7,7]` 做对了 | `[-7,7]` 对不对 |
| golden self-consistency（maxΔ=0） | ∅ 非正确性 | golden 可复现/确定 | 算子正确 |
| `eval_qlinear`（4-prompt top-1） | ⚠️ 仅一致（精度 smoke） | GPTQ 没灾难性坏模型（4 条） | 正式精度（要 PPL） |

**关键标记**：凡是依赖 **clamp bound（-7 vs -8）** 的检查（Part1/4、W2），其 PASS **只信"机制没 bug"，不信"[-7,7] 对"**。`[-7,7]` vs `[-8,7]` 哪个是部署真值，未验，待 ④NPU 校准（独立源 = NPU 内核）裁定；已用 `ref_w4a8.INT4_NEG_CLIP` 标记待定。

**判据一句话**：可信 ⟺ 标准独立于被测 AND spec 锚到独立源。标准与被测共享 spec → PASS 只剩"一致"；标准=被测自身 → PASS 不是 correctness。spec 正确性只能来自**独立交叉腿**（dopt 代码 for dopt 范围、NPU 内核 for 部署范围、手算算术 for公式），自洽夹具抓不到 spec 选择错。



## 八、现状：已建 vs 待建

### 本目录已建（`dopt_op_benchmark/`）

| 文件 | 作用 | 在体系里的位置 |
|---|---|---|
| `gen_golden.py` | 跑 dopt QLinear fakequant，hook 抓每个算子 (输入激活, 输出) 存 golden | ①的 **W4A16 dopt 仿真 I/O golden**：可作**精度参照** + **权重网格交叉** |
| `compare.py` | 自洽校验（参考算子重现自己的 golden，必为 0）+ 候选算子对齐模板 | ③的对比 harness 模板（待接 Tier 0/候选算子）|
| `golden/manifest.json` | 契约（operator_impl=dopt.QLinear.fakequant, atol=rtol=1e-2）| 标杆数据契约 |
| `golden/e2e_logits.pt` | (4, vocab) 末端 logits golden | ①端到端 golden |
| `golden/op_io/*.pt` | 20 算子 × 4 prompt 的 (input_act, output_act) | ②③的算子 I/O golden（W4A16 语义）|
| `run.sh` | 薄封装（env + 自动 torchaudio shim + dispatch gen/compare/selfcheck）| 入口 |
| `shims/torchaudio/` | torchaudio 损坏时的 stub（自动启用）| 环境兼容 |
| `ref_w4a8.py` | **Tier 0**：极简纯 torch W4A8 s8s4 reference（独立于 converter 代码，每步一行）| **算子正确性的标准**（②③④对齐基准）|
| `bedrock/hand_fixtures.py` | **Tier 0 bedrock**：手算微夹具（人算 expected，5 case：权重量化/round-half-to-even/clamp/dynamic_quant/全 W4A8 matmul=-698/127）| 钉 Tier 0 到"人的算术"，不依赖任何代码 |
| `verify_materialize.py` | **②materialize 对齐 Tier 0**：`materialize_int4_per_group_symmetric` q vs Tier 0 q（bit-exact）+ `pack_int4` round-trip + `transdata_scale` round-trip；合成夹具 + 真实 dopt 权重 | 验 converter `quant_swap/materialize.py` 的 weight 量化/打包/scale 存储符合数学定义 |
| `verify_quant_matmul.py` | **③cann_lm 算子链对齐 Tier 0**：A `dynamic_quant`、B `quant_matmul`(unpacked)、C `QuantLinear.forward`(packed 链) vs `ref_dynamic_quant`/`ref_quant_matmul`/`ref_w4a8_linear` | 验 `dynamic_quant.py`+`quant_matmul.py`+`quant_linear.py` 的 wiring；C surface eager 路径断裂 finding |

**关键定位**：当前 golden 是 **W4A16 dopt fakequant**（权重 int4 反量化 + fp 激活 + fp matmul，`input_state=False`）。它能当：
- ①的**精度参照**（量化后 vs 原始 fp16 的输出对比，见外部 `eval_qlinear.py`）；
- ②的**权重网格交叉校验**（验证 `materialize` 的 q == dopt 的 q_ref，两者都该是 round(w/s)）。

**它不能当** ③④的算子标准——W4A16 不覆盖 W4A8 的激活量化，且 dopt 与 converter 同仓可能同源。

### 外部已建（未进本目录）

- `examples/test_converter/dopt_quantization_st/`：①的产物契约校验（`verify_output.py`）+ GPTQ 跑通（`run_st.py`），不含精度。
- `/tmp/opencode/dopt_eval/eval_qlinear.py`：①的 4-prompt 精度冒烟（top-1 4/4, cos 0.985）——非正式、未进仓。

### 待建（从可信度根 → 叶）

1. **④NPU 校准**：同输入跑 Tier 0 vs OM `QuantBatchMatmulV4`，存校准报告（钉"规格↔硬件"，**顺带定 -7 vs -8**：喂含 -8 权重看内核表示 -8 还是 clamp -7）→ 之后 Tier 0 才能在 CPU 便宜地当标准。
2. **①⑤正式精度标杆**：PPL（百~千样本）替代 4-prompt 冒烟，进 ST。
3. **③C finding 修复**：`QuantLinear.forward`(packed) eager CPU 断裂（CPU impl 要 unpacked）；若要 eager 回归，改 QuantLinear 存 unpacked 或修 CPU impl 解包。

> 进度：Tier 0 ✓ → ②materialize ✓ → ③cann_lm 算子链 ✓（A/B PASS, C surface eager 断裂 finding）→ **P1 已修复**（materialize clamp [-7,7]→[-8,7]，对齐 dopt/标准；`test_materialize.py` 同步改；126 UT pass；Part5 转回归守卫）→ 当前 ④NPU 校准（最后确认内核 [-8,7]）。

### 已修复的问题

- **P1（materialize clamp）**：`materialize.py:125-129` 由 `clamp(-7,7)` 改 `clamp(-8,7)`（加 `qmin=-(1<<(nbits-1))=-8`），对齐 dopt `qmin_max.py:124-125` + 标准 signed int4 二补码全范围。修复前丢每层 1万~5万个 -8（共 11.5万/3层，每个偏 +1 量子）；修复后 `verify_materialize Part5` 验 0 改动（回归守卫）。同步改 `tests/py/converter/test_materialize.py`（原测试编码了 [-7,7] bug）+ Tier 0 `INT4_NEG_CLIP=-8` + 手算夹具 W2/W2b。126 UT pass。
- **待修**：P2（QuantLinear eager CPU packed 断裂，不影响部署）、③C。

---

## 九、一句话

验证量化 = 沿 5 道工序逐个查；**查"精度"拿原图（fp16）比，查"工序"拿跟代码无关的数学定义比（手算钉 + NPU 校准）**，两套标准不能混。本目录当前：①的 W4A16 dopt 仿真 golden（精度参照 + 权重网格交叉）+ **Tier 0 数学 reference（`ref_w4a8.py` + 手算夹具 5/5 PASS，算子正确性的标准已就绪）**；下一步接 ②materialize/③cann_lm 算子对齐到 Tier 0。
