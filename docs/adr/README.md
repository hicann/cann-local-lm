# 架构决策记录 (ADR)

> 记录 CANN-LM 项目中重大架构决策的上下文、选择与后果，供未来追溯"为什么这样选"。

## 何时写 ADR

满足以下任一条件时写 ADR（而非仅写在设计 .md 里）：

- **算子/技术路径 pivot**：如自研算子 → 复用 CANN 原生算子（方案 A → 路径 C' 这类）
- **跨模块架构选型**：如 quant 与 export 的内存桥接 vs 一体化、subprocess vs 进程内
- **不可逆或高代价回退的决策**：如注册算子、落盘格式、IR 节点结构
- **多方案权衡后选定其一**：记录被否决的替代方案与理由

不写 ADR 的场景：函数重命名、行号更新、bugfix、局部重构（这些靠设计 .md + 代码注释）。

### 触发 checklist（任一 Yes → 写 ADR）

拿到一个改动，过一遍这 8 个问题，任一答 Yes 就写 ADR；全 No 则改设计 .md + 代码注释即可：

| # | 判断问题 | Yes 时的典型场景 |
|---|---------|-----------------|
| 1 | 是否在两个以上技术方案间做选择？ | 自研算子 vs 复用 CANN 原生；V3 vs V4；内存桥接 vs 磁盘桥接 vs subprocess |
| 2 | 是否引入或删除一个 custom op / GE 算子注册？ | `reg_register.h` 新增/删除 `REG_OP`；新加 `cann_lm::` schema+fake+converter 全套 |
| 3 | 回退这个改动的代价是否很高？ | 需改多文件 / 重导出所有 epair / 重编译 OM / 已有产物不兼容 |
| 4 | 是否跨模块影响链路结构？ | quant↔export↔loader↔saver 的桥接方式；prefill-decode 双图 vs single 图 |
| 5 | 是否改变权重/scale 的落盘格式或内存布局？ | weight_scale fp32→int64 位模式；预转置 vs 自然朝向；nibble-packed vs unpack |
| 6 | 是否改变环境依赖？ | 引入/移除 torch_npu（stub 路径 vs 真实 torch_npu）；torchair 源码编译 vs bundled |
| 7 | 是否改变部署形态？ | host-embed vs 图内 embed；lm_head 在 NPU vs host；tie_word_embeddings 解 tie |
| 8 | 三个月后回头看，是否需要知道"为什么当时这么选"？ | 否决了某方案但理由非显然；workaround 的取舍（如 dummy bias） |

### 典型场景 → 是否写 ADR

| 场景 | 写 ADR？ | 对应 checklist |
|------|---------|---------------|
| 方案 A `quant_matmul_s8s4` → 路径 C' `npu_quant_matmul` | ✅（已发生，未补录，留历史档案段） | 1,2,3,8 |
| weight_scale 落盘 fp32→int64（方案 A 改 buffer vs B 仅落盘） | ✅ | 3,5 |
| GPTQ 一键式选方案 C（subprocess）vs A（改 CANN 轻量化工具）vs B（内存） | ✅（已发生） | 1,4,6 |
| converter 不装 torch_npu 走 stub 路径 | ✅（已发生，记在 cli_design.md） | 6,8 |
| ~~`swap_to_quant_linear_from_dopt` 拆成 `_swap_linear_from_sd` + `swap_from_dopt`~~ | ✅ 已落地（2026-08-31）：`swap_to_quant_linear_from_dopt` 删除，`_swap_linear_from_sd` + `swap_from_dopt` 唯一路径 |
| `materialize_int4_with_scale` 改名 `materialize_int4_per_group_symmetric` | ❌ 重命名 | — |
| 函数行号偏移、签名参数收敛为 dataclass | ❌ | — |
| 新增 `QuantLinear.bias=None` 时 GE converter 塞 dummy zeros[N] | ⚠️ workaround 取舍（属路径 C' 决策的一部分，不单写） | 8（已含在路径 C' ADR 里） |

> 判断口径：**决策的"为什么"若只看代码看不出來 → 写 ADR**。代码能自解释的（重命名/拆分/行号）不写。

## Status 字段

| 状态 | 含义 |
|------|------|
| `Proposed` | 已提出，待评审/拍板 |
| `Accepted` | 已采纳，当前生效 |
| `Superseded` | 被后续 ADR 取代（注明被谁取代） |
| `Deprecated` | 弃用但无替代 |

设计 .md 顶部"状态"字段引用对应 ADR 的 Status。Status 变更时**必须**同步更新对应设计 .md 顶部状态字段。

## 命名约定

- `NNNN-kebab-case-title.md`（NNNN = 4 位序号，从 0001 起）
- `0000-template.md` = 本目录的模板（不作为正式 ADR）
- 序号单调递增，不复用被 Superseded 的号

## 与设计 .md 的关系

| 文档 | 承载内容 | 易变性 |
|------|---------|--------|
| `docs/adr/*.md` | 决策的"为什么"——上下文、权衡、被否决的替代方案 | 低（决策一旦写下基本不变，只改 Status） |
| `docs/cann_lm_*/*.md` | 设计的"是什么"——函数清单、签名、流程、行号锚定 | 高（随代码演进） |

设计 .md 顶部加 `> 状态: 当前采用 | 最后核对: YYYY-MM-DD`（有对应 ADR 时附链接）。ADR 是事实来源，设计 .md 反映当前实现状态。

## 索引

| ADR | 标题 | Status |
|-----|------|--------|
| [ADR-0001](./0001-load-model-cpu-quant-subprocess-device.md) | converter 主进程 load_model 写死 CPU，轻量化工具 subprocess 支持设备选择 | Accepted |

> 历史已发生的 pivot（如方案 A `quant_matmul_s8s4` → 路径 C' `npu_quant_matmul`）不补录 ADR，保留在对应设计 .md 的"历史档案"段追溯。
