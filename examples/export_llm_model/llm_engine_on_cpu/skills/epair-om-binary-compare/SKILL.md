---
name: epair-om-binary-compare
description: |
  二进制比较两个 .om（或 .pt2/.epair）编译产物，判断是否同源、差异在哪、差异是否实质。
  当用户提到以下场景时使用：
  - 比较两个 .om / .pt2 / .epair 文件是否一致（"一样吗"、"相同吗"）
  - 两次导出/编译产物比对、回归验证（重构后产物是否等价）
  - .om 字节级 diff、找差异字节、cmp/md5 比对
  - 解释 .om 差异原因（GE 编译非确定性 ID / 时间戳 / 路径）
  - 判断 .om 差异是否影响模型内容（权重/图 vs 元信息）
version: 1.0.0
metadata:
  author: assistant
  category: model-verification
  model: qwen3-0.6b
  reference: torch-delegate PR #93
allowedTools:
  - read
  - write
  - bash
  - grep
  - glob
---

# epair/.om 二进制比较技能

## 适用场景

两个编译产物（`.om` / `.pt2` / `.epair`）想确认是否同源、是否等价、差异在哪、差异是否实质影响模型内容。典型：重构 converter/C++ 后跑回归，比新旧两个 .om 是否一致。

## 核心方法论（4 步）

### Step 1：大小 + md5（快速判等）
```bash
ls -l A.om B.om            # 精确大小
md5sum A.om B.om           # 整体哈希
```
- 大小不同 → 必然不同。
- 大小同 + md5 同 → 完全一致。
- **大小同 + md5 不同 → 大概率"近乎一致"，进入 Step 2 定位差异**（同源编译、仅元信息不同最常见）。

### Step 2：cmp 定位首个差异 + 差异总量
```bash
cmp A.om B.om                       # 首个差异字节（如 "differ: byte 1750, line 64"）
cmp -l A.om B.om | wc -l            # 差异字节数（如 6318）
cmp -l A.om B.om | awk '{print $1}' | sort -n | awk 'NR==1{min=$1}{max=$1} END{print "min="min,"max="max,"count="NR,"span="max-min+1}'
```
- `count / 总字节` = 差异占比。差异占比极小（如 6318 / 1.5G = 0.0004%）→ **几乎一致**。
- `span` vs `count`：若 span 远大于 count（如 span=6.4M 但 count=6318）→ 差异**散布成小簇**（非连续块），典型是元信息区里的零散字段。

### Step 3：看差异 offset 分布（判断在头部还是权重区）
```bash
cmp -l A.om B.om | awk '{print $1}' | sort -n | (head -20; echo ...; tail -20)
```
- 差异集中在文件**前几 MB**（头部元信息区）→ 后端权重/编译图主体一致。
- 差异散布到 GB 级权重区 → 权重本身不同（严重，非同源或权重错）。

### Step 4：dump 差异区内容，识别差异是什么
```bash
# 首个差异附近，两文件各 dump 字符串 + hex
dd if=A.om bs=1 skip=1700 count=200 2>/dev/null | strings -n1
dd if=B.om bs=1 skip=1700 count=200 2>/dev/null | strings -n1
dd if=A.om bs=1 skip=1600 count=400 2>/dev/null | xxd | head -25
dd if=B.om bs=1 skip=1600 count=400 2>/dev/null | xxd | head -25
```
对比两文件的字符串/hex，定位差异字段。

## 关键经验：GE 编译产物的非确定性 ID

**GE（Graph Engine）编译 .om 时会植入非确定性字段**，导致同模型同编译两次跑出的 .om **字节不同但实质等价**：

| 非确定性字段 | 样例 | 位置 |
|------|------|------|
| 动态常量节点 ID | `dynamic_const_1747858_435` vs `dynamic_const_1508749_435` | 头部图描述区，~6 字节小簇 |
| 编译时间戳 | 嵌入元信息 | 头部 |
| 源文件路径串 | 编译时 CWD 相关 | 头部 |

特征：差异字节**集中在文件前几 MB 的元信息区**，成 ~6 字节小簇散布，**后端权重区（GB 级）逐字节一致**。差异占比 < 0.001%。

→ 判定：**同模型 + 同编译，差异纯粹是 GE 编译期 ID/时间戳，不影响模型内容**（图结构 + 算子 + 权重一致）。

## 实战样例：qwen3-0.6B 两个 .om 比较

两份 `.om` 均由 qwen3-0.6B + stub GE 编译生成（不同时间）：

| 指标 | 值 |
|------|------|
| 精确大小 | **完全一致** 1,511,769,165 字节（1.41 GiB） |
| md5 | `9ae705214e76cb555b6f6f0fd539b5b7` vs `46746f3db24f53a38d59ccd7d147af74`（不同） |
| 差异字节数 | **6318 / 1.5G**（0.0004%） |
| 首个差异 | byte 1750 |
| 差异范围 | byte 1750 ~ 6457384（span ~6.4MB，count 6318 → 散布小簇） |
| 差异内容 | `dynamic_const_<ID>_435` 的 ID 数字不同（`1747858` vs `1508749`），其余 GE 元信息（`origin_shape_initialized`/`placement`/`OUTPUT_IS_VAR`/`DNN_VM_GE_LOCAL_OP_STORER` 等）逐字节同 |
| 结论 | **实质等价**：同模型同编译，差异仅 GE 编译期自增 ID，图+权重主体一致 |

## 判定流程图

```
md5 同? ──YES──> 完全一致
   │NO
   ├─ 大小同 + 差异占比 <0.1% + 集中头部? ──YES──> 实质等价（GE 非确定性 ID/时间戳）
   │
   ├─ 差异散布到权重区(GB级)? ──YES──> 权重不同（严重，需查源）
   │
   └─ 差异在图结构区(算子/拓扑)? ──YES──> 图不同（重构/导出逻辑变了，需回归）
```

## 命令速查

```bash
# 一键：大小+md5+首差异+差异计数+范围
A=path/a.om; B=path/b.om
stat -c '%n %s bytes' "$A" "$B"
md5sum "$A" "$B"
cmp "$A" "$B"
cmp -l "$A" "$B" | wc -l
cmp -l "$A" "$B" | awk '{print $1}' | sort -n | awk 'NR==1{min=$1}{max=$1} END{print "min="min,"max="max,"count="NR,"span="max-min+1}'

# dump 首差异附近内容
OFF=1700; LEN=300
dd if="$A" bs=1 skip=$OFF count=$LEN 2>/dev/null | xxd | head -20
dd if="$B" bs=1 skip=$OFF count=$LEN 2>/dev/null | xxd | head -20
```

## 注意

- `.om` 是 GE 编译后的二进制（含权重 + 编译图 + 元信息），**不可读**，只能二进制 diff。
- `.pt2` / `.epair`（归档格式）同理可 cmp；`.epair` 若是目录（ep_json+weights_*.bin），先比 `ep_json_with_ascend_ir.json`（`json.dumps(sort_keys=True)` 规范化后 diff）+ `weight_info.json` + 各 `weights_*.bin` 的 md5。
- GE 编译**非确定性**是已知行为（节点 ID 自增、时间戳），回归比对时**容忍这部分头部元信息差异**，重点看权重区 + 图拓扑是否一致。
- 若需严格可复现：固定 GE 编译期 seed / 关闭时间戳（取决于 GE 是否支持），否则同模型两次编译必有微小字节差异。
