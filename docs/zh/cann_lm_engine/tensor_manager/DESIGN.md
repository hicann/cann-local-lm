# TensorManager 设计文档

简体中文 | [English](../../../en/cann_lm_engine/tensor_manager/DESIGN.md)

| 字段 | 值 |
|------|-----|
| Spec ID | — |
| Level | L3 核心组件（`cann_lm/engine/core/tensor_manager.h/.cpp`） |
| 状态 | **已落地**（v0.12 backend 代理；v0.14 增 KV 批量通道 `CreateKvCacheTensor`，C-04 落地） |
| 版本 | 0.2 |

## §1 范围与目标

TensorManager 是 engine 侧的 **tensor 申请统一入口**（单例代理），为输入/输出 tensor 与
KV cache tensor 提供申请通道；与 KvCachePool（M-02）分工：

- **TensorManager**：申请通道——`CreateTensor`（每步重填的输入/输出 tensor，modelId
  关联已加载模型）与 `CreateKvCacheTensor`（KV 池专用批量通道，无 modelId——池在模型
  Load 前创建，不绑定具体模型）；内部委托 backend `CreateTensorBuffer(TensorDesc,
  DeviceType)` 工厂，engine Status 经 `map_backend_status` 映射
- **KvCachePool**：跨步累积的 KV cache tensor 的**语义与生命周期**（槽位/光标/驱逐/复用，
  持有 `shared_ptr<ITensorBuffer>` RAII），构造时**单次调用** `CreateKvCacheTensor`
  申请全部 `2×num_hidden_layers` 个 tensor

**设计决策（D-13 补充 → v0.14 演进）**：KV cache 的驱逐/复用等特有操作自包含在
KvCachePool；其底层内存的申请/释放经 TensorManager 统一入口（C-04 落地——取代 v0.1
草案的 `AllocKvBuffer`→`AttachBuffer` 注入式分工，v0.1 草案未实现）。

## §2 当前状态

**已落地**（`core/tensor_manager.h/.cpp`）：

- 单例 `Instance()`；`backend_type_ = DetectAvailableDevice()` 运行期探测（NDK 优先）
- `CreateTensor(TensorDesc, modelId, out*)`：`ModelExecutor::Load` 后按 built descs 申请
  `input_tensors_`/`output_tensors_`（model_executor 消费）
- `CreateKvCacheTensor(num_hidden_layers, kv_cache_max_len, num_attention_kv_heads,
  batch, num_attention_head_dims, DataType, out*)`：**单次创建全部** `2×num_hidden_layers`
  个 per-layer K/V tensor（KvCachePool 构造消费）；进接口统一校验（任一参数 ≤ 0 /
  dtype UNDEFINED / out 空 → INVALID_ARGUMENT）+ uint64 乘积溢出防护 + all-or-nothing
  （任一失败 → FAILURE，out 不产出）
- `DataTypeSize(DataType)`：dtype → 字节数 inline 辅助
- `Release(shared_ptr)`：RAII 兜底（显式释放可选）

v0.1 草案的 `Init/GetStepTensors/AllocKvBuffer/FreeKvBuffer` 未实现，被上述实际形态取代
（InputAssembler 的每步临时 tensor 现为 `ModelExecutor::Load` 申请的 `input_tensors_`
工位，按 name→index 查找填充，不经池化）。

## §3 对外接口（实况）

```cpp
class TensorManager {
public:
    static TensorManager& Instance();

    /// 按 desc + modelId 申请 tensor（输入/输出；内部委托 CreateTensorBuffer）。
    Status CreateTensor(const TensorDesc& desc, int modelId,
                        std::shared_ptr<ITensorBuffer>* out);

    /// KV cache 池专用批量通道：一次调用创建全部 2×num_hidden_layers 个 tensor。
    /// 维度 [kv_cache_max_len, num_attention_kv_heads, batch, num_attention_head_dims]，
    /// 命名 kv_cache_k_<layer> / kv_cache_v_<layer>，输出序 [k0,v0,k1,v1,...]。
    Status CreateKvCacheTensor(int num_hidden_layers, int kv_cache_max_len,
                               int num_attention_kv_heads, int batch,
                               int num_attention_head_dims, DataType dtype,
                               std::vector<std::shared_ptr<ITensorBuffer>>* out);

    /// 显式释放（可选；RAII 亦自动释放）。
    Status Release(const std::shared_ptr<ITensorBuffer>& t);
};
```

## §4 与其他模块的关系

| 模块 | 关系 |
|------|------|
| KvCachePool (M-02) | 构造时**单次**调 `CreateKvCacheTensor` 申请全部 KV tensor；槽位/光标/驱逐归 KvCachePool |
| ModelExecutor (M-01) | Load 后按 built descs + modelId `CreateTensor` 申请 `input_tensors_`/`output_tensors_` |
| InputAssembler (M-03) | 填 ModelExecutor 暴露的 `input_tensors_` 工位（不经 TensorManager） |
| backend | 委托 `cann_llm_engine::CreateTensorBuffer(TensorDesc, DeviceType)` 工厂 |

## §5 内存排布

- KV Cache（v0.14）：**每层独立 K/V tensor**（共 `2×num_hidden_layers` 个），单 tensor
  维度 `[kv_cache_max_len][num_attention_kv_heads][batch][num_attention_head_dims]`，
  序 `[k0,v0,k1,v1,...]`，命名 `kv_cache_k_<layer>`/`kv_cache_v_<layer>`（满足 om 按层
  绑定整层 K/V）；dtype 由 meta_json `kv_cache_type` 解析（小写词表 fp16/fp32…，缺省/
  未知回退 FLOAT16）；容量/形状参数自 meta_json model_config 段权威
- 输入/输出 tensor：按 built descs（backend 模型定义）

## §6 落地计划（v0.1 草案 → 现状）

1. ~~定义 `ITensorAllocator` 接口~~ → 未采用；v0.12 起直接经 backend `CreateTensorBuffer` 工厂
2. ~~TensorManager 池化分配/释放~~ → 落地为单例代理（`CreateTensor`/`CreateKvCacheTensor`），无池化
3. ~~KvCachePool 从 AttachBuffer 切换为 TensorManager 注入~~ → **已完成（v0.14）**：
   `CreateKvCacheTensor` 单次批量申请；旧 `AttachBuffer`/`PoolHandle`/`SlotOffset` 已删除
4. ~~InputAssembler 从裸 `shared_ptr<Tensor>` 切换为 TensorManager 获取~~ → 形态演进：
   Load 后按 descs 申请工位，assembler 按名填工位（name→index）

## §7 变更日志

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1 | 2026-08-19 | 初版：期望/规范状态文档（`AllocKvBuffer`→`AttachBuffer` 注入式草图；代码未提供）。 |
| 0.2 | 2026-09-08 | 对齐实况：TM 为 tensor 申请统一入口（`CreateTensor` + KV 批量通道 `CreateKvCacheTensor`）；C-04 落地——KvCachePool 构造单次申请全部 `2×num_hidden_layers` 个 per-layer K/V tensor（meta_json kv 形状参数驱动，dims `[kv_cache_max_len][kv_heads][batch][head_dims]`）；旧 `AttachBuffer`/`PoolHandle`/`SlotOffset` 删除。§1/§2/§3/§4/§5/§6 重写。 |
