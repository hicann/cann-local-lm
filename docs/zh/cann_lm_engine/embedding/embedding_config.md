简体中文 | [English](../../../en/cann_lm_engine/embedding/embedding_config.md)

该文件指定 embedding 模块的配置参数输入结构：

example:

```json
{
    "embedding_config": {
        "embedding_weights": "gemma4.embedding_weights",
        "embedding_quant_mode": "per_token",
        "embedding_load_mode": "in_memory",
        "embedding_type": "fp16",
        "embedding_scale_type": "fp16",
        "token_identity_type": "int8",
        "token_identity_scale_type": "fp32"
    }
}
```

## 键说明

| 键 | 含义 | 缺省 |
|---|---|---|
| `embedding_weights` | 单文件路径（四段布局，见下）；空 = 不加载（tokenId 部署形态） | 必填（启用 embedding 时） |
| `embedding_quant_mode` | scale 粒度：`per_token` / `per_group` | `per_token` |
| `embedding_load_mode` | 加载方式：`in_memory` / `in_flash` | `in_memory` |
| `embedding_type` | 主表存储类型：`fp32` / `fp16` / `int8` | `fp32` |
| `embedding_scale_type` | 主 scale 存储类型：`none` 以外的 `fp32` / `fp16` | `none` |
| `token_identity_type` | identity 表（per_layer weight）读取类型：`fp32` / `fp16` / `int8` | `fp16` |
| `token_identity_scale_type` | identity scale（per_layer scale）读取类型：`fp32` / `fp16` | `fp16` |

- `embedding_dequant_scale` 键已废弃：`embedding_weights` 单文件承载全部 scale
  段；残留非空值仅产生 WARN，不影响解析。
- identity 相关段是否读取由部署图工位判定（以图为准）；`token_identity_type`
  两键在主表为空（tokenId 形态）时仍独立解析。

## 单文件四段布局

`embedding_weights` 指向的文件为四段紧排（段内行优先，无文件头、无段间
padding）；段长为 0 的段物理不存在，后段偏移自然前移：

```
┌──────────────────────────────┬ offset 0
│ seg1 embedding weight        │ L1 = vocab × hidden × elem(embedding_type)
├──────────────────────────────┬ L1
│ seg2 per_layer weight        │ L2 = vocab × W × elem(token_identity_type)
│   identity 表 [vocab, W]     │   W = 图工位 dims 尾维连乘（未启用 → 0）
├──────────────────────────────┬ L1 + L2
│ seg3 weight scale            │ L3 = vocab × groups × elem(embedding_scale_type)
├──────────────────────────────┬ L1 + L2 + L3
│ seg4 per_layer scale         │ L4 = vocab × elem(token_identity_scale_type)
└──────────────────────────────┴ total = L1 + L2 + L3 + L4
```

identity 启用时（图带工位），文件总长须精确等于四段和（引擎加载期校验）。
