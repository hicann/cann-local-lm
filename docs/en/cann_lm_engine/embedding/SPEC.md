# SR20260828001 — CANN-LM Engine TextEmbedding Module

English | [简体中文](../../../zh/cann_lm_engine/embedding/SPEC.md)

## Chapter 1 Overview

This specification defines the business behavior of the C++ embedding module under `cann_local_lm/engine/core/component/text_embedding`. This module provides the CANN-LM inference engine with embedding table loading, host-side table lookup (gather), and model input slot filling capabilities: loading low-precision embedding tables and dequantization scale parameters from the file system into memory, or (InFlash mode) leaving the table in the file system and reading on demand; gathering low-precision rows by index on the host side, and filling the results into model input slots (`input_embed` / `embed_scales`) in the type declared by the om graph `TensorDesc`. Dequantization (dequant) is **always done on the NPU side**; this module does not perform any dequantization math on the host side. This module is an engine-internal C++ component, consumed only by the pipeline / model_executor layer, and **does not expose** any cross-language or external interfaces.

Requirement ID: **SR20260828001**. Software layer: **framework** (engine core reusable library, includes backward compatibility chapter).

The reader should understand within 30 seconds: this spec defines a C++ embedding module's behavior contract that "loads low-precision tables + host gather + proactively fills slots", dual-path (tokenId / embedding), supports perToken / perGroup quantization granularity, consumed internally by the engine.

## Chapter 2 Glossary

| Term | Definition |
|------|------|
| embedding table | 2D matrix of `[vocab_size, hidden_dim]`, row v is vocabulary entry v's embedding vector, stored in file system as low-precision type |
| storage type | byte layout type of embedding / scale data in the **file system** (`EmbeddingType` / `EmbeddingScaleType`), specified by user at Load time |
| fill type | actual data type written to model input slot, determined by om graph `TensorDesc.dataType`; when inconsistent with storage type, module **must** do type conversion before filling |
| scale / dequantization parameter | coefficient needed to restore low-precision embedding to high precision; stored in the same single file as the embedding table (third segment) |
| PerToken | quantization granularity: each vocabulary entry (one row of table) corresponds to **one** scale |
| PerGroup | quantization granularity: one row of embedding divided into groups by `group_size`, **each group** has its own scale; one tokenId corresponds to multiple dequantization parameters |
| group_size | element count per group in PerGroup mode (hidden dimension direction); required in PerGroup mode |
| gather / table lookup | operation of copying corresponding row from table by tokenId index; host-side gather is pure index-based copy (stride copy), no dequantization |
| tokenId path | inference path where om graph has built-in embedding lookup: **embedding table already included in model weights loaded with graph**, graph does gather + dequant internally, input slot holds token id; table not passed through this module |
| embedding path | inference path where om graph has no built-in embedding: host-side gather low-precision rows fill `input_embed`, NPU-side dequant |
| InMemory / InFlash | table loading method: InMemory = entire table mmap read-only mapped; InFlash = table stays in file system, no resident memory, read directly from file by row offset during gather/fill |
| slot / input slot | model om graph input tensor slot (`input_embed`, `embed_scales`, etc.), see `docs/model_executor/INPUT_INFO.md` |
| name→index mapping | `TensorDesc.name` → `input_tensors_` index mapping table, established at model_executor Load |
| ITensorBuffer | backend abstract tensor container (`backend/api/tensor_buffer.h`): `GetData()` / `GetSize()` / `GetDesc()` |
| TensorDesc | om graph tensor metadata snapshot: `name` / `dataType` / `format` / `dims` / `size` |
| type conversion | host-side data representation change (e.g. INT8 value expanded to FLOAT16), **does not multiply scale**; strictly distinguished from dequantization (multiply scale, NPU side) |
| D-07 | model_executor design accepted decision: tokenId goes through model internal embedding, embedding table loaded by pipeline calling embedding module (C-02), model_executor does not re-implement table lookup |
| C-02 | call contract number pipeline→embedding module in model_executor design |
| RSS | process resident physical memory set, used to measure table loading memory footprint |

## Chapter 3 Background

### 3.1 Current State

`cann_local_lm/engine/core/component/text_embedding/text_embedding.h/.cpp` is currently a stub era stub implementation: `TextEmbedding::LoadAndBind()` directly returns `FAILURE`, header comment notes "polymorphism pending docs/embedding DESIGN". Same file also holds `ModalityEntry` (multimodal placeholder positioning entry, another responsibility consumed by Preprocessor). `INPUT_INFO.md` records `embed_scales` slot filler as embedding module (currently pending). User requirement original input see `docs/embedding/proposal.md`.

### 3.2 Problem

- **No table loading capability**: models without om graph built-in embedding (embedding path) cannot obtain input embedding; `embed_scales` slot has no filler.
- **Quantized deployment blocked**: low-precision (INT8, etc.) embedding table + scale quantized deployment form lacks host-side loading and gather channel.
- **Contract missing**: `LoadAndBind()` stub signature doesn't match real requirements (path, type, quantization granularity, slot filling), no behavior spec to follow.

### 3.3 Expectations and Confirmed Design Decisions

Module responsibility determined via brainstorm (2026-08-27) and user three-round decisions (2026-08-28) as:

| Dimension | Conclusion |
|------|------|
| Responsibility positioning | **Dual path**: tokenId path (om graph built-in embedding, table already in weights loaded with graph, module only fills token id) | embedding path (host gather low-precision rows, NPU dequant) |
| Dequantization location | **Always NPU side**; host only reads low-precision table + scale and fills slots, host does not do dequantization math |
| File format | **embedding_weights single file** four segments packed contiguously: main table raw 2D `[vocab, hidden]` → per_layer table → main scale → per_layer scale (no file header, no inter-segment padding); scale segment shape determined by quantization granularity (see 4.1.1) |
| Quantization granularity | initialization input parameter `quantMode`: **PerToken** (one scale per vocabulary entry) or **PerGroup** (one tokenId's embedding corresponds to multiple dequantization parameters, `GroupSize` required) |
| Table loading method | optional: **InMemory** (entire table mmap read-only mapped, default) or **InFlash** (table not read into memory, read directly from file by row offset during gather/fill) |
| Structure approach | **Single class + type tag dispatch** (no polymorphic hierarchy; host gather type-agnostic); when some type has differentiated host logic (e.g. INT4 unpack) upgrade to strategy/polymorphic, interface shape unchanged |
| Table delivery method | **No BindToModel**: table/scale not weight-bound. tokenId path no table needed (table in weights); embedding path table/scale as **input slot data** proactively filled by this module |
| Slot filling attribution | **Module proactively fills** (approach B): `FillInputs(...)` depends on model_executor's input bundle + name→index mapping |
| Fill type | actual fill type per target slot `TensorDesc.dataType`; user specifies file system storage type; mismatch **must** implement type conversion before filling |
| gather output | `Gather` **simultaneously outputs** gathered low-precision rows (`out_embed`) and per-token scales collected by tokenId (`out_scales`) |

### 3.4 Module Boundary

- **This module does**: embedding_weights single-file four-segment loading (InMemory whole-table mmap / InFlash keep in file read by row), host gather (including per-token scale collection), type conversion, input slot filling (including tokenId path's token id filling).
- **This module does not do**: dequantization (NPU), tokenId path table delivery (**table already included in model weights loaded with graph**, not through this module), other slot filling like attention_mask/position_ids (InputAssembler), multimodal placeholder replacement (Preprocessor, via `ModalityEntry`), table memory underlying allocation strategy (via ITensorBuffer abstraction), InFlash table file hot update.
- **Upstream/downstream basically unchanged, except one wiring change**: `PipelineExecutorFactory::create` still passes `TextEmbedding` by value; `Preprocessor` still holds non-owning `TextEmbedding*`; **`Scheduler` constructor signature needs to add `TextEmbedding*` non-owning pointer** (`input_embed` slot's sole writer in current code is `Scheduler`'s two `SetInputData` call sites; embedding/tokenId both paths' slot filling connected via `FillInputs`, see DESIGN §2.2 wiring point); only `LoadAndBind()` stub replaced by new interface.

## Chapter 4 Functional Requirements

### 4.1 Behavior Specification

#### 4.1.1 Initialization and Loading (Load)

- **Preconditions**: caller (pipeline, M-02 at init phase) holds complete `EmbeddingConfig` (including table loading method `load_mode`). config passed via **meta_json `embedding_config` section → pipeline inline parsing step directly calls `EmbeddingConfig::FromSection`** channel (SR20260903001 supersedes U-8 route b — channel part superseded, parsing semantics maintained, 2026-09-04 user decision); assembly mapping see DESIGN C-01 §2.
- **Processing logic**:
  - Module **must** read the main table from the first segment (offset 0) of the single file at config's `table_path`, byte layout is raw 2D `[vocab_size, hidden_dim]` row-major contiguous. Reading method determined by `load_mode`:
    - **InMemory** (default): main-table segment mmap read-only (RAII munmap, row pointers point directly into mapped pages).
    - **InFlash**: table **not read into memory** — Load only validates file metadata (open file, validate byte size, record row stride = `hidden_dim × elem_size`), keeps file open; subsequent `Gather` / `FillInputs` read directly from file by row offset (`id × stride`) via positional read (`pread`-like semantics).
  - Module **must** read the main scale segment (third segment, offset = main table + identity table length) from the `table_path` single file into host-side heap memory when `scale_type ≠ None`. **scale is always in-memory** (InFlash not applicable — scale is `vocab` or `vocab × groups` scalars, much smaller than table, and randomly accessed by index on every gather). scale element count determined by quantization granularity:
    - **PerToken**: `vocab_size` elements (v-th element is table's v-th row's scale), segment byte size = `vocab_size × scale_elem_size`.
    - **PerGroup**: `vocab_size × groups` elements row-major (token v's g-th group scale at index `v × groups + g`), where `groups = hidden_dim / group_size`; segment byte size = `vocab_size × groups × scale_elem_size`.
  - Module **must** validate in PerGroup mode that `group_size` is provided, `group_size > 0`, `group_size ≤ hidden_dim` and `hidden_dim % group_size == 0`, any failure returns `INVALID_ARGUMENT`.
  - Module **must** validate file byte size (InMemory actual mapping, InFlash `stat` metadata) satisfies the main-table-segment expectation (≥ passes, later segments may follow), and the main scale segment reads in full (short read returns `INVALID_ARGUMENT`); exact validation of the whole-file four-segment total is performed by LoadTokenIdentity (when identity is enabled).
  - Module **must** support `scale_type = None` (deployment form without the main scale segment): no scale loaded, subsequent `Gather`'s `out_scales` produces no data.
  - Module **must** return `INVALID_ARGUMENT` when file missing / unreadable, no exception thrown.
  - Module **should** record table scale (vocab_size / hidden_dim / type / quantization granularity / loading method) after successful load for accessor query.
- **vocab_size / hidden_dim source**: provided as Load config input parameter by caller (expected from ModelConfig; see open item U-2).
- **tokenId path relationship with Load**: tokenId path's table already included in model weights loaded with graph, this module's `Load` **only serves embedding path**; tokenId path's `FillInputs` does not require prior Load (see 4.1.3).

#### 4.1.2 Table Lookup (Gather)

- **Preconditions**: module successfully Loaded; caller provides `ids[count]` and two output buffers.
- **Processing logic**:
  - Module **must** copy corresponding low-precision row from table for each tokenId (`hidden_dim × elem_size` bytes, stride copy) to `out_embed`, layout is `[count, hidden_dim]` contiguous. Copy is **pure byte moving**, unrelated to storage type, **no dequantization**.
    - **InMemory**: copy from memory table buffer.
    - **InFlash**: positional read from table file by row offset `ids[i] × stride` (`pread`-like semantics, doesn't move file cursor, multi-thread concurrent).
  - Module **must** simultaneously collect scales by tokenId to `out_scales` (scale always in-memory, see 4.1.1):
    - PerToken: `count` elements (i-th element = table's `ids[i]`-th row's scale).
    - PerGroup: `count × groups` elements (token i's g-th group scale at index `i × groups + g`).
  - Module **must** return `INVALID_ARGUMENT` when any `ids[i] ≥ vocab_size` (reject entire batch, no partial output).
  - Module **must** return `INVALID_ARGUMENT` when `out_embed` / `out_scales` capacity (`GetSize()`) insufficient.
  - Module **must** return `INVALID_STATE` when not Loaded.
  - Module **must** return `FAILURE` when InFlash read fails (I/O error / short read) (file truncated/replaced after Load etc. runtime corruption belongs to this path).
  - Module **should** guarantee `Gather` is `const` read-only operation, can be called concurrently in multi-threaded (see Chapter 5).
  - Module **may** do memory locality optimization for InFlash batch random reads (e.g. sort by file offset then sequential read), but **must not** change `out_embed` row order semantics.
  - When `scale_type = None`, module **must** skip `out_scales` write and return `SUCCESS`.

#### 4.1.3 Slot Filling (FillInputs)

- **Preconditions**: caller (model_executor / pipeline) provides input bundle (`vector<shared_ptr<ITensorBuffer>>`) and name→index mapping; Embedding path also requires module successfully Loaded.
- **Processing logic**:
  - Module **must** support two filling paths, selected by `path` parameter:
    - **Embedding path** (`EmbeddingPath::Embedding`):
      - Module **must** execute 4.1.2's gather for `ids` (rows + per-token scale; both InMemory / InFlash supported).
      - Module **must** fill gathered rows into `input_embed` slot: **actual write type per that slot's `TensorDesc.dataType`**; when inconsistent with storage type, module **must** complete type conversion before filling (conversion semantics see 4.1.4).
      - Module **must** fill collected scales into `embed_scales` slot (skip when `scale_type = None`, see open item U-6): PerToken fills `[count]`, PerGroup fills `[count × groups]`.
    - **TokenId path** (`EmbeddingPath::TokenId`):
      - **embedding table already included in model weights loaded with graph**, this module does not handle table/scale; module **must** fill token id into `input_embed` slot (per slot `TensorDesc.dataType`, supports INT32/INT64, rest returns `INVALID_ARGUMENT`), om graph does gather + dequant internally.
      - TokenId path **does not require module to Load first**: `FillInputs(TokenId, ...)` can normally fill token id in unloaded state.
  - Module **must** locate slots via name→index mapping; `input_embed` missing returns `NOT_FOUND`; Embedding path with `scale_type ≠ None` and `embed_scales` missing returns `NOT_FOUND`.
  - Module **must** return `INVALID_STATE` when Embedding path not Loaded.
- **Path selection attribution**: by model_executor based on whether om has built-in embedding (D-07 existing logic), not this module's self-decision.

#### 4.1.4 Type Conversion

- **Semantics**: type conversion is host-side **data representation change** (e.g. INT8 integer value expanded to FLOAT16 same-value float), **does not multiply scale** — multiplying scale is dequantization, always on NPU side.
- **Conversion matrix** (storage type → slot TensorDesc type):
  - Same type: direct byte copy.
  - INT8 → FLOAT16 / FLOAT32; FLOAT16 → FLOAT32: element-wise value expansion.
  - scale storage Fp16 → `embed_scales` slot FLOAT32: element-wise value expansion.
  - **Other combinations (including INT4, any conversion involving INT4/BF16) not supported in this version**, return `INVALID_ARGUMENT` (see open item U-3).
- Module **must** return `INVALID_ARGUMENT` when conversion target buffer capacity insufficient.

#### 4.1.5 Dual-Path Data Flow (End-to-End)

- **Embedding path**: `Load(table+scale, InMemory or InFlash)` → `FillInputs(Embedding, ids)`: internal gather low-precision rows (memory copy or file per-row read) → type conversion → fill `input_embed`; collect scale → fill `embed_scales` → NPU dequant → graph subsequent computation.
- **TokenId path**: embedding table **already included in model weights loaded with graph** (not through this module) → `FillInputs(TokenId, ids)`: token id fill `input_embed` (no prior Load needed) → NPU gather + dequant.

#### 4.1.6 Error Handling General Rules

- All failure paths return `Status` error codes from `infra/status.h` (`INVALID_ARGUMENT` / `INVALID_STATE` / `NOT_FOUND` / `FAILURE`), no exception cross-layer.
- NULL pointer input (`ids == nullptr` and `count > 0`, etc.) returns `INVALID_ARGUMENT`; empty `shared_ptr` output buffer (`Gather`'s `out_embed` / `out_scales`) and slot bundle empty buffer (`FillInputs`) also returns `INVALID_ARGUMENT`.
- After load failure, module **must** stay unloaded (can re-Load), no half-loaded state.
- InFlash mode table file runtime corruption after Load (truncation / I/O error / short read) returns `FAILURE` at `Gather` / `FillInputs`.

### 4.2 Interface Requirements

```cpp
// —— Types (engine internal, subset mapping of non-backend DataType) ——
enum class EmbeddingType     { Fp32, Fp16, Int8, Int4 };   // main table segment storage type
enum class EmbeddingScaleType{ None, Fp32, Fp16 };         // main scale segment storage type
enum class EmbeddingQuantMode{ PerToken, PerGroup };       // scale granularity
enum class EmbeddingPath     { TokenId, Embedding };       // FillInputs path selection
enum class EmbeddingLoadMode { InMemory, InFlash };        // table loading method: entire table into memory / stay in file read by row

struct EmbeddingConfig {
    std::string           table_path;   // embedding_weights single file (four segments: main table → identity table → main scale → identity scale)
    EmbeddingType         table_type;
    EmbeddingScaleType    scale_type;
    EmbeddingQuantMode    quant_mode;
    size_t                group_size;   // PerGroup required; PerToken ignored
    size_t                vocab_size;   // table row count (source ModelConfig, see U-2)
    size_t                hidden_dim;   // hidden dimension length (source ModelConfig, see U-2)
    EmbeddingLoadMode     load_mode = EmbeddingLoadMode::InMemory;  // table loading method (scale always in-memory)

    // per_layer_token_identity (gemma4): read types come from meta keys; row
    // width W is backfilled by pipeline from graph slot dims (not a meta field)
    EmbeddingType         token_identity_type = EmbeddingType::Fp16;
    EmbeddingScaleType    token_identity_scale_type = EmbeddingScaleType::Fp16;
    size_t                token_identity_row_width = 0;

    /// Assembly entry (SR20260903001: supersedes U-8 route b's FromModelConfig — D-12 direct parameterization):
    /// embedding_config section 7 strings + cross-section vocab/hidden → EmbeddingConfig (string→enum parse validation).
    /// weights empty → SUCCESS and table_path empty (caller doesn't Load based on this, tokenId deployment form).
    static Status FromSection(const std::string& weights,
                              const std::string& type, const std::string& scale_type,
                              const std::string& quant_mode, const std::string& load_mode,
                              const std::string& identity_type, const std::string& identity_scale_type,
                              size_t vocab_size, size_t hidden_dim, EmbeddingConfig* out);
};

class TextEmbedding {
public:
    TextEmbedding() = default;

    /// 4.1.1: load main-table segment + (optional) main scale segment (single-file four-segment layout).
    /// InMemory whole-table mmap into mapped pages; InFlash only validates file metadata and keeps file
    /// open, gather reads file directly by row offset; scale always in-memory.
    Status Load(const EmbeddingConfig& config);

    /// gemma4 per_layer_token_identity: second table source isomorphic to Load (identity table
    /// segment + identity scale segment, same single file); whole-file total length exact-equality
    /// against the four-segment sum.
    Status LoadTokenIdentity(const EmbeddingConfig& config);

    /// 4.1.2: embedding path host table lookup (copy low-precision rows by index + collect scale, no dequantization).
    /// InMemory copy from memory; InFlash pread by row offset.
    Status Gather(const TokenId* ids, size_t count,
                  std::shared_ptr<ITensorBuffer> out_embed,
                  std::shared_ptr<ITensorBuffer> out_scales) const;

    /// 4.1.3: proactively fill model input slots (approach B: depends on input bundle + name→index).
    /// TokenId path does not require prior Load (table in model weights); Embedding path requires Loaded.
    Status FillInputs(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                      const std::unordered_map<std::string, size_t>& name_index,
                      EmbeddingPath path,
                      const TokenId* ids, size_t count);

    /// Accessors (pipeline / UT use).
    EmbeddingType                  TableType() const;   // unloaded → Fp32 (sentinel) + Loaded() discriminates
    EmbeddingLoadMode              LoadMode() const;    // unloaded → InMemory (sentinel)
    bool                           Loaded() const;
    bool                           IdentityLoaded() const;
};
```

- `ModalityEntry` retained in same header file (multimodal placeholder, another responsibility of Preprocessor, this spec does not constrain its behavior).
- Old stub `LoadAndBind()` replaced by `Load` and deleted (see Chapter 7).

### 4.3 Timing

```
pipeline init (M-02, only embedding path needs):
  TextEmbedding emb;
  emb.Load(config)                     ← table segment (InMemory whole-table mmap / InFlash only open file+validate) + main scale segment (always in-memory)
  emb.LoadTokenIdentity(config)        ← identity table/scale segments (when graph carries per_layer slots; same single file, total length exact-equality)

each forward step (input preparation phase, actual caller is Scheduler's prefill/decode two input preparation points; before/parallel with InputAssembler filling other slots):
  emb.FillInputs(input_tensors_, input_name_index_, path, ids, count)
      ├─ Embedding path: gather (memory copy/file per-row read) → type conversion → fill input_embed + embed_scales
      └─ TokenId path:   fill token id (table in model weights, no Load needed)

UT / debug direct query:
  emb.Gather(ids, count, out_embed, out_scales)
```

Timing constraints: `Load` before all `Gather` / `FillInputs` in embedding path; tokenId path's `FillInputs` has no Load prerequisite. `FillInputs` and Forward relative timing belongs to model_executor input preparation flow (`INPUT_INFO.md` §3), this module is not aware. InFlash mode table file **must** remain accessible during module lifecycle (Load opens, destructor closes).

## Chapter 5 Non-Functional Requirements

| Dimension | Metric | Test Condition | Evidence Source |
|------|------|----------|----------|
| gather throughput | InMemory host gather reaches reasonable proportion of contiguous memcpy bandwidth (budget target, no measured baseline, open U-5) | aarch64, INT8 table `[65536, 4096]`, batch gather timing | budget target |
| InFlash gather latency | single batch gather (≤ max_seqLen rows) file read latency acceptable (budget target, no measured baseline, open U-5; batch random reads can do offset sorting optimization, see 4.1.2) | aarch64, same table scale InFlash mode, vs InMemory baseline | budget target |
| table load memory | InMemory resident ≈ total single-file length (mmap read-only mapped pages; with identity enabled covers all four segments) + scale heap memory; **InFlash resident ≈ scale bytes + file descriptor** (table zero resident) | RSS measurement after load | design constraint (single resident / zero resident) |
| thread safety | same instance `Gather` (const) can be concurrent (InFlash via `pread`-like positional read, shared fd also concurrent); `Load` not concurrent, `Load` not concurrent with `Gather`/`FillInputs` | concurrency test cases TC_3xx | design constraint |
| error observability | all failure paths return error codes, no silent swallowing, no exception escape | exception TC_1xx | behavior contract |
| portability | g++ 13 / C++17, aarch64 (Ascend host) passes | CI compile + GoogleTest | environmental fact |

Self-check: each item measurable, verifiable; no N/A items; open items registered at end of document.

## Chapter 6 Constraints and Dependencies

- **Hardware dependencies**: host-side pure CPU (file read + memcpy + type conversion); no NPU operator dependency (dequantization in graph).
- **Software dependencies**: C++17 (g++ 13 verified), GoogleTest (UT). Does not depend on ACL/NDK concrete backend class names — only interacts via `backend/api/tensor_buffer.h`'s `ITensorBuffer` / `TensorDesc` abstraction (consistent with model_executor v0.4 "backend compile-time determined" principle).
- **Cross-module interactions**:
  - `model_executor`: consumes `FillInputs` (input preparation phase); provides input bundle and name→index; path selection (D-07).
  - `pipeline`: M-02 calls `Load` at init phase; `PipelineExecutorFactory::create` passes `TextEmbedding` by value (interface not broken); `Scheduler` holds non-owning `TextEmbedding*` and calls `FillInputs` at input preparation phase (constructor signature adds pointer parameter, see 3.4 and DESIGN §2.2 wiring point).
  - `Preprocessor`: holds non-owning `TextEmbedding*`; `ModalityEntry` consumer.
  - `INPUT_INFO.md`: 6 slots source of truth; `embed_scales` filler = this module (landing this time).
- Self-check: no implicit dependencies; explicitly does not depend on dynamic library hot unload, does not do NPU memory management.

## Chapter 7 Backward Compatibility

- This module is engine-internal C++ component, does not expose external interfaces, no external API binary compatibility burden.
- **Internal interface change**: `LoadAndBind()` stub replaced by `Load(const EmbeddingConfig&)`. Current repo-wide consumption of this symbol is zero-value (stub always returns `FAILURE`), replacement has no behavior regression risk; `PipelineExecutorFactory` / `Preprocessor` holding and pass-by-value unchanged; `Scheduler` constructor signature adds `TextEmbedding*` (see 3.4, wiring change).
- **Behavior alignment**: `embed_scales` slot goes from "pending" to "this module fills" (embedding path), is INPUT_INFO's designated filler landing, not semantic change.
- Self-check: framework layer retains this chapter; no external interface compatibility burden.

## Chapter 8 Verification Standards

> Numbering: TC_0xx normal / TC_1xx exception / TC_2xx boundary / TC_3xx concurrency. UT in `tests/cpp/engine/`, GoogleTest; NPU-side dequantization results not in host UT scope (backend dependency), UT only verifies host-side loading / lookup / conversion / slot filling.

### Normal Paths (TC_0xx)

**TC_001 PerToken load success** (related 4.1.1; category: normal path)
- Given INT8 table segment `[vocab=16, hidden=8]` + Fp32 main scale segment `[16]` (single file) + PerToken config
- When `Load(config)`
- Then returns SUCCESS; `Loaded()` true, `TableType()` Int8; full 16-row gather and main scale segment byte-for-byte hit (behavior-level equivalent, replacing former Table()/Scale() size assertions)

**TC_002 PerGroup load success** (related 4.1.1; category: normal path)
- Given INT8 table segment `[16, 8]` + Fp32 main scale segment `[16 × 4]` (own two-segment file, W=0; group_size=2, groups=4) + PerGroup config
- When `Load(config)`
- Then returns SUCCESS; Gather of last vocab entry collects last 4 group scales + last row, byte-for-byte matching the fixture tail (behavior-level equivalent)

**TC_003 PerToken gather correctness** (related 4.1.2; category: normal path)
- Given TC_001 loaded, ids = {3, 0, 15}
- When `Gather(ids, 3, out_embed, out_scales)`
- Then `out_embed` byte-for-byte equals table's rows 3/0/15 concatenated; `out_scales` equals scale[3]/scale[0]/scale[15]

**TC_004 PerGroup gather scale collection** (related 4.1.2; category: normal path)
- Given TC_002 loaded, ids = {5}
- When `Gather(&id, 1, out_embed, out_scales)`
- Then `out_scales`'s 4 elements sequentially equal scale segment indices `5×4+0..3` values

**TC_005 Embedding path FillInputs fills correct slots** (related 4.1.3; category: normal path)
- Given loaded (INT8 storage), mock input bundle contains `input_embed` (desc FLOAT16 `[3, 8]`) and `embed_scales` (desc FLOAT32 `[3]`), name→index mapping
- When `FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids={3,0,15}, 3)`
- Then `input_embed` data is table's rows 3/0/15 INT8 values expanded to FLOAT16; `embed_scales` is scale[3]/scale[0]/scale[15] as FLOAT32

**TC_006 TokenId path FillInputs (no Load needed)** (related 4.1.3; category: normal path)
- Given mock `input_embed` desc INT32 `[3]`, module **not loaded**
- When `FillInputs(..., EmbeddingPath::TokenId, ids={3,0,15}, 3)`
- Then `input_embed` data = {3, 0, 15} INT32 representation (table in model weights, not through this module)

**TC_010 TokenId path INT64 slot** (related 4.1.3; category: normal path)
- Given mock `input_embed` desc INT64 `[3]`, module **not loaded**
- When `FillInputs(..., EmbeddingPath::TokenId, ids={3,0,15}, 3)`
- Then `input_embed` data = {3, 0, 15} INT64 representation (uint32→INT64 value expansion)

**TC_007 Same-type no-conversion direct fill** (related 4.1.4; category: normal path)
- Given storage type FLOAT16, slot desc FLOAT16
- When FillInputs (Embedding path)
- Then data byte-for-byte equals gathered original rows (no conversion path hit)

**TC_008 InFlash load and gather correctness** (related 4.1.1/4.1.2; category: normal path)
- Given TC_001 same table file and config, but `load_mode = InFlash`
- When `Load(config)` then `Gather(ids={3,0,15}, 3, out_embed, out_scales)`
- Then returns SUCCESS; `LoadMode()` is InFlash, `Loaded()` true; `out_embed`/`out_scales` byte-for-byte identical to TC_003 (InMemory); after load, truncating the table file to the main-table-segment end, Gather still succeeds (rows go through fd pread, scale already read into heap memory at Load time)

**TC_009 InFlash mode FillInputs (Embedding path)** (related 4.1.3; category: normal path)
- Given InFlash loaded (INT8 storage), mock `input_embed` desc FLOAT16 + `embed_scales` desc FLOAT32
- When `FillInputs(..., EmbeddingPath::Embedding, ids, 3)`
- Then both slots' data byte-for-byte identical to TC_005 (InMemory same config)

### Exception Paths (TC_1xx)

**TC_101 table file missing** (related 4.1.1) → `Load` returns `INVALID_ARGUMENT`, `Loaded()` false, can re-Load successfully.
**TC_102 file size mismatch** (related 4.1.1) → main table segment short (rejected before mapping) or main scale segment short read, `INVALID_ARGUMENT`; trailing bytes beyond later segments pass.
**TC_103 PerGroup params illegal** (related 4.1.1) → group_size=0 / not provided / not divisible by hidden, each returns `INVALID_ARGUMENT`.
**TC_105 tokenId out of bounds** (related 4.1.2) → any id ≥ vocab → `INVALID_ARGUMENT`, `out_embed` not partially written.
**TC_106 query before load** (related 4.1.2/4.1.3) → `Gather` and `FillInputs(Embedding, ...)` return `INVALID_STATE`; `FillInputs(TokenId, ...)` unaffected (see TC_006).
**TC_107 slot missing** (related 4.1.3) → name→index no `input_embed` or (Embedding path and scale_type≠None) no `embed_scales` → `NOT_FOUND`.
**TC_108 unsupported conversion combination** (related 4.1.4) → storage INT4 → slot FLOAT16 → `INVALID_ARGUMENT`.
**TC_109 output capacity insufficient** (related 4.1.2/4.1.4) → `out_embed`/`out_scales` GetSize too small → `INVALID_ARGUMENT`.
**TC_110 NULL input** (related 4.1.6) → `ids=nullptr, count>0` → `INVALID_ARGUMENT`.
**TC_111 InFlash runtime file corruption** (related 4.1.6) → InFlash loaded then truncate table file, `Gather` / `FillInputs` (Embedding path) returns `FAILURE`.
**TC_112 empty buffer input** (related 4.1.6) → `Gather` passes empty `shared_ptr` `out_embed`/`out_scales`; `FillInputs` passes slot bundle with empty buffer → `INVALID_ARGUMENT`, no dereference crash.
**TC_113 FromSection unknown enum string** (related 4.1.1/4.2, SR20260903001 D-12) → `embedding_config` section's type key = `"fp8"` etc. unknown value → `INVALID_ARGUMENT`; `weights` empty → SUCCESS and `table_path` empty (no-load signal).

### Boundary (TC_2xx)

**TC_201 count=0** (related 4.1.2) → `Gather(ids, 0, ...)` returns SUCCESS, writes no bytes.
**TC_202 scale_type=None** (related 4.1.1/4.1.2) → no main scale segment load success; Gather skips `out_scales` returns SUCCESS; FillInputs skips `embed_scales` (handling see U-6).
**TC_203 vocab boundary id** (related 4.1.2) → id = vocab-1 success, id = vocab rejected.
**TC_204 load failure recoverable** (related 4.1.6) → failed Load then fix config and Load successfully.

### Concurrency (TC_3xx)

**TC_301 concurrent Gather (InMemory)** (related Chapter 5) → 4 threads concurrent Gather same/different ids on same loaded instance, results consistent with serial, no crash (optional TSan).
**TC_302 concurrent Gather (InFlash)** (related Chapter 5) → 4 threads share same InFlash instance (shared fd) concurrent Gather, results consistent with serial, no crash; positional reads must not interfere with each other's cursor.

## Open Items Register

| ID | Open Item | Category | Blocks spec release | Resolution Phase |
|----|--------|------|----------------|----------|
| ~~U-1~~ | ~~tokenId path table/scale into graph channel~~ **resolved (2026-08-28 user confirmed)**: tokenId path needs no extra embedding table — **table already included in model weights loaded with graph**, this module only fills token id | system boundary | no (resolved) | merged into 4.1.1/4.1.3/4.1.5 |
| U-2 | authoritative source of `vocab_size` / `hidden_dim` (this spec assumes Load input parameter, from ModelConfig). After U-8 decision source decided = `ModelConfig` extension field (landed together with embedding channel), only field naming/type alignment remains | interface contract | no | implementation phase with ModelConfig extension landing |
| U-3 | INT4 / BF16 storage type: backend `DataType` has no corresponding enum, INT4 needs bit unpacking — not supported in this version | functional scope | no (enum placeholder, conversion matrix explicit exclusion) | upgrade structure approach when real demand (polymorphic/strategy) |
| ~~U-4~~ | ~~`INPUT_INFO.md` correction~~ **completed (2026-08-29 with implementation landing, INPUT_INFO v0.4)**: `embed_scales` dimensions added PerGroup `×groups`; `input_embed` dtype changed to "per TensorDesc, varies by path"; filler/flow description synchronized with FillInputs wiring | documentation consistency | no (completed) | closed |
| U-5 | gather throughput / InFlash read latency / load memory budget no measured baseline | acceptance criteria | no (budget target) | implementation phase micro-benchmark |
| U-6 | `embed_scales` slot handling when `scale_type=None` (don't fill vs fill 1.0) | behavior contract | no (this spec takes "skip, don't fill") | confirm when first no-scale model integrated |
| U-7 | InFlash mode table file lifecycle: must remain accessible during module lifecycle (4.3 already constrained); whether file replacement/hot update scenarios need support | deployment constraint | no (this spec doesn't support hot update, corruption goes to TC_111's FAILURE path) | extend when hot update demand arises |
| ~~U-8~~ | ~~`EmbeddingConfig` user→pipeline transfer channel unresolved~~ **resolved (2026-08-29 user decision, route b); later partially superseded by SR20260903001 (2026-09-04 user decision)**: channel changed to meta_json `embedding_config` section → pipeline inline parsing step directly calls `EmbeddingConfig::FromSection` (8 parameters direct, D-12; ModelConfig no longer carries embedding extension fields); vocab/hidden taken from model_config section across sections. Parsing and validation semantics maintained per this spec | interface contract | no (resolved) | 4.1.1 and DESIGN C-01/F-01 synchronized |

**Blocking-level open items cleared**: U-1 resolved (table in weights); U-8 resolved (channel via meta_json→ModelConfig, 2026-08-29); U-2/U-4 are alignment-type (U-2 source decided, only field alignment remains); U-3/U-5/U-6/U-7 are scope/budget/deployment type. No trust/authorization boundary, no security redline related open items. **All open items no longer block implementation entry.**

## Consistency Self-Check (Generation Phase)

- Interface contract coverage: 4.2 each semantic interface traceable to 4.1 behavior (Load↔4.1.1, Gather↔4.1.2, FillInputs↔4.1.3, accessors↔4.1.1). ✓
- Test coverage: 4.1 each "must" has TC in Chapter 8 (load/lookup/slot fill/conversion/error/boundary/concurrency). ✓
- Terminology consistency: all document terms defined in Chapter 2 (including PerToken/PerGroup/storage type/fill type distinction). ✓
- Timing consistency: 4.3 timing consistent with 4.1 and INPUT_INFO §3. ✓
- Layer adaptation: framework layer includes Chapter 7. ✓
- Decision traceability: 3.3 decision table each corresponds to brainstorm session (2026-08-27) and user confirmation (2026-08-28). ✓
