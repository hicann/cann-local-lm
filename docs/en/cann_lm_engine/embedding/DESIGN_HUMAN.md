<!--
Human-readable version of the design document. Content is equivalent to DESIGN.md, but narrative-first with numbering as a supplement.
The machine-readable authoritative version remains ./DESIGN.md (with complete ID cross-references).
v0.1: Corresponds to SPEC SR20260828001 initial draft (including U-1 resolved: TokenId path table is in weights).
v0.2: External review (+2, 2026-08-29) release condition revisions — Scheduler wiring point, U-8 channel registration,
      OHOS CMake registration and several L-level corrections; technical decisions (D-01~D-11) unchanged.
-->

# TextEmbedding Module Design (Human-Readable Version)

English | [简体中文](../../../zh/cann_lm_engine/embedding/DESIGN_HUMAN.md)

> Companion: `SPEC.md` (behavior specification), `DESIGN.md` (machine-readable authoritative design, with complete numbered cross-references).
> This document is a narrative rewrite of DESIGN.md for easy reading; technical decisions are completely consistent.

## What This Module Does

In one sentence: **It is the "embedding porter" in the inference engine — loading low-precision embedding tables and dequantization parameters from the file system (or simply leaving them in the file and reading on demand), looking up the corresponding rows by token id before forward pass, and filling the results into the model's input slots in the type required by the om graph. It only moves bytes, only does value representation expansion, and never performs dequantization — the scale multiplication step happens on the NPU.**

It sits between `pipelineExecutor` (upper-level scheduler) and the om graph input slots (`input_embed` / `embed_scales`), doing three things:

1. **Load**: Read embedding table files and scale files. The table can be read entirely into memory (InMemory), or not loaded into memory, keeping a file handle and reading on demand each time (InFlash).
2. **Gather**: Given a set of token ids, copy the corresponding low-precision rows from the table by row offset, and simultaneously collect each token's dequantization parameters (scale).
3. **FillInputs**: Type-convert the gathered rows and write them into the `input_embed` slot, and write scales into the `embed_scales` slot — the write type is determined by the slot's own `TensorDesc`, not by the file storage type.

It is an engine-internal C++ component, consumed **only** by pipeline / model_executor, and does not expose any bindings externally.

## Four Key Points for Understanding This Module

### 1. Host Only Moves Bytes, Dequantization Always on NPU

This is the foundation of all decisions. The embedding table is low-precision (INT8, etc.), and with scale it can be restored to high precision. What this module does on the host side — copying rows, collecting scales, type conversion — is all **byte moving and value representation expansion**: INT8's 5 becomes FLOAT16's 5.0, the value doesn't change, nothing is multiplied. Multiplying by scale (actual dequantization) happens inside the om graph on the NPU. So host-side gather is storage-type-agnostic (pure stride copy), and type conversion is quantization-agnostic (pure value expansion) — these two "irrelevances" make the implementation very thin.

### 2. Dual Path: Whether the Table Is in Weights Determines How Much This Module Does

- **tokenId path** (om graph has built-in embedding): **The table is already included in the model weights and loaded with the graph**. This module does nothing to load; `FillInputs` just fills token ids into the `input_embed` slot and is done — it doesn't even require a prior Load. The NPU does its own table lookup and dequantization inside the graph.
- **embedding path** (om graph has no built-in embedding): This module's full home ground — Load table → host gather → fill `input_embed` + `embed_scales` → NPU dequantize.

Which path to take is not decided by this module itself, but by model_executor based on whether the om graph has built-in embedding (that's D-07, already accepted in the model_executor design).

### 3. What Type to Fill, the Slot Decides

The `EmbeddingType` (Fp32/Fp16/Int8) specified by the user at Load time describes the **byte layout of the file on disk**; while the actual data type written into the `input_embed` slot is taken from that slot's `TensorDesc.dataType` (whatever the om graph declares). When the two don't match, the module does type conversion before filling (element-wise value expansion). Same type means direct copy. Unsupported combinations (INT4, BF16, etc.) are explicitly rejected, returning `INVALID_ARGUMENT`.

### 4. Quantization Granularity: PerToken vs PerGroup

The shape of the scale file is determined by the initialization parameter `quantMode`:

- **PerToken**: One scale per vocabulary entry (one row of the table), so the file has `[vocab]` values.
- **PerGroup**: One row of embedding is sliced into several groups along the hidden dimension (group size = `GroupSize`, required and must evenly divide hidden_dim), **each group has its own scale** — one tokenId corresponds to multiple dequantization parameters. File shape `[vocab × (hidden/group_size)]`, token v's g-th group scale is at index `v×groups+g`.

Accordingly, when Gather collects out_scales: PerToken collects 1 per token, PerGroup collects `groups` per token.

## Structure: One Public Class, Two Internal Helpers

Externally there is only one concrete class `TextEmbedding` (the original brainstorm chose "single class with type tag dispatch" over "polymorphic hierarchy" — because host-side logic is type-agnostic, polymorphism has no landing point). Internally it splits into two helpers that don't appear in the public header:

- **TableSource (M-02)**: Table data source strategy. The unified interface is just one `ReadRow(id, dst)` — give me a token id, I put that row of bytes into your buffer. Two implementations:
  - `MemoryTableSource`: Entire table in memory buffer, ReadRow is memcpy.
  - `FlashTableSource`: Only has fd and row width, ReadRow is a single `pread` (reads at `id × row_bytes` offset, doesn't move file cursor, naturally multi-thread concurrent).
  - This is the first landing point for "reserved strategy hooks": future INT4 bit unpacking and other differentiated logic extends at this layer.
- **TypeConverter (M-03)**: Pure function set, a support matrix (storage type × slot dtype), same-type memcpy, different-type value expansion, rest rejected. Stateless, no dependencies.

All three land in the `core/component/text_embedding.{h,cpp}` single file pair. The `ModalityEntry` (multimodal placeholder entry) in the same header file is the neighboring Preprocessor's responsibility, kept in place unchanged.

## Trade-offs Between Two Loading Methods

| | InMemory (default) | InFlash |
|---|---|---|
| Table resident in memory | ≈ table file bytes | 0 (only scale + one fd) |
| gather cost | memcpy | one pread syscall per row |
| Suitable for | throughput-first (server ACL side) | memory-constrained (device-side OHOS) |

Both are **runtime configurations** (`EmbeddingConfig.load_mode`), not compile-time either-or — the same build artifact can run both. Scale is always in-memory regardless of mode: it has only vocab (or vocab×groups) scalars, much smaller than the table, and is randomly accessed by index on every gather, so keeping it in a file is not cost-effective.

InFlash's cost is file lifecycle: after Load, the table file must always be there (the module holds the fd until destruction). If the file is truncated or replaced mid-way, `pread` short-reads on gather, returning `FAILURE`. Hot update is not supported — address when needed (SPEC open item U-7).

## Interface

```cpp
Status Load(const EmbeddingConfig& config);

Status Gather(const TokenId* ids, size_t count,
              std::shared_ptr<ITensorBuffer> out_embed,
              std::shared_ptr<ITensorBuffer> out_scales) const;   // concurrent-safe

Status FillInputs(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                  const std::unordered_map<std::string, size_t>& name_index,
                  EmbeddingPath path,
                  const TokenId* ids, size_t count);

// Accessors: Table() / Scale() / TableType() / LoadMode() / Loaded()
```

`EmbeddingConfig`: two paths, two types, quantization granularity + group_size, vocab/hidden (from ModelConfig), load_mode.

About `FillInputs` parameters — it takes model_executor's input slot array (`input_tensors_`) and name→index mapping (`input_name_index_`). For this, `ModelExecutor` adds two **read-only** accessors (`InputTensors()` / `InputNameIndex()`), an additive change that doesn't touch existing interfaces. This module only writes to slots named `input_embed` and `embed_scales`; out-of-bounds writes are defects — it doesn't interfere with the other three slots written by InputAssembler (position_ids / new_kv_cache_pos / attention_mask).

## A Complete Forward Pass (Embedding Path)

```
init phase:   pipeline gets ModelConfig (meta_json deserialization, existing plan owned by ModelConfig)
               → EmbeddingConfig::FromModelConfig(mc, &config)   ← U-8 resolved: route b
               → only if table_path non-empty: text_embedding_.Load(config)
              (InMemory: entire table+scale read into memory; InFlash: open+fstat verify+keep fd)
              (table_path empty = tokenId deployment form, skip Load — before meta_json deserialization lands
               ModelConfig fields default to empty, compatible with current behavior)

each forward step:  input preparation phase — specifically happens at two input preparation points within Scheduler::Step
              (prefill: take next chunk from queue; decode: Preprocessor.PrepareDecodeStep each active request's last token):
                text_embedding_->FillInputs(executor.InputTensors(),
                                             executor.InputNameIndex(),
                                             path, ids, count)
                  ├─ Embedding path: per-token ReadRow (memcpy or pread) → type convert → write input_embed
                  │                 (ids from chunk / decode input, count = number of rows)
                  └─ collect scale per token → write embed_scales
              (simultaneously InputAssembler fills the other three slots; KV pool slots bound at Load)
              executor.Forward(opts) → NPU dequantize + graph compute → logits
```

To clarify where the "input preparation phase" actually is: **RunLoop itself does not fill inputs**, it only loops calling `scheduler_.Step()`; in the current code, the sole writer of `input_embed` is the two `SetInputData` calls in Scheduler (one for prefill, one for decode). So the wiring lands at these two points: `Scheduler` constructor adds a `TextEmbedding*` non-owning pointer (same pattern as the `Preprocessor*` it already holds, pipeline passes `&text_embedding_` in), and the two input preparation points replace `SetInputData(chunk)` with `FillInputs(...)` — the tokenId path is also replaced (the module conveniently does INT32/INT64 value conversion, replacing bare memcpy). After replacement, `SetInputData` has zero calls in the repo, kept as "external direct-fill channel" API, not deleted. **Within the same iteration, `input_embed` is only allowed to be written once** — `SetInputData` and `FillInputs` must not both be called in sequence; this is a hard rule (otherwise the later write overwrites the earlier, behavior depends on call order).

The tokenId path is simple: `FillInputs(..., TokenId, ids, count)` converts ids to the slot dtype (INT32/INT64) and writes to `input_embed`; the rest is handled by in-graph weights.

## Concurrency Model (and Why There Are No Locks)

After Load, all the module's members (table buffer or fd, scale buffer, config snapshot, row width) are all read-only. Read-only data concurrent access doesn't need locks; InFlash's `pread` is positional read, multiple threads share one fd each reading their own offset, POSIX guarantees safety. So this module **has no locks** — the contract has only two rules: don't call Load concurrently (naturally serial during init), don't interleave Load and queries. `Gather` is const and can be freely concurrent; `FillInputs` writes to slot buffers, which inherently live in the single-threaded domain of the input preparation phase.

## Error Handling

All go through `infra/status.h`'s six coarse codes, no exceptions thrown:

- **INVALID_ARGUMENT**: file missing/wrong size, PerGroup parameters illegal, scale config self-contradictory (type=None but path given), token id out of bounds (reject entire batch, no partial output), output capacity insufficient, NULL input, unsupported type conversion combination.
- **INVALID_STATE**: embedding path Gather/FillInputs called without Load (tokenId path exempt).
- **NOT_FOUND**: name→index doesn't have `input_embed` (or `embed_scales` when scale is expected).
- **FAILURE**: InFlash runtime file corruption (pread short-read / I/O error); TensorManager buffer allocation failure.

Load failure keeps the unloaded state; after fixing config, Load can be retried; no half-loaded state.

## How Testing Is Done

GoogleTest, in `tests/cpp/engine/`, UT target `ut_text_embedding`. Core approach: **NPU-side dequantization is not in UT scope** (backend dependency), UT only verifies three things on the host side — loading, table lookup, slot filling. SPEC Chapter 8 has the complete TC list (TC_001–TC_302), a few worth mentioning:

- **InFlash vs InMemory consistency** (TC_008/009): Same table file loaded in both modes, gather/fill results byte-for-byte identical — this is the anchor for InFlash correctness.
- **Type conversion matrix**: Each supported combination verified element-by-element for value correctness (including negative INT8, Fp16 denormalized values); each unsupported combination verified for rejection.
- **mock slot bundle**: Directly use `CreateTensorBuffer` to build input bundle + name_index by desc, no need to mock TensorManager (stub backend has no side effects, use real singleton).
- **Concurrency** (TC_301/302): 4 threads concurrent Gather, results consistent with serial; InFlash shared fd concurrent pread doesn't interfere.
- **grep assertion**: `text_embedding.{h,cpp}` must not contain concrete backend class names.

## Connection with Current State (Migration Notes)

The current `text_embedding.h/.cpp` is a stub era stub: `LoadAndBind()` always returns FAILURE. This is an in-place rewrite: `LoadAndBind` is deleted (after confirming zero-value calls via repo-wide grep), replaced by the `Load/Gather/FillInputs` trio. Upstream and downstream are basically unchanged: `PipelineExecutorFactory::create` still passes `TextEmbedding` by value, `Preprocessor` still holds a non-owning pointer, `ModalityEntry` kept in place; **the only exception is the `Scheduler` constructor signature adding `TextEmbedding*`** (wiring needed, see "input preparation phase" section above). When registering UT, remember to hook both the main CMakeLists and `ohos/CMakeLists.txt` **on both sides** — the OHOS side uses explicit per-target copy; if you add to the main chain but not OHOS, this test won't run on HarmonyOS (`ut_sampler`/`ut_request_handle` in the repo have already drifted this way once, don't add another).

One existing issue to fix along the way: `INPUT_INFO.md` (the source of truth for model_executor input slots) records `embed_scales` dimension as `batch×seqLen×1` which only holds for PerToken (PerGroup should be `×groups`), and `input_embed` recorded as FLOAT16 is a stub placeholder (actually follows TensorDesc, varies by path). Fix together with the implementation PR (SPEC open item U-4).

## What Gaps Were Left (Open Items Quick View)

- **U-2**: The source field names for vocab/hidden, align with ModelConfig at implementation time (input form already defined).
- **U-3**: INT4/BF16 not supported in this version — real demand is the trigger condition for M-02/M-03 strategy upgrade.
- **U-5**: Throughput/latency/memory budgets are all budget targets, micro-benchmark calibration at implementation time.
- **U-6**: Models without scale, `embed_scales` slot skipped (current approach); confirm when first no-scale model is integrated.
- **U-7**: InFlash files don't support hot update, corruption goes to FAILURE.
- ~~**U-8** (added 2026-08-29 external review)~~: `EmbeddingConfig` channel from user config to pipeline — **resolved (2026-08-29 user decision) route b**: meta_json → `ModelConfig` extends embedding section (plain string/integer fields, serialization-friendly) → `EmbeddingConfig::FromModelConfig` assembles (string→enum parsing and validation in this module, dependency direction is legal). L1 `EngineOptions`/C ABI untouched. vocab/hidden also placed in ModelConfig (U-2 source is thus settled).
- New implementation-phase review item: table/scale buffer allocated via `TensorManager::CreateTensor` with `modelId=0` sentinel semantics, confirm at real backend implementation that host buffer is not affected by modelId binding.
