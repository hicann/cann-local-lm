# Gemma4 / Qwen3.5 Lightweight Performance and Numerical Regression Log

Fixed capacity 5120, batch=1, BF16; inputs of 128/512/2048 tokens, with 32 output tokens per run.
Trajectories are taken from the first greedy result of the initial PTE; subsequent eager/PTE runs are fed the same prefix, and output does not stop early on EOS.
Each case has 1 warmup and 3 measurements, reporting the median; model loading, file reading, logging, and logits snapshots are not timed.
TTFT includes request state initialization, mask, data movement, prefill, and token selection; total latency is the sum of the per-step times after removing snapshot overhead.
TPOT=(total latency−TTFT)/31; throughput=32/total latency in seconds. The raw wall_with_capture_ms is retained for auditing.
These are end-to-end inference metrics on fixed trajectories; they do not include tokenization and are not serving concurrency throughput.
Cosine compares the final logits over the full vocabulary; step=0 is prefill. Layer-by-layer checks and MMLU are not part of this quick set.
The initial eager run uses the same model code, static capacity, and stateful decode as the PTE; the PTE uses standard export.
Each command runs both models sequentially by default; please keep the NPU exclusive. Numerical WARN is for manual review and does not automatically overwrite the baseline.

## 2026-09-30T03:25:31.834669+00:00 — baseline — gemma4

Results directory: `/mnt/workspace/executorch-npu-eval/light_baseline_20260930_v2/gemma4`; see `metadata.json` for configuration/commit/artifact fingerprints.

| Case | Backend | Min cosine vs eager | Min cosine vs initial PTE | top-1 agreement vs PTE | TTFT ms | TPOT ms | Total latency ms | tokens/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99988709 | 1.00000000 | 100.00% | 107.21 | 30.60 | 1053.48 | 30.375 | BASELINE |
| short (128) | eager | 1.00000000 | 0.99988709 | 100.00% | 213.99 | 200.50 | 6429.48 | 4.977 | BASELINE |
| medium (512) | pte | 0.99994653 | 1.00000000 | 100.00% | 116.16 | 30.35 | 1056.88 | 30.278 | BASELINE |
| medium (512) | eager | 1.00000000 | 0.99994653 | 96.88% | 211.38 | 194.80 | 6248.66 | 5.121 | BASELINE |
| long (2048) | pte | 0.99997342 | 1.00000000 | 100.00% | 289.70 | 31.29 | 1259.04 | 25.416 | BASELINE |
| long (2048) | eager | 1.00000000 | 0.99997342 | 100.00% | 247.81 | 192.62 | 6220.21 | 5.145 | BASELINE |

## 2026-09-30T03:41:14.565549+00:00 — baseline — qwen3_5

Results directory: `/mnt/workspace/executorch-npu-eval/light_baseline_20260930_v2/qwen3_5`; see `metadata.json` for configuration/commit/artifact fingerprints.

| Case | Backend | Min cosine vs eager | Min cosine vs initial PTE | top-1 agreement vs PTE | TTFT ms | TPOT ms | Total latency ms | tokens/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99985068 | 1.00000000 | 100.00% | 1058.08 | 12.40 | 1442.69 | 22.181 | BASELINE |
| short (128) | eager | 1.00000000 | 0.99985068 | 96.88% | 1731.52 | 126.86 | 5664.11 | 5.650 | BASELINE |
| medium (512) | pte | 0.99957259 | 1.00000000 | 100.00% | 1696.56 | 12.28 | 2076.90 | 15.408 | BASELINE |
| medium (512) | eager | 1.00000000 | 0.99957259 | 100.00% | 1612.48 | 127.80 | 5574.33 | 5.741 | BASELINE |
| long (2048) | pte | 0.99985404 | 1.00000000 | 100.00% | 4265.02 | 12.87 | 4668.01 | 6.855 | BASELINE |
| long (2048) | eager | 1.00000000 | 0.99985404 | 100.00% | 1676.02 | 126.31 | 5591.71 | 5.723 | BASELINE |

## 2026-09-30T03:44:00.448941+00:00 — Baseline replay (no optimization) — gemma4

Results directory: `/mnt/workspace/executorch-npu-eval/light_replay_20260930/gemma4`; see `metadata.json` for configuration/commit/artifact fingerprints.

| Case | Backend | Min cosine vs eager | Min cosine vs initial PTE | top-1 agreement vs PTE | TTFT ms | TPOT ms | Total latency ms | tokens/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99988709 | 1.00000000 | 100.00% | 80.34 | 30.73 | 1035.09 | 30.915 | PASS |
| medium (512) | pte | 0.99994653 | 1.00000000 | 100.00% | 118.54 | 30.81 | 1075.32 | 29.759 | PASS |
| long (2048) | pte | 0.99997342 | 1.00000000 | 100.00% | 286.13 | 32.89 | 1307.89 | 24.467 | PASS |


## Validation and current fluctuation range

Validated through the runner build, 4 comparison-logic tests, and actual runs covering 3 cases for each of the two models with 3 measurements per case.
For Gemma4, running the three cases consecutively from a single load versus loading them separately produced identical trajectories and full logits, verifying state isolation between cases.
In the Gemma4 no-optimization replay, the model/PTE/PTD/runner fingerprints and the evaluation source fingerprint were unchanged; all three cases PASSed, and the logits were identical to the initial PTE.

With the same code and the same artifacts, the total-latency fluctuation in this run is as follows (not an optimization gain):

- short: total latency change -1.75%.
- medium: total latency change +1.74%.
- long: total latency change +3.88%.

Short-input TTFT went from 107.21 ms to 80.34 ms, showing clear fluctuation between individual runs. The above is the observed range of a single replay, not a statistical confidence interval; small gains of the same magnitude should be re-measured. PASS/WARN in the tables only indicates the numerical regression verdict, not a performance improvement.

The initial eager and PTE are not exactly equal token by token: Gemma4 medium and Qwen3.5 short each differ in top-1 on 1/32 steps; these initial differences are retained, and subsequent checks mainly look for new drift relative to the initial PTE.

Historically, a single Qwen3.5 PTE load took about 13.5 minutes, and a similarly long load wait was observed in this run. The evaluation loads each model once and runs all three cases; load time is not mixed into TTFT/TPOT/total latency/output throughput.
## 2026-09-30T07:51:11.824141+00:00 — GDN stage 1: fused prefill and BF16 state — qwen3_5

Results directory: `/mnt/workspace/executorch-npu-eval/qwen_gdn_stage01_20260930/regression/qwen3_5`; see `metadata.json` for configuration/commit/artifact fingerprints.

| Case | Backend | Min cosine vs eager | Min cosine vs initial PTE | top-1 agreement vs PTE | TTFT ms | TPOT ms | Total latency ms | tokens/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99984300 | 0.99984394 | 100.00% | 84.42 | 12.05 | 459.50 | 69.641 | WARN |
| medium (512) | pte | 0.99960461 | 0.99979134 | 100.00% | 60.51 | 11.60 | 420.24 | 76.147 | WARN |
| long (2048) | pte | 0.99974154 | 0.99975077 | 100.00% | 131.77 | 11.70 | 494.37 | 64.729 | WARN |


## GDN stage 1: export cost and error breakdown

Environment: executorch_npu; loads CANN 9.1.0 and `/mnt/workspace/libtorch_delegate/set_env.sh`. The fusion switch is off by default, and the original baseline is not overwritten.

The export times below are compared against historical records; the old full export was not re-run, so this is not a single controlled A/B:

| Item | Historical reference s | This run s |
|---|---:|---:|
| PT2 prefill capture | 263.61 | 20.04 |
| PT2 prefill save | 158.05 | 14.85 |
| PT2 total | 450.29 | 63.05 |
| PTE total | 13863.54 | 666.64 |

Prefill new-graph statistics: `{"aten.matmul.default": 13, "npu.npu_chunk_gated_delta_rule.default": 18}`.

Numerical breakdown (minimum cosine per case):

| Case | PTE vs fused eager | Fused eager vs BF16 state only | Fused eager vs initial eager |
|---|---:|---:|---:|
| short | 0.99984300 | 0.99983443 | 0.99979635 |
| medium | 0.99967152 | 0.99981586 | 0.99981846 |
| long | 0.99965519 | 0.99982986 | 0.99978250 |

Current regression verdict: **WARN**. The original thresholds are kept; WARN indicates that numerical drift exists, so no claim of unchanged accuracy can be made on this basis. For a single dynamic GDN PTE, 8 lengths are element-wise identical to the native operator, proving that the conversion path itself passes this targeted check.

Full artifacts, raw logits, and diagnostic records: `/mnt/workspace/executorch-npu-eval/qwen_gdn_stage01_20260930`.
## 2026-09-30T09:08:24.783011+00:00 — Recipes stage 2/3/4 combination — qwen3_5

Results directory: `/mnt/workspace/executorch-npu-eval/qwen_recipe234_20260930/regression/qwen3_5`; see `metadata.json` for configuration/commit/artifact fingerprints.

| Case | Backend | Min cosine vs eager | Min cosine vs initial PTE | top-1 agreement vs PTE | TTFT ms | TPOT ms | Total latency ms | tokens/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99984938 | 0.99984077 | 100.00% | 49.86 | 9.71 | 350.84 | 91.210 | WARN |
| medium (512) | pte | 0.99950715 | 0.99968010 | 100.00% | 60.43 | 9.99 | 371.50 | 86.138 | WARN |
| long (2048) | pte | 0.99963177 | 0.99964774 | 100.00% | 112.88 | 9.80 | 416.58 | 76.816 | WARN |


## Recipes stage 2/3/4 combination: relative to stage 1

Adopts the FA, QKVZ/BA and QKV/gate projection fusion, RMSNorm/Add-RMSNorm, and recurrent GDN schemes from recipes-infer 92d9e1f; adapted to the existing contiguous KV Cache with capacity 5120 and BF16 state. The default reference path and the stage-1 artifacts are both retained.

| Case | TTFT speedup | TPOT speedup | Total latency speedup | Min cosine vs stage 1 | top-1 agreement |
|---|---:|---:|---:|---:|---:|
| short | 1.693 | 1.241 | 1.310 | 0.99986479 | 100.00% |
| medium | 1.001 | 1.161 | 1.131 | 0.99960874 | 100.00% |
| long | 1.167 | 1.194 | 1.187 | 0.99974587 | 100.00% |

Comparison of PTE vs the optimized eager:

- short: minimum cosine 0.99986483, top-1 agreement 100.00%.
- medium: minimum cosine 0.99971009, top-1 agreement 100.00%.
- long: minimum cosine 0.99973002, top-1 agreement 100.00%.

The verdict against the initial baseline is **WARN**; the original thresholds are not relaxed. Artifacts and raw results: `/mnt/workspace/executorch-npu-eval/qwen_recipe234_20260930`.
## 2026-09-30T14:44:35.957546+00:00 — Gemma4 recipes fusion adaptation — gemma4

Results directory: `/mnt/workspace/executorch-npu-eval/gemma_recipe_20260930/regression/gemma4`; see `metadata.json` for configuration/commit/artifact fingerprints.

| Case | Backend | Min cosine vs eager | Min cosine vs initial PTE | top-1 agreement vs PTE | TTFT ms | TPOT ms | Total latency ms | tokens/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| short (128) | pte | 0.99997014 | 0.99991453 | 100.00% | 96.39 | 31.27 | 1065.90 | 30.022 | WARN |
| medium (512) | pte | 0.99994846 | 0.99997083 | 100.00% | 105.15 | 29.52 | 1020.16 | 31.368 | WARN |
| long (2048) | pte | 0.99997441 | 0.99997545 | 100.00% | 210.76 | 32.71 | 1224.77 | 26.127 | WARN |


## Gemma4 E2B recipes fusion adaptation

Following the cann-recipes-infer 92d9e1f Gemma4 report: FA v2 (contiguous BNSD, explicit window mask, scale=1), RMSNorm/AddRMSNorm, RoPE lookup table, and partial rotation paired by halves. Shared KV, GELU tanh, PLE, and the final softcap are retained. Capacity 5120; the original baseline is unchanged. After PT2 capture the graph is passed directly to the compiler without serializing a PT2 file, so the capture time cannot be compared directly with historical times that include serialization.

- short PTE vs the optimized eager: minimum cosine 0.99993527, top-1 agreement 100.00%.
- medium PTE vs the optimized eager: minimum cosine 0.99996993, top-1 agreement 100.00%.
- long PTE vs the optimized eager: minimum cosine 0.99997646, top-1 agreement 100.00%.

Initial-baseline verdict **WARN**; thresholds are not relaxed. Results `/mnt/workspace/executorch-npu-eval/gemma_recipe_20260930`; the large PTE/PTD files are stored in `/tmp/gemma_recipe_20260930/pte`, with artifacts/pte as symlinks, and have not yet been backed up to a persistent volume.

## 2026-10-10 — vLLM Ascend single-card comparison

Single 910B3 card; BF16; TP=1; inputs 128/512/2048, output 32; 1 warmup and 3 measurements, taking the median of each metric. vLLM 0.30.0+empty / vllm-ascend 0.19.1rc2.dev2667+g7fca7cae5, pinned image manifest c030d15d93a7d9b3c24347637929b0defe07dade7042704b5fdf65f4d378d36e.

The following uses FULL_DECODE_ONLY graph mode, with ACL graph replay confirmed active. TTFT/total latency are the offline-engine end-to-end request times, including scheduling/IPC but excluding model loading, tokenization, text decoding, and HTTP. TPOT=(total latency−TTFT)/31; output throughput=32/total latency in seconds.

| Model | Input tokens | TTFT ms | TPOT ms | Total latency ms | Output tokens/s |
|---|---:|---:|---:|---:|---:|
| gemma4 | 128 | 163.91 | 19.71 | 775.06 | 41.287 |
| gemma4 | 512 | 173.54 | 20.33 | 809.61 | 39.525 |
| gemma4 | 2048 | 157.64 | 18.51 | 732.89 | 43.663 |
| qwen3_5 | 128 | 178.04 | 56.36 | 1925.26 | 16.621 |
| qwen3_5 | 512 | 168.46 | 56.22 | 1912.85 | 16.729 |
| qwen3_5 | 2048 | 175.39 | 56.55 | 1920.79 | 16.660 |

Concurrent limited-batch output throughput (512 input/32 output per request): Gemma4 at concurrency 4/8 is 138.14/274.75 tokens/s; Qwen3.5 is 58.80/116.41 tokens/s. This is not the maximum throughput under sustained HTTP load.

Gemma4 required two compatibility fixes: quant_method attribute protection for BF16 vision initialization, and the paged-cache path for the first prefill with shared KV. Across single-request and concurrent runs, 10 targeted attention comparisons produced element-wise identical outputs; model weights and the shared-KV structure are retained. Gemma4 long's final token and the concurrent outputs show differences; no full numerical regression/MMLU was run, so no claim of unchanged accuracy is made. Qwen eager/graph outputs for the three cases and the concurrent runs are consistent.

The historical PTE uses fixed trajectories, while this vLLM run uses free greedy generation; the input token IDs are exactly the same as the historical baseline, but this is not a strict same-trajectory A/B. The historical recipes verdict remains WARN; the data is retained and not overwritten. Full eager/graph comparisons, configuration, patches, per-run raw data, and reproduction scripts: `/mnt/workspace/executorch-npu-eval/vllm_compare_20261010/REPORT.md`.


## 2026-10-12 — msprof collection

Uses the retained full PTE/PTD of Gemma4 recipes and Qwen3.5 recipes234, with SHA256 matching the original optimization records. Capacity 5120, BF16, batch=1; inputs 128/512/2048, output 32, 1 warmup and 1 collection, frozen trajectories, with the two models run sequentially.

Results for all three cases of both models and the timeline/operator CSVs have been exported; the collection includes loading and warmup, and profiled times are not written back to the performance baseline. Qwen exited normally. After Gemma's three result sets completed, a heap-memory exit error occurred while the msprof return code was still 0; the separate long-case supplementary collection status is FAILED, and the cause of the anomaly has not yet been located.

Commands, fingerprints, numerical comparisons, and Chinese-language notes: `/mnt/workspace/executorch-npu-eval/msprof_20261012`; the full raw PROF and exported files are archived at `/mnt/workspace/executorch-npu-eval/msprof_20261012/profiles.tar.gz`.

Supplementary comparison: the same standalone Gemma long case exits normally (exit=0) without msprof, while both the original msprof collection and the separate supplementary collection hit an exit error after the full results were written. The six sets of logits for both models are element-wise identical to the retained optimized PTE results; the root cause of the anomaly has not yet been located.


### 2026-10-12 — Qwen cache copy source and compilation path review

The original full PT2, the actual TorchDelegate GE conversion, current torch_npu native targeted profiling, and the existing PTE trace have all been reviewed. The native scatter_update_ has only Scatter; the functional Scatter in the PTE introduces 12 KV TensorMoves. The explicit Linear copy_ is eliminated in the graph, but decode still has 36 recurrent-state Transposes; the 18 IR TensorMoves of the functional GDN are not issued separately, so IR counts cannot substitute for task counts. Address-level attribution of the trailing 60 MiB D2D remains to be confirmed.

Evidence and re-analysis report: /mnt/workspace/executorch-npu-eval/msprof_20261012/cache_reanalysis/evidence. The model/PTE were not modified.


### 2026-10-12: Custom GE Scatter cache in-place update pass (subgraph prototype)
- skill: ge-fusion-pass-skill, GitCode master e7a4876f72724382aab1d659413d054205f42dd2; case: /mnt/workspace/executorch-npu-eval/ge_scatter_reinplace_20261012.
- The C++ graph pass short-circuits explicitly permitted Data→TensorMove→Scatter into Data→Scatter at BeforeInferShape. Loaded as a private OPP; without permission nothing is replaced by default, and it skips cases where the old cache is still read by other paths.
- Real PT2→PTE→NPU: BF16 [1,2,5120,256] cache, 16 rounds across baseline/custom, with the full cache and read results bit-identical; the skill's three graph checks and eight output comparisons passed.
- Final PreRunAfterBuild graphs: TensorMove drops from 1 to 0 for both prefill/decode, while the old-cache observer negative case stays at 1; device task counts and whole-network speedup were not measured.
- The on-site offline BuildModel does not support ge.exec.inputReuseMemIndexes; the prototype uses the experimental process flag CANN_ET_SCATTER_REINPLACE_INPUTS to explicitly declare rewritable inputs.
- Full Qwen/Gemma recompilation, whole-network accuracy, TTFT/TPOT/throughput, and independent control-edge/subgraph fault injection were not run; existing whole-network baseline conclusions are unchanged. See validation-report.md in the case root for the validation report.


### 2026-10-13: Analysis of TransData/slicing/Cast/Transpose in existing Qwen profiling
The data is still the 2026-10-12 full-PTE collection, official rounds prefill4/decode95; no re-export or model modification.
- TransData: prefill 38 / 1.60358ms, decode 36 / 0.28552ms. All decode entries are NCHW↔NC1HWC0 FLOAT conversions for the 18-layer convolution; prefill additionally has 2 aclnnMm TransData totaling 0.91244ms, with shape/format N/A and the specific conversion targets to be investigated.
- StridedSlice: prefill 257 / 3.78862ms, decode 244 / 0.63294ms; covering projection split/chunk, partial RoPE, convolution cropping, and state windows. The 18 prefill trailing-4-column state extractions total 0.97742ms.
- Cast: prefill 269 / 2.19472ms, decode 208 / 0.32176ms. The 97 decode FLOAT→BF16 scalar Casts all come from lifted constants, so constant folding should be investigated; numerically sensitive conversions for convolution, normalization, gating, etc. cannot simply be removed.
- Transpose: prefill 79 / 1.26728ms, decode 36 / 0.34558ms. All decode entries swap the last two axes of the GDN state before and after across 18 layers; a unified state storage layout is worth investigating.
Cumulative task time is not directly savable end-to-end latency. Follow-up focus areas are full causal-conv chain fusion, GDN state layout, constant Casts, prefill trailing slices, and LMhead-related conversions.
Evidence directory: /mnt/workspace/executorch-npu-eval/msprof_20261012/analysis_iterations_3_63, with the full GE traceability in cache_reanalysis.


## 2026-10-13 — Qwen3.5 GDN state layout validation started

Runs serially under nohup: layout tests → full PT2 export → PTE compilation from the saved PT2 → old/new PTE accuracy and performance comparison → current eager validation → old/new msprof collection → report. Only the BF16 state [B,H,V,K] layout is validated, with scalar Cast folding disabled in the PTE export process; the full recipes234 baseline is retained and re-measured. batch=1, cache=5120, inputs 128/512/2048, output 32, 1 warmup, 3 official measurements, with fixed inputs and token trajectories; profiling is collected separately. The focus is on counting actual device GDN state Transposes, TensorMoves grouped by shape, and asynchronous D2D copies.

Task and logs: /mnt/workspace/executorch-npu-eval/gdn_state_layout_20261013; large artifacts: /tmp/gdn_state_layout_20261013. The source snapshot and SHA256 are frozen, and any step failure stops automatically. Results will be appended after completion.


## 2026-10-13 — Qwen3.5 GDN state layout validation interrupted

Error: `RuntimeError('layout_tests exit=1; see logs/layout_tests.log')`. Evidence directory: `/mnt/workspace/executorch-npu-eval/gdn_state_layout_20261013`. Previous baselines and historical WARNs are retained.


## 2026-10-13 — Qwen3.5 GDN layout and scalar Cast folding joint validation started

At the user's request, the existing scalar Cast folding is enabled while validating the BF16 recurrent state [B,H,V,K] layout. The previous round stopped before export due to a state I/O test failure (6 passed, 1 failed), producing no PT2/PTE. The issue where PyTorch functionalization eliminates the implicit conversion of cross-dtype copy_ has been reproduced; explicit BF16 conversion is restored to keep the exported graph consistent with the cache contract, and the conversion becomes a no-op on the normal BF16 path. The previous failure logs are fully retained.

Serial under nohup: GDN and Cast unit tests → PT2 save → PTE compilation → old/new accuracy and performance validation → current eager validation → old/new msprof → report. The full recipes234 baseline is retained and re-measured; cache5120, batch1, inputs 128/512/2048, output 32, 1 warmup, 3 official runs. The focus is on recording changes in Transpose, TensorMove, D2D, and Cast, plus the actual number of scalar folds. Performance is the joint gain of the two modifications and cannot be attributed individually.

Task directory: /mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013; large artifacts: /tmp/gdn_state_cast_20261013; results to be appended.


## 2026-10-13 — Qwen3.5 GDN state layout validation interrupted

Error: `RuntimeError('pte_export exit=1; see logs/pte_export.log')`. Evidence directory: `/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013`. Previous baselines and historical WARNs are retained.


### Qwen3.5 joint validation: resuming from the PTE stage

All 36 GDN/Cast tests passed; the full PT2 export took 65.03 seconds, with both prefill.pt2 and decode.pt2 saved. The first PTE attempt stopped before importing TorchDelegate; the prefill partition recorded 97 scalar Casts actually folded. The background script now preserves the PYTHONPATH set by set_env.sh and prepends the source snapshot; imports of TorchDelegate and of the frozen source have both been verified. The existing PT2 is reused, resuming from the PTE stage under nohup and then running the subsequent accuracy, performance, and profiling steps. The failure scene is preserved at /mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/attempt_01_environment_failure.


## 2026-10-13 — Qwen3.5 profiling statistics correction

The first report filtered by the outer Model ID and missed the bulk of decode's static GE tasks under nested Model ID=1. The first report's conclusion of "no new TensorMove" has been withdrawn; its data and scripts are retained at /mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/reports_before_nested_model_fix. The original collection and the numerical/performance results are unchanged; all device tasks are re-counted over mutually non-overlapping actual iteration time windows, fully verifying 18 GDN and 6 full-attention operators per window. The corrected report below is authoritative.


## 2026-10-13 — Qwen3.5 GDN state layout and scalar Cast folding validation

The BF16 recurrent state is unified to `[B,H,V,K]`, with the current scalar Cast folding implementation enabled at the same time. The PT2→PTE flow is fixed; the old recipes234 PTE is fully retained and re-measured.

batch=1, cache=5120, inputs 128/512/2048, output 32; frozen inputs and teacher-forcing trajectories; performance with 1 warmup and 3 measurements taking the median; profiling collected separately and serially for the old/new artifacts, each with 1 warmup and 1 measurement.

| Input | TTFT old→new ms | TPOT old→new ms | Total latency old→new ms | Output throughput old→new token/s | Min cosine (vs current PTE) | Accuracy |
|---|---|---|---|---|---|---|
| short | 62.828→54.401 | 10.750→9.703 | 394.457→353.567 | 81.124→90.506 | 1.0000000000 | PASS |
| medium | 67.580→64.791 | 10.453→10.463 | 393.589→389.133 | 81.303→82.234 | 1.0000000000 | PASS |
| long | 113.503→117.367 | 9.883→10.210 | 419.161→433.876 | 76.343→73.754 | 1.0000000000 | PASS |

All GDN state Transposes eliminated: True; new TensorMoves: True; new D2D copies: False.

Actual scalar Cast fold counts per compilation partition: [97, 97]. Both optimizations are enabled simultaneously, so the performance gain cannot be attributed solely to the layout change.

| Official iteration | state Transpose old→new | All Transpose old→new | All Cast old→new | TensorMove old→new | New data movement |
|---|---|---|---|---|---|
| short_prefill | 18→0 | 79→61 | 269→269 | 12→12 | [] |
| short_decode | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |
| medium_prefill | 18→0 | 79→61 | 269→269 | 12→12 | [] |
| medium_decode_first | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |
| medium_decode | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |
| long_prefill | 18→0 | 79→61 | 269→269 | 12→12 | [] |
| long_decode | 36→0 | 36→0 | 208→111 | 13→31 | [{"input_shapes": "1,16,128,128", "input_dtypes": "DT_BF16", "extra_count": 18}] |

The initial eager comparison, current eager comparison, old-PTE re-measurement stability, and per-run accuracy and performance ranges are also recorded. The historical recipes234 WARN against the initial eager is not overwritten. Profiling is counted by actual device tasks, shape, copy direction, and size; the number of TensorMove nodes in the IR alone cannot establish that data movement actually occurred. Decode's static GE tasks use nested Model ID=1, which differs from the outer step_trace ID, and are included in the same iteration time window; each time window verifies 18 GDN and 6 full-attention operators with no iteration overlap.

Full evidence: `/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013`; report: `/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/reports/comparison.json`.
Old trace: `/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/profiles/current_baseline/raw/PROF_000001_20261013152541501_03519402OCPKNDCB/mindstudio_profiler_output/msprof_20261013152630.json`; new trace: `/mnt/workspace/executorch-npu-eval/gdn_state_cast_20261013/profiles/native_gdn_layout/raw/PROF_000001_20261013152708619_03519829RFLHBMNJ/mindstudio_profiler_output/msprof_20261013152755.json`.

### Source analysis of the 18 new decode state TensorMoves

The source is `_ge_concrete_graph/ge_converter/custom/recurrent_gated_delta_rule.py:64` in the current TorchAir: the functional converter first performs `state_copy = ge.TensorMove(state)`, then hands the copy to `ge.RecurrentGatedDeltaRule`, which updates the state. The saved torch_npu C++ source likewise corresponds to calling the native recurrent GDN after `state.clone()`; one per linear-attention layer, 18 in total, each BF16 [1,16,128,128], 512 KiB.

The old and new PT2 graph JSONs have been checked directly. Old state input chain: cache→transpose→to(BF16)→contiguous→functional GDN, with the output transposed back to the cache layout; new: cache→to(BF16)→functional GDN. In the new version the cache itself is already contiguous BF16, [B,H,V,K], so the same-dtype conversion and contiguous no longer generate separate state temporary storage.

These 18 TensorMoves already existed in the old full GE audit IR: for the first layer, Transpose_115→Identity_1 (the GE mapping of clone)→TensorMove_1→RecurrentGatedDeltaRule. The old profiling actually issued only Transpose→GDN→Transpose; the new version actually issues TensorMove→GDN. So the model did not add 18 copy_ operations; rather, the pre-existing functional input-protection copies are now issued independently. The old transpose produces a temporary buffer that can be exclusively owned, while the new version reads the persistent cache graph input directly, which explains the difference in copy-elimination conditions between the two. The specific GE pass responsible for the old copy elimination/memory reuse has not been confirmed, so it is not attributed to any proven reinplace pass.

The state TensorMoves added in this round sit before the GDN; the copy_ of cache.update_recurrent_state sits after the GDN and also existed in the old version. A functionalized aten.copy with the same dtype/shape can return src directly in the current converter, so these 18 tasks cannot be assigned to cache write-back merely because copy_ appears in Python. At runtime the state input uses current and the output uses next, with Commit swapping the two; removing the converter's protection copy or switching to in-place GDN requires validating export functionalization, input aliasing/lifetime, and state I/O binding together, so TensorMove cannot simply be replaced with Identity.

Official decode iteration 95 for the 512 input: the old 36 state Transposes total 336.60 us; the new 18 state TensorMoves total 190.46 us, the related state-movement task count goes 36→18, and cumulative task time is reduced by about 146.14 us (43.4%). The 12 large K/V TensorMoves are unchanged. New TensorMoves do not mean a net increase in state movement, nor can they directly explain the previously observed 3.5% end-to-end regression on the 2048 case.


## 2026-10-13 — GDN / Scatter shared GE pass subgraph and persistent state validation

The same C++ pass keeps the Scatter input0 rule and adds a RecurrentGatedDeltaRule state input4 rule; the preceding TensorMove is eliminated only for exclusively owned, modifiable cache inputs, while GDN outputs 0/1 are preserved. The real GE has 8 valid inputs, compatible with the two trailing empty optional inputs among the 10 anchors.

The isolated build on CANN9.1/910B3 passed. GDN baseline/custom produced 16 sets of full outputs and states that are bit-identical, with the old-state observer negative case correctly skipped; the real stateful runner's 4 prefills + 7 decode steps produced bit-identical logits throughout; all 16 Scatter regression sets are bit-identical. The outputs of the two profiles are also bit-identical, with no application exit anomalies observed.

Single-layer session, official decode iteration 9: state TensorMove baseline 1 / 7.58us, custom 0; 512KiB D2D is 0 across the entire capture range on both sides, so the copy was not shifted to output DMA. The runtime current/next implementation was not modified.

This is single-layer prototype validation and does not mean the full Qwen has eliminated the 18 state copies or already has whole-network speedup gains; it has not been integrated into the production compilation path, and the full Qwen/Gemma have not been recompiled. External address aliasing, failure rollback, and independent control-edge/subgraph fault injection are not covered; G3 DEGRADED/NOT_RUN. Source, validation, and the step-by-step matrix: /mnt/workspace/executorch-npu-eval/ge_cache_reinplace_20261013.


## 2026-10-13 — Qwen GDN GE pass whole-network compilation validation started

With user authorization, runs serially under nohup: decode GE input audit → full PTE compilation → accuracy/performance re-test without and with the pass → msprof on both sides → report. Reuses the full prefill/decode PT2 in /tmp/gdn_state_cast_20261013/pt2 and the frozen source; compared against the current layout-unification + Cast-folding PTE, only decode's 18 owned recurrent state inputs are allowed to be made in-place, with prefill/Scatter unchanged. cache5120, inputs 128/512/2048, output 32, batch1, 1 warmup, 3 official measurements; fixed inputs and trajectories; the same-source eager reference reuses the previous round's results. The focus is on verifying the 18 state TensorMoves and 512KiB D2D across the entire capture range.

Task directory: /mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013; large artifacts: /tmp/qwen_ge_cache_reinplace_20261013. The original baseline and PT2 are retained; results to be appended.


## 2026-10-13 — Qwen3.5 full-model GDN GE pass validation

Both the baseline and the optimized side use the BF16 `[B,H,V,K]` state layout and the same scalar Cast folding. The saved full PT2 is reused to compile the PTE; only decode's 18 recurrent state inputs are allowed to be made in-place, while prefill and Scatter keep baseline behavior.

batch=1, cache=5120, inputs 128/512/2048, output 32; frozen inputs and teacher-forcing trajectories; performance with 1 warmup and 3 measurements taking the median; profiling collected separately and serially for the old/new artifacts, each with 1 warmup and 1 measurement.

| Input | TTFT old→new ms | TPOT old→new ms | Total latency old→new ms | Output throughput old→new token/s | Min cosine (vs current PTE) | Accuracy |
|---|---|---|---|---|---|---|
| short | 54.817→49.361 | 9.830→9.438 | 359.536→341.946 | 89.004→93.582 | 1.0000000000 | PASS |
| medium | 58.530→59.488 | 9.538→9.790 | 353.362→362.030 | 90.559→88.390 | 1.0000000000 | PASS |
| long | 111.569→109.721 | 9.413→9.441 | 403.361→403.582 | 79.333→79.290 | 1.0000000000 | PASS |

All GDN state Transposes eliminated: True; new TensorMoves: False; new D2D copies: True.

Scalar Cast fold counts per compilation partition: [97, 97]; this optimization is kept in this round.

Official decode state TensorMove counts: {'short_decode': {'baseline': 18, 'optimized': 0}, 'medium_decode': {'baseline': 18, 'optimized': 0}, 'long_decode': {'baseline': 18, 'optimized': 0}}; whether all 18 are eliminated: True.

New 512KiB state D2D events across the entire capture range: 3348.

| Official iteration | state Transpose old→new | All Transpose old→new | All Cast old→new | TensorMove old→new | New data movement |
|---|---|---|---|---|---|
| short_prefill | 0→0 | 61→61 | 269→269 | 12→12 | [] |
| short_decode | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |
| medium_prefill | 0→0 | 61→61 | 269→269 | 12→12 | [] |
| medium_decode_first | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |
| medium_decode | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |
| long_prefill | 0→0 | 61→61 | 269→269 | 12→12 | [] |
| long_decode | 0→0 | 0→0 | 111→111 | 31→13 | [{"operation": "device to device", "size_bytes": 524288, "extra_count": 18}] |

The initial eager comparison, current eager comparison, old-PTE re-measurement stability, and per-run accuracy and performance ranges are also recorded. The historical recipes234 WARN against the initial eager is not overwritten. Profiling is counted by actual device tasks, shape, copy direction, and size; the number of TensorMove nodes in the IR alone cannot establish that data movement actually occurred. Decode's static GE tasks use nested Model ID=1, which differs from the outer step_trace ID, and are included in the same iteration time window; each time window verifies 18 GDN and 6 full-attention operators with no iteration overlap.

Full evidence: `/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013`; report: `/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013/reports/comparison.json`.
Old trace: `/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013/profiles/without_pass/raw/PROF_000001_20261013192758665_03559383PCRQKRFN/mindstudio_profiler_output/msprof_20261013192844.json`; new trace: `/mnt/workspace/executorch-npu-eval/qwen_ge_cache_reinplace_20261013/profiles/with_pass/raw/PROF_000001_20261013192919780_03559803MHHIPBKP/mindstudio_profiler_output/msprof_20261013193006.json`.


## 2026-10-13 — Qwen prefill/decode single-buffer state validation

Comparison: the same GDN GE pass, BF16 layout, and Cast folding; double buffer → single buffer. Reuses the validated PT2.
Inputs 128/512/2048, output 32; performance with 1 warmup and 3 measurements taking the median; profiling collected separately.

| Case | TTFT ms | TPOT ms | Total latency ms | Output token/s | Accuracy |
|---|---|---|---|---|---|
| short | 43.568→52.061 | 9.417→9.936 | 335.054→362.436 | 95.507→88.291 | PASS |
| medium | 52.667→54.492 | 9.999→9.922 | 362.027→363.250 | 88.391→88.094 | PASS |
| long | 112.051→109.705 | 10.290→9.750 | 432.258→410.988 | 74.030→77.861 | PASS |

512 KiB state D2D across all captures: 3348 → 0.
Acceptance passed: True. Full evidence: `/mnt/workspace/executorch-npu-eval/state_single_buffer_20261013/full`.

State movement is attributed by shape and size; 18 GDN and 6 Full Attention operators are verified per window.


## 2026-10-13 — Qwen single-buffer alternating performance re-test

Reuses the model and the frozen runner, with 4 paired alternating-order sessions, 3 warmups and 7 measurements per case; 28 runs per version per case in total. Case order is also rotated.
The logits of all measured requests are bit-identical to the retained baseline. Percentages are the median of the four paired sessions' individual median changes.

| Input | TTFT ms double→single | TPOT ms double→single | Total latency ms double→single | Throughput token/s double→single | Paired total latency change | Range across four pairs |
|---|---|---|---|---|---|---|
| short | 45.574→46.516 | 9.539→9.137 | 343.871→330.519 | 93.059→96.817 | -1.81% | -5.97% ~ +4.43% |
| medium | 53.590→51.385 | 9.462→9.280 | 344.060→341.865 | 93.007→93.605 | -3.89% | -7.52% ~ +5.62% |
| long | 108.275→106.947 | 9.581→9.540 | 405.064→402.435 | 79.000→79.517 | -1.88% | -5.64% ~ +4.41% |

Raw per-run measurements, paired results, and environment snapshot: `/tmp/qwen_perf_recheck_20261013`. This round does not modify source, re-export, or treat profiled times as performance results.

Analysis conclusion: the previously observed +8.17% total-latency regression on short input did not reproduce consistently. For all three input lengths, the first three paired sessions were faster and the fourth was slower, showing clear inter-session fluctuation; four pairs of data are insufficient to claim a stable speedup and cannot be precisely attributed to CPU scheduling or device frequency. It is recommended to keep the single-buffer implementation, which is verified correct and eliminates state movement, and to record performance as a slight improving trend.

For the selected short decode iteration with profiling: cumulative operator time 5195.66→5193.04 μs, cumulative asynchronous copy time 170.64→20.78 μs. This supports that data movement did decrease, but these values come from separate profiling and cannot directly substitute for unprofiled request latency.

The first re-test in this round was interrupted because /mnt/workspace filled up; it is preserved in full at /tmp/qwen_perf_recheck_20261013/attempt1_disk_full and is excluded from the statistics above. The valid re-test exited with code 0, with 168 measurements in total, and all logits are bit-identical to the retained baseline.


## 2026-10-14 — Qwen single-buffer alternating performance re-test

Reuses the model and the frozen runner, with 4 paired alternating-order sessions, 3 warmups and 7 measurements per case; 28 runs per version per case in total. Case order is also rotated.
The logits of all measured requests are bit-identical to the retained baseline. Percentages are the median of the four paired sessions' individual median changes.

| Input | TTFT ms double→single | TPOT ms double→single | Total latency ms double→single | Throughput token/s double→single | Paired total latency change | Range across four pairs |
|---|---|---|---|---|---|---|
| short | 44.154→44.847 | 9.349→9.234 | 335.071→333.423 | 95.502→95.976 | -1.39% | -2.65% ~ -0.50% |
| medium | 51.613→51.346 | 9.458→9.209 | 348.101→338.816 | 91.927→94.447 | -1.70% | -5.37% ~ +0.40% |
| long | 109.328→107.804 | 10.256→9.633 | 427.685→405.793 | 74.883→78.874 | -2.83% | -7.53% ~ +0.40% |

Raw per-run measurements, paired results, and environment snapshot: `/tmp/qwen_perf_recheck_20261014`. This round does not modify source, re-export, or treat profiled times as performance results.


## 2026-10-16 — Full evaluation of both models on the latest code started

Source `f37cb9b56d603e6b2b72da0c05310770698b2339` plus the workspace modifications at launch; snapshot: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/source`. Background PID `3934960`.
Runs serially: rebuild, PT2→PTE, full 5-shot MMLU, and fixed-trajectory performance tests. No completed results for this round yet.
Task status and logs: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/status.json`, `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/run.log`; results are appended after each model completes.


## 2026-10-16 — Latest-code MMLU and performance — gemma4

The source snapshot is based on `f37cb9b56d603e6b2b72da0c05310770698b2339` plus all uncommitted modifications at launch; provenance and SHA256: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/source_manifest.json`.
The runner is rebuilt from that snapshot and PT2→PTE is re-exported; executed through the current repository's default compilation entry, with no additional experimental GE pass injected.
BF16, batch=1, cache=5120. Full MMLU: 57 subjects, 14,042 questions, 5-shot, reusing historical token inputs, with no truncation and no sampling.

| MMLU | This round | Historical standard PTE | Change |
|---|---:|---:|---:|
| Correct count | 6853/14042 | 6877/14042 | -24 |
| Accuracy | 48.8036% | 48.9745% | -0.1709 pp |

Answer agreement rate: 97.7354%; 112 previously correct now wrong, 88 previously wrong now correct.
Per-question scores, predictions, and per-subject results: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/gemma4/mmlu/summary.jsonl`, `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/gemma4/mmlu/summary.json`.

Performance is a separate unprofiled fixed-trajectory test; inputs 128/512/2048, output 32, 3 warmups and 7 measurements. Model loading and logits diagnostic snapshots are excluded from the four metrics.

| Input | TTFT ms | TPOT ms | Total latency ms | Output token/s | Min cosine vs initial PTE | top-1 agreement | Numerical verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| short (128) | 89.763 | 32.824 | 1106.729 | 28.914 | 0.99991453 | 100.00% | WARN |
| medium (512) | 88.830 | 29.350 | 1000.238 | 31.992 | 0.99997083 | 100.00% | WARN |
| long (2048) | 200.617 | 33.801 | 1247.631 | 25.649 | 0.99997545 | 100.00% | WARN |

Full source, environment, PT2/PTE fingerprints, and per-stage logs: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/gemma4`; see `artifacts.json` for the shared runner fingerprint.
Historical numerical baselines and WARNs are all retained; thresholds are not relaxed. The wall-clock time of full scoring includes loading and is not used as the fixed-trajectory inference latency above.


## 2026-10-16 — qwen3_5 latest-code evaluation interrupted

Error: `CalledProcessError(1, ['/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/runner', 'session', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/pte/model.pte', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/pte/ascend.weights.package.ge_aot.ptd', 'prefill', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/inputs/short.i64', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/inputs/short.positions.i64', 'decode', '128', '1,1', '32', '0', 'light_eval', '--max-seq-len', '5120', '--eval-output', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/pte', '--warmup', '3', '--repeats', '7', '--causal-mask', '--eval-manifest', '/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/pte/cases.tsv'])`. Logs: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/logs`. Partial scoring is not marked as complete; evaluation of the other model continues.


## 2026-10-16 — Qwen3.5 full MMLU resumed independently

The original task's decode performance check failed; scoring uses prefill, which passed the smoke test. The resumed run PID is `3974189`, reusing and verifying this round's PTE/PTD and runner without redoing the export. No complete Qwen score yet.


## 2026-10-16 — Qwen3.5 full MMLU resumed-run results

Reuses this round's latest-code snapshot and the completed PT2/PTE and runner; no re-export and no additional GE pass injected.
Full 57 subjects, 14,042 questions, 5-shot, no truncation. Scoring goes through prefill; the decode performance issue is kept separately.

MMLU: **41.2263% (5789/14042)**.
Relative to the historical standard PTE: +0.1139 pp; answer agreement rate 98.5472%.
54 previously correct now wrong; 70 previously wrong now correct.

Per-question and per-subject results: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/mmlu/summary.jsonl`, `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/mmlu/summary.json`.
The performance evaluation still fails: `decode logits must be valid float32 or bfloat16 with vocabulary last`; there are no valid complete performance results, so the four metrics cannot be reported.
Original failure log: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/qwen3_5/performance/pte/runner.log`.
This section only marks the full MMLU as complete; it does not mark the decode performance issue as passed.


## 2026-10-16 — Qwen decode NaN isolation control started

Reproduced: all 248,320 prefill logits are finite; all 248,320 logits of the first decode step are NaN, with normal shape/dtype. Cross combinations of old/new artifacts and runtimes are all rejected by the ABI check, so no attribution can be made on that basis.
The isolation control changes only decode's 18 recurrent states to double, reusing this round's PT2; all other model computation, weights, and the compilation entry are unchanged. Repository source is not modified and the scored PTE is not overwritten. PID `3990154`, directory: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`. No control conclusion yet.


## 2026-10-16 — Qwen decode NaN isolation control started

Reproduced: all 248,320 prefill logits are finite; all 248,320 logits of the first decode step are NaN, with normal shape/dtype. Cross combinations of old/new artifacts and runtimes are all rejected by the ABI check, so no attribution can be made on that basis.
The isolation control changes only decode's 18 recurrent states to double, reusing this round's PT2; all other model computation, weights, and the compilation entry are unchanged. Repository source is not modified and the scored PTE is not overwritten. PID `3995653`, directory: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`. No control conclusion yet.


## 2026-10-16 — Qwen decode NaN isolation control started

Reproduced: all 248,320 prefill logits are finite; all 248,320 logits of the first decode step are NaN, with normal shape/dtype. Cross combinations of old/new artifacts and runtimes are all rejected by the ABI check, so no attribution can be made on that basis.
The isolation control changes only decode's 18 recurrent states to double, reusing this round's PT2; all other model computation, weights, and the compilation entry are unchanged. Repository source is not modified and the scored PTE is not overwritten. PID `4003672`, directory: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`. No control conclusion yet.


## 2026-10-16 — Qwen decode recurrent state double-buffer control results

This is an isolated diagnostic control and does not replace the original latest-code single-buffer failure result; the original full MMLU still comes from the original artifacts.

| Input | TTFT ms | TPOT ms | Total latency ms | token/s | Numerical verdict |
|---|---:|---:|---:|---:|---|
| short | 43.061 | 9.577 | 341.013 | 93.838 | WARN |
| medium | 49.163 | 9.571 | 346.272 | 92.413 | WARN |
| long | 104.704 | 9.174 | 389.291 | 82.201 | WARN |

Only the buffer mode of the decode recurrent state is changed. Full evidence: `/mnt/workspace/executorch-npu-eval/latest_mmlu_perf_20261016/state_alias_control`.
If this control passes, it only shows that the issue is related to that binding path; it cannot yet be localized to a specific CANN/runtime operation.


## 2026-10-19 — Qwen single buffer decision defect confirmed

With the same October 16 PTE/PTD and the same diagnostic runtime, single produced all-NaN on the first decode; changing only the runtime binding of the 18 decode recurrent states to double after loading yields finite output. No OM was recompiled, ruling out recompilation and old/new ABI differences as explanations for this control.
A formal compiler fix is being validated: functional GDN that keeps TensorMove does not declare boundary co-location and falls back to double; genuine direct references and paths that do not read the old state keep single. Evidence: `/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019`.


## 2026-10-19 — Qwen recurrent state single-buffer decision fix and double-buffer fallback

The root cause is located to an overly broad alias declaration in the compiler: the GDN state output references the result of TensorMove rather than the original Data input; the original decision treated these two storages as the same boundary and allowed co-located binding. A single-variable runtime control with the same PTE reproduced the NaN and eliminated it with double.
Formal fix: this protection-copy path falls back to double; genuine direct references and single paths that do not read the old state are retained. No new model options were added and CANN/TorchAir were not modified; the full scoring history is retained.

Before the fix, 2 regression cases failed and 1 passed; after the fix, the state-planning/manifest/backend/compiler-context tests pass. A normal default export yields a double contract for the 18 decode recurrent states; 3 cases × 7 runs × 32 tokens are all finite, with logits bit-identical to the validated double-buffer control. The initial numerical thresholds remain WARN and are not relaxed.

| Input | TTFT ms | TPOT ms | Total latency ms | token/s | Numerical verdict |
|---|---:|---:|---:|---:|---|
| short | 44.917 | 9.331 | 335.820 | 95.289 | WARN |
| medium | 50.786 | 9.514 | 344.986 | 92.757 | WARN |
| long | 105.044 | 9.307 | 393.499 | 81.322 | WARN |

Applied patch: `/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019/fix.diff`; new artifacts: `/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019/pte`; full evidence: `/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019`.
The full MMLU was not re-run in this round; the previous score comes from the original October 16 artifacts and cannot be labeled as a re-test result of the new artifacts.


## 2026-10-19 — GDN state in-place formal integration validation

The default pipeline is now GE graph lowering → conservative in-place rewriting based on the device-only state contract → actual topology buffer planning → offline compilation. It does not rely on external OPP pass injection, hand-written Data indices, or model recipe switches; the previous round's strict alias fix is retained, and unmatched cases continue to use double.
The scope is limited to RecurrentGatedDeltaRule with 8 inputs, 2 outputs, and static BF16 [1,H,128,128]. The protection copy and state must have no additional readers or control edges, and the output must correspond to the same state contract; a failed graph rewrite aborts the export.
A normal Qwen export actually rewrites 18 Data→TensorMove→GDN occurrences into Data→GDN and subsequently plans 18 decode recurrent singles. Continuous decode and cross-request reset are covered over 3 cases × 7 runs × 32 tokens; all outputs are finite and bit-identical to the safe double control. The initial baseline and thresholds are unchanged, and the original numerical verdict remains WARN; the full MMLU was not re-run in this round.

| Input | New single TPOT ms | Same-day double TPOT ms | TPOT reduction | New TTFT ms | New token/s |
|---|---:|---:|---:|---:|---:|
| short | 9.731 | 12.067 | 19.35% | 47.049 | 91.676 |
| medium | 10.719 | 10.659 | -0.56% | 58.926 | 81.454 |
| long | 9.626 | 11.821 | 18.57% | 110.039 | 78.330 |

The gains in the table are measured from same-day sequential runs and do not mean frequency/load noise has been ruled out. Graph-rewrite evidence alone is not equivalent to runtime task elimination; without msprof collection, no final TensorMove task count is claimed.
New artifacts: `/mnt/workspace/executorch-npu-eval/gdn_reinplace_integration_20261019/pte`; full evidence and patch: `/mnt/workspace/executorch-npu-eval/gdn_reinplace_integration_20261019`; the safe double-buffer artifacts are retained: `/mnt/workspace/executorch-npu-eval/single_buffer_fix_20261019/pte`.


## 2026-10-19 — single buffer / reinplace scheme withdrawn at the user's request

The current code uniformly restores the current/next double buffer and removes the Python GE in-place rewrite, automatic single planning, the manifest buffer_mode field, and the runtime co-located binding branch. The original model adaptation, fusion, state contracts, and evaluation tools are retained. Previous single buffer / reinplace results serve only as historical experiment records and no longer represent the current default implementation. The buffer_mode field in old experimental PTEs will be rejected; they must be re-exported and used with the runtime rebuilt in this round. Pre-rollback backup and patch: `/mnt/workspace/executorch-npu-eval/double_buffer_rollback_20261019`; build and regression validation are in progress.


## 2026-10-19 — Double-buffer rollback validation completed

The single buffer / reinplace code and its dedicated tests have been withdrawn, and the schema is synchronized. State inputs are bound fixedly to current and outputs fixedly to next, swapped on success; read-only states are not swapped. C++ / Python regressions pass, and real NPU probes verify distinct input/output addresses, successful swapping, no commit on failure, no swapping for read-only, and reset/release.

The runtime is rebuilt from the current rollback code and Qwen exports normally: the 18 decode GDNs keep TensorMove, and no state contract contains buffer_mode any more. 3 input types × 7 runs × 32 tokens are all finite, with logits bit-identical to the validated safe double control. The original numerical WARN and thresholds are retained. Gemma was not re-exported and the full MMLU was not re-run in this round.

Current runtime: `/mnt/workspace/executorch_npu/build/release/executorch_npu_stateful_runner`; new Qwen PTE: `/mnt/workspace/executorch-npu-eval/double_buffer_rollback_20261019/pte`; evidence and rollback patch: `/mnt/workspace/executorch-npu-eval/double_buffer_rollback_20261019`.


## 2026-10-19 — Full evaluation of both models on the current double buffer started

Source `f37cb9b56d603e6b2b72da0c05310770698b2339` plus the workspace modifications at launch; snapshot: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/source`. Background PID `12041`.
Freezes the current double-buffer code and rebuilds the runner; Gemma re-exports the PTE, while Qwen reuses the double-buffer artifacts verified for source consistency. Both models re-run the full 57-subject / 14,042-question / 5-shot MMLU and the fixed-trajectory performance tests. No completed results for this round yet.
Task status and logs: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/status.json`, `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/run.log`; results are appended after each model completes.


## 2026-10-19 — Current double-buffer full MMLU and performance — gemma4

The source snapshot is based on `f37cb9b56d603e6b2b72da0c05310770698b2339` plus all uncommitted modifications at launch; provenance and SHA256: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/source_manifest.json`.
The runner is rebuilt from that snapshot; fixed current/next double buffer, swapped on success, with no buffer_mode / reinplace.
Gemma: after verifying the model/export code and PT2 fingerprints, the PTE is re-exported with the current default compiler.
BF16, batch=1, cache=5120. Full MMLU: 57 subjects, 14,042 questions, 5-shot, reusing historical token inputs, with no truncation and no sampling.

| MMLU | This round | Historical standard PTE | Change |
|---|---:|---:|---:|
| Correct count | 6853/14042 | 6877/14042 | -24 |
| Accuracy | 48.8036% | 48.9745% | -0.1709 pp |

Answer agreement rate: 97.7354%; 112 previously correct now wrong, 88 previously wrong now correct.
Per-question scores, predictions, and per-subject results: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/gemma4/mmlu/summary.jsonl`, `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/gemma4/mmlu/summary.json`.

Performance is a separate unprofiled fixed-trajectory test; inputs 128/512/2048, output 32, 3 warmups and 7 measurements. Model loading and logits diagnostic snapshots are excluded from the four metrics.

| Input | TTFT ms | TPOT ms | Total latency ms | Output token/s | Min cosine vs initial PTE | top-1 agreement | Numerical verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| short (128) | 88.816 | 30.943 | 1049.607 | 30.488 | 0.99991453 | 100.00% | WARN |
| medium (512) | 98.056 | 29.320 | 1006.293 | 31.800 | 0.99997083 | 100.00% | WARN |
| long (2048) | 215.141 | 32.893 | 1232.509 | 25.963 | 0.99997545 | 100.00% | WARN |

Full source, environment, PT2/PTE fingerprints, and per-stage logs: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/gemma4`; see `artifacts.json` for the shared runner fingerprint.
Historical numerical baselines and WARNs are all retained; thresholds are not relaxed. The wall-clock time of full scoring includes loading and is not used as the fixed-trajectory inference latency above.


## 2026-10-19 — Current double-buffer full MMLU and performance — qwen3_5

The source snapshot is based on `f37cb9b56d603e6b2b72da0c05310770698b2339` plus all uncommitted modifications at launch; provenance and SHA256: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/source_manifest.json`.
The runner is rebuilt from that snapshot; fixed current/next double buffer, swapped on success, with no buffer_mode / reinplace.
Qwen: item-by-item source verification matches the double-buffer artifacts that passed regression; that PTE is reused, and this round re-scores and re-measures performance.
BF16, batch=1, cache=5120. Full MMLU: 57 subjects, 14,042 questions, 5-shot, reusing historical token inputs, with no truncation and no sampling.

| MMLU | This round | Historical standard PTE | Change |
|---|---:|---:|---:|
| Correct count | 5789/14042 | 5773/14042 | +16 |
| Accuracy | 41.2263% | 41.1124% | +0.1139 pp |

Answer agreement rate: 98.5472%; 54 previously correct now wrong, 70 previously wrong now correct.
Per-question scores, predictions, and per-subject results: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/qwen3_5/mmlu/summary.jsonl`, `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/qwen3_5/mmlu/summary.json`.

Performance is a separate unprofiled fixed-trajectory test; inputs 128/512/2048, output 32, 3 warmups and 7 measurements. Model loading and logits diagnostic snapshots are excluded from the four metrics.

| Input | TTFT ms | TPOT ms | Total latency ms | Output token/s | Min cosine vs initial PTE | top-1 agreement | Numerical verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| short (128) | 50.673 | 9.609 | 348.122 | 91.922 | 0.99984077 | 100.00% | WARN |
| medium (512) | 54.051 | 9.403 | 345.717 | 92.561 | 0.99968010 | 100.00% | WARN |
| long (2048) | 113.402 | 9.827 | 418.499 | 76.464 | 0.99964774 | 100.00% | WARN |

Full source, environment, PT2/PTE fingerprints, and per-stage logs: `/mnt/workspace/executorch-npu-eval/double_buffer_mmlu_perf_20261019/qwen3_5`; see `artifacts.json` for the shared runner fingerprint.
Historical numerical baselines and WARNs are all retained; thresholds are not relaxed. The wall-clock time of full scoring includes loading and is not used as the fixed-trajectory inference latency above.
