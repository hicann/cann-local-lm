# Lightweight End-to-End Regression

Goal: quickly compare the final logits and inference latency before and after an optimization, using three fixed text inputs and 32 output steps. This is not an MMLU replacement.

## Usage

In the `executorch_npu` environment, pass the model, PTE, and runner paths directly; per-run JSON configurations are not saved in the repository. The example below uses Gemma; for Qwen, change to `--model qwen3_5` and specify the corresponding paths.

```bash
bash examples/scripts/run_light_eval.sh baseline \
  --model gemma4 --model-path /mnt/workspace/models/gemma-4-E2B-it \
  --pte-dir /path/to/baseline/pte --output /path/to/light_baseline --stage baseline

bash examples/scripts/run_light_eval.sh run \
  --model gemma4 --model-path /mnt/workspace/models/gemma-4-E2B-it \
  --pte-dir /path/to/candidate/pte --baseline /path/to/light_baseline \
  --output /path/to/light_stage_01 --stage 'stage_01: optimization details'
```

You can also pass `--runner`, `--capacity`, `--output-tokens`, `--warmup`, `--repeats`, and `--device-id`. The run configuration and metadata are written to the result directory. `--config` is only for optional external configuration/replay of old results. Subsequent runs execute only the candidate PTE without re-running eager, and append to `docs/performance_log.md` by default; overwriting is forbidden when a complete baseline exists.

Model-related optimizations require regenerating the PTE through the existing export entry point; the evaluation does not export automatically. Runtime optimizations require rebuilding the runner first.

## Fixed Conditions

- Cache capacity 5120, batch=1, BF16; input lengths 128/512/2048; 32 output steps.
- For each case, the first warmup greedily generates 32 tokens with the initial PTE and freezes them; eager / PTE then both use the same history tokens for teacher forcing. EOS does not truncate.
- Each model is loaded only once; the three cases share the resident model, each with 1 warmup and 3 measurements. Loading time is not counted in the four metrics. Historically, loading the current Qwen artifacts takes about 13.5 minutes and is the main source of waiting in a full evaluation round.
- eager uses the repository's existing ExportWrapper + a custom static Cache, executing step by step; it is not the old MMLU's use_cache=False, nor a correctness oracle implemented independently of this repository.
- The two models run serially, with exclusive NPU access during measurement; avoid concurrent export/compilation or other compute tasks.

## Metric Boundaries

Timing starts at request state initialization, covers mask construction, H2D/D2H, prefill, decode, and per-step greedy token selection, and ends when the last token is available. Tokenization and input file reading are completed beforehand. The PTE runtime synchronously copies outputs to the host; eager performs token selection after `.cpu()`.

The logits snapshot is copied only after each step's timing ends, and the next timing segment is then restarted. Total latency is the sum of the segment durations, excluding diagnostic copy time; the raw wall-clock value `wall_with_capture_ms` is also retained. The timing measures the end-to-end model inference duration under a fixed trajectory, not service latency including request queuing, tokenization, network, or text rendering. Snapshots may still perturb the CPU cache; all phases are measured in the same way.

- TTFT: time until the first output is available.
- TPOT: sum of the remaining 31 step durations / 31.
- Total latency: sum of the 32 step measurement intervals.
- Output throughput: 32 / total latency in seconds.

The four metrics are recorded for each of the three runs, and the table takes the median of each separately, so small formula-level discrepancies may exist among the four medians.

## Numerical Comparison

The final full-vocabulary logits of every step of every run are saved (FP32 files, one step per line, step=0 is prefill), and per-step cosine similarity against the initial eager/PTE is computed. The report includes the average/minimum cosine of the worst run, the worst step, and the top-1 agreement rate; the raw JSON retains per-step results. Non-finite values and zero norms are checked. Incompatible comparisons are rejected when inputs, capacity, or weights/tokenizer change.

The initial record does not assume eager/PTE are exactly equal. The subsequent WARN tolerance is `max(1e-5, 5 * initial PTE repeat cosine error)`; a warning is raised if the value relative to the initial PTE exceeds the tolerance, drops more than the tolerance below the initial PTE's minimum cosine relative to eager, or the top-1 agreement rate falls below the baseline's own repeat agreement rate. This threshold is the initial rule for numerical drift checking and does not guarantee task accuracy; model run errors, missing results, or non-finite outputs fail immediately and no successful-phase report is written.

## Results

Each model directory contains `metadata.json` (commit, diff, software versions, device, weight/PTE/runner SHA256), `config.json`, frozen inputs, three runs of raw logits and timings per backend per case, and `summary.json`. The baseline always remains unchanged; optimization phases use new directories. Run commands and full runner logs are also retained.
