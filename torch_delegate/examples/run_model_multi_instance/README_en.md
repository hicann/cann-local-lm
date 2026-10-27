# run_model_multi_instance

English | [简体中文](README.md)

Multi-process concurrent end-to-end example: it calls `td::EpairModelLoader::RunModel` (synchronous interface) to verify **multi-instance concurrent execution**. The parent process spawns multiple worker processes (one process per instance). Inside each worker, a producer thread and a consumer thread call `RunModel` on their own instance in parallel, exposing host-side multi-process submission and device-side multi-stream parallelism.

## Why multi-process (one process per instance)

The deployed `RunModel` wraps GE's synchronous `RunGraph`: `InnerSession::RunGraph` takes a
**process-level static mutex** (`try_to_lock`, held until the graph finishes), so concurrent calls
from threads in the same process are rejected with `GE_SESS_ALREADY_RUNNING` and serialized. One process per instance (each with its own
`GeSession`/ACL context/Device stream) bypasses the in-process shared lock and restores device-side multi-stream concurrency.

## Model (deep and wide)

`export_matmul_epair.py` builds a network with `depth` stacked layers, each holding `branches` **independent** MatMul+ReLU
branches (default 24 layers × 8 branches ≈ 600 operators: 8 MatMul + 9 ReLU + 8 Add per layer;
weights are reused across layers, keeping memory usage bounded):

- Many layers and operators: a single `RunModel` takes milliseconds of Device execution time, so concurrency effects are stable and observable;
- Independent branches in the graph: each layer has several mutually independent MatMul+ReLU branches, giving a large and regular operator graph.

## Concurrency Metrics

Two phases (`CLOCK_MONOTONIC`, cross-process timestamps are directly comparable):

1. **SOLO**: each instance runs one task alone (serial baseline, the parent releases instances one by one) to obtain the single-task duration `t_solo`;
2. **RUN**: all instances execute the remaining tasks concurrently.

| Metric | Meaning | Interpretation |
|------|------|------|
| `host_max_inflight` | Maximum overlap of `RunModel` call intervals from a sweep line (number of processes inside the call at the same time) | ≥2 means host-side concurrent submission; ≈workers means fully loaded |
| `device_parallelism` | `concurrent tasks × t_solo / wall` (`RunModel` blocks until the graph finishes, so wall reflects Device throughput) | ≈workers means Device multi-stream parallelism is sufficient; ≈1 means serialized |
| `avg concurrency` | `Σ task duration / wall` (reference value) | order-of-magnitude reference |

## Prerequisites

1. The `torch_delegate` whl is built and installed (`pip install <torch_delegate whl>`);
   `TORCH_DELEGATE_ROOT` is derived automatically from the installed whl, so no `source set_env.sh` is needed
2. `<cann_path>/cann/set_env.sh` has been sourced
3. Model and inputs generated:
   ```bash
   python3 export_matmul_epair.py --dim 1024 --depth 24 --branches 8
   ```
   This generates `matmul_model.epair` and `input_0.bin` (`dim × dim` float32)

## One-Shot Run

```bash
bash run.sh                       # default: 8 processes (instances), 48 tasks

# Overridable parameters:
INSTANCES=8 TASKS=48 DIM=1024 DEPTH=24 BRANCHES=8 bash run.sh
MIN_PARALLEL_RATIO=0.5 bash run.sh         # concurrency gate ratio
TD_MULTI_TIMEOUT_S=900 bash run.sh         # barrier timeout (seconds)
```

`run.sh` checks that all tasks succeed and checksums match, **and** enforces the concurrency gate:
`host_max_inflight >= 2`, `device_parallelism >= INSTANCES * MIN_PARALLEL_RATIO`.

## Manual Build and Run

```bash
cmake -S . -B build \
    -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
    -DCANN_ROOT="${ASCEND_HOME_PATH}" \
    -DTORCH_PKG_DIR="$(python3 -c 'import os,torch;print(os.path.dirname(torch.__file__))')"
cmake --build build -j

# <epair> [workers=8] [tasks=48] [dim=1024], requires tasks >= 2*workers
./build/run_model_multi_instance matmul_model.epair 8 48 1024
```

## Where Device-Side Multi-Stream Comes From

**Cross-process multi-stream**: each process has its own ACL context and streams, so the execution streams of different instances are independent. When multiple processes call `RunModel` concurrently, the operators of each instance run in parallel on different streams.

> Note: automatic multi-stream within a GE graph (`ge.autoMultistreamParallelMode`) is an experimental feature and is not
> enabled in this example; device-side parallelism comes entirely from cross-process multi-streams.

## Observing Multi-Stream Parallelism with msprof (Optional)

```bash
msprof --output=./prof_multi_instance \
  --application="./build/run_model_multi_instance matmul_model.epair 8 48 1024"
```

Inspect the timeline: operators from different processes should land on different streams with overlapping execution intervals.

> Note: msprof's device-side operator collection is registered per device exclusively, so with multiple processes `op_summary` usually
> contains operators from only one worker (use the application-level `device_parallelism` metric to judge overall parallelism).

## Expected Results

- All `tasks` tasks succeed and output checksums match;
- `host_max_inflight >= 2` and `device_parallelism` close to `workers`, i.e. Device-side operators run on multiple streams in parallel.

> [!NOTE]
> A return from `RunModel` means the graph has finished executing and the outputs (host side) can be read directly; each instance is called only by
> its own worker process, and there is no GE shared lock between processes.
