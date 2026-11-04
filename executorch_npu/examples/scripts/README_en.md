# Example scripts

English | [简体中文](README.md)

Run these scripts from any working directory. The repository root is resolved
from the script location; Python and installed wheels come from the active environment.
Activate your Python environment before running them, for example:

```bash
conda activate executorch_npu
bash examples/scripts/run.sh qwen3_5 export-pt2 --help
```

Load your CANN and delegate setup scripts first, or provide their paths through
`CANN_ENV_SCRIPT` and `TORCH_DELEGATE_ENV_SCRIPT`. An existing `ASCEND_HOME_PATH`
with a `set_env.sh` is also recognized. Dataset downloads do not require CANN.

| Variable | Behavior |
|---|---|
| `EXECUTORCH_NPU_ROOT` | Optional repository root override; otherwise derived from this directory. |
| `CANN_ENV_SCRIPT` | Optional CANN setup script; use this to select a CANN version. |
| `TORCH_DELEGATE_ENV_SCRIPT` | Optional installed delegate setup script. |
| `TORCH_DELEGATE_REPO` | Optional delegate source checkout for Python/bridge library lookup; point it at the in-repo `../torch_delegate` when not installing the delegate whl. |
| `CANN_ET_CONDA_ENV` | Activate this conda environment only when explicitly set. |
| `CONDA_SETUP_SCRIPT` | Optional conda shell initialization script when conda is not on PATH. |
| `EXECUTORCH_SDK` | Optional installed ExecuTorch SDK directory; otherwise discovered from Python. |
| `FLATC_EXECUTABLE` | Optional flatc override; otherwise use the installed ExecuTorch wheel. |
| `CANN_ET_BUILD_DIR` | Runner build directory; defaults to `build/release` under the repository. |
| `CANN_ET_BUILD_JOBS` | Build parallelism; defaults to 8. |
| `CANN_ET_BOOTSTRAP_SCRIPT` | Optional user network setup for `download_mmlu_bootstrap.sh`. |

Proxy variables and Hugging Face cache configuration are inherited unchanged
from the caller. Configure them for your network before downloading datasets.
