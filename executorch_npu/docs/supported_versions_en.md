# Supported Versions

## Current Support Baseline

ExecuTorch 1.5.0 is the first official release supported by ExecuTorch-NPU. Earlier versions are outside
the support scope.

| Component | Supported version or range |
|---|---|
| Platform | Linux aarch64 |
| Python | 3.10–3.14; currently validated with 3.11 |
| ExecuTorch | `1.5.0` |
| PyTorch | CPU wheel `>=2.9.0,<2.14.0`; 2.11.0 is the default development profile |
| torchao | `>=0.18.0,<0.19`, installed by ExecuTorch |
| Torchvision | Matched to Torch, see the table below; required only for model examples |
| Transformers | `5.17.0`; required only for the Qwen3.5 / Gemma4 examples, kept consistent with `requirements-examples.txt` |
| CMake | Backend build `>=3.24` |
| C++ | C++17, currently validated with GCC 9.4 |
| CANN | 9.1.0 and 9.2.0 validation environments |
| Manifest framing | 1 |
| Runtime protocol | 1 |
| Artifact ABI | 1 |

Neither ExecuTorch 1.5 nor ExecuTorch-NPU installs Torch automatically. All supported combinations must first
explicitly install the CPU wheel from the official PyTorch CPU index. The default profile is:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.11.0+cpu" "torchvision==0.26.0+cpu"
python -m pip install "executorch==1.5.0"
```

Validated model dependency combinations:

| Torch | Torchvision |
|---|---|
| 2.9.0+cpu | 0.24.0+cpu |
| 2.10.0+cpu | 0.25.0+cpu |
| 2.11.0+cpu | 0.26.0+cpu |
| 2.12.0+cpu | 0.27.0+cpu |
| 2.13.0+cpu | 0.28.0+cpu |

Non-CPU wheels automatically selected by PyPI are prohibited. The PyPI Linux aarch64
wheels for Torch 2.11.0–2.13.0 pull in CUDA runtime libraries and do not belong to the Ascend export profile; 2.9.0 and 2.10.0 also uniformly use
the official CPU index to avoid the installation source varying by version.

The full ExecuTorch wheel must be used. The minimal wheel does not contain the headers and Runtime libraries
required for ExecuTorch-NPU builds and C++ application integration.

## Validation Status

Currently completed:

- Compiled the ExecuTorch-NPU Backend with the ExecuTorch 1.5.0 Linux aarch64 wheel.
- C++ unit tests pass.
- 74 Python core tests pass, covering EXIR lowering, named data, PTE/PTD serialization,
  multi-method preprocess, and the state contract.
- Compiled and linked the ordinary Runner and the Stateful Runner against the `libexecutorch.so` in the wheel.
- Torch 2.9.0+cpu, 2.10.0+cpu, 2.11.0+cpu, 2.12.0+cpu, and 2.13.0+cpu
  all completed the 74 Python core tests.
- Each of the Torch versions above rebuilt a standalone TorchAir in its corresponding environment and was validated through real MobileNetV2
  GE/OM compilation, PTE/PTD export, and the Ascend910B3 C++ Runner:
  `ASCEND_EXECUTORCH_OK values=1000`.

Still to be completed:

- ExecuTorch 1.5.0 end-to-end regression on other registered SoCs.
- ExecuTorch 1.5.0 real-NPU validation of Stateful Prefill/Decode.

Until these validations are complete, the documentation must not describe "Python core tests pass" as "all models and devices have completed
end-to-end validation".

## Older Version Boundaries

| ExecuTorch | Status | Reason |
|---|---|---|
| 0.x | Not supported | PyPI has no current Linux aarch64 wheel; the interfaces and artifact protocols predate the current implementation |
| 1.0.0–1.2.0 | Not supported | The wheel lacks `executorch/runtime/backend/interface.h`; the current default build cannot run |
| 1.3.1–1.4.1 | Not supported | The Backend can compile, but the wheel lacks the standalone C++ Runtime SDK required by the current Runner |
| 1.5.0 | Support baseline | The wheel provides the complete C++ SDK for the first time; the build, core export interfaces, and Runner compilation are validated |

Even if an older version passes some Python tests, that alone does not make it a supported version.

## Nightly

Nightly is only for forward-compatibility validation and is outside the stable support scope. Torch must be explicitly included when installing:

```bash
python -m pip install --upgrade --pre executorch torch \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

CI should record the full nightly version and git commit, for example:

```text
executorch-1.6.0.dev20260916+cpu
ac4f33f90f7dff650e3c982fa34a3cc84848b366
```

Nightly build results cannot automatically expand the stable-version support scope.

## Validated Devices

The following ACL SoC names have been validated:

```text
Ascend910_9362
Ascend910B3
Ascend910B4
Ascend310P3
Ascend310B1
```

Loading succeeds when the `acl_soc_name` recorded in the exported artifact exactly matches the return value of
`aclrtGetSocName()` on the running device; mismatches are rejected (`DelegateInvalidCompatibility`).
