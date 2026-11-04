# cann-local-lm

English | [简体中文](README.md)

## 🔥 Latest News

- [2026/10] Converter current capabilities: HuggingFace model → epair export (torch.export + FX→GE conversion + weight separation), supporting W4A8 quantization (GPTQ offline + runtime activated dynamic quantization). Engine current capabilities: three-layer inference engine architecture, supporting KV Cache management, asynchronous streaming inference, multi-turn continuation, request cancellation, adapted to Kirin platform, providing C++ API and Python API entry points.

## 🚀 Overview

cann-local-lm aims to provide large model conversion tools and end-to-end inference engines based on CANN for edge devices, enabling lightweight, low-latency, and low-memory efficient inference for various large language models.

## ⬇️ Source Code Download

After the environment is prepared, download the branch source code matching the CANN version. The general command is as follows, replace `${tag_version}` with the branch tag name:

```bash
# General command: git clone -b ${tag_version} https://gitcode.com/cann/cann-local-lm.git
git clone -b 2026_09_22 https://gitcode.com/cann/cann-local-lm.git
```

> Note: If the matching branch source code already exists in the environment, you can skip this step.

## 📦 Supported Models and Chips

| Model    | Architecture Features                                  | Supported Chips |
| ---------| ------------------------------------------------------|-----------------|
| gemma4   | KV sharing + sliding window + PLE (Per-Layer Embedding)| KirinX90        |

## 📂 Directory Structure

```
cann-local-lm/
├── cann_local_lm/                  # Core source code
│   ├── converter/            # Export toolchain (HF model → epair → omc compilation)
│   ├── engine/               # Inference engine (omc loading + inference execution)
│   └── models/               # Model definitions (gemma4 + fused_ops operators)
├── torch_delegate/           # torch-delegate: epair serialization/loading + C++ backend + .venv build
│   ├── csrc/                 # C++ backend (epair archive loading → GE Graph → OM)
│   ├── torch_delegate/       # Python package (api.py + converter/ + epair/ + core/)
│   ├── cmake/                # torch-delegate dedicated CMake modules
│   ├── docs/                 # torch-delegate design docs + build guide
│   ├── tests/                # torch-delegate unit tests (C++ GTest + Python pytest)
│   ├── build.sh              # One-stop build (deps → cpp → wheel → .run)
│   └── examples/             # Examples (export loading/accuracy comparison/async execution)
├── executorch_npu/           # ExecuTorch-NPU: ExecuTorch Ascend AOT Backend (PyTorch → PTE/PTD → NPU execution)
│   ├── python/executorch_npu/ # Python package (AscendPartitioner + compiler + passes + artifacts)
│   ├── runtime/              # C++ backend (AscendBackend static registration + Manifest/SHA-256 + Stateful API)
│   ├── tools/                # C++ validation runners (regular/stateful + GDN operator registration)
│   ├── examples/             # Examples (mobilenet_v2 / qwen3_5 / gemma4 + MMLU evaluation)
│   ├── docs/                 # Bilingual docs (build/export/runtime/design/troubleshooting/supported versions)
│   ├── tests/                # Unit tests (C++ GTest + Python pytest)
│   ├── third_party/          # nlohmann/json (vendored, MIT)
│   └── build.sh              # One-stop build (C++ backend → ctest → wheel)
├── docs/                     # Design docs (converter + engine + build + version evolution)
├── examples/                 # End-to-end use cases (omc compilation + NPU inference verification)
├── scripts/                  # Build scripts (build.sh + build.conf + set_env*.sh + cann_lm_compile/)
├── tests/                    # Unit tests (cpp/ + python/)
├── ci/                       # CI/CD
└── cmake/                    # CMake common modules (modules/ + third_party/ + TorchDelegatePackage)
```

### Directory Introduction

| Directory                     | Responsibility                        | Key Files                                                                                                                                                              |
| ------------------------------| --------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **cann_local_lm/converter/**        | HF model → epair → omc export toolchain | `cann_lm_converter.py`(CLI entry) / `loader.py`(loading) / `saver.py`(saving) / `quant/`(quantization:swap/materialize/auto_quant/dopt_wrapper) / `export/`(ge_converter + C++ csrc) |
| **cann_local_lm/engine/**           | omc inference engine                   | `api/`(Python interface) / `core/`(core logic) / `backend/`(C++ backend) / `pipeline/`(inference pipeline)                                                              |
| **cann_local_lm/models/**           | Model definitions + custom operators   | `gemma4/`(Gemma4ForCausalLM + export_wrapper) / `fused_ops/`(cann_local_lm:: + npu:: operator schema+fake-meta)                                                              |
| **torch_delegate/**           | epair serialization/loading + C++ backend | `csrc/backend/`(EpairModelLoader → GE Graph → OM) / `torch_delegate/`(api.py + converter + epair + core) / `build.sh`(one-click build) / `examples/`(examples)        |
| **executorch_npu/**           | ExecuTorch Ascend AOT Backend      | `python/executorch_npu/`(AscendPartitioner + AscendBackend.preprocess: EP→EPAIR→GE→OM→PTE/PTD) / `runtime/`(C++ backend: PTE/PTD loading + Manifest validation + ACL execution) / `tools/`(C++ runners) / `examples/`(mobilenet_v2/qwen3_5/gemma4) / `build.sh`(one-click build) |
| **docs/cann_lm_converter/**   | converter design docs                  | `cli/cli_design.md` / `export/export_design.md` / `quant/quant_design.md` / `README.md`(overview) / `version_evolution.md`(version evolution)                        |
| **examples/test_converter/**  | End-to-end test cases                  | `export_llm_model/`(C++ omc compilation + om_npu_verify inference verification) / `cann_lm_cli_export/run_pd6_cases.sh`(6 cases)                                      |
| **scripts/**                  | Build scripts                          | `build.sh`(top-level check/deps/third_party/cpp/tests/wheel/package/all) / `build.conf`(versions+git source+self-locating derived paths) / `set_env.sh`(converter+compile unified env, real CANN no stubs) / `set_env_cann.sh`(③ NPU) / `_env_common.sh`(common) / `cann_lm_compile/build_cpp.sh`(C++ .so+cann_lm_compile) / `build_test.sh`(ASAN tests) / `build_third_party.sh`(self-build json/gtest/makeself) / `cann_lm_{converter,engine}/build_wheel.sh`(whl packaging) |
| **tests/**                    | Unit tests                             | `cpp/`(C++ GTest; run: `bash scripts/build.sh tests` → ASAN+ctest) / `python/`(pytest)                                                                                |
| **ci/**                       | CI/CD                                  | Pipeline definitions, build scripts, image configurations                                                                                                             |
| **cmake/**                    | CMake shared                           | `modules/`(DelegateCommonFlags + Find*.cmake + TorchDelegateCannCommon) / `third_party/`(ExternalProject download json/gtest/makeself)                                |

## ⚡️ Quick Start

See [Quick Start](docs/en/QUICKSTART.md) for details, covering installation, epair export, quantization export, omc compilation, and deployment to HarmonyOS devices.

## 📚 Documentation

| Document | Content |
|----------|---------|
| [Quick Start](docs/en/QUICKSTART.md) | Installation / epair export / quantization export / omc compilation / deployment to HarmonyOS |
| [Build Guide](docs/en/build_guide.md) | Build prerequisites / one-click build+install+usage / quick reference / FAQ |
| [Environment Dependencies](docs/en/env_dependencies.md) | Build/runtime dependency classification / P1 venv / P2 torch_npu / multi-env quick reference table |
| [Version Evolution](docs/en/version_evolution.md) | Release cadence / compatibility table (HarmonyOS / CANN / chip / PyTorch / TorchNPU) |
| [CLI Design](docs/en/cann_lm_converter/cli/cli_design.md) | CLI entry / convert() / command-line arguments / call chain |
| [Export Design](docs/en/cann_lm_converter/export/export_design.md) | loader / saver / ExportWrapper / C++ loading |
| [Quantization Design](docs/en/cann_lm_converter/quant/quant_design.md) | W4A8 scheme / lightweight tool GPTQ / swap / operator specs |
| [Converter Overview](docs/en/cann_lm_converter/README.md) | Cross-module call chain / model differences / accuracy summary |

## 🌐 Ecosystem Integration

The following projects have upstream/downstream dependencies or collaborative relationships with cann-local-lm:

- **[GE (Graph Engine)](https://gitcode.com/cann/ge)**: CANN Graph Engine; cann-local-lm's FX→GE conversion and omc compilation are both based on GE capabilities
- **[ops-math](https://gitcode.com/cann/ops-math)**: CANN numerical computation operator library, providing basic math operator support for model export
- **[ops-nn](https://gitcode.com/cann/ops-nn)**: CANN neural network operator library, providing NN layer operator support for model export
- **[ops-transformer](https://gitcode.com/cann/ops-transformer)**: CANN LLM operator library, providing Transformer structure operator support for model export
- **[TorchAir](https://gitcode.com/Ascend/torchair)**: Integrates GE into PyTorch graph mode; cann-local-lm converter completes FX→GE conversion via torchair
- **[CANN Toolkit](https://hiascend.com/software/cann)**: CANN software package (atc / acl), providing omc compilation and NPU runtime

## 📝 Related Information

- [Contributing Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
- [Build Guide](docs/en/build_guide.md)
- [Affiliated SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/device-ai)

## 💬 Feedback and Communication

- **Issue Feedback**: Submit issues via GitCode [Issues](https://gitcode.com/cann/cann-local-lm/issues)
- **Community Interaction**: Participate in discussions via GitCode [Discussions](https://gitcode.com/cann/cann-local-lm/discussions)

---

PS: The features and documentation of this project are continuously being updated and improved. Stay tuned for the latest version.
