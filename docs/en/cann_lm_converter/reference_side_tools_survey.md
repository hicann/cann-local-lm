# Side LLM Tool CLI and Documentation Survey

English | [简体中文](../../zh/cann_lm_converter/reference_side_tools_survey.md)

> Survey of 6 side/conversion LLM tools' CLI entry / command style / doc structure / command reference / capabilities, compared with cann-lm-converter.
> Code cloned at `/mnt/workspace/side_tools/` (shallow --depth 1, skip LFS).
>
> Survey date: 2026-09-01 | Related: [CLI-013](./TODO_notes.md#cli-013-todo-side-llm-tool-cli-and-doc-delivery-survey)

## Table of Contents

- [Survey Objects](#survey-objects)
- [1. mlx-lm (Apple, Python+MLX)](#1-mlx-lm-apple-pythonmlx)
- [2. ollama (Go)](#2-ollama-go)
- [3. executorch (PyTorch side)](#3-executorch-pytorch-side)
- [4. mlc-llm (MLC)](#4-mlc-llm-mlc)
- [5. litert-lm (Google, Bazel)](#5-litert-lm-google-bazel)
- [6. coremltools (Apple)](#6-coremltools-apple)
- [Comparison Table (6 tools + cann-lm)](#comparison-table-6-tools-cann-lm)
- [cann-lm-converter comparison + implications](#cann-lm-converter-comparison-implications)
- [Key difference: export tool vs side runtime](#key-difference-export-tool-vs-side-runtime)

---

<a id="survey-objects"></a>
## Survey Objects

| Tool | Positioning | Language/stack | Code path |
|------|------|---------|----------|
| mlx-lm | Full-chain side runtime (convert+quantize+inference+serve+fine-tune) | Python+MLX (Apple Silicon) | `/mnt/workspace/side_tools/mlx-lm/` |
| ollama | Full-chain side runtime (run+serve+pull) | Go (cobra) | `/mnt/workspace/side_tools/ollama/` |
| executorch | Full-chain side deployment engine (export+quantize+inference+serve) | C++/Python (PyTorch) | `/mnt/workspace/side_tools/executorch/` |
| mlc-llm | Full-chain side deployment engine (compile+serve+multi-platform SDK) | C++/Python (TVM) | `/mnt/workspace/side_tools/mlc-llm/` |
| litert-lm | Consumer-side orchestration (consumes pre-exported .litertlm, inference+serve) | Python (click, Bazel) | `/mnt/workspace/side_tools/litert-lm/` |
| coremltools | Conversion tool (PyTorch/TF→CoreML) | Python (C API) | `/mnt/workspace/side_tools/coremltools/` |

---

<a id="1-mlx-lm-apple-pythonmlx"></a>
## 1. mlx-lm (Apple, Python+MLX)

### CLI Entry

**setup.py `console_scripts` 18** (dot multi-entry + main dispatcher):

| console_script | Points to |
|---|---|
| `mlx_lm` | `mlx_lm.cli:main` (main dispatcher) |
| `mlx_lm.generate`/`.chat`/`.convert`/`.server`/`.cache_prompt`/`.benchmark`/`.evaluate`/`.perplexity` | `mlx_lm.<cmd>:main` |
| `mlx_lm.lora`/`.fuse` | `mlx_lm.<cmd>:main` (fine-tune) |
| `mlx_lm.awq`/`.dwq`/`.dynamic_quant`/`.gptq` | `mlx_lm.quant.<cmd>:main` (learned quantization) |
| `mlx_lm.manage`/`.upload`/`.share` | `mlx_lm.<cmd>:main` (model management) |

`mlx_lm/cli.py` (53 lines) is a **hand-written subcommand dispatcher**: `sys.argv.pop(1)` takes subcommand + `importlib.import_module(f"mlx_lm.{subcommand}")` dynamic loading + calls `submodule.main()`. **Not** argparse subparsers/typer/click.

### Command Style

**Dual-track** (each command two forms, same behavior):
- Dot multi-entry: `mlx_lm.generate --prompt "..."`
- Subcommand: `mlx_lm generate --prompt "..."` (via cli.py dispatch)

README mainly promotes dot form.

### 17 Commands List

| Command | Capability | Module |
|------|------|------|
| generate | One-shot inference | `mlx_lm/generate.py` |
| chat | Interactive REPL | `mlx_lm/chat.py` |
| convert | HF→MLX conversion+basic quant+upload | `mlx_lm/convert.py` |
| server | OpenAI-compatible HTTP API | `mlx_lm/server.py` |
| cache_prompt | Long-context prompt cache | `mlx_lm/cache_prompt.py` |
| benchmark | tok/sec+memory benchmark | `mlx_lm/benchmark.py` |
| evaluate | MMLU Pro eval (lm-eval) | `mlx_lm/evaluate.py` |
| perplexity | Perplexity | `mlx_lm/perplexity.py` |
| lora | LoRA/QLoRA/DoRA/full fine-tune | `mlx_lm/lora.py` + `tuner/` |
| fuse | Fuse adapter+GGUF export | `mlx_lm/fuse.py` |
| awq/dwq/dynamic_quant/gptq | 4 learned quantization | `mlx_lm/quant/` |
| manage | HF cache scan/delete | `mlx_lm/manage.py` |
| upload | Upload HF Hub | `mlx_lm/upload.py` |
| share | Local share endpoint | `mlx_lm/share.py` |

### Doc Structure

**No independent docs site** (no mkdocs/sphinx). Two layers:
- Top-level `README.md` (290 lines): Quick Start / Python API / Streaming / Sampling / Command Line / Long Prompts / Supported Models / Large Models
- `mlx_lm/` files by feature (not by command):
  - `SERVER.md` (server OpenAI API field table)
  - `LORA.md` (lora+fuse fine-tune/data format/GGUF)
  - `LEARNED_QUANTS.md` (awq/dwq/dynamic_quant/gptq comparison/tuning)
  - `BENCHMARKS.md` (benchmark/evaluate)
  - `MANAGE.md` (manage)
- `mlx_lm/examples/` (9 files: generate_response.py/chat.py/batch_generate/sharded/tool_use etc.)

### Command Reference

**`--help` is authoritative** (argparse `help=`), README repeatedly guides "`mlx_lm.<cmd> --help`". Key commands have independent `.md` with **hand-written param tables** (SERVER.md Request/Response Fields). **No auto-generated param tables**. All 18 `main()` use stdlib `argparse`, no third-party CLI framework.

### Main Capabilities

| Capability | Implementation |
|------|------|
| Convert | `convert` (HF→MLX, `-q` basic quant, `--upload-repo`) |
| Quantize | Basic (`convert -q`) + 4 learned quant (awq/dwq/dynamic_quant/gptq) + KV cache quant (`--kv-bits`) |
| Inference | `generate` (batch)/`chat` REPL/`cache_prompt`/speculative decoding (`--draft-model`)/rotary KV (`--max-kv-size`)/prefill stepping |
| serve (OpenAI API) | `server` (port 8080, `/v1/chat/completions`, streaming/logprobs/draft_model) |
| Fine-tune | `lora` (LoRA/QLoRA/DoRA/full, YAML config, wandb) + `fuse` (fuse+GGUF) |
| Eval | `benchmark`/`evaluate` (MMLU Pro)/`perplexity` |
| Manage | `manage`/`upload`/`share` |
| GGUF | `fuse --export-gguf` (Mistral/Llama fp16) |
| Distributed | `mx.distributed` (`examples/sharded_generate.py`) |
| Tool calling | `mlx_lm/tool_parsers/` (13 parsers: pythonic/gemma4/glm47 etc.) |

### Key Files

- `setup.py` (18 console_scripts)
- `mlx_lm/__main__.py` (`python -m mlx_lm`)
- `mlx_lm/cli.py` (main dispatcher)
- `mlx_lm/__init__.py` (Python API: load/generate/stream_generate/convert)
- `mlx_lm/{generate,chat,convert,server,cache_prompt,benchmark,evaluate,perplexity,lora,fuse,manage,upload,share}.py`
- `mlx_lm/quant/{awq,dwq,dynamic_quant,gptq}.py`
- `mlx_lm/tuner/` (lora/dora/trainer/datasets)
- `mlx_lm/models/` (50+ models)
- `mlx_lm/tool_parsers/` (13 parsers)
- `README.md` + `mlx_lm/{SERVER,LORA,LEARNED_QUANTS,BENCHMARKS,MANAGE}.md`
- `mlx_lm/examples/` (9 files)

### Param Tables + Code Details (in-depth survey 2026-09-01)

**Code details**:
- `setup.py:52-72` console_scripts 18: 1 dispatcher `mlx_lm = mlx_lm.cli:main` + 17 dot entry direct to each module's `main` (4 quant commands `mlx_lm.awq/dwq/dynamic_quant/gptq` point to `mlx_lm.quant.<cmd>:main`)
- `cli.py:7-53` main dispatcher: `sys.argv.pop(1)` takes subcommand (modifies argv to pass through rest) + `importlib.import_module(f"mlx_lm.{subcommand}")` on-demand dynamic load + `submodule.main()`; `subpackages` dict maps 4 quant commands to `quant` subpackage; `--version`/`-h`/`--help` special-cased; miss raises ValueError. **Doesn't use** argparse subparsers/click/typer
- `__main__.py:3-5`: `python -m mlx_lm` → `cli.main()`
- **Dual-track**: dot entry (`mlx_lm.generate`) direct to module main vs subcommand (`mlx_lm generate`) via dispatcher, both call same `submodule.main()`, behavior consistent; README promotes dot
- **Single command main**: `setup_arg_parser()` (argparse definition independent function, reusable) + `main()` calls `parser.parse_args()` (generate.py example)

#### `mlx_lm generate` (29 params)

Model/loading: `--model` `--trust-remote-code` `--adapter-path` `--quantize-activations`/`-qa`

Input/prompt: `--prompt`/`-p` (`-` reads stdin) `--system-prompt` `--prefill-response` `--extra-eos-token`(nargs+) `--ignore-chat-template` `--use-default-chat-template` `--chat-template-config`(JSON)

Generation/sampling: `--max-tokens`/`-m` `--temp` `--top-p` `--min-p` `--top-k` `--xtc-probability` `--xtc-threshold` `--min-tokens-to-keep` `--seed`

KV cache: `--max-kv-size`(rotary KV) `--kv-bits` `--kv-group-size`(default 64) `--quantized-kv-start` `--prompt-cache-file` `--prefill-step-size`

Speculative decoding: `--draft-model` `--num-draft-tokens`(default 3)

Output: `--verbose`(str2bool default True)

#### `mlx_lm convert` (11 params, HF→MLX convert+quant+upload)

| Param | Type | Default | Description |
|---|---|---|---|
| `--hf-path`/`--model` | str | — | HF model path/repo (required) |
| `--mlx-path` | str | mlx_model | MLX save path |
| `--upload-repo` | str | — | Upload HF repo |
| `--trust-remote-code` | flag | False | Trust remote code |
| `-q`/`--quantize` | flag | — | Generate quantized model |
| `--q-group-size` | int | None | Quant group size |
| `--q-bits` | int | None | Bits per weight |
| `--q-mode` | str | affine | Quant mode: affine/mxfp4/nvfp4/mxfp8 |
| `--quant-predicate` | str | — | Mixed-bit recipe (choices=QUANT_RECIPES) |
| `-d`/`--dequantize` | flag | False | Dequantize |
| `--dtype` | str | None | Non-quant param type (choices=MODEL_CONVERSION_DTYPES) |

#### `mlx_lm.awq` (11 params, AWQ learned quantization)

| Param | Type | Default | Description |
|---|---|---|---|
| `--model`/`-m` | str | mlx-community/Qwen2.5-7B-Instruct-bf16 | Model |
| `--mlx-path` | str | mlx_model | Save path |
| `--bits` | int | 4 | Weight bits |
| `--group-size` | int | 64 | Weight group size |
| `--embed-bits` | int | 4 | Embedding bits |
| `--embed-group-size` | int | 32 | Embedding group size |
| `--num-samples` | int | 128 | Calibration sample count |
| `--sequence-length` | int | 512 | Calibration sequence length |
| `--n-grid` | int | 20 | AWQ search grid |
| `--seed` | int | 123 | Random seed |
| `--trust-remote-code` | flag | — | Trust remote code |

#### `mlx_lm.gptq` (10 params, GPTQ learned quantization)

| Param | Type | Default | Description |
|---|---|---|---|
| `--model`/`-m` | str | <hf_model_id> | Model |
| `--mlx-path` | str | mlx_model | Save path |
| `--bits` | int | 4 | GPTQ layer bits |
| `--group-size` | int | 64 | GPTQ layer group size |
| `--fallback-bits` | int | 6 | **Non-GPTQ layer** bits |
| `--fallback-group-size` | int | 64 | Non-GPTQ layer group size |
| `--num-samples` | int | -1 | Calibration sample count (-1 all) |
| `--sequence-length` | int | 512 | Calibration sequence length |
| `--seed` | int | 123 | Random seed |
| `--trust-remote-code` | flag | — | Trust remote code |

**awq vs gptq**: 8 common (`--model`/`--mlx-path`/`--bits`/`--group-size`/`--num-samples`/`--sequence-length`/`--seed`/`--trust-remote-code`); awq unique `--embed-bits`/`--embed-group-size` (embedding separately quantized) + `--n-grid` (search grid); gptq unique `--fallback-bits`/`--fallback-group-size` (non-GPTQ layer fallback quant, mixed bit width).

**vs cann-lm-converter**: mlx-lm awq/gptq are independent quant commands (params directly control bits/group/embed/fallback), produces MLX quantized model; cann-lm folds quant into `--quant-method gptq` (spawns CANN lightweight tool subprocess), quant params (W4A8/group128/A8) determined by dopt_config `Quant_INT8-4_ECO` strategy, CLI doesn't expose. mlx-lm flexible (any bits/group), cann-lm fixed (W4A8 strategy).

---

<a id="2-ollama-go"></a>
## 2. ollama (Go)

### CLI Entry

**Go binary (cobra)**, no console_scripts:
- `main.go` (13 lines): `cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))`
- `NewCLI()` in `cmd/cmd.go:2286` (registers all subcommands, `rootCmd.AddCommand(...)` at once)
- Build: `cmake -B build . && cmake --build build` (incl. C++ native payload) or `go build .` (pure Go)

### Command Style

**Unified `ollama <cmd>` (cobra)**, 13 subcommands + launch integration + independent bench:

| Subcommand | Use | Handler | Key flags |
|--------|-----|---------|-----------|
| `serve` (alias `start`) | serve | RunServer | 20+ `OLLAMA_*` env (appendEnvDocs) |
| `create` | create MODEL | CreateHandler | `-f/--file`, `-q/--quantize`, `--experimental` |
| `show` | show MODEL | ShowHandler | `--license/--modelfile/--parameters/--system` |
| `run` | run MODEL [PROMPT] | RunHandler | `--keepalive`, `--format`, `--think`, `--truncate` |
| `stop` | stop MODEL | StopHandler | — |
| `pull`/`push` | pull/push MODEL | Pull/PushHandler | `--insecure` |
| `signin`/`signout` (hidden, alias login/logout) | signin/signout | Signin/SignoutHandler | — |
| `list` (alias `ls`)/`ps` | list/ps | List/ListRunningHandler | — |
| `cp`/`rm` | cp/rm | Copy/DeleteHandler | — |
| `runner`/`gpu-discover` (hidden) | runner/gpu-discover | — | passthrough |
| `launch` | launch [INTEGRATION] | LaunchIntegration | `--model`, `--config`, `--yes` |

`launch` integrates 18+ apps (claude/codex/codex_app/opencode/copilot/droid/deepseek_harness/cline/kimi/muse/pi/poolside/qwen/vscode/hermes/openclaw/omp/claude_desktop), each integration a `cmd/launch/<name>.go`.

Independent binaries: `bench` (`cmd/bench/`, Go flag, not cobra), `runner` (`cmd/runner/`, llama.cpp runner subprocess).

### Doc Structure

- Repo `README.md` (356 lines): Download / Get started / Coding / AI assistant / Chat / REST API / Python/JS / Backends / Documentation / Community Integrations (very long)
- `docs/` (Mintlify-driven, deployed docs.ollama.com):
  - `docs.json` (site config, 3 tabs: Guide/Integrations/API Reference)
  - `cli.mdx` (CLI Reference, example-style)
  - `api.md` (54KB REST API) + `openapi.yaml` (58KB)
  - `modelfile.mdx` (Modelfile Reference)
  - `docs/api/` (7 mdx: introduction/authentication/streaming/usage/errors/openai-compat/anthropic-compat)
  - `docs/capabilities/` (7 mdx: embeddings/streaming/structured-outputs/thinking/tool-calling/vision/web-search)
  - `docs/integrations/` (27 mdx: claude-code/codex/opencode etc.)
  - `development.md`/`quickstart.mdx`/`faq.mdx`/`gpu.mdx`/`cloud.mdx` etc.

### Command Reference

Three-layer complementary:
1. `--help` (cobra auto): `serve` appends 20+ `OLLAMA_*` env (`OLLAMA_HOST`/`OLLAMA_DEBUG`/`OLLAMA_CONTEXT_LENGTH`/`OLLAMA_KEEP_ALIVE`/`OLLAMA_NUM_PARALLEL`/`OLLAMA_FLASH_ATTENTION` etc.)
2. `docs.ollama.com/cli` (`docs/cli.mdx`): example-driven (### Run a model → code block)
3. README inline (`ollama run gemma4`/`ollama launch claude`/`curl` etc.)

### Main Capabilities

| Capability | Implementation |
|------|------|
| Run | `ollama run MODEL [PROMPT]` (interactive REPL, `cmd/interactive.go`) |
| serve (REST API) | `ollama serve` (OpenAI-compatible `/v1/*` + Anthropic-compatible) |
| pull/push | registry models |
| create (Modelfile) | `ollama create -f Modelfile` (`--quantize`/`--experimental`) |
| Embeddings | `ollama run embeddinggemma`/`nomic-embed-text` (`--truncate`/`--dimensions`) |
| Multimodal | `ollama run gemma4 "What's in this image? /path.png"` |
| thinking | `run --think true|high|medium|low`/`--hidethinking` |
| Integration launch | `ollama launch [INTEGRATION]` (Claude Code/Codex/OpenCode/Copilot/Droid etc.) |
| cloud models | `:cloud` suffix |
| Python/JS SDK | `ollama-python`/`ollama-js` |

### Key Files

- `main.go` (Go binary entry)
- `cmd/cmd.go` (`NewCLI()` + 13 Handlers)
- `cmd/launch/` (`launch.go` + 18 integrations `*.go` + `registry.go`)
- `cmd/runner/main.go` (llama.cpp runner subprocess)
- `cmd/bench/bench.go` (independent benchmark)
- `cmd/interactive.go` (run REPL)/`cmd/agent_tui.go` (root no-arg TUI)
- `api/` (REST server)/`openai/`/`anthropic/` (compat layer)
- `README.md` + `docs/` (Mintlify site)

---

<a id="3-executorch-pytorch-side"></a>
## 3. executorch (PyTorch side)

### CLI Entry

**pyproject.toml `[project.scripts]` only `flatc`** (FlatBuffers compiler), no top-level dispatcher:
```
[project.scripts]
flatc = "executorch.data.bin:flatc"
```
All tools run as **`python -m <module.path>`** independently, no `executorch <subcommand>` aggregate entry, no `__main__.py`. Each tool has its own `argparse` + `if __name__ == "__main__": main()`.

C++ binaries built scattered (`examples/models/llama/main.cpp`/`extension/llm/runner/`), via `CMakeLists.txt` + `CMakePresets.json` + `Makefile` to `cmake-out/`.

### Command Style

**Mixed: flat single-script + Hydra dot-config**:

| Style | Tool | Description |
|------|------|------|
| `python -m module` + argparse | inspector_cli/diff_pte/size_analysis_tool/llm_server/install_executorch | Each one ArgumentParser, no subparsers |
| Hydra dot + YAML | `export_llm` (main LLM export) | `base.model_class=llama3` `quantization.qmode=8da4w` `--config xxx.yaml` |
| Makefile target + CMake preset | C++ runner | `make llama-cpu`/`make llama-cuda-debug` |

Main "commands" (independent modules):

| Command | Entry | Style |
|------|------|------|
| export (LLM) | `python -m executorch.extension.llm.export.export_llm` | Hydra + `--config` |
| export (Llama legacy) | `python -m executorch.examples.models.llama.export_llama` | argparse + Hydra dual-mode |
| serve | `python -m executorch.examples.llm_server.python.server` | argparse (FastAPI) |
| finetune | `python examples/llm_pte_finetuning/runner.py` | argparse + torchtune YAML |
| inspect | `python -m executorch.devtools.inspector.inspector_cli` | argparse |
| diff pte | `python -m executorch.devtools.pte_tool.diff_pte` | positional args |
| size analysis | `python -m executorch.devtools.size_analysis_tool.size_analysis_tool` | argparse |
| install | `python install_executorch.py` | argparse |
| flatc (sole console_script) | `flatc` | — |

No `convert`/`run` top-level verb. "export"=export, "run"=runner(C++)/server, "convert" via export_llm internal 7 steps (load→quantize→export→partition→.pte).

### Doc Structure

**Sphinx primary** (`docs/source/`) + README + per-dir README + `.claude`/`.wiki`:

```
README.md                    # Root (Quick Start 3 steps + platform table + model list)
docs/
├── README.md                # Sphinx build instructions
├── Makefile                 # sphinx
  source/
    ├── index.md             # toctree (intro/quick-start/edge/backends/llm/advanced/tools/api)
    ├── *.md (60+ files, flat) # organized by section
    ├── llm/                 # LLM topic (11 articles: export-llm/run-with-c-plus-plus/run-on-android etc.)
    ├── backends/            # per-backend docs
    └── tutorials_source/   # tutorials
examples/                    # end-to-end cases (by topic)
    ├── models/              # model zoo (60+: llama/gemma3/whisper/stable_diffusion etc.)
    ├── portable/ selective_build/ devtools/ demo-apps/
    ├── xnnpack/ arm/ qualcomm/ samsung/ cadence/ apple/ mediatek/ nxp/
    └── llm_server/ llm_pte_finetuning/
.claude/  .wiki/             # skills reference + tribal knowledge (2200+ issue synthesis)
```

### Command Reference

**No unified command reference page**, scattered:
1. `--help` (argparse auto)
2. Per-tool README (`extension/llm/export/README.md` Hydra usage + YAML; `examples/llm_server/python/README.md` server flag; `examples/llm_pte_finetuning/README.md` fine-tune; `extension/llm/runner/README.md` 33KB C++ runner API)
3. `docs/source/llm/export-llm.md` (LLM export tutorial-style)
4. `Makefile help` target (C++ runner build targets)

Param table style: Hydra tools use YAML + dot CLI overrides (schema in `LlmConfig` dataclass); plain tools classic argparse; LLM server kebab-case flags.

### Main Capabilities

| Capability | Implementation |
|------|------|
| Export (torch.export→edge→.pte) | `extension/llm/export/` (export_llm.py / builder.py LLMEdgeManager / export_passes.py / partitioner_lib.py); 7 steps: instantiate→load weights→transform/quantize→torch.export→pt2e→partition→.pte |
| Quantize | `extension/llm/export/quantize.py` (dual-path: TorchAO source-level `8da4w`/`int8` + pt2e graph-level QNN/CoreML/Vulkan) |
| Inference (C++ runtime) | `extension/llm/runner/` (TextLLMRunner/MultiModalRunner + pybindings `Runtime.get()`; KV cache/attention sink/dynamic shape) |
| serve (OpenAI-compatible) | `examples/llm_server/` (FastAPI control plane + C++ worker JSONL; `/v1/chat/completions` streaming + tool calling Hermes/Qwen XML) |
| Fine-tune | `examples/llm_pte_finetuning/` (torchtune LoRA, first .pte checkpoint→train, CPU/fp32) |
| Model zoo | `examples/models/` (60+: llama/gemma3/phi_4_mini/llava/whisper/stable_diffusion etc.) |
| DevTools | `devtools/` (inspector/etdump/etrecord/pte_tool/size_analysis/visualization/bundled_program) |
| Multi backend | `backends/` (xnnpack/coreml/qnn/mediatek/samsung/arm/cadence/nxp/vulkan/openvino/cuda/metal/mlx) |
| Cross-platform SDK | `extension/apple/`/`extension/android/` + Swift Package |

### Key Files

- `pyproject.toml` (`[project.scripts]` only flatc)
- `setup.py` (130KB dynamic deps + wheel)
- `CMakeLists.txt` (63KB)/`CMakePresets.json`/`Makefile`
- `install_executorch.py` (install CLI)
- `extension/llm/export/export_llm.py` (Hydra main)/`builder.py` (LLMEdgeManager)/`config/llm_config.py` (LlmConfig schema)
- `extension/llm/runner/` (C++ runtime + pybindings + README 33KB)
- `examples/llm_server/python/server.py` (FastAPI serve)
- `examples/llm_pte_finetuning/` (fine-tune)
- `devtools/` (inspector_cli/diff_pte/size_analysis)
- `docs/source/` (Sphinx 60+ md)+ `examples/models/` (60+ models)

---

<a id="4-mlc-llm-mlc"></a>
## 4. mlc-llm (MLC)

### CLI Entry

**No `[project.scripts]`** (`pyproject.toml` full-text search `project.scripts`/`console_scripts`/`entry_points` all miss). `mlc_llm` command comes from **alias**:
```bash
export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
alias mlc_llm="python -m mlc_llm"
```
Doc fallback: "command not found: mlc_llm, try `python -m mlc_llm --help`".

Real entry: `python/mlc_llm/__main__.py` (67 lines) — **hand-written argv[1] dispatch**: `ArgumentParser` parses `sys.argv[1:2]` to get subcommand (choices hardcoded 8), `if/elif` lazy import `mlc_llm.cli.<name>` calls `main(sys.argv[2:])`. **Not** typer/click/argparse subparsers.

C++ `CMakeLists.txt` only builds libs (`libmlc_llm.so`/`mlc_llm_static`/`mlc_llm_module` + optional `mlc_llm_cpp_tests`), **no CLI binary**.

### Command Style

**Subcommand `mlc_llm <cmd>` (argparse hand-written)**, 8:

| Subcommand | Purpose | Key params |
|--------|------|---------|
| `compile` | Compile model→model lib (.so/.dylib/.dll/.tar/.wasm) | `model`/`--quantization`/`--model-type`/`--device`/`--host`/`--opt`/`--output` |
| `convert_weight` | HF weights→MLC quant format (params_shard_*.bin+tensor-cache.json) | `config`/`--quantization`/`--source`/`--source-format`/`--output`/`--lora-adapter` |
| `gen_config` | Generate mlc-chat-config.json + tokenizer | `config`/`--quantization`/`--conv-template`/`--context-window-size`/`--output` |
| `chat` | Interactive CLI chat | `model`/`--device`/`--model-lib`/`--overrides` |
| `serve` | OpenAI-compatible REST server | `model`/`--device`/`--mode`(local/interactive/server)/`--speculative-mode`/`--prefix-cache-mode`/`--host`/`--port` |
| `package` | Package iOS/Android model lib+weights | `--package-config`/`--output` |
| `calibrate` | Quantization calibration (ShareGPT) | `model`/`--model-lib`/`--dataset`/`--num-calibration-samples` |
| `router` | disaggregation routing | `model`/`--router-mode`(disagg/round-robin)/`--pd-balance-factor` |

Aux: `python -m mlc_llm.bench` (benchmark)/`cli/delivery` (HF continuous delivery)/`cli/worker` (Disco distributed).

Help text centralized `interface/help.py` (HELP dict, 273 lines), each param help concatenates `HELP[key]`. Actual implementation delegated to `mlc_llm.interface.<cmd>`, CLI layer only parses+type auto-detect (`auto_config`/`auto_target`/`auto_device`).

### Doc Structure

**Three layers**:

| Layer | Path | Tech | Deploy |
|----|------|------|------|
| README | `README.md` | Markdown | GitHub |
| Sphinx docs | `docs/` | Sphinx + RST + `sphinx_rtd_theme` | `llm.mlc.ai/docs` |
| Jekyll site | `site/` | Jekyll + `jekyll-theme-mlc` | `llm.mlc.ai` (home) |

`docs/` toctree (`index.rst`):
- `get_started/` (quick_start/introduction)
- `install/` (mlc_llm/tvm/conda/gpu/emcc)
- `compilation/` (compile_models 1092 lines/convert_weights/package_libraries/configure_quantization/define_new_models)
- `deploy/` (cli/rest/python_engine/webllm/ios/android/ide_integration/mlc_chat_config)
- `microserving/tutorial.rst` (cross-engine disaggregation)
- `community/` (guideline/faq)

### Command Reference

- `--help`: each subcommand independent ArgumentParser, `mlc_llm <cmd> --help` outputs usage; top `mlc_llm --help` lists 8 subcommands
- docs site RST param tables (`docs/deploy/cli.rst` chat / `docs/deploy/rest.rst` serve / `docs/compilation/compile_models.rst` compile/convert_weight/gen_config inline): **hand-written RST tables**, positional args uppercase placeholders (`MODEL`/`CONFIG`)+ `--flag`

### Main Capabilities

| Capability | Implementation |
|------|------|
| compile (TVM compile) | `cli/compile.py` + `interface/compile.py`; platforms cuda/metal/vulkan/rocm/opencl/webgpu/iphone/android/auto; cross-compile(`--host` LLVM triple); `--opt` O0-O3 + fine-grained knob; `--system-lib-prefix` |
| convert_weight | `cli/convert_weight.py`; source-format huggingface-torch/safetensor/awq; quant AWQ/GroupQuantize/FTQuantize/FP8/BlockScale/PerTensor/NoQuantize; `--lora-adapter`(PEFT merge) |
| gen_config | `cli/gen_config.py`; conv-template/context-window/sliding-window/prefill-chunk/tensor-parallel/pipeline-parallel/disaggregation |
| serve (OpenAI-compatible) | `cli/serve.py` (264 lines); mode local/interactive/server; speculative(small_draft/eagle/medusa); prefix cache(radix tree); hybrid/chunked prefill;tensor parallel;embedding model;API key;CORS |
| chat | `cli/chat.py`; REPL `/help`/`/exit`/`/stats`/`/metrics`/`/set temperature=0.5` |
| package | `cli/package.py`; iOS/Android model lib+weights bundle |
| calibrate | `cli/calibrate.py`; ShareGPT calibration |
| router | `cli/router.py`; prefill-decode disaggregation |
| bench | `python -m mlc_llm.bench`; fixed concurrency/request rate; TTFT; CUDA profile; CSV |
| JIT compile | `interface/jit.py`; `--model-lib` unspecified auto-JIT (`MLC_JIT_POLICY` ON/OFF/REDO/READONLY) |
| Multi-platform SDK | `ios/` (MLCChat+MLCSwift)/`android/` (MLCChat+mlc4j)/`web/` (WebLLM npm) |

### Key Files

- `pyproject.toml` (no [project.scripts], scikit-build-core)
- `python/mlc_llm/__main__.py` (CLI main entry, argv[1] dispatch)
- `python/mlc_llm/cli/` (8 subcommands + delivery/lib_delivery/model_metadata/worker)
- `python/mlc_llm/interface/` (actual implementation + help.py HELP dict + jit.py)
- `python/mlc_llm/support/argparse.py` (enhanced ArgumentParser)
- `CMakeLists.txt` (only builds libs, no CLI binary)
- `cpp/` (C++ engine: serve/metadata/json_ffi/tokenizers/multi_gpu)
- `docs/` (Sphinx RST)+ `site/` (Jekyll)
- `python/mlc_llm/model/` (45 model architectures)+ `quantization/` (7 quant algorithms)
- `ios/`/`android/`/`web/` (platform SDK)

---

<a id="5-litert-lm-google-bazel"></a>
## 5. litert-lm (Google, Bazel)

### CLI Entry

**Bazel Python binary (click)**, not C++:
- Bazel target: `//python/litert_lm_cli:litert-lm` (`pytype_strict_binary`, `main = "main.py"`)
- PyPI wheel `litert-lm`, `console_scripts` `litert-lm = litert_lm_cli.main:main` (`BUILD:301-305`)
- `python/litert_lm_cli/main.py` → `main()` → `cli()` click group

Command registration: **module-level `register(cli)` + `cli.add_command()`**:
```python
# main.py
_serve_module.register(cli); _convert_module.register(cli); _list_module.register(cli)
_import_module.register(cli); _describe_module.register(cli); _delete_module.register(cli)
_rename_module.register(cli); _benchmark_module.register(cli); _run_module.register(cli)
_pack_module.register(cli); _unpack_module.register(cli)
```
Shared flags use `common.py` decorator factory (`common_inference_options`/`config_option`/`huggingface_options`).

### Command Style

**Single binary + subcommands (click Group)**, 11 in 3 groups (`help_formatter.py` COMMAND_SECTIONS):

| Group | Commands | Files |
|------|------|------|
| Inference | `run`/`benchmark`/`serve` | `commands/run.py`/`benchmark.py`/`serve.py` |
| Model Management | `list`/`import`/`describe`/`rename`/`delete` | `commands/list.py`/`import.py`/`describe.py`/`rename.py`/`delete.py` |
| File Commands | `pack`/`unpack`/`convert`(stub) | `commands/pack.py`/`unpack.py`/`convert.py` |

Per command:
- `run`: inference (`--prompt`/`--backend` cpu|gpu|npu/`--speculative-decoding`/`--attachment` multimodal/`--thinking`/`--top-k/p/temperature`/`--cache`/`--android` ADB)
- `benchmark`: prefill/decode timing (`--runs`, Android ADB)
- `serve`: HTTP (OpenAI/Gemini-compatible, `openai_handler.py`/`gemini_handler.py`)
- `list`/`import`/`describe`/`rename`/`delete`: model management (`~/.litert-lm`)
- `pack`/`unpack`: `.litertlm` container pack/unpack (`litert_lm_builder`)
- `convert`: **stub unimplemented** (28 lines, not add_command)

### Doc Structure

- `README.md` (product overview, Release v0.7~v0.16)
- `docs/` (`README.md` index):
  - `getting-started/` (`build-and-run.md` Bazel 7.6.1 Win/Linux/Mac/Android;`cmake.md` experimental)
  - `api/cpp/` (conversation/constrained-decoding/tool-use/tool-use-antlr)
  - `api/kotlin/` (getting_started.md)
  - `litert_lm_builder.md` (builder library CLI+Python API)
- Per Python package README (links external `ai.google.dev/edge/litert-lm/*`)
- **Full CLI command reference external link** `ai.google.dev/edge/litert-lm/cli`, repo only has build/API guide

### Command Reference

- `--help`/`-h` (click `ColorHelpFormatter` green titles/cyan usage, `show_default=True`, `max_content_width=120`)
- flags: `@click.option` decorator, `textwrap.dedent` multi-line help; `click.Choice` (`--backend` cpu|gpu|npu); `is_flag`/`flag_value`; `click.IntRange/FloatRange`; `multiple=True` (`--attachment`/`--cors-origin`)
- `config.json` (JSON Schema `config_schema.json`, v0.15 centralized CLI configuration): flags > config.json > engine default; `--config` specifies path (default `~/.litert-lm/config.json`)
- No independent param table md, `--help` + docstring Args section (`run.py:608-650`) authoritative

### Main Capabilities

| Capability | Implementation |
|------|------|
| Export (litert-torch) | **Not in litert-lm CLI** (no litert-torch/torch.export references); positioning **consumer-side**, consumes upstream pre-exported .litertlm/.tflite; `convert` is stub |
| Pack .litertlm | `pack`/`unpack` + `litert_lm_builder` (LitertLmFileBuilder: tflite+tokenizer+metadata)+ `litert-lm-peek` |
| Quantize | Consumer-side (pre-quantized artifacts); `--activation-data-type` fp32|fp16|int16|int8 (experimental, hidden) |
| Inference | `run` + `litert_lm.Engine`/`Session`/`Conversation`; prefill-decode (`session.run_prefill`→`run_decode_async` streaming) |
| Backend | `--backend` cpu|gpu|npu; `--cpu-thread-count`/`--gpu-decode-steps-per-sync`/`--ringbuffers-local-attention`(KV memory saving)/`--cache` disk|memory|no |
| speculative | `--speculative-decoding` MTP (Multi-Token Prediction) drafters (Gemma4) |
| Multi-platform | `prebuilt/` (android_arm64/x86_64/ios_arm64/sim/macos/linux_arm64/windows); Android ADB (`adb_engine`/`adb_benchmark`); v0.16 C API prebuilt all platforms |
| Multimodal/Tool use | `run --attachment`/`--preset`; `ToolEventHandler` function calling; constrained decoding (C++ ANTLR) |
| serve | OpenAI/Gemini-compatible (`/v1/chat/completions`/`/v1/embeddings`) |

### Key Files

- `python/litert_lm_cli/main.py` (CLI main entry, click group, command registration)
- `python/litert_lm_cli/BUILD` (litert-lm binary/wheel target)
- `python/litert_lm_cli/{common,help_formatter,config,model}.py` (shared flag/color help/config parsing)
- `python/litert_lm_cli/commands/{run,benchmark,serve,list,import,describe,rename,delete,pack,unpack,convert}.py`
- `python/litert_lm/` (Engine/Session/Conversation/EmbeddingEngine/Benchmark)
- `python/litert_lm_builder/` (LitertLmFileBuilder/litertlm_peek)
- `docs/` + per-package README
- `prebuilt/` (all-platform C API)
- `WORKSPACE`/`.bazelversion`/`.bazelrc` (Bazel)

---

<a id="6-coremltools-apple"></a>
## 6. coremltools (Apple)

### CLI Entry

**No CLI** (pure Python API):
- No `pyproject.toml` (repo root)
- `setup.py` (115 lines) only `packages=find_packages()`+`install_requires`+`package_data`, **no `entry_points`**
- No `__main__.py` (`coremltools/` glob confirms)
- No argparse/click/typer (grep zero hits)
- `MANIFEST.in` (1 line `include README.md`)
- `CMakeLists.txt` (285 lines) builds C++ extensions (`libmilstoragepython.so`/`libmodelpackage.so`/`libcoremlpython.so` macOS only + `kmeans1d`), **no CLI target**

### Command Style

**Pure Python API**: `import coremltools as ct; ct.convert(...)`:
- `coremltools/__init__.py:129` → `from .converters import ... convert`
- `coremltools/converters/_converters_entry.py:69` `convert(model, source="auto", inputs=None, outputs=None, classifier_config=None, minimum_deployment_target=None, convert_to=None, compute_precision=None, skip_model_load=False, compute_units=_ComputeUnit.ALL, package_dir=None, debug=False, pass_pipeline=None, states=None)` (14 kwargs)
- Top-level API: `ct.ComputeUnit`/`ct.target`(macOS13/iOS15)/`ct.precision`(FLOAT16/FLOAT32)/`ct.transform`(FP16ComputePrecision)/`ct.PassPipeline`/`ct.TensorType`/`ct.ImageType`/`ct.Shape`/`ct.RangeDim`/`ct.EnumeratedShapes`/`ct.ClassifierConfig`
- source: auto/tensorflow/pytorch/milinternal; convert_to: mlprogram/neuralnetwork/milinternal

### Doc Structure

**Three Sphinx sites**:

| Site | Source | Theme | Deploy |
|------|------|------|------|
| API Reference | `docs/` | `sphinx_rtd_theme` + autodoc/numpydoc/sphinx_gallery | `gh-pages/` root |
| Guide & Examples | `docs-guides/` | `sphinx_book_theme` + myst_parser(Markdown) + sphinx_copybutton | `gh-pages/docs-guides/` |
| Core ML Format Spec | `mlmodel/docs/` | protobuf spec | `gh-pages/mlmodel/` |

Key docs:
- `README.md` (install/capabilities/resources)
- `BUILDING.md` (CMake/conda/zsh, `scripts/build.sh`)
- `docs-guides/source/` (~90 articles: convert-pytorch/convert-tensorflow/opt-quantization/opt-palettization/opt-pruning/opt-joint-compression/flexible-inputs/stateful-models etc.)
- `docs/source/` (autodoc stub: `coremltools.converters.convert.rst` etc.)
- `docs/examples/optimize/torch/` (sphinx_gallery runnable: pruning/palettization/quantization)
- `examples/README.md` (only links external readme.io)

### Command Reference

**No `--help`** (no CLI). Param docs two types:
1. `convert()` numpy docstring (`_converters_entry.py:85-1190`, 1100+ lines) — Sphinx autodoc + numpydoc renders API page (Parameters section per-param `name : type (optional)` + code snippets)
2. Guide Markdown prose (`docs-guides/source/convert-*.md`/`conversion-options.rst`) covers workflows

### Main Capabilities

| Capability | Implementation |
|------|------|
| Convert PyTorch→CoreML | `converters/mil/frontend/torch/` (TorchScript + torch.export EDGE dialect;`source="pytorch"`) |
| Convert TensorFlow→CoreML | `converters/mil/frontend/tensorflow/` (TF1.x frozen/`.pb`/`.h5`/SavedModel + TF2.x keras/`.h5`/concrete;`source="tensorflow"`) |
| Non-neural | `converters/{libsvm,sklearn,xgboost,lightgbm}/` |
| ML Program/NeuralNetwork | `convert_to`/`minimum_deployment_target`(iOS15+ mlprogram, old neuralnetwork; milinternal debug) |
| Quantize/Palettize/Prune (CoreML side) | `optimize/coreml/` (OpLinearQuantizerConfig/OpPalettizerConfig/OpMagnitudePrunerConfig + linear_quantize_weights/palettize_weights/prune_weights/decompress_weights/get_weights_metadata; MIL graph passes) |
| Quantize/Palettize/Prune (PyTorch training side) | `optimize/torch/` (base_model_optimizer/layerwise_compression/palettization/pruning/quantization/optimization_config, optional install) |
| compute_precision | `convert(compute_precision=...)` + `ct.transform.FP16ComputePrecision(op_selector=...)` |
| Graph pass pipeline | `converters/mil/mil/passes/pass_pipeline.py` → `ct.PassPipeline`(EMPTY/CLEANUP/DEFAULT_PRUNING/DEFAULT_PALETTIZATION,remove_passes/set_options) |
| Predict/verify (macOS only) | `models/model.py` + `coremlpython/`(Obj-C++ bridge, calls Core ML framework);`skip_model_load=True` non-macOS skips loading only export |
| iOS deployment | `modelrunner/` (Xcode project `ModelRunner.xcworkspace`); artifacts `.mlpackage`/`.mlmodel` Xcode integration |

### Key Files

- `setup.py` (no console_scripts)
- `CMakeLists.txt`/`Makefile` (C++ extension build, `scripts/*.sh`)
- `coremltools/__init__.py` (top-level API: convert/ComputeUnit/target/precision/transform/PassPipeline/type descriptors)
- `coremltools/converters/_converters_entry.py` (`convert()` 1190 lines, docstring command reference)
- `coremltools/converters/mil/` (MIL + frontend torch/tf + backend milprogram/neuralnetwork + graph passes)
- `coremltools/optimize/{coreml,torch}/` (quantize/palettize/prune)
- `coremltools/models/` (MLModel class/ml_program/neural_network/utils/compute_plan)
- `coremltools/modelrunner/` (iOS verification app)
- `coremlpython/`/`milstoragepython/`/`mlmodel/`/`modelpackage/` (C++/Obj-C++ native modules)
- `docs/` (API Sphinx)+ `docs-guides/` (Guide Sphinx)+ `mlmodel/docs/` (Format Spec)

---

<a id="comparison-table-6-tools-cann-lm"></a>
## Comparison Table (6 tools + cann-lm)

| Dimension | mlx-lm | ollama | executorch | mlc-llm | litert-lm | coremltools | **cann-lm-converter** |
|------|--------|--------|-----------|---------|-----------|-------------|----------------------|
| **entry registration** | console_scripts 18 (setup.py) | Go binary (cobra) | console_scripts only flatc | None (alias python -m) | console_scripts 1 (click) | None (pure API) | **console_scripts 1 ✅** (pyproject `[project.scripts]`) |
| **command style** | Dual-track (dot+subcommand, hand-written dispatcher) | Unified `ollama <cmd>` (cobra) | Scattered `python -m` (Hydra+argparse) | Subcommand (argparse hand-written) | Subcommand (click Group) | `ct.convert()` top-level function | **Single command argparse** (no subcommand) |
| **CLI framework** | argparse (hand-written dispatch) | cobra (Go) | argparse + Hydra | argparse (hand-written dispatch) | click | None | argparse |
| **doc structure** | README + feature md (no docs site) | README + Mintlify docs site | Sphinx docs/+examples/+.claude/+.wiki | README + Sphinx RST + Jekyll site | README + docs (external CLI link) | Three Sphinx sites (API/Guide/Spec) | **docs/ by directory** + README nav + INDEX |
| **command reference** | `--help` + hand-written md | `--help`(cobra)+ docs site | Per README + tutorial + Makefile help | `--help` + RST param tables | `--help`(color)+ config.json | docstring (autodoc) | **cli/INDEX ✅** + cli_design param tables |
| **param table style** | Hand-written (argparse help=) | cobra auto + env append | Hydra YAML + dataclass schema | Hand-written RST (positional uppercase) | click auto + JSON Schema | numpy docstring (autodoc) | Hand-written (argparse help= + markdown table) |
| **capability scope** | Convert+4 quant+inference+serve+LoRA+manage+GGUF | Run+serve+pull+create+integration+multimodal | Export+quant+inference+serve+fine-tune+DevTools+12 backends | compile+convert+serve+chat+multi-platform SDK | Consumer-side pack/unpack+inference+serve | Convert+quant+palettize+prune+iOS | **Only convert** (export epair) |
| **env vars** | — | `OLLAMA_*` (20+, serve --help append) | — | — | config.json (JSON Schema) | — | env table (cli_design, no `CANN_LM_ARG_*` style) |
| **Positioning** | Full-chain side runtime | Full-chain side runtime | Full-chain side deployment engine | Full-chain side deployment engine | Consumer-side orchestration | Conversion tool | **Conversion tool** (export epair for C++ delegate) |

---

<a id="cann-lm-converter-comparison-implications"></a>
## cann-lm-converter comparison + implications

### Current State Comparison

cann-lm-converter is a **conversion tool** (export epair for C++ delegate loading), positioning close to **coremltools** (convert/export), not ollama/mlc-llm full-chain runtime. cann-lm-engine (independent `scripts/cann_lm_engine/pyproject.toml`) handles serve/inference.

| cann-lm current state | 6 tools commonality | Evaluation |
|--------------|-----------|------|
| `cann-lm-converter` console_scripts ✅ | mlx-lm/executorch/litert-lm use console_scripts;ollama Go binary;mlc-llm alias;coremltools none | **Standard** (better than mlc-llm alias/coremltools no CLI, close to mlx-lm/litert-lm) |
| Single command argparse (no subcommand) | mlx-lm dual-track;ollama/mlc-llm/litert-lm subcommand;executorch scattered;coremltools API | **Reasonable** (converter single responsibility convert; if extend serve/verify then subcommand-ize) |
| docs/ by directory + INDEX | llama.cpp docs/ + executorch Sphinx;mlx-lm feature md;ollama/mlc-llm docs site | **Close to llama.cpp/executorch** (can retain) |
| cli/INDEX command reference ✅ | mlx-lm `--help`;ollama docs site;executorch per README;mlc-llm RST;litert-lm `--help`+config;coremltools docstring | **Already has** (close to mlx-lm/executorch) |
| Only convert | 6 tools mostly full-chain (convert+inference+serve+fine-tune) | **Reasonable** (cann-lm-engine independent serve/inference; converter keeps convert) |

### Implications

1. **console_scripts `cann-lm-converter` ✅** already registered (more standard than mlc-llm alias/coremltools no CLI, close to mlx-lm/executorch/litert-lm)
2. **Single command suits converter single responsibility** (convert export epair); if extend serve/verify (like mlx-lm/ollama/mlc-llm full-chain), can subcommand-ize:
   - mlx-lm dual-track (dot+subcommand, hand-written dispatcher) — simple, but many dot entries
   - litert-lm click Group + `register(cli)` — modular, recommended
   - Reference `cann-lm convert`/`cann-lm serve`/`cann-lm verify`
3. **docs/ by directory close to llama.cpp + executorch Sphinx** (can retain); can borrow:
   - mlx-lm feature md (SERVER.md/LEARNED_QUANTS.md style, by feature files)
   - mlc-llm/ollama docs site (independent command reference page, Mintlify/Sphinx)
4. **Command reference cli/INDEX ✅** already added; can borrow:
   - llama.cpp `llama-gen-docs` auto-generate (`tools/cli/README.md` from `--help`, avoid manual maintenance)
   - litert-lm `config.json` (JSON Schema, centralized CLI configuration)
   - ollama `OLLAMA_*` env vars (param table marks env, cann-lm can `CANN_LM_ARG_*` style)
5. **Capability only convert** (side tools mostly full-chain): cann-lm-engine independent (serve/inference, like ollama/mlc-llm run/serve); converter keeps convert reasonable; if merge CLI reference ollama/mlc-llm `convert`/`serve`/`run` subcommands

### Optional Improvements (low priority, non-blocking)

| Improvement | Borrow from | Priority |
|------|------|--------|
| Command reference auto-gen (`--help` → README) | llama.cpp `llama-gen-docs` | Medium (avoid manual param table maintenance) |
| Env var parameterization (`CANN_LM_ARG_*`) | ollama `OLLAMA_*` | Low (env table sufficient) |
| config.json CLI config | litert-lm `config.json` (JSON Schema) | Low (converter few params) |
| Subcommand-ize (if responsibility expands) | litert-lm click `register(cli)` | Low (pending serve/verify expansion) |
| Feature md (SERVER.md style) | mlx-lm SERVER.md/LEARNED_QUANTS.md | Low (docs/ by directory already covers) |

---

<a id="key-difference-export-tool-vs-side-runtime"></a>
## Key difference: export tool vs side runtime

6 tools two categories:

| Category | Tools | Positioning | cann-lm corresponding |
|------|------|------|-------------|
| **Full-chain side runtime** (convert+inference+serve) | mlx-lm/ollama/executorch/mlc-llm/litert-lm | Convert+quant+inference+serve(+fine-tune) | **cann-lm-engine** (independent, serve/inference) |
| **Conversion tool** (export/convert, for downstream runtime loading) | coremltools | PyTorch/TF→CoreML, no inference | **cann-lm-converter** (HF→epair, for C++ delegate loading) |

cann-lm-converter positioning close to **coremltools** (convert/export), not ollama/mlc-llm full-chain. So:
- cann-lm-converter **single command + console_scripts** reasonable (like coremltools's `ct.convert()`, but cann-lm uses CLI not Python API)
- cann-lm-engine (independent pyproject) handles serve/inference (like ollama/mlc-llm run/serve)
- If converter+engine merge CLI, reference litert-lm click subcommands (`cann-lm convert`/`cann-lm serve`)

---

## Related

- Legacy issue: [CLI-013](./TODO_notes.md#cli-013-todo-side-llm-tool-cli-and-doc-delivery-survey) (all action items ✅: subcommand-ization evaluation keep single command + console_scripts registered + command reference page cli/INDEX added)
- cann-lm packaging: `scripts/cann_lm_converter/pyproject.toml` (console_scripts `cann-lm-converter`) + `scripts/cann_lm_engine/pyproject.toml` (engine independent)
- Design docs: [cli_design.md](./cli/cli_design.md) "Command-line parameters / Complete call chain", [cli/INDEX.md](./cli/INDEX.md) "Command Reference"
- Code: `cann_lm/converter/cann_lm_converter.py` (CLI entry)
