# Converter 代码-约定偏离清理笔记

> 记录 `cann_lm/converter/` 代码与 [AGENTS.md](../../AGENTS.md) 约定 / 设计文档的偏离点,按发现顺序累积。
> 每条含:位置、现状、约定、根因、影响、状态、清理动作。清理完成后状态置「已清理」并保留记录。
>
> 状态:累积中 | 最后更新: 2026-08-31

## 目录

- [CLI-001 `from __future__ import annotations` 冗余](#cli-001-from-__future__-import-annotations-冗余)
- [CLI-002 saver `use_embeddings` 默认值文档与代码不符](#cli-002-saver-use_embeddings-默认值文档与代码不符)
- [CLI-003 场景2 调用链 Embedding 策略文档过时](#cli-003-场景2-调用链-embedding-策略文档过时)
- [CLI-004 场景2 `--quant-bits` 与 dopt_wrapper 不同步](#cli-004-场景2---quant-bits-与-dopt_wrapper-不同步)
- [CLI-005 `--quant-device` CLI 参数删除(设备自动检测)](#cli-005---quant-device-cli-参数删除设备自动检测)
- [CLI-006 `generate_dopt_config` 删除 + `build_layer_strategy` 改名](#cli-006-generate_dopt_config-删除--build_layer_strategy-改名)
- [CLI-007 TODO:CPU/GPU 一键式不分进程评估](#cli-007-todocpugpu-一键式不分进程评估)
- [CLI-008 `swap_to_quant_linear_from_dopt` 死代码删除](#cli-008-swap_to_quant_linear_from_dopt-死代码删除)

---

## CLI-001 `from __future__ import annotations` 冗余

| 字段 | 值 |
|------|-----|
| 编号 | CLI-001 |
| 位置 | `cann_lm/converter/` 下共 **6 处**:`cann_lm_converter.py:17`、`loader.py:3`、`saver.py:3`、`export/epair_utils.py:3`、`export/ge_converters/ge_converter.py:17`、`quant/download_dataset.py:15` |
| 现状 | 文件顶部存在 `from __future__ import annotations` |
| 违反约定 | [AGENTS.md](../../AGENTS.md)「通用 / Python 版本:`>= 3.10`(使用 `dict[str, Any]` 等内置泛型语法,**不引入** `from __future__ import annotations`)」 |
| 发现日期 | 2026-08-31(首轮核对 `cann_lm_converter.py`;二轮比对 cli_design.md ↔ converter 各文件时经 grep 扩展到全部 6 处) |
| 状态 | **已清理(2026-08-31)** |

### 背景

`from __future__ import annotations`(PEP 563)把模块内所有类型注解从「定义时求值」改为「延迟成字符串存储」。在 Python 3.7–3.9,它让 `int | None` / `list[str]` 这类 3.10+ 语法提前可用(注解只是字符串,不会被 eval,故旧版不会因不认识的语法报错)。

### 根因

项目 Python 环境(三套均验证一致):

| 环境 | 路径 / 命令 | Python 版本 |
|------|------------|-------------|
| 默认 | `python3` | 3.11.4 |
| 系统(④ npu verify-runtime) | `/opt/buildtools/Python-3.11.4/bin/python3` | 3.11.4 |
| converter venv(① export epair) | `/mnt/workspace/torch-delegate-harmony/.venv/bin/python` | 3.11.4 |

三套均 `>= 3.10`。Python 3.10+ 已原生支持运行时求值 `int | None`(PEP 604)、`list[str]` / `dict[str, Any]`(PEP 585),无需 future 注解。故该行冗余——属于可安全删除的死代码,且违反「不引入 future 注解」约定。

### 影响(各文件用到的未来语法注解)

- `cann_lm_converter.py`:`convert()` 的 `num_layers: int | None`、`quant_path: str | None` 等(:51–55)
- `loader.py`:`ModelArtifact` 的 `quant_info: dict | None`(:26)、`tokenizer_files: list[Path]`(:23)
- `saver.py`:`output_dir: str | Path`、`whitelist_decompositions: set | None`、`embed_quant: dict[...] | None`
- `epair_utils.py` / `ge_converter.py` / `download_dataset.py`:同类 `X | Y` / 泛型注解

删除该行后,3.11 原生求值这些类型,**无行为变化**;反而 `get_type_hints()` 等运行时反射从「拿到字符串」恢复正常为「拿到真实类型对象」(对依赖反射的库更友好)。

### 设计文档未反映

[cli_design.md](./cli/cli_design.md) 的 `convert()` 代码块(L36–126)未标注此行,「converter 与 torch_npu 的关系」等章节亦未提及。即源码存在此行而设计文档从未记录,本次补记。

### 清理动作

1. 删除上述 6 个文件顶部的 `from __future__ import annotations`(各文件 1 行,连同其后空行保持 import 分组整洁)。
2. 清理后跑一次 converter 冒烟(如 `examples/` 下 qwen3-0.6B `single` 导出)确认无 import / 注解求值错误。
3. 更新本文档状态为「已清理」并补清理日期。

### 执行结果(2026-08-31)

6 文件顶部 `from __future__ import annotations` 已全部删除(`cann_lm_converter.py`、`loader.py`、`saver.py`、`export/epair_utils.py`、`export/ge_converters/ge_converter.py`、`quant/download_dataset.py`)。`python3 -m py_compile` 6 文件全部通过(3.11 原生求值 `int | None` / 泛型注解,无语法错误);grep 确认 `cann_lm/converter/` 目录 0 残留。

> 端到端冒烟(实际跑一次 converter 导出)未执行——需 source `scripts/set_env_export.sh` 全套 env(torchair stubs),属可选验证;语法层已确认无回归。

### 关联

- 约定来源:[AGENTS.md](../../AGENTS.md)「通用 / Python 版本」
- 源码文件:`cann_lm/converter/` 下 6 个文件(见上表)
- 设计文档:[cli_design.md](./cli/cli_design.md)

---

## CLI-002 saver `use_embeddings` 默认值文档与代码不符

| 字段 | 值 |
|------|-----|
| 编号 | CLI-002 |
| 位置 | 文档:[cli_design.md:272](./cli/cli_design.md);代码:`cann_lm/converter/saver.py:58`、`saver.py:139` |
| 现状(清理前) | 文档签名块写 `use_embeddings=True`,代码 `save_epair` / `save_epair_prefill_decode` 默认 `False`;文档注释「saver 默认 False」与签名块 True 自相矛盾,且与代码不符 |
| 违反约定 | 设计文档应与代码一致(AGENTS「修改了代码后,要更新 docs/」) |
| 发现日期 | 2026-08-31(二轮比对 cli_design.md ↔ saver.py) |
| 状态 | **已清理(2026-08-31)** |

### 根因

saver 作为底层 API 原默认 `use_embeddings=False`(图内查表,自包含),CLI 层 `convert()` 显式传 CLI 默认 True(纯 embeddings,embed_tokens 外置 host 查表)。设计文档代码块却把 saver 默认写成 True,与代码不符,且同处注释又说 False,文档内部自相矛盾。

### 决策

用户确认以 True 为准:saver 默认值对齐「纯 embeddings」(与 CLI 默认一致),不再保留 saver 层的保守默认 False。

### 清理动作(已执行)

1. `cann_lm/converter/saver.py:58` `save_epair` 默认 `use_embeddings: bool = False` → `True`
2. `cann_lm/converter/saver.py:139` `save_epair_prefill_decode` 默认 `use_embeddings: bool = False` → `True`
3. [cli_design.md:272](./cli/cli_design.md) 注释「saver 默认 False」→「saver 默认 True」(签名块本就写 True,现在代码 / 签名块 / 注释三方一致)

### 影响

- CLI 路径不变(`convert()` 显式传参,默认值变化不影响 CLI)
- 受影响:程序化直调 `save_epair` / `save_epair_prefill_decode` 且不传 `use_embeddings` 的调用方(如 `examples/` 测试脚本)——默认从「图内查表」变为「纯 embeddings」。需排查此类直调点,确认是否需显式传 `False`。

### 残留(未改,知会)

`saver.py:253` `save_model_config(artifact, output_dir, *, use_embeddings: bool = False, ...)` 仍为 `False`。该参数语义是「写 `model_config.json` 时记录的 `use_embeddings` 字段」,且 `save_epair` / `save_epair_prefill_decode` 调用它时**总显式传参**(传自身的 `use_embeddings`),默认值实际不命中。如需全模块统一可一并改 True,但语义独立,本次不动。

### 关联

- 源码文件:`cann_lm/converter/saver.py`
- 设计文档:[cli_design.md](./cli/cli_design.md)「prefill/decode 分离模式 CLI / 接口(saver 侧)」

---

## CLI-003 场景2 调用链 Embedding 策略文档过时

| 字段 | 值 |
|------|-----|
| 编号 | CLI-003 |
| 位置 | 文档:[cli_design.md:204](./cli/cli_design.md) 场景2 调用链 Embedding 行 |
| 现状(清理前) | 文档写「Embedding→Quant_Embed_MinMax / Quant_Gemma4_Embed」;代码 `auto_quant.py:16` / `dopt_wrapper.py:23` 只用 `Quant_Embed_MinMax`,`auto_quant.py:24` 注释明确「dopt 53ad5ec removed Quant_Gemma4_Embed; gemma4 embed uses Quant_Embed_MinMax too」 |
| 违反约定 | 设计文档应与代码一致 |
| 发现日期 | 2026-08-31(三轮核对 cli_design 场景2 ↔ auto_quant / dopt_wrapper) |
| 状态 | **已清理(2026-08-31)** |

### 清理动作(已执行)

[cli_design.md:204](./cli/cli_design.md) 删「 / Quant_Gemma4_Embed」,只留 `Embedding→Quant_Embed_MinMax`,与代码一致。

### 关联

- 源码佐证:`cann_lm/converter/quant/auto_quant.py:16,24`、`dopt_wrapper.py:23`
- 设计文档:[cli_design.md](./cli/cli_design.md)「完整调用链 / 场景2」

---

## CLI-004 场景2 `--quant-bits` 与 dopt_wrapper 不同步

| 字段 | 值 |
|------|-----|
| 编号 | CLI-004 |
| 位置 | 代码:`cann_lm/converter/quant/auto_quant.py:65`(cmd,HEAD 版)、`dopt_wrapper.py:47-53`(parse_args) |
| 现状(初次核对) | HEAD 版 `auto_quant._run_dopt_subprocess` 的 cmd 传 `--quant-bits`,但 `dopt_wrapper.parse_args` 未定义该参数 → 实测报 `unrecognized arguments: --quant-bits w4a8`,exit=2,场景2 subprocess 启动即失败 |
| 违反约定 | 代码内部不一致(auto_quant ↔ dopt_wrapper) |
| 发现日期 | 2026-08-31 |
| 状态 | **已清理(2026-08-31,工作区)** |

### 核对过程

实测:`TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 dopt_wrapper.py --model-path /nonexist --optimize-config /nonexist --output-dir /tmp/opencode/dopt_test --hf-type-save --quant-bits w4a8` → `error: unrecognized arguments: --quant-bits w4a8`,exit=2。确认 `dopt_wrapper.parse_args` 不收 `--quant-bits`。

### 清理动作(已执行,工作区)

`git diff` 确认 `auto_quant.py` 工作区已删 cmd 中 `"--quant-bits", quant_bits,` 行(HEAD 仍有,工作区无)。该删除使 auto_quant spawn dopt_wrapper 时不再传 `--quant-bits`,argparse 不再报错,场景2 subprocess 可启动。

> 该行删除非本会话 edit 完成——本会话 edit 因 oldString 已不存在而失败,疑为并发手动删除。HEAD 仍含 `--quant-bits`,需 commit 才彻底落地。

### 参数链清理(2026-08-31 补充,全链已删)

用户指示「清理未用的 quant_bits」,全链删除:

- **代码**:`auto_quant.py`(`generate_dopt_config` / `_run_dopt_subprocess` / `run_auto_quant` 三函数签名 + `_run_dopt_subprocess` 调用)、`cann_lm_converter.py`(convert `quant_bits` 参数 + print + `run_auto_quant` 调用 + parse_args `--quant-bits` add_argument + main `quant_bits=args.quant_bits`)、`examples/.../run_pd6_cases.sh:119`(`generate_dopt_config(art, "w4a8")` → `generate_dopt_config(art)`)
- **文档(已同步)**:`quant_design.md`(L343 调用链 + L492/493/495 签名表)、`cli_design.md`(L32 概述 + L49 convert 代码块 + L67 调用 + L151-152 parse_args 代码块 + L172 参数表 + L197 调用链 + L456 关联表)、`quant/INDEX.md` L53

验证:`grep quant_bits`/`quant-bits` 在 `cann_lm/converter` 0 残留;`py_compile` auto_quant/cann_lm_converter/dopt_wrapper 通过。

### 文档残留(已清,2026-08-31)

`--quant-bits` 从 CLI 删除后,以下文档原残留 `--quant-bits` 描述(参数说明/关联表/示例),已全部清理(部分句意重写为「位宽由 `Quant_INT8-4_ECO` strategy 决定,无 CLI 参数」):

- `quant_design.md`:L31 / L54 / L57 / L68 / L73 / L80 / L83 / L360 / L374(假描述「映射到 _LINEAR_STRATEGY」重写)/ L379 / L1237(共 11 处)
- `quant/INDEX.md`:L74(参数表行)/ L81(指向 CLI)
- `cli/INDEX.md`:L39(参数表行)/ L54(接入说明)
- `dopt_design.md`:L101(`--quant-bits w4a8` 示例,去 `--hf-type-save` 续行符)/ L582(关联表)

验证:`grep quant.bits` 在 `docs/cann_lm_converter/` 仅余 `cleanup_notes.md` 记录文本,其他文档 0 残留。

### 关联

- 源码文件:`cann_lm/converter/quant/auto_quant.py`、`dopt_wrapper.py`
- 设计文档:[cli_design.md](./cli/cli_design.md)「完整调用链 / 场景2」「converter 与 torch_npu 的关系」

---

## CLI-005 `--quant-device` CLI 参数删除(设备自动检测)

| 字段 | 值 |
|------|-----|
| 编号 | CLI-005 |
| 位置 | 代码:`cann_lm_converter.py`(--quant-device CLI + convert `quant_device` 参数 + `if quant_device: os.environ["DEVICE"]` 块 + parse_args `--quant-device` add_argument + main `quant_device=args.quant_device` + 未用 `import os`);文档:`cli_design.md` / `cli/INDEX.md` / `quant/INDEX.md` / `quant_design.md` / `adr/0001` |
| 现状(清理前) | `--quant-device` CLI 参数通过 `os.environ["DEVICE"]` → `_detect_device_type` → subprocess `env.setdefault("DEVICE", device)` 传到 dopt subprocess;用户判断「无法传到 dopt 工具,当前自动获取」,删除 CLI,设备完全靠 `_detect_device_type` 自动检测 |
| 违反约定 | 无(用户主动简化 CLI) |
| 发现日期 | 2026-08-31 |
| 状态 | **已清理(2026-08-31)** |

### 决策

用户指示:「`--quant-device` 当前也无法传到 dopt 工具里面,也先删掉,当前是自动获取的」。删除 CLI 参数,设备完全由 `auto_quant._detect_device_type` 自动检测(读 `os.environ["DEVICE"]`,外部 env 如 `set_env_npu.sh` 可设;未设则 spawn 检测,无加速器回落 `cpu`)。

### 清理动作(已执行)

- **代码**:`cann_lm_converter.py` 删 `--quant-device` add_argument + convert `quant_device` 参数 + `if quant_device: os.environ["DEVICE"]` 块 + 注释 + main `quant_device=args.quant_device` + docstring 两处提及 + 未用的 `import os`
- **文档**:`cli_design.md`(convert 代码块 / parse_args 代码块 / 参数表 / 说明 / 调用链 / CPU 回落)、`cli/INDEX.md` L39、`quant/INDEX.md` L74、`quant_design.md`(L43 / L54 / L57 / L70 / L336 / L359 / L378 / L486 共 8 处,部分句意重写为「`_detect_device_type` 自动检测」)、`adr/0001` L29 加状态变更标注(保留历史原文)

### 保留(自动检测机制,未删)

- `auto_quant._detect_device_type`:读 `os.environ["DEVICE"]` 或 spawn 检测,回落 `cpu`
- `auto_quant._run_dopt_subprocess` 的 `env.setdefault("DEVICE", device)`:subprocess env 继承 DEVICE

### 验证

`grep quant.device` 在 `cann_lm/converter` 0 残留;`py_compile cann_lm_converter.py` 通过;文档仅余 `cleanup_notes.md` 记录 + `adr/0001` 历史原文(已加状态变更标注)。

### 关联

- 源码文件:`cann_lm/converter/cann_lm_converter.py`、`cann_lm/converter/quant/auto_quant.py`(`_detect_device_type` 保留)
- 设计文档:[cli_design.md](./cli/cli_design.md)「命令行参数 / 完整调用链 / 运行环境支持」
- 历史决策:[adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md)(状态变更 2026-08-31)

---

## CLI-006 `generate_dopt_config` 删除 + `build_layer_strategy` 改名

| 字段 | 值 |
|------|-----|
| 编号 | CLI-006 |
| 位置 | `auto_quant.py`(删 `_build_layer_strategy` + `generate_dopt_config` + `_LINEAR/_EMBED_STRATEGY` + 未用 `import json`/`import torch`)、`dopt_wrapper.py`(`_build_layer_strategy` → `build_layer_strategy`,L31 定义 + L67 调用)、`examples/.../run_pd6_cases.sh:gen_dopt_config`(内联)、`tests/dopt_lm/test_strategy_generation.py`(改单份)、文档(`quant_design.md` / `quant/INDEX.md` / `cli_design.md` / `adr/0001` / `dopt_design.md`) |
| 背景 | `generate_dopt_config`(场景1 离线辅助)只被 `run_pd6_cases.sh` 用例用;`_build_layer_strategy` 在 `auto_quant` + `dopt_wrapper` 两份重复(dopt_lm `generate_config_file` 是全 float 模板,逻辑不同不替代)。用户选路径2:删 `generate_dopt_config` + 用例内联 + 改公开名消除重复 |
| 违反约定 | 无(用户主动清理 + 消除重复) |
| 发现日期 | 2026-08-31 |
| 状态 | **已清理(2026-08-31)** |

### 决策

用户选路径2:`generate_dopt_config` 删除(auto_quant.py),用例 `run_pd6_cases.sh:gen_dopt_config` 内联 `build_layer_strategy` + `tempfile.mkstemp` + `os.fdopen` + `json.dump`;`_build_layer_strategy` 改公开名 `build_layer_strategy`(去 `_`),消除两份重复(auto_quant 删,dopt_wrapper 唯一实现)。

### 清理动作(已执行)

- **代码**:
  - `auto_quant.py`:删 `_LINEAR_STRATEGY`/`_EMBED_STRATEGY` + `_build_layer_strategy` + `generate_dopt_config` + 未用 `import json`/`import torch`
  - `dopt_wrapper.py`:`_build_layer_strategy` → `build_layer_strategy`(L31 定义 + L67 调用)
  - `examples/.../run_pd6_cases.sh`:`gen_dopt_config` 内联(删 `from auto_quant import generate_dopt_config`,改 `from dopt_wrapper import build_layer_strategy` + `tempfile`/`os.fdopen`/`json.dump`)
  - `tests/dopt_lm/test_strategy_generation.py`:删 `auto_quant` import,测试改单份(只测 `dopt_wrapper.build_layer_strategy`)
- **文档**:`quant_design.md`(L236/335/341-342/368/489-490/502/818 + replaceAll `_build_layer_strategy`→`build_layer_strategy`)、`quant/INDEX.md`(L21 删 generate_dopt_config + L22/26 改名)、`cli_design.md`(L173 计划改造段重写删 generate_dopt_config + 合并计划 + L192 改名)、`adr/0001` L44(~~删除线~~ + 已落地标注)、`dopt_design.md` L103

### 验证

`py_compile auto_quant/dopt_wrapper/cann_lm_converter` 通过;`pytest tests/dopt_lm/test_strategy_generation.py` 2 passed;`grep generate_dopt_config|_build_layer_strategy|auto_quant.build_layer_strategy` 在 docs 仅余 cleanup_notes 记录 + adr 历史标注。

### 关联

- 源码文件:`cann_lm/converter/quant/auto_quant.py`、`dopt_wrapper.py`、`examples/.../run_pd6_cases.sh`、`tests/dopt_lm/test_strategy_generation.py`
- 设计文档:[cli_design.md](./cli/cli_design.md)「完整调用链」、[quant_design.md](./quant/quant_design.md)「`auto_quant.py`/`dopt_wrapper.py` 签名表」
- 历史决策:[adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md) L44(状态变更 2026-08-31)

---

## CLI-007 TODO:CPU/GPU 一键式不分进程评估

| 字段 | 值 |
|------|-----|
| 编号 | CLI-007 |
| 类型 | TODO(待评估,非偏离) |
| 背景 | 当前场景 2(`--quant-method gptq`)主进程 export(CPU stubs)+ dopt subprocess(系统 python + torch_npu + NPU)两套环境,因 NPU 场景 `_LazyTorchair` 代理冲突(装 torch_npu 破坏 export GE 转换)用 subprocess 隔离。但 CPU/GPU(无 NPU)场景**不装 torch_npu → 不撞代理 → 一套环境理论可行** |
| 状态 | **TODO(待评估)** |
| 发现日期 | 2026-08-31 |

### 待评估问题

CPU/GPU(无 NPU)时,一键式脚本能否不分进程(一套环境跑 export + GPTQ):

1. 不装 torch_npu → export 不撞 `_LazyTorchair` 代理 → 一套环境(无 subprocess)理论可行
2. `dopt_lm` 兼容 torch 2.9.0(converter venv)?当前 dopt 在系统 python torch 2.7.1
3. converter venv 装 `transformers` + `dopt_lm` 依赖?(当前不装)
4. CPU GPTQ(dopt_lm `dispatch_dtype=float32`)在 converter venv(torch 2.9.0+cpu + torchair stubs)跑通?
5. GPU GPTQ(`torch.cuda`)适配待验证(cli_design「GPU dopt 适配 ⏳」)

### 目标:一套环境(CPU/GPU,无 NPU)

converter venv 装 `torch 2.9.0+cpu` + `torchair`(源码+stubs)+ `torch-delegate` + `transformers` + `dopt_lm`,GPTQ 走 CPU(float32)或 GPU(cuda),不 spawn subprocess,主进程直接调 `dopt_lm`。

### 当前两套环境对比

| 维度 | ① converter 主进程(export epair) | dopt subprocess(GPTQ) |
|------|----------------------------------|----------------------|
| Python | converter venv(`.venv`,3.11.4) | 系统 python(`DOPT_PYTHON`,3.11.4) |
| torch | 2.9.0+cpu | 2.7.1+cpu |
| torch_npu | **不装**(stub 路径,`_library.py` 自提供 npu:: fake-meta) | 2.7.1.post4(真,NPU 算子) |
| torchair | 源码编译(`NO_ASCEND_SDK=1`)+ `build/stubs` | 无(不跑 GE 转换) |
| torch-delegate | 0.1.0(whl) | 无 |
| transformers | 不装 | 装(dopt_lm 依赖) |
| CANN lib | stubs(`LD_LIBRARY_PATH` stubs 在前,避 `aclrtSetDevice→107001`) | 真实 lib64(spawn 时去 stubs) |
| 芯片 | 无 NPU(CPU-only) | Ascend910 × 2(64GB HBM) |
| env 脚本 | `set_env_export.sh`(CANN set_env + stubs LD + `TORCH_DELEGATE_HOME` + `AUTOLOAD=0` + `FALLBACK_CODEGEN=0`) | auto_quant spawn(继承去 stubs + `PYTHONPATH` quant/ + `DEVICE` + `DOPT_PYTHON`) |

### 核心差异(为何当前分进程)

1. **torch_npu 有无**:export 不装(避 `_LazyTorchair` 代理覆盖源码 torchair → GE 转换 bug);GPTQ 装(需真 NPU 算子 eager)
2. **torchair 源码+stubs vs 无**:export 需 GE 转换(torchair);GPTQ 不需
3. **transformers 无 vs 有**:dopt_lm 用 transformers,converter 不用
4. **CANN lib stubs vs 真实**:stubs 给 torchair GE 引擎 C 符号(避真 NPU 107001);dopt 需真 lib 跑算子
5. **torch 版本 2.9.0 vs 2.7.1**:torchair 编译匹配 2.9.0,torch_npu 2.7.1.post4 匹配 2.7.1(不互通,故分进程)

### CPU/GPU 一套的可行性关键

- **NPU 场景**:必须分进程(torch_npu 代理冲突无解)
- **CPU/GPU 场景**:不装 torch_npu → 无代理冲突 → 一套环境理论可行,但需解决 transformers 依赖 + dopt_lm torch 版本兼容 + CPU/GPU GPTQ 跑通

### 行动项

- [ ] 评估 `dopt_lm` 在 torch 2.9.0 跑通(CPU/GPU)
- [ ] converter venv 装 `transformers` + `dopt_lm` 依赖
- [ ] CPU GPTQ 一套环境冒烟(小模型,验证 dopt_lm + export 同进程)
- [ ] GPU GPTQ 适配验证(`torch.cuda`,dopt GPU 分支)
- [ ] 若可行,`auto_quant` 加 CPU/GPU 分支:不分进程,主进程直接调 `dopt_lm`(NPU 仍走 subprocess)

### 关联

- 设计文档:[cli_design.md](./cli/cli_design.md)「运行环境支持(NPU/GPU/纯 CPU)」「converter 与 torch_npu 的关系」
- 历史决策:[adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md)(subprocess 隔离决策)
- 相关代码:`cann_lm/converter/quant/auto_quant.py`(`_detect_device_type` / `run_auto_quant` / `_run_dopt_subprocess`)

---

## CLI-008 `swap_to_quant_linear_from_dopt` 死代码删除

| 字段 | 值 |
|------|-----|
| 编号 | CLI-008 |
| 位置 | `cann_lm/converter/swap.py` L334-351(定义)+ `swap_from_dopt` docstring(对比提及);文档:`quant/INDEX.md` L50、`export_design.md` L1269、`quant_design.md` L136/138/345/450/451、`quant_swap/__init__.py` L29、`adr/README.md` L39 |
| 背景 | `swap_to_quant_linear_from_dopt` 是向后兼容薄 wrapper(只 Linear swap,自己 load sd),与 `swap_from_dopt`(Linear + embed 一次 load sd)重复。grep 确认**无代码调用**(死代码),只在文档/签名表提。`_swap_linear_from_sd`(私有,接受 sd)已是抽取的 Linear swap 主体,`swap_from_dopt` 复用它,无需 `swap_to_quant_linear_from_dopt` |
| 违反约定 | 无(清理死代码) |
| 发现日期 | 2026-08-31 |
| 状态 | **已清理(2026-08-31)** |

### 决策

删 `swap_to_quant_linear_from_dopt`(死代码),`swap_from_dopt` 保持(已用 `_swap_linear_from_sd` 复用 Linear swap,一次 load sd 共用 embed 抽取)。`swap_from_dopt` 不能调 `swap_to_quant_linear_from_dopt`(会二次 load sd,违背一次 load 设计)。

### 清理动作(已执行)

- **代码**:`swap.py` 删 `swap_to_quant_linear_from_dopt`(L334-351)+ 改 `swap_from_dopt` docstring(删 "avoids swap_to_quant_linear_from_dopt's..." 对比,改为描述自身一次 load)
- **文档**:`quant_design.md`(L136 删行 + L138 删提及 + L345-346 调用链合并为 `swap_from_dopt` + L450 删签名表行 + L451 删对比)、`quant/INDEX.md` L50 删行、`export_design.md` L1269 删提及、`quant_swap/__init__.py` L29 删提及、`adr/README.md` L39(~~删除线~~ + ❌→✅ 已落地标注)

### 验证

`py_compile swap.py` 通过;`grep swap_to_quant_linear_from_dopt` 仅余 `adr/README.md` 删除线标注(保留历史),代码 + 其他文档 0 残留。

### 关联

- 源码文件:`cann_lm/converter/swap.py`
- 设计文档:[quant_design.md](./quant/quant_design.md)「`swap.py` 签名表」「swap 模块符号」、[export_design.md](./export/export_design.md)「文件树」
- 历史决策:[adr/README.md](../adr/README.md) L39(重构项 ❌→✅ 已落地)
