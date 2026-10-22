# Converter 代码-约定偏离清理笔记

> 记录 `cann_lm/converter/` 代码与 [AGENTS.md](../../AGENTS.md) 约定 / 设计文档的偏离点,按发现顺序累积。
> 每条含:位置、现状、约定、根因、影响、状态、清理动作。清理完成后状态置「已清理」并保留记录。
>
> 状态:累积中 | 最后更新: 2026-09-01

## 目录

- [CLI-001 `from __future__ import annotations` 冗余](#cli-001-from-__future__-import-annotations-冗余)
- [CLI-002 saver `use_embeddings` 默认值文档与代码不符](#cli-002-saver-use_embeddings-默认值文档与代码不符)
- [CLI-003 场景2 调用链 Embedding 策略文档过时](#cli-003-场景2-调用链-embedding-策略文档过时)
- [CLI-004 场景2 `--quant-bits` 与 dopt_wrapper 不同步](#cli-004-场景2---quant-bits-与-dopt_wrapper-不同步)
- [CLI-005 `--quant-device` CLI 参数删除(设备自动检测)](#cli-005---quant-device-cli-参数删除设备自动检测)
- [CLI-006 `generate_dopt_config` 删除 + `build_layer_strategy` 改名](#cli-006-generate_dopt_config-删除--build_layer_strategy-改名)
- [CLI-007 TODO:CPU/GPU 一键式不分进程评估](#cli-007-todocpugpu-一键式不分进程评估)
- [CLI-008 `swap_to_quant_linear_from_dopt` 死代码删除](#cli-008-swap_to_quant_linear_from_dopt-死代码删除)
- [CLI-009 `swap_from_dopt` 参数精简(删 skip/act_quant)](#cli-009-swap_from_dopt-参数精简删-skipact_quant)
- [CLI-010 TODO:load_model 加载 HF 原始 Linear fp 浪费(swap 丢弃)](#cli-010-todoload_model-加载-hf-原始-linear-fp-浪费swap-丢弃)
- [CLI-011 TODO:build_layer_strategy 与 strategy.py 一致性](#cli-011-todobuild_layer_strategy-与-strategypy-一致性)
- [CLI-012 TODO:gemma4 PLE 量化未适配(_build_external_extras 用 HF 原始 float)](#cli-012-todogemma4-ple-量化未适配_build_external_extras-用-hf-原始-float)
- [CLI-013 TODO:端侧 LLM 工具 CLI 与文档提供方式调研](#cli-013-todo端侧-llm-工具-cli-与文档提供方式调研)
- [CLI-014 TODO:PD 分离导出只存一份权重,decode 权重丢弃(PD 权重不一致时失效)](#cli-014-todopd-分离导出只存一份权重decode-权重丢弃pd-权重不一致时失效)
- [CLI-015 TODO:参考 weight_format_design.md 实现权重 ND 到 NZ 格式转换](#cli-015-todo参考-weight_format_designmd-实现权重-nd-到-nz-格式转换)
- [CLI-016 TODO:epair 权重与 const 分离](#cli-016-todoepair-权重与-const-分离)
- [CLI-017 TODO:torch-delegate 支持 torchair 打桩后整体整理安装脚本](#cli-017-todotorch-delegate-支持-torchair-打桩后整体整理安装脚本)
- [CLI-018 TODO:参考 mlx-lm 环境处理简化 cann-lm 多套环境](#cli-018-todo参考-mlx-lm-环境处理简化-cann-lm-多套环境)
- [CLI-019 TODO:导出 epair 时区分 p_/c_/b_ 三类数据](#cli-019-todo导出-epair-时区分-p_cb-三类数据)

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
| 默认 | `python3` | 3.12.9 |
| 系统(④ npu verify-runtime) | `$DOPT_PYTHON` | 3.12.9 |
| converter venv(① export epair) | `$TORCH_DELEGATE_HOME/.venv/bin/python` | 3.12.9 |

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
2. 清理后跑一次 converter 冒烟(如 `examples/` 下 qwen3.5-0.8B `single` 导出)确认无 import / 注解求值错误。
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

验证:`grep quant.bits` 在 `docs/cann_lm_converter/` 仅余 `TODO_notes.md` 记录文本,其他文档 0 残留。

### 关联

- 源码文件:`cann_lm/converter/quant/auto_quant.py`、`dopt_wrapper.py`
- 设计文档:[cli_design.md](./cli/cli_design.md)「完整调用链 / 场景2」「converter 与 torch_npu 的关系」

---

## CLI-005 `--quant-device` CLI 参数删除(设备自动检测)

| 字段 | 值 |
|------|-----|
| 编号 | CLI-005 |
| 位置 | 代码:`cann_lm_converter.py`(--quant-device CLI + convert `quant_device` 参数 + `if quant_device: os.environ["DEVICE"]` 块 + parse_args `--quant-device` add_argument + main `quant_device=args.quant_device` + 未用 `import os`);文档:`cli_design.md` / `cli/INDEX.md` / `quant/INDEX.md` / `quant_design.md` / `adr/0001` |
| 现状(清理前) | `--quant-device` CLI 参数通过 `os.environ["DEVICE"]` → `_detect_device_type` → subprocess `env.setdefault("DEVICE", device)` 传到 CANN 轻量化工具 subprocess;用户判断「无法传到轻量化工具,当前自动获取」,删除 CLI,设备完全靠 `_detect_device_type` 自动检测 |
| 违反约定 | 无(用户主动简化 CLI) |
| 发现日期 | 2026-08-31 |
| 状态 | **已清理(2026-08-31)** |

### 决策

用户指示:「`--quant-device` 当前也无法传到轻量化工具里面,也先删掉,当前是自动获取的」。删除 CLI 参数,设备完全由 `auto_quant._detect_device_type` 自动检测(读 `os.environ["DEVICE"]`,外部 env 如 `set_env_npu.sh` 可设;未设则 spawn 检测,无加速器回落 `cpu`)。

### 清理动作(已执行)

- **代码**:`cann_lm_converter.py` 删 `--quant-device` add_argument + convert `quant_device` 参数 + `if quant_device: os.environ["DEVICE"]` 块 + 注释 + main `quant_device=args.quant_device` + docstring 两处提及 + 未用的 `import os`
- **文档**:`cli_design.md`(convert 代码块 / parse_args 代码块 / 参数表 / 说明 / 调用链 / CPU 回落)、`cli/INDEX.md` L39、`quant/INDEX.md` L74、`quant_design.md`(L43 / L54 / L57 / L70 / L336 / L359 / L378 / L486 共 8 处,部分句意重写为「`_detect_device_type` 自动检测」)、`adr/0001` L29 加状态变更标注(保留历史原文)

### 保留(自动检测机制,未删)

- `auto_quant._detect_device_type`:读 `os.environ["DEVICE"]` 或 spawn 检测,回落 `cpu`
- `auto_quant._run_dopt_subprocess` 的 `env.setdefault("DEVICE", device)`:subprocess env 继承 DEVICE

### 验证

`grep quant.device` 在 `cann_lm/converter` 0 残留;`py_compile cann_lm_converter.py` 通过;文档仅余 `TODO_notes.md` 记录 + `adr/0001` 历史原文(已加状态变更标注)。

### 关联

- 源码文件:`cann_lm/converter/cann_lm_converter.py`、`cann_lm/converter/quant/auto_quant.py`(`_detect_device_type` 保留)
- 设计文档:[cli_design.md](./cli/cli_design.md)「命令行参数 / 完整调用链 / 运行环境支持」
- 历史决策:[adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md)(状态变更 2026-08-31)

---

## CLI-006 `generate_dopt_config` 删除 + `build_layer_strategy` 改名

| 字段 | 值 |
|------|-----|
| 编号 | CLI-006 |
| 位置 | `auto_quant.py`(删 `_build_layer_strategy` + `generate_dopt_config` + `_LINEAR/_EMBED_STRATEGY` + 未用 `import json`/`import torch`)、`dopt_wrapper.py`(`_build_layer_strategy` → `build_layer_strategy`,L31 定义 + L67 调用)、`examples/.../run_pd6_cases.sh:gen_dopt_config`(内联)、`tests/py/dopt_lm/test_strategy_generation.py`(改单份)、文档(`quant_design.md` / `quant/INDEX.md` / `cli_design.md` / `adr/0001`) |
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
  - `tests/py/dopt_lm/test_strategy_generation.py`:删 `auto_quant` import,测试改单份(只测 `dopt_wrapper.build_layer_strategy`)
- **文档**:`quant_design.md`(L236/335/341-342/368/489-490/502/818 + replaceAll `_build_layer_strategy`→`build_layer_strategy`)、`quant/INDEX.md`(L21 删 generate_dopt_config + L22/26 改名)、`cli_design.md`(L173 计划改造段重写删 generate_dopt_config + 合并计划 + L192 改名)、`adr/0001` L44(~~删除线~~ + 已落地标注)

### 验证

`py_compile auto_quant/dopt_wrapper/cann_lm_converter` 通过;`pytest tests/py/dopt_lm/test_strategy_generation.py` 2 passed;`grep generate_dopt_config|_build_layer_strategy|auto_quant.build_layer_strategy` 在 docs 仅余 TODO_notes 记录 + adr 历史标注。

### 关联

- 源码文件:`cann_lm/converter/quant/auto_quant.py`、`dopt_wrapper.py`、`examples/.../run_pd6_cases.sh`、`tests/py/dopt_lm/test_strategy_generation.py`
- 设计文档:[cli_design.md](./cli/cli_design.md)「完整调用链」、[quant_design.md](./quant/quant_design.md)「`auto_quant.py`/`dopt_wrapper.py` 签名表」
- 历史决策:[adr/0001](../adr/0001-load-model-cpu-quant-subprocess-device.md) L44(状态变更 2026-08-31)

---

## CLI-007 TODO:CPU/GPU 一键式不分进程评估

| 字段 | 值 |
|------|-----|
| 编号 | CLI-007 |
| 类型 | TODO(待评估,非偏离) |
| 背景 | 当前场景 2(`--quant-method gptq`)主进程 export(CPU stubs)+ 轻量化工具 subprocess(系统 python + torch_npu + NPU)两套环境,因 NPU 场景 `_LazyTorchair` 代理冲突(装 torch_npu 破坏 export GE 转换)用 subprocess 隔离。但 CPU/GPU(无 NPU)场景**不装 torch_npu → 不撞代理 → 一套环境理论可行** |
| 状态 | **TODO(待评估)** |
| 发现日期 | 2026-08-31 |

### 待评估问题

CPU/GPU(无 NPU)时,一键式脚本能否不分进程(一套环境跑 export + GPTQ):

1. 不装 torch_npu → export 不撞 `_LazyTorchair` 代理 → 一套环境(无 subprocess)理论可行
2. `dopt_lm` 兼容 torch 2.9.0(converter venv)?当前轻量化工具在系统 python torch 2.7.1
3. converter venv 装 `transformers` + `dopt_lm` 依赖?(当前不装)
4. CPU GPTQ(dopt_lm `dispatch_dtype=float32`)在 converter venv(torch 2.9.0+cpu + torchair stubs)跑通?
5. GPU GPTQ(`torch.cuda`)适配待验证(cli_design「GPU 轻量化工具适配 ⏳」)

### 目标:一套环境(CPU/GPU,无 NPU)

converter venv 装 `torch 2.9.0+cpu` + `torchair`(源码+stubs)+ `torch-delegate` + `transformers` + `dopt_lm`,GPTQ 走 CPU(float32)或 GPU(cuda),不 spawn subprocess,主进程直接调 `dopt_lm`。

### 当前两套环境对比

| 维度 | ① converter 主进程(export epair) | 轻量化工具 subprocess(GPTQ) |
|------|----------------------------------|----------------------|
| Python | converter venv(`.venv`,3.12.9) | 系统 python(`DOPT_PYTHON`,3.12.9 自动探测) |
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
4. **CANN lib stubs vs 真实**:stubs 给 torchair GE 引擎 C 符号(避真 NPU 107001);轻量化工具需真 lib 跑算子
5. **torch 版本 2.9.0 vs 2.7.1**:torchair 编译匹配 2.9.0,torch_npu 2.7.1.post4 匹配 2.7.1(不互通,故分进程)

### CPU/GPU 一套的可行性关键

- **NPU 场景**:必须分进程(torch_npu 代理冲突无解)
- **CPU/GPU 场景**:不装 torch_npu → 无代理冲突 → 一套环境理论可行,但需解决 transformers 依赖 + dopt_lm torch 版本兼容 + CPU/GPU GPTQ 跑通

### 行动项

- [ ] 评估 `dopt_lm` 在 torch 2.9.0 跑通(CPU/GPU)
- [ ] converter venv 装 `transformers` + `dopt_lm` 依赖
- [ ] CPU GPTQ 一套环境冒烟(小模型,验证 dopt_lm + export 同进程)
- [ ] GPU GPTQ 适配验证(`torch.cuda`,轻量化工具 GPU 分支)
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

---

## CLI-009 `swap_from_dopt` 参数精简(删 skip/act_quant)

| 字段 | 值 |
|------|-----|
| 编号 | CLI-009 |
| 位置 | `cann_lm/converter/swap.py` L334-355(swap_from_dopt 签名 + 内部调用);文档:`quant_design.md` L448、`quant/INDEX.md` L49、`README.md` L47 |
| 背景 | `swap_from_dopt(model, dopt_path, skip=DEFAULT_SKIP, act_quant=True)` 的 `skip`/`act_quant` 参数:无代码调用方传(唯一调用方 `cann_lm_converter.py:100` 只传 `model+dopt_path`,用默认)。`skip=DEFAULT_SKIP`(固定跳 embed_tokens)、`act_quant=True`(W4A8,当前不支持 W4A16)均为固定值,无外部覆盖需求 |
| 违反约定 | 无(参数精简) |
| 发现日期 | 2026-08-31 |
| 状态 | **已清理(2026-08-31)** |

### 决策

删 `swap_from_dopt` 的 `skip`/`act_quant` 参数,内部硬编码 `skip=DEFAULT_SKIP`/`act_quant=True` 传 `_swap_linear_from_sd`。`_swap_linear_from_sd` 保留参数(核心逻辑接口,灵活性)。

### 清理动作(已执行)

- **代码**:`swap.py` `swap_from_dopt` 签名删 `skip`/`act_quant`,内部 `_swap_linear_from_sd` 调用改 `skip=DEFAULT_SKIP, act_quant=True`
- **文档**:`quant_design.md` L448 签名表(签名 `(model, dopt_path)`,参数数 4→2,补注硬编码)、`quant/INDEX.md` L49(删 `...`)、`README.md` L47(删 `...`)

### 验证

`py_compile swap.py` 通过;`swap_from_dopt` 唯一调用方 `cann_lm_converter.py:100` 只传 `(artifact.model, dopt_out_path)`,删参数后兼容。

### 关联

- 源码文件:`cann_lm/converter/swap.py`
- 调用方:`cann_lm/converter/cann_lm_converter.py:100`
- 设计文档:[quant_design.md](./quant/quant_design.md)「`swap.py` 签名表」

---

## CLI-010 TODO:load_model 加载 HF 原始 Linear fp 浪费(swap 丢弃)

| 字段 | 值 |
|------|-----|
| 编号 | CLI-010 |
| 类型 | TODO(待优化,非偏离) |
| 背景 | `load_model(hf_path)` 加载 HF 原始(含 Linear fp16 权重)→ cann_lm 模型;`swap_from_dopt` 的 `_swap_linear_from_sd` 从轻量化工具产物 sd 读 int4+scale,`setattr` 替换 Linear→QuantLinear,**不读 model Linear 的 weight**。故 load_model 加载的 Linear HF fp 被替换丢弃,浪费 IO+内存(场景2 内存峰值:subprocess transformers + 主进程 cann_lm 含被丢弃的 Linear fp) |
| 状态 | **TODO(待评估)** |
| 发现日期 | 2026-08-31 |

### 浪费点

`_swap_linear_from_sd`(swap.py:221-331)遍历 `model.named_modules()`,对 `nn.Linear` 从 sd(轻量化工具产物)读 `{name}.weight`+`{name}.quant_op...s`,materialize int4,`setattr` 替换 QuantLinear。**完全不读 model 自身 Linear 的 `.weight`**(只定位模块名 + setattr 替换)。所以 load_model 加载的 Linear HF fp 被丢弃,不进最终产物。

| 模块 | load_model 加载(HF fp) | swap 后 | 浪费 |
|------|-------------------------|---------|------|
| `nn.Linear` | HF 原始 fp16 | 丢弃(setattr 替换 QuantLinear,int4 来自轻量化工具) | ✅ IO+内存 |
| `nn.Embedding` | HF 原始 fp16 | 抽 int8(若轻量化工具量化 embed),否则保留 fp | 部分 |
| LayerNorm/其他 | HF 原始 fp16 | 保留(轻量化工具不动) | 必要 |

### 为什么当前 load HF 原始(设计权衡)

1. cann_lm 模型结构需 HF 原始构建(`model_registry` + cann_lm 模型类 + key 重映射)
2. 非 Linear(LayerNorm)需 HF 原始 fp(轻量化工具不动,但轻量化工具产物层名可能带 `model.language_model.` 等 VLM 前缀,load_model key 重映射面向 HF 原始)
3. 简单:load_model 通用加载(不区分 Linear/非 Linear),swap 替换

### 优化方向(未做)

- **load_model 加载轻量化工具产物**(`load_model(dopt_out_path)`):Linear = fakequant fp(与 swap 同源),避免 HF 原始 Linear fp 浪费。但需 load_model key 重映射适配轻量化工具产物层名(剥 `language_model.` 等)+ 轻量化工具产物含完整模块权重
- **load_model 跳过 Linear weight(meta 占位)**:只构建结构 + 非 Linear 权重,Linear meta tensor 占位(反正是替换)。省 Linear fp IO+内存。但 load_model 通用(`param.copy_`),需加选择性跳过逻辑

### 行动项

- [ ] 评估 load_model 加载轻量化工具产物的 key 重映射兼容性(Qwen3_5 `language_model.` 前缀)
- [ ] 评估 load_model 跳过 Linear weight(meta)的可行性 + swap 是否需 Linear 模块存在(setattr 替换)
- [ ] 对比优化前后内存峰值(场景2 小模型冒烟)

### 关联

- 源码:`cann_lm/converter/cann_lm_converter.py:77`(load_model)+ `cann_lm/converter/swap.py:221`(_swap_linear_from_sd,不读 model Linear weight)
- 设计文档:[cli_design.md](./cli/cli_design.md)「完整调用链」(load_model → swap)、[quant_design.md](./quant/quant_design.md)「swap.py 签名表」
- 相关:CLI-007(场景2 内存峰值,subprocess + 主进程)

---

## CLI-011 TODO:build_layer_strategy 与 strategy.py 一致性

| 字段 | 值 |
|------|-----|
| 编号 | CLI-011 |
| 类型 | TODO(待验证/同步) |
| 背景 | `dopt_wrapper.build_layer_strategy` 生成 `dopt_config.json` 层策略(`quant_strategy` 名:`Quant_INT8-4_ECO` / `Quant_Embed_MinMax`)。轻量化工具读 strategy 名查 `dopt_lm/quantizer/strategy.py` 的 `strategy_dict` 拿完整策略(weight + input)。需确保 build_layer_strategy 的 strategy 名与 strategy.py 定义对应,效果一致 |
| 状态 | **已验证(2026-09-01,三者一致,无需改代码)** |
| 发现日期 | 2026-08-31 |

### dopt_lm/quantizer/strategy.py 策略定义(参考)

```python
"Quant_INT8-4_ECO": {
    "weight": {
        "blocksize": 128, "group_size": 128, "bit": 4,
        "per_channel": True, "weight_algo": "omni_group_min_max",
    },
    "input": {
        "bit": 8, "per_channel": False,
        "input_algo": "dyn_s8", "unsigned_quant": False,
    },
},
"Quant_Embed_MinMax": {
    "weight": {
        "bit": 8, "per_channel": True, "weight_algo": "min_max",
        "fp16": True, "tie_lmhead": False,
    }
},
```

### 当前实现对照

- `dopt_wrapper.build_layer_strategy`(dopt_wrapper.py:31):赋 `quant_strategy` 名(`Quant_INT8-4_ECO` for Linear,`Quant_Embed_MinMax` for Embedding),轻量化工具读查 strategy.py 拿完整策略 → 效果应一致(若 strategy 名对应)
- `swap.py _STRATEGY_DEFAULT_PARAMS`(swap.py:138):硬编码 `Quant_INT8-4_ECO` weight 5 字段(group_size:128, bit:4, blocksize:128, per_channel:True, weight_algo:omni_group_min_max)→ 与 strategy.py weight 部分一致(当前)
- swap.py 不含 input 部分(A8):swap 只物化 weight int4,input A8 由 `npu::npu_dynamic_quant` 在 QuantLinear forward 处理(`act_quant=True` 硬编码 W4A8)

### 待验证/同步

1. build_layer_strategy 的 strategy 名(`Quant_INT8-4_ECO`/`Quant_Embed_MinMax`)与 strategy.py `strategy_dict` key 一致(轻量化工具能查到)
2. swap.py `_STRATEGY_DEFAULT_PARAMS` 与 strategy.py weight 部分同步(strategy.py 更新时,swap.py 硬编码需同步;swap.py:137 注释「converter venv 不装 dopt_lm, so hardcoded; update here when adding a weight-quant strategy」)
3. 轻量化工具 GPTQ input 策略(A8 `dyn_s8`)与 swap W4A8(`act_quant=True`)一致(轻量化工具产 A8 量化模型,swap 物化 W4A8,需对应)

### 行动项

- [x] 验证 build_layer_strategy strategy 名与 strategy.py strategy_dict key 对应 → **一致**:`Quant_INT8-4_ECO` / `Quant_Embed_MinMax` 均在 strategy_dict(dopt_wrapper.py:22-23 赋值,strategy.py:35,50 定义)
- [x] 同步 swap.py _STRATEGY_DEFAULT_PARAMS 与 strategy.py weight → **已同步**:swap.py:138-143 `Quant_INT8-4_ECO` weight 5 字段(group_size:128, bit:4, blocksize:128, per_channel:True, weight_algo:omni_group_min_max)与 strategy.py:36-42 完全一致;注释(swap.py:133-137)已注明「aligned with strategy.py」「dopt 53ad5ec group_size 256→128 synced」
- [x] 确认轻量化工具 input 策略(A8 dyn_s8)与 swap act_quant=True(W4A8)一致 → **一致**:strategy.py Quant_INT8-4_ECO input(bit:8, input_algo:dyn_s8, unsigned_quant:False)=A8 动态量化;swap act_quant=True(CLI-009 硬编码)→ QuantLinear forward 由 `npu::npu_dynamic_quant` 处理 A8;轻量化工具产 W4A8 模型(weight int4 + input A8),swap 物化 W4A8,对应

### 验证结果(2026-09-01)

三者全部一致,无需改代码:

| 验证点 | 对照(source : line) | 结论 |
|--------|----------------------|------|
| strategy 名对应 | build_layer_strategy 赋 `Quant_INT8-4_ECO`(Linear)/`Quant_Embed_MinMax`(Embedding)(dopt_wrapper.py:22-23,39,42);strategy_dict 含两 key(strategy.py:35,50) | ✅ 一致 |
| weight 字段同步 | swap.py _STRATEGY_DEFAULT_PARAMS 5 字段(group_size:128/bit:4/blocksize:128/per_channel:True/omni_group_min_max)(swap.py:138-143)= strategy.py Quant_INT8-4_ECO weight(strategy.py:36-42) | ✅ 完全一致 |
| input/A8 对应 | strategy.py input(bit8/dyn_s8/unsigned False)(strategy.py:43-48)=A8;swap act_quant=True(CLI-009)→`npu::npu_dynamic_quant` A8(swap.py:482) | ✅ W4A8 对应 |

> 维护约束(swap.py:134 注释):converter venv 不装 dopt_lm,`_STRATEGY_DEFAULT_PARAMS` 硬编码;**strategy.py 新增 weight-quant strategy 时须手动同步 swap.py**。当前仅 `Quant_INT8-4_ECO` 一个 weight strategy,已同步;`Quant_Embed_MinMax` 为 embedding-side(swap.py:182 注释,不走 Linear swap,不在 _STRATEGY_DEFAULT_PARAMS)。

### 关联

- 源码:`cann_lm/converter/quant/dopt_wrapper.py:31`(build_layer_strategy)、`cann_lm/converter/swap.py:138`(_STRATEGY_DEFAULT_PARAMS)、`cann_lm/converter/quant/dopt_lm/quantizer/strategy.py`(策略定义)
- 相关:CLI-009(swap_from_dopt 删 act_quant,硬编码 W4A8)

---

## CLI-012 TODO:gemma4 PLE 量化未适配(_build_external_extras 用 HF 原始 float)

| 字段 | 值 |
|------|-----|
| 编号 | CLI-012 |
| 类型 | TODO(bug,待修复) |
| 背景 | gemma4 PLE(`embed_tokens_per_layer`)被轻量化工具量化(`Quant_Embed_MinMax` → QEmbedding int8 per-channel + scale),`host_weights_*.bin` pack `_q`/`_scale`(int8 `[vocab,8960]` + float32 `[vocab]`)。但 `_build_external_extras`(gemma4/export_wrapper.py:273-283)用 `m.embed_tokens_per_layer(input_ids)`(**HF 原始 float**,swap_embeddings_from_dopt 不替换 Embedding 模块),图输入 `per_layer_token_identity` float(**无 scale**),与 embed_tokens 量化处理(int8+scale 作图输入 + 图 dequant)不对称 |
| 状态 | **已修复(2026-09-02)** |
| 发现日期 | 2026-08-31 |

### 证据

**host_weights_info.json**(epair_gemma4_quant_path):
```json
{ "name": "model.model.embed_tokens_per_layer.weight_q", "dtype": "int8", "shape": [262144, 8960] },
{ "name": "model.model.embed_tokens_per_layer.weight_scale", "dtype": "float32", "shape": [262144] }
```
PLE 打包了 `weight_q`(int8)+ `weight_scale`(float32)——轻量化工具量化了 PLE。

**轻量化工具产物** model.safetensors:
```
model.embed_tokens_per_layer.weight [262144, 8960] (fakequant fp16)
model.embed_tokens_per_layer.quant_op.weight_quantizer.s [262144] (scale)
model.embed_tokens_per_layer.quant_op.weight_quantizer.bit/unsigned_quant/offset/max/min
```
轻量化工具用 QEmbedding 量化 PLE(int8 + scale + meta)。

**_build_external_extras**(gemma4/export_wrapper.py:280):
```python
token_identity = m.embed_tokens_per_layer(input_ids).reshape(...).detach()   # HF 原始 float
```
`m.embed_tokens_per_layer` 是 cann_lm 模型的 Embedding(load_model 加载 HF 原始 fp,swap 不替换)。

### 问题

| | embed_tokens(quant) | PLE(embed_tokens_per_layer,quant) |
|---|---|---|
| 轻量化工具量化 | ✅ int8+scale | ✅ int8+scale |
| host_weights pack | ✅ _q + _scale | ✅ _q + _scale |
| **导出时图输入** | `emb_q`(int8)+ `emb_scale`(从 `_quant_embeddings` gather) | `per_layer_token_identity`(**HF 原始 float**,从 `m.embed_tokens_per_layer` 查表) |
| forward | `int8*scale→float` dequant(图内) | 直接用 float(无 dequant,无 scale 参数) |

1. **数值不一致**:导出时图 trace 用 **HF 原始 PLE**(_build_external_extras);host 推理传 **轻量化工具量化 PLE dequant**(`int8*scale`)。HF 原始 ≠ 量化 dequant → 图期望原始数值,host 传量化数值 → **精度不匹配**
2. **forward 无 scale**:`per_layer_token_identity` 单 tensor(无 `_scale` 参数),图不 dequant PLE;host 需在 host 侧 dequant 再传图——但图 trace 用 HF 原始(不是 dequant),host 传 dequant,数值对不上
3. **与 embed_tokens 不对称**:embed_tokens quant 走 int8+scale 作图输入 + 图 dequant;PLE quant 却走 HF 原始 float(未适配 int8)

### 修复方向

1. **`_build_external_extras` 适配 PLE 量化**:检测 `_quant_embeddings` 含 PLE(`model.model.embed_tokens_per_layer.weight`)→ 从中取 int8+scale,gather `input_ids` → `ple_q`(int8)+ `ple_scale`(per-token scale)
2. **forward 加 `per_layer_token_identity_scale` 参数**:`per_layer_token_identity`(int8)* `per_layer_token_identity_scale` → float dequant(类似 `embeddings`/`embeddings_scale`)
3. **get_sample_inputs 传 `ple_q` + `ple_scale`**(quant 时,与非 quant 的 float PLE 分支区分)
4. **host 推理脚本适配**:读 host_weights PLE `_q`/`_scale`,gather + 传 int8+scale(图 dequant),或 host dequant 传 float(图不 dequant,需与导出一致)

### 行动项

- [x] ~~`_build_external_extras` 加 quant 分支~~ → 实现在 `get_sample_inputs(quant)` 直接从 `_quant_embeddings` gather `ple_q`+`ple_scale`(`gemma4/export_wrapper.py:258-269`),不调 `_build_external_extras`(后者保持非 quant float 分支)
- [x] forward 加 `per_layer_token_identity_scale` 参数 + dequant 逻辑(`gemma4/export_wrapper.py:61,70-72`,`int8*scale→float`,scale `[b,s]` broadcast over `[num_layers,hidden_per_layer_input]`)
- [x] get_sample_inputs(quant) 传 `ple_q`+`ple_scale`(`:258-269`);非 quant/图内分支加 `None` ple_scale 保持 11 元素返回顺序(`:275,281`)
- [x] host 推理脚本适配 PLE int8+scale → `verify_gemma4_epair_quant.py`(epair 回环验证,读轻量化工具产物 + `swap_embeddings_from_dopt` + `set_quant_embeddings`,host 不需手动 gather — `get_sample_inputs` 已 gather)
- [x] 验证:case 4(gemma4 场景1)epair 导出跑通(`n_quantized=277,weights 1484.6 MB,host_weights 2753.6 MB`,含 PLE `_q`/`_scale`)+ `verify_gemma4_epair_quant.py` ep.module(traced) vs wrapper(eager) `max_diff=0`(PLE dequant 在 trace 中正确)

### 修复(2026-09-02)

**PLE quant 适配 epair 路径**(3 处代码 + 1 验证脚本):

1. **`cann_lm/models/gemma4/export_wrapper.py`**:
   - `forward`(:49-72)加 `per_layer_token_identity_scale` 参数 + PLE dequant(`per_layer_token_identity(int8) * per_layer_token_identity_scale → float`,仿 `embeddings/embeddings_scale`)
   - `get_sample_inputs(quant)`(:246-269)从 `_quant_embeddings` 取 PLE int8+scale,gather `input_ids`→`ple_q`+`ple_scale`;非 quant/图内分支加 `None` ple_scale 保持 11 元素返回顺序
 2. **`cann_lm/models/fused_ops/fused_infer_attention_score.py` `_fa_cpu`**(:92-100):兼容 gemma4 mask 方阵 `[b,1,kv_max,kv_max]` bool(截前 seq_len 行匹配 `attn_weights [b,heads,seq_len,kv_max]` + bool→float `True→-inf`);非方阵 float mask(qwen3.5 等)不受影响
3. **`examples/.../debug_gemma4/verify_gemma4_epair_quant.py`**(新,97 行):epair 回环验证(load+轻量化工具+set_quant_embeddings+export+ep.module vs wrapper logits 一致+PLE dequant sanity)

**验证结果**(`verify_gemma4_epair_quant.py --hf-path gemma-4-E2B-it --quant-path /tmp/opencode/gemma4_quant_output`):
- PLE quantized=True(`embed_tokens`+`embed_tokens_per_layer` 均 int8+scale)
- `ple_q (1,128,35,256) int8, ple_scale (1,128) f32`(gemma4 35 层 × 256 hidden_per_layer)
- **ep.module(traced) vs wrapper(eager) logits max_diff=0.000000e+00** → forward 的 PLE dequant 在 torch.export trace 中正确保留,host 传 ple_q+ple_scale 时图内 dequant 与 eager 一致
- PLE dequant: min=-0.4219 max=0.4980 mean=0.0001 nan=False inf=False(数值合理)
- `[PASS] gemma4 quant PLE epair 回环一致`

> OM verify 路径(`run_gemma4_gen.py`,fakequant float PLE)未适配(OM 算子不支持 + 用 fakequant 非 int8),不在 CLI-012 范围。生产 epair host 推理脚本(用 `epair_model_loader` C++)的 PLE 适配由 C++ 按 ep_json 输入顺序自动读 `per_layer_token_identity_scale`(待生产 host 脚本落地确认)。

### 关联

- 源码:`cann_lm/models/gemma4/export_wrapper.py:273-283`(_build_external_extras)、`cann_lm/models/export_wrapper.py:44-60`(forward per_layer_token_identity/embeddings_scale)、`cann_lm/converter/swap.py:35-56`(swap_embeddings_from_dopt,不替换 Embedding)
- 产物:`host_weights_info.json`(PLE `_q`/`_scale`)、轻量化工具 `model.safetensors`(PLE fakequant+scale)
- 相关文档:[feature_embedding_design.md](./feature_embedding_design.md)「gemma4 PLE 特殊」「轻量化工具对 gemma4 的实现」
- 相关:CLI-011(build_layer_strategy 与 strategy.py 一致性)

---

## CLI-013 TODO:端侧 LLM 工具 CLI 与文档提供方式调研

| 字段 | 值 |
|------|-----|
| 编号 | CLI-013 |
| 类型 | TODO(调研,非偏离) |
| 背景 | cann-lm-converter 当前 CLI 是单脚本 `python cann_lm_converter.py --hf-path ... --mode ...`(argparse,无子命令),文档在 `docs/cann_lm_converter/` 分目录。为评估 CLI/文档提供方式是否需对齐端侧 LLM 工具主流做法,调研 mlx-lm 等端侧工具的命令行入口 + 文档组织 |
| 状态 | **TODO(调研完成,待评估采纳)** |
| 发现日期 | 2026-09-01 |

### 调研对象

端侧(本地/设备侧)LLM 推理或转换工具,与 cann-lm-converter(导出 epair 供 C++ delegate 加载)定位可比(均含「转换/量化 + 运行」链路):

- **mlx-lm**(Apple Silicon,Python+MLX):转换/量化 + 生成 + 微调
- **llama.cpp**(C/C++,多后端含 CANN):GGUF 转换 + cli/serve 推理
- **mlc-ai/mlc-llm**(MLC,多平台):convert/build/serve/chat
- **ollama**(Go):run/pull/serve(运行时,转换在服务侧)

### 调研发现:命令行工具提供方式

| 工具 | CLI 入口风格 | entry point 机制 | 子命令示例 |
|------|-------------|-------------------|-----------|
| mlx-lm | `mlx_lm.<cmd>` 多 entry(点分 namespace) | `console_scripts` 注册多个(`mlx_lm.generate`/`.chat`/`.convert`/`.server`/`.cache_prompt`/`.benchmark`) | `mlx_lm.generate --model ... --prompt ...` |
| llama.cpp | 统一 `llama <subcmd>` | C++ binary(`llama`)+ Python 脚本(`convert_hf_to_gguf.py` 顶层) | `llama cli -hf ...` / `llama serve -hf ...` |
| mlc-llm | 统一 `mlc_llm <subcmd>`(typer) | `console_scripts` | `mlc_llm convert` / `mlc_llm build` / `mlc_llm chat` / `mlc_llm serve` |
| ollama | 统一 `ollama <cmd>` | Go binary | `ollama run` / `ollama pull` / `ollama serve` / `ollama list` |

**共性**:端侧工具普遍用「**统一入口 + 子命令**」(`llama cli/serve`、`mlc_llm convert/build`、`ollama run/pull`),而非单脚本;mlx-lm 用点分多 entry(`mlx_lm.generate`)是 Python 包惯例的变体。子命令粒度按「动作」划分(cli/serve/convert/build/chat/run/pull),每子命令独立 `--help`。

**cann-lm-converter 现状**:`python cann_lm_converter.py --hf-path ... --mode prefill-decode ...`(单脚本 + argparse,`--mode` 区分 prefill-decode/single,`--quant-method`/`--quant-path` 区分量化场景)。无子命令,无 console_scripts entry(脚本直调)。

### 调研发现:文档提供方式

| 工具 | 文档结构 | 命令参考 | 示例 |
|------|---------|---------|------|
| mlx-lm | README.md 单文件(Quick Start/Command Line/Python API/各功能章节)+ `LORA.md`(微调)+ `examples/`(generate_response.py 等)+ `AGENTS.md`/`CLAUDE.md` | README「Command Line」章节 + 每命令 `-h` | `examples/` 目录 |
| llama.cpp | README.md(主,Quick Start/Description/Backends)+ `docs/`(按主题分文件:build.md/docker.md/android.md/multi-gpu.md/backend/*.md)+ `tools/<tool>/README.md`(每工具独立)+ wiki | `tools/cli/README.md` + `tools/server/README.md`(每工具 README)+ `docs/build.md` | `examples/` |
| mlc-llm | docs site(独立站点)+ README | docs site 命令参考页 | `examples/` |
| ollama | docs site(独立站点) | docs site 命令参考 | — |

**共性**:文档分两类——**单 README + examples/**(mlx-lm,轻量)或 **README + docs/ 分主题 + 每工具 README**(llama.cpp,结构化)。命令参考要么在 README 章节,要么每子命令/工具独立 README(llama.cpp `tools/<t>/README.md`)。`-h`/`--help` 是 CLI 自文档兜底。llama.cpp 的「docs/ 按主题分文件 + tools/<t>/README.md 每工具」结构与 cann-lm 当前 `docs/cann_lm_converter/{cli,quant,export}/` 分目录接近。

**cann-lm-converter 现状**:`docs/cann_lm_converter/` 分目录(cli/cli_design.md、quant/quant_design.md、export/export_design.md、README.md(总览)、TODO_notes.md 等),接近 llama.cpp docs/ 分主题模式;无独立「命令参考」页(CLI 参数散在 cli_design.md);examples/ 下有用例脚本(run_pd6_cases.sh)但无独立 example 文档。

### 对 cann-lm-converter 的启示(待评估)

1. **CLI 入口**:
   - 当前单脚本 `python cann_lm_converter.py` 适合 converter 单一职责(导出 epair)。若后续扩展「验证/加载/serve」等动作,可参考 `llama cli/serve` 或 `mlc_llm convert/build/serve` 子命令化(`cann-lm convert`/`cann-lm serve`)
   - entry point:当前 whl(torch_delegate)已装,但 `cann_lm_converter.py` 仍脚本直调。可考虑 console_scripts 注册 `cann-lm-converter` 命令(参考 mlx-lm `mlx_lm.convert` / mlc-llm `mlc_llm convert`),pip install 后直接调用,免去 `python cann_lm_converter.py`
2. **文档**:
   - 当前 `docs/cann_lm_converter/` 分目录接近 llama.cpp docs/ 模式(可保留)。可补「命令参考」独立页(CLI 参数汇总表,参考 mlx-lm README「Command Line」章节或 llama.cpp `tools/cli/README.md`),现在参数散在 cli_design.md
   - examples/ 用例脚本(run_pd6_cases.sh)无独立 README,可参考 llama.cpp `tools/<t>/README.md` 给每用例/工具补 README
3. **`-h`/`--help` 自文档**:确保 `cann_lm_converter.py --help` 输出完整可用(参考 mlx-lm「Use -h to see a list of available options for a command」)

### 行动项

- [x] 评估 cann-lm-converter 是否需子命令化 → **保持单命令**(converter 单一职责导出 epair,无 serve/verify 动作;子命令化待职责扩展再议)
- [x] 评估 console_scripts 注册 `cann-lm-converter` 命令 → **已完成**:`scripts/cann_lm_converter/pyproject.toml:20-21` `cann-lm-converter = "cann_lm.converter.cann_lm_converter:main"`,pip install 后直调
- [x] 补「命令参考」独立文档页 → **已完成**:cli/INDEX.md 加「命令参考」节(console_scripts 入口 `cann-lm-converter` + 用法示例 5 类)
- [ ] examples/ 用例补 README(参考 llama.cpp `tools/<t>/README.md`)

### 关联

- 调研来源:mlx-lm(github.com/ml-explore/mlx-lm)、llama.cpp(github.com/ggml-org/llama.cpp)、mlc-llm(github.com/mlc-ai/mlc-llm)、ollama(github.com/ollama/ollama)
- 相关代码:`cann_lm/converter/cann_lm_converter.py`(CLI)、`pyproject.toml`(entry point,torch_delegate 已有)
- 设计文档:[cli_design.md](./cli/cli_design.md)(CLI 参数/调用链)、[README.md](./README.md)

---

## CLI-014 TODO:PD 分离导出只存一份权重,decode 权重丢弃(PD 权重不一致时失效)

| 字段 | 值 |
|------|-----|
| 编号 | CLI-014 |
| 类型 | TODO(遗留设计假设限制,非偏离) |
| 位置 | `cann_lm/converter/saver.py:220-238`(prefill `save_weights=True` / decode `save_weights=False`)+ docstring:142-148「shared weights (packed once)」;设计:[export_design.md:936](./export/export_design.md)「权重共享:两张图 state_dict 相同」 |
| 现状 | `save_epair_prefill_decode` 假设 prefill/decode 两图 state_dict 相同(同一模型),只 prefill 存权重(`consolidated_weights_save(prefill_epair, save_weights=True)` 写 `weights_*.bin` + `weight_info.json`),decode `save_weights=False`(只写 `decode_ep_json_with_ascend_ir.json`,权重丢弃) |
| 违反约定 | 无(设计假设,非偏离) |
| 发现日期 | 2026-09-01 |
| 状态 | **已实施（B，2026-09-28）**：v2 hash 去重统一格式（单图/PD 都 v2，去 `version` 字段 + 去 v1 兼容 + 去 `inline_fused_attn_seq_constants`/`inline_weight_scale_dtype`）。验证：qwen3.5/gemma4 端到端（单轮/多轮/长 prompt 多 chunk）全通过 |

### 设计假设(当前)

`save_epair_prefill_decode` docstring(:142-148)与 [export_design.md:936](./export/export_design.md) 明确:
- **权重共享**:两张图 state_dict 相同(同一模型),`pack_weights` 只执行一次,`weight_info.json` + `weights_*.bin` 只写一份
- prefill 存权重(`save_weights=True`),decode `save_weights=False`(权重丢弃,只写 ep_json)
- C++ 加载侧(`GeWeightLoader`)读 `weight_info.json` + `weights_*.bin`,prefill/decode 两图共用同一份权重(层名一致)

### 遗留问题

当 **PD 分离且权重不一致**时(非同一模型/state_dict 不同),decode 权重丢弃会出错:

1. decode `save_weights=False` → decode 引用的权重节点不进 `weights_*.bin` / `weight_info.json`
2. C++ 加载 decode graph 时,decode ep_json 引用的权重在 prefill 的 `weight_info.json` 找不到对应条目(层名/shape/value 不同)→ 加载失败或错配
3. 场景:PD 不同量化(prefill W8A8 / decode W4A8)、PD disaggregated 部署(两份权重)、PD 层名差异(带前缀)

### 当前可行的前提

仅当 prefill 与 decode **同一模型、同 state_dict**(权重组完全一致)时,decode 复用 prefill 权重安全。这是当前 `--mode prefill-decode` 的隐含约束,`run_pd6_cases.sh` 各 case 均同一模型导出 PD 两图,满足此约束。

### 关键修正:C++ 已按图编译(原方向 2 已就绪)

`run_compile.sh:8-9` + `compile_main.cc:61-73`:编译按图分别调用——每次 `cann_lm_compile <model_json> <weights_index>` 处理一张图,`EpairModelLoader` 接收 per-graph 的 `weights_index`。故原「待解决方向 2:C++ 加载侧 per-graph 权重索引」**已满足**。当前 decode 能编译,是 `run_compile.sh` 让 decode 复用 prefill 的 `weight_info.json`(PD 同名命中)。**缺口纯在 Python 侧**(`save_epair_prefill_decode` decode `save_weights=False` 丢权重),非 C++ 不支持。

且**单 artifact API 下 PD 权重必然相同**(同一模型仅 seq_len 不同 trace)→ 当前 dedup 是**正确的,非 bug**。「不一致」只源于下方场景。

### 场景(PD 权重真不一致的触发)

| 场景 | 权重是否真不同 | 触发 |
|------|--------------|------|
| ① disaggregated 部署 | 相同值,两台机器各需自包含一份 | 部署模式(非导出不一致) |
| ② PD 不同量化(prefill W8A8/decode W4A8) | 真不同值 | 需双 artifact API(当前单 artifact 产不出) |
| ③ 层名前缀差异(不同 wrapping) | 同值异名 | 同②,需双 artifact |

### 解法

**A. 务实(Python-only,推荐先做)**:`consolidated_weights_save`/`_pack_weights_to_dir` 加 `weights_prefix` 参数 + `save_epair_prefill_decode` 加 `pd_split_weights` 开关(或自动判定)。关=现状(PD 同,decode `save_weights=False` 复用,零开销);开=prefill 写 `weight_info_prefill.json`+`weights_prefill_*.bin`,decode `save_weights=True` 写 `weight_info_decode.json`+`weights_decode_*.bin`(两图自包含)。`model_config.json` 记 `pd_weight_split`。**C++ 零改动**;`run_compile.sh` 已支持显式 `<model_json> <weights_index>`(:9),split 时给 decode 传 `weight_info_decode.json`。覆盖①②③。

**B. hash 去重(进阶,长期)**:仿 GE `FileConstant`(`external_weight.md`)——统一 `weights_*.bin` 按 content hash 去重 + `weight_info.json` 改两级索引(`blobs:{hash:{offset,size,format,kind}}` + `graphs:{prefill:{name→hash},decode:{name→hash}}`)。PD 同值→共一 hash(自动 dedup);异值→两 hash;部分共享(共享 embed+不同 Linear)→共享 dedup、不同各存。缺点:两级索引 + C++ `GeWeightLoader` 改读 `graphs[G][name]→hash→blobs[hash]`,改动大。仅「部分共享 PD」成真才值得。

**C. 双 artifact API(场景②③必需)**:`save_epair_prefill_decode(prefill_artifact, decode_artifact=None, ...)`——decode 传独立 artifact 产真不同权重 PD;`None` 回退现状。与 A 组合(split 开关+双 artifact)。

**D. 一致性 assert(安全网)**:单 artifact dedup 路径加 `assert prefill_sd == decode_sd`(name/shape/dtype/bytes),违反 fail-fast(而非静默产坏 epair)。零成本护现状假设。

**推荐**:先 A+D(A 复用已就绪的 per-graph 编译,无 C++ 改动;D 护栏);C 留给②③真出现;B 仅长期部分共享场景。与 CLI-015(`format` ND/NZ)+ CLI-019(`kind` p_/c_/b_)协同:split 后的 per-graph `weight_info_{prefill,decode}.json` 条目同载二字段,三 TODO 一起落地最省。

### 行动项

- [ ] **A**:`consolidated_weights_save`/`_pack_weights_to_dir` 加 `weights_prefix`;`save_epair_prefill_decode` 加 `pd_split_weights` 开关;`model_config.json` 记 `pd_weight_split`;`run_compile.sh` auto-find 加 `weight_info_decode` fallback(或显式传) — **未做（B 已覆盖 A 的目标且更优）**
- [ ] **D**:单 artifact 路径加 PD state_dict 一致性 assert(fail-fast) — **未做（v2 hash 去重本身即 fail-fast：PD 不同值各存，不再隐式共享）**
- [ ] **C**(场景②③):`save_epair_prefill_decode` 扩展可选 `decode_artifact` — **未做（双 artifact 场景②③未出现）**
- [x] **B**(已实施,2026-09-28):hash 去重 + 两级索引 + C++ `GeWeightLoader` 改读 — `_pack_weights_v2`/`_pack_weights_pd_dedup`(epair_utils.py) + `consolidated_weights_save_pd`(PD)/`consolidated_weights_save` 改调(单图统一 v2) + `ge_weight_loader.cc` v2 两级(`graphs[G][name]→hash→blobs[hash]`) + 去 v1 循环(无 graphs 报错) + `ge_graph_builder`/`epair_model_loader` graph_name 推断(ep_json filename→prefill/decode/single) + 去 `version` 字段(C++ 检测 `graphs`/`blobs` 存在) + 去 `inline_fused_attn_seq_constants`/`inline_weight_scale_dtype`(v2 解决 actual_seq PD 共享冲突,不需 inline workaround)
- [x] 文档:[export_design.md](./export/export_design.md)「PD 权重分离场景(hash 去重 v2)」节(B 方案设计 + schema + Python/C++ 接口 + 验证矩阵)
- [x] 验证(2026-09-28):converter UT 127 passed;qwen3.5-0.8B + gemma4-E2B 端到端(导出 v2→OM 编译 C++ v2 加载→NPU 出字);单图 `graphs["single"]` + PD `graphs["prefill"/"decode"]` hash 去重;多轮对话 + 长 prompt 多 chunk prefill;去 inline 后出字一致(v2 从根本解决 actual_seq 共享冲突)

### 关联

- 源码:`cann_lm/converter/saver.py:220-238`(save_epair_prefill_decode 权重保存)、`cann_lm/converter/export/epair_utils.py:131`(consolidated_weights_save save_weights 语义)、C++ `GeWeightLoader`(权重加载)
- 设计文档:[export_design.md](./export/export_design.md)「prefill/decode 分离导出 / 权重共享」(L928-962)
- 相关:CLI-010(load_model 加载浪费,swap 丢弃 Linear fp)

---

## CLI-015 TODO:参考 weight_format_design.md 实现权重 ND 到 NZ 格式转换

| 字段 | 值 |
|------|-----|
| 编号 | CLI-015 |
| 类型 | TODO(遗留问题/待实现,非偏离) |
| 位置 | 设计:[weight_format_design.md](./export/weight_format_design.md);代码待实现:`cann_lm/converter/export/weight_format/`(`layout.py` `nd2nz` + `rules.py` `resolve_format`)+ `weight_format_rules.json` + `epair_utils._pack_weights_to_dir` 集成 + `inline_weight_layout` + C++ `aten_tensor_desc_mapper.cc:112` `MapGeLayoutString` / `ge_weight_loader.cc:115` `Pt2LayoutFromGeString` |
| 背景 | `weight_format_design.md` 设计了权重落盘从 ND(行主序)转 NZ(FractNZ 16×16 块主序),让 Ascend cube 计算权重 Const 直接以 `FORMAT_FRACTAL_NZ` 进 GE 图,避免 OM 编译/运行期 transdata。Stage 1(fp16 `MatMul` Linear 权重 `*_proj`)已设计,但 `weight_format/` 包 + `weight_format_rules.json` **0 实现**(`ls` 确认 NOT implemented) |
| 违反约定 | 无(待实现) |
| 发现日期 | 2026-09-01 |
| 状态 | **L2 通路跑通(bare-matmul + nn.Linear A3, NPU max_diff=0);L3 整网待验(2026-09-07)** |

### 设计已就绪(Stage 1,weight_format_design.md)

- `weight_format/layout.py`:`nd2nz(tensor, block=16, pad=True)` [M,K] ND → [ceil(M/b),ceil(K/b),b,b] 块主序(非 b 倍数维内部补零)
- `weight_format/rules.py`:`load_rules`(读 shipped `weight_format_rules.json`)+ `resolve_format(name, is_quantized)`(按 `name_pattern` + `op_type` 分流 MatMul/QuantBatchMatmulV4)
- `weight_format_rules.json`:默认配置(`*_proj.weight` → NZ;`QuantBatchMatmulV4` → ND Stage 2 待研究)
- `epair_utils._pack_weights_to_dir` 集成:插 `resolve_format` + `nd2nz`(仿 `transdata_scale` 先例),`weight_info.json` 条目加 `"format"` 字段
- **环境变量总开关** `CANN_LM_ENABLE_WEIGHT_TRANS`（默认关闭，`1` 启用）：env 未设 → packer 跳过 `resolve_format` + `is_quantized` 探测，全 ND 落盘，代码 dormant 零风险；仅 Python 导出侧，C++ 由 ep_json layout 字符串驱动无 env
- `inline_weight_layout`:ep_json `tensor_values.layout` + `ascend_ir.desc.layout` ND→NZ
- C++ `MapGeLayoutString`/`Pt2LayoutFromGeString` 加 `"NZ"` case

### 不做(Stage 2,含强制研究门)

量化 `QuantBatchMatmulV4` int4 nibble-packed 权重 NZ——标准 `nd2nz` 会打散 nibble 配对,需专用 `pack_int4_nz`,但 int4-NZ 字节布局由 CANN V4/V3 runtime kernel 决定,不能凭推断实现。研究门:① `atc --mode=1` 解析现有量化 OM 看有无 TransData 残留(CANN 自动转则量化 weight 保持 ND);② 读 GE 源码 `quant_batch_matmul_v4` op_graph + 950-ops int4 NZ kernel 拿字节布局。当前 ND 量化 OM 已跑通,Stage 2 是性能优化非正确性必需。

### 待验证

1. MatMul NZ 输入接受性:`reg_register.h` MatMul 是否接受 weight Const `FORMAT_FRACTAL_NZ` 而不插运行期 transdata(需最小 LLM（截断 N 层）NZ 导出 + OM 编译验证,参数化放 `examples/`)
2. NZ shape 语义:实现按 GE 标准(逻辑 shape + 内部 pad),OM 编译实测确认 kernel 正确读 partial block

### 行动项

- [x] Python `weight_format/{layout,rules}.py` + `weight_format_rules.json` + `epair_utils._pack_weights_to_dir` env-gated `nd2nz` + `weight_info.format` 字段(13 测试过)
- [x] C++ `model_types.h`(format_str) + `ge_weight_loader.cc`(读 format) + `aten_tensor_desc_mapper.cc`(`MapGeLayoutString` NZ case) + `ge_graph_builder.cc:CreateConstNode`(2D shape + FRACTAL_NZ, 不 reshape) + input desc NZ 覆盖
- [x] ~~`inline_weight_layout`~~ 已弃用(ep_json pt2 枚举无 NZ;改走 `weight_info.format` + C++ input 覆盖)
- [x] L2 bare-matmul 通路跑通(A3):`nd2nz` permute `(2,0,1,3)` + 2D shape + FRACTAL_NZ(不 reshape)+ input NZ 覆盖 → OM 编译过 + NPU max_diff=0(2026-09-07)
- [x] L2 nn.Linear 通路跑通(A3):skip Transpose + `transpose_x2=True` + `nz_transpose_passthrough` 移到 `GraphBuildContext`(跨节点共享)→ NPU max_diff=0(2026-09-07)
- [x] L2 非 16 对齐 shape 限制确认(130×130):GE `CheckTensorIsValid` 不认 NZ padding → 非 16 对齐不可行(2026-09-07)
- [ ] L3 整网(A3)
- [ ] C++ 回退死路(Transpose perm 预扫 + ① 死路代码)

### 关联

- 设计文档:[weight_format_design.md](./export/weight_format_design.md)(Stage 1 设计 + Stage 2 研究门)
- 相关代码:`cann_lm/converter/export/epair_utils.py`(`_pack_weights_to_dir` 集成点 + `inline_weight_scale_dtype` 先例)、C++ `aten_tensor_desc_mapper.cc`/`ge_weight_loader.cc`/`ge_graph_builder.cc`
- 相关:CLI-014(PD 权重产物布局)、CLI-016(epair 权重与 const 分离)

---

## CLI-016 TODO:epair 权重与 const 分离

| 字段 | 值 |
|------|-----|
| 编号 | CLI-016 |
| 类型 | TODO(遗留问题/待澄清+待实现) |
| 位置 | `cann_lm/converter/export/epair_utils.py`(`_pack_weights_to_dir`/`ConsolidatedWeights`,权重与 const 混合落盘)+ `weight_info.json`(索引)+ C++ `GeWeightLoader`(加载) |
| 背景 | epair 当前 `weights_*.bin` 把模型权重(Linear `*.weight`、Embedding、norm、bias 等)与图常量(const tensor,如 fused_attn_seq 常量、shape/dtype 常量等)混合落盘到一个 bin,`weight_info.json` 统一索引。需将「权重」与「const」分离(分文件/分类索引),便于 C++ 加载侧区分权重 Const 节点与图常量,支持差异化处理(如 NZ 格式只作用于权重 const 保持 ND;或 const 外置/共享) |
| 违反约定 | 无(待澄清+待实现) |
| 发现日期 | 2026-09-01 |
| 状态 | **TODO(待澄清范围+待实现)** |

### 待澄清

1. 「const」范围:仅图常量(`ascend_ir` 中的 Const 节点,如 inline 的 fused_attn_seq 常量、scalar 常量)?还是含所有非 Linear-weight 张量(norm/bias/embed)?
2. 分离粒度:分文件(`weights_*.bin` vs `consts_*.bin`)+ 分别索引(`weight_info.json`/`const_info.json`)?还是同文件分类索引(`weight_info.json` 加 `"kind": "weight"|"const"` 字段)?
3. C++ 加载侧:`GeWeightLoader` 是否需区分 weight/const 走不同加载路径(如 weight 支持 NZ 格式,const 保持 ND;const 可能外置/共享)

### 待解决方向

1. **产物布局**:`weights_*.bin`(权重)+ `consts_*.bin`(图常量)分文件 + `weight_info.json`/`const_info.json` 分索引;或同 bin + `kind` 字段分类
2. **C++ 加载**:`GeWeightLoader` 按 kind 区分加载(weight 走 NZ/格式转换路径,const 走 ND)
3. **与 CLI-015 协同**:NZ 格式转换只作用于权重(const 保持 ND),分离后 CLI-015 的 `resolve_format` 只匹配 weight kind
4. **model_config.json**:记录 const 分离标记

### 行动项

- [ ] 澄清「const」范围(图常量 vs 非权重张量)与分离粒度(分文件 vs 分类索引)
- [ ] 设计权重/const 分离产物布局(`consts_*.bin` + `const_info.json` 或 `kind` 字段)
- [ ] C++ `GeWeightLoader` 按 kind 区分加载路径
- [ ] 与 CLI-015(NZ 格式)协同:resolve_format 只匹配 weight kind
- [ ] 文档:export_design.md 补权重/const 分离说明

### 关联

- 源码:`cann_lm/converter/export/epair_utils.py`(`_pack_weights_to_dir`/`ConsolidatedWeights`/`weight_info.json`)、C++ `GeWeightLoader`
- 设计文档:[export_design.md](./export/export_design.md)(产物布局/权重加载)、[weight_format_design.md](./export/weight_format_design.md)
- 相关:CLI-015(weight_format NZ,只作用于权重)、CLI-014(PD 权重产物布局)

---

## CLI-017 TODO:torch-delegate 支持 torchair 打桩后整体整理安装脚本

| 字段 | 值 |
|------|-----|
| 编号 | CLI-017 |
| 类型 | TODO(遗留问题/待 torch-delegate 支持后整理) |
| 位置 | `torch-delegate-harmony/build.sh`(只构建 torch-delegate whl + .run,不含 torchair stubs);torchair 安装独立步骤(clone + configure `NO_ASCEND_SDK=1` + `build_and_install.sh` + cp `build/dist/torchair` 到 venv + `LD_LIBRARY_PATH` 设 stubs,见 SKILL.md 步骤3);CANN-LM `scripts/set_env_export.sh`(`LD_LIBRARY_PATH` stubs 在前) |
| 背景 | 当前 converter env(torch-delegate venv + torchair stubs)搭建分散:torch-delegate `build.sh` 只产 `torch_delegate` whl + .run,torchair stubs 需手动 `git clone gitcode.com/Ascend/torchair` + `NO_ASCEND_SDK=1 bash configure` + `bash build_and_install.sh` + 手动 cp `build/dist/torchair`(build_and_install 可能未自动装) + 设 `LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:...`(SKILL.md 步骤3 + AGENTS.md converter env)。torchair stubs 是 converter env 必需(给 torchair GE 引擎 C 符号,避真 NPU `aclrtSetDevice→107001`),但不在 torch-delegate 安装脚本内。等 torch-delegate 支持 torchair 打桩方式(如 `build.sh` 集成 torchair stubs 构建/安装),再整体整理安装脚本(一键搭 converter env) |
| 违反约定 | 无(待 torch-delegate 支持) |
| 发现日期 | 2026-09-01 |
| 状态 | **TODO(待 torch-delegate 支持 torchair 打桩后整理)** |

### 现状(分散的搭建步骤)

converter env = torch-delegate venv + torchair stubs,搭建分散:

1. **torch-delegate**(`build.sh`):venv 创建 + torch 2.9.0+cpu + 三方库 + C++ 编译 + whl + .run(产出 `torch_delegate-0.1.0-*.whl/.run`)
2. **torchair**(手动,SKILL.md 步骤3):`git clone --recursive gitcode.com/Ascend/torchair` + `NO_ASCEND_SDK=1 bash configure` + `bash build_and_install.sh <site-packages>` + 手动 cp `build/dist/torchair`(build_and_install 可能未自动装) + `pip install packaging`
3. **stubs env**(CANN-LM `set_env_export.sh`):`LD_LIBRARY_PATH=${TORCHAIR_ROOT}/build/stubs:${ASCEND_HOME_PATH}/lib64:...`(stubs 在前,避 `aclrtSetDevice→107001`)

问题:torchair stubs 是 converter env 必需(无 stubs → torchair GE 引擎调真 CANN `aclrtSetDevice` → 107001 无设备),但 torch-delegate `build.sh` 不含 torchair 构建/安装,需手动多步。env 跨 torch-delegate(`/mnt/workspace/torch-delegate-harmony`)+ torchair(`/mnt/workspace/torchair`)+ CANN-LM `set_env_export.sh` 三处。

### 待 torch-delegate 支持

- torch-delegate `build.sh` 集成 torchair stubs 构建选项(如 `--with_torchair_stubs`,调 torchair `configure NO_ASCEND_SDK=1` + `build_and_install` + 装 `dist/torchair` 到 venv)
- 或 torchair 作为 torch-delegate 依赖(pip install 或 build.sh 拉取 + 编译 stubs)
- 产出:torch-delegate 安装后 venv 自带 torchair + stubs,无需手动多步

### 整理目标(待支持后)

1. **一键搭 converter env**:`torch-delegate build.sh --with_torchair`(或类似)产 whl + .run + 装 torchair stubs 到 venv
2. **env 脚本简化**:CANN-LM `set_env_export.sh` 只 source CANN + 设 `LD_LIBRARY_PATH`(stubs 路径由 torch-delegate 安装记录/默认),无需手动维护 torchair 路径
3. **文档统一**:SKILL.md 步骤3(torchair 手动)合并到 torch-delegate 安装文档,或 torch-delegate 文档说明 torchair stubs 集成

### 行动项

- [ ] 等 torch-delegate 支持 torchair 打桩方式(`build.sh` 集成或依赖)
- [ ] 整理安装脚本:torch-delegate `build.sh` 一键产 whl + .run + torchair stubs
- [ ] CANN-LM `set_env_export.sh` 简化(stubs 路径由 torch-delegate 安装默认)
- [ ] SKILL.md / AGENTS.md 文档统一(torchair stubs 集成到 torch-delegate 安装)

### 关联

- 安装脚本:`torch-delegate-harmony/build.sh`(torch-delegate 构建)、`torchair/build_and_install.sh`(torchair 构建)
- env:CANN-LM `scripts/set_env_export.sh`(converter env,`LD_LIBRARY_PATH` stubs)、`scripts/_env_common.sh`(`TORCHAIR_ROOT`/`TORCH_DELEGATE_HOME` 默认)
- 文档:`examples/.../skills/torch-delegate-epair/SKILL.md`(步骤3 torchair 手动安装)、CANN-LM AGENTS.md「运行环境设置(converter)」

---

## CLI-018 TODO:参考 mlx-lm 环境处理简化 cann-lm 多套环境

| 字段 | 值 |
|------|-----|
| 编号 | CLI-018 |
| 类型 | TODO(遗留问题/调研+简化方向,非偏离) |
| 位置 | 调研:mlx-lm `setup.py`(extras + install_requires)+ mlx 库设备抽象(`mx.default_device`/`mx.device_info`)+ 4 env 变量代码内设;cann-lm `scripts/set_env_{export,compile,generate,npu}.sh` + `_env_common.sh` + AGENTS.md「运行环境设置」 |
| 背景 | mlx-lm 环境处理极简:单一 mlx 库(pip install mlx-lm 即用,macOS 自动拉 mlx,Linux 选 extras cpu/cuda)+ 设备全交 mlx 库(`mx.default_device()` 自动选 macOS→Metal / Linux→CPU)+ 仅 4 env 变量(`TOKENIZERS_PARALLELISM`/`MLXLM_USE_MODELSCOPE`/`MLX_GDN_PACKED`/`TRANSFORMERS_NO_ADVISORY_WARNINGS`,代码内设,无需 source)+ wired memory(`mx.set_wired_limit`)+ 无 env 脚本/stubs/LD_LIBRARY_PATH 管理。cann-lm 相反:4 套 env(converter venv+stubs / compile / generate / npu)+ `LD_LIBRARY_PATH` stubs 在前避 `aclrtSetDevice→107001` + `ASCEND_HOME_PATH` + `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + `TORCH_DELEGATE_*` + torchair 源码编译 + torch_npu 不装(stub 路径)。根因:mlx-lm 依赖单一 mlx 库(Apple 官方,设备抽象完整);cann-lm 跨 CANN+torchair+torch_npu+torch-delegate 多组件,组件间 ABI/版本耦合(torchair 编译匹配 torch 2.9.0,torch_npu 2.7.1 不互通) |
| 违反约定 | 无(调研+简化方向) |
| 发现日期 | 2026-09-01 |
| 状态 | **TODO(调研完成,简化方向待评估)** |

### mlx-lm 环境处理(调研)

| 维度 | mlx-lm | cann-lm |
|------|--------|---------|
| 环境数量 | 单一(mlx 库统一设备抽象) | 多套(converter venv+stubs / compile / generate / npu) |
| 安装 | `pip install mlx-lm`(macOS 自动拉 mlx,Linux extras) | build.sh + torchair 手动 + CANN 升级 + stubs(CLI-017 待整理) |
| 设备管理 | mlx 库 `mx.default_device()` 自动 | 手动 set_env_export.sh + LD_LIBRARY_PATH stubs + ASCEND_HOME_PATH |
| env 脚本 | 无(pip install 即用) | 4 套 set_env_*.sh |
| env 变量 | 4 个(代码内设) | 多个(ASCEND_HOME_PATH/LD_LIBRARY_PATH/TORCH_DEVICE_BACKEND_AUTOLOAD/TORCH_DELEGATE_*,需 source) |
| 平台 | Apple Silicon 单一 | Ascend NPU + CANN + torchair + torch_npu 多组件 |
| 打桩 | 无需(mlx 直接跑) | stub 路径(torchair stubs 避 aclrtSetDevice→107001) |

### 简化方向(待评估,受多组件耦合限制)

1. **单一依赖收敛**:torch-delegate build.sh 集成 torchair stubs(CLI-017)后,`pip install torch-delegate` 或 build.sh 一键产 whl+stubs,接近 mlx-lm `pip install mlx-lm`
2. **设备抽象层**:cann-lm 无 mlx 库对应的统一设备抽象(CANN/torchair/torch_npu 各管各的);长期可封装 device 抽象层(如 `cann_lm.device.is_available()` 对标 `mx.default_device()`),但受多组件限制
3. **env 内设**:mlx-lm env 变量代码内设(`os.environ[...]=...`),无需 source;cann-lm 部分可内设(如 `TRANSFORMERS_NO_ADVISORY_WARNINGS` 已仿),但 LD_LIBRARY_PATH/ASCEND_HOME_PATH 需 source(因 CANN lib 路径)
4. **env 脚本简化**:CLI-017 后 set_env_export.sh 简化(stubs 路径由 torch-delegate 安装默认);4 套 env 收敛(若 converter/compile/generate/npu 能合并)
5. **打桩透明化**:mlx-lm 无 stubs(mlx 直接跑);cann-lm stubs 是 torchair GE 引擎避真 NPU 的权宜,长期若 torchair 支持纯 CPU 转换(无 GE 引擎调 aclrt)可去 stubs

### 行动项

- [ ] 评估单一依赖收敛(torch-delegate build.sh 集成 torchair stubs 后,接近 pip install 即用)
- [ ] 评估 device 抽象层(对标 `mx.default_device`,受多组件限制)
- [ ] env 变量内设(能内设的仿 mlx-lm 代码内设,减少 source)
- [ ] env 脚本简化(依赖 CLI-017,4 套 env 收敛)
- [ ] 打桩透明化评估(长期,若 torchair 支持纯 CPU 转换可去 stubs)

### 关联

- 调研来源:mlx-lm `setup.py` + `mlx_lm/{generate,server,utils,__init__}.py`(device/env 处理)
- 相关 cann-lm:`scripts/set_env_*.sh` + `_env_common.sh` + AGENTS.md「运行环境设置(converter)」
- 相关遗留:**CLI-017**(torchair 打桩集成 + 安装脚本整理,本条的环境简化依赖 CLI-017)
- 调研文档:[reference_side_tools_survey.md](./reference_side_tools_survey.md) mlx-lm 章节 + 环境处理对比

---

## CLI-019 TODO:导出 epair 时区分 p_/c_/b_ 三类数据

| 字段 | 值 |
|------|-----|
| 编号 | CLI-019 |
| 类型 | TODO(待决策+待实现) |
| 位置 | `cann_lm/converter/export/epair_utils.py`(`_pack_weights_to_dir`:三类混合落盘,虽已遍历 `graph_signature` 的 `inputs_to_parameters`/`inputs_to_buffers`/`inputs_to_lifted_tensor_constants` 但未记录类别)+ `weight_info.json`(条目无类别字段)+ C++ `ge_weight_loader.cc`/`ge_graph_builder.cc:CreateConstNode`(三类统一建 Const) |
| 背景 | epair 导出时 `_pack_weights_to_dir` 把 parameter/const/buffer 三类数据混合写进同一 `weights_*.bin` + `weight_info.json`(条目仅 `{name,offset,size,format}`,**无类别标识**)。三类在 ep_json `tensor_values` 按前缀区分:`p_*`(parameter,模型权重)、`c_*`(const,lifted tensor 常量)、`b_*`(buffer,非持久状态);权威源是 `graph_signature`(pack 时已遍历但未持久化类别)。需在导出时区分,因为:(1) NZ 格式转换(CLI-015)只应作用于 `p_*` 的 `*_proj.weight`,不应碰 `c_*`/`b_*`;(2) C++ 加载侧对三类语义不同(parameter 可 NZ/外置,const 保持 ND/可共享,buffer 非持久);(3) 与 CLI-016 粒度直接相关 |
| 违反约定 | 无(待决策) |
| 发现日期 | 2026-09-05 |
| 状态 | **TODO(待决策方案+待实现)** |

### 两种解法

| 方案 | 产物布局 | 索引 | C++ 加载侧 |
|------|---------|------|-----------|
| ① 分开文件保存 | `weights_*.bin`(p_)+ `consts_*.bin`(c_)+ `buffers_*.bin`(b_)分文件 | `weight_info.json`/`const_info.json`/`buffer_info.json` 分索引 | `GeWeightLoader` 按文件天然分流 |
| ② weight_info 加字段 | 同一 `weights_*.bin` 混合 | `weight_info.json` 条目加 `"kind": "parameter"\|"const"\|"buffer"` | `GeWeightLoader` 读 `kind` 分流 |

### 决策考量

- 方案①边界清晰(物理分离),但改产物布局(3 套 bin+index)+ C++ 加载分流,改动面大;prefill/decode 共享权重(p_)仍只存一份
- 方案②改动小(加 1 字段,pack 时按 `graph_signature` 判定类别),C++ 按 `kind` 走不同路径(NZ 只对 parameter kind);仍混合 bin
- 与 CLI-015 协同:NZ(`weight_info.format`)只作用于 `kind=="parameter"` 的 `*_proj.weight`,const/buffer 保持 ND
- 与 CLI-016 关系:CLI-016 是「权重 vs const」2-way,本条细化为「p_/c_/b_」3-way(把 CLI-016 的「权重」拆成 parameter+buffer)。**解决本条即解决 CLI-016 的「待澄清 1/2」(const 范围 + 粒度)**,建议两条合并决策

### 行动项

- [ ] 决策方案①(分文件)还是②(加 kind 字段)
- [ ] 若②:`_pack_weights_to_dir` 按 `graph_signature` 给条目加 `"kind"`(parameter/buffer/lifted_tensor_constant)
- [ ] C++ `GeWeightLoader`/`CreateConstNode` 按 `kind` 分流(NZ 只对 parameter)
- [ ] 与 CLI-015(NZ format)、CLI-016(权重/const 分离)协同

### 关联

- 源码:`cann_lm/converter/export/epair_utils.py`(`_pack_weights_to_dir`)+ C++ `ge_weight_loader.cc`/`ge_graph_builder.cc:CreateConstNode`
- 相关:**CLI-015**(NZ 格式,只作用于 parameter)、**CLI-016**(权重与 const 分离,本条细化其粒度)

---

## CLI-020 TODO:torch.export 无 output name 参数,saver 需自动设置语义名

| 字段 | 值 |
|------|-----|
| 编号 | CLI-020 |
| 类型 | TODO(待实现) |
| 位置 | `cann_lm/converter/saver.py`(`save_epair`/`save_epair_prefill_decode`:`torch.export.export` 后未设 output name)+ `cann_lm/models/export_wrapper.py`(`make_input_output_names` 只用于 ONNX,未覆盖 delta_out) |
| 背景 | `torch.export.export(model, args, strict=False)` **无直接参数**设 output name。输出名从产生该输出的 FX 算子节点名派生(如 `matmul`/`sin`/`cos`/`index_select`),非语义名(如 `logits`/`past_key_0_out`)。返回 dict 时 key 会丢失(torch.export flatten 成 tuple,name 仍从算子名派生)。实测 torch 2.9.0+cpu:`OutputSpec.arg = TensorArgument(name='matmul')` — `name` 字段可后处理改写,改后 `ep.module()` 正常跑 |
| 违反约定 | 无(待实现) |
| 发现日期 | 2026-09-16 |
| 状态 | **TODO(方案已验证,待实现)** |

### 背景:为何需要语义 output name

epair 导出链路 `torch.export → torch_delegate.export_to_epair → ep_json`,ep_json 的 `signature.output_specs` 里每个 output 的 `arg.name` 沿用 FX 算子节点名(如 `matmul_3`/`index_select_1`),无语义。下游消费方(OM runtime/C++ 加载/host 调度)需按位置而非名字识别输出(logits/past_kv_out/delta_out),可读性差、易错位。

### 现状缺口

1. **`ExportWrapper.make_input_output_names()` 已有约定但只用于 ONNX**(`export_wrapper.py:229`):生成 `["logits", "past_key_{i}_out", "past_value_{i}_out"]`,epair 路径未调用
2. **delta_out 未覆盖**:基类 `make_input_output_names` 只生成 logits + past_kv_out(`2×num_layers`),qwen3.5 `forward` 还返回 `delta_out_flat`(每层 conv+rec,`2×num_layers` 个,`export_wrapper.py:135-140`),names 数量 < output 数量
3. **saver 未后处理**:`save_epair`/`save_epair_prefill_decode` 在 `torch.export.export()` 后直接 `torch_delegate.export_to_epair(ep, config)`,未改 `ep.graph_signature.output_specs[i].arg.name`

### 方案(已验证 torch 2.9 可行)

**第 1 步**:补全 `make_input_output_names()` 含 delta_out(基类或 qwen3.5 override):

```python
def make_input_output_names(self) -> tuple[list[str], list[str]]:
    n = self._num_layers()
    input_names = ["input_ids", "attention_mask", "position_ids", "new_kv_cache_pos"]
    for i in range(n):
        input_names.extend([f"past_key_{i}", f"past_value_{i}"])
    output_names = ["logits"]
    for i in range(n):
        output_names.extend([f"past_key_{i}_out", f"past_value_{i}_out"])
    if self._has_delta_out():  # 子类 override: qwen3.5 True, gemma4 False
        for i in range(n):
            output_names.extend([f"delta_conv_{i}_out", f"delta_rec_{i}_out"])
    return input_names, output_names
```

**第 2 步**:saver.py 在 `torch.export.export()` 后、`torch_delegate.export_to_epair()` 前后处理 output_specs:

```python
ep = torch.export.export(wrapper, sample_inputs, strict=False)
_, output_names = wrapper.make_input_output_names()
for spec, name in zip(ep.graph_signature.output_specs, output_names):
    spec.arg.name = name  # TensorArgument.name 可改
for node in ep.graph.nodes:  # 同步 graph output node name
    if node.op == "output":
        for n, name in zip(node.args[0], output_names):
            n.name = name
        break
with lite_export_context(lite_config):
    epair = torch_delegate.export_to_epair(ep, config)  # ep_json 沿用自定义 name
```

### 边界处理

- **delta_out 可选**:`forward` 返回 `delta_out_flat` 仅当 `delta_states is not None`(`export_wrapper.py:136`)。`save_epair` 的 sample_inputs 不传 delta_states → delta_out_flat 为空 → output 数量 = `1 + 2×num_layers`。`make_input_output_names` 多出的 delta names 被 `zip` 自动截断,不影响
- **zip 短板**:names 不足时 fallback 用算子名(避免 mismatch),需加 `if len(names) < len(specs)` 兜底
- **prefill/decode 共享**:`make_input_output_names` 不依赖 seq_len/prefill/decode,prefill/decode 用同一组 names,无 prefill/decode 差异问题(与 CLI-014 的权重共享无冲突)

### 行动项

- [ ] `ExportWrapper.make_input_output_names()` 补 delta_out names + 加 `_has_delta_out()` hook(qwen3.5 override True)
- [ ] `saver.py` 加 `_apply_output_names(ep, wrapper)` helper,在 `save_epair`/`save_epair_prefill_decode` 的 `torch.export` 后调用
- [ ] 验证 ep_json `signature.output_specs` 的 `arg.name` 为语义名(`logits`/`past_key_0_out`/...)
- [ ] 文档:export_design.md 补 output name 自动设置说明

### 关联

- 源码:`cann_lm/converter/saver.py`(`save_epair`/`save_epair_prefill_decode`)+ `cann_lm/models/export_wrapper.py`(`make_input_output_names`/`forward` 返回 logits+past_kv_flat+delta_out_flat)
- 相关:**CLI-014**(PD 权重共享,本条的 output name 不影响共享)、**CLI-016/CLI-019**(权重/const 分离,已部分实现 `_is_weight` 字段——见下注)
- 注:CLI-016/CLI-019 描述的「weight_info.json 加类别字段」已实现为 `_is_weight`(bool,二态:parameter+buffer=true / const=false),`epair_utils.py:_pack_weights_to_dir` 落盘该字段 + C++ `ge_weight_loader.cc:LoadConsolidated` 读取填 `TensorRecordInfo.is_parameter` + `ge_graph_builder.cc:CreateConstNode` 注入 GE Const 节点私有 attr `_is_weight`(dump 图可见,OM 被 GE OptimizeStage1 清理)。CLI-016/CLI-019 的 NZ 按 kind 分流(NZ 只对 parameter)尚未实现,仍为 TODO

