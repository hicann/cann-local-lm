# 贡献指南

简体中文 | [English](CONTRIBUTING_en.md)

感谢您对 CANN-LM 项目的关注！欢迎通过以下方式参与贡献。

## 如何贡献

### 提交 Issue

- 使用 GitCode [Issues](https://gitcode.com/cann/cann-lm/issues) 提交问题
- 请描述清楚问题现象、复现步骤、环境信息（CANN 版本 / torch 版本 / 芯片型号）
- Bug 报告请附上错误日志和最小复现用例

### 提交 Pull Request

1. **Fork 本仓**并拉取到本地
2. **创建分支**：`git checkout -b feat/your-feature` 或 `fix/your-bugfix`
3. **修改代码**，遵循以下编码规范：
   - Python `>= 3.10`，使用内置泛型语法（`dict[str, Any]` 而非 `Dict[str, Any]`）
   - 单函数控制在 50 行以内，参数不超过 5 个
   - 不写过渡性注释，复杂设计写入 `docs/`
   - C++ 遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)，`-Wall -Wextra -Werror` 零警告
4. **跑通单元测试**（提交前强制）：
   ```bash
   # Python 测试 (converter venv + TORCH_DEVICE_BACKEND_AUTOLOAD=0)
   TORCH_DEVICE_BACKEND_AUTOLOAD=0 <venv_python> -m pytest tests/python/llm_models/ tests/python/converter/ tests/python/dopt_lm/ -q

   # C++ 测试 (真 CANN env, ASAN)
   bash scripts/build.sh cpp --asan && ctest --test-dir build --output-on-failure
   ```
5. **提交 PR**，在描述中说明改动范围和验证结果

## 开发规范

### cann_lm/ 代码修改

- **bug 修复必须给推导与修改逻辑**：提供根因证据链（代码行号 / spec 文档 / 测试输出 / NPU 实测），禁止凭印象猜测
- **测试时禁改 cann_lm/**：跑测试用例时不要修改 `cann_lm/` 下任何代码，优先改测试代码，无法通过再确认改主仓
- **环境前置**：任何依赖 torchair 的脚本运行前须先 `source scripts/set_env.sh`，C++ 编译须 `source scripts/set_env_cann.sh`

### 测试代码通用性

- 测试代码不硬编码绝对路径，通过 CLI 参数传入（`--hf-path` / `--output-dir` / `--soc-version` 等）
- 端到端测试用例及产物统一放置于 `examples/` 路径下，按用例名分目录组织

### 端到端测试

- 所有端到端测试用例及生成产物（模型文件、权重、OM 等）统一放置于 `examples/`
- 用例环境以文件夹为单位统一环境前置条件，须与 `scripts/` 下环境脚本相匹配

## 文档更新

修改代码后须同步更新对应文档：

| 修改范围 | 须更新文档 |
|----------|-----------|
| `cann_lm/converter/` | `docs/cann_lm_converter/` 下对应设计文档 |
| `cann_lm/models/fused_ops/` | `docs/zh/cann_lm_converter/models/fused_ops/fused_ops.md` |
| `cann_lm/converter/export/` | `docs/zh/cann_lm_converter/export/export_design.md` |
| `cann_lm/converter/quant/` | `docs/zh/cann_lm_converter/quant/quant_design.md` |
| CLI 命令行参数 | `docs/zh/cann_lm_converter/cli/cli_design.md` |

## 版本配套

请选择与 CANN 版本配套的分支/标签源码，使用 master 分支可能存在版本不匹配风险。详见 [release 仓库](https://gitcode.com/cann/release-management)。
