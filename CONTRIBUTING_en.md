# Contributing Guide

English | [简体中文](CONTRIBUTING.md)

Thank you for your interest in the CANN-LM project! You are welcome to contribute in the following ways.

## How to Contribute

### Submitting an Issue

- Use GitCode [Issues](https://gitcode.com/cann/cann-lm/issues) to report problems
- Clearly describe the problem symptoms, reproduction steps, and environment information (CANN version / torch version / chip model)
- Bug reports should include error logs and a minimal reproduction case

### Submitting a Pull Request

1. **Fork this repository** and clone it locally
2. **Create a branch**: `git checkout -b feat/your-feature` or `fix/your-bugfix`
3. **Modify the code**, following these coding conventions:
   - Python `>= 3.10`, use built-in generic syntax (`dict[str, Any]` instead of `Dict[str, Any]`)
   - Keep single functions within 50 lines, with no more than 5 parameters
   - Avoid transitional comments; write complex designs into `docs/`
   - C++ follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), zero warnings under `-Wall -Wextra -Werror`
4. **Pass unit tests** (mandatory before submission):
   ```bash
   # Python tests (converter venv + TORCH_DEVICE_BACKEND_AUTOLOAD=0)
   TORCH_DEVICE_BACKEND_AUTOLOAD=0 <venv_python> -m pytest tests/python/llm_models/ tests/python/converter/ tests/python/dopt_lm/ -q

   # C++ tests (real CANN env, ASAN)
   bash scripts/build.sh cpp --asan && ctest --test-dir build --output-on-failure
   ```
5. **Submit the PR**, explaining the scope of changes and verification results in the description

## Development Conventions

### cann_lm/ Code Modifications

- **Bug fixes must provide reasoning and modification logic**: provide a root cause evidence chain (code line numbers / spec docs / test output / NPU actual measurement), guessing from impression is prohibited
- **Do not modify cann_lm/ during testing**: do not modify any code under `cann_lm/` when running test cases; prioritize modifying test code, and only confirm modifying the main repository if tests cannot pass
- **Environment prerequisites**: before running any torchair-dependent script, you must first `source scripts/set_env.sh`; C++ compilation requires `source scripts/set_env_cann.sh`

### Test Code Generality

- Test code should not hardcode absolute paths; pass them via CLI parameters (`--hf-path` / `--output-dir` / `--soc-version`, etc.)
- End-to-end test cases and artifacts should be placed uniformly under the `examples/` path, organized in directories by case name

### End-to-End Testing

- All end-to-end test cases and generated artifacts (model files, weights, OM, etc.) should be placed uniformly under `examples/`
- Case environments should unify environment prerequisites per folder, and must match the environment scripts under `scripts/`

## Documentation Updates

After modifying code, you must synchronously update the corresponding documentation:

| Modification Scope | Documentation to Update |
|----------|-----------|
| `cann_lm/converter/` | Corresponding design docs under `docs/cann_lm_converter/` |
| `cann_lm/models/fused_ops/` | `docs/zh/cann_lm_converter/models/fused_ops/fused_ops.md` |
| `cann_lm/converter/export/` | `docs/zh/cann_lm_converter/export/export_design.md` |
| `cann_lm/converter/quant/` | `docs/zh/cann_lm_converter/quant/quant_design.md` |
| CLI command-line parameters | `docs/zh/cann_lm_converter/cli/cli_design.md` |

## Version Compatibility

Please select the branch/tag source code that matches your CANN version; using the master branch may carry version mismatch risks. See the [release repository](https://gitcode.com/cann/release-management) for details.
