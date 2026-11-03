# 仓库目录结构重组

简体中文 | [English](../../en/adr/0003-repo-layout-restructure.md)

项目目录混乱：Python 绑定散在 `cann_local_lm/python/`、docs 未与源码 context 对齐、tests 混 ut/st/e2e、vendored 源码与 find module 混放、`generate/` 预留占位冗余（真实 OHOS 构建已在 `scripts/cann_lm_engine/`）、根 `build.sh` 与 `scripts/build.sh` 双入口重复。决定按 engine/converter context 对齐源码与 docs；`tests/` 只放 ut、`examples/` 放 st/端到端；vendored 源码统一入 `cmake/third_party/`（与 find module 同根）；构建脚本统一 `scripts/`（根 `build.sh` 作 thin entry 保 `bash build.sh` 习惯）；删 `generate/` 占位与 `deploy_ohos/` tracked 产物。

## Considered Options

- 维持现状：目录与 context 不一致、入口重复、tests 边界模糊——否决。
- 一次性大重构 squash 提交：难 review、单点回滚粒度粗——否决。
- **分步提交**：按决策各一个 commit，便于 review 与单点回滚——采纳。

## Consequences

- `cann_local_lm/engine/CMakeLists.txt`（5 处）与 `cann_local_lm/engine/ohos/CMakeLists.txt`（5 处）的 `third_party` include 路径同步改为 `cmake/third_party/nlohmann`。
- 根 `build.sh` 的 `--pkg`/`-u` 逻辑迁入 `scripts/build.sh` 作 `pkg`/`ut` 子命令；docs/examples 内 11 处 `bash build.sh` 引用不变（thin entry 保习惯）。
- `scripts/cann_lm_engine/pyproject.toml` + `build_wheel.sh` 对 `cann_local_lm/python/` 的引用更新为 `cann_local_lm/engine/python/`。
- 新增 `GLOSSARY.md`（ut/st/e2e/4 套环境/vendored vs find module/context 边界术语）。

## 关联

- `docs/adr/0002-env-build-isolation.md`（4 套构建环境；③ generate 现由 `scripts/cann_lm_engine/build_ohos.sh` 承载）。
