# generate/ — 鸿蒙交叉编译 (预留)

本目录承载 **③ generate 环境**：面向鸿蒙 PC 的交叉编译产物构建。

## 当前状态：预留占位

代码与构建**暂未实现**。仅建立目录结构与工具链占位 stub，待以下信息确定后补全：

- 鸿蒙 PC 上运行的目标产物形态（`.so` / `.om` / `.hap`）
- OHOS NDK / 交叉编译工具链安装路径
- 目标 ABI（`aarch64-linux-ohos` ?）
- 与 ② compile 的产物复用关系（是否复用 `libtorch_delegate_backend.so` / epair）

## 占位内容

- `cmake/toolchains/ohos-aarch64.cmake` — OHOS 交叉编译 toolchain 文件 **注释化模板**（不可用，待补全）
- `scripts/set_env_generate.sh` — 直 source，打印"预留未实现"

## 与三套环境的关系

| 环境 | 职责 | 状态 |
|------|------|------|
| ① converter | HF→epair + 量化 (Linux CPU venv) | 已就绪 |
| ② compile | epair→OM (C++ 编译 + OM 编译, 真 CANN) | 已就绪 |
| ③ generate | 鸿蒙交叉编译 | **预留** |
| ④ npu | verify-runtime (dopt GPTQ / NPU 出字 / OM 验证) | 已就绪 |

详见 `docs/adr/0002-env-build-isolation.md`。
