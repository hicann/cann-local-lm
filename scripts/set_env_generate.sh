#!/bin/bash
# set_env_generate.sh — ③ generate env (鸿蒙交叉编译, 预留占位)
#
# 待以下信息确定后补全:
#   - 鸿蒙 PC 上运行的目标产物形态 (.so / .om / .hap)
#   - OHOS NDK / 交叉编译工具链安装路径
#   - 目标 ABI (aarch64-linux-ohos ?)
# 见 generate/cmake/toolchains/ohos-aarch64.cmake (注释化 stub) + docs/adr/0002
echo "[set_env] ③ generate (鸿蒙交叉编译) — 预留, 暂未实现"
echo "  generate/ + cmake/toolchains/ohos-aarch64.cmake 占位 stub 已建"
echo "  目标产物 (so/om/hap) 与 OHOS NDK 路径确定后补全此脚本"
return 0 2>/dev/null || exit 0
