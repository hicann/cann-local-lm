# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""torch-delegate-config 命令行入口。

仅提供构建期信息：C++ 制品前缀与包版本。运行期库路径不由本工具注入，
需由使用方自行配置 LD_LIBRARY_PATH（见 docs/zh/guide/build.md）。
"""

import argparse
import os
import sys

from ._version import __version__

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torch-delegate-config",
        description="Print TorchDelegate build-time information.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cmake-prefix",
        action="store_true",
        help="print the C++ prefix (put it into CMAKE_PREFIX_PATH for find_package(TorchDelegate))",
    )
    group.add_argument(
        "--version",
        action="store_true",
        help="print the torch-delegate version",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    print(PACKAGE_DIR if args.cmake_prefix else __version__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
