# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
__all__ = ["logger"]

from functools import lru_cache
import logging
import sys
import os
import threading
from datetime import datetime


class _MillisecFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = datetime.fromtimestamp(record.created)
        s = ct.strftime("%Y-%m-%d %H:%M:%S")
        s += ".{:03d}".format(ct.microsecond // 1000)
        return s


def _get_logger(*, level=logging.ERROR, output=sys.stdout, name=None):
    _logger = logging.getLogger(name)
    _logger.handlers.clear()
    _logger.setLevel(level)

    formatter = _MillisecFormatter(
        f"[%(levelname)s] TORCH_DELEGATE({os.getpid()},{os.path.basename(sys.executable)})"
        f" %(asctime)s [%(filename)s:%(lineno)d] {threading.get_native_id()} %(message)s"
    )

    if output:
        console_handler = logging.StreamHandler(output)
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

    @lru_cache
    def _warning_once(msg):
        _logger.warning(msg)

    _logger.warning_once = _warning_once

    return _logger


logger = _get_logger(
    name="torch_delegate",
    level=logging.DEBUG if os.getenv("TORCH_DELEGATE_DEBUG") == "1" else logging.WARN,
)
