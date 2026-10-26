# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
测试 torch_delegate 日志系统
"""

import os
import logging
from io import StringIO

from torch_delegate.core.utils import logger, _get_logger, _MillisecFormatter


class TestLoggerLevels:
    """测试日志级别"""

    def test_default_level_is_warn(self):
        """默认日志级别为 WARN"""
        assert logger.level == logging.WARN

    def test_debug_level_explicit(self):
        """显式传入 DEBUG 级别时生效"""
        debug_logger = _get_logger(level=logging.DEBUG, name="test_debug")
        assert debug_logger.level == logging.DEBUG

    def test_warn_prints_by_default(self):
        """WARN 级别日志默认打印"""
        output = StringIO()
        test_logger = _get_logger(level=logging.WARN, output=output, name="test_warn")
        test_logger.warning("test warning message")
        assert "test warning message" in output.getvalue()

    def test_info_not_printed_by_default(self):
        """INFO 级别默认不打印（WARN 为默认级别）"""
        output = StringIO()
        test_logger = _get_logger(level=logging.WARN, output=output, name="test_info")
        test_logger.info("test info message")
        assert output.getvalue() == ""

    def test_debug_printed_when_enabled(self):
        """DEBUG 级别在启用时打印"""
        output = StringIO()
        test_logger = _get_logger(
            level=logging.DEBUG, output=output, name="test_debug_print"
        )
        test_logger.debug("test debug message")
        assert "test debug message" in output.getvalue()


class TestWarningOnce:
    """测试 warning_once 功能"""

    def test_warning_once_prints_single_time(self):
        """warning_once 只打印一次"""
        output = StringIO()
        test_logger = _get_logger(level=logging.WARN, output=output, name="test_once")

        test_logger.warning_once("warning once")
        test_logger.warning_once("warning once")
        test_logger.warning_once("warning once")

        log_content = output.getvalue()
        assert log_content.count("warning once") == 1

    def test_warning_once_different_messages(self):
        """不同消息各打印一次"""
        output = StringIO()
        test_logger = _get_logger(
            level=logging.WARN, output=output, name="test_diff_once"
        )

        test_logger.warning_once("message A")
        test_logger.warning_once("message B")
        test_logger.warning_once("message A")

        log_content = output.getvalue()
        assert log_content.count("message A") == 1
        assert log_content.count("message B") == 1


class TestLoggerFormat:
    """测试日志格式"""

    def test_format_contains_pid(self):
        """日志格式包含进程 ID"""
        output = StringIO()
        test_logger = _get_logger(level=logging.WARN, output=output, name="test_format")
        test_logger.warning("format test")

        log_content = output.getvalue()
        assert str(os.getpid()) in log_content

    def test_format_contains_level(self):
        """日志格式包含级别"""
        output = StringIO()
        test_logger = _get_logger(level=logging.WARN, output=output, name="test_level")
        test_logger.warning("level test")

        log_content = output.getvalue()
        assert "[WARNING]" in log_content

    def test_format_contains_file_lineno(self):
        """日志格式包含文件名和行号"""
        output = StringIO()
        test_logger = _get_logger(level=logging.WARN, output=output, name="test_file")
        test_logger.warning("file test")

        log_content = output.getvalue()
        assert "[" in log_content

    def test_format_contains_timestamp(self):
        """日志格式包含时间戳"""
        output = StringIO()
        test_logger = _get_logger(level=logging.WARN, output=output, name="test_time")
        test_logger.warning("time test")

        log_content = output.getvalue()
        assert (
            "2026-" in log_content or "2025-" in log_content or "2024-" in log_content
        )


class TestMillisecFormatter:
    """测试毫秒格式化器"""

    def test_formatter_has_milliseconds(self):
        """时间戳包含毫秒"""
        formatter = _MillisecFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        formatted_time = formatter.formatTime(record)

        assert "." in formatted_time
        assert len(formatted_time.split(".")[-1]) == 3
