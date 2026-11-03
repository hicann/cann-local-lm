"""Pytest path bootstrap + host .so/stub env for Tier B tests.

(1) Make `import cann_local_lm` resolve to the bindings package without requiring
    `pip install -e .` (keeps the test run dependency-free). 绑定源在
    cann_local_lm/engine/python/cann_local_lm/ (engine 目录重构后移入, conftest 旧路径
    cann_local_lm/python 已废)。
(2) On host, auto-point CANN_LM_LIB at the host-built libcann_lm_engine.so and
    CANN_LM_NDK_LIB_DIR at the build dir (stub .so for OH_NN*) so Tier B tests
    needing a loadable .so + NDK backend run without manual env. 只在未设且 .so
    实际存在时注入(允许外部 override)。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PKG_ROOT = os.path.normpath(os.path.join(_REPO, "cann_local_lm", "engine", "python"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# examples/ 上 sys.path: test_bbit/test_generate `from bbit import ...`(bbit.py 在
# examples/, 是 bbit.cpp 的 python 镜像,纯 helper 在此测)。
_EXAMPLES = os.path.join(_REPO, "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

# host SHARED .so + stub_ndk for Tier B: probe candidate host build dirs.
for _bdir in ("build_host2", "build_host"):
    _so = os.path.join(_REPO, _bdir, "libcann_lm_engine.so")
    if os.path.isfile(_so):
        os.environ.setdefault("CANN_LM_LIB", _so)
        os.environ.setdefault("CANN_LM_NDK_LIB_DIR", os.path.join(_REPO, _bdir))
        break

# Tokenizer fixture 自动生成(combined 1+2):bpe_fixture_tokenizer.json 是 gitignored
# 生成物(.gitignore: fixtures/),由 tests/cpp/engine/gen_fixture.py
# 用 HF `tokenizers` 库生成。有该库的环境下,pytest 开跑前在此自动生成 fixture,
# test_token_count.TestBpeExactAgainstGolden 即真跑;无该库(裸 CI/本机)时不生成,
# 那批测试自带 skip-guard 跳过(见该文件)。本块绝不抛错——gen 失败仅意味着依赖该
# fixture 的测试会 skip,不影响采集。
_FIX_DIR = os.path.join(_REPO, "tests", "cpp", "engine")
_BPE_FIX = os.path.join(_FIX_DIR, "bpe_fixture_tokenizer.json")
if not os.path.isfile(_BPE_FIX):
    try:
        import tokenizers  # noqa: F401  探测库在不在;gen 脚本本身 import 它
    except ImportError:
        pass
    else:
        import subprocess

        _GEN = os.path.join(_REPO, "tests", "cpp", "engine", "gen_fixture.py")
        if os.path.isfile(_GEN):
            try:
                subprocess.run(
                    [sys.executable, _GEN],
                    check=True,
                    cwd=_REPO,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
