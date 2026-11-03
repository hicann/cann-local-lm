"""cann_lm_converter.py CLI argparse 单元测试 — 参数解析 (不跑真导出).

测 parse_args() 的 argparse 行为:
- --help 退出码 0 + 含关键参数名
- 缺 --hf-path (required) → SystemExit (code 2)
- --mode 非法 choices → SystemExit
- --quant-path / --quant-method 互斥 → SystemExit
- 合法参数 → Namespace 字段 + 默认值
跳过: convert() 全流程 (需真实 HF 模型 + torchair GE 转换).
"""

import sys

import pytest

from cann_local_lm.converter.cann_lm_converter import parse_args


def _run_parse(argv):
    """临时替换 sys.argv 调 parse_args (argparse 读 sys.argv)."""
    old = sys.argv
    sys.argv = ["cllm-converter"] + argv
    try:
        return parse_args()
    finally:
        sys.argv = old


def _expect_exit(argv, code=2):
    old = sys.argv
    sys.argv = ["cllm-converter"] + argv
    try:
        parse_args()
    except SystemExit as e:
        assert e.code == code, f"expected exit {code}, got {e.code}"
        return e
    finally:
        sys.argv = old
    pytest.fail("expected SystemExit, none raised")


# ── --help ──


def test_help_exits_zero_and_lists_key_args(capsys):
    with pytest.raises(SystemExit) as exc:
        _run_parse(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for key in ["--hf-path", "--output-dir", "--target-soc", "--mode", "--num-layers"]:
        assert key in out


# ── required / choices / 互斥 校验 ──


def test_missing_required_hf_path_exits_nonzero(capsys):
    _expect_exit([])
    err = capsys.readouterr().err
    assert "hf-path" in err


def test_missing_required_output_dir_exits_nonzero(capsys):
    _expect_exit(["--hf-path", "/x"])
    assert "output-dir" in capsys.readouterr().err


def test_invalid_dtype_exits_nonzero(capsys):
    # --dtype 已删除 (commit 73fdb01), 改测 --target-soc 任意字符串均接受 (dtype 由 _derive_dtype 派生)
    args = _run_parse(
        ["--hf-path", "/x", "--output-dir", "/y", "--target-soc", "Ascend910B"]
    )
    assert args.target_soc == "Ascend910B"


def test_invalid_mode_exits_nonzero(capsys):
    _expect_exit(["--hf-path", "/x", "--output-dir", "/y", "--mode", "fast"])
    assert "invalid choice" in capsys.readouterr().err


def test_quant_path_and_quant_method_mutually_exclusive(capsys):
    _expect_exit(
        [
            "--hf-path",
            "/x",
            "--output-dir",
            "/y",
            "--quant-path",
            "/q",
            "--quant-method",
            "gptq",
        ]
    )
    assert "not allowed" in capsys.readouterr().err


def test_invalid_quant_method_exits_nonzero(capsys):
    _expect_exit(["--hf-path", "/x", "--output-dir", "/y", "--quant-method", "awq"])
    assert "invalid choice" in capsys.readouterr().err


# ── 合法参数 → Namespace ──


def test_valid_minimal_args_namespace_and_defaults():
    args = _run_parse(["--hf-path", "/data/model", "--output-dir", "/out"])
    assert args.hf_path == "/data/model"
    assert args.output_dir == "/out"
    assert (
        args.target_soc == "KirinX90"
    )  # 默认 (commit 73fdb01: dtype 由 --target-soc 派生)
    assert args.mode == "prefill-decode"  # 默认
    assert args.use_embeddings is True  # 默认
    assert args.num_layers is None
    assert args.quant_path is None
    assert args.quant_method is None


def test_target_soc_overrides_default():
    args = _run_parse(
        ["--hf-path", "/x", "--output-dir", "/y", "--target-soc", "Ascend910B"]
    )
    assert args.target_soc == "Ascend910B"


def test_no_use_embeddings_flag():
    args = _run_parse(["--hf-path", "/x", "--output-dir", "/y", "--no-use-embeddings"])
    assert args.use_embeddings is False


def test_num_layers_parsed_as_int():
    args = _run_parse(["--hf-path", "/x", "--output-dir", "/y", "--num-layers", "4"])
    assert args.num_layers == 4


def test_single_mode_choice():
    args = _run_parse(["--hf-path", "/x", "--output-dir", "/y", "--mode", "single"])
    assert args.mode == "single"


def test_quant_path_alone_accepted():
    args = _run_parse(["--hf-path", "/x", "--output-dir", "/y", "--quant-path", "/q"])
    assert args.quant_path == "/q"
    assert args.quant_method is None


def test_quant_method_gptq_alone_accepted():
    args = _run_parse(
        ["--hf-path", "/x", "--output-dir", "/y", "--quant-method", "gptq"]
    )
    assert args.quant_method == "gptq"
    assert args.quant_path is None
