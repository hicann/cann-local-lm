"""Tests for the cann_local_lm top-level CLI (subcommand dispatcher, mlx_lm-style).

Pure: only the argparse dispatch is exercised (parse `cann_local_lm generate ...` →
command=generate, args.func is generate.run, args wired). The actual generate
run needs the .so (device).
"""

from cann_local_lm import generate
from cann_local_lm import cli


def _parse(*argv):
    return cli.build_parser().parse_args(list(argv))


class TestCliDispatch:
    def test_generate_subcommand_routes_to_generate_run(self):
        args = _parse("generate", "--config", "tc.json", "--prompt", "hi")
        assert args.command == "generate"
        assert args.func is generate.run  # dispatcher → generate.run
        assert args.config == "tc.json"
        assert args.prompt == "hi"

    def test_generate_subcommand_args_match_add_args(self):
        # the subcommand accepts the same flags generate.add_args registers
        args = _parse("generate", "--config", "x", "--max-tokens", "5", "--stop", "END")
        assert args.max_tokens == 5
        assert args.stop_strings == ["END"]

    def test_no_subcommand_errors(self):
        import pytest

        with pytest.raises(SystemExit):
            _parse()  # subcommand required

    def test_unknown_subcommand_errors(self):
        import pytest

        with pytest.raises(SystemExit):
            _parse("bogus", "--config", "x")
