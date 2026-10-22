#!/usr/bin/env python3
"""cann_lm — top-level CLI (subcommand dispatcher, mlx_lm-style).

`cann_lm generate ...` is the primary form; the standalone `cann_lm.generate`
also works (both registered in pyproject). Future subcommands (chat, server, ...)
slot in as additional `sub.add_parser(...)` calls in build_parser().
"""
import argparse
import os
import sys

# Make `import cann_lm` resolve when run directly or via -m without pip.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cann_lm import generate as _generate  # the generate subcommand handler


def build_parser():
    ap = argparse.ArgumentParser(
        prog="cann_lm",
        description="CANN-LM CLI. Subcommands: generate (more to come).")
    sub = ap.add_subparsers(dest="command", required=True, metavar="<command>")

    # generate
    g = sub.add_parser("generate", help="generate text from a prompt")
    _generate.add_args(g)
    g.set_defaults(func=_generate.run)

    # future subcommands slot in here, e.g.:
    # chat = sub.add_parser("chat", help="interactive chat")
    # _chat.add_args(chat); chat.set_defaults(func=_chat.run)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
