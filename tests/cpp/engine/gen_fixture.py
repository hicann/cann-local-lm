#!/usr/bin/env python3
"""Tokenizer 测试资产生成（SR20260728001 v0.7，M-8 A+B 分层）。

A 层：小型 fixture tokenizer.json（Qwen 风格 / gemma 风格）+ Python oracle
（tokenizers 库）预生成的 golden ids，入库供离线 UT 使用（运行时零 Python 依赖）。
B 层：tmp/qwen、tmp/gemma 真实产物的 golden ids（产物本体 gitignore，不入库）。

用法：
    python3 tests/cpp/engine/gen_fixture.py          # 生成全部
    python3 tests/cpp/engine/gen_fixture.py --real   # 仅重新生成 B 层 golden

注意：A 层 Qwen fixture 的 Split 正则携带真实 Qwen 正则（\\p{L} 等属性类），
golden 由 Python oracle 按其计算；C++ 侧使用 D-10 正则（POSIX 类）——
golden 语料已避开豁免类输入（Latin↔CJK 直接相邻混排、ü 开头词、大写缩写、
非 ASCII 空白），两侧切分一致（见 DESIGN D-10 §4 豁免登记）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

# __file__ = <repo>/tests/cpp/engine/gen_fixture.py → 仓库根在
# parents[3] (engine → cpp → tests → repo root)
REPO = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO / "tests" / "cpp" / "engine"
TMP_DIR = REPO / "tmp"

# 真实 Qwen 预切分正则（tokenization_qwen3_5.py:25 / tokenizer.json pre_tokenizer）
QWEN_PRETOKENIZE_REGEX = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| """
    r"""?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

# A 层 golden 语料：避开 D-10 豁免类（混排直接相邻/ü 开头/大写缩写/非 ASCII 空白）
QWEN_GOLDEN_TEXTS = [
    "Hello world",
    "你好世界",
    "it's a test",
    "hello<|endoftext|>world",
    "  leading and trailing spaces   ",
    "tabs\tand\nnewlines",
    "数字 12345 混合",
    "punct!@#$%^&*()",
    "aü ok",
    "mixed 你好 english",
    "<tool_call>run<|im_end|>",
    "",
]

BPE_GOLDEN_TEXTS = [
    "hello world",
    "你好",
    "Hello, Gemma!",
    "<start_of_image>caption",
    "  double  spaces  ",
    "byte fallback: é 😀",
    "The quick brown fox",
    "",
]


def build_qwen_fixture() -> Tokenizer:
    """Qwen 风格 fixture：BPE + Split(真实 Qwen 正则) + ByteLevel。"""
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = None  # NFC 跳过（U-4：C++ 侧不实现归一化，oracle 同口径）
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(QWEN_PRETOKENIZE_REGEX), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "Hello world and Welcome to the Tokenizer Test Suite",
        "hello world and welcome to the tokenizer test suite",
        "你好世界 这是一次分词测试",
        "it's a test with punctuation and numbers 12345",
        "aüb ok mixed content with tabs\tand spaces",
        "leading and trailing spaces   plus newlines\n\n",
        "punct!@#$%^&*() and CAPS LOCK TEXT",
        "mixed 你好 english words 数字 12345",
        "hello<|endoftext|>world <tool_call>run<|im_end|>",
    ]
    # 初始字母表 = ByteLevel 全 256 映射字符（保证任意输入不缺字）
    initial_alphabet = []
    n = 0
    for b in range(256):
        if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255):
            initial_alphabet.append(chr(b))
        else:
            initial_alphabet.append(chr(256 + n))
            n += 1
    trainer = BpeTrainer(
        vocab_size=420,
        show_progress=False,
        special_tokens=[],
        initial_alphabet=initial_alphabet,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.add_special_tokens(["<|endoftext|>"])
    tok.add_tokens(["<tool_call>"])
    return tok


def build_bpe_fixture() -> Tokenizer:
    """gemma 风格 fixture：BPE(byte_fallback) + Replace(" "→▁) + Split + Sequence 解码。"""
    # 手工词表：控制 token + 字母/数字 + <0xXX> 全集 + 若干合并词
    vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    next_id = 4
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789▁,.!?-':;()"
    for c in chars:
        vocab[c] = next_id
        next_id += 1
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = next_id
        next_id += 1
    # 合并链（(A,B) 对；中间产物与最终词均须入 vocab 供查表）
    merge_pairs = [
        ("h", "e"),
        ("he", "l"),
        ("hel", "l"),
        ("hell", "o"),  # → hello
        ("▁", "w"),
        ("▁w", "o"),
        ("▁wo", "r"),
        ("▁wor", "l"),
        ("▁worl", "d"),  # → ▁world
        ("▁", "t"),
        ("▁t", "e"),
        ("▁te", "s"),
        ("▁tes", "t"),  # → ▁test
        ("o", "k"),
        ("▁", "o"),
        ("▁o", "k"),  # → ok / ▁ok
    ]
    for a, b in merge_pairs:
        vocab.setdefault(a + b, next_id)
        next_id += 1
    tok = Tokenizer(
        models.BPE(
            vocab=vocab, merges=merge_pairs, unk_token="<unk>", byte_fallback=True
        )
    )
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.pre_tokenizer = pre_tokenizers.Split(" ", behavior="merged_with_previous")
    tok.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
    )
    tok.add_special_tokens(["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_image>"])
    return tok


def dump_golden(tok: Tokenizer, texts: list[str]) -> dict:
    out = {"vocab_size": tok.get_vocab_size(), "cases": []}
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(ids, skip_special_tokens=True)
        out["cases"].append({"text": text, "ids": ids, "decoded": decoded})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="仅重新生成 B 层 golden")
    args = ap.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.real:
        qwen = build_qwen_fixture()
        qwen.save(str(FIXTURE_DIR / "qwen_fixture_tokenizer.json"))
        (FIXTURE_DIR / "qwen_fixture_golden.json").write_text(
            json.dumps(
                dump_golden(qwen, QWEN_GOLDEN_TEXTS), ensure_ascii=False, indent=1
            ),
            encoding="utf-8",
        )
        print(f"[A] qwen fixture + golden -> {FIXTURE_DIR}")

        bpe = build_bpe_fixture()
        bpe.save(str(FIXTURE_DIR / "bpe_fixture_tokenizer.json"))
        (FIXTURE_DIR / "bpe_fixture_golden.json").write_text(
            json.dumps(
                dump_golden(bpe, BPE_GOLDEN_TEXTS), ensure_ascii=False, indent=1
            ),
            encoding="utf-8",
        )
        print(f"[A] bpe fixture + golden -> {FIXTURE_DIR}")

    # B 层：真实产物 golden（语料避开 D-10 豁免类）
    real_texts = [
        "Hello world",
        "你好世界",
        "it's a test",
        "The quick brown fox jumps over the lazy dog",
        "中文 与 English 混合 的句子",
        "数字与标点：123, 456.789!",
        "hello <|im_end|> world",
        "😀 emoji 表情",
    ]
    for name in ("qwen", "gemma"):
        artifact = TMP_DIR / name / "tokenizer.json"
        if not artifact.exists():
            print(f"[B] {artifact} 缺失，跳过")
            continue
        tok = Tokenizer.from_file(str(artifact))
        (FIXTURE_DIR / f"{name}_real_golden.json").write_text(
            json.dumps(dump_golden(tok, real_texts), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(
            f"[B] {name} real golden (vocab_size={tok.get_vocab_size()}) -> {FIXTURE_DIR}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
