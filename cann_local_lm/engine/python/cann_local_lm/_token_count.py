"""Lightweight stdlib-only token counter for the chat budget guard.

Reads an HF-fast `tokenizer.json` (pure JSON) and runs a minimal BPE encode for
the SentencePiece-style byte-fallback BPE layout (normalizer Replace(" "->"▁")
+ Split(" ", MergedWithPrevious) pre-tokenizer + BPE merges + byte fallback +
added-token pre-split) — the Gemma fixture path. Counts match the engine's
tokenizer EXACTLY for that layout (verified vs tests/cpp/engine/
bpe_fixture_golden.json).

For other layouts (Qwen/GPT-4 ByteLevel BPE with a \\p{L}\\p{N}\\p{M} regex
pre-tokenizer; SentencePiece .model protobuf; anything unparseable) the counter
falls back to a conservative len(text) estimate (over-counts ASCII ~4x, ~exact
for CJK) — safe for a budget guard (never under-counts CJK), at the cost of
dropping history earlier than strictly needed. Exact Qwen counting would need
the `regex` package (\\p{...} unicode property classes) which the no-new-imports
constraint excludes, or a C-ABI tokenize call (follow-up).

No third-party imports — pure stdlib (json + os). Cached per path.
"""

import json
import os

__all__ = ["count_tokens", "encode", "TokenizeUnsupported"]


class TokenizeUnsupported(ValueError):
    """Raised by encode() when the tokenizer layout is not the supported
    SentencePiece-style byte-fallback BPE (Qwen/ByteLevel/SPM .model/...).
    count_tokens() catches this and falls back to len(text)."""


_cache = {}  # tokenizer_path -> layout dict or None (unsupported / unloadable)


def _load(path):
    lay = _cache.get(path, "MISS")
    if lay != "MISS":
        return lay
    lay = _parse(path)
    _cache[path] = lay
    return lay


def _parse(path):
    """Read tokenizer.json, validate the supported layout, return a layout dict
    or None (unsupported / unloadable). Supported = BPE model + Replace(' '->'▁')
    normalizer + Split(String=' ', ...) pre-tokenizer (or null/Sequence thereof)."""
    try:
        with open(path, "rb") as f:
            j = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(j, dict):
        return None
    model = j.get("model") or {}
    if model.get("type") != "BPE":
        return None
    # normalizer: only Replace(String=' ', content='▁') is exactly supported
    norm = j.get("normalizer")
    if not (
        isinstance(norm, dict)
        and norm.get("type") == "Replace"
        and (norm.get("pattern") or {}).get("String") == " "
        and norm.get("content") == "▁"
    ):
        return None
    # pre_tokenizer: only Split(String=' ', ...) / null / Sequence thereof
    if not _is_simple_space_split(j.get("pre_tokenizer")):
        return None
    vocab = model.get("vocab") or {}
    merges = model.get("merges") or []
    rank = {}
    for i, m in enumerate(merges):
        if isinstance(m, list) and len(m) == 2:
            rank[(m[0], m[1])] = i
        elif isinstance(m, str) and " " in m:
            a, b = m.split(" ", 1)
            rank[(a, b)] = i
    added_items = []
    for t in j.get("added_tokens") or []:
        if isinstance(t, dict) and t.get("content") and isinstance(t.get("id"), int):
            added_items.append((t["content"], t["id"]))
    # longest-first so '<start_of_image>' wins over '<' in added-token matching
    added_items.sort(key=lambda ci: len(ci[0]), reverse=True)
    return {
        "vocab": vocab,
        "rank": rank,
        "byte_fallback": bool(model.get("byte_fallback")),
        "added_items": added_items,
    }


def _is_simple_space_split(pretok):
    if pretok is None:
        return True
    if not isinstance(pretok, dict):
        return False
    if pretok.get("type") == "Split":
        return (pretok.get("pattern") or {}).get("String") == " "
    if pretok.get("type") == "Sequence":
        return all(
            _is_simple_space_split(s) for s in (pretok.get("pretokenizers") or [])
        )
    return False


def _split_added(text, added_items):
    """Split text into [(segment, added_id_or_None), ...]. Added tokens
    (longest-first, left-to-right, non-overlapping) become single segments with
    their id; gaps between are None-id segments to be normalized+BPE'd. Matches
    on the raw (pre-normalization) text — correct for special added tokens."""
    if not added_items or not text:
        return [(text, None)]
    segments = []
    i = 0
    gap_start = 0
    n = len(text)
    while i < n:
        match = None
        for content, tid in added_items:
            if content and text.startswith(content, i):
                match = (content, tid)
                break
        if match:
            content, tid = match
            if i > gap_start:
                segments.append((text[gap_start:i], None))
            segments.append((content, tid))
            i += len(content)
            gap_start = i
        else:
            i += 1
    if gap_start < n:
        segments.append((text[gap_start:], None))
    return segments


def _bpe(piece, rank):
    """Standard BPE: repeatedly merge the lowest-rank adjacent pair until none
    applies. Returns the final list of subword symbols (strings)."""
    syms = list(piece)
    while len(syms) > 1:
        best_i, best_r = None, None
        for i in range(len(syms) - 1):
            r = rank.get((syms[i], syms[i + 1]))
            if r is not None and (best_r is None or r < best_r):
                best_i, best_r = i, r
        if best_i is None:
            break
        # merge ALL non-overlapping occurrences of the chosen pair in one pass
        a, b = syms[best_i], syms[best_i + 1]
        merged = a + b
        new = []
        i = 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                new.append(merged)
                i += 2
            else:
                new.append(syms[i])
                i += 1
        syms = new
    return syms


def _byte_tokens(ch, vocab):
    """byte_fallback: one CJK/unknown char -> its UTF-8 bytes as <0xHH> tokens."""
    out = []
    for b in ch.encode("utf-8"):
        tok = "<0x%02X>" % b
        tid = vocab.get(tok)
        if tid is None:
            raise TokenizeUnsupported(
                f"byte_fallback vocab missing {tok!r} for char {ch!r}"
            )
        out.append(tid)
    return out


def encode(text, path):
    """Encode text -> list of token ids, EXACT for the supported layout.
    Raises TokenizeUnsupported for unsupported layouts (Qwen/ByteLevel/SPM .model/
    unparseable). count_tokens() catches this and falls back to len(text)."""
    lay = _load(path)
    if lay is None:
        raise TokenizeUnsupported(path)
    vocab = lay["vocab"]
    rank = lay["rank"]
    bf = lay["byte_fallback"]
    added_items = lay["added_items"]
    ids = []
    for seg, tid in _split_added(text, added_items):
        if tid is not None:
            ids.append(tid)
            continue
        # gap segment: normalize Replace(' '->'▁'); then Split(' ', MergedWithPrev)
        # is a no-op (no spaces remain post-normalization) -> one BPE piece.
        normed = seg.replace(" ", "▁")
        if not normed:
            continue
        for sym in _bpe(normed, rank):
            if sym in vocab:
                ids.append(vocab[sym])
            elif bf:
                for ch in sym:
                    ids.extend(_byte_tokens(ch, vocab))
            else:
                # no byte fallback + unk subword — unsupported for exact counting
                raise TokenizeUnsupported(
                    f"unk subword {sym!r} and byte_fallback is off"
                )
    return ids


def count_tokens(text, path):
    """Token count for the chat budget guard. Exact for the supported
    SentencePiece-BPE layout; conservative len(text) fallback otherwise (and on
    any internal gap, e.g. a byte_fallback vocab missing a <0xHH> entry)."""
    try:
        return len(encode(text, path))
    except (TokenizeUnsupported, ValueError):
        return len(text)
