"""Exercise byte boundaries with synthetic strings only; no model required."""

import ctypes as ct
import unittest

from cann_local_lm.engine import _make_callback


def emit(callback, token, raw):
    buffer = ct.create_string_buffer(raw)
    callback(token, ct.cast(buffer, ct.c_void_p), len(raw), None)


class StreamingUtf8Test(unittest.TestCase):
    def test_every_partition_of_multibyte_characters(self):
        for text in ("你", "好", "é", "€", "한", "あ", "😀", "𐍈", "️", "́"):
            raw = text.encode("utf-8")
            for mask in range(1 << (len(raw) - 1)):
                with self.subTest(text=text, mask=mask):
                    pieces = []
                    start = 0
                    for i in range(1, len(raw)):
                        if mask & (1 << (i - 1)):
                            pieces.append(raw[start:i])
                            start = i
                    pieces.append(raw[start:])
                    events = []
                    callback, _ = _make_callback(lambda t, s: events.append((t, s)))
                    for token, piece in enumerate(pieces):
                        emit(callback, token, piece)
                    self.assertEqual("".join(s for _, s in events), text)
                    self.assertEqual([t for t, _ in events], list(range(len(pieces))))

    def test_mixed_text_byte_by_byte(self):
        text = "abc你好 é € 😀 xyz"
        output = []
        callback, _ = _make_callback(lambda _, s: output.append(s))
        for i, byte in enumerate(text.encode("utf-8")):
            emit(callback, i, bytes([byte]))
        self.assertEqual("".join(output), text)

    def test_decoders_are_independent(self):
        outputs = [[], []]
        first, _ = _make_callback(lambda _, s: outputs[0].append(s))
        second, _ = _make_callback(lambda _, s: outputs[1].append(s))
        emit(first, 0, b"\xe4")
        emit(second, 0, b"\xc3")
        emit(first, 1, b"\xbd\xa0")
        emit(second, 1, b"\xa9")
        self.assertEqual(["".join(v) for v in outputs], ["你", "é"])

    def test_empty_and_invalid_chunks(self):
        output = []
        callback, _ = _make_callback(lambda _, s: output.append(s))
        emit(callback, 0, b"")
        emit(callback, 1, b"\xe4")
        emit(callback, 2, b"")
        emit(callback, 3, b"\xbd\xa0\xff!")
        self.assertEqual(output, ["", "", "", "你�!"])

    def test_callback_exception_does_not_reset_decoder(self):
        output = []

        def receive(token, text):
            if token == 0:
                raise ValueError("test callback")
            output.append(text)

        callback, _ = _make_callback(receive)
        emit(callback, 0, b"\xe4")
        emit(callback, 1, b"\xbd\xa0")
        self.assertEqual(output, ["你"])


if __name__ == "__main__":
    unittest.main()
