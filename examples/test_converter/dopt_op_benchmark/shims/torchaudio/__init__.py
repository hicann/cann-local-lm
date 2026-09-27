__version__ = "2.11.0"
import types

functional = types.SimpleNamespace(resample=lambda *a, **k: None)
compliance = types.SimpleNamespace(kaldi=types.SimpleNamespace(fbank=lambda *a, **k: None))


def load(*args, **kwargs):
    raise RuntimeError("torchaudio stub: audio I/O unavailable in this env")
