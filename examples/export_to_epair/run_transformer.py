"""Load and run the SimpleTransformer with saved weights.

Usage:
  # from project root (CANN-LM_5883/):
  python -m cann_lm.converter.export.run_transformer

  # from anywhere:
  python /path/to/cann_lm/converter/export/run_transformer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- ensure project root is on sys.path ---
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # CANN-LM_5883/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ------------------------------------------

import torch
from safetensors.torch import load_file

from cann_lm.converter.export.simple_transformer import SimpleTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(weights_path: str | Path, device: str = DEVICE) -> SimpleTransformer:
    model = SimpleTransformer(256, 64, 8, 256, 2, 128).eval().to(device)
    path = Path(weights_path)
    if path.suffix == ".safetensors":
        state_dict = load_file(str(path), device=device)
    else:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def generate(model: SimpleTransformer, prompt: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
    model.eval()
    tokens = prompt
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens


def main() -> None:
    root = Path(__file__).resolve().parent
    model = load_model(root / "simple_transformer.safetensors")

    prompt = torch.randint(0, 256, (1, 8))
    output = generate(model, prompt, max_new_tokens=16)

    print(f"Prompt shape:   {prompt.shape}")
    print(f"Output shape:   {output.shape}")
    print(f"Prompt tokens:  {prompt[0].tolist()}")
    print(f"Output tokens:  {output[0].tolist()}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
