import torch
from torch import nn, Tensor


class SimpleNetworkWithConstBuffer(nn.Module):
    """A minimal network exercising all three graph-signature input kinds.

    Architecture:
        x → matmul(w1) → mul(cos_cached) → mul(gate_mask) → mul(sin_cached) → matmul(w2) → add(bias) → out

    The three input kinds map to weight_info.json ``_is_weight`` classification
    (see cann_lm/converter/export/epair_utils.py:_pack_weights_to_dir):

        - weight (nn.Parameter):            w1, w2, bias           → _is_weight=True
        - buffer (register_buffer):         cos_cached, sin_cached → _is_weight=True
        - const (plain attribute tensor):   gate_mask              → _is_weight=False

    weight + buffer are nn.Module state (state_dict entries, shared across
    prefill/decode); const is a trace-time lifted tensor constant (independent
    of input, produced by torch.export from a plain module attribute that is
    neither a Parameter nor a registered buffer).
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        # weight: trainable model parameters → inputs_to_parameters
        self.w1 = nn.Parameter(torch.randn(dim, dim))
        self.w2 = nn.Parameter(torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        # buffer: module state, non-trainable → inputs_to_buffers
        # persistent=False mirrors qwen3.5 Qwen3_5TextRotaryEmbedding cos/sin caches
        self.register_buffer("cos_cached", torch.cos(torch.arange(dim, dtype=torch.float32) * 0.1),
                             persistent=False)
        self.register_buffer("sin_cached", torch.sin(torch.arange(dim, dtype=torch.float32) * 0.1),
                             persistent=False)
        # const: plain attribute tensor (NOT register_buffer/Parameter) → inputs_to_lifted_tensor_constants
        # torch.export lifts it as a trace-time constant (fixed gate/scale mask, input-independent)
        self.gate_mask = torch.linspace(0.5, 1.5, dim)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.matmul(x, self.w1)             # weight (Parameter)
        h = h * self.cos_cached                   # buffer (register_buffer)
        h = h * self.gate_mask                    # const (lifted tensor constant)
        h = h * self.sin_cached                   # buffer (register_buffer)
        h = torch.matmul(h, self.w2) + self.bias  # weight (Parameter) + weight (Parameter)
        return h


def main() -> None:
    dim = 64
    model = SimpleNetworkWithConstBuffer(dim).eval()
    x = torch.randn(2, dim)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Buffers: {sum(b.numel() for b in model.buffers()):,}")
    print(f"Const attrs (non-registered tensors): gate_mask {tuple(model.gate_mask.shape)}")


if __name__ == "__main__":
    main()
