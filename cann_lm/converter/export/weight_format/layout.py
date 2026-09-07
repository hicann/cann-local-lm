"""Weight layout transforms: ND row-major -> FractNZ block-major (Ascend cube-native)."""

import torch


def nd2nz(tensor: torch.Tensor, block: int = 16, pad: bool = True) -> torch.Tensor:
    """[M, K] ND -> FractNZ [ceil(M/b), ceil(K/b), b, b] block-major (same dtype, contiguous).

    Non-block-multiple dims are zero-padded (partial block); the logical shape
    stays [M, K] (pad is internal to the physical block). Dtype-agnostic: call
    after any dtype view (e.g. bfloat16->int16) since this only rearranges bytes.
    """
    if tensor.dim() != 2:
        raise ValueError(f"nd2nz expects 2D, got {tensor.dim()}D")
    M, K = tensor.shape
    m, k = (M + block - 1) // block, (K + block - 1) // block
    pad_m, pad_k = m * block - M, k * block - K
    if pad_m or pad_k:
        if not pad:
            raise ValueError(f"shape {(M, K)} not block{block}-aligned and pad=False")
        tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_m))
    return tensor.reshape(m, block, k, block).permute(2, 0, 1, 3).contiguous()
