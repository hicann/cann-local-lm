"""Weight layout transforms: ND row-major -> FractNZ/FractZ block-major (Ascend cube-native)."""

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


def pack_int4_nz(tensor: torch.Tensor) -> torch.Tensor:
    """KirinX90 s8s4 FractZ (format "Z"): int4pack int32 [K, N//8] -> GM fractal.

    Differs from nd2nz (standard NZ, 16x16 blocks for fp16/int8):
    - Output 4D shape [ceil(K/64), ceil(N/16), 16, 8] matches ge_graph_builder.cc
    - K block=64 int4, N block=16 int4 (= 2 int32), C0=8 int4 (= 1 int32)
    - Within each block K varies fastest (Z-order): flat = (j%2)*64 + (i%64),
      then reshaped to [16, 8]

    Non-aligned dims are zero-padded to match ge_graph_builder.cc ceil-div 4D shape.
    """
    if tensor.dim() != 2:
        raise ValueError(f"pack_int4_nz expects 2D, got {tensor.dim()}D")
    N, K_packed = tensor.shape
    k1 = (K_packed + 7) // 8
    n1 = N // 16
    pad_k = k1 * 8 - K_packed
    pad_n = n1 * 16 - N
    if pad_k or pad_n:
        tensor = torch.nn.functional.pad(tensor, (0, pad_n, 0, pad_k))
    # [K, N_packed] -> [k1, 64, n1, 2] -> permute -> [k1, n1, 2, 64] -> [k1, n1, 16, 8]
    return tensor.reshape(n1, 16, k1, 8).permute(2, 0, 1, 3).contiguous()
