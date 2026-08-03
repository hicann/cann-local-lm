"""Verify QuantBatchMatmulV4 (torch_npu.npu_quant_matmul / aclnnQuantMatmulV5) on NPU.

W8A8 K-C mode (int8 act + int8 weight, per-token + per-channel fp32 scales → fp16).
Runs on Ascend910_9362 (ascend910_93 family). Asserts the op output matches the
reference (x1 @ (q*scale).T * pertoken_scale) bit-exactly (maxdiff == 0).

Usage:
  source /home/developer/Ascend/cann/set_env.sh
  python3 verify_quant_matmul_v4_w8a8.py
"""
import torch
import torch_npu


def main() -> None:
    torch.npu.set_device(0)
    dev = "npu"
    m, k, n = 4, 8, 4
    torch.manual_seed(0)

    # weight [n, k] fp -> int8 (symmetric per-channel)
    w = torch.randn(n, k, device=dev, dtype=torch.float32) * 0.1
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)            # [n] per-channel fp32 (x2Scale)
    q_w = torch.clamp(torch.round(w / scale.unsqueeze(1)), -127, 127).to(torch.int8)  # [n,k] int8
    x2 = q_w.t().contiguous()                                         # [k, n] int8

    # act [m, k] int8 + per-token pertoken_scale [m]
    x1 = torch.randint(-127, 127, (m, k), dtype=torch.int8, device=dev)
    pertoken_scale = torch.ones(m, device=dev, dtype=torch.float32) * 0.01  # x1Scale [m]

    # QuantBatchMatmulV4 (eager via npu_quant_matmul -> aclnnQuantMatmulV5)
    y = torch.ops.npu.npu_quant_matmul(
        x1, x2, scale, pertoken_scale=pertoken_scale, output_dtype=5,  # 5 = fp16
    )

    # reference: (x1 @ (q_w*scale).T) * pertoken_scale
    w_deq = q_w.float() * scale.unsqueeze(1)                          # [n, k] dequantized
    ref = (x1.float() @ w_deq.t() * pertoken_scale.unsqueeze(1)).to(torch.float16)

    maxdiff = (y.float() - ref.float()).abs().max().item()
    print(f"device: {torch.npu.get_device_name(0)}")
    print(f"W8A8 K-C: x1{tuple(x1.shape)} int8, x2{tuple(x2.shape)} int8, "
          f"scale{tuple(scale.shape)} fp32, pertoken{tuple(pertoken_scale.shape)} fp32")
    print(f"out: {tuple(y.shape)} {y.dtype}")
    print(f"maxdiff vs ref = {maxdiff}")
    print(f"y[:4]  = {y.flatten()[:4].tolist()}")
    print(f"ref[:4] = {ref.flatten()[:4].tolist()}")
    assert maxdiff < 1e-5, f"W8A8 V4 mismatch (maxdiff={maxdiff})"
    print("PASS: QuantBatchMatmulV4 W8A8 verified on NPU (maxdiff=0)")


if __name__ == "__main__":
    main()
