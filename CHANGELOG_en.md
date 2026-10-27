# CHANGELOG

English | [简体中文](CHANGELOG.md)

> This document records important changes for each version, listed in reverse chronological order.

## 2026.10

- Converter current capabilities: HuggingFace model → epair export (torch.export + FX→GE conversion + weight separation), supports W4A8 quantization
- Engine current capabilities: three-layer inference engine architecture, supports Continuous Batching, KV Cache management, asynchronous streaming inference, multi-turn continuation, request cancellation
- Fix omc output reading by actual dtype (FP16/BF16 golden comparison false positive)
- CLI parameter simplification: dtype derived from `--target-soc`, `--use-embeddings` fixed to True
- Quantization operator dump + simulated inference + hook consistency verification
- Converter documentation cleanup (removed 8 deprecated docs, merged converter_design into README)
