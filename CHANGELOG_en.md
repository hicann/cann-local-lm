# CHANGELOG

English | [简体中文](CHANGELOG.md)

> This document records important changes for each version, listed in reverse chronological order.

## 2026.11

- Embedding weights and scale merged into a single `embedding_weights` file (four-segment layout: main table → per_layer table → main scale → per_layer scale; no file header, no inter-segment padding); the engine read side now loads the single file — table/scale are replaced atomically in one file, eliminating silent mismatch from updating only one of them, and one fewer deployment artifact
- `embedding_dequant_scale` config key deprecated: a leftover non-empty value only produces a WARN and does not affect parsing; existing dual-file artifacts must be re-exported under the new four-segment layout before loading
- With identity (per_layer segments) enabled, the whole-file total length must exactly equal the four-segment sum (load-time validation, guarding against row width W mismatching the file/graph)

## 2026.10

- Converter current capabilities: HuggingFace model → epair export (torch.export + FX→GE conversion + weight separation), supports W4A8 quantization
- Engine current capabilities: three-layer inference engine architecture, supports Continuous Batching, KV Cache management, asynchronous streaming inference, multi-turn continuation, request cancellation
- Fix omc output reading by actual dtype (FP16/BF16 golden comparison false positive)
- CLI parameter simplification: dtype derived from `--target-soc`, `--use-embeddings` fixed to True
- Quantization operator dump + simulated inference + hook consistency verification
- Converter documentation cleanup (removed 8 deprecated docs, merged converter_design into README)
