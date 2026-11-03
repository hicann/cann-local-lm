English | [简体中文](../../../zh/cann_lm_engine/embedding/proposal.md)

Purpose: Design the embedding module
Function: Convert tokenId to embedding via table lookup

Feature points:
1. Initialization: load the embedding_weights single file (four-segment layout: main table → per_layer table → main scale → per_layer scale; the latter two may be absent). Inputs include the embedding file path, embeddingType (data type), and embeddingScaleType (dequantization parameter type)
2. Via table lookup, convert tokenId to an embedding array. Inputs are an int32_t array, the inputEmbed address (obtain data address from tensorBuffer), data size, the inputEmbedDeQuantScale address, and data size
