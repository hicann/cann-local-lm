English | [简体中文](../../../zh/cann_lm_engine/embedding/proposal.md)

Purpose: Design the embedding module
Function: Convert tokenId to embedding via table lookup

Feature points:
1. Initialization: load the embedding file and the embedding_dequant_scale dequantization parameter file (may not exist). Inputs include the embedding file path, dequantization parameter file path, embeddingType (data type), and embeddingScaleType (dequantization parameter type)
2. Via table lookup, convert tokenId to an embedding array. Inputs are an int32_t array, the inputEmbed address (obtain data address from tensorBuffer), data size, the inputEmbedDeQuantScale address, and data size
