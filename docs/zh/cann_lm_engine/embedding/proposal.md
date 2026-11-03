简体中文 | [English](../../../en/cann_lm_engine/embedding/proposal.md)

目的：设计embedding模块
功能：通过查表将tokenId转为embedding

功能点：
1、初始化，加载 embedding_weights 单文件（四段布局：主表 → per_layer 表 → 主 scale → per_layer scale；后两段可缺省），输入包含 embedding 文件 path、embeddingType（数据类型）、embeddingScaleType（反量化参数类型）
2、通过查表，将tokenId转为embedding数组，输入为int32_t数组，inputEmbed地址(tensorBuffer取数据地址)、数据大小、inputEmbedDeQuantScale地址、数据大小
