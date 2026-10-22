目的：设计embedding模块
功能：通过查表将tokenId转为embedding

功能点：
1、初始化，加载embedding文件、embedding_dequant_scale反量化参数文件（可能不存在），输入包含embedding文件path、反量化参数文件path、embeddingType（数据类型）、embeddingScaleType（反量化参数类型）
2、通过查表，将tokenId转为embedding数组，输入为int32_t数组，inputEmbed地址(tensorBuffer取数据地址)、数据大小、inputEmbedDeQuantScale地址、数据大小