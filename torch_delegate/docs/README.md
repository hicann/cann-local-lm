# TorchDelegate 文档

简体中文 | [English](README_en.md)

本目录提供TorchDelegate的用户指南、API参考、设计文档和约束说明。

## 文档分层

| 层级 | 目录 | 内容 |
|------|------|------|
| 用户指南 | `zh/guide/` | 编译安装、epair打包、Fallback Codegen使用、环境变量 |
| API参考 | `zh/api/` | Python API、C++ API |
| 设计文档 | `zh/features/` | 整体架构、epair格式、Fallback Codegen、Lite Context等设计资料 |
| 约束说明 | `zh/constraints/` | CANN Stub、CXX11 ABI兼容性等开发约束 |

英文文档位于`en/`目录，目录结构和文件名与`zh/`完全一致。

## 阅读路径

- 首次使用：先阅读[《编译安装》](zh/guide/build.md)，再按[《epair打包指南》](zh/guide/build_epair.md)完成模型导出。
- Python接入：阅读[《Python API参考》](zh/api/python_api.md)。
- C++部署：阅读[《EpairModelLoader API》](zh/api/epair_model_loader.md)。
- 自定义算子：阅读[《Fallback Codegen用户指南》](zh/guide/fallback_codegen_guide.md)。
- 了解实现：阅读[《整体架构》](zh/features/architecture.md)和[《Epair格式设计》](zh/features/epair_format_design.md)。

## 文档维护约定

- 新增用户指南放在`zh/guide/`，新增API放在`zh/api/`，新增设计文档放在`zh/features/`，新增约束说明放在`zh/constraints/`。
- 中文文档修改或新增时，同步修改或新增`en/`下同名文件。
- 文件名使用英文小写，多个单词使用下划线连接。
- 历史设计文档在文首标注设计时点和适用范围。
- Python API当前按模块维护在单个`python_api.md`中，作为本项目对“每个API一个文件”规则的登记例外；新增独立API时优先拆分为单独文件。
