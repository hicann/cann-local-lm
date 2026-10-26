# Claude Code Prompt 模板（中文）

本文档提供一组 **可复用的 Prompt 模板**，用于在 Claude Code 或其他 coding agent 中进行：

* 架构设计
* 方案讨论
* 代码实现
* 重构
* Debug
* 文档生成

目标：

* 保持 Prompt 结构清晰
* 减少上下文膨胀
* 固化设计决策
* 提升 AI 协作稳定性

# 架构设计 Prompt

```
在执行任务之前：

请先阅读并遵守：
@AGENTS.md

参考架构：
@docs/architecture.md

任务：
设计 <子系统名称> 的架构。

需要输出：

1. 子系统职责
2. 模块划分
3. 核心接口
4. 数据流
5. 关键设计约束

要求：

- 只讨论设计
- 不生成实现代码
- 保持结构清晰
```

---

# 模块实现 Prompt

```
在开始之前：

请阅读并遵守：
@AGENTS.md

架构文档：
@docs/architecture.md

模块设计：
@docs/design/<module>.md

任务：
实现 <模块名称>。

要求：

- 严格遵循架构
- 不引入未讨论的功能
- 保持模块边界
- 模块验证、有依赖的 pip 包时，注意使用 venv 环境
```

---

# 重构 Prompt

```
在执行任务前：

请阅读并遵守：
@AGENTS.md

架构参考：
@docs/architecture.md

目标模块：
@src/<module>

目标：

- 简化结构
- 提高可读性
- 与架构保持一致
```

---

# Bug 分析 Prompt

```
在执行任务之前：

请阅读并遵守：
@AGENTS.md

架构：
@docs/architecture.md

相关代码：
@src/<module>

问题描述：

<描述 bug>

任务：

1. 分析可能原因
2. 定位模块
3. 提出修复方案

先不要修改代码。
```

---

# 代码审查 Prompt

```
在执行任务前：

请阅读并遵守：
@AGENTS.md

架构参考：
@docs/architecture.md

代码：
@src/<module>

检查：

- 是否符合架构
- 是否存在潜在 bug
- 是否有性能风险
```

---

# Prompt：让 Claude Code 生成一个 Skill

在开始之前：

请先阅读并遵守：
@AGENTS.md


任务：

为本项目创建一个新的 Claude Code skill。


Skill 目标：

<在这里填写 skill 的用途，例如：代码审查 / 机制分析 / 调用链分析 等>


要求：

1. 设计一个可复用的 skill prompt
2. 该 prompt 应适用于 Claude Code
3. 该 prompt 应该是结构化的，并且稳定可复用
4. 避免依赖具体业务上下文
5. 优先使用 @path 引用代码或文档


生成的内容包括：

1. Skill 名称
2. Skill 使用场景
3. Skill Prompt（可直接使用）
4. 示例调用方式


输出文件：

`.claude/skills/<skill_name>.md`


Skill Prompt 的结构应包含：

- 阅读 AGENTS.md
- 任务描述
- 输入对象（例如 @src/...）
- 分析或执行步骤
- 输出格式


生成的 skill 文件应该是完整的 Markdown 文档，可以直接被 Claude Code 使用。

---
