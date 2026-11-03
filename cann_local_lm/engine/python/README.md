# cann-local-lm (Python bindings)

CANN-LM 推理引擎的 Python 绑定 —— 基于 ctypes 的 **C ABI 透传层**，与 `cann_local_lm/engine/api/c/c_api.h` 字段对字段 1:1 对应。零运行时依赖（纯 Python 标准库），`.so` 惰性加载，既可在设备上做真实推理，也能在无引擎库时导入测试。

## 特性

- **零依赖**：`requires-python >= 3.8`，`dependencies = []`（纯 stdlib `ctypes`，`pip install` 即用）
- **1:1 透传**：结构体定义、函数原型、枚举值与 C 头文件逐一对应，无额外逻辑
- **惰性加载**：`import cann_local_lm` 不触碰 `.so`；首次构造 `Engine`（或显式调用 `get_library()`）时才解析库
- **Pythonic 封装**：`Engine` / `Context` / `Request` / `Result` 句柄包装，上下文管理器自动释放 C 内存
- **同步 + 异步**：`generate()` 阻塞返回 `Result`；`generate_async()` 返回 `Request`，支持流式回调
- **CLI 开箱即用**：`cllm generate` / `cllm chat` 两条命令，config 驱动，mlx_lm 风格

## 目录结构

```
cann_local_lm/engine/python/
├── pyproject.toml          # 包定义 (cann-local-lm 0.1.0, 零依赖, 3 个 console-script)
└── cann_local_lm/
    ├── __init__.py          # 公共 API 导出
    ├── engine.py            # Engine/Context/Request/Result 包装类 + marshaling helpers
    ├── cli.py               # 顶层 CLI 子命令调度器 (generate, chat)
    ├── generate.py          # generate 子命令 (config → Engine → 同步生成 → 打印)
    ├── chat.py              # chat 子命令 (交互式多轮对话 REPL)
    ├── config.py            # testcase.json 配置加载器 (camelCase → snake_case)
    ├── _ffi.py              # ctypes 透传层 + 库发现/加载
    ├── _enums.py            # Status / FinishReason / RequestStatus 枚举
    ├── exceptions.py        # CannLmError / CannLmStatusError / raise_for_status
    └── _token_count.py      # 轻量 token 计数器 (chat 预算守卫, 纯 stdlib)
```

## 安装

### 1. 安装 Python 包

```bash
cd cann_local_lm/engine/python
pip install .            # 普通安装
pip install -e .[dev]    # 开发模式（含 pytest>=7）
```

> 零运行时依赖，无需 torch / numpy / transformers 等任何三方库。

### 2. 准备引擎动态库 `libcann_lm_engine.so`

Python 包本身不包含 C 引擎库，需另行构建（架构须与运行 Python 匹配）：

```bash
# 仓库根目录：构建 C++ 引擎 + cann_lm_compile 工具
bash scripts/build.sh cpp    # 详见 docs/build_guide.md
```

构建产物或部署 `.so` 后，通过以下任一方式让 Python 找到它（**发现顺序**）：

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 | `--lib PATH` | CLI 显式指定（最高优先级） |
| 2 | `CANN_LM_LIB` | 环境变量，指向 `.so` 文件或其目录 |
| 3 | 仓库内候选路径 | 自动探测（见下） |

仓库内自动探测的候选路径（`_ffi.py` 中 `_repo_root()` 相对）：

- `deploy_ohos/libcann_lm_engine.so` — OHOS 设备部署
- `build_ohos/libcann_lm_engine.so` — OHOS 交叉构建产物
- `model_final/libcann_lm_engine.so` — 签名部署产物

> 若 `.so` 存在但 `CDLL` 加载失败，通常是缺少运行时依赖（`libhiai_ascendcl.so` / `libneural_network_core.so` / `libhiai_foundation.so`）。将其加入 `LD_LIBRARY_PATH` 即可。

## 快速开始（CLI）

两条命令均 **config 驱动**：`--config` 提供 `testcase.json`（模型路径 + tokenizer + 采样默认值），CLI 参数可覆盖。

### generate — 单次文本生成

```bash
cllm generate --config testcase.json --prompt "你好"
cllm generate --config testcase.json --prompt "你好" --max-tokens 128 --stop </s>
# 模块模式（调试用）：
python -m cann_local_lm.generate --config testcase.json --prompt "你好"
```

### chat — 交互式多轮对话

```bash
cllm chat --config testcase.json
# 进入 REPL：user> 输入，assistant> 回复；/quit 或 Ctrl-D 退出
```

### 通用参数

| 参数 | 说明 |
|------|------|
| `--config PATH` | **必填**，`testcase.json` |
| `--prompt TEXT` | 提示词（仅 generate；chat 走 stdin） |
| `--max-tokens N` | 覆盖配置的 `maxGenTokens` |
| `--stop STR` | 停止字符串（可重复，覆盖 `stopSeq`） |
| `--lib PATH` | 显式指定 `libcann_lm_engine.so` 路径 |
| `--show-paths` | 打印配置解析出的路径 + 磁盘存在性后退出（**不加载 .so，不建 Engine**，用于排查路径问题） |

### 退出码

| 码 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 引擎/生成失败（含部分输出） |
| 2 | 配置/用法错误，或无法加载引擎库 |

### `--show-paths` 排查示例

```bash
cllm generate --config testcase.json --show-paths
# 输出每个配置路径的存在性，快速定位 "路径设了但引擎说找不到" 的问题
```

## 配置文件 `testcase.json`

`config.py` 将 JSON 的 camelCase 字段归一化为 snake_case，供引擎消费。字段映射：

| JSON 字段 | 归一化字段 | 说明 |
|-----------|-----------|------|
| `tokenizer_type` | `tokenizer_type` | 分词器类型 |
| `tokenizer_path` | `tokenizer_path` | 分词器路径 |
| `model_path_prefill` | `prefill_omc_path` | prefill 图路径 |
| `model_path_decode` | `decode_omc_path` | decode 图路径 |
| `model_meta_config` | `meta_json_path` | meta 配置 JSON 路径 |
| `model_weight_dir_prefill` | `prefill_weight` | prefill 权重目录（**必填**） |
| `model_weight_dir_decode` | `decode_weight` | decode 权重目录（**必填**） |
| `sentences[].prompt` | `prompt` | 提示词 |
| `sentences[].maxGenTokens` | `max_tokens` | 最大生成 token 数 |
| `sentences[].topK` | `top_k` | top-k 采样 |
| `sentences[].topP` | `top_p` | top-p 采样 |
| `sentences[].temperature` | `temperature` | 温度 |
| `sentences[].seed` | `seed` | 随机种子 |
| `sentences[].repetitionPenalty` | `repetition_penalty` | 重复惩罚 |
| `sentences[].stopSeq` | `stop_strings` | 停止字符串列表 |

示例 `testcase.json`：

```json
{
  "tokenizer_type": "gemma",
  "tokenizer_path": "/path/to/tokenizer.json",
  "model_path_prefill": "/path/to/prefill.omc",
  "model_path_decode":  "/path/to/decode.omc",
  "model_meta_config":  "/path/to/meta.json",
  "model_weight_dir_prefill": "/path/to/weights/prefill",
  "model_weight_dir_decode":  "/path/to/weights/decode",
  "sentences": [
    {
      "prompt": "Hello, my name is",
      "maxGenTokens": 64,
      "topK": 0,
      "topP": 1.0,
      "temperature": 1.0,
      "seed": 0,
      "repetitionPenalty": 1.0,
      "stopSeq": []
    }
  ]
}
```

> CLI 使用 `sentences[0]` 的采样默认值；`--prompt` / `--max-tokens` / `--stop` 可覆盖。

## Python API

### 同步生成

```python
import cann_local_lm

# 1. 构建模型产物（路径形式；也可用 buffer 形式内联数据）
artifact = cann_local_lm.build_model_artifact(
    prefill_omc_path="/path/to/prefill.omc",
    decode_omc_path="/path/to/decode.omc",
    meta_json_path="/path/to/meta.json",
    prefill_weight="/path/to/weights/prefill",
    decode_weight="/path/to/weights/decode",
)

# 2. 构建引擎选项 + 采样参数
opts = cann_local_lm.build_engine_options(
    default_chunk_size=1024, max_concurrent_prefill=1,
    tokenizer_type="gemma", tokenizer_path="/path/to/tokenizer.json",
)
sp = cann_local_lm.build_sampling_params(temperature=0.8, top_k=40, top_p=0.9, seed=42)

# 3. 创建引擎（此时加载 .so + 模型；路径不存在会 fast-fail）
with cann_local_lm.Engine(artifact=artifact, options=opts) as eng:
    ctx = eng.create_context()
    try:
        r = eng.generate("Hello, my name is", ctx,
                         sampling_params=sp, max_tokens=128)
        print(r.text)            # 生成文本
        print(r.token_ids)       # token id 列表
        print(r.finish_reason)   # FinishReason.MAX_TOKENS / STOP_STRING / ...
    finally:
        eng.destroy_context(ctx.id)
```

### 无模型形式（c_api.h no-model）

```python
import cann_local_lm
with cann_local_lm.Engine() as eng:          # artifact=None, options=None
    ctx = eng.create_context()
    r = eng.generate("hello", ctx)     # Result on SUCCESS
```

### 异步生成 + 流式回调

```python
import cann_local_lm

with cann_local_lm.Engine(artifact=artifact, options=opts) as eng:
    ctx = eng.create_context()
    try:
        def on_token(token_id, text):
            print(text, end="", flush=True)   # 每个 token 回调（pipeline 线程）

        req = eng.generate_async(
            "讲个故事", ctx,
            sampling_params=sp, max_tokens=256,
            callback=on_token,
        )
        # 非阻塞轮询
        if not req.is_done:
            req.wait()          # 阻塞直到完成 -> Result
        # 或: req.try_get_result()  # 非阻塞取结果（未完成抛 INVALID_STATE）
    finally:
        eng.destroy_context(ctx.id)
```

> **回调注意**：`callback(token_id, text)` 在 pipeline 内部线程触发，**不可阻塞或调用 Engine API**。Python 异常不会跨 C 边界传播（被吞掉）。

### 核心 API 速览

| 类/函数 | 说明 |
|---------|------|
| `Engine(artifact=None, options=None, lib_path=None)` | 引擎包装；上下文管理器自动 `close()` |
| `Engine.create_context()` → `Context` | 创建对话上下文 |
| `Engine.generate(prompt, context, ...)` → `Result` | 同步生成；非 SUCCESS 抛 `CannLmStatusError`（带 `.result` 部分输出） |
| `Engine.generate_async(prompt, context, ..., callback=None)` → `Request` | 异步生成 + 流式回调 |
| `Context.id` / `.cancel()` / `.destroy()` | 上下文 id / 取消 / 销毁 |
| `Request.status` / `.is_done` / `.wait()` / `.try_get_result()` / `.cancel()` | 异步请求生命周期 |
| `Result.token_ids` / `.text` / `.finish_reason` | 推理输出（Python 拥有，C 内存已释放） |

### marshaling helpers

| 函数 | 用途 |
|------|------|
| `build_model_artifact(...)` | 路径 / buffer → `ModelArtifact`（buffer 为 view，需保活） |
| `build_engine_options(...)` | → `EngineOptions` |
| `build_sampling_params(...)` | → `SamplingParams`（默认值与 C++ 一致：temp=1.0, top_p=1.0） |
| `build_request_options(...)` | → `RequestOptions`（max_tokens / stop_token_ids / stop_strings） |
| `build_buffer(data)` | bytes/None → `Buffer` |

## 库加载

```python
import cann_local_lm

# 检查 .so 是否可加载（测试 skip-guard 用）
if not cann_local_lm.is_library_available():
    print("引擎库不可用")

# 显式加载（缓存）
lib = cann_local_lm.get_library("/path/to/libcann_lm_engine.so")
```

发现顺序：`--lib` / `get_library(path)` → `$CANN_LM_LIB` → 仓库内候选路径。加载失败抛 `CannLmError`，错误信息列出所有尝试过的候选路径及失败原因。

## 枚举与错误处理

### 枚举（1:1 对应 c_api.h）

```python
from cann_local_lm import Status, FinishReason, RequestStatus

# Status: SUCCESS=0, FAILURE=1, INVALID_ARGUMENT=2, INVALID_STATE=3, NOT_FOUND=4, CANCELED=5
# FinishReason: MAX_TOKENS=0, STOP_TOKEN=1, STOP_STRING=2, CANCELED=3, ERROR=4
# RequestStatus: QUEUED=0, PREFILLING=1, DECODING=2, COMPLETED=3, CANCELED=4, FAILED=5
```

### 异常

```python
from cann_local_lm import CannLmError, CannLmStatusError, raise_for_status
```

| 异常 | 说明 |
|------|------|
| `CannLmError` | 所有绑定错误的基类（含库未找到） |
| `CannLmStatusError` | C ABI 返回非 SUCCESS 时抛出；含 `.status`(Status)、`.func_name`(C 函数名)、`.message`、`.result`(部分输出) |
| `raise_for_status(status, func_name, message=None)` | 非 SUCCESS 抛异常，SUCCESS 时 no-op |

**状态契约**（来自 c_api.cpp）：`Result` 在 `SUCCESS` 或 `FAILURE` 时填充。故 `FAILURE` 时仍有部分 `Result`，绑定层将其附在 `CannLmStatusError.result` 上，部分输出永不丢失。

```python
try:
    r = eng.generate(prompt, ctx, max_tokens=64)
except CannLmStatusError as e:
    print(f"生成失败: {e.status.name}")
    if e.result is not None:
        print(f"部分输出: {e.result.text}")
```

## chat 子命令特性

- **多轮对话 REPL**：`user>` 输入 → `assistant>` 回复，历史拼合为 `Role: content` 纯文本 prompt
- **轮次上限**：20 轮（`MAX_ROUNDS`）
- **token 预算**：累计 prompt 不超过 16384 tokens（`MAX_PROMPT_TOKENS`），超额时 FIFO 丢弃最旧历史
- **token 计数**：纯 stdlib 实现（`_token_count.py`），对 SentencePiece-style byte-fallback BPE 精确计数；其他布局保守回退到 `len(text)`（CJK 近似精确，绝不欠计）
- **动画指示**：生成时显示 `thinking.` → `thinking..` → `thinking...` 动画（区分"正在工作"与"卡死"）

## 开发

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行测试
pytest
```

模块约定：以下划线前缀的 `_ffi` / `_enums` / `_token_count` 为内部模块；公共 API 全部从 `__init__.py` 导出。marshaling helpers 在无引擎库时亦可测试（`.so` 仅在 `Engine` 构造时加载）。
