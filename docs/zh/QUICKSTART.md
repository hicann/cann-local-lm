# 快速入门

简体中文 | [English](../en/QUICKSTART.md)

> 端到端用例：HF模型 → epair → omc → 部署。环境依赖见 [env_dependencies.md](env_dependencies.md)，构建命令见 [构建指南](build_guide.md)。

> **前置**：CANN + P1 venv已就绪（见 [构建指南](build_guide.md) Part 1 + 运行时前置），且已拿到 `cann_local_lm_converter-*.run`（自行构建见 [构建指南](build_guide.md) §2.1）。鸿蒙部署还需签名 engine wheel `cann_local_lm_engine-*+signed*.whl`（`--pkg` 或 `bash scripts/cann_lm_engine/build_package_ohos.sh` 产出）。

---

## 安装

```bash
# 1. 解压 .run 包（自动运行 set_env.sh 打印环境提示）
bash /path/to/cann_local_lm_converter-0.1.0-aarch64.run --extract=/opt/cann-local-lm

# 2. 装 cann_local_lm whl 进 P1 venv（converter 用，提供 cllm-converter CLI）
export CONVERTER_PYTHON=/path/to/venv/bin/python
$CONVERTER_PYTHON -m pip install /opt/cann-local-lm/wheels/cann_local_lm-*.whl

# 3.（仅量化需要）设 DOPT_PYTHON 指向装了 torch_npu 的系统 python
export DOPT_PYTHON=/path/to/python3
# 或自动探测（需先 source CANN）：source $HOME/Ascend/cann/set_env.sh && source /opt/cann-local-lm/set_env.sh --npu
# 首次部署若系统 python 缺 cann_local_lm/dopt_lm：$DOPT_PYTHON -m pip install /opt/cann-local-lm/wheels/cann_local_lm-*.whl --no-deps --user
```

---

## 端到端用例：HF → epair → omc → 部署

### 0. 准备env

```bash
source $HOME/Ascend/cann/set_env.sh    # CANN
source /opt/cann-local-lm/set_env.sh         # cann-local-lm（LD_LIBRARY_PATH + GE 配置）
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 # 阻止 torch autoload npu backend（P1 venv 无 torch_npu）
export CONVERTER_PYTHON=/path/to/venv/bin/python   # P1 venv python
```

### ① HF → epair（导出prefill + decode）

> 需提前从 [HuggingFace](https://huggingface.co)或 [魔搭社区](https://modelscope.cn)下载模型到本地，`--hf-path` 指向下载目录。下载方式：
> ```bash
> # HuggingFace
> pip install huggingface_hub
> huggingface-cli download <org>/<model-name> --local-dir /path/to/model
>
> # 魔搭社区（国内更快）
> pip install modelscope
> modelscope download --model <org>/<model-name> --local_dir /path/to/model
> ```

**普通导出：**

```bash
cllm-converter \
    --hf-path <你的模型路径> \
    --output-dir ./output/e2e \
    --mode prefill-decode --target-soc KirinX90
```

> 导出正常结束时进程会自动退出（日志末行 `Done. Output: <dir>`）。若在部分环境下进程未退出，产物此时已全部落盘，可直接 kill。

**量化导出（可选）：**

> **量化特殊要求**：
> - GPTQ量化需 **NPU设备加速**（dopt子进程跑在NPU上），需额外准备 **P2环境**（系统python + torch_npu，与P1 venv隔离，不能共用）。
> - 需先 `source $HOME/Ascend/cann/set_env.sh` 再 `source /opt/cann-local-lm/set_env.sh --npu`（自动探测 `DOPT_PYTHON` + 设 `DEVICE=npu`），探测失败时手动 `export DOPT_PYTHON=/path/to/python3`。
> - torchaudio版本须与系统python的torch一致，否则报 `undefined symbol`。
> - 无NPU时可跳过GPTQ，用 `--quant-path` 消费已有dopt产物做离线量化。

```bash
# NPU 量化 env（先 CANN 再 --npu，后者探测 DOPT_PYTHON 依赖 ASCEND_HOME_PATH）
source $HOME/Ascend/cann/set_env.sh
source /opt/cann-local-lm/set_env.sh --npu

# 方式一：converter 自动调 dopt（一步到位，需 NPU）
cllm-converter \
    --hf-path <你的模型路径> \
    --output-dir ./output/quant \
    --target-soc KirinX90 \
    --quant-method gptq

# 方式二：手动跑 dopt + converter 消费（两步，dopt 需 NPU，converter 不需）
# 准备：从 cann_local_lm 包目录拷校准数据和配置到工作目录
PKG_QUANT=$($DOPT_PYTHON -c "import cann_local_lm.converter.quant as q; print(q.__path__[0])")
mkdir -p ./output/dopt_quant
cp -r "$PKG_QUANT/calibrate_sets" ./output/dopt_quant/
cp "$PKG_QUANT/config.yaml" ./output/dopt_quant/
# config.yaml: 可配置校准数据集（train_files）、采样数（ptq_samples）、序列长度（cutoff_len）等
# dopt_config.json: 逐层量化策略（layer_strategy），每层可单独设 quant_strategy（如 Quant_INT8-4_ECO / Quant_Embed_MinMax / float 跳过）

# Step 1a: 首次运行（生成 dopt_config.json 并退出）
cd ./output/dopt_quant
$DOPT_PYTHON -m dopt_lm.opt_main \
    --model-path <你的模型路径> \
    --dopt-config ./dopt_config.json \
    --optimize-config ./config.yaml \
    --output-dir .
# Step 1b: 编辑 dopt_config.json 量化策略（Linear→Quant_INT8-4_ECO, Embedding→Quant_Embed_MinMax, 可按层粒度调）
# Step 1c: 重跑（执行 GPTQ 量化）
$DOPT_PYTHON -m dopt_lm.opt_main \
    --model-path <你的模型路径> \
    --dopt-config ./dopt_config.json \
    --optimize-config ./config.yaml \
    --output-dir .
cd ../..    # 回到仓根

# Step 2: converter 消费 dopt 产物（离线量化，不需 NPU）
cllm-converter \
    --hf-path <你的模型路径> \
    --output-dir ./output/quant \
    --target-soc KirinX90 \
    --quant-path ./output/dopt_quant
```

> 量化方案（W4A8）见 [量化设计](cann_lm_converter/quant/quant_design.md)。

### ② epair → omc（编译）

```bash
BIN=/opt/cann-local-lm/cann_lm_compile
OUT=./output/e2e       # 普通导出；量化导出用 ./output/quant
for st in prefill decode; do
  mkdir -p "$OUT/$st"
  "$BIN" "$OUT/${st}_ep_json_with_ascend_ir.json" "$OUT/weight_info.json" "$OUT/$st/$st"
done
```

---

## 产物总览

① 导出 + ② 编译后的完整产物（`$OUT/` = `./output/e2e` 或 `./output/quant`）：

| 文件 | 来源 | 说明 |
|------|------|------|
| `prefill_ep_json_with_ascend_ir.json` | ①导出 | prefill段EP JSON（②编译输入） |
| `decode_ep_json_with_ascend_ir.json` | ①导出 | decode段EP JSON（②编译输入） |
| `weight_info.json` | ①导出 | 权重索引（②编译输入） |
| `weights_0.bin` | ①导出 | 权重数据（②编译加载） |
| `host_weights_info.json` | ①导出 | host侧权重索引（embedding等不在图中的权重，②编译加载） |
| `host_weights_0.bin` | ①导出 | host侧权重数据（embedding等，②编译加载） |
| `model.json` | ①导出 | **推理用 meta 配置**（含 `model_config` / `embedding_config` / `tokenizer_config` 三段），部署时填 `model_meta_config` |
| `model_config.json` | ①导出 | HF 侧模型配置快照（供查阅，**不是** engine 的 `model_meta_config`） |
| `tokenizer.json` / `tokenizer_config.json` | ①导出 | tokenizer（推理用） |
| `prefill/prefill.omc` | ②编译 | prefill段omc（部署到设备跑） |
| `decode/decode.omc` | ②编译 | decode段omc |
| `prefill\|decode/weight/` | ②编译 | 权值分离时外置权值目录（含 `delegate_graph_weight_combined` / `meta.json` / `SubGraph_*.weight` / `weight_0.weight`） |

---

## 部署到鸿蒙设备

### 1. 推送 engine + 模型产物

```bash
# engine 签名 wheel（.so 内嵌在 cann_local_lm/lib/，--pkg 或 build_package_ohos.sh 产出）
hdc file send build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl /data/local/tmp/cann_lm_engine/

# 模型产物（omc + 权值 + tokenizer + model_config）
hdc file send ./output/e2e /data/local/tmp/cann_lm_engine/
```

### 2. 设备上安装 engine + 编写配置

```bash
hdc shell
cd /data/local/tmp/cann_lm_engine

# 装 engine wheel（.so 自动从 cann_local_lm/lib/ 发现）
pip3 install cann_local_lm_engine-0.1.0+signed-py3-none-any.whl

# 编写 testcase.json（模型路径 + tokenizer + 采样参数）
cat > testcase.json <<'EOF'
{
    "tokenizer_type": "BPE",
    "tokenizer_path": "/data/local/tmp/cann_lm_engine/output/e2e/tokenizer.json",
    "model_path_prefill": "/data/local/tmp/cann_lm_engine/output/e2e/prefill/prefill.omc",
    "model_path_decode": "/data/local/tmp/cann_lm_engine/output/e2e/decode/decode.omc",
    "model_meta_config": "/data/local/tmp/cann_lm_engine/output/e2e/model.json",
    "model_weight_dir_prefill": "/data/local/tmp/cann_lm_engine/output/e2e/prefill/weight",
    "model_weight_dir_decode": "/data/local/tmp/cann_lm_engine/output/e2e/decode/weight",
    "sentences": [{
        "prompt": "你好",
        "seed": 99, "topK": 1, "topP": 1.0,
        "temperature": 1.0, "maxGenTokens": 128
    }]
}
EOF
```

> **无 pip 时**：`unzip cann_local_lm_engine-*.whl -d . && export PYTHONPATH=.:$PYTHONPATH && export CANN_LM_LIB=./cann_local_lm/lib/libcann_lm_engine.so`
>
> **权值目录**：编译时若未设 `GE_EXTERNAL_WEIGHT=2`（内嵌权值），`model_weight_dir_*` 可指向 omc 同目录。

### 3. 运行推理

```bash
# 单次生成
cllm generate --config testcase.json --prompt "你好"

# 交互式多轮对话
cllm chat --config testcase.json

# 排查路径（不加载 .so，仅检查文件存在性）
cllm generate --config testcase.json --show-paths
```

> `.so` 发现顺序：`--lib PATH` > `$CANN_LM_LIB` > wheel 内 `cann_local_lm/lib/`。加载失败时将 `.so` 所在目录加入 `LD_LIBRARY_PATH`（`libc++_shared.so` 需同目录）。
>
> C++ 黑盒测试（bbit/smoke）可通过 `bash build.sh --smoke` 单独构建，推送后 `./bbit --config testcase.json` 运行。详见 [engine Python README](../../cann_local_lm/engine/python/README.md)。
