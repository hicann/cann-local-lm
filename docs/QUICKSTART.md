# 快速入门

> 端到端用例：HF 模型 → epair → omc → 部署。环境依赖见 [env_dependencies.md](env_dependencies.md)，构建命令见 [构建指南](build_guide.md)。

> **前置**：CANN + P1 venv 已就绪（见 [构建指南](build_guide.md) Part 1 + 运行时前置），且已拿到 `cann_lm_compile_converter-*.run`（自行构建见 [构建指南](build_guide.md) §2.1）。鸿蒙部署还需 `cann_lm_engine_ohos-*.run`（同 §2.1 产出）。

---

## 安装

```bash
# 1. 解压 .run 包（自动运行 set_env.sh 打印环境提示）
bash /path/to/cann_lm_compile_converter-0.1.0-aarch64.run --extract=/opt/cann-device-lm

# 2. 装 cann_lm whl 进 P1 venv（converter 用，提供 cann-device-lm-converter CLI）
export CONVERTER_PYTHON=/path/to/venv/bin/python
$CONVERTER_PYTHON -m pip install /opt/cann-device-lm/wheels/cann_lm-*.whl

# 3.（仅量化需要）先 source NPU env 探测 DOPT_PYTHON，再装 cann_lm whl 进系统 python
source /opt/cann-device-lm/set_env.sh --npu
$DOPT_PYTHON -m pip install /opt/cann-device-lm/wheels/cann_lm-*.whl --no-deps --user
# 注：若自动探测失败，手动 export DOPT_PYTHON=/path/to/python3 后重跑上一行
```

---

## 端到端用例：HF → epair → omc → 部署

### 0. 准备 env

```bash
source $HOME/Ascend/cann/set_env.sh    # CANN
source /opt/cann-device-lm/set_env.sh         # cann-device-lm（LD_LIBRARY_PATH + GE 配置）
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 # 阻止 torch autoload npu backend（P1 venv 无 torch_npu）
export CONVERTER_PYTHON=/path/to/venv/bin/python   # P1 venv python
```

### ① HF → epair（导出 prefill + decode）

> 需提前从 [HuggingFace](https://huggingface.co) 或 [魔搭社区](https://modelscope.cn) 下载模型到本地，`--hf-path` 指向下载目录。下载方式：
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
$CONVERTER_PYTHON -m cann_lm.converter.cann_lm_converter \
    --hf-path <你的模型路径> \
    --output-dir ./output/e2e \
    --mode prefill-decode --target-soc KirinX90
```

> converter 后台运行会在 GE finalize 处挂起（已知），产物此前已落盘。可后台运行 + 轮询产物文件 + kill 绕过。

**量化导出（可选）：**

> **量化特殊要求**：
> - GPTQ 量化需 **NPU 设备加速**（dopt 子进程跑在 NPU 上），需额外准备 **P2 环境**（系统 python + torch_npu 2.7.1.post4，与 P1 venv 的 torch 2.9.0 版本不同，不能共用）。
> - 需 `source /opt/cann-device-lm/set_env.sh --npu`（自动探测 `DOPT_PYTHON` + 设 `DEVICE=npu`），探测失败时手动 `export DOPT_PYTHON=/path/to/python3`。
> - torchaudio 版本须与 torch 2.7.1 一致（`pip install torchaudio==2.7.1`），否则报 `undefined symbol`。
> - 无 NPU 时可跳过 GPTQ，用 `--quant-path` 消费已有 dopt 产物做离线量化。

```bash
# NPU 量化 env（--npu 自动探测 DOPT_PYTHON + 设 DEVICE=npu）
source /opt/cann-device-lm/set_env.sh --npu

# 方式一：converter 自动调 dopt（一步到位，需 NPU）
$CONVERTER_PYTHON -m cann_lm.converter.cann_lm_converter \
    --hf-path <你的模型路径> \
    --output-dir ./output/quant \
    --target-soc KirinX90 \
    --quant-method gptq

# 方式二：手动跑 dopt + converter 消费（两步，dopt 需 NPU，converter 不需）
# 准备：从 cann_lm 包目录拷校准数据和配置到工作目录
PKG_QUANT=$($DOPT_PYTHON -c "import cann_lm.converter.quant as q; print(q.__path__[0])")
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
$CONVERTER_PYTHON -m cann_lm.converter.cann_lm_converter \
    --hf-path <你的模型路径> \
    --output-dir ./output/quant \
    --target-soc KirinX90 \
    --quant-path ./output/dopt_quant
```

> 量化方案（W4A8）见 [量化设计](cann_lm_converter/quant/quant_design.md)。

### ② epair → omc（编译）

```bash
BIN=/opt/cann-device-lm/cann_lm_compile
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
| `prefill_ep_json_with_ascend_ir.json` | ①导出 | prefill 段 EP JSON（②编译输入） |
| `decode_ep_json_with_ascend_ir.json` | ①导出 | decode 段 EP JSON（②编译输入） |
| `weight_info.json` | ①导出 | 权重索引（②编译输入） |
| `weights_0.bin` | ①导出 | 权重数据（②编译加载） |
| `host_weights_info.json` | ①导出 | host 侧权重索引（embedding 等不在图中的权重，②编译加载） |
| `host_weights_0.bin` | ①导出 | host 侧权重数据（embedding 等，②编译加载） |
| `model_config.json` | ①导出 | 模型配置（推理用） |
| `tokenizer.json` / `tokenizer_config.json` | ①导出 | tokenizer（推理用） |
| `prefill/prefill.omc` | ②编译 | prefill 段 omc（部署到设备跑） |
| `decode/decode.omc` | ②编译 | decode 段 omc |
| `prefill\|decode/weight/*.bin` | ②编译 | 权值分离时外置权值 |

---

## 部署到鸿蒙设备

将 omc + 权值推到鸿蒙设备，用 `cann_lm_engine_ohos-*.run` 产出的 engine 跑：

```bash
# 鸿蒙设备上安装 engine（见 build_guide.md §2.2）
bash cann_lm_engine_ohos-0.1.0-aarch64.run --extract=/data/local/tmp/cann_lm_engine

# 推 omc + 权值到设备
hdc file send ./output/e2e /data/local/tmp/cann_lm_engine/

# 运行
cd /data/local/tmp/cann_lm_engine && ./run.sh
```
