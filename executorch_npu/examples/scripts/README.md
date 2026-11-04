# 示例脚本

简体中文 | [English](README_en.md)

可以从任意工作目录运行这些脚本。仓库根目录由脚本所在位置解析；Python和已安装的wheel来自当前激活的环境。运行前请先激活Python环境，例如：

```bash
conda activate executorch_npu
bash examples/scripts/run.sh qwen3_5 export-pt2 --help
```

请先加载CANN和delegate的setup脚本，或通过 `CANN_ENV_SCRIPT` 和 `TORCH_DELEGATE_ENV_SCRIPT` 提供其路径。已存在的、带有 `set_env.sh` 的 `ASCEND_HOME_PATH` 也会被识别。数据集下载不需要CANN。

| 变量 | 行为 |
|---|---|
| `EXECUTORCH_NPU_ROOT` | 可选的仓库根目录覆盖项；否则由本目录推导。 |
| `CANN_ENV_SCRIPT` | 可选的CANN setup脚本；用于选择CANN版本。 |
| `TORCH_DELEGATE_ENV_SCRIPT` | 可选的已安装delegate setup脚本。 |
| `TORCH_DELEGATE_REPO` | 可选的delegate源码检出路径，用于Python/bridge库查找；未安装delegate whl时，将其指向仓内的 `../torch_delegate`。 |
| `CANN_ET_CONDA_ENV` | 仅在显式设置时激活该conda环境。 |
| `CONDA_SETUP_SCRIPT` | 当conda不在PATH中时，可选的conda shell初始化脚本。 |
| `EXECUTORCH_SDK` | 可选的已安装ExecuTorch SDK目录；否则从Python中发现。 |
| `FLATC_EXECUTABLE` | 可选的flatc覆盖项；否则使用已安装的ExecuTorch wheel。 |
| `CANN_ET_BUILD_DIR` | Runner构建目录；默认为仓库下的 `build/release`。 |
| `CANN_ET_BUILD_JOBS` | 构建并行度；默认为8。 |
| `CANN_ET_BOOTSTRAP_SCRIPT` | 供 `download_mmlu_bootstrap.sh` 使用的可选用户网络setup。 |

代理变量和Hugging Face缓存配置会原样从调用方继承。下载数据集前，请根据你的网络进行配置。
