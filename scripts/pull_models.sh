#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# pull_models.sh — 从 ModelScope / HuggingFace 拉取 HF 格式模型到本地
#
# 实现: git clone + git-lfs, 不依赖 modelscope/huggingface_hub python 包
#       (符合 AGENTS.md "不修改系统环境" 原则, 仅需系统 git-lfs)
# 鲁棒性: clone 阶段禁用 smudge (GIT_LFS_SKIP_SMUDGE=1), 权重由 git lfs pull
#         独立拉取 — 可中断/可恢复 (重跑即续传); 完成后校验 safetensors 非指针
# 目录: <output-dir>/<org>/<name>  (匹配 HF/MS cache 约定, 避免命名冲突)
#
# 前置: git-lfs 已安装 (校验: `git lfs version`)
#
# 用法:
#   bash scripts/pull_models.sh Qwen/Qwen3.5-0.8B unsloth/gemma-4-E2B-it
#   bash scripts/pull_models.sh --output-dir /data/models Qwen/Qwen3-0.6B
#   bash scripts/pull_models.sh --source huggingface Qwen/Qwen3-0.6B
#   bash scripts/pull_models.sh --jobs 3 Qwen/Qwen3.5-0.8B Qwen/Qwen3-0.6B Qwen/Qwen3.5-9B
#   bash scripts/pull_models.sh --skip-existing Qwen/Qwen3.5-9B   # 已存在则跳过 (不续传)
#   bash scripts/pull_models.sh --no-lfs Qwen/Qwen3.5-0.8B         # 仅元数据 (config 等, 不下权重)
#
# 环境覆盖: OUTPUT_DIR (默认 ./models), SOURCE (默认 modelscope), JOBS (默认 1)
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-./models}"
SOURCE="${SOURCE:-modelscope}"
SKIP_EXISTING=0
NO_LFS=0
JOBS="${JOBS:-1}"
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)    OUTPUT_DIR="$2"; shift 2;;
        --source)        SOURCE="$2"; shift 2;;
        --jobs)          JOBS="$2"; shift 2;;
        --skip-existing) SKIP_EXISTING=1; shift;;
        --no-lfs)        NO_LFS=1; shift;;
        -h|--help)       grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
        --) shift; while [[ $# -gt 0 ]]; do MODELS+=("$1"); shift; done;;
        *)               MODELS+=("$1"); shift;;
    esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: 至少指定一个 model_id (org/name 格式, 如 Qwen/Qwen3-0.6B). 用 --help 查看用法" >&2
    exit 1
fi
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
    echo "ERROR: --jobs 需为正整数, 得到: $JOBS" >&2; exit 1
fi
if ! git lfs version >/dev/null 2>&1; then
    echo "ERROR: git-lfs 未安装 (apt install git-lfs 或 https://git-lfs.com)" >&2
    exit 1
fi
git lfs install >/dev/null 2>&1 || true   # 确保 LFS filters 已注册 (幂等, 非致命)

# org/name -> clone URL
build_url() {
    local model_id="$1"
    case "$SOURCE" in
        modelscope)  echo "https://www.modelscope.cn/${model_id}.git";;
        huggingface) echo "https://huggingface.co/${model_id}.git";;
        *) echo "ERROR: 未知 source: $SOURCE (modelscope|huggingface)" >&2; exit 1;;
    esac
}

# 拉取单个模型: skip-smudge clone + git lfs pull; 目标已存在则续传 (除非 --skip-existing)
pull_one() {
    local model_id="$1"
    local target="$OUTPUT_DIR/$model_id"
    local url; url="$(build_url "$model_id")"

    if [[ -d "$target" ]]; then
        if [[ $SKIP_EXISTING -eq 1 || $NO_LFS -eq 1 ]]; then
            echo "[skip]  $model_id (已存在)"; return 0
        fi
        echo "[resume] $model_id 续传权重 (上次可能中断)..."
        git -C "$target" lfs pull
        return $?
    fi

    mkdir -p "$(dirname "$target")"
    echo "[clone] $model_id  ->  $target"
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$url" "$target"

    if [[ $NO_LFS -eq 1 ]]; then
        echo "[meta]  $model_id 仅元数据, 跳过权重"; return 0
    fi
    echo "[lfs]   $model_id 拉取权重 (可中断, 重跑即恢复)..."
    git -C "$target" lfs pull
}

# 校验: 权重文件非 LFS 指针 (<1KB 视为指针)
verify_one() {
    local model_id="$1"
    local target="$OUTPUT_DIR/$model_id"
    if [[ ! -d "$target" ]]; then
        echo "[fail]  $model_id (未拉取, 目标不存在)" >&2; return 1
    fi
    [[ $NO_LFS -eq 1 ]] && { echo "[meta]  $model_id"; return 0; }

    local bad=0 f sz
    while IFS= read -r -d '' f; do
        sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
        if [[ $sz -lt 1024 ]]; then
            echo "[warn]  $model_id 权重为 LFS 指针 (未下载完整): $(basename "$f")" >&2
            bad=1
        fi
    done < <(find "$target" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -print0 2>/dev/null)
    [[ $bad -eq 0 ]] && echo "[ok]    $model_id"
    return $bad
}

echo "=== Pull Models ==="
echo "  source    : $SOURCE"
echo "  output dir: $OUTPUT_DIR"
echo "  models    : ${#MODELS[@]} 个 (${MODELS[*]})"
echo "  jobs      : $JOBS  $([[ $NO_LFS -eq 1 ]] && echo '| no-lfs (仅元数据)' || echo '| 拉权重')"
[[ $SKIP_EXISTING -eq 1 ]] && echo "  skip-existing: 已开启"
echo ""

# 拉取 (单个失败不中断其他; 失败状态由后续 verify 准确报告)
if [[ "$JOBS" -le 1 ]]; then
    for m in "${MODELS[@]}"; do
        pull_one "$m" || echo "[fail]  $m 拉取失败 (见上)" >&2
    done
else
    running=0
    for m in "${MODELS[@]}"; do
        (pull_one "$m" || echo "[fail]  $m 拉取失败 (见上)" >&2) &
        running=$((running+1))
        if (( running >= JOBS )); then
            wait -n 2>/dev/null || true
            running=$((running-1))
        fi
    done
    wait 2>/dev/null || true
fi

# 校验所有 (串行, 准确报告每个模型状态 — 这是退出码的 source of truth)
echo ""
echo "=== Verify ==="
VERIFY_FAIL=0
for m in "${MODELS[@]}"; do
    verify_one "$m" || VERIFY_FAIL=$((VERIFY_FAIL+1))
done

echo ""
if [[ $VERIFY_FAIL -eq 0 ]]; then
    echo "=== 完成: ${#MODELS[@]} 个模型全部就绪 ($OUTPUT_DIR) ==="
else
    echo "=== 部分失败: verify=${VERIFY_FAIL}/${#MODELS[@]} (重跑可恢复中断的下载) ==="
fi
exit $VERIFY_FAIL
