#!/usr/bin/env bash
# 构建 + 运行 Sample（Sampler）C++ 验收。任意 cwd 可跑。
# 用法： bash scripts/cann_lm_engine/build_sample.sh
# 退出码：0=全部测试通过；非零=编译失败或有测试失败。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAMPLE_DIR="$REPO_ROOT/cann_local_lm/engine/core/sample"

# CANN 环境（若有则 source；原型不依赖 CANN，仅生产版需要。无则跳过。）
if [ -f /home/developer/Ascend/cann/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /home/developer/Ascend/cann/set_env.sh 2>/dev/null || true
fi

cd "$SAMPLE_DIR"

echo "==> 编译 sampler.cc + test/sampler_ut.cc"
# -Wno-missing-field-initializers：designated initializer 合法地省略字段走默认值，该 warning 无意义。
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers \
    sampler.cc test/sampler_ut.cc -o test/sampler_ut

echo "==> 运行 test/sampler_ut"
./test/sampler_ut
