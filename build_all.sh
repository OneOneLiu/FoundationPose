#!/usr/bin/env bash
set -euo pipefail

# 记住当前目录
DIR=$(pwd)
PYTHON_EXECUTABLE=$(which python3)

echo "Using Python interpreter: ${PYTHON_EXECUTABLE}"

# 1) 编译 C++ binding （mycpp）
echo "=== Building mycpp ==="
cd "${DIR}/mycpp"
rm -rf build
mkdir -p build && cd build
cmake .. \
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

# 2) 如果你有本地 Kaolin 源码并且需要重新安装，请取消下面注释并修改路径
# echo "=== Installing local Kaolin (skip if already installed system-wide) ==="
# cd "${DIR}/path/to/your/kaolin"
# rm -rf build *.egg-info
# "${PYTHON_EXECUTABLE}" -m pip install -e .

# 3) 编译并安装 bundlesdf/mycuda
echo "=== Installing bundlesdf/mycuda ==="
cd "${DIR}/bundlesdf/mycuda"
rm -rf build *.egg-info
"${PYTHON_EXECUTABLE}" -m pip install -e .

# 回到工作目录
cd "${DIR}"
echo "=== build_all.sh complete ==="
