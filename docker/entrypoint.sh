#!/usr/bin/env bash
# 显式 source conda，激活 my
. /opt/conda/etc/profile.d/conda.sh
conda activate my

# 执行后续命令（比如 bash，或 ros2 launch …）
exec "$@"