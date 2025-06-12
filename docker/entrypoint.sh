#!/usr/bin/env bash

# source ROS
source /opt/ros/humble/setup.bash

# 执行后续命令（比如 bash，或 ros2 launch …）
exec "$@"