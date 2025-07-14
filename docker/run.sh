#!/usr/bin/env bash
set -e

IMAGE="co-vins-ros2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# 解析选项
FORCE_BUILD=false
while getopts "b" opt; do
  case "$opt" in
    b) FORCE_BUILD=true ;;
    *) ;;
  esac
done

# 1) 构建镜像（可通过 -b 强制构建）
if $FORCE_BUILD || [[ -z "$(docker images -q ${IMAGE}:latest 2> /dev/null)" ]]; then
  echo "正在构建镜像 ${IMAGE}:latest …"
  # 尝试从现有同名镜像读取缓存
  docker build \
    --cache-from ${IMAGE}:latest \
    -t ${IMAGE}:latest \
    "$SCRIPT_DIR"
else
  echo "镜像 ${IMAGE}:latest 已存在，跳过构建"
fi

# 2) 切到工作区，执行依赖安装和编译
docker run -it --rm \
  -v "${REPO_ROOT}":/workspace \
  -w /workspace \
  ${IMAGE}:latest \
  bash -c "\
    source /opt/ros/foxy/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install && \
    source install/setup.bash && \
    ros2 launch your_package your_launch.py\
  "

