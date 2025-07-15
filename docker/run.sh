#!/usr/bin/env bash
set -euo pipefail


IMAGE_NAME="co-vins-ros2"

# Dockerfile 路径（相对于脚本所在目录）
DOCKERFILE_PATH="docker/dockerfile"

# 构建镜像
echo "  Building Docker image ${IMAGE_NAME}..."
docker build -f "${DOCKERFILE_PATH}" -t "${IMAGE_NAME}" .

# 运行容器
# 如果传入参数，就把参数当成容器内部要执行的命令；否则进入交互式 shell
if [ $# -gt 0 ]; then
  echo "  Running container and executing: $*"
  docker run --rm -it \
    --network host \
    --privileged \
    "${IMAGE_NAME}" \
    "$@"
else
  echo "  Running container (interactive shell)"
  docker run --rm -it \
    --network host \
    --privileged \
    "${IMAGE_NAME}"
fi

