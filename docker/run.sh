#!/usr/bin/env bash
#
# Run the ABot-World Gradio demo inside Docker.
#
# Usage:
#   bash docker/run.sh                       # start the Gradio demo on GPU 0
#   CUDA_ID=1 bash docker/run.sh             # select a GPU
#   PORT=8080 bash docker/run.sh             # map a different host port
#   bash docker/run.sh bash                  # drop into an interactive shell
#
# Requirements: Docker with the NVIDIA Container Toolkit installed.
# The image contains only the runtime environment: this script mounts the
# whole repository (including ./checkpoints) into the container, so it must
# be run from a local clone of ABot-World (see docker/README.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="${IMAGE:-abot-world:latest}"
PORT="${PORT:-2233}"
CUDA_ID="${CUDA_ID:-0}"

mkdir -p "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/outputs" "$HOME/.cache/huggingface"

echo "=== ABot-World Docker ==="
echo "  Image: $IMAGE"
echo "  GPU:   CUDA_ID=$CUDA_ID"
echo "  Port:  http://localhost:$PORT"
echo "========================="

docker run --rm -it \
    --gpus all \
    --ipc=host \
    --shm-size=32GB \
    --ulimit nofile=65536:65536 \
    -e CUDA_ID="$CUDA_ID" \
    -v "$PROJECT_ROOT:/workspace/ABot-World" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -p "$PORT:2233" \
    "$IMAGE" \
    "$@"
