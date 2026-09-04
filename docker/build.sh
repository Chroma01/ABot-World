#!/usr/bin/env bash
#
# Build the ABot-World Docker image.
#
# Usage:
#   bash docker/build.sh                             # default image tag abot-world:latest
#   IMAGE=abot-world:v0.1 bash docker/build.sh       # custom image tag
#   TORCH_CUDA_ARCH_LIST="12.0" bash docker/build.sh # build SageAttention for RTX 5090 only (faster)
#   MINICONDA_URL=<mirror-url> bash docker/build.sh  # use a Miniconda mirror (e.g. TUNA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="${IMAGE:-abot-world:latest}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9;9.0;12.0}"
# CAUTION: SageAttention's build parallelism multiplies (see docker/Dockerfile);
# raise MAX_JOBS only on hosts with plenty of RAM.
MAX_JOBS="${MAX_JOBS:-4}"

echo "=== Building ABot-World Docker image ==="
echo "  Image:               $IMAGE"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS:            $MAX_JOBS"
echo "========================================="

DOCKER_BUILDKIT=1 docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    ${MINICONDA_URL:+--build-arg MINICONDA_URL="$MINICONDA_URL"} \
    ${HTTP_PROXY:+--build-arg HTTP_PROXY="$HTTP_PROXY"} \
    ${HTTPS_PROXY:+--build-arg HTTPS_PROXY="$HTTPS_PROXY"} \
    ${NO_PROXY:+--build-arg NO_PROXY="$NO_PROXY"} \
    -t "$IMAGE" \
    "$PROJECT_ROOT"

echo "Done. Run the demo with: bash docker/run.sh"
