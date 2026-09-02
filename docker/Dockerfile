# =============================================================================
# ABot-World Docker image
#
# Follows the installation steps in README.md (Setup section):
#   Ubuntu 22.04 + CUDA 12.8 + Python 3.12 + PyTorch 2.8.0 (cu128)
#   + FlashAttention 2.8.1 (prebuilt wheel)
#   + SageAttention (built from source)
#   + lightx2v_kernel (built from source, requires CUTLASS)
#
# Build (run from the repository root):
#   docker build -f docker/Dockerfile -t abot-world:latest .
# or simply:
#   bash docker/build.sh
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# CUDA architectures for SageAttention (works without a GPU at build time).
#   8.6 -> RTX 3090, 8.9 -> RTX 4090, 9.0 -> H100/H20, 12.0 -> RTX 5090
# Note: lightx2v_kernel is compiled for sm_120a (Blackwell) only, as upstream.
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"
# Parallel compile jobs. CAUTION: SageAttention's build parallelism multiplies
# (EXT_PARALLEL x MAX_JOBS ninja jobs x nvcc --threads), and each compile unit
# can peak at 3-6 GB RAM. The defaults below keep peak usage under ~30 GB.
ARG MAX_JOBS=4
# Miniconda installer URL; override for faster mirrors, e.g. TUNA:
#   https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential ninja-build \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Conda environment (Python 3.12, as in README) ────────────────────────────
RUN wget --progress=dot:giga "${MINICONDA_URL}" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    # Accept Anaconda channel ToS non-interactively (no-op on older conda)
    && { /opt/conda/bin/conda tos accept --override-channels \
           --channel https://repo.anaconda.com/pkgs/main \
           --channel https://repo.anaconda.com/pkgs/r || true; } \
    && /opt/conda/bin/conda create -n aworld python=3.12 -y \
    && /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/envs/aworld/bin:/opt/conda/bin:$PATH

# ── PyTorch (CUDA 12.8) ──────────────────────────────────────────────────────
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ── FlashAttention (prebuilt wheel) ──────────────────────────────────────────
# NOTE: keep the original wheel filename; pip rejects renamed wheels.
RUN wget --progress=dot:giga -P /tmp \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
    && pip install "/tmp/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
    && rm /tmp/flash_attn-*.whl

# ── SageAttention (from source) ──────────────────────────────────────────────
RUN pip install packaging ninja \
    && git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && cd /tmp/SageAttention \
    # EXT_PARALLEL=1: build extensions sequentially; --threads 2 overrides the
    # hardcoded --threads=8 in setup.py to bound per-nvcc memory.
    && EXT_PARALLEL=1 NVCC_APPEND_FLAGS="--threads 2" MAX_JOBS=${MAX_JOBS} \
       TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
       python setup.py install \
    && rm -rf /tmp/SageAttention

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ── lightx2v_kernel (from source, requires CUTLASS) ──────────────────────────
RUN pip install uv \
    && git clone --depth 1 https://github.com/NVIDIA/cutlass.git /tmp/cutlass \
    && git clone --depth 1 https://github.com/ModelTC/LightX2V.git /tmp/LightX2V \
    && cd /tmp/LightX2V/lightx2v_kernel \
    && MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} \
       uv build --wheel \
           -Cbuild-dir=build . \
           -Ccmake.define.CUTLASS_PATH=/tmp/cutlass \
           --color=always \
           --no-build-isolation \
    && pip install dist/*.whl --force-reinstall --no-deps \
    && rm -rf /tmp/cutlass /tmp/LightX2V

# ── Checkpoint download tools (used at runtime, see docker/README.md) ────────
RUN pip install -U huggingface_hub modelscope

# ── Project mount point ─────────────────────────────────────────────────────────────
# The image contains only the runtime environment; the ABot-World repository
# is mounted here at runtime (see docker/run.sh).
WORKDIR /workspace/ABot-World

# Gradio demo port (web_client/config.py: SERVER_PORT = 2233)
EXPOSE 2233

CMD ["bash", "web_client/run.sh"]
