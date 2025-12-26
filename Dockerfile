# PointTransformerV3 Dockerfile
# Based on official PTv3 installation instructions
# https://github.com/Pointcept/PointTransformerV3
#
# Supports: RTX 3090 (8.6), RTX 4090 (8.9), RTX 5090 (10.0), A100 (8.0)

# =============================================================================
# Build Arguments
# =============================================================================
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.9.0

# =============================================================================
# Base Image - CUDA 13 with cuDNN
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# Re-declare ARGs after FROM
ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG CUDA_VERSION

# =============================================================================
# Environment Variables
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# GPU Architecture List:
# - 8.0: A100, A30
# - 8.6: RTX 3090, RTX 3080, A40
# - 8.9: RTX 4090, RTX 4080, L40
# - 10.0: RTX 5090, RTX 5080 (Blackwell)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0"
ENV MAX_JOBS=4

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Miniconda Installation
# =============================================================================
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# =============================================================================
# Create Conda Environment
# =============================================================================
RUN conda create -n ptv3 python=${PYTHON_VERSION} -y \
    && conda clean -afy

# Activate environment for subsequent commands
SHELL ["conda", "run", "-n", "ptv3", "/bin/bash", "-c"]

# =============================================================================
# PyTorch Installation (CUDA 13.0)
# =============================================================================
# PyTorch 2.9+ supports CUDA 13
RUN pip install torch==${PYTORCH_VERSION} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# =============================================================================
# Core Python Dependencies
# =============================================================================
RUN pip install --no-cache-dir \
    ninja \
    numpy \
    h5py \
    pyyaml \
    tensorboard \
    tensorboardx \
    yapf \
    addict \
    einops \
    scipy \
    plyfile \
    termcolor \
    timm \
    tqdm \
    wandb \
    open3d

# =============================================================================
# PyTorch Geometric and Extensions
# =============================================================================
# Build from source for CUDA 13 compatibility (prebuilt wheels may be ABI-incompatible)
# This takes ~20 minutes but ensures compatibility
ENV MAX_JOBS=8
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster

RUN pip install --no-cache-dir torch-geometric

# =============================================================================
# spconv (Sparse Convolutions)
# =============================================================================
# Try cu130 first, fallback to cu124, then cu120
RUN pip install --no-cache-dir spconv-cu130 2>/dev/null \
    || pip install --no-cache-dir spconv-cu124 2>/dev/null \
    || pip install --no-cache-dir spconv-cu120 \
    || echo "Warning: spconv installation failed, will need to build from source"

# =============================================================================
# Flash Attention (Required for PTv3 performance)
# =============================================================================
# Flash Attention 2.x supports Ampere (8.x), Ada (8.9), and Blackwell (10.x)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention installation failed - PTv3 will work without it (set enable_flash=False)"

# =============================================================================
# Working Directory
# =============================================================================
WORKDIR /workspace

# =============================================================================
# Copy PointTransformerV3 Code
# =============================================================================
COPY . /workspace/PointTransformerV3/

# =============================================================================
# Build pointops Extension
# =============================================================================
WORKDIR /workspace/PointTransformerV3/Pointcept/libs/pointops

RUN pip install -e . || python setup.py install

# =============================================================================
# Build pointops2 Extension (if exists)
# =============================================================================
WORKDIR /workspace/PointTransformerV3/Pointcept/libs/pointops2

RUN if [ -f "setup.py" ]; then pip install -e . || python setup.py install; fi

# =============================================================================
# Build pointgroup_ops Extension (if exists)
# =============================================================================
WORKDIR /workspace/PointTransformerV3/Pointcept/libs/pointgroup_ops

RUN if [ -f "setup.py" ]; then pip install -e . || python setup.py install; fi

# =============================================================================
# Additional Dependencies for CLDHits Instance Segmentation
# =============================================================================
WORKDIR /workspace

RUN pip install --no-cache-dir \
    pyarrow \
    pandas \
    awkward \
    scikit-learn \
    hdbscan \
    matplotlib \
    seaborn

# =============================================================================
# Set Default Working Directory
# =============================================================================
WORKDIR /workspace/PointTransformerV3

# =============================================================================
# Entrypoint
# =============================================================================
# Activate conda environment by default
RUN echo "conda activate ptv3" >> ~/.bashrc
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ptv3"]
CMD ["/bin/bash"]

# =============================================================================
# Usage Instructions
# =============================================================================
# Build:
#   docker build -t ptv3:latest .
#
# Run interactively:
#   docker run --gpus all -it --rm \
#     -v /path/to/data:/data \
#     -v /path/to/experiments:/experiments \
#     ptv3:latest /bin/bash
#
# Run training:
#   docker run --gpus all -it --rm \
#     -v /eventssl-vol:/eventssl-vol \
#     ptv3:latest python train_instance.py --config configs/cld_instance.yaml
#
# =============================================================================
# Supported GPUs
# =============================================================================
# GPU              | Compute Capability | Architecture
# -----------------|-------------------|---------------
# A100             | 8.0               | Ampere
# RTX 3090/3080    | 8.6               | Ampere
# RTX 4090/4080    | 8.9               | Ada Lovelace
# RTX 5090/5080    | 10.0              | Blackwell
#
# Notes:
# - CUDA 13.0 required for Blackwell (RTX 50xx) support
# - Flash Attention works on all supported architectures
# - Adjust MAX_JOBS based on available RAM (each job uses ~2GB during compilation)
