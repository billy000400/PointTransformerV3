# PTv3 Installation for PyTorch 2.9+ and CUDA 13.0

This guide adapts the official PTv3 installation for newer PyTorch and CUDA versions.

## Requirements

- Ubuntu 22.04+
- CUDA 13.0+
- PyTorch 2.9.0+
- Python 3.10+

## Environment Setup

```bash
# Option 1: Using conda (recommended)
conda create -n ptv3-cu13 python=3.11 -y
conda activate ptv3-cu13
conda install ninja -y

# Install PyTorch 2.9 with CUDA 13.0
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Option 2: Using existing environment
# Ensure you have PyTorch 2.9+ with CUDA 13 support
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Install Dependencies

```bash
# Core dependencies
pip install numpy h5py pyyaml tensorboard tensorboardx
pip install yapf addict einops scipy plyfile termcolor timm tqdm wandb

# Open3D (optional, for visualization)
pip install open3d
```

## PyTorch Geometric Extensions

For CUDA 13.0 / PyTorch 2.9, you need to build from source:

```bash
# Build torch-scatter, torch-sparse, torch-cluster from source
pip install torch-scatter torch-sparse torch-cluster

# This will compile for your CUDA version - takes ~10-20 minutes
# If you have issues, try:
pip install torch-scatter torch-sparse torch-cluster --no-cache-dir
```

## spconv (Sparse Convolutions)

```bash
# Try cu130 first, then fall back
pip install spconv-cu130 || pip install spconv-cu124 || pip install spconv-cu120
```

## Build pointops

```bash
cd /path/to/PointTransformerV3/Pointcept/libs/pointops

# Build CUDA extension
pip uninstall pointops -y 2>/dev/null
python setup.py install

# Verify
python -c "import pointops; from pointops._C import knn_query_cuda; print('OK')"
```

## Flash Attention (Optional but Recommended)

```bash
# For Ampere/Ada/Blackwell GPUs (sm80+)
pip install flash-attn --no-build-isolation

# If installation fails, PTv3 works without it (set enable_flash=False in config)
```

## CLDHits-specific Dependencies

```bash
pip install pyarrow pandas scikit-learn hdbscan matplotlib seaborn
```

## Verify Installation

```bash
cd /path/to/PointTransformerV3
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

import pointops
print('pointops: OK')

import spconv
print('spconv: OK')

import torch_scatter
print('torch_scatter: OK')

from model import PointTransformerV3
print('PTv3 model: OK')
"
```

## GPU Compatibility

| GPU | Compute Capability | Status |
|-----|-------------------|--------|
| A100 | 8.0 | Tested |
| RTX 3090 | 8.6 | Tested |
| RTX 4090 | 8.9 | Should work |
| RTX 5090 | 10.0 | Requires CUDA 13+ |

## Troubleshooting

### torch-scatter build fails
```bash
# Install with specific PyTorch/CUDA versions
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### spconv import error
```bash
# Try older CUDA version wheel
pip install spconv-cu120
```

### pointops build fails
```bash
# Set CUDA architectures manually
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;10.0"
python setup.py install
```
