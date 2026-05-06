#!/bin/bash
# Setup script to initialize the tdmpc2-codex environment correctly.
# This prevents the sequence of missing dependency and path errors encountered during remote deployments.

set -e

echo "================================================="
echo "TD-MPC2 Environment Setup"
echo "================================================="

# 0. Set PYTHONPATH to include the current directory for correct imports
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH set to $PYTHONPATH"

# 1. Install OS-level graphics dependencies required for EGL rendering by dm-control and mujoco
echo "[1/4] Installing OS-level OpenGL dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libosmesa6-dev libglew-dev patchelf mesa-utils
else
    echo "Warning: apt-get not found. Ensure OS-level OpenGL libraries are installed."
fi

# 2. Install correct PyTorch version depending on the GPU architecture (sm_120 for RTX 50-series)
echo "[2/4] Installing PyTorch..."
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1 || echo "None")
PYTHON_EXEC=$(which python3)
echo "Using Python: $PYTHON_EXEC"

if [[ "$GPU_NAME" == *"RTX 50"* ]]; then
    echo "Detected RTX 50-series: $GPU_NAME. Installing cu132 nightly build to support sm_120..."
    $PYTHON_EXEC -m pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
else
    echo "Detected $GPU_NAME. Installing standard cu124 stable build..."
    $PYTHON_EXEC -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# 3. Install project Python dependencies
echo "[3/4] Installing Python dependencies..."
$PYTHON_EXEC -m pip install -q wandb hydra-core omegaconf termcolor tensordict torchrl gymnasium mujoco imageio imageio-ffmpeg moviepy pandas tqdm kornia hydra-submitit-launcher submitit absl-py matplotlib
$PYTHON_EXEC -m pip install -q dm-control

# 4. Patch config.yaml to remove submitit_local plugin requirement
echo "[4/4] Patching config.yaml for local execution..."
if [ -f "config.yaml" ]; then
    # Use python to safely modify YAML and avoid parser errors
    python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
if 'defaults' in cfg:
    del cfg['defaults']
with open('config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
    echo "Patched config.yaml successfully."
else
    echo "Warning: config.yaml not found in current directory."
fi

# 5. Environment Probe
echo "-------------------------------------------------"
echo "Running Environment Probe..."
python3 -c "
import torch
print(f'Torch CUDA available: {torch.cuda.is_available()}')
try:
    import matplotlib.pyplot as plt
    print('matplotlib imported successfully')
    from dm_control import suite
    print('dm_control.suite imported successfully')
    env = suite.load(\"acrobot\", \"swingup\")
    print('acrobot-swingup loaded successfully')
except Exception as e:
    print(f'Environment Probe FAILED: {e}')
    import sys; sys.exit(1)
"

echo "================================================="
echo "Setup Complete."
echo "================================================="
