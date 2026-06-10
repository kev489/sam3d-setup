#!/bin/bash
set -e

# SAM 3D Objects - Full Setup Script
# Run this on a Linux machine with NVIDIA GPU (32GB+ VRAM) and mamba/conda installed.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # full setup (env + checkpoints)
#   ./setup.sh --env-only   # just create the environment, skip checkpoint download
#   ./setup.sh --ckpt-only  # just download checkpoints (env already exists)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM3D_DIR="$SCRIPT_DIR/sam-3d-objects"
UPSTREAM_REPO="https://github.com/facebookresearch/sam-3d-objects.git"
UPSTREAM_COMMIT="f91db411c50efee93d8db7aeb323885650f6f722"

ENV_ONLY=false
CKPT_ONLY=false

for arg in "$@"; do
    case $arg in
        --env-only)  ENV_ONLY=true ;;
        --ckpt-only) CKPT_ONLY=true ;;
    esac
done

# ---- Preflight checks ----
echo "=== SAM 3D Objects Setup ==="
echo ""

if ! command -v git &> /dev/null; then
    echo "ERROR: git is required."
    exit 1
fi

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: SAM 3D Objects requires Linux. You are on $(uname)."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. An NVIDIA GPU is required."
    exit 1
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "Detected GPU memory: ${GPU_MEM} MiB"
if (( GPU_MEM < 30000 )); then
    echo "WARNING: SAM 3D requires 32GB+ VRAM. You have ${GPU_MEM} MiB. Proceeding anyway..."
fi

# Check for mamba or conda
if command -v mamba &> /dev/null; then
    CONDA_CMD=mamba
elif command -v conda &> /dev/null; then
    CONDA_CMD=conda
else
    echo "ERROR: Neither mamba nor conda found. Install miniforge/miniconda first."
    echo "  Quick install: curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3-Linux-x86_64.sh"
    exit 1
fi
echo "Using: $CONDA_CMD"

# ---- Official source checkout ----
echo ""
echo "=== Step 0: Restoring official SAM 3D Objects source ==="

if [[ ! -d "$SAM3D_DIR/.git" ]]; then
    git clone "$UPSTREAM_REPO" "$SAM3D_DIR"
fi

if [[ -n "$(git -C "$SAM3D_DIR" status --porcelain)" ]]; then
    echo "ERROR: $SAM3D_DIR has local changes."
    echo "Move or commit them before running setup so they are not overwritten."
    exit 1
fi

git -C "$SAM3D_DIR" fetch origin "$UPSTREAM_COMMIT"
git -C "$SAM3D_DIR" checkout --detach "$UPSTREAM_COMMIT"

# ---- Environment setup ----
if [[ "$CKPT_ONLY" == false ]]; then
    echo ""
    echo "=== Step 1: Creating conda environment ==="
    cd "$SAM3D_DIR"

    $CONDA_CMD env create -f environments/default.yml || {
        echo "Environment may already exist. Trying to update..."
        $CONDA_CMD env update -f environments/default.yml
    }

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate sam3d-objects

    echo ""
    echo "=== Step 2: Installing dependencies ==="

    export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

    pip install -e '.[dev]'
    pip install -e '.[p3d]'

    echo ""
    echo "=== Step 3: Installing inference dependencies ==="

    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install -e '.[inference]'

    echo ""
    echo "=== Step 4: Applying patches ==="
    chmod +x patching/hydra
    ./patching/hydra

    echo ""
    echo "Environment setup complete!"
fi

# ---- Checkpoint download ----
if [[ "$ENV_ONLY" == false ]]; then
    echo ""
    echo "=== Step 5: Downloading model checkpoints ==="

    # Make sure we're in the right dir and env
    cd "$SAM3D_DIR"
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate sam3d-objects 2>/dev/null || true

    pip install 'huggingface-hub[cli]<1.0'

    # Check HuggingFace auth
    if ! huggingface-cli whoami &> /dev/null; then
        echo ""
        echo "You need to authenticate with HuggingFace first."
        echo "1. Create an access token at https://huggingface.co/settings/tokens"
        echo "2. Request access to the model at https://huggingface.co/facebook/sam-3d-objects"
        echo "3. Run: huggingface-cli login"
        echo ""
        echo "After authenticating, re-run: ./setup.sh --ckpt-only"
        exit 1
    fi

    TAG=hf
    mkdir -p checkpoints

    echo "Downloading checkpoints (this may take a while)..."
    huggingface-cli download \
        --repo-type model \
        --local-dir "checkpoints/${TAG}-download" \
        --max-workers 1 \
        facebook/sam-3d-objects

    mv "checkpoints/${TAG}-download/checkpoints" "checkpoints/${TAG}"
    rm -rf "checkpoints/${TAG}-download"

    echo ""
    echo "Checkpoints downloaded to: $SAM3D_DIR/checkpoints/$TAG/"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To use SAM 3D Objects:"
echo "  conda activate sam3d-objects"
echo "  cd $SAM3D_DIR"
echo "  python demo.py"
echo ""
echo "Or run the test script:"
echo "  ./test_setup.sh"
