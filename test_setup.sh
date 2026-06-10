#!/bin/bash
set -e

# Quick test to verify SAM 3D Objects is installed correctly.
# Run after setup.sh completes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM3D_DIR="$SCRIPT_DIR/sam-3d-objects"

echo "=== SAM 3D Objects - Setup Verification ==="
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate sam3d-objects

cd "$SAM3D_DIR"

echo "1. Checking Python version..."
python --version

echo ""
echo "2. Checking PyTorch + CUDA..."
python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "3. Checking core imports..."
python -c "
import sys
sys.path.append('notebook')
from inference import Inference, load_image, load_single_mask
print('   All core imports OK')
"

echo ""
echo "4. Checking checkpoints..."
TAG=hf
if [ -f "checkpoints/${TAG}/pipeline.yaml" ]; then
    echo "   Checkpoints found at checkpoints/${TAG}/"
    echo "   pipeline.yaml exists"
else
    echo "   WARNING: Checkpoints not found at checkpoints/${TAG}/pipeline.yaml"
    echo "   Run: ./setup.sh --ckpt-only"
fi

echo ""
echo "5. Checking custom reconstruction tools..."
for script in \
    "$SCRIPT_DIR/scripts/batch_reconstruct.py" \
    "$SCRIPT_DIR/scripts/reconstruct.py" \
    "$SCRIPT_DIR/scripts/compose_scene.py" \
    "$SCRIPT_DIR/scripts/compose_mesh_scene.py"; do
    test -f "$script"
    python -m py_compile "$script"
done
echo "   Custom tools found and syntax-checked"

echo ""
echo "=== Verification Complete ==="
echo ""
echo "If all checks passed, run the demo:"
echo "  cd $SAM3D_DIR && python demo.py"
