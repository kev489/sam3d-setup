#!/bin/bash
set -e

# Load common MIT cluster modules, then delegate to the portable setup script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== SAM 3D Objects Cluster Setup ==="
echo "Loading modules..."
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module was not found"
module load anaconda3 2>/dev/null || \
    module load miniconda 2>/dev/null || \
    echo "WARN: no conda module was found"

exec "$SCRIPT_DIR/setup.sh" "$@"
