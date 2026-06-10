# SAM 3D Objects Setup and Reconstruction Tools

Lightweight setup and reconstruction tooling for Meta's
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).

This repository does not vendor Meta's source code, model checkpoints, Kaolin,
or generated meshes. `setup.sh` restores the pinned official source and
downloads the gated checkpoints from Hugging Face.

## Requirements

- Linux x86-64
- NVIDIA GPU with approximately 32 GB or more VRAM
- CUDA 12.1
- Conda or Mamba
- Git
- Access to
  [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)

## Setup

```bash
git clone https://github.com/kev489/sam3d-setup.git
cd sam3d-setup
huggingface-cli login
./setup.sh
./test_setup.sh
```

On a module-based cluster:

```bash
./setup_cluster.sh
```

Setup supports:

```bash
./setup.sh --env-only
./setup.sh --ckpt-only
```

The official source is pinned to:

```text
facebookresearch/sam-3d-objects
f91db411c50efee93d8db7aeb323885650f6f722
```

## Custom Tools

- `scripts/reconstruct.py`: reconstruct one object and estimate metric depth.
- `scripts/batch_reconstruct.py`: reconstruct several object masks.
- `scripts/compose_scene.py`: combine Gaussian splats into one scene.
- `scripts/compose_mesh_scene.py`: combine exported meshes using pose metadata.

Example inputs are under `examples/`. See
[docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md) for commands.

## Recovery

See [RESTORE.md](RESTORE.md) for a complete description of how to recreate
source, checkpoints, dependencies, examples, and generated outputs after a
machine cleanup.

## Storage Policy

Do not commit:

- `sam-3d-objects/`
- `sam-3d-objects/checkpoints/`
- Conda environments
- Generated `.ply`, `.obj`, or `.glb` files

These are reproducible or downloadable and are excluded by `.gitignore`.

## License

Meta's SAM 3D source and model are governed by the included SAM License. The
custom scripts retain their upstream copyright notices where applicable.
