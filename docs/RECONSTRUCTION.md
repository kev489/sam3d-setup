# Reconstruction Runbook

Run these commands from the root of this repository after completing setup.

## Input Rules

- Use one binary mask per object.
- White pixels represent the object; black pixels represent the background.
- Grayscale or RGB masks are supported.
- Avoid RGBA masks whose alpha channel is solid white.

Convert an RGBA mask to grayscale when needed:

```bash
python -c "from PIL import Image; import numpy as np; img=np.array(Image.open('mask.png')); Image.fromarray((img[:,:,0]>0).astype(np.uint8)*255).save('mask.png')"
```

## Batch Reconstruction

```bash
conda run -n sam3d-objects python scripts/batch_reconstruct.py \
  --image examples/rob_scene.jpg \
  --masks examples/airpods.png examples/marker.png examples/snowCone.png \
  --names airpods marker snowCone \
  --output-dir examples/outputs \
  --seed 42
```

With known object sizes:

```bash
conda run -n sam3d-objects python scripts/batch_reconstruct.py \
  --image examples/rob_scene.jpg \
  --masks examples/airpods.png examples/marker.png examples/snowCone.png \
  --names airpods marker snowCone \
  --real-sizes-mm 55 140 100 \
  --output-dir examples/outputs \
  --seed 42
```

Each object can produce:

- `<name>.ply`: Gaussian splat
- `<name>.obj`: raw mesh
- `<name>.glb`: colored mesh when available
- `<name>.json`: pose and optional metric-depth metadata

## Single Object

```bash
conda run -n sam3d-objects python scripts/reconstruct.py \
  --image examples/rob_scene.jpg \
  --mask examples/airpods.png \
  --real-size-mm 55 \
  --output-ply examples/outputs/airpods.ply \
  --seed 42
```

## Gaussian Scene Composition

```bash
conda run -n sam3d-objects python scripts/compose_scene.py \
  --image examples/rob_scene.jpg \
  --masks examples/airpods.png examples/marker.png examples/snowCone.png \
  --names airpods marker snowCone \
  --output-dir examples/outputs \
  --seed 42 \
  --render-gif
```

This produces posed and normalized scene splats plus an optional turntable GIF.

## Mesh Scene Composition

After batch reconstruction:

```bash
conda run -n sam3d-objects python scripts/compose_mesh_scene.py \
  --input-dir examples/outputs \
  --names airpods marker snowCone \
  --output-dir examples/outputs
```

The pose transform uses PyTorch3D's row-vector convention: `vertices @ R`.

## Troubleshooting

- `CUDA OOM`: reconstruct one object at a time.
- Missing checkpoints: run `./setup.sh --ckpt-only`.
- Missing source checkout: run `./setup.sh --env-only`.
- Incorrect full-scene mask: convert RGBA masks to grayscale.
- Missing GLB output: install the optional mesh-processing dependencies.
