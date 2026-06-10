# Restore This Project

This document describes how to recreate the complete SAM 3D working
environment after deleting all local source checkouts, model checkpoints,
build products, and generated reconstruction outputs.

## 1. Clone This Repository

```bash
git clone https://github.com/kev489/sam3d-setup.git
cd sam3d-setup
```

This restores the setup scripts, custom reconstruction tools, runbook, and
small example inputs.

## 2. Obtain Model Access

Request access to:

```text
https://huggingface.co/facebook/sam-3d-objects
```

Create a Hugging Face token and authenticate:

```bash
huggingface-cli login
```

## 3. Restore Source, Environment, and Checkpoints

On a Linux GPU machine:

```bash
./setup.sh
./test_setup.sh
```

On a module-based cluster:

```bash
./setup_cluster.sh
```

The setup script:

1. Clones `https://github.com/facebookresearch/sam-3d-objects.git`.
2. Checks out commit `f91db411c50efee93d8db7aeb323885650f6f722`.
3. Creates or updates the `sam3d-objects` Conda environment.
4. Installs PyTorch3D, Kaolin, and inference dependencies.
5. Applies the required Hydra patch.
6. Downloads checkpoints to `sam-3d-objects/checkpoints/hf/`.

To restore only one part:

```bash
./setup.sh --env-only
./setup.sh --ckpt-only
```

## 4. Recreate Example Outputs

Run the commands in [docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md).
Generated files are written under `examples/outputs/` and intentionally are not
stored in Git.

The sample scene and masks are:

```text
examples/rob_scene.jpg
examples/airpods.png
examples/marker.png
examples/snowCone.png
```

Using seed `42` provides repeatable results for a fixed source revision,
environment, image, and masks.

## 5. Optional Kaolin Source Checkout

Normal setup installs Kaolin from NVIDIA's package index. A separate Kaolin
source checkout is not required.

To reproduce the historical source checkout used during development:

```bash
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
git checkout ba9824e394b074099fbae7d5218e68c6362e9ecf
```

Do not commit Kaolin build directories or compiled `.so` files.

## 6. Historical Backup

An older snapshot was also uploaded to the private
`MIT-CLEAR-Lab/Shared-Autonomy-Demo` repository at commit:

```text
a5c31b40cfbd7d3af44a0c063e838cf6cb76a3b3
```

That snapshot contains generated outputs and a broken Kaolin submodule pointer.
Use this repository, official upstream source, and Hugging Face checkpoints as
the canonical recovery path.
