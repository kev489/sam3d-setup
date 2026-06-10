# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
batch_reconstruct.py — Batch 3D reconstruction producing mesh (.glb), gaussian splat (.ply),
and metadata (.json) for each object mask.

Usage:
    python batch_reconstruct.py --image scene.jpg --masks airpods.png marker.png snowCone.png \
        --names airpods marker snowCone --output-dir outputs/ --seed 42

Each object gets: <name>.ply, <name>.glb, <name>.json
"""

import os
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
import json
import argparse
import numpy as np
from PIL import Image

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sam3d_dir = os.path.join(_repo_dir, "sam-3d-objects")
_notebook_dir = os.path.join(_sam3d_dir, "notebook")
sys.path.insert(0, _sam3d_dir)
sys.path.insert(0, _notebook_dir)
from inference import Inference, load_image, load_mask, WHITELIST_FILTERS, BLACKLIST_FILTERS, check_hydra_safety

DEFAULT_CHECKPOINT = os.path.join(_sam3d_dir, "checkpoints", "hf", "pipeline.yaml")


def load_pipeline(checkpoint):
    """Load the full pipeline directly (not the Inference wrapper) so we can enable mesh postprocessing."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    config = OmegaConf.load(checkpoint)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = os.path.dirname(checkpoint)

    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)

    pipeline = instantiate(config)
    return pipeline


def merge_mask_to_rgba(image, mask):
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = mask_u8[..., None]
    return np.concatenate([image[..., :3], mask_u8], axis=-1)


def estimate_metric_depth(pointmap, mask, real_size_mm):
    """Use a known object size to recover metric depth from MoGe pointmap."""
    ph, pw = pointmap.shape[:2]
    if mask.shape != (ph, pw):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.array(mask_img.resize((pw, ph), Image.NEAREST)) > 0

    obj_pts = pointmap[mask]  # Nx3
    mins = obj_pts.min(dim=0).values
    maxs = obj_pts.max(dim=0).values
    extent = maxs - mins

    face_extent = max(extent[0].item(), extent[1].item())
    real_size_m = real_size_mm / 1000.0
    scale_factor = real_size_m / face_extent

    median_z = obj_pts[:, 2].median().item()
    mean_x = obj_pts[:, 0].mean().item()
    mean_y = obj_pts[:, 1].mean().item()

    return {
        "distance_cm": round(median_z * scale_factor * 100, 1),
        "x_offset_cm": round(mean_x * scale_factor * 100, 1),
        "y_offset_cm": round(mean_y * scale_factor * 100, 1),
        "scale_factor": float(scale_factor),
        "extent_x_mm": round(extent[0].item() * scale_factor * 1000, 1),
        "extent_y_mm": round(extent[1].item() * scale_factor * 1000, 1),
        "extent_z_mm": round(extent[2].item() * scale_factor * 1000, 1),
    }


def run_single_object(pipeline, image, mask, seed, real_size_mm):
    """Run inference for one object, returning all outputs."""
    import torch

    rgba = merge_mask_to_rgba(image, mask)

    output = pipeline.run(
        rgba,
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
        stage1_inference_steps=None,
        pointmap=None,
    )

    # Metadata
    rotation = output["rotation"].squeeze().tolist()
    translation = output["translation"].squeeze().tolist()
    scale = output["scale"].squeeze().tolist()

    meta = {
        "rotation_quaternion": rotation,
        "translation": translation,
        "scale": scale,
        "seed": seed,
    }

    if real_size_mm is not None:
        depth_info = estimate_metric_depth(output["pointmap"], mask, real_size_mm)
        meta["metric_depth"] = depth_info

    return output, meta


def save_mesh_obj(mesh_result, path):
    """Export a MeshExtractResult as a Wavefront OBJ file."""
    vertices = mesh_result.vertices.float().cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_outputs(output, meta, output_dir, name):
    """Save .ply, .obj, .glb (if available), and .json for one object."""
    os.makedirs(output_dir, exist_ok=True)

    # Gaussian splat PLY
    ply_path = os.path.join(output_dir, f"{name}.ply")
    output["gs"].save_ply(ply_path)
    print(f"  Saved gaussian splat: {ply_path}")

    # Mesh OBJ (raw mesh from decoder)
    if "mesh" in output and output["mesh"] is not None:
        mesh_list = output["mesh"]
        mesh_obj = mesh_list[0] if isinstance(mesh_list, list) else mesh_list
        obj_path = os.path.join(output_dir, f"{name}.obj")
        save_mesh_obj(mesh_obj, obj_path)
        print(f"  Saved mesh OBJ:       {obj_path}")
    else:
        print(f"  WARNING: No mesh output available for {name}")

    # Mesh GLB (if postprocessing was enabled)
    if output.get("glb") is not None:
        glb_path = os.path.join(output_dir, f"{name}.glb")
        output["glb"].export(glb_path)
        print(f"  Saved mesh GLB:       {glb_path}")

    # Metadata JSON
    json_path = os.path.join(output_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata:       {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch 3D reconstruction: mesh + gaussian splat + metadata")
    parser.add_argument("--image", required=True, help="Path to base scene image")
    parser.add_argument("--masks", nargs="+", required=True, help="Paths to binary mask images (one per object)")
    parser.add_argument("--names", nargs="+", required=True, help="Names for each object (used in output filenames)")
    parser.add_argument("--real-sizes-mm", nargs="+", type=float, default=None, help="Known real sizes in mm (one per object, optional)")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (same for all objects)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to pipeline config")
    args = parser.parse_args()

    assert len(args.masks) == len(args.names), "Number of masks must match number of names"
    if args.real_sizes_mm:
        assert len(args.real_sizes_mm) == len(args.names), "Number of real-sizes must match number of names"

    print(f"Loading pipeline from {args.checkpoint}...")
    pipeline = load_pipeline(args.checkpoint)

    print(f"Loading base image: {args.image}")
    image = load_image(args.image)

    for i, (mask_path, name) in enumerate(zip(args.masks, args.names)):
        print(f"\n{'='*60}")
        print(f"Processing object {i+1}/{len(args.names)}: {name}")
        print(f"{'='*60}")

        print(f"Loading mask: {mask_path}")
        mask = load_mask(mask_path)

        # Resize image to mask if needed
        img = image.copy()
        if img.shape[:2] != mask.shape[:2]:
            print(f"Resizing image from {img.shape[:2]} to match mask {mask.shape[:2]}")
            img_pil = Image.fromarray(img).resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
            img = np.array(img_pil)

        real_size = args.real_sizes_mm[i] if args.real_sizes_mm else None

        print("Running inference...")
        output, meta = run_single_object(pipeline, img, mask, args.seed, real_size)

        meta["source_image"] = os.path.basename(args.image)
        meta["mask_file"] = os.path.basename(mask_path)
        meta["object_name"] = name

        save_outputs(output, meta, args.output_dir, name)

    print(f"\n{'='*60}")
    print(f"Done! All outputs saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
