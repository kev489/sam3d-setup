# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
reconstruct.py — Single-image 3D reconstruction with metric depth estimation.

Usage:
    python reconstruct.py --image <path> --mask <path> --real-size-mm 57 --output-ply cube.ply

Arguments:
    --image         Path to the input image (RGB or RGBA, any standard format)
    --mask          Path to the binary mask (white = object, black = background)
    --real-size-mm  Real-world size of the object's largest visible dimension in mm (default: 57 for Rubik's cube)
    --output-ply    Path to save the gaussian splat .ply file (optional)
    --seed          Random seed for reproducibility (default: 42)
"""

import os
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
import argparse
import numpy as np
from PIL import Image

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sam3d_dir = os.path.join(_repo_dir, "sam-3d-objects")
sys.path.insert(0, _sam3d_dir)
sys.path.insert(0, os.path.join(_sam3d_dir, "notebook"))
from inference import Inference, load_image, load_single_mask, load_mask

DEFAULT_CHECKPOINT = os.path.join(_sam3d_dir, "checkpoints", "hf", "pipeline.yaml")


def load_mask_from_path(mask_path):
    mask = load_mask(mask_path)
    return mask


def estimate_metric_depth(pointmap, mask, real_size_mm):
    """
    Use a known object size to recover metric depth from MoGe pointmap.

    Args:
        pointmap: HxWx3 tensor of 3D points in camera space (MoGe units)
        mask: HxW boolean numpy array (True = object pixels)
        real_size_mm: known real-world size of the object in mm

    Returns:
        dict with metric estimates
    """
    import torch

    # resize mask to pointmap resolution if needed
    ph, pw = pointmap.shape[:2]
    if mask.shape != (ph, pw):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.array(mask_img.resize((pw, ph), Image.NEAREST)) > 0

    # extract 3D points belonging to the object
    obj_pts = pointmap[mask]  # Nx3

    mins = obj_pts.min(dim=0).values
    maxs = obj_pts.max(dim=0).values
    extent = maxs - mins  # X, Y, Z extents in MoGe units

    # calibrate using X/Y face extents (more reliable than Z for a frontal view)
    face_extent = max(extent[0].item(), extent[1].item())
    real_size_m = real_size_mm / 1000.0
    scale_factor = real_size_m / face_extent  # meters per MoGe unit

    mean_z = obj_pts[:, 2].mean().item()
    median_z = obj_pts[:, 2].median().item()
    mean_x = obj_pts[:, 0].mean().item()
    mean_y = obj_pts[:, 1].mean().item()

    return {
        "distance_cm": round(median_z * scale_factor * 100, 1),
        "x_offset_cm": round(mean_x * scale_factor * 100, 1),
        "y_offset_cm": round(mean_y * scale_factor * 100, 1),
        "scale_factor": scale_factor,
        "extent_x_mm": round(extent[0].item() * scale_factor * 1000, 1),
        "extent_y_mm": round(extent[1].item() * scale_factor * 1000, 1),
        "extent_z_mm": round(extent[2].item() * scale_factor * 1000, 1),
        "moge_units": {
            "mean_z": round(mean_z, 4),
            "x_extent": round(extent[0].item(), 4),
            "y_extent": round(extent[1].item(), 4),
            "z_extent": round(extent[2].item(), 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="3D reconstruction with metric depth")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to binary mask")
    parser.add_argument("--real-size-mm", type=float, default=57.0, help="Known object size in mm (default: 57 for Rubik's cube)")
    parser.add_argument("--output-ply", default=None, help="Path to save .ply gaussian splat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to pipeline config")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    inference = Inference(args.checkpoint, compile=False)

    print(f"Loading image: {args.image}")
    image = load_image(args.image)

    print(f"Loading mask: {args.mask}")
    mask = load_mask_from_path(args.mask)

    # ensure image and mask are the same size
    if image.shape[:2] != mask.shape[:2]:
        print(f"Resizing image from {image.shape[:2]} to match mask {mask.shape[:2]}")
        img_pil = Image.fromarray(image).resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        image = np.array(img_pil)

    print("Running inference...")
    output = inference(image, mask, seed=args.seed)

    print("\n--- 3D Reconstruction Results ---")
    print(f"Object rotation (quaternion): {output['rotation'].squeeze().tolist()}")

    depth = estimate_metric_depth(output["pointmap"], mask, args.real_size_mm)

    print(f"\nMetric estimates (calibrated using known size = {args.real_size_mm}mm):")
    print(f"  Distance from camera:  {depth['distance_cm']} cm")
    print(f"  Lateral offset (X):    {depth['x_offset_cm']} cm  (+ = right)")
    print(f"  Vertical offset (Y):   {depth['y_offset_cm']} cm  (+ = down)")
    print(f"  Measured size X:       {depth['extent_x_mm']} mm")
    print(f"  Measured size Y:       {depth['extent_y_mm']} mm")
    print(f"  Measured size Z:       {depth['extent_z_mm']} mm")

    if args.output_ply:
        output["gs"].save_ply(args.output_ply)
        print(f"\nGaussian splat saved to: {args.output_ply}")

    return depth


if __name__ == "__main__":
    main()
