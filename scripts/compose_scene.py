# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
compose_scene.py — Compose multiple object reconstructions into a single scene.

Runs inference per-object, then uses make_scene() to place all gaussians
in the shared camera frame and exports the combined scene as a .ply.

Usage:
    python compose_scene.py --image <scene.jpg> --masks <m1.png> <m2.png> ... --names <n1> <n2> ... --output-dir <dir> --seed 42
"""

import os
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
import json
import numpy as np
from PIL import Image

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sam3d_dir = os.path.join(_repo_dir, "sam-3d-objects")
_notebook_dir = os.path.join(_sam3d_dir, "notebook")
sys.path.insert(0, _sam3d_dir)
sys.path.insert(0, _notebook_dir)
from inference import Inference, load_image, load_mask, make_scene, ready_gaussian_for_video_rendering, render_video

DEFAULT_CHECKPOINT = os.path.join(_sam3d_dir, "checkpoints", "hf", "pipeline.yaml")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compose multi-object scene from individual masks")
    parser.add_argument("--image", required=True, help="Path to base scene image")
    parser.add_argument("--masks", nargs="+", required=True, help="Paths to binary mask images")
    parser.add_argument("--names", nargs="+", required=True, help="Names for each object")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--render-gif", action="store_true", help="Also render a turntable GIF")
    args = parser.parse_args()

    assert len(args.masks) == len(args.names)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    inference = Inference(args.checkpoint, compile=False)

    print(f"Loading base image: {args.image}")
    image = load_image(args.image)

    # Run inference per object, keeping raw output dicts for make_scene
    outputs = []
    for i, (mask_path, name) in enumerate(zip(args.masks, args.names)):
        print(f"\n[{i+1}/{len(args.names)}] Reconstructing: {name}")
        mask = load_mask(mask_path)

        img = image.copy()
        if img.shape[:2] != mask.shape[:2]:
            print(f"  Resizing image from {img.shape[:2]} to match mask {mask.shape[:2]}")
            img_pil = Image.fromarray(img).resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
            img = np.array(img_pil)

        output = inference(img, mask, seed=args.seed)
        outputs.append(output)
        print(f"  Done.")

    # Compose scene
    print(f"\nComposing scene from {len(outputs)} objects...")
    scene_gs = make_scene(*outputs)

    # Save posed scene (before normalization)
    posed_path = os.path.join(args.output_dir, "scene_posed.ply")
    scene_gs.save_ply(posed_path)
    print(f"Saved posed scene: {posed_path}")

    # Normalize for rendering
    scene_gs = ready_gaussian_for_video_rendering(scene_gs, in_place=True)

    normalized_path = os.path.join(args.output_dir, "scene_normalized.ply")
    scene_gs.save_ply(normalized_path)
    print(f"Saved normalized scene: {normalized_path}")

    # Optional turntable GIF
    if args.render_gif:
        import imageio
        print("Rendering turntable GIF...")
        video = render_video(scene_gs, r=1, fov=60, resolution=512)["color"]
        gif_path = os.path.join(args.output_dir, "scene.gif")
        imageio.mimsave(gif_path, video, format="GIF", duration=1000 / 30, loop=0)
        print(f"Saved GIF: {gif_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
