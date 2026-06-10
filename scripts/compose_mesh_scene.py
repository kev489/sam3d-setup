# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
compose_mesh_scene.py — Compose multiple object meshes into a single scene.

Reads the raw OBJ meshes (Z-up, canonical frame) and JSON metadata,
applies scale -> rotation -> translation to place each object in the
shared camera frame, then exports a combined OBJ and GLB.

Usage:
    python compose_mesh_scene.py --input-dir ../sam_3d_segments/outputs --names airpods marker snowCone --output-dir ../sam_3d_segments/outputs
"""

import os
import json
import argparse
import numpy as np


def load_obj(path):
    """Load vertices and faces from a Wavefront OBJ file."""
    vertices = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith("f "):
                face = [int(x.split("/")[0]) - 1 for x in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def save_obj(path, vertices, faces, vert_offset=0):
    """Save vertices and faces as OBJ. Returns number of vertices written."""
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        lines.append(f"f {f[0]+1+vert_offset} {f[1]+1+vert_offset} {f[2]+1+vert_offset}")
    return "\n".join(lines), len(vertices)


def quat_wxyz_to_rotation_matrix(q_wxyz):
    """Convert wxyz quaternion to 3x3 rotation matrix (same as pytorch3d.quaternion_to_matrix)."""
    w, x, y, z = q_wxyz
    # Standard quaternion to rotation matrix
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def transform_vertices(vertices, scale, rotation_quat_wxyz, translation):
    """
    Apply the same transform chain as the repo's compose_transform:
    Scale -> Rotate -> Translate

    PyTorch3D uses row-vector convention: p_new = p @ R (NOT R @ p).
    The raw OBJ mesh is in Z-up canonical frame, same as the pose parameters.
    Quaternion is wxyz (pytorch3d convention).
    """
    R = quat_wxyz_to_rotation_matrix(rotation_quat_wxyz)

    # Scale (isotropic)
    s = scale[0]
    v = vertices * s

    # Rotate — row-vector convention: v @ R (matching pytorch3d)
    v = v @ R

    # Translate
    v = v + np.array(translation)

    return v


def z_up_to_y_up(vertices):
    """Convert from Z-up to Y-up for GLB export: (x,y,z) -> (x,z,-y)."""
    conv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    return vertices @ conv


def main():
    parser = argparse.ArgumentParser(description="Compose object meshes into a single scene")
    parser.add_argument("--input-dir", required=True, help="Directory with per-object .obj and .json files")
    parser.add_argument("--names", nargs="+", required=True, help="Object names (must match filenames)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as input-dir)")
    parser.add_argument("--y-up", action="store_true", help="Convert final scene to Y-up (for GLB viewers)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    all_obj_parts = []
    vert_offset = 0

    for name in args.names:
        obj_path = os.path.join(args.input_dir, f"{name}.obj")
        json_path = os.path.join(args.input_dir, f"{name}.json")

        print(f"Loading {name}...")
        vertices, faces = load_obj(obj_path)
        with open(json_path) as f:
            meta = json.load(f)

        print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
        print(f"  Scale: {meta['scale'][0]:.4f}, Translation: {meta['translation']}")

        # Apply pose transform in Z-up frame
        transformed = transform_vertices(
            vertices,
            meta["scale"],
            meta["rotation_quaternion"],
            meta["translation"],
        )

        print(f"  Transformed bounds: {transformed.min(axis=0).round(3)} to {transformed.max(axis=0).round(3)}")

        all_obj_parts.append((transformed, faces, vert_offset, name))
        vert_offset += len(vertices)

    # Write combined OBJ (Z-up, camera frame)
    scene_obj_path = os.path.join(output_dir, "scene_mesh.obj")
    with open(scene_obj_path, "w") as f:
        running_offset = 0
        for verts, faces, _, name in all_obj_parts:
            f.write(f"# Object: {name}\n")
            f.write(f"o {name}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1+running_offset} {face[1]+1+running_offset} {face[2]+1+running_offset}\n")
            running_offset += len(verts)
    print(f"\nSaved combined scene (Z-up): {scene_obj_path}")

    # Also export Y-up version for GLB-compatible viewers
    scene_yup_path = os.path.join(output_dir, "scene_mesh_yup.obj")
    with open(scene_yup_path, "w") as f:
        running_offset = 0
        for verts, faces, _, name in all_obj_parts:
            verts_yup = z_up_to_y_up(verts)
            f.write(f"# Object: {name}\n")
            f.write(f"o {name}\n")
            for v in verts_yup:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1+running_offset} {face[1]+1+running_offset} {face[2]+1+running_offset}\n")
            running_offset += len(verts)
    print(f"Saved combined scene (Y-up): {scene_yup_path}")

    # Export GLB via trimesh
    try:
        import trimesh
        running_offset = 0
        meshes = []
        for verts, faces, _, name in all_obj_parts:
            verts_yup = z_up_to_y_up(verts)
            mesh = trimesh.Trimesh(vertices=verts_yup, faces=faces, process=False)
            mesh.metadata["name"] = name
            meshes.append(mesh)

        scene = trimesh.Scene()
        for mesh, (_, _, _, name) in zip(meshes, all_obj_parts):
            scene.add_geometry(mesh, node_name=name)

        glb_path = os.path.join(output_dir, "scene_mesh.glb")
        scene.export(glb_path)
        print(f"Saved combined GLB (Y-up):  {glb_path}")
    except ImportError:
        print("trimesh not available, skipping GLB export")

    print("\nDone!")


if __name__ == "__main__":
    main()
