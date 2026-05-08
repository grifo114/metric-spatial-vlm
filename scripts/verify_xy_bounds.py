"""
verify_xy_bounds.py
===================
Round-trip verification of the world-XY → image-pixel transform.

For a chosen scene, projects each object's XY centroid (from the manifest)
into pixel coordinates using a candidate xy_bounds, and overlays a red 'X'
at every projected position on top of the existing top-down render.

If the red Xs land on top of the numbered circles → bounds are correct.
If they're shifted, scaled, or flipped → adjust and rerun.

Usage:
    python verify_xy_bounds.py \
        --render renders/scene0142_00/topdown_marked.png \
        --manifest benchmark/objects_manifest_test_official_stage1.csv \
        --scene scene0142_00 \
        --bounds -2.5 -3.0 4.0 3.5 \
        --out verify_scene0142_00.png

Alternative: --bounds-from-mesh path/to/scene_vh_clean_2.ply
will compute the bounds from the mesh instead of taking them as args.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def world_to_image(x, y, xy_bounds, image_size):
    x_min, y_min, x_max, y_max = xy_bounds
    W, H = image_size
    px = (x - x_min) / (x_max - x_min) * W
    py = (y_max - y) / (y_max - y_min) * H   # y axis flipped for image
    return px, py


def square_pad(x_min, y_min, x_max, y_max):
    """Expand the smaller axis so the bounds are square (matches a 1024x1024
    render that didn't distort aspect ratio)."""
    dx = x_max - x_min
    dy = y_max - y_min
    if dx > dy:
        cy = (y_min + y_max) / 2
        return x_min, cy - dx / 2, x_max, cy + dx / 2
    else:
        cx = (x_min + x_max) / 2
        return cx - dy / 2, y_min, cx + dy / 2, y_max


def compute_pipeline_bounds(scene_df):
    """Reproduces the world bounds used by render_scene_numbered() in script 83.
    
    The pipeline:
      1. Centers the camera on the MEAN of the object centroids (not the mesh).
      2. Sets a SQUARE extent = max(x_ext, y_ext, 1.0) * 1.25,
         where x_ext/y_ext come from the OBJECT AABBs (not the mesh).
      3. Renders top-down with Y up, no rotation.
    """
    cx = float(scene_df["centroid_x"].mean())
    cy = float(scene_df["centroid_y"].mean())
    x_ext = float(scene_df["aabb_max_x"].max() - scene_df["aabb_min_x"].min())
    y_ext = float(scene_df["aabb_max_y"].max() - scene_df["aabb_min_y"].min())
    extent = max(x_ext, y_ext, 1.0) * 1.25
    return (cx - extent / 2, cy - extent / 2,
            cx + extent / 2, cy + extent / 2)


def find_centroid_columns(df):
    """Try a few common column-name conventions."""
    candidates = [
        ("centroid_x", "centroid_y"),
        ("cx", "cy"),
        ("center_x", "center_y"),
        ("x", "y"),
        ("c_x", "c_y"),
    ]
    for cx_name, cy_name in candidates:
        if cx_name in df.columns and cy_name in df.columns:
            return cx_name, cy_name
    raise SystemExit(
        f"Couldn't find centroid columns. Available columns:\n  {list(df.columns)}\n"
        "Edit find_centroid_columns() to add yours.")


def find_id_column(df):
    for n in ("object_id", "instance_id", "id", "obj_id"):
        if n in df.columns:
            return n
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--render", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--bounds", type=float, nargs=4,
                   metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    p.add_argument("--bounds-from-mesh", type=str,
                   help="path to scene mesh (PLY/OBJ) — overrides --bounds")
    p.add_argument("--from-pipeline", action="store_true",
                   help="reproduce render_scene_numbered() bounds from manifest "
                        "(centered on mean object centroid, square extent x 1.25). "
                        "RECOMMENDED.")
    p.add_argument("--no-square-pad", action="store_true",
                   help="don't expand bounds to a square (use if render is rectangular)")
    p.add_argument("--image-size", type=int, nargs=2, default=None,
                   help="override pixel size; default = read from image")
    p.add_argument("--out", default="verify_overlay.png")
    args = p.parse_args()

    # Load manifest, filter to this scene
    df = pd.read_csv(args.manifest)
    if "scene_id" not in df.columns:
        raise SystemExit(f"Expected 'scene_id' column. Got: {list(df.columns)}")
    df = df[df["scene_id"] == args.scene].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"No objects for scene {args.scene}")
    print(f"Found {len(df)} objects in {args.scene} (raw)")

    # Pipeline filters out invalid objects before computing camera params
    if args.from_pipeline and "is_valid_object" in df.columns:
        n_before = len(df)
        df = df[df["is_valid_object"] == True].reset_index(drop=True)
        print(f"  after is_valid_object filter: {len(df)} "
              f"({n_before - len(df)} excluded)")
    elif args.from_pipeline:
        print("  (no 'is_valid_object' column found — using all objects)")

    # Resolve bounds
    used_pipeline = False
    if args.from_pipeline:
        x_min, y_min, x_max, y_max = compute_pipeline_bounds(df)
        used_pipeline = True
        print(f"Pipeline bounds (from manifest, mean-centroid + 1.25× extent):")
        print(f"  x[{x_min:.3f}, {x_max:.3f}], y[{y_min:.3f}, {y_max:.3f}]")
    elif args.bounds_from_mesh:
        try:
            import trimesh
        except ImportError:
            raise SystemExit("pip install trimesh  # to use --bounds-from-mesh")
        m = trimesh.load(args.bounds_from_mesh, force="mesh")
        v = np.asarray(m.vertices)
        x_min, x_max = float(v[:, 0].min()), float(v[:, 0].max())
        y_min, y_max = float(v[:, 1].min()), float(v[:, 1].max())
        print(f"From mesh: x[{x_min:.3f}, {x_max:.3f}], y[{y_min:.3f}, {y_max:.3f}]")
    elif args.bounds:
        x_min, y_min, x_max, y_max = args.bounds
    else:
        raise SystemExit("Pass one of: --from-pipeline | --bounds-from-mesh | --bounds")

    # The pipeline already produces square bounds, so skip the square-pad in that case
    if not used_pipeline and not args.no_square_pad:
        x_min, y_min, x_max, y_max = square_pad(x_min, y_min, x_max, y_max)
        print(f"After square-pad: x[{x_min:.3f}, {x_max:.3f}], y[{y_min:.3f}, {y_max:.3f}]")

    bounds = (x_min, y_min, x_max, y_max)

    cx_col, cy_col = find_centroid_columns(df)
    id_col = find_id_column(df)
    print(f"Using centroid columns: {cx_col}, {cy_col}")

    # Load render
    img = np.asarray(Image.open(args.render).convert("RGB"))
    H_img, W_img = img.shape[:2]
    image_size = tuple(args.image_size) if args.image_size else (W_img, H_img)
    if (W_img, H_img) != image_size:
        print(f"Note: forcing projection to use {image_size}, "
              f"but image is {W_img}x{H_img}")
    print(f"Using image_size = {image_size}")

    # Project
    pxs, pys, labels = [], [], []
    for _, row in df.iterrows():
        x, y = float(row[cx_col]), float(row[cy_col])
        px, py = world_to_image(x, y, bounds, image_size)
        pxs.append(px); pys.append(py)
        labels.append(str(row[id_col]) if id_col else f"#{_}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(img)
    ax.scatter(pxs, pys, s=180, marker="x", color="red",
               linewidths=2.5, zorder=10,
               label="projected XY centroid")
    for px, py, lab in zip(pxs, pys, labels):
        ax.annotate(lab, (px, py), color="red", fontsize=8, zorder=11,
                    xytext=(6, -10), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="red", lw=0.5, alpha=0.85))

    ax.set_title(f"{args.scene} — bounds: "
                 f"({bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f})\n"
                 f"red X should overlap each numbered circle",
                 fontsize=10)
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {args.out}")
    print("\nVisual checklist:")
    print("  Xs aligned with circles            → bounds OK ✓")
    print("  Xs shifted left/right or up/down   → bounds asymmetric or padded")
    print("  Xs scaled (clustered or spread)    → wrong xy_bounds extent")
    print("  Xs mirrored (left↔right)           → check sign of x axis")
    print("  Xs flipped (top↔bottom)            → toggle the y-flip in world_to_image")


if __name__ == "__main__":
    main()