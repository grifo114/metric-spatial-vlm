"""
04_baseline_pixel2d.py

Baseline A: estimates distance between two objects using the ratio of
their projected 2D centroid distance to image width, scaled by a fixed
room diagonal of 5.0 metres.

Uses ALL poses (from 03b) to find the best frame for each pair.
No neural network — pure geometry projection.

Input:  results/selected_pairs.csv
        data/scannet/poses/<scene>/
Output: results/predictions_baseline_a.csv
"""

import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm

INPUT_CSV   = "results/selected_pairs.csv"
POSES_DIR   = "data/scannet/poses"
OUTPUT_CSV  = "results/predictions_baseline_a.csv"

ROOM_DIAGONAL_M = 5.0

# ── Helpers ──────────────────────────────────────────────────────

def load_intrinsics(scene_id):
    path = os.path.join(POSES_DIR, scene_id, "intrinsics.json")
    with open(path) as f:
        return json.load(f)

def load_all_poses(scene_id):
    """Return dict {frame_idx: 4x4 np.array}."""
    pose_dir = os.path.join(POSES_DIR, scene_id)
    poses = {}
    for fname in sorted(os.listdir(pose_dir)):
        if not fname.endswith(".txt"):
            continue
        idx  = int(fname.replace(".txt", ""))
        pose = np.loadtxt(os.path.join(pose_dir, fname), dtype=np.float64)
        if np.isfinite(pose).all() and not np.allclose(pose, 0):
            poses[idx] = pose
    return poses

def project(centroid_world, intrinsics, pose_c2w):
    """
    Project a 3D world point onto the image plane.
    Returns (u, v) or None if behind camera.
    """
    w2c = np.linalg.inv(pose_c2w)
    pt  = w2c @ np.array([*centroid_world, 1.0])
    x, y, z = pt[:3]
    if z <= 0.1:
        return None
    u = intrinsics["fx"] * x / z + intrinsics["cx"]
    v = intrinsics["fy"] * y / z + intrinsics["cy"]
    return (u, v)

def find_best_frame(centroids_a, centroids_b, intrinsics, poses):
    """
    Find the frame where both objects project inside image bounds
    and are most centred. Returns (pa, pb) pixel coords or None.
    """
    W = intrinsics["width"]
    H = intrinsics["height"]
    margin = 10

    best_pa, best_pb = None, None
    best_score = -np.inf

    for idx, pose in poses.items():
        pa = project(centroids_a, intrinsics, pose)
        pb = project(centroids_b, intrinsics, pose)

        if pa is None or pb is None:
            continue

        ua, va = pa
        ub, vb = pb

        in_a = margin < ua < W-margin and margin < va < H-margin
        in_b = margin < ub < W-margin and margin < vb < H-margin
        if not in_a or not in_b:
            continue

        # Prefer frame where both projections are well inside image
        score = -(abs(ua - W/2) + abs(va - H/2) +
                  abs(ub - W/2) + abs(vb - H/2))

        if score > best_score:
            best_score = score
            best_pa, best_pb = pa, pb

    return (best_pa, best_pb) if best_pa is not None else None

def pixel_to_metric(pa, pb, image_width):
    px_dist = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
    return (px_dist / image_width) * ROOM_DIAGONAL_M

# ── Main ─────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Baseline A — Pixel 2D")
    print(f"Processing {len(df)} pairs across "
          f"{df['scene_id'].nunique()} scenes...\n")

    # Cache poses per scene
    poses_cache = {}
    intr_cache  = {}

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pairs"):
        scene_id = row["scene_id"]

        if scene_id not in poses_cache:
            poses_cache[scene_id] = load_all_poses(scene_id)
            intr_cache[scene_id]  = load_intrinsics(scene_id)

        poses      = poses_cache[scene_id]
        intrinsics = intr_cache[scene_id]

        ca = [row["cx_a"], row["cy_a"], row["cz_a"]]
        cb = [row["cx_b"], row["cy_b"], row["cz_b"]]

        result = find_best_frame(ca, cb, intrinsics, poses)

        if result is None:
            pred   = float("nan")
            status = "no_valid_frame"
        else:
            pa, pb = result
            pred   = round(pixel_to_metric(pa, pb, intrinsics["width"]), 4)
            status = "ok"

        results.append({
            "scene_id":        row["scene_id"],
            "range":           row["range"],
            "gt_distance_m":   row["gt_distance_m"],
            "pred_distance_m": pred,
            "label_a":         row["label_a"],
            "label_b":         row["label_b"],
            "method":          "baseline_a_pixel2d",
            "status":          status,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    valid = out_df[out_df["status"] == "ok"].copy()
    valid["abs_error"] = (valid["pred_distance_m"] - valid["gt_distance_m"]).abs()
    valid["rel_error"] = valid["abs_error"] / valid["gt_distance_m"] * 100

    print(f"\n{'─'*50}")
    print(f"Pairs processed : {len(out_df)}")
    print(f"Valid           : {len(valid)}")
    print(f"Failed          : {len(out_df) - len(valid)}")

    if len(valid) > 0:
        print(f"\nMAE             : {valid['abs_error'].mean():.4f} m")
        print(f"Mean Rel. Error : {valid['rel_error'].mean():.2f} %")
        print(f"\nBy range:")
        for rng in ["short", "medium", "long"]:
            sub = valid[valid["range"] == rng]
            if len(sub) == 0:
                continue
            print(f"  {rng:<8}  n={len(sub):>3}  "
                  f"MAE={sub['abs_error'].mean():.3f} m  "
                  f"RelErr={sub['rel_error'].mean():.1f}%")

    print(f"\nOutput: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
