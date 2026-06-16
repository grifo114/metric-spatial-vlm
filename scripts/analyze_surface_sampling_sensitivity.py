from pathlib import Path
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path("/Users/jeffersonlopes/metric-spatial-vlm")
sys.path.insert(0, str(ROOT))

from scipy.spatial import cKDTree

from src.geometry.geometry_ops import load_points_npz


ROOT = Path("/Users/jeffersonlopes/metric-spatial-vlm")

GT_PATH = ROOT / "benchmark" / "ground_truth_distance_nearest_test_official_stage1.csv"
OBJECTS_PATH = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"

OUT_DIR = ROOT / "results" / "sampling_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_POINTS_LIST = [512, 1000, 2000, 5000, 10000, 20000]
SEEDS = [0, 1, 2, 3, 4]


def subsample_points(points: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points)
    rng = np.random.default_rng(seed)

    if len(points) == 0:
        raise ValueError("Empty point set.")

    # If the object has fewer points than requested, sample with replacement.
    replace = len(points) < n_points
    idx = rng.choice(len(points), size=n_points, replace=replace)
    return points[idx]


def main():
    gt = pd.read_csv(GT_PATH)
    objects = pd.read_csv(OBJECTS_PATH)

    # Keep only distance queries.
    gt = gt[gt["operator"] == "distance"].copy()

    # object_id -> points_path
    obj_to_points = {}
    obj_to_npoints = {}

    for _, row in objects.iterrows():
        object_id = str(row["object_id"])
        points_path = Path(str(row["points_path"]))

        if not points_path.is_absolute():
            points_path = ROOT / points_path

        obj_to_points[object_id] = points_path
        obj_to_npoints[object_id] = int(row["n_points"])

    rows = []

    print(f"Loaded {len(gt)} distance ground-truth pairs.")
    print(f"Loaded {len(objects)} objects from manifest.")

    for _, row in gt.iterrows():
        query_id = row["query_id"]
        scene_id = row["scene_id"]
        obj_a = str(row["gt_object_a"])
        obj_b = str(row["gt_object_b"])
        gt_distance_m = float(row["gt_distance_m"])

        if obj_a not in obj_to_points:
            print(f"[SKIP] object A not found in manifest: {obj_a}")
            continue

        if obj_b not in obj_to_points:
            print(f"[SKIP] object B not found in manifest: {obj_b}")
            continue

        path_a = obj_to_points[obj_a]
        path_b = obj_to_points[obj_b]

        if not path_a.exists():
            print(f"[SKIP] points file not found for A: {path_a}")
            continue

        if not path_b.exists():
            print(f"[SKIP] points file not found for B: {path_b}")
            continue

        pts_a_full = load_points_npz(path_a)
        pts_b_full = load_points_npz(path_b)

        # Operational reference: all stored points.
        full_distance = float(surface_distance(pts_a_full, pts_b_full))

        # Difference between stored gt_distance_m and recomputed full-point distance.
        full_vs_gt_abs_error = abs(full_distance - gt_distance_m)

        for n_points in N_POINTS_LIST:
            for seed in SEEDS:
                pts_a = subsample_points(pts_a_full, n_points, seed)
                pts_b = subsample_points(pts_b_full, n_points, seed + 1000003)

                sampled_distance = float(surface_distance(pts_a, pts_b))

                rows.append(
                    {
                        "query_id": query_id,
                        "scene_id": scene_id,
                        "object_a": obj_a,
                        "object_b": obj_b,
                        "available_points_a": len(pts_a_full),
                        "available_points_b": len(pts_b_full),
                        "n_points_requested": n_points,
                        "seed": seed,
                        "sampled_distance_m": sampled_distance,
                        "full_point_distance_m": full_distance,
                        "stored_gt_distance_m": gt_distance_m,
                        "abs_error_vs_full_m": abs(sampled_distance - full_distance),
                        "abs_error_full_vs_stored_gt_m": full_vs_gt_abs_error,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No rows were generated. Check paths and object IDs.")

    raw_path = OUT_DIR / "surface_point_sampling_sensitivity_raw.csv"
    df.to_csv(raw_path, index=False)

    summary = (
        df.groupby("n_points_requested")
        .agg(
            n=("abs_error_vs_full_m", "count"),
            median_abs_error_m=("abs_error_vs_full_m", "median"),
            mean_abs_error_m=("abs_error_vs_full_m", "mean"),
            p90_abs_error_m=("abs_error_vs_full_m", lambda x: np.percentile(x, 90)),
            p95_abs_error_m=("abs_error_vs_full_m", lambda x: np.percentile(x, 95)),
            max_abs_error_m=("abs_error_vs_full_m", "max"),
        )
        .reset_index()
    )

    summary_path = OUT_DIR / "surface_point_sampling_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    seed_summary = (
        df.groupby(["query_id", "n_points_requested"])
        .agg(
            min_distance_m=("sampled_distance_m", "min"),
            max_distance_m=("sampled_distance_m", "max"),
            std_distance_m=("sampled_distance_m", "std"),
        )
        .reset_index()
    )

    seed_summary["range_distance_m"] = (
        seed_summary["max_distance_m"] - seed_summary["min_distance_m"]
    )

    seed_global = (
        seed_summary.groupby("n_points_requested")
        .agg(
            median_seed_range_m=("range_distance_m", "median"),
            p90_seed_range_m=("range_distance_m", lambda x: np.percentile(x, 90)),
            p95_seed_range_m=("range_distance_m", lambda x: np.percentile(x, 95)),
            max_seed_range_m=("range_distance_m", "max"),
            median_seed_std_m=("std_distance_m", "median"),
        )
        .reset_index()
    )

    seed_path = OUT_DIR / "surface_point_sampling_seed_sensitivity_summary.csv"
    seed_global.to_csv(seed_path, index=False)

    gt_check = (
        df[["query_id", "full_point_distance_m", "stored_gt_distance_m", "abs_error_full_vs_stored_gt_m"]]
        .drop_duplicates()
        .copy()
    )

    gt_check_path = OUT_DIR / "surface_full_vs_stored_gt_check.csv"
    gt_check.to_csv(gt_check_path, index=False)

    print("\nSampling-density sensitivity:")
    print(summary.to_string(index=False))

    print("\nSeed sensitivity:")
    print(seed_global.to_string(index=False))

    print("\nFull-point distance vs stored gt_distance_m:")
    print(gt_check["abs_error_full_vs_stored_gt_m"].describe().to_string())

    print("\nWrote:")
    print(raw_path)
    print(summary_path)
    print(seed_path)
    print(gt_check_path)


if __name__ == "__main__":
    main()  