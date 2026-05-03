from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path("/Users/jeffersonlopes/metric-spatial-vlm")
sys.path.insert(0, str(ROOT))

GT_PATH = ROOT / "benchmark" / "ground_truth_distance_nearest_test_official_stage1.csv"
OBJECTS_PATH = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"

OUT_DIR = ROOT / "results" / "geometric_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def aabb_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    Minimum Euclidean distance between two axis-aligned bounding boxes.

    If boxes overlap along an axis, the separation on that axis is zero.
    If they overlap in all axes, the distance is zero.
    """
    a_min = np.array([row_a["aabb_min_x"], row_a["aabb_min_y"], row_a["aabb_min_z"]], dtype=float)
    a_max = np.array([row_a["aabb_max_x"], row_a["aabb_max_y"], row_a["aabb_max_z"]], dtype=float)

    b_min = np.array([row_b["aabb_min_x"], row_b["aabb_min_y"], row_b["aabb_min_z"]], dtype=float)
    b_max = np.array([row_b["aabb_max_x"], row_b["aabb_max_y"], row_b["aabb_max_z"]], dtype=float)

    sep = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(sep))


def centroid_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    ca = np.array([row_a["centroid_x"], row_a["centroid_y"], row_a["centroid_z"]], dtype=float)
    cb = np.array([row_b["centroid_x"], row_b["centroid_y"], row_b["centroid_z"]], dtype=float)
    return float(np.linalg.norm(ca - cb))


def main():
    gt = pd.read_csv(GT_PATH)
    objects = pd.read_csv(OBJECTS_PATH)

    gt = gt[gt["operator"] == "distance"].copy()
    objects = objects[objects["is_valid_object"] == True].copy()

    object_map = {
        str(row["object_id"]): row
        for _, row in objects.iterrows()
    }

    rows = []

    print(f"Loaded {len(gt)} distance queries.")
    print(f"Loaded {len(objects)} valid objects.")

    for _, q in gt.iterrows():
        query_id = q["query_id"]
        scene_id = q["scene_id"]
        obj_a_id = str(q["gt_object_a"])
        obj_b_id = str(q["gt_object_b"])
        surface_ref = float(q["gt_distance_m"])

        if obj_a_id not in object_map:
            print(f"[SKIP] object A not found: {obj_a_id}")
            continue

        if obj_b_id not in object_map:
            print(f"[SKIP] object B not found: {obj_b_id}")
            continue

        obj_a = object_map[obj_a_id]
        obj_b = object_map[obj_b_id]

        d_centroid = centroid_distance(obj_a, obj_b)
        d_aabb = aabb_distance(obj_a, obj_b)

        rows.append({
            "query_id": query_id,
            "scene_id": scene_id,
            "object_a": obj_a_id,
            "label_a": obj_a["label_norm"],
            "object_b": obj_b_id,
            "label_b": obj_b["label_norm"],
            "surface_ref_distance_m": surface_ref,
            "centroid_distance_m": d_centroid,
            "aabb_distance_m": d_aabb,
            "centroid_abs_error_m": abs(d_centroid - surface_ref),
            "aabb_abs_error_m": abs(d_aabb - surface_ref),
            "aabb_minus_surface_m": d_aabb - surface_ref,
            "centroid_minus_surface_m": d_centroid - surface_ref,
            "aabb_improvement_over_centroid_m": abs(d_centroid - surface_ref) - abs(d_aabb - surface_ref),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No rows generated.")

    raw_path = OUT_DIR / "aabb_distance_baseline_per_query.csv"
    df.to_csv(raw_path, index=False)

    summary = pd.DataFrame([
        {
            "representation": "centroid",
            "n": len(df),
            "mae_m": df["centroid_abs_error_m"].mean(),
            "medae_m": df["centroid_abs_error_m"].median(),
            "p90_abs_error_m": np.percentile(df["centroid_abs_error_m"], 90),
            "p95_abs_error_m": np.percentile(df["centroid_abs_error_m"], 95),
            "max_abs_error_m": df["centroid_abs_error_m"].max(),
        },
        {
            "representation": "aabb",
            "n": len(df),
            "mae_m": df["aabb_abs_error_m"].mean(),
            "medae_m": df["aabb_abs_error_m"].median(),
            "p90_abs_error_m": np.percentile(df["aabb_abs_error_m"], 90),
            "p95_abs_error_m": np.percentile(df["aabb_abs_error_m"], 95),
            "max_abs_error_m": df["aabb_abs_error_m"].max(),
        },
        {
            "representation": "surface_point_set",
            "n": len(df),
            "mae_m": 0.0,
            "medae_m": 0.0,
            "p90_abs_error_m": 0.0,
            "p95_abs_error_m": 0.0,
            "max_abs_error_m": 0.0,
        },
    ])

    summary_path = OUT_DIR / "aabb_distance_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)

    improvement = {
        "n": len(df),
        "mean_improvement_over_centroid_m": df["aabb_improvement_over_centroid_m"].mean(),
        "median_improvement_over_centroid_m": df["aabb_improvement_over_centroid_m"].median(),
        "aabb_better_than_centroid_count": int((df["aabb_abs_error_m"] < df["centroid_abs_error_m"]).sum()),
        "aabb_better_than_centroid_pct": 100.0 * (df["aabb_abs_error_m"] < df["centroid_abs_error_m"]).mean(),
        "aabb_equal_centroid_count": int((df["aabb_abs_error_m"] == df["centroid_abs_error_m"]).sum()),
        "aabb_worse_than_centroid_count": int((df["aabb_abs_error_m"] > df["centroid_abs_error_m"]).sum()),
    }

    improvement_df = pd.DataFrame([improvement])
    improvement_path = OUT_DIR / "aabb_distance_baseline_improvement.csv"
    improvement_df.to_csv(improvement_path, index=False)

    print("\nGeometric baseline summary:")
    print(summary.to_string(index=False))

    print("\nAABB improvement over centroid:")
    for k, v in improvement.items():
        print(f"{k}: {v}")

    print("\nWrote:")
    print(raw_path)
    print(summary_path)
    print(improvement_path)


if __name__ == "__main__":
    main()
