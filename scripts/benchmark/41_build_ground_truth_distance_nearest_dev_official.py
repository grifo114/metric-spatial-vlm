from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import load_points_npz, surface_distance

MANIFEST = ROOT / "benchmark" / "objects_manifest_dev_official.csv"
QUERIES = ROOT / "benchmark" / "queries_dev_official_distance_nearest_final.csv"
OUT_PATH = ROOT / "benchmark" / "ground_truth_distance_nearest_dev_official.csv"

def main():
    manifest = pd.read_csv(MANIFEST)
    queries = pd.read_csv(QUERIES)

    manifest = manifest[manifest["is_valid_object"] == True].copy()
    manifest_by_id = {row["object_id"]: row for _, row in manifest.iterrows()}

    rows = []

    for i, q in queries.reset_index(drop=True).iterrows():
        op = q["operator"]
        query_id = f"dev_official_dn_{i:04d}"

        if op == "distance":
            object_a = q["object_a"]
            object_b = q["object_b"]

            row_a = manifest_by_id[object_a]
            row_b = manifest_by_id[object_b]

            pts_a = load_points_npz(ROOT / row_a["points_path"])
            pts_b = load_points_npz(ROOT / row_b["points_path"])

            gt_value = surface_distance(pts_a, pts_b)

            rows.append({
                "query_id": query_id,
                "scene_id": q["scene_id"],
                "operator": op,
                "structured_query": q["structured_query"],
                "gt_object_a": object_a,
                "gt_object_b": object_b,
                "gt_distance_m": float(gt_value),
                "gt_answer_object": "",
            })

        elif op == "nearest":
            rows.append({
                "query_id": query_id,
                "scene_id": q["scene_id"],
                "operator": op,
                "structured_query": q["structured_query"],
                "gt_object_a": q["reference_object"],
                "gt_object_b": "",
                "gt_distance_m": float(q["answer_distance_m"]),
                "gt_answer_object": q["answer_object"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print()
    print(df["operator"].value_counts(dropna=False))
    print()
    print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()