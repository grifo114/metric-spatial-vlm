from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import is_between_xy, is_aligned_xy

MANIFEST = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"
QUERIES = ROOT / "benchmark" / "queries_test_official_stage1_relational_binary_labeled_repaired.csv"
OUT_PATH = ROOT / "results" / "benchmark_v1" / "test_stage1_relational_binary_predictions.csv"

def centroid_xyz(row: pd.Series) -> np.ndarray:
    return row[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)

def main():
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["is_valid_object"] == True].copy()
    queries = pd.read_csv(QUERIES)

    by_id = {row["object_id"]: row for _, row in manifest.iterrows()}
    rows = []

    for _, q in queries.iterrows():
        op = q["operator"]

        if op == "between":
            row_x = by_id[q["object_x"]]
            row_a = by_id[q["object_a"]]
            row_b = by_id[q["object_b"]]

            cx = centroid_xyz(pd.Series(row_x))
            ca = centroid_xyz(pd.Series(row_a))
            cb = centroid_xyz(pd.Series(row_b))

            pred_label = int(is_between_xy(
                cx, ca, cb,
                tau_between=float(q.get("tau_between_m", 0.30))
            ))

        elif op == "aligned":
            row_a = by_id[q["object_a"]]
            row_b = by_id[q["object_b"]]
            row_c = by_id[q["object_c"]]

            ca = centroid_xyz(pd.Series(row_a))
            cb = centroid_xyz(pd.Series(row_b))
            cc = centroid_xyz(pd.Series(row_c))

            pred_label = int(is_aligned_xy(
                ca, cb, cc,
                tau_align=float(q.get("tau_align_m", 0.25))
            ))

        else:
            continue

        rows.append({
            "binary_query_id": q["binary_query_id"],
            "scene_id": q["scene_id"],
            "operator": op,
            "structured_query": q["structured_query"],
            "pred_label": pred_label,
        })

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved predictions: {OUT_PATH}")
    print()
    print(out_df.head(20).to_string(index=False))
    print()
    print(out_df["operator"].value_counts(dropna=False))

if __name__ == "__main__":
    main()