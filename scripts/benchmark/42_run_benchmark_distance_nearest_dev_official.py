from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import load_points_npz, surface_distance, nearest_object_by_surface

MANIFEST = ROOT / "benchmark" / "objects_manifest_dev_official.csv"
GT = ROOT / "benchmark" / "ground_truth_distance_nearest_dev_official.csv"
QUERIES = ROOT / "benchmark" / "queries_dev_official_distance_nearest_final.csv"
OUT_PATH = ROOT / "results" / "benchmark_v1" / "dev_official_distance_nearest_surface_predictions.csv"

def main():
    manifest = pd.read_csv(MANIFEST)
    gt = pd.read_csv(GT)
    queries = pd.read_csv(QUERIES)

    manifest = manifest[manifest["is_valid_object"] == True].copy()

    by_scene = {scene_id: g.copy() for scene_id, g in manifest.groupby("scene_id")}
    by_id = {row["object_id"]: row for _, row in manifest.iterrows()}
    gt_by_query = {row["structured_query"]: row for _, row in gt.iterrows()}

    rows = []

    for _, q in queries.iterrows():
        scene_id = q["scene_id"]
        op = q["operator"]
        query_id = gt_by_query[q["structured_query"]]["query_id"]

        if op == "distance":
            row_a = by_id[q["object_a"]]
            row_b = by_id[q["object_b"]]

            pts_a = load_points_npz(ROOT / row_a["points_path"])
            pts_b = load_points_npz(ROOT / row_b["points_path"])

            pred = surface_distance(pts_a, pts_b)

            rows.append({
                "query_id": query_id,
                "scene_id": scene_id,
                "operator": op,
                "structured_query": q["structured_query"],
                "pred_distance_m": float(pred),
                "pred_answer_object": "",
            })

        elif op == "nearest":
            ref_row = by_id[q["reference_object"]]
            ref_points = load_points_npz(ROOT / ref_row["points_path"])

            scene_df = by_scene[scene_id]
            target_df = scene_df[scene_df["label_norm"] == q["target_category"]].copy()

            candidate_dict = {}
            for _, r in target_df.iterrows():
                candidate_dict[r["object_id"]] = load_points_npz(ROOT / r["points_path"])

            pred_obj, pred_dist = nearest_object_by_surface(ref_points, candidate_dict)

            rows.append({
                "query_id": query_id,
                "scene_id": scene_id,
                "operator": op,
                "structured_query": q["structured_query"],
                "pred_distance_m": float(pred_dist),
                "pred_answer_object": pred_obj,
            })

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved predictions: {OUT_PATH}")
    print()
    print(out_df["operator"].value_counts(dropna=False))
    print()
    print(out_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()