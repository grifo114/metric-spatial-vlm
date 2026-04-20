from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"
QUERIES_PATH = ROOT / "benchmark" / "queries_test_official_stage1_distance_nearest_final.csv"


def load_points_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    return data["points"]


def surface_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    best = float("inf")

    for pa in points_a:
        d2 = np.sum((points_b - pa[None, :]) ** 2, axis=1)
        local = float(np.sqrt(d2.min()))
        if local < best:
            best = local

    for pb in points_b:
        d2 = np.sum((points_a - pb[None, :]) ** 2, axis=1)
        local = float(np.sqrt(d2.min()))
        if local < best:
            best = local

    return best


def centroid_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    a = row_a[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
    b = row_b[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
    return float(np.linalg.norm(a - b))


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    queries = pd.read_csv(QUERIES_PATH)

    manifest = manifest[manifest["is_valid_object"] == True].copy()
    by_object = {row["object_id"]: row for _, row in manifest.iterrows()}

    nearest_queries = queries[queries["operator"] == "nearest"].copy()

    rows = []

    for _, q in nearest_queries.iterrows():
        scene_id = q["scene_id"]
        ref_object = q["reference_object"]
        target_category = q["target_category"]

        if ref_object not in by_object:
            continue

        ref_row = by_object[ref_object]
        ref_points = load_points_npz(ROOT / ref_row["points_path"])

        candidates = manifest[
            (manifest["scene_id"] == scene_id)
            & (manifest["label_norm"] == target_category)
        ].copy()

        candidates = candidates[candidates["object_id"] != ref_object].copy()
        if len(candidates) == 0:
            continue

        best_centroid = None
        best_centroid_d = float("inf")
        best_surface = None
        best_surface_d = float("inf")

        per_candidate = []

        for _, cand in candidates.iterrows():
            cd = centroid_distance(ref_row, cand)
            sp = load_points_npz(ROOT / cand["points_path"])
            sd = surface_distance(ref_points, sp)

            per_candidate.append(
                {
                    "object_id": cand["object_id"],
                    "label_norm": cand["label_norm"],
                    "centroid_distance": cd,
                    "surface_distance": sd,
                }
            )

            if cd < best_centroid_d:
                best_centroid_d = cd
                best_centroid = cand["object_id"]

            if sd < best_surface_d:
                best_surface_d = sd
                best_surface = cand["object_id"]

        same_winner = best_centroid == best_surface

        per_candidate_sorted_centroid = sorted(per_candidate, key=lambda x: x["centroid_distance"])
        per_candidate_sorted_surface = sorted(per_candidate, key=lambda x: x["surface_distance"])

        rows.append(
            {
                "scene_id": scene_id,
                "reference_object": ref_object,
                "target_category": target_category,
                "winner_centroid": best_centroid,
                "winner_surface": best_surface,
                "winner_same": same_winner,
                "best_centroid_distance": best_centroid_d,
                "best_surface_distance": best_surface_d,
                "ranking_centroid": json.dumps([x["object_id"] for x in per_candidate_sorted_centroid]),
                "ranking_surface": json.dumps([x["object_id"] for x in per_candidate_sorted_surface]),
            }
        )

    out = pd.DataFrame(rows)

    out_dir = ROOT / "results" / "benchmark_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "nearest_divergence_analysis.csv"
    out.to_csv(out_csv, index=False)

    total = len(out)
    divergent = int((~out["winner_same"]).sum()) if total > 0 else 0

    print(f"Saved: {out_csv}")
    print(f"Total nearest queries analyzed: {total}")
    print(f"Queries with different winners: {divergent}")
    if total > 0:
        print(f"Fraction with different winners: {divergent / total:.4f}")

    if divergent > 0:
        print("\nExamples with divergence:")
        print(
            out[~out["winner_same"]][
                ["scene_id", "reference_object", "target_category", "winner_centroid", "winner_surface"]
            ].head(10).to_string(index=False)
        )


if __name__ == "__main__":
    main()