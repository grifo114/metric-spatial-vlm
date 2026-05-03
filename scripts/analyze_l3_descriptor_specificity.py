from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path("/Users/jeffersonlopes/metric-spatial-vlm")
sys.path.insert(0, str(ROOT))

from src.grounding.spatial_descriptor import (
    compute_scene_frame,
    descriptor_level3,
)

GT_PATH = ROOT / "benchmark" / "ground_truth_distance_nearest_test_official_stage1.csv"
OBJECTS_PATH = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"

OUT_DIR = ROOT / "results" / "descriptor_specificity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_l3_descriptor(obj_id, scene_objects, query_categories):
    row = scene_objects[scene_objects["object_id"] == obj_id]
    if row.empty:
        raise ValueError(f"Object not found: {obj_id}")
    row = row.iloc[0]

    frame = compute_scene_frame(scene_objects)

    # IMPORTANT:
    # L3 excludes queried categories, as described in the paper.
    return descriptor_level3(
        row,
        scene_objects,
        frame,
        language="en",
        excluded_categories=query_categories,
    )


def count_same_category_with_descriptor(scene_objects, category, descriptor, query_categories):
    frame = compute_scene_frame(scene_objects)
    candidates = scene_objects[scene_objects["label_norm"] == category].copy()

    count = 0
    ids = []

    for _, obj in candidates.iterrows():
        d = descriptor_level3(
            obj,
            scene_objects,
            frame,
            language="en",
            excluded_categories=query_categories, 
        )
        if d == descriptor:
            count += 1
            ids.append(obj["object_id"])

    return count, ids, len(candidates)


def main():
    gt = pd.read_csv(GT_PATH)
    objects = pd.read_csv(OBJECTS_PATH)

    gt = gt[gt["operator"] == "distance"].copy()
    objects = objects[objects["is_valid_object"] == True].copy()

    rows = []

    for _, q in gt.iterrows():
        query_id = q["query_id"]
        scene_id = q["scene_id"]
        obj_a = q["gt_object_a"]
        obj_b = q["gt_object_b"]

        scene_objects = objects[objects["scene_id"] == scene_id].copy()

        row_a = scene_objects[scene_objects["object_id"] == obj_a]
        row_b = scene_objects[scene_objects["object_id"] == obj_b]

        if row_a.empty or row_b.empty:
            print(f"[SKIP] missing GT object for {query_id}")
            continue

        label_a = row_a.iloc[0]["label_norm"]
        label_b = row_b.iloc[0]["label_norm"]

        query_categories = {label_a, label_b}

        desc_a = get_l3_descriptor(obj_a, scene_objects, query_categories)
        desc_b = get_l3_descriptor(obj_b, scene_objects, query_categories)

        n_match_a, ids_match_a, n_candidates_a = count_same_category_with_descriptor(
            scene_objects, label_a, desc_a, query_categories
        )

        n_match_b, ids_match_b, n_candidates_b = count_same_category_with_descriptor(
            scene_objects, label_b, desc_b, query_categories
        )

        total_pairs = n_candidates_a * n_candidates_b

        # If labels are the same, avoid self-pair counting if needed.
        # In this benchmark, distance pairs are usually different categories,
        # but this keeps the analysis safer.
        if label_a == label_b:
            total_pairs = n_candidates_a * max(n_candidates_b - 1, 0)

        compatible_pairs = n_match_a * n_match_b
        if label_a == label_b and desc_a == desc_b:
            compatible_pairs = n_match_a * max(n_match_b - 1, 0)

        rows.append({
            "query_id": query_id,
            "scene_id": scene_id,
            "object_a": obj_a,
            "label_a": label_a,
            "object_b": obj_b,
            "label_b": label_b,
            "descriptor_a_l3": desc_a,
            "descriptor_b_l3": desc_b,
            "n_candidates_a": n_candidates_a,
            "n_candidates_b": n_candidates_b,
            "n_same_category_with_desc_a": n_match_a,
            "n_same_category_with_desc_b": n_match_b,
            "unique_l3_a": n_match_a == 1,
            "unique_l3_b": n_match_b == 1,
            "unique_l3_both": (n_match_a == 1 and n_match_b == 1),
            "total_candidate_pairs": total_pairs,
            "compatible_pairs_by_l3": compatible_pairs,
            "pair_reduction_ratio": compatible_pairs / total_pairs if total_pairs else np.nan,
            "matching_ids_a": "|".join(ids_match_a),
            "matching_ids_b": "|".join(ids_match_b),
        })

    df = pd.DataFrame(rows)

    raw_path = OUT_DIR / "l3_descriptor_specificity_per_query.csv"
    df.to_csv(raw_path, index=False)

    summary = {
        "n_queries": len(df),
        "unique_l3_a_pct": 100 * df["unique_l3_a"].mean(),
        "unique_l3_b_pct": 100 * df["unique_l3_b"].mean(),
        "unique_l3_both_pct": 100 * df["unique_l3_both"].mean(),
        "median_pair_reduction_ratio": df["pair_reduction_ratio"].median(),
        "p25_pair_reduction_ratio": df["pair_reduction_ratio"].quantile(0.25),
        "p75_pair_reduction_ratio": df["pair_reduction_ratio"].quantile(0.75),
        "n_l3_reduces_to_single_pair": int((df["compatible_pairs_by_l3"] == 1).sum()),
        "l3_reduces_to_single_pair_pct": 100 * (df["compatible_pairs_by_l3"] == 1).mean(),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = OUT_DIR / "l3_descriptor_specificity_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nL3 descriptor specificity summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nWrote:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
