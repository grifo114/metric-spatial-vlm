from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_dev_distance_nearest_candidates.jsonl"
OUT_PATH = ROOT / "benchmark" / "queries_dev_distance_nearest_review.csv"


def main():
    rows = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # colunas de revisão manual
    df["review_keep"] = ""
    df["review_reason"] = ""
    df["review_notes"] = ""

    preferred_cols = [
        "scene_id",
        "operator",
        "structured_query",
        "natural_query",
        "object_a",
        "label_a",
        "object_b",
        "label_b",
        "surface_distance_m",
        "centroid_distance_m",
        "reference_object",
        "reference_label",
        "target_category",
        "answer_object",
        "answer_distance_m",
        "second_best_object",
        "second_best_distance_m",
        "margin_to_second_m",
        "status",
        "review_keep",
        "review_reason",
        "review_notes",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols].sort_values(["operator", "scene_id", "structured_query"])

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved review sheet: {OUT_PATH}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()