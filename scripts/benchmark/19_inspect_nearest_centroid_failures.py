from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GT = ROOT / "benchmark" / "ground_truth_distance_nearest.csv"
SURF = ROOT / "results" / "pilot" / "benchmark_distance_nearest_predictions.csv"
CENT = ROOT / "results" / "pilot" / "baseline_centroid_distance_nearest_predictions.csv"
OUT = ROOT / "results" / "pilot" / "nearest_centroid_failures.csv"

def main():
    gt = pd.read_csv(GT)
    surf = pd.read_csv(SURF).rename(columns={
        "pred_answer_object": "surface_pred_answer_object",
        "pred_distance_m": "surface_pred_distance_m",
    })
    cent = pd.read_csv(CENT).rename(columns={
        "pred_answer_object": "centroid_pred_answer_object",
        "pred_distance_m": "centroid_pred_distance_m",
    })

    df = gt.merge(surf, on=["query_id", "scene_id", "operator", "structured_query"], how="inner")
    df = df.merge(cent, on=["query_id", "scene_id", "operator", "structured_query"], how="inner")

    ndf = df[df["operator"] == "nearest"].copy()
    failures = ndf[ndf["centroid_pred_answer_object"] != ndf["gt_answer_object"]].copy()

    failures.to_csv(OUT, index=False)

    print(f"Saved: {OUT}")
    print()
    if len(failures) == 0:
        print("No centroid nearest failures found.")
    else:
        print(failures[[
            "query_id",
            "scene_id",
            "structured_query",
            "gt_answer_object",
            "surface_pred_answer_object",
            "centroid_pred_answer_object",
            "gt_distance_m",
            "surface_pred_distance_m",
            "centroid_pred_distance_m",
        ]].to_string(index=False))

if __name__ == "__main__":
    main()