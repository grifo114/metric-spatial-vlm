from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GT = ROOT / "benchmark" / "ground_truth_distance_nearest_dev_official.csv"
SURF = ROOT / "results" / "benchmark_v1" / "dev_official_distance_nearest_surface_predictions.csv"
CENT = ROOT / "results" / "benchmark_v1" / "dev_official_distance_nearest_centroid_predictions.csv"

OUT_TXT = ROOT / "results" / "benchmark_v1" / "dev_official_surface_vs_centroid_summary.txt"
OUT_CSV = ROOT / "results" / "benchmark_v1" / "dev_official_surface_vs_centroid_per_query.csv"

def main():
    gt = pd.read_csv(GT)
    surf = pd.read_csv(SURF).rename(columns={
        "pred_distance_m": "surface_pred_distance_m",
        "pred_answer_object": "surface_pred_answer_object",
    })
    cent = pd.read_csv(CENT).rename(columns={
        "pred_distance_m": "centroid_pred_distance_m",
        "pred_answer_object": "centroid_pred_answer_object",
    })

    df = gt.merge(surf, on=["query_id", "scene_id", "operator", "structured_query"], how="inner")
    df = df.merge(cent, on=["query_id", "scene_id", "operator", "structured_query"], how="inner")

    lines = []

    ddf = df[df["operator"] == "distance"].copy()
    if len(ddf) > 0:
        ddf["surface_abs_error"] = (ddf["surface_pred_distance_m"] - ddf["gt_distance_m"]).abs()
        ddf["centroid_abs_error"] = (ddf["centroid_pred_distance_m"] - ddf["gt_distance_m"]).abs()
        ddf["centroid_minus_surface"] = ddf["centroid_abs_error"] - ddf["surface_abs_error"]

        lines.append("DISTANCE")
        lines.append(f"n = {len(ddf)}")
        lines.append(f"Surface MAE  = {ddf['surface_abs_error'].mean():.6f}")
        lines.append(f"Centroid MAE = {ddf['centroid_abs_error'].mean():.6f}")
        lines.append(f"Surface MedAE  = {np.median(ddf['surface_abs_error']):.6f}")
        lines.append(f"Centroid MedAE = {np.median(ddf['centroid_abs_error']):.6f}")
        lines.append(f"Mean improvement (centroid - surface) = {ddf['centroid_minus_surface'].mean():.6f}")
        lines.append("")

    ndf = df[df["operator"] == "nearest"].copy()
    if len(ndf) > 0:
        ndf["surface_correct"] = ndf["surface_pred_answer_object"] == ndf["gt_answer_object"]
        ndf["centroid_correct"] = ndf["centroid_pred_answer_object"] == ndf["gt_answer_object"]

        lines.append("NEAREST")
        lines.append(f"n = {len(ndf)}")
        lines.append(f"Surface Top-1  = {ndf['surface_correct'].mean():.6f}")
        lines.append(f"Centroid Top-1 = {ndf['centroid_correct'].mean():.6f}")
        lines.append(f"Centroid failures = {(~ndf['centroid_correct']).sum()}")
        lines.append("")

    df.to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved per-query comparison: {OUT_CSV}")
    print(f"Saved summary: {OUT_TXT}")

if __name__ == "__main__":
    main()