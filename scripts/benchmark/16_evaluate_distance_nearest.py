from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "benchmark" / "ground_truth_distance_nearest.csv"
PRED = ROOT / "results" / "pilot" / "benchmark_distance_nearest_predictions.csv"
OUT_METRICS = ROOT / "results" / "pilot" / "benchmark_distance_nearest_metrics.txt"

def main():
    gt = pd.read_csv(GT)
    pred = pd.read_csv(PRED)

    df = gt.merge(pred, on=["query_id", "scene_id", "operator", "structured_query"], how="inner")

    lines = []

    # distance
    ddf = df[df["operator"] == "distance"].copy()
    if len(ddf) > 0:
        err = np.abs(ddf["pred_distance_m"] - ddf["gt_distance_m"])
        mae = float(err.mean())
        medae = float(np.median(err))
        rmse = float(np.sqrt(np.mean((ddf["pred_distance_m"] - ddf["gt_distance_m"]) ** 2)))

        lines.append("DISTANCE")
        lines.append(f"n = {len(ddf)}")
        lines.append(f"MAE   = {mae:.6f}")
        lines.append(f"MedAE = {medae:.6f}")
        lines.append(f"RMSE  = {rmse:.6f}")
        lines.append("")

    # nearest
    ndf = df[df["operator"] == "nearest"].copy()
    if len(ndf) > 0:
        acc = float((ndf["pred_answer_object"] == ndf["gt_answer_object"]).mean())

        lines.append("NEAREST")
        lines.append(f"n = {len(ndf)}")
        lines.append(f"Top-1 Accuracy = {acc:.6f}")
        lines.append("")

    text = "\n".join(lines)
    OUT_METRICS.write_text(text, encoding="utf-8")

    print(text)
    print(f"Saved metrics: {OUT_METRICS}")

if __name__ == "__main__":
    main()