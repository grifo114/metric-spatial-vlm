"""
07_evaluate.py

Consolidates all predictions and computes final evaluation metrics.

Inputs:  results/predictions_baseline_a.csv
         results/predictions_baseline_b.csv
         results/predictions_proposed_rgbd.csv
Output:  results/metrics_final.csv
         results/metrics_by_range.csv
         results/metrics_summary.txt
"""

import os
import numpy as np
import pandas as pd

OUTPUT_DIR = "results"

FILES = {
    "Baseline A — Pixel 2D":       "results/predictions_baseline_a.csv",
    "Baseline B — DPT Monocular":  "results/predictions_baseline_b.csv",
    "Proposed — RGBD Fusion":      "results/predictions_proposed_rgbd.csv",
}

def compute_metrics(df):
    valid = df[df["status"] == "ok"].copy()
    if len(valid) == 0:
        return {}
    valid["abs_error"] = (valid["pred_distance_m"] - valid["gt_distance_m"]).abs()
    valid["rel_error"] = valid["abs_error"] / valid["gt_distance_m"] * 100
    return {
        "n_total":   len(df),
        "n_valid":   len(valid),
        "n_failed":  len(df) - len(valid),
        "pct_valid": round(len(valid) / len(df) * 100, 1),
        "MAE":       round(valid["abs_error"].mean(), 4),
        "MedianAE":  round(valid["abs_error"].median(), 4),
        "RMSE":      round(np.sqrt((valid["abs_error"]**2).mean()), 4),
        "RelErr":    round(valid["rel_error"].mean(), 2),
        "MAE_short":   round(valid[valid["range"]=="short"]["abs_error"].mean(), 4)
                       if len(valid[valid["range"]=="short"]) > 0 else None,
        "MAE_medium":  round(valid[valid["range"]=="medium"]["abs_error"].mean(), 4)
                       if len(valid[valid["range"]=="medium"]) > 0 else None,
        "MAE_long":    round(valid[valid["range"]=="long"]["abs_error"].mean(), 4)
                       if len(valid[valid["range"]=="long"]) > 0 else None,
        "RelErr_short":  round(valid[valid["range"]=="short"]["rel_error"].mean(), 2)
                         if len(valid[valid["range"]=="short"]) > 0 else None,
        "RelErr_medium": round(valid[valid["range"]=="medium"]["rel_error"].mean(), 2)
                         if len(valid[valid["range"]=="medium"]) > 0 else None,
        "RelErr_long":   round(valid[valid["range"]=="long"]["rel_error"].mean(), 2)
                         if len(valid[valid["range"]=="long"]) > 0 else None,
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    all_valid   = []

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    for method_name, fpath in FILES.items():
        df = pd.read_csv(fpath)
        df["method_name"] = method_name
        m  = compute_metrics(df)

        all_metrics.append({"method": method_name, **m})

        valid = df[df["status"] == "ok"].copy()
        valid["abs_error"] = (valid["pred_distance_m"] - valid["gt_distance_m"]).abs()
        valid["rel_error"] = valid["abs_error"] / valid["gt_distance_m"] * 100
        valid["method_name"] = method_name
        all_valid.append(valid)

        print(f"\n{method_name}")
        print(f"  Pairs evaluated : {m['n_total']}")
        print(f"  Valid           : {m['n_valid']} ({m['pct_valid']}%)")
        print(f"  MAE             : {m['MAE']} m")
        print(f"  Median AE       : {m['MedianAE']} m")
        print(f"  RMSE            : {m['RMSE']} m")
        print(f"  Mean Rel. Error : {m['RelErr']} %")
        print(f"  By range:")
        for rng in ["short", "medium", "long"]:
            mae = m.get(f"MAE_{rng}")
            rel = m.get(f"RelErr_{rng}")
            if mae is not None:
                print(f"    {rng:<8}  MAE={mae:.3f} m  RelErr={rel:.1f}%")

    # Improvement over Baseline A
    metrics_df = pd.DataFrame(all_metrics)

    mae_a    = metrics_df.loc[metrics_df.method.str.contains("Baseline A"), "MAE"].values[0]
    mae_b    = metrics_df.loc[metrics_df.method.str.contains("Baseline B"), "MAE"].values[0]
    mae_rgbd = metrics_df.loc[metrics_df.method.str.contains("RGBD"),       "MAE"].values[0]

    imp_vs_a = (mae_a - mae_rgbd) / mae_a * 100
    imp_vs_b = (mae_b - mae_rgbd) / mae_b * 100

    print(f"\n{'─'*60}")
    print(f"IMPROVEMENT — Proposed RGBD vs Baseline A : {imp_vs_a:.1f}% MAE reduction")
    print(f"IMPROVEMENT — Proposed RGBD vs Baseline B : {imp_vs_b:.1f}% MAE reduction")
    print(f"{'─'*60}")

    # Save summary CSV
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_final.csv"), index=False)

    # Save by-range CSV
    if all_valid:
        combined = pd.concat(all_valid, ignore_index=True)
        by_range = combined.groupby(["method_name", "range"]).agg(
            n=("abs_error", "count"),
            MAE=("abs_error", "mean"),
            MedianAE=("abs_error", "median"),
            RMSE=("abs_error", lambda x: np.sqrt((x**2).mean())),
            RelErr=("rel_error", "mean"),
        ).round(4).reset_index()
        by_range.to_csv(os.path.join(OUTPUT_DIR, "metrics_by_range.csv"), index=False)

    # Save text summary
    summary = []
    summary.append("FINAL EVALUATION RESULTS")
    summary.append("=" * 60)
    summary.append(f"Dataset     : ScanNet v2")
    summary.append(f"Scenes      : 20")
    summary.append(f"Total pairs : 162")
    summary.append("")
    for row in all_metrics:
        summary.append(f"{row['method']}")
        summary.append(f"  MAE={row['MAE']} m  RelErr={row['RelErr']}%  Valid={row['n_valid']}/{row['n_total']}")
    summary.append("")
    summary.append(f"Proposed RGBD vs Baseline A: -{imp_vs_a:.1f}% MAE")
    summary.append(f"Proposed RGBD vs Baseline B: -{imp_vs_b:.1f}% MAE")

    with open(os.path.join(OUTPUT_DIR, "metrics_summary.txt"), "w") as f:
        f.write("\n".join(summary))

    print(f"\nOutputs saved to results/:")
    print(f"  metrics_final.csv")
    print(f"  metrics_by_range.csv")
    print(f"  metrics_summary.txt")

if __name__ == "__main__":
    main()
