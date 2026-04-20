from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import is_between_xy, is_aligned_xy

MANIFEST = ROOT / "benchmark" / "objects_manifest_dev_official.csv"
DATASET = ROOT / "benchmark" / "queries_dev_official_relational_binary_labeled_repaired.csv"

OUT_CSV = ROOT / "results" / "benchmark_v1" / "dev_official_relational_threshold_sensitivity.csv"
OUT_TXT = ROOT / "results" / "benchmark_v1" / "dev_official_relational_threshold_sensitivity.txt"

TAU_BETWEEN_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
TAU_ALIGN_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

def centroid_xyz(row: pd.Series) -> np.ndarray:
    return row[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {
        "n": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

def main():
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["is_valid_object"] == True].copy()
    dataset = pd.read_csv(DATASET)

    by_id = {row["object_id"]: row for _, row in manifest.iterrows()}
    rows = []

    between_df = dataset[dataset["operator"] == "between"].copy()
    if len(between_df) > 0:
        for tau in TAU_BETWEEN_VALUES:
            y_true = []
            y_pred = []

            for _, q in between_df.iterrows():
                row_x = by_id[q["object_x"]]
                row_a = by_id[q["object_a"]]
                row_b = by_id[q["object_b"]]

                cx = centroid_xyz(pd.Series(row_x))
                ca = centroid_xyz(pd.Series(row_a))
                cb = centroid_xyz(pd.Series(row_b))

                pred = int(is_between_xy(cx, ca, cb, tau_between=float(tau)))

                y_true.append(int(q["binary_label"]))
                y_pred.append(pred)

            y_true = np.asarray(y_true, dtype=int)
            y_pred = np.asarray(y_pred, dtype=int)

            m = binary_metrics(y_true, y_pred)
            rows.append({
                "operator": "between",
                "tau": tau,
                **m,
            })

    aligned_df = dataset[dataset["operator"] == "aligned"].copy()
    if len(aligned_df) > 0:
        for tau in TAU_ALIGN_VALUES:
            y_true = []
            y_pred = []

            for _, q in aligned_df.iterrows():
                row_a = by_id[q["object_a"]]
                row_b = by_id[q["object_b"]]
                row_c = by_id[q["object_c"]]

                ca = centroid_xyz(pd.Series(row_a))
                cb = centroid_xyz(pd.Series(row_b))
                cc = centroid_xyz(pd.Series(row_c))

                pred = int(is_aligned_xy(ca, cb, cc, tau_align=float(tau)))

                y_true.append(int(q["binary_label"]))
                y_pred.append(pred)

            y_true = np.asarray(y_true, dtype=int)
            y_pred = np.asarray(y_pred, dtype=int)

            m = binary_metrics(y_true, y_pred)
            rows.append({
                "operator": "aligned",
                "tau": tau,
                **m,
            })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    lines = []
    for op in ["between", "aligned"]:
        sub = out_df[out_df["operator"] == op].copy()
        lines.append(op.upper())
        lines.append(sub.to_string(index=False))
        lines.append("")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(out_df.to_string(index=False))
    print()
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved TXT: {OUT_TXT}")

if __name__ == "__main__":
    main()