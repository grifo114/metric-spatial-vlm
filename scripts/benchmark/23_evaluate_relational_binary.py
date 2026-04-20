from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GT = ROOT / "benchmark" / "queries_dev_relational_binary_labeled_repaired.csv"
PRED = ROOT / "results" / "pilot" / "benchmark_relational_binary_predictions.csv"
OUT_TXT = ROOT / "results" / "pilot" / "benchmark_relational_binary_metrics.txt"

def binary_metrics(df: pd.DataFrame):
    tp = int(((df["binary_label"] == 1) & (df["pred_label"] == 1)).sum())
    tn = int(((df["binary_label"] == 0) & (df["pred_label"] == 0)).sum())
    fp = int(((df["binary_label"] == 0) & (df["pred_label"] == 1)).sum())
    fn = int(((df["binary_label"] == 1) & (df["pred_label"] == 0)).sum())

    acc = (tp + tn) / len(df) if len(df) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {
        "n": len(df),
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
    gt = pd.read_csv(GT)
    pred = pd.read_csv(PRED)

    df = gt.merge(
        pred,
        on=["binary_query_id", "scene_id", "operator", "structured_query"],
        how="inner"
    )

    lines = []

    overall = binary_metrics(df)
    lines.append("OVERALL")
    for k, v in overall.items():
        lines.append(f"{k} = {v}")
    lines.append("")

    for op in ["between", "aligned"]:
        sub = df[df["operator"] == op].copy()
        m = binary_metrics(sub)
        lines.append(op.upper())
        for k, v in m.items():
            lines.append(f"{k} = {v}")
        lines.append("")

    text = "\n".join(lines)
    OUT_TXT.write_text(text, encoding="utf-8")

    print(text)
    print(f"Saved metrics: {OUT_TXT}")

if __name__ == "__main__":
    main()