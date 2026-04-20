from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_dev_official_relational_binary_labeled.csv"
OUT_PATH = ROOT / "benchmark" / "queries_dev_official_relational_binary_labeled_repaired.csv"

def main():
    df = pd.read_csv(IN_PATH)

    df["pairing_group"] = df["pairing_group"].astype(str)
    df["binary_label"] = df["binary_label"].astype(int)

    df["binary_query_id"] = df["pairing_group"] + df["binary_label"].map({1: "__pos", 0: "__neg"})

    if df["binary_query_id"].duplicated().any():
        raise RuntimeError("Duplicated binary_query_id found after repair.")

    cols = ["binary_query_id"] + [c for c in df.columns if c != "binary_query_id"]
    df = df[cols]

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved repaired binary dataset: {OUT_PATH}")
    print()
    print(df[["binary_query_id", "pairing_group", "operator", "binary_label", "structured_query"]].head(20).to_string(index=False))
    print()
    print("Unique binary_query_id:", df["binary_query_id"].nunique())
    print("Total rows:", len(df))

if __name__ == "__main__":
    main()  