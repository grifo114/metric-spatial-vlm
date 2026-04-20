from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_dev_master.csv"

OUT_A = ROOT / "benchmark" / "queries_dev_regression_retrieval.csv"
OUT_B = ROOT / "benchmark" / "queries_dev_relational_binary.csv"

def main():
    df = pd.read_csv(IN_PATH)

    df_a = df[df["operator"].isin(["distance", "nearest"])].copy()
    df_b = df[df["operator"].isin(["between", "aligned"])].copy()

    df_a.to_csv(OUT_A, index=False)
    df_b.to_csv(OUT_B, index=False)

    print(f"Saved: {OUT_A} ({len(df_a)} queries)")
    print(f"Saved: {OUT_B} ({len(df_b)} queries)")
    print()
    print("Block A:")
    print(df_a["operator"].value_counts(dropna=False))
    print()
    print("Block B:")
    print(df_b["operator"].value_counts(dropna=False))

if __name__ == "__main__":
    main()