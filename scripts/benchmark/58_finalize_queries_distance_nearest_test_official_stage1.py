from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_test_official_stage1_distance_nearest_review_filled.csv"
OUT_PATH = ROOT / "benchmark" / "queries_test_official_stage1_distance_nearest_final.csv"

def main():
    df = pd.read_csv(IN_PATH)

    df["review_keep"] = df["review_keep"].fillna("").astype(str).str.strip().str.lower()
    final_df = df[df["review_keep"] == "yes"].copy()

    final_df.to_csv(OUT_PATH, index=False)

    print(f"Saved final queries: {OUT_PATH}")
    print()
    print(final_df["operator"].value_counts(dropna=False))
    print()
    print(final_df[["scene_id", "operator", "structured_query"]].to_string(index=False))
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_test_official_stage1_distance_nearest_review_filled.csv"
OUT_PATH = ROOT / "benchmark" / "queries_test_official_stage1_distance_nearest_final.csv"

def main():
    df = pd.read_csv(IN_PATH)

    df["review_keep"] = df["review_keep"].fillna("").astype(str).str.strip().str.lower()
    final_df = df[df["review_keep"] == "yes"].copy()

    final_df.to_csv(OUT_PATH, index=False)

    print(f"Saved final queries: {OUT_PATH}")
    print()
    print(final_df["operator"].value_counts(dropna=False))
    print()
    print(final_df[["scene_id", "operator", "structured_query"]].to_string(index=False))

if __name__ == "__main__":
    main()