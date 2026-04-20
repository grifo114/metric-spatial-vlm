from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

A_PATH = ROOT / "benchmark" / "queries_dev_official_distance_nearest_final.csv"
B_PATH = ROOT / "benchmark" / "queries_dev_official_between_aligned_final.csv"

OUT_CSV = ROOT / "benchmark" / "queries_dev_official_master.csv"
OUT_JSONL = ROOT / "benchmark" / "queries_dev_official_master.jsonl"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "scene_id",
        "operator",
        "structured_query",
        "natural_query",
        "status",
        "review_keep",
        "review_reason",
        "review_notes",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df

def main():
    df_a = pd.read_csv(A_PATH)
    df_b = pd.read_csv(B_PATH)

    df_a = normalize_columns(df_a)
    df_b = normalize_columns(df_b)

    master = pd.concat([df_a, df_b], ignore_index=True)
    master = master.sort_values(["operator", "scene_id", "structured_query"]).reset_index(drop=True)

    master.insert(0, "query_id", [f"dev_official_q_{i:04d}" for i in range(len(master))])

    master.to_csv(OUT_CSV, index=False)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for _, row in master.iterrows():
            f.write(row.to_json(force_ascii=False) + "\n")

    print(f"Saved master CSV:   {OUT_CSV}")
    print(f"Saved master JSONL: {OUT_JSONL}")
    print()
    print("Queries by operator:")
    print(master["operator"].value_counts(dropna=False))
    print()
    print("Queries by scene:")
    print(master["scene_id"].value_counts().sort_index())
    print()
    print("Total queries:", len(master))

if __name__ == "__main__":
    main()