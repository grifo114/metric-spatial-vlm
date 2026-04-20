from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DN_PATH = ROOT / "benchmark" / "queries_dev_distance_nearest_final.csv"
BA_PATH = ROOT / "benchmark" / "queries_dev_between_aligned_final.csv"

OUT_CSV = ROOT / "benchmark" / "queries_dev_master.csv"
OUT_JSONL = ROOT / "benchmark" / "queries_dev_master.jsonl"

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
    df_dn = pd.read_csv(DN_PATH)
    df_ba = pd.read_csv(BA_PATH)

    df_dn = normalize_columns(df_dn)
    df_ba = normalize_columns(df_ba)

    master = pd.concat([df_dn, df_ba], ignore_index=True)

    master = master.sort_values(["operator", "scene_id", "structured_query"]).reset_index(drop=True)
    master.insert(0, "query_id", [f"dev_q_{i:04d}" for i in range(len(master))])

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

if __name__ == "__main__":
    main()