from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_dev_distance_nearest_review.csv"
OUT_REPAIRED = ROOT / "benchmark" / "queries_dev_distance_nearest_review_repaired.csv"
OUT_FINAL = ROOT / "benchmark" / "queries_dev_distance_nearest_final.csv"

def main():
    df = pd.read_csv(IN_PATH)

    for col in ["review_keep", "review_reason", "review_notes"]:
        if col not in df.columns:
            df[col] = ""

    df["review_keep"] = df["review_keep"].fillna("").astype(str).str.strip().str.lower()
    df["review_notes"] = df["review_notes"].fillna("").astype(str).str.strip().str.lower()

    # mover yes/no da coluna review_notes para review_keep quando review_keep estiver vazio
    mask_move = df["review_keep"].eq("") & df["review_notes"].isin(["yes", "no", "yse"])
    df.loc[mask_move, "review_keep"] = df.loc[mask_move, "review_notes"]

    # corrigir typo
    df["review_keep"] = df["review_keep"].replace({"yse": "yes"})

    # correções metodológicas obrigatórias
    forced_no = {
        'nearest("scene0030_00__door_069", "chair")',
        'nearest("scene0030_00__table_051", "chair")',
        'nearest("scene0114_00__cabinet_010", "chair")',
        'distance("scene0011_00__monitor_018", "scene0011_00__table_017")',
        'distance("scene0050_00__desk_005", "scene0050_00__door_008")',
    }

    df.loc[df["structured_query"].isin(forced_no), "review_keep"] = "no"

    df.to_csv(OUT_REPAIRED, index=False)

    final_df = df[df["review_keep"] == "yes"].copy()
    final_df.to_csv(OUT_FINAL, index=False)

    print(f"Saved repaired review sheet: {OUT_REPAIRED}")
    print(f"Saved final queries: {OUT_FINAL}")
    print()
    print(final_df["operator"].value_counts(dropna=False))
    print()
    print(final_df[["scene_id", "operator", "structured_query"]].to_string(index=False))

if __name__ == "__main__":
    main()