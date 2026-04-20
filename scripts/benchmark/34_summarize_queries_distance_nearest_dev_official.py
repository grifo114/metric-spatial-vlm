from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_dev_official_distance_nearest_final.csv"
OUT_PATH = ROOT / "benchmark" / "queries_dev_official_distance_nearest_summary.csv"

def main():
    df = pd.read_csv(IN_PATH)

    summary = (
        df.groupby(["scene_id", "operator"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    summary["total"] = summary.drop(columns=["scene_id"]).sum(axis=1)
    summary = summary.sort_values(["total", "scene_id"], ascending=[False, True])

    summary.to_csv(OUT_PATH, index=False)

    print(f"Saved summary: {OUT_PATH}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Total queries:", len(df))
    print(df["operator"].value_counts(dropna=False))

if __name__ == "__main__":
    main()