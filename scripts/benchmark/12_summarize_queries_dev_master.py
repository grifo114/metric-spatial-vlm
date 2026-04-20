from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "queries_dev_master.csv"
OUT_SUMMARY = ROOT / "benchmark" / "queries_dev_master_summary.csv"

def main():
    df = pd.read_csv(IN_PATH)

    op_scene = (
        df.groupby(["scene_id", "operator"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    op_scene["total"] = op_scene.drop(columns=["scene_id"]).sum(axis=1)
    op_scene = op_scene.sort_values(["total", "scene_id"], ascending=[False, True])

    op_scene.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved summary: {OUT_SUMMARY}")
    print()
    print(op_scene.to_string(index=False))
    print()
    print("Total queries:", len(df))
    print("By operator:")
    print(df["operator"].value_counts(dropna=False))

if __name__ == "__main__":
    main()