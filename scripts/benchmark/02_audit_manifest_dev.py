from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark" / "objects_manifest_dev.csv"
OUT_SUMMARY = ROOT / "benchmark" / "scene_manifest_summary_dev.csv"

TARGET_COLS = ["chair", "table", "desk", "sofa", "bed", "door", "cabinet", "monitor"]

def main():
    df = pd.read_csv(MANIFEST)

    valid_df = df[df["is_valid_object"] == True].copy()

    summary = (
        valid_df.groupby(["scene_id", "label_norm"])
        .size()
        .unstack(fill_value=0)
    )

    for col in TARGET_COLS:
        if col not in summary.columns:
            summary[col] = 0

    summary = summary[TARGET_COLS].copy()

    summary["n_surface"] = summary["table"] + summary["desk"]
    summary["n_furniture"] = summary["chair"] + summary["sofa"] + summary["bed"]
    summary["n_reference"] = summary["door"] + summary["cabinet"] + summary["monitor"]

    summary["supports_distance"] = summary["n_surface"] >= 1
    summary["supports_nearest"] = (summary["n_surface"] >= 1) & (summary["n_furniture"] >= 2)
    summary["supports_between"] = summary["n_surface"] >= 2
    summary["supports_aligned"] = (summary["n_surface"] + summary["n_furniture"] + summary["n_reference"]) >= 3

    summary = summary.reset_index().sort_values(
        ["supports_between", "supports_nearest", "supports_distance", "scene_id"],
        ascending=[False, False, False, True]
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved summary: {OUT_SUMMARY}\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()