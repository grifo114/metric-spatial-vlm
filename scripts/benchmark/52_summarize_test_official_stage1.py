from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "scenes_test_official_stage1.csv"

def main():
    df = pd.read_csv(IN_PATH)

    print("Counts by room_group:")
    print(df["room_group"].value_counts(dropna=False))
    print()
    print("Counts by family:")
    print(df["family_id"].nunique())
    print()
    print(df[[
        "scene_id",
        "family_id",
        "scene_type",
        "room_group",
        "selection_score",
        "n_valid_objects",
        "n_valid_categories",
    ]].to_string(index=False))

if __name__ == "__main__":
    main()