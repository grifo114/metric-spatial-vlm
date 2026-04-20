from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

POOL_PATH = ROOT / "benchmark" / "scene_candidate_pool_ranked.csv"
DEV_PATH = ROOT / "benchmark" / "scenes_dev_official.csv"

OUT_POOL = ROOT / "benchmark" / "scene_test_candidate_pool.csv"
OUT_FAMILIES = ROOT / "benchmark" / "scene_test_candidate_families.txt"

def family_id(scene_id: str) -> str:
    return scene_id.split("_")[0]

def main():
    pool = pd.read_csv(POOL_PATH)
    dev = pd.read_csv(DEV_PATH)

    pool["family_id"] = pool["scene_id"].apply(family_id)
    dev["family_id"] = dev["scene_id"].apply(family_id)

    dev_families = set(dev["family_id"])
    remaining = pool[~pool["family_id"].isin(dev_families)].copy()

    remaining = remaining.sort_values(
        ["room_group", "selection_score", "n_valid_objects", "n_valid_categories", "scene_id"],
        ascending=[True, False, False, False, True]
    )

    remaining.to_csv(OUT_POOL, index=False)

    fams = sorted(remaining["family_id"].unique())
    OUT_FAMILIES.write_text("\n".join(fams), encoding="utf-8")

    print(f"Saved pool: {OUT_POOL}")
    print(f"Saved families: {OUT_FAMILIES}")
    print()
    print("Remaining candidate scenes by room_group:")
    print(remaining["room_group"].value_counts(dropna=False))
    print()
    print("Remaining unique families by room_group:")
    print(remaining.groupby("room_group")["family_id"].nunique())
    print()
    print("Top 20 remaining candidates:")
    print(remaining.head(20).to_string(index=False))
    print()
    print("Total remaining candidate scenes:", len(remaining))
    print("Total remaining unique families:", remaining["family_id"].nunique())

if __name__ == "__main__":
    main()