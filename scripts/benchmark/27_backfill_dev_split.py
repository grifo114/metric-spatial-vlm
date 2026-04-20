from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV_CSV = ROOT / "benchmark" / "scenes_dev_official.csv"
DEV_TXT = ROOT / "benchmark" / "scenes_dev_official.txt"
POOL = ROOT / "benchmark" / "scene_candidate_pool_ranked.csv"

TARGET_TOTAL = 20

def family_id(scene_id: str) -> str:
    return scene_id.split("_")[0]

def main():
    dev = pd.read_csv(DEV_CSV)
    pool = pd.read_csv(POOL)

    if "family_id" not in dev.columns:
        dev["family_id"] = dev["scene_id"].apply(family_id)

    pool["family_id"] = pool["scene_id"].apply(family_id)

    used_scene_ids = set(dev["scene_id"])
    used_families = set(dev["family_id"])

    remaining = pool[
        (~pool["scene_id"].isin(used_scene_ids)) &
        (~pool["family_id"].isin(used_families))
    ].copy()

    remaining = remaining.sort_values(
        ["selection_score", "n_valid_objects", "n_valid_categories", "scene_id"],
        ascending=[False, False, False, True]
    )

    need = TARGET_TOTAL - len(dev)
    if need <= 0:
        print("Dev split already has 20 or more scenes.")
        return

    add_rows = remaining.head(need).copy()

    final_df = pd.concat([dev, add_rows], ignore_index=True)
    final_df = final_df.sort_values(
        ["room_group", "selection_score", "scene_id"],
        ascending=[True, False, True]
    )

    final_df.to_csv(DEV_CSV, index=False)
    DEV_TXT.write_text("\n".join(final_df["scene_id"].tolist()), encoding="utf-8")

    print(f"Added {len(add_rows)} scene(s).")
    print()
    print("Added scenes:")
    print(add_rows[[
        "scene_id",
        "family_id",
        "scene_type",
        "room_group",
        "selection_score",
        "n_valid_objects",
        "n_valid_categories",
    ]].to_string(index=False))
    print()
    print("Final counts by room_group:")
    print(final_df["room_group"].value_counts(dropna=False))
    print()
    print("Unique families:", final_df["family_id"].nunique())
    print("Total scenes:", len(final_df))

if __name__ == "__main__":
    main()