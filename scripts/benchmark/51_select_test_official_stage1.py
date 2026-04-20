from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "benchmark" / "scene_test_candidate_pool.csv"
OUT_CSV = ROOT / "benchmark" / "scenes_test_official_stage1.csv"
OUT_TXT = ROOT / "benchmark" / "scenes_test_official_stage1.txt"
OUT_FAMILIES = ROOT / "benchmark" / "scenes_test_official_stage1_families.txt"

TARGETS = {
    "bedroom": 5,
    "living_room": 3,
    "office_study": 6,
    "mixed_other": 6,
}

def family_id(scene_id: str) -> str:
    return scene_id.split("_")[0]

def choose_best_per_family(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "family_id" not in work.columns:
        work["family_id"] = work["scene_id"].apply(family_id)

    work = work.sort_values(
        ["family_id", "selection_score", "n_valid_objects", "n_valid_categories", "scene_id"],
        ascending=[True, False, False, False, True]
    )

    best = work.groupby("family_id", as_index=False).first()
    best = best.sort_values(
        ["room_group", "selection_score", "n_valid_objects", "n_valid_categories", "scene_id"],
        ascending=[True, False, False, False, True]
    )
    return best

def main():
    df = pd.read_csv(IN_PATH).copy()

    if "family_id" not in df.columns:
        df["family_id"] = df["scene_id"].apply(family_id)

    best = choose_best_per_family(df)

    selected_rows = []
    used_families = set()

    # seleção estratificada por grupo
    for group, target_n in TARGETS.items():
        sub = best[best["room_group"] == group].copy()

        chosen = 0
        for _, row in sub.iterrows():
            fam = row["family_id"]
            if fam in used_families:
                continue
            selected_rows.append(row)
            used_families.add(fam)
            chosen += 1
            if chosen >= target_n:
                break

        print(f"{group}: selected {chosen} / target {target_n}")

    selected = pd.DataFrame(selected_rows).copy()

    # backfill caso algum grupo fique curto
    need = sum(TARGETS.values()) - len(selected)
    if need > 0:
        remaining = best[~best["family_id"].isin(used_families)].copy()
        remaining = remaining.sort_values(
            ["selection_score", "n_valid_objects", "n_valid_categories", "scene_id"],
            ascending=[False, False, False, True]
        )
        add_rows = remaining.head(need).copy()
        selected = pd.concat([selected, add_rows], ignore_index=True)
        used_families.update(add_rows["family_id"].tolist())
        print(f"Backfill added: {len(add_rows)} scene(s)")

    selected = selected.sort_values(
        ["room_group", "selection_score", "scene_id"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    selected.to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(selected["scene_id"].tolist()), encoding="utf-8")
    OUT_FAMILIES.write_text("\n".join(selected["family_id"].tolist()), encoding="utf-8")

    print()
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved TXT: {OUT_TXT}")
    print(f"Saved families TXT: {OUT_FAMILIES}")
    print()
    print(selected[[
        "scene_id",
        "family_id",
        "scene_type",
        "room_group",
        "selection_score",
        "n_valid_objects",
        "n_valid_categories",
    ]].to_string(index=False))
    print()
    print("Counts by room_group:")
    print(selected["room_group"].value_counts(dropna=False))
    print()
    print("Unique families:", selected["family_id"].nunique())
    print("Total scenes:", len(selected))

if __name__ == "__main__":
    main()