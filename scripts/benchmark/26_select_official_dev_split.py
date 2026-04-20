from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "scene_candidate_pool_ranked.csv"
OUT_DEV = ROOT / "benchmark" / "scenes_dev_official.txt"
OUT_DEV_CSV = ROOT / "benchmark" / "scenes_dev_official.csv"

TARGETS = {
    "bedroom": 4,
    "living_room": 5,
    "office_study": 5,
    "mixed_other": 6,
}

def family_id(scene_id: str) -> str:
    # scene0006_02 -> scene0006
    return scene_id.split("_")[0]

def main():
    df = pd.read_csv(IN_PATH).copy()
    df["family_id"] = df["scene_id"].apply(family_id)

    selected_rows = []
    used_families = set()

    # seleção estratificada por grupo
    for group, target_n in TARGETS.items():
        sub = df[df["room_group"] == group].copy()

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

    # ordenar para ficar estável
    selected = selected.sort_values(["room_group", "selection_score", "scene_id"], ascending=[True, False, True])

    selected.to_csv(OUT_DEV_CSV, index=False)
    OUT_DEV.write_text("\n".join(selected["scene_id"].tolist()), encoding="utf-8")

    print()
    print(f"Saved dev CSV: {OUT_DEV_CSV}")
    print(f"Saved dev TXT: {OUT_DEV}")
    print()
    print(selected[[
        "scene_id",
        "family_id",
        "scene_type",
        "room_group",
        "selection_score",
        "n_valid_objects",
        "n_valid_categories"
    ]].to_string(index=False))
    print()
    print("Counts by room_group:")
    print(selected["room_group"].value_counts(dropna=False))
    print()
    print("Unique families:", selected["family_id"].nunique())
    print("Total scenes:", len(selected))

if __name__ == "__main__":
    main()