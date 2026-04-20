from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "benchmark" / "scene_inventory.csv"
OUT_PATH = ROOT / "benchmark" / "scene_candidate_pool_ranked.csv"


def normalize_room_group(scene_type: str) -> str:
    s = str(scene_type).strip().lower()

    if any(k in s for k in ["bedroom", "hotel"]):
        return "bedroom"

    if any(k in s for k in ["living room", "lounge", "family room"]):
        return "living_room"

    if any(k in s for k in ["office", "study", "conference", "meeting room", "computer lab", "classroom"]):
        return "office_study"

    return "mixed_other"


def build_score(df: pd.DataFrame) -> pd.Series:
    score = 0.0

    score += 2.0 * df["n_valid_categories"]
    score += 0.5 * df["n_valid_objects"]

    score += 1.5 * (df["n_valid_chair"] >= 2).astype(float)
    score += 1.5 * ((df["n_valid_table"] + df["n_valid_desk"]) >= 2).astype(float)
    score += 1.0 * ((df["n_valid_door"] + df["n_valid_cabinet"] + df["n_valid_monitor"]) >= 1).astype(float)

    score += 0.5 * (df["n_valid_sofa"] >= 1).astype(float)
    score += 0.5 * (df["n_valid_bed"] >= 1).astype(float)
    score += 0.5 * (df["n_valid_monitor"] >= 1).astype(float)

    return score


def main():
    df = pd.read_csv(IN_PATH)

    df = df[df["status"] == "ok"].copy()
    df = df[df["is_candidate"] == True].copy()

    df["room_group"] = df["scene_type"].apply(normalize_room_group)
    df["selection_score"] = build_score(df)

    keep_cols = [
        "scene_id",
        "scene_type",
        "room_group",
        "n_valid_objects",
        "n_valid_categories",
        "n_valid_chair",
        "n_valid_table",
        "n_valid_desk",
        "n_valid_sofa",
        "n_valid_bed",
        "n_valid_door",
        "n_valid_cabinet",
        "n_valid_monitor",
        "selection_score",
    ]
    df = df[keep_cols].sort_values(
        ["room_group", "selection_score", "n_valid_objects", "n_valid_categories", "scene_id"],
        ascending=[True, False, False, False, True]
    )

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved ranked pool: {OUT_PATH}")
    print()
    print("Candidate counts by room_group:")
    print(df["room_group"].value_counts(dropna=False))
    print()
    print("Top 10 overall:")
    print(df.sort_values(["selection_score", "n_valid_objects"], ascending=[False, False]).head(10).to_string(index=False))
    print()

    for group in ["bedroom", "living_room", "office_study", "mixed_other"]:
        sub = df[df["room_group"] == group].head(10)
        print(f"Top scenes for {group}:")
        if len(sub) == 0:
            print("  none")
        else:
            print(sub.to_string(index=False))
        print()


if __name__ == "__main__":
    main()