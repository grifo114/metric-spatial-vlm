from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"
OUT_DIR = ROOT / "benchmark" / "demo_alias_maps"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()

    df = pd.read_csv(MANIFEST)
    df = df[(df["scene_id"] == args.scene_id) & (df["is_valid_object"] == True)].copy()

    if len(df) == 0:
        raise RuntimeError(f"Nenhum objeto válido encontrado para {args.scene_id}")

    df = df.sort_values(["label_norm", "object_id"]).reset_index(drop=True)

    counters = {}
    aliases = []

    for _, row in df.iterrows():
        label = row["label_norm"]
        counters[label] = counters.get(label, 0) + 1
        alias = f"{label}{counters[label]}"
        aliases.append(alias)

    df["alias"] = aliases

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.scene_id}_alias_map.csv"
    df[[
        "scene_id",
        "alias",
        "object_id",
        "label_norm",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "points_path",
    ]].to_csv(out_path, index=False)

    print(f"Saved alias map: {out_path}")
    print()
    print(df[["alias", "label_norm", "object_id"]].to_string(index=False))

if __name__ == "__main__":
    main()