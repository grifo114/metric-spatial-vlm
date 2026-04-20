from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark" / "objects_manifest_dev_official.csv"
OUT_DIR = ROOT / "benchmark" / "scene_objects_dev_official"

TARGET_COLS = [
    "scene_id",
    "object_id",
    "label_norm",
    "n_points",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "extent_x",
    "extent_y",
    "extent_z",
]

def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["is_valid_object"] == True].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scene_id, g in df.groupby("scene_id"):
        g = g[TARGET_COLS].sort_values(["label_norm", "object_id"])
        out_path = OUT_DIR / f"{scene_id}_valid_objects.csv"
        g.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()