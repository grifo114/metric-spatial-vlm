from pathlib import Path
import pandas as pd

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.geometry.geometry_ops import (
    load_points_npz,
    centroid_distance,
    surface_distance,
    is_between_xy,
    is_aligned_xy,
)

MANIFEST = ROOT / "benchmark" / "objects_manifest_dev.csv"


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["is_valid_object"] == True].copy()

    # exemplo usando scene0114_00
    scene_df = df[df["scene_id"] == "scene0114_00"].copy()

    chair = scene_df[scene_df["object_id"] == "scene0114_00__chair_004"].iloc[0]
    table = scene_df[scene_df["object_id"] == "scene0114_00__table_018"].iloc[0]
    desk = scene_df[scene_df["object_id"] == "scene0114_00__desk_013"].iloc[0]

    pts_chair = load_points_npz(ROOT / chair["points_path"])
    pts_table = load_points_npz(ROOT / table["points_path"])
    pts_desk = load_points_npz(ROOT / desk["points_path"])

    d_cent = centroid_distance(pts_chair, pts_table)
    d_surf = surface_distance(pts_chair, pts_table)

    print("scene0114_00")
    print(f"chair: {chair['object_id']}")
    print(f"table: {table['object_id']}")
    print(f"desk : {desk['object_id']}")
    print()
    print(f"centroid distance = {d_cent:.4f} m")
    print(f"surface distance  = {d_surf:.4f} m")
    print()

    c_chair = chair[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
    c_table = table[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
    c_desk = desk[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)

    print(f"is_between_xy(chair, desk, table) = {is_between_xy(c_chair, c_desk, c_table)}")
    print(f"is_aligned_xy(chair, desk, table) = {is_aligned_xy(c_chair, c_desk, c_table)}")


if __name__ == "__main__":
    main()