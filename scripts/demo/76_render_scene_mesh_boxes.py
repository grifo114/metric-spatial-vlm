from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]

ALIAS_DIR = ROOT / "benchmark" / "demo_alias_maps"
MANIFEST_PATH = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"
SCANS_DIR = ROOT / "data" / "scannet" / "scans"
OUT_DIR = ROOT / "artifacts" / "demo_visual"


def load_scene_mesh(scene_id: str) -> pv.PolyData:
    scene_dir = SCANS_DIR / scene_id
    ply_files = sorted(scene_dir.glob("*vh_clean_2.ply"))
    if not ply_files:
        raise FileNotFoundError(f"PLY da cena não encontrado para {scene_id}")

    ply = PlyData.read(str(ply_files[0]))

    vertex = ply["vertex"]
    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    points = np.column_stack([x, y, z])

    mesh = pv.PolyData(points)

    if "face" in ply:
        faces_raw = ply["face"].data["vertex_indices"]
        faces = []
        for f in faces_raw:
            f = list(f)
            faces.append([len(f)] + f)
        mesh.faces = np.hstack(faces).astype(np.int64)

    color_names = set(vertex.data.dtype.names or [])
    if {"red", "green", "blue"}.issubset(color_names):
        r = np.asarray(vertex["red"], dtype=np.uint8)
        g = np.asarray(vertex["green"], dtype=np.uint8)
        b = np.asarray(vertex["blue"], dtype=np.uint8)
        rgb = np.column_stack([r, g, b])
        mesh["rgb"] = rgb

    return mesh


def build_scene_df(scene_id: str) -> pd.DataFrame:
    alias_path = ALIAS_DIR / f"{scene_id}_alias_map.csv"
    if not alias_path.exists():
        raise FileNotFoundError(f"Alias map não encontrado: {alias_path}")

    alias_df = pd.read_csv(alias_path)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[
        (manifest["scene_id"] == scene_id) &
        (manifest["is_valid_object"] == True)
    ].copy()

    df = alias_df.merge(
        manifest[
            [
                "object_id",
                "label_norm",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "aabb_min_x",
                "aabb_min_y",
                "aabb_min_z",
                "aabb_max_x",
                "aabb_max_y",
                "aabb_max_z",
            ]
        ],
        on=["object_id", "label_norm", "centroid_x", "centroid_y", "centroid_z"],
        how="left",
    )

    return df


def make_box(row: pd.Series) -> pv.PolyData:
    bounds = (
        float(row["aabb_min_x"]),
        float(row["aabb_max_x"]),
        float(row["aabb_min_y"]),
        float(row["aabb_max_y"]),
        float(row["aabb_min_z"]),
        float(row["aabb_max_z"]),
    )
    return pv.Box(bounds=bounds)


def label_offset(alias: str, idx: int) -> float:
    base = 0.18
    extra = 0.12 * (idx % 3)
    return base + extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--highlight", nargs="*", default=[])
    parser.add_argument("--show_labels", action="store_true")
    parser.add_argument("--save_screenshot", action="store_true")
    args = parser.parse_args()

    scene_id = args.scene_id
    highlight_aliases = list(dict.fromkeys(args.highlight))
    highlight_set = set(highlight_aliases)

    mesh = load_scene_mesh(scene_id)
    df = build_scene_df(scene_id)

    plotter = pv.Plotter(window_size=(1600, 1000))
    plotter.set_background("white")

    if "rgb" in mesh.array_names:
        plotter.add_mesh(
            mesh,
            scalars="rgb",
            rgb=True,
            lighting=True,
            smooth_shading=False,
            show_edges=False,
        )
    else:
        plotter.add_mesh(
            mesh,
            color="lightgray",
            lighting=True,
            smooth_shading=False,
            show_edges=False,
        )

    # Bounding boxes
    for _, row in df.iterrows():
        alias = row["alias"]
        box = make_box(row)

        if alias in highlight_set:
            color = "red"
            line_width = 5
            opacity = 1.0
        else:
            color = "lightskyblue"
            line_width = 1.0
            opacity = 0.12

        plotter.add_mesh(
            box,
            style="wireframe",
            color=color,
            line_width=line_width,
            opacity=opacity,
        )

    # Labels apenas dos objetos destacados
    if args.show_labels and highlight_aliases:
        label_points = []
        label_texts = []

        for idx, alias in enumerate(highlight_aliases):
            sub = df[df["alias"] == alias]
            if len(sub) == 0:
                continue

            row = sub.iloc[0]
            z_offset = label_offset(alias, idx)

            label_points.append([
                float(row["centroid_x"]),
                float(row["centroid_y"]),
                float(row["centroid_z"] + z_offset),
            ])
            label_texts.append(alias)

        if label_points:
            plotter.add_point_labels(
                np.asarray(label_points),
                label_texts,
                font_size=18,
                point_size=0,
                text_color="black",
                shape_color="white",
                shape_opacity=0.65,
                margin=2,
                always_visible=True,
            )

    title = f"{scene_id}"
    if highlight_aliases:
        title += " | highlight: " + ", ".join(highlight_aliases)

    plotter.add_text(title, position="upper_left", font_size=12, color="black")

    plotter.enable_anti_aliasing()
    plotter.show_grid(color="lightgray")
    plotter.camera_position = "iso"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.save_screenshot:
        shot_path = OUT_DIR / f"{scene_id}_mesh_boxes.png"
        plotter.show(screenshot=str(shot_path))
        print(f"Screenshot salva em: {shot_path}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()