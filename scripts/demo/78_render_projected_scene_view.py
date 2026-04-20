from __future__ import annotations

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import pyvista as pv
from plyfile import PlyData
from PIL import Image, ImageDraw, ImageFont

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


def box_corners(row: pd.Series) -> np.ndarray:
    x0, y0, z0 = row["aabb_min_x"], row["aabb_min_y"], row["aabb_min_z"]
    x1, y1, z1 = row["aabb_max_x"], row["aabb_max_y"], row["aabb_max_z"]
    return np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def project_points(
    pts_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    cam_up: np.ndarray,
    view_angle_deg: float,
    width: int,
    height: int,
):
    forward = normalize(cam_target - cam_pos)
    right = normalize(np.cross(forward, cam_up))
    true_up = normalize(np.cross(right, forward))

    rel = pts_world - cam_pos[None, :]
    x_cam = rel @ right
    y_cam = rel @ true_up
    z_cam = rel @ forward

    valid = z_cam > 1e-6

    fy = 0.5 * height / math.tan(math.radians(view_angle_deg) / 2.0)
    fx = fy

    u = fx * (x_cam / np.maximum(z_cam, 1e-6)) + width / 2.0
    v = -fy * (y_cam / np.maximum(z_cam, 1e-6)) + height / 2.0

    proj = np.column_stack([u, v, z_cam])
    return proj, valid


def render_base_image(mesh: pv.PolyData, width: int, height: int, out_png: Path):
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
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

    plotter.camera_position = "iso"
    plotter.enable_anti_aliasing()
    plotter.show(screenshot=str(out_png), auto_close=False)

    cam = plotter.camera
    cam_pos = np.array(cam.position, dtype=float)
    cam_target = np.array(cam.focal_point, dtype=float)
    cam_up = np.array(cam.up, dtype=float)
    view_angle = float(cam.view_angle)

    plotter.close()

    return cam_pos, cam_target, cam_up, view_angle


def clamp_box(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    return x0, y0, x1, y1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--highlight", nargs="*", default=[])
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--show_labels", action="store_true")
    args = parser.parse_args()

    scene_id = args.scene_id
    highlight_aliases = set(args.highlight)

    mesh = load_scene_mesh(scene_id)
    df = build_scene_df(scene_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_png = OUT_DIR / f"{scene_id}_base_render.png"
    final_png = OUT_DIR / f"{scene_id}_projected_boxes.png"

    cam_pos, cam_target, cam_up, view_angle = render_base_image(
        mesh=mesh,
        width=args.width,
        height=args.height,
        out_png=base_png,
    )

    img = Image.open(base_png).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    visible_count = 0

    for _, row in df.iterrows():
        alias = row["alias"]
        corners = box_corners(row)

        proj, valid = project_points(
            pts_world=corners,
            cam_pos=cam_pos,
            cam_target=cam_target,
            cam_up=cam_up,
            view_angle_deg=view_angle,
            width=args.width,
            height=args.height,
        )

        if valid.sum() < 4:
            continue

        uv = proj[valid][:, :2]

        x0 = float(np.min(uv[:, 0]))
        y0 = float(np.min(uv[:, 1]))
        x1 = float(np.max(uv[:, 0]))
        y1 = float(np.max(uv[:, 1]))

        if x1 < 0 or y1 < 0 or x0 >= args.width or y0 >= args.height:
            continue

        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, args.width, args.height)

        if (x1 - x0) < 6 or (y1 - y0) < 6:
            continue

        visible_count += 1

        if alias in highlight_aliases:
            color = (220, 20, 60, 255)
            width = 4
        else:
            color = (30, 144, 255, 170)
            width = 2

        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

        if args.show_labels:
            tx = x0 + 2
            ty = max(0, y0 - 12)
            text_box = [tx - 1, ty - 1, tx + 8 * len(alias), ty + 10]
            draw.rectangle(text_box, fill=(255, 255, 255, 180))
            draw.text((tx, ty), alias, fill=(0, 0, 0, 255), font=font)

    title = f"{scene_id} | visible objects projected: {visible_count}"
    tx, ty = 12, 12
    draw.rectangle([tx - 4, ty - 4, tx + 10 * len(title), ty + 16], fill=(255, 255, 255, 200))
    draw.text((tx, ty), title, fill=(0, 0, 0, 255), font=font)

    if highlight_aliases:
        subtitle = "highlight: " + ", ".join(sorted(highlight_aliases))
        sy = 34
        draw.rectangle([tx - 4, sy - 4, tx + 10 * len(subtitle), sy + 16], fill=(255, 255, 255, 200))
        draw.text((tx, sy), subtitle, fill=(160, 0, 0, 255), font=font)

    img.save(final_png)
    print(f"Base render: {base_png}")
    print(f"Projected view: {final_png}")
    print(f"Visible projected objects: {visible_count}")


if __name__ == "__main__":
    main()