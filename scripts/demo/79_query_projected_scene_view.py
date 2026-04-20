from __future__ import annotations

from pathlib import Path
import argparse
import math
import re
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

TAU_BETWEEN = 0.30
TAU_ALIGN = 0.25


def normalize_text(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("distância", "distancia")
        .replace("próximo", "proximo")
        .replace("está", "esta")
        .replace("estão", "estao")
    )


def slugify_query(text: str, max_len: int = 80) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len] if text else "query"


def load_points_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    return data["points"]


def surface_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    best = float("inf")

    for pa in points_a:
        d2 = np.sum((points_b - pa[None, :]) ** 2, axis=1)
        local = float(np.sqrt(d2.min()))
        if local < best:
            best = local

    for pb in points_b:
        d2 = np.sum((points_a - pb[None, :]) ** 2, axis=1)
        local = float(np.sqrt(d2.min()))
        if local < best:
            best = local

    return best


def is_between_xy(cx: np.ndarray, ca: np.ndarray, cb: np.ndarray, tau_between: float = 0.30) -> bool:
    ax, ay = ca[0], ca[1]
    bx, by = cb[0], cb[1]
    xx, xy = cx[0], cx[1]

    ab = np.array([bx - ax, by - ay], dtype=float)
    axv = np.array([xx - ax, xy - ay], dtype=float)

    ab_norm2 = float(np.dot(ab, ab))
    if ab_norm2 < 1e-12:
        return False

    t = float(np.dot(axv, ab) / ab_norm2)
    if t < 0.0 or t > 1.0:
        return False

    proj = np.array([ax, ay], dtype=float) + t * ab
    perp = float(np.linalg.norm(np.array([xx, xy], dtype=float) - proj))
    return perp <= tau_between


def is_aligned_xy(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, tau_align: float = 0.25) -> bool:
    p1 = np.array([c1[0], c1[1]], dtype=float)
    p2 = np.array([c2[0], c2[1]], dtype=float)
    p3 = np.array([c3[0], c3[1]], dtype=float)

    v = p3 - p1
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return False

    w = p2 - p1
    dist = abs(v[0] * w[1] - v[1] * w[0]) / n
    return float(dist) <= tau_align


def centroid_xyz(row: pd.Series) -> np.ndarray:
    return row[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)


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
                "points_path",
            ]
        ],
        on=["object_id", "label_norm", "centroid_x", "centroid_y", "centroid_z", "points_path"],
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
    return v if n < 1e-12 else v / n


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


def answer_query(df_scene: pd.DataFrame, query: str):
    by_alias = {row["alias"]: row for _, row in df_scene.iterrows()}
    text = normalize_text(query)

    m = re.match(r"qual a distancia entre ([a-z0-9_]+) e ([a-z0-9_]+)\?", text)
    if m:
        a1, a2 = m.group(1), m.group(2)
        if a1 not in by_alias or a2 not in by_alias:
            return "Alias não encontrado.", {a1, a2}

        r1 = by_alias[a1]
        r2 = by_alias[a2]
        p1 = load_points_npz(ROOT / r1["points_path"])
        p2 = load_points_npz(ROOT / r2["points_path"])
        d = surface_distance(p1, p2)
        return f"Resposta: a distância entre {a1} e {a2} é {d:.4f} m.", {a1, a2}

    m = re.match(r"qual ([a-z_]+) esta mais proximo de ([a-z0-9_]+)\?", text)
    if m:
        category, ref_alias = m.group(1), m.group(2)
        if ref_alias not in by_alias:
            return "Alias de referência não encontrado.", {ref_alias}

        ref_row = by_alias[ref_alias]
        ref_points = load_points_npz(ROOT / ref_row["points_path"])

        sub = df_scene[df_scene["label_norm"] == category].copy()
        if len(sub) == 0:
            return f"Não há objetos da categoria {category} nesta cena.", {ref_alias}

        best_alias = None
        best_d = float("inf")
        for _, row in sub.iterrows():
            alias = row["alias"]
            if alias == ref_alias:
                continue
            pts = load_points_npz(ROOT / row["points_path"])
            d = surface_distance(ref_points, pts)
            if d < best_d:
                best_d = d
                best_alias = alias

        if best_alias is None:
            return "Nenhum candidato válido encontrado.", {ref_alias}

        return f"Resposta: o objeto da categoria {category} mais próximo de {ref_alias} é {best_alias}, a {best_d:.4f} m.", {ref_alias, best_alias}

    m = re.match(r"([a-z0-9_]+) esta entre ([a-z0-9_]+) e ([a-z0-9_]+)\?", text)
    if m:
        x_alias, a_alias, b_alias = m.group(1), m.group(2), m.group(3)
        if x_alias not in by_alias or a_alias not in by_alias or b_alias not in by_alias:
            return "Um ou mais aliases não foram encontrados.", {x_alias, a_alias, b_alias}

        cx = centroid_xyz(pd.Series(by_alias[x_alias]))
        ca = centroid_xyz(pd.Series(by_alias[a_alias]))
        cb = centroid_xyz(pd.Series(by_alias[b_alias]))
        pred = is_between_xy(cx, ca, cb, tau_between=TAU_BETWEEN)
        return f"Resposta: {'sim' if pred else 'não'}, {x_alias} {'está' if pred else 'não está'} entre {a_alias} e {b_alias}.", {x_alias, a_alias, b_alias}

    m = re.match(r"([a-z0-9_]+), ([a-z0-9_]+) e ([a-z0-9_]+) estao alinhados\?", text)
    if m:
        a1, a2, a3 = m.group(1), m.group(2), m.group(3)
        if a1 not in by_alias or a2 not in by_alias or a3 not in by_alias:
            return "Um ou mais aliases não foram encontrados.", {a1, a2, a3}

        c1 = centroid_xyz(pd.Series(by_alias[a1]))
        c2 = centroid_xyz(pd.Series(by_alias[a2]))
        c3 = centroid_xyz(pd.Series(by_alias[a3]))
        pred = is_aligned_xy(c1, c2, c3, tau_align=TAU_ALIGN)
        return f"Resposta: {'sim' if pred else 'não'}, {a1}, {a2} e {a3} {'estão' if pred else 'não estão'} alinhados.", {a1, a2, a3}

    return "Consulta fora do padrão esperado.", set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--show_all_labels", action="store_true")
    args = parser.parse_args()

    scene_id = args.scene_id
    query = args.query

    mesh = load_scene_mesh(scene_id)
    df = build_scene_df(scene_id)

    answer_text, highlight_aliases = answer_query(df, query)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    query_slug = slugify_query(query)

    base_png = OUT_DIR / f"{scene_id}_query_base.png"
    final_png = OUT_DIR / f"{scene_id}_{query_slug}.png"

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
            color = (30, 144, 255, 110)
            width = 2

        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

        if args.show_all_labels or alias in highlight_aliases:
            tx = x0 + 2
            ty = max(0, y0 - 12)
            text_box = [tx - 1, ty - 1, tx + 8 * len(alias), ty + 10]
            draw.rectangle(text_box, fill=(255, 255, 255, 190))
            draw.text((tx, ty), alias, fill=(0, 0, 0, 255), font=font)

    title = f"{scene_id} | visible objects projected: {visible_count}"
    tx, ty = 12, 12
    draw.rectangle([tx - 4, ty - 4, tx + 10 * len(title), ty + 16], fill=(255, 255, 255, 210))
    draw.text((tx, ty), title, fill=(0, 0, 0, 255), font=font)

    answer_y = 36
    answer_box_right = min(args.width - 10, tx + 8 * len(answer_text) + 10)
    draw.rectangle(
        [tx - 4, answer_y - 4, answer_box_right, answer_y + 16],
        fill=(255, 255, 255, 220),
    )
    draw.text((tx, answer_y), answer_text, fill=(120, 0, 0, 255), font=font)

    img.save(final_png)

    print(f"Query: {query}")
    print(answer_text)
    print(f"Imagem gerada: {final_png}")
    print(f"Objetos destacados: {sorted(highlight_aliases)}")


if __name__ == "__main__":
    main()