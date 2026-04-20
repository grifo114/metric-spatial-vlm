from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import (
    load_points_npz,
    surface_distance,
    is_between_xy,
    is_aligned_xy,
)

ALIAS_DIR = ROOT / "benchmark" / "demo_alias_maps"
MANIFEST_PATH = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"
SCANS_DIR = ROOT / "data" / "scannet" / "scans"
OUT_DIR = ROOT / "artifacts" / "demo_visual"

TAU_BETWEEN = 0.30
TAU_ALIGN = 0.25


def get_scene_id() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene_id", default="scene0142_00")
    args, _ = parser.parse_known_args()
    return args.scene_id


def normalize_text(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("distância", "distancia")
        .replace("próximo", "proximo")
        .replace("está", "esta")
        .replace("estão", "estao")
    )


def centroid_xyz(row: pd.Series) -> np.ndarray:
    return row[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)


@st.cache_data
def load_scene_df(scene_id: str) -> pd.DataFrame:
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


@st.cache_data
def load_scene_mesh_arrays(scene_id: str):
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

    faces = None
    if "face" in ply:
        faces_raw = ply["face"].data["vertex_indices"]
        packed = []
        for f in faces_raw:
            f = list(f)
            packed.extend([len(f)] + f)
        faces = np.asarray(packed, dtype=np.int64)

    rgb = None
    names = set(vertex.data.dtype.names or [])
    if {"red", "green", "blue"}.issubset(names):
        r = np.asarray(vertex["red"], dtype=np.uint8)
        g = np.asarray(vertex["green"], dtype=np.uint8)
        b = np.asarray(vertex["blue"], dtype=np.uint8)
        rgb = np.column_stack([r, g, b])

    return points, faces, rgb


def build_plotter(scene_id: str, df_scene: pd.DataFrame, role_map: dict[str, str], show_all_boxes: bool):
    points, faces, rgb = load_scene_mesh_arrays(scene_id)

    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    plotter.set_background("white")

    mesh = pv.PolyData(points)
    if faces is not None:
        mesh.faces = faces

    if rgb is not None:
        mesh["rgb"] = rgb
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

    role_colors = {
        "query_a": "red",
        "query_b": "limegreen",
        "query_c": "orange",
        "reference": "orange",
        "answer": "red",
    }

    # caixas
    for _, row in df_scene.iterrows():
        alias = row["alias"]
        highlight = alias in role_map

        if not show_all_boxes and not highlight:
            continue

        bounds = (
            float(row["aabb_min_x"]),
            float(row["aabb_max_x"]),
            float(row["aabb_min_y"]),
            float(row["aabb_max_y"]),
            float(row["aabb_min_z"]),
            float(row["aabb_max_z"]),
        )
        box = pv.Box(bounds=bounds)

        if highlight:
            color = role_colors.get(role_map[alias], "red")
            line_width = 5
            opacity = 1.0
        else:
            color = "lightskyblue"
            line_width = 1.0
            opacity = 0.10

        plotter.add_mesh(
            box,
            style="wireframe",
            color=color,
            line_width=line_width,
            opacity=opacity,
        )

    # labels só dos destacados
    if role_map:
        label_points = []
        label_texts = []

        for idx, alias in enumerate(role_map.keys()):
            sub = df_scene[df_scene["alias"] == alias]
            if len(sub) == 0:
                continue

            row = sub.iloc[0]
            z_offset = 0.18 + 0.12 * (idx % 3)

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
                font_size=14,
                point_size=0,
                text_color="black",
                shape_color="white",
                shape_opacity=0.65,
                margin=2,
                always_visible=True,
            )

    plotter.camera_position = "iso"
    plotter.enable_anti_aliasing()
    return plotter


def render_scene_png(scene_id: str, df_scene: pd.DataFrame, role_map: dict[str, str], show_all_boxes: bool) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{scene_id}_streamlit_render.png"

    plotter = build_plotter(scene_id, df_scene, role_map, show_all_boxes)
    plotter.show(screenshot=str(out_path), auto_close=True)
    return out_path


def answer_query(df_scene: pd.DataFrame, query: str):
    by_alias = {row["alias"]: row for _, row in df_scene.iterrows()}
    text = normalize_text(query)

    m = re.match(r"qual a distancia entre ([a-z0-9_]+) e ([a-z0-9_]+)\?", text)
    if m:
        a1, a2 = m.group(1), m.group(2)
        if a1 not in by_alias or a2 not in by_alias:
            return "Alias não encontrado.", {}

        r1 = by_alias[a1]
        r2 = by_alias[a2]
        p1 = load_points_npz(ROOT / r1["points_path"])
        p2 = load_points_npz(ROOT / r2["points_path"])
        d = surface_distance(p1, p2)

        return f"Resposta: a distância entre {a1} e {a2} é {d:.4f} m.", {
            a1: "query_a",
            a2: "query_b",
        }

    m = re.match(r"qual ([a-z_]+) esta mais proximo de ([a-z0-9_]+)\?", text)
    if m:
        category, ref_alias = m.group(1), m.group(2)
        if ref_alias not in by_alias:
            return "Alias de referência não encontrado.", {}

        ref_row = by_alias[ref_alias]
        ref_points = load_points_npz(ROOT / ref_row["points_path"])

        sub = df_scene[df_scene["label_norm"] == category].copy()
        if len(sub) == 0:
            return f"Não há objetos da categoria {category} nesta cena.", {}

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
            return "Nenhum candidato válido encontrado.", {}

        return f"Resposta: o objeto da categoria {category} mais próximo de {ref_alias} é {best_alias}, a {best_d:.4f} m.", {
            ref_alias: "reference",
            best_alias: "answer",
        }

    m = re.match(r"([a-z0-9_]+) esta entre ([a-z0-9_]+) e ([a-z0-9_]+)\?", text)
    if m:
        x_alias, a_alias, b_alias = m.group(1), m.group(2), m.group(3)
        if x_alias not in by_alias or a_alias not in by_alias or b_alias not in by_alias:
            return "Um ou mais aliases não foram encontrados.", {}

        cx = centroid_xyz(pd.Series(by_alias[x_alias]))
        ca = centroid_xyz(pd.Series(by_alias[a_alias]))
        cb = centroid_xyz(pd.Series(by_alias[b_alias]))
        pred = is_between_xy(cx, ca, cb, tau_between=TAU_BETWEEN)

        return f"Resposta: {'sim' if pred else 'não'}, {x_alias} {'está' if pred else 'não está'} entre {a_alias} e {b_alias}.", {
            x_alias: "query_a",
            a_alias: "query_b",
            b_alias: "query_c",
        }

    m = re.match(r"([a-z0-9_]+), ([a-z0-9_]+) e ([a-z0-9_]+) estao alinhados\?", text)
    if m:
        a1, a2, a3 = m.group(1), m.group(2), m.group(3)
        if a1 not in by_alias or a2 not in by_alias or a3 not in by_alias:
            return "Um ou mais aliases não foram encontrados.", {}

        c1 = centroid_xyz(pd.Series(by_alias[a1]))
        c2 = centroid_xyz(pd.Series(by_alias[a2]))
        c3 = centroid_xyz(pd.Series(by_alias[a3]))
        pred = is_aligned_xy(c1, c2, c3, tau_align=TAU_ALIGN)

        return f"Resposta: {'sim' if pred else 'não'}, {a1}, {a2} e {a3} {'estão' if pred else 'não estão'} alinhados.", {
            a1: "query_a",
            a2: "query_b",
            a3: "query_c",
        }

    return "Consulta fora do padrão esperado.", {}


scene_id = get_scene_id()
df_scene = load_scene_df(scene_id)

st.set_page_config(layout="wide")
st.title("Demo visual de consultas espaciais em cena 3D")
st.caption(f"Cena: {scene_id}")

if "answer_text" not in st.session_state:
    st.session_state.answer_text = "Cena carregada. Aguardando consulta."
    st.session_state.role_map = {}

left, right = st.columns([3, 1])

with right:
    st.subheader("Consulta")
    query = st.text_input(
        "Digite a consulta",
        value="qual a distancia entre cabinet2 e table4?"
    )
    show_all_boxes = st.checkbox("Mostrar caixas de todos os objetos", value=True)
    run = st.button("Executar consulta", type="primary")

    st.subheader("Aliases disponíveis")
    st.dataframe(
        df_scene[["alias", "label_norm"]],
        use_container_width=True,
        height=600,
    )

if run:
    answer_text, role_map = answer_query(df_scene, query)
    st.session_state.answer_text = answer_text
    st.session_state.role_map = role_map

img_path = render_scene_png(
    scene_id=scene_id,
    df_scene=df_scene,
    role_map=st.session_state.role_map,
    show_all_boxes=show_all_boxes,
)

with left:
    st.image(str(img_path), use_container_width=True)

st.markdown(f"**{st.session_state.answer_text}**")