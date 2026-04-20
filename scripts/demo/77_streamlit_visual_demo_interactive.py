from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def triangulate_faces(face_indices) -> np.ndarray:
    triangles = []
    for face in face_indices:
        f = list(face)
        if len(f) < 3:
            continue
        if len(f) == 3:
            triangles.append(f)
        else:
            for j in range(1, len(f) - 1):
                triangles.append([f[0], f[j], f[j + 1]])
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


@st.cache_data
def load_mesh_for_plotly(scene_id: str, max_triangles: int = 500000):
    scene_dir = SCANS_DIR / scene_id
    ply_files = sorted(scene_dir.glob("*vh_clean_2.ply"))
    if not ply_files:
        raise FileNotFoundError(f"PLY da cena não encontrado para {scene_id}")

    ply = PlyData.read(str(ply_files[0]))
    vertex = ply["vertex"]

    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)

    triangles = np.empty((0, 3), dtype=np.int32)
    if "face" in ply:
        face_indices = ply["face"].data["vertex_indices"]
        triangles = triangulate_faces(face_indices)

    if len(triangles) > max_triangles:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(triangles), size=max_triangles, replace=False)
        triangles = triangles[keep]

    names = set(vertex.data.dtype.names or [])
    facecolor = None
    if {"red", "green", "blue"}.issubset(names) and len(triangles) > 0:
        r = np.asarray(vertex["red"], dtype=np.float32)
        g = np.asarray(vertex["green"], dtype=np.float32)
        b = np.asarray(vertex["blue"], dtype=np.float32)

        tri_rgb = np.stack(
            [
                r[triangles].mean(axis=1),
                g[triangles].mean(axis=1),
                b[triangles].mean(axis=1),
            ],
            axis=1,
        ).astype(np.uint8)

        facecolor = [f"rgb({rr},{gg},{bb})" for rr, gg, bb in tri_rgb]

    return x, y, z, triangles, facecolor


def make_box_trace(row: pd.Series, color: str, width: float, opacity: float):
    x0, y0, z0 = row["aabb_min_x"], row["aabb_min_y"], row["aabb_min_z"]
    x1, y1, z1 = row["aabb_max_x"], row["aabb_max_y"], row["aabb_max_z"]

    corners = {
        0: (x0, y0, z0),
        1: (x1, y0, z0),
        2: (x1, y1, z0),
        3: (x0, y1, z0),
        4: (x0, y0, z1),
        5: (x1, y0, z1),
        6: (x1, y1, z1),
        7: (x0, y1, z1),
    }

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    xs, ys, zs = [], [], []
    for a, b in edges:
        xa, ya, za = corners[a]
        xb, yb, zb = corners[b]
        xs += [xa, xb, None]
        ys += [ya, yb, None]
        zs += [za, zb, None]

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        showlegend=False,
        hoverinfo="skip",
    )


def build_figure(scene_id: str, df_scene: pd.DataFrame, answer_text: str, role_map: dict[str, str], show_all_boxes: bool):
    x, y, z, triangles, facecolor = load_mesh_for_plotly(scene_id)

    fig = go.Figure()

    if len(triangles) > 0:
        mesh_kwargs = dict(
            x=x,
            y=y,
            z=z,
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1, roughness=0.9),
            lightposition=dict(x=100, y=200, z=300),
            name="scene",
            showscale=False,
            hoverinfo="skip",
        )
        if facecolor is not None:
            mesh_kwargs["facecolor"] = facecolor
        else:
            mesh_kwargs["color"] = "lightgray"

        fig.add_trace(go.Mesh3d(**mesh_kwargs))
    else:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=1.2, color="gray", opacity=0.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    role_colors = {
        "query_a": "#ef4444",
        "query_b": "#22c55e",
        "query_c": "#f59e0b",
        "reference": "#f59e0b",
        "answer": "#ef4444",
    }

    for _, row in df_scene.iterrows():
        alias = row["alias"]
        highlight = alias in role_map

        if not show_all_boxes and not highlight:
            continue

        if highlight:
            color = role_colors.get(role_map[alias], "#ef4444")
            width = 10
            opacity = 1.0
        else:
            color = "rgba(30,144,255,0.35)"
            width = 2
            opacity = 0.22

        fig.add_trace(make_box_trace(row, color=color, width=width, opacity=opacity))

    # labels apenas dos objetos destacados
    if role_map:
        label_points = []
        label_texts = []
        label_colors = []

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
            label_colors.append("black")

        arr = np.asarray(label_points, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                mode="text",
                text=label_texts,
                textfont=dict(size=14, color=label_colors),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"{scene_id}<br><sup>{answer_text}</sup>",
        margin=dict(l=0, r=0, t=70, b=0),
        height=850,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
    )
    return fig


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
st.title("Demo visual interativa de consultas espaciais em cena 3D")
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

fig = build_figure(
    scene_id=scene_id,
    df_scene=df_scene,
    answer_text=st.session_state.answer_text,
    role_map=st.session_state.role_map,
    show_all_boxes=show_all_boxes,
)

with left:
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"**{st.session_state.answer_text}**")