from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
import webbrowser
import numpy as np
import pandas as pd
from plyfile import PlyData
import plotly.graph_objects as go

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


def load_scene_points(scene_id: str, max_points: int = 120000) -> np.ndarray:
    scene_dir = SCANS_DIR / scene_id
    ply_files = sorted(scene_dir.glob("*vh_clean_2.ply"))
    if not ply_files:
        raise FileNotFoundError(f"PLY da cena não encontrado para {scene_id}")

    ply = PlyData.read(str(ply_files[0]))
    vertex = ply["vertex"]
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )

    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]

    return xyz


def make_box_trace(row: pd.Series, color: str, width: float, opacity: float, name: str):
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
        name=name,
        showlegend=False,
    )


def render_scene(scene_id: str, df_scene: pd.DataFrame, answer_text: str, role_map: dict[str, str], out_html: Path):
    pts = load_scene_points(scene_id)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.8, color="lightgray", opacity=0.40),
            name="scene",
            showlegend=False,
        )
    )

    default_color = "#3b82f6"
    role_colors = {
        "query_a": "#ef4444",
        "query_b": "#22c55e",
        "query_c": "#f59e0b",
        "reference": "#f59e0b",
        "answer": "#ef4444",
        "candidate": "#22c55e",
    }

    # caixas
    for _, row in df_scene.iterrows():
        alias = row["alias"]
        role = role_map.get(alias)
        color = role_colors.get(role, default_color)
        width = 10 if role else 2
        opacity = 1.0 if role else 0.28

        fig.add_trace(make_box_trace(row, color=color, width=width, opacity=opacity, name=alias))

    # labels
    label_colors = []
    for _, row in df_scene.iterrows():
        alias = row["alias"]
        role = role_map.get(alias)
        label_colors.append(role_colors.get(role, default_color))

    fig.add_trace(
        go.Scatter3d(
            x=df_scene["centroid_x"],
            y=df_scene["centroid_y"],
            z=df_scene["centroid_z"],
            mode="text",
            text=df_scene["alias"],
            textfont=dict(size=10, color=label_colors),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"{scene_id}<br><sup>{answer_text}</sup>",
        margin=dict(l=0, r=0, t=70, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    webbrowser.open(out_html.resolve().as_uri())


def answer_query(df_scene: pd.DataFrame, query: str):
    by_alias = {row["alias"]: row for _, row in df_scene.iterrows()}
    text = normalize_text(query)

    # distance
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

        answer = f"Resposta: a distância entre {a1} e {a2} é {d:.4f} m."
        roles = {a1: "query_a", a2: "query_b"}
        return answer, roles

    # nearest
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

        answer = f"Resposta: o objeto da categoria {category} mais próximo de {ref_alias} é {best_alias}, a {best_d:.4f} m."
        roles = {ref_alias: "reference", best_alias: "answer"}
        return answer, roles

    # between
    m = re.match(r"([a-z0-9_]+) esta entre ([a-z0-9_]+) e ([a-z0-9_]+)\?", text)
    if m:
        x_alias, a_alias, b_alias = m.group(1), m.group(2), m.group(3)
        if x_alias not in by_alias or a_alias not in by_alias or b_alias not in by_alias:
            return "Um ou mais aliases não foram encontrados.", {}

        cx = centroid_xyz(pd.Series(by_alias[x_alias]))
        ca = centroid_xyz(pd.Series(by_alias[a_alias]))
        cb = centroid_xyz(pd.Series(by_alias[b_alias]))
        pred = is_between_xy(cx, ca, cb, tau_between=TAU_BETWEEN)

        answer = f"Resposta: {'sim' if pred else 'não'}, {x_alias} {'está' if pred else 'não está'} entre {a_alias} e {b_alias}."
        roles = {x_alias: "query_a", a_alias: "query_b", b_alias: "query_c"}
        return answer, roles

    # aligned
    m = re.match(r"([a-z0-9_]+), ([a-z0-9_]+) e ([a-z0-9_]+) estao alinhados\?", text)
    if m:
        a1, a2, a3 = m.group(1), m.group(2), m.group(3)
        if a1 not in by_alias or a2 not in by_alias or a3 not in by_alias:
            return "Um ou mais aliases não foram encontrados.", {}

        c1 = centroid_xyz(pd.Series(by_alias[a1]))
        c2 = centroid_xyz(pd.Series(by_alias[a2]))
        c3 = centroid_xyz(pd.Series(by_alias[a3]))
        pred = is_aligned_xy(c1, c2, c3, tau_align=TAU_ALIGN)

        answer = f"Resposta: {'sim' if pred else 'não'}, {a1}, {a2} e {a3} {'estão' if pred else 'não estão'} alinhados."
        roles = {a1: "query_a", a2: "query_b", a3: "query_c"}
        return answer, roles

    return "Consulta fora do padrão esperado.", {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()

    alias_path = ALIAS_DIR / f"{args.scene_id}_alias_map.csv"
    if not alias_path.exists():
        raise FileNotFoundError(f"Alias map não encontrado: {alias_path}")

    alias_df = pd.read_csv(alias_path)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[(manifest["scene_id"] == args.scene_id) & (manifest["is_valid_object"] == True)].copy()

    df_scene = alias_df.merge(
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

    out_html = OUT_DIR / f"{args.scene_id}_visual_demo.html"

    print(f"Cena: {args.scene_id}")
    print("Aliases disponíveis:")
    print(df_scene[['alias', 'label_norm']].to_string(index=False))
    print()
    print("Exemplos:")
    print("  qual a distancia entre cabinet2 e table4?")
    print("  qual chair esta mais proximo de cabinet1?")
    print("  chair5 esta entre desk1 e table4?")
    print("  cabinet1, chair2 e table3 estao alinhados?")
    print()
    print("Digite 'sair' para encerrar.")
    print()

    render_scene(args.scene_id, df_scene, "Cena carregada. Aguardando consulta.", {}, out_html)

    while True:
        q = input("Consulta> ").strip()
        if q.lower() in {"sair", "exit", "quit"}:
            break
        if not q:
            continue

        answer, roles = answer_query(df_scene, q)
        render_scene(args.scene_id, df_scene, answer, roles, out_html)
        print(answer)
        print(f"Visual atualizado em: {out_html}\n")


if __name__ == "__main__":
    main()