#!/usr/bin/env python3
"""
90_generate_human_annotation.py

Gera imagens anotadas e CSV template para validação humana do bloco
relacional (between e aligned) do test_official_stage1.

Seleciona 40 exemplos estratificados (10 between-pos, 10 between-neg,
10 aligned-pos, 10 aligned-neg) e para cada um:
  - Renderiza a cena com os objetos relevantes destacados
  - Sobrepõe a natural_query como legenda na imagem
  - Exporta a imagem numerada (ex: "ann_001_between_pos.jpg")
  - Gera CSV template com colunas para 3 anotadores

O CSV template deve ser preenchido manualmente:
  annotator_1, annotator_2, annotator_3 → 1 (Sim/Verdadeiro) ou 0 (Não/Falso)

Uso:
    python scripts/90_generate_human_annotation.py
    python scripts/90_generate_human_annotation.py --n 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
BENCHMARK    = ROOT / "benchmark"
SCANS_DIR    = ROOT / "data" / "scannet" / "scans"
MANIFEST_CSV = BENCHMARK / "objects_manifest_test_official_stage1.csv"
BINARY_CSV   = BENCHMARK / "queries_test_official_stage1_relational_binary_labeled_repaired.csv"
ANN_DIR      = ROOT / "artifacts" / "human_annotation"
OUT_TEMPLATE = ROOT / "results" / "benchmark_v1" / "human_annotation_template.csv"

IMG_W, IMG_H  = 1024, 768
COLOR_X  = (210, 40,  40)   # vermelho — objeto X (between) ou A (aligned)
COLOR_A  = (40,  160, 40)   # verde    — objeto A (between) ou B (aligned)
COLOR_B  = (40,  80,  210)  # azul     — objeto B (between) ou C (aligned)
COLOR_DIM = (160, 160, 160)


# ---------------------------------------------------------------------------
# Amostragem estratificada
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n_per_stratum: int,
                       seed: int = 42) -> pd.DataFrame:
    """Amostra n exemplos por (operator × binary_label)."""
    rng = np.random.default_rng(seed)
    parts = []
    for (op, lbl), g in df.groupby(["operator", "binary_label"]):
        n = min(n_per_stratum, len(g))
        idx = rng.choice(len(g), size=n, replace=False)
        parts.append(g.iloc[idx])
    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _load_mesh(scene_id: str) -> pv.PolyData:
    scene_dir = SCANS_DIR / scene_id
    ply_files = sorted(scene_dir.glob("*_vh_clean_2.ply"))
    if not ply_files:
        raise FileNotFoundError(f"PLY não encontrado: {scene_id}")
    ply = PlyData.read(str(ply_files[0]))
    v   = ply["vertex"]
    pts = np.column_stack([np.asarray(v["x"], np.float32),
                           np.asarray(v["y"], np.float32),
                           np.asarray(v["z"], np.float32)])
    mesh = pv.PolyData(pts)
    if "face" in ply:
        raw = ply["face"].data["vertex_indices"]
        faces = []
        for f in raw:
            f = list(f)
            faces.append([len(f)] + f)
        if faces:
            mesh.faces = np.hstack(faces).astype(np.int64)
    names = set(v.data.dtype.names or [])
    if {"red","green","blue"}.issubset(names):
        mesh["rgb"] = np.column_stack([
            np.asarray(v["red"],   np.uint8),
            np.asarray(v["green"], np.uint8),
            np.asarray(v["blue"],  np.uint8)])
    return mesh


def render_annotation_image(
    scene_id: str,
    manifest: pd.DataFrame,
    highlight: dict[str, tuple],   # object_id -> (color, short_label)
    natural_query: str,
    out_path: Path,
) -> None:
    """Render top-down com objetos destacados e query sobreposta."""
    pv.OFF_SCREEN = True
    mesh = _load_mesh(scene_id)
    sdf  = manifest[(manifest["scene_id"] == scene_id) &
                    (manifest["is_valid_object"] == True)].copy()

    plotter = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H])
    plotter.set_background("white")

    if "rgb" in mesh.array_names:
        mesh["colors"] = mesh["rgb"].astype(float) / 255.0
        plotter.add_mesh(mesh, scalars="colors", rgb=True, opacity=0.60)
    else:
        plotter.add_mesh(mesh, color="lightgray", opacity=0.60)

    for _, row in sdf.iterrows():
        oid = row["object_id"]
        x0,y0,z0 = row["aabb_min_x"],row["aabb_min_y"],row["aabb_min_z"]
        x1,y1,z1 = row["aabb_max_x"],row["aabb_max_y"],row["aabb_max_z"]
        box = pv.Box(bounds=(x0,x1,y0,y1,z0,z1))
        if oid in highlight:
            color, _ = highlight[oid]
            plotter.add_mesh(box, style="wireframe",
                             color=[c/255. for c in color], line_width=4)
        else:
            plotter.add_mesh(box, style="wireframe",
                             color=[c/255. for c in COLOR_DIM], line_width=1)

    cx = sdf["centroid_x"].mean()
    cy = sdf["centroid_y"].mean()
    z_top = float(sdf["aabb_max_z"].max()) + 6.0
    ext = max(sdf["aabb_max_x"].max() - sdf["aabb_min_x"].min(),
              sdf["aabb_max_y"].max() - sdf["aabb_min_y"].min(), 1.0) * 1.25

    plotter.camera_position = [(cx, cy, z_top), (cx, cy, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = ext / 2.0

    img_arr = plotter.screenshot(return_img=True)
    plotter.close()

    img  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font_q  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font_q  = ImageFont.load_default()
        font_sm = font_q

    # Caixa de texto para a query (parte inferior)
    bg_y = IMG_H - 60
    draw.rectangle([0, bg_y, IMG_W, IMG_H], fill=(240, 240, 240))
    draw.text((10, bg_y + 8),  natural_query, fill=(0, 0, 0), font=font_q)
    draw.text((10, bg_y + 34), "Responda: Sim (1) ou Não (0)",
              fill=(80, 80, 80), font=font_sm)

    # Legenda de cores
    leg_y = 10
    for oid, (color, label) in highlight.items():
        draw.rectangle([10, leg_y, 28, leg_y + 16], fill=color)
        draw.text((33, leg_y), label, fill=(0, 0, 0), font=font_sm)
        leg_y += 22

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=88)


# ---------------------------------------------------------------------------
# Build highlight map por operador
# ---------------------------------------------------------------------------

def build_highlight(row: pd.Series) -> dict[str, tuple]:
    op = str(row["operator"])
    h  = {}
    if op == "between":
        # between(X, A, B): X está entre A e B?
        if pd.notna(row.get("object_x")):
            h[str(row["object_x"])] = (COLOR_X,
                f"X: {row.get('label_x','?')}")
        if pd.notna(row.get("object_a")):
            h[str(row["object_a"])] = (COLOR_A,
                f"A: {row.get('label_a','?')}")
        if pd.notna(row.get("object_b")):
            h[str(row["object_b"])] = (COLOR_B,
                f"B: {row.get('label_b','?')}")
    elif op == "aligned":
        # aligned(A, B, C): A, B e C estão alinhados?
        if pd.notna(row.get("object_a")):
            h[str(row["object_a"])] = (COLOR_X,
                f"A: {row.get('label_a','?')}")
        if pd.notna(row.get("object_b")):
            h[str(row["object_b"])] = (COLOR_A,
                f"B: {row.get('label_b','?')}")
        if pd.notna(row.get("object_c")):
            h[str(row["object_c"])] = (COLOR_B,
                f"C: {row.get('label_c','?')}")
    return h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="Exemplos por estrato (default=10 → 40 total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    binary_df = pd.read_csv(BINARY_CSV)
    binary_df = binary_df[binary_df["review_keep"] == "yes"].copy()
    manifest  = pd.read_csv(MANIFEST_CSV)

    sample = stratified_sample(binary_df, args.n, seed=args.seed)
    print(f"Amostra selecionada: {len(sample)} exemplos")
    print(sample.groupby(["operator","binary_label"]).size())

    ANN_DIR.mkdir(parents=True, exist_ok=True)

    template_rows = []
    n_ok, n_err = 0, 0

    for i, row in sample.iterrows():
        ann_num  = n_ok + 1
        op       = str(row["operator"])
        lbl      = int(row["binary_label"])
        lbl_str  = "pos" if lbl == 1 else "neg"
        fname    = f"ann_{ann_num:03d}_{op}_{lbl_str}.jpg"
        out_path = ANN_DIR / fname

        print(f"  [{ann_num}/{len(sample)}] {fname}", end=" ", flush=True)

        highlight  = build_highlight(row)
        natural_q  = str(row["natural_query"])
        scene_id   = str(row["scene_id"])

        if not highlight:
            print("SKIP (highlight vazio)")
            n_err += 1
            continue

        try:
            render_annotation_image(scene_id, manifest, highlight,
                                    natural_q, out_path)
            print("OK")
            n_ok += 1
        except Exception as e:
            print(f"ERRO: {e}")
            n_err += 1
            continue

        template_rows.append({
            "ann_num":         ann_num,
            "image_file":      fname,
            "binary_query_id": row["binary_query_id"],
            "scene_id":        scene_id,
            "operator":        op,
            "natural_query":   natural_q,
            "geometric_label": lbl,
            "annotator_1":     "",   # preencher: 1 ou 0
            "annotator_2":     "",
            "annotator_3":     "",
            "notes":           "",
        })

    template_df = pd.DataFrame(template_rows)
    template_df.to_csv(OUT_TEMPLATE, index=False, sep=",")

    print(f"\n{'='*60}")
    print(f"Imagens geradas: {n_ok}/{len(sample)} (erros: {n_err})")
    print(f"Diretório:       {ANN_DIR}")
    print(f"CSV template:    {OUT_TEMPLATE}")
    print(f"\nInstruções para anotadores:")
    print(f"  1. Abrir cada imagem em {ANN_DIR}/")
    print(f"  2. Ler a pergunta na parte inferior da imagem")
    print(f"  3. Preencher as colunas annotator_1/2/3 no CSV:")
    print(f"     1 = Sim (verdadeiro)   0 = Não (falso)")
    print(f"  4. Salvar o CSV preenchido como human_annotation_filled.csv")
    print(f"  5. Rodar: python scripts/91_analyze_human_validation.py")


if __name__ == "__main__":
    main()