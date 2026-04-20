#!/usr/bin/env python3
"""
89_run_directional_benchmark.py

Avalia o operador `above` no dataset gerado pelo script 88 e executa
um baseline VLM (GPT-4.1) com render isométrico da cena — o único tipo
de imagem que expõe relações verticais ao modelo.

Etapas:
  1. Avaliação geométrica (critério aabb_min_z vs aabb_max_z)
  2. Baseline VLM com render isométrico + bounding boxes coloridas
  3. Análise comparativa: geométrico vs VLM
  4. Figuras e tabela LaTeX para a dissertação

Uso:
    python scripts/89_run_directional_benchmark.py           # geom + VLM
    python scripts/89_run_directional_benchmark.py --geo-only
    OPENAI_API_KEY=sk-... python scripts/89_run_directional_benchmark.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
BENCHMARK    = ROOT / "benchmark"
SCANS_DIR    = ROOT / "data" / "scannet" / "scans"
MANIFEST_CSV = BENCHMARK / "objects_manifest_test_official_stage1.csv"
QUERIES_CSV  = BENCHMARK / "queries_test_official_stage1_above_binary.csv"
RENDERS_DIR  = ROOT / "artifacts" / "directional_renders"
RESULTS_DIR  = ROOT / "results" / "benchmark_v1"
FIGS_DIR     = ROOT / "artifacts" / "figures"

OUT_GEO_CSV  = RESULTS_DIR / "directional_above_geometric_results.csv"
OUT_VLM_CSV  = RESULTS_DIR / "directional_above_vlm_results.csv"
FIG_PDF      = FIGS_DIR / "fig_directional_above_comparison.pdf"
FIG_PNG      = FIGS_DIR / "fig_directional_above_comparison.png"

TAU_ABOVE    = 0.0    # limiar geométrico (0 = sem sobreposição)
MODEL        = "gpt-4.1"
MAX_RETRIES  = 3
RETRY_DELAY  = 6
IMG_W, IMG_H = 1024, 768

COLOR_A  = (210, 40,  40)   # vermelho — objeto A (potencialmente acima)
COLOR_B  = (40,  80, 210)   # azul     — objeto B
COLOR_DIM = (160, 160, 160)


# ---------------------------------------------------------------------------
# Operador geométrico above
# ---------------------------------------------------------------------------

def geometric_above(row: pd.Series, manifest: pd.DataFrame,
                    tau: float = TAU_ABOVE) -> int:
    """1 se aabb_min_z(A) >= aabb_max_z(B) - tau, senão 0."""
    oid_a = row["object_a"]
    oid_b = row["object_b"]
    rows_a = manifest[manifest["object_id"] == oid_a]
    rows_b = manifest[manifest["object_id"] == oid_b]
    if rows_a.empty or rows_b.empty:
        return -1
    min_z_a = float(rows_a.iloc[0]["aabb_min_z"])
    max_z_b = float(rows_b.iloc[0]["aabb_max_z"])
    return int(min_z_a >= max_z_b - tau)


# ---------------------------------------------------------------------------
# Render isométrico (expõe relações verticais)
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


def render_isometric(scene_id: str, manifest: pd.DataFrame,
                     oid_a: str, oid_b: str, out_path: Path) -> None:
    """Render isométrico (45° de elevação) para expor relações verticais."""
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

    def add_box(row, color, lw):
        x0,y0,z0 = row["aabb_min_x"],row["aabb_min_y"],row["aabb_min_z"]
        x1,y1,z1 = row["aabb_max_x"],row["aabb_max_y"],row["aabb_max_z"]
        box = pv.Box(bounds=(x0,x1,y0,y1,z0,z1))
        plotter.add_mesh(box, style="wireframe",
                         color=[c/255. for c in color], line_width=lw)

    for _, row in sdf.iterrows():
        oid = row["object_id"]
        if oid == oid_a:
            add_box(row, COLOR_A, 4)
        elif oid == oid_b:
            add_box(row, COLOR_B, 4)
        else:
            add_box(row, COLOR_DIM, 1)

    # Câmera isométrica (45° de elevação, vista diagonal)
    cx = sdf["centroid_x"].mean()
    cy = sdf["centroid_y"].mean()
    cz = sdf["centroid_z"].mean()
    ext = max(
        sdf["aabb_max_x"].max() - sdf["aabb_min_x"].min(),
        sdf["aabb_max_y"].max() - sdf["aabb_min_y"].min(),
    ) * 1.5

    plotter.camera_position = [
        (cx + ext, cy - ext, cz + ext * 0.8),
        (cx, cy, cz),
        (0, 0, 1),
    ]

    img_arr = plotter.screenshot(return_img=True)
    plotter.close()

    img  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    # Legenda
    draw.rectangle([10, 10, 30, 30], fill=COLOR_A)
    draw.text((35, 12), "A (potencialmente acima)", fill=(0,0,0), font=font)
    draw.rectangle([10, 38, 30, 58], fill=COLOR_B)
    draw.text((35, 40), "B (potencialmente abaixo)", fill=(0,0,0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=88)


# ---------------------------------------------------------------------------
# GPT-4.1
# ---------------------------------------------------------------------------

PROMPT_ABOVE = """\
Você está observando uma cena 3D de ambiente interno em perspectiva isométrica.
A caixa VERMELHA é o objeto A: {label_a} ({oid_a})
A caixa AZUL é o objeto B: {label_b} ({oid_b})

Consulta: "{natural_query}"

Considerando a posição vertical (altura) dos dois objetos na cena 3D, \
o objeto A (vermelho) está fisicamente acima do objeto B (azul)?

Responda APENAS com: Sim   ou   Não"""


def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_gpt4(api_key: str, prompt: str, img_path: Path) -> str:
    payload = {
        "model": MODEL, "max_tokens": 10,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(img_path)}",
                           "detail": "high"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def call_retry(api_key, prompt, img_path):
    for attempt in range(MAX_RETRIES):
        try:
            return call_gpt4(api_key, prompt, img_path)
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt+1))
    return "ERROR"


def parse_vlm(response: str) -> int | None:
    r = response.strip().lower()
    if r.startswith("sim") or r == "yes":
        return 1
    if r.startswith("não") or r.startswith("nao") or r == "no":
        return 0
    return None


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    tp = sum(t==1 and p==1 for t,p in zip(y_true, y_pred))
    tn = sum(t==0 and p==0 for t,p in zip(y_true, y_pred))
    fp = sum(t==0 and p==1 for t,p in zip(y_true, y_pred))
    fn = sum(t==1 and p==0 for t,p in zip(y_true, y_pred))
    acc  = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {"accuracy":prec,"precision":prec,"recall":rec,"f1":f1,
            "accuracy_correct": acc, "tp":tp,"tn":tn,"fp":fp,"fn":fn}


# ---------------------------------------------------------------------------
# Figura comparativa
# ---------------------------------------------------------------------------

def make_figure(geo_metrics: dict, vlm_metrics: dict, n: int) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    C_GEO = "#009E73"
    C_VLM = "#E69F00"

    metrics = ["Acurácia", "Precisão", "Revocação", "F1"]
    geo_v   = [geo_metrics["accuracy_correct"], geo_metrics["precision"],
               geo_metrics["recall"],            geo_metrics["f1"]]
    vlm_v   = [vlm_metrics["accuracy_correct"], vlm_metrics["precision"],
               vlm_metrics["recall"],            vlm_metrics["f1"]]

    x   = range(len(metrics))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars_g = ax.bar([i - w/2 for i in x], geo_v, width=w, color=C_GEO,
                    label="Motor geométrico", edgecolor="black", linewidth=0.7)
    bars_v = ax.bar([i + w/2 for i in x], vlm_v, width=w, color=C_VLM,
                    label="VLM (GPT-4.1, render isométrico)",
                    edgecolor="black", linewidth=0.7)

    for bar in list(bars_g) + list(bars_v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Valor", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.40)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.97, f"N = {n} exemplos binários",
            ha="right", va="top", transform=ax.transAxes, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIG_PDF, bbox_inches="tight", dpi=150)
    plt.savefig(FIG_PNG, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figura: {FIG_PDF}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo-only", action="store_true")
    parser.add_argument("--dry-run", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not args.geo_only and not api_key:
        sys.exit("OPENAI_API_KEY não definida. Use --geo-only ou defina a chave.")

    df       = pd.read_csv(QUERIES_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)

    if args.dry_run > 0:
        df = df.head(args.dry_run * 2)  # *2 para incluir pos e neg

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Avaliação geométrica -------------------------------------------
    print(f"\n=== Operador geométrico above (τ={TAU_ABOVE}) ===")
    geo_preds = []
    for _, row in df.iterrows():
        pred = geometric_above(row, manifest, TAU_ABOVE)
        geo_preds.append(pred)

    df["geo_pred"] = geo_preds
    geo_valid = df[df["geo_pred"] >= 0].copy()
    geo_m = compute_metrics(geo_valid["binary_label"].tolist(),
                            geo_valid["geo_pred"].tolist())

    print(f"  N={len(geo_valid)}  "
          f"acc={geo_m['accuracy_correct']:.3f}  "
          f"prec={geo_m['precision']:.3f}  "
          f"rec={geo_m['recall']:.3f}  "
          f"F1={geo_m['f1']:.3f}")
    df.to_csv(OUT_GEO_CSV, index=False)
    print(f"  Salvo: {OUT_GEO_CSV}")

    if args.geo_only:
        print("\nModo --geo-only. VLM baseline pulado.")
        return

    # ---- Baseline VLM -----------------------------------------------
    print(f"\n=== VLM baseline above (GPT-4.1, render isométrico) ===")
    vlm_rows = []
    done_ids: set[str] = set()
    if OUT_VLM_CSV.exists():
        done_df  = pd.read_csv(OUT_VLM_CSV)
        done_ids = set(done_df["binary_query_id"].tolist())
        vlm_rows = done_df.to_dict("records")
        print(f"Resumindo: {len(done_ids)} já processadas")

    # Cache de renders por cena
    rendered: set[str] = set()

    for _, row in df.iterrows():
        qid      = str(row["binary_query_id"])
        scene_id = str(row["scene_id"])
        oid_a    = str(row["object_a"])
        oid_b    = str(row["object_b"])
        lab_a    = str(row["label_a"])
        lab_b    = str(row["label_b"])
        nq       = str(row["natural_query"])
        gt       = int(row["binary_label"])

        if qid in done_ids:
            continue

        print(f"  {qid}", end=" ", flush=True)

        # Render (um por scene+par, reusado para pos e neg)
        pair_key = f"{scene_id}__{oid_a}__{oid_b}"
        render_path = RENDERS_DIR / f"{pair_key}.jpg"
        if pair_key not in rendered and not render_path.exists():
            try:
                render_isometric(scene_id, manifest, oid_a, oid_b, render_path)
                rendered.add(pair_key)
            except Exception as e:
                print(f"RENDER_ERR: {e}")
                vlm_rows.append({"binary_query_id":qid,"scene_id":scene_id,
                                  "binary_label":gt,"vlm_response":"RENDER_ERROR",
                                  "vlm_pred":None,"error":str(e)})
                _append(OUT_VLM_CSV, vlm_rows[-1])
                continue

        prompt = PROMPT_ABOVE.format(
            label_a=lab_a, oid_a=oid_a,
            label_b=lab_b, oid_b=oid_b,
            natural_query=nq)

        try:
            response = call_retry(api_key, prompt, render_path)
            vlm_pred = parse_vlm(response)
            print(f"GT={gt} VLM={response!r} pred={vlm_pred}")
        except Exception as e:
            print(f"API_ERR: {e}")
            response = "API_ERROR"
            vlm_pred = None

        r = {"binary_query_id":qid,"scene_id":scene_id,
             "binary_label":gt,"vlm_response":response,
             "vlm_pred":vlm_pred,"error":None}
        vlm_rows.append(r)
        _append(OUT_VLM_CSV, r)
        time.sleep(0.6)

    # ---- Análise comparativa -----------------------------------------
    vlm_df = pd.read_csv(OUT_VLM_CSV)
    vlm_df = vlm_df[vlm_df["vlm_pred"].notna()].copy()
    vlm_df["vlm_pred"] = vlm_df["vlm_pred"].astype(int)

    vlm_m = compute_metrics(vlm_df["binary_label"].tolist(),
                            vlm_df["vlm_pred"].tolist())

    print(f"\n--- Resumo comparativo ---")
    print(f"  Geométrico: acc={geo_m['accuracy_correct']:.3f}  F1={geo_m['f1']:.3f}")
    print(f"  VLM:        acc={vlm_m['accuracy_correct']:.3f}  F1={vlm_m['f1']:.3f}")
    print(f"  Parse rate VLM: {len(vlm_df)}/{len(df)}")

    print_latex(geo_m, vlm_m, len(geo_valid), len(vlm_df))
    make_figure(geo_m, vlm_m, len(geo_valid))
    print("Script 89 concluído.")


def _append(path: Path, data: dict) -> None:
    row = pd.DataFrame([data])
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, mode="w", header=True, index=False)


def print_latex(geo: dict, vlm: dict, n_geo: int, n_vlm: int) -> None:
    def f(v): return f"{v:.3f}"
    print("\n" + "="*65)
    print("TABELA LATEX — operador above")
    print("="*65)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Resultados do operador direcional \textit{above} no "
          r"conjunto de teste. O motor geométrico usa o critério "
          r"$\text{aabb\_min\_z}(A) \geq \text{aabb\_max\_z}(B)$; o "
          r"baseline VLM recebe um render isométrico da cena.}")
    print(r"\label{tab:directional-above}")
    print(r"\begin{tabular}{lccccl}")
    print(r"\toprule")
    print(r"\textbf{Método} & \textbf{N} & \textbf{Acurácia} & "
          r"\textbf{Precisão} & \textbf{Revocação} & \textbf{F1} \\")
    print(r"\midrule")
    print(f"Motor geométrico & {n_geo} & {f(geo['accuracy_correct'])} & "
          f"{f(geo['precision'])} & {f(geo['recall'])} & {f(geo['f1'])} \\\\")
    print(f"VLM (GPT-4.1)    & {n_vlm} & {f(vlm['accuracy_correct'])} & "
          f"{f(vlm['precision'])} & {f(vlm['recall'])} & {f(vlm['f1'])} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*65)


if __name__ == "__main__":
    main()