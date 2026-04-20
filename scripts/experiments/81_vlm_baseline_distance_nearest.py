#!/usr/bin/env python3
"""
81_vlm_baseline_distance_nearest.py

Baseline VLM (GPT-4.1 vision) para operadores distance e nearest
no test_official_stage1.

Para cada query, gera uma imagem top-down da cena com os objetos
relevantes destacados por bounding boxes coloridas. A imagem é enviada
ao GPT-4.1 com um prompt estruturado. Resultados salvos incrementalmente.

Uso:
    OPENAI_API_KEY=sk-... python scripts/81_vlm_baseline_distance_nearest.py
    OPENAI_API_KEY=sk-... python scripts/81_vlm_baseline_distance_nearest.py --dry-run 5
    OPENAI_API_KEY=sk-... python scripts/81_vlm_baseline_distance_nearest.py --skip-render
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR  = ROOT / "benchmark"
SCANS_DIR      = ROOT / "data" / "scannet" / "scans"
ARTIFACTS_DIR  = ROOT / "artifacts" / "vlm_baseline_renders"
RESULTS_DIR    = ROOT / "results" / "benchmark_v1"

GT_CSV       = BENCHMARK_DIR / "ground_truth_distance_nearest_test_official_stage1.csv"
QUERIES_CSV  = BENCHMARK_DIR / "queries_test_official_stage1_distance_nearest_final.csv"
MANIFEST_CSV = BENCHMARK_DIR / "objects_manifest_test_official_stage1.csv"
OUTPUT_CSV   = RESULTS_DIR / "vlm_baseline_distance_nearest_raw.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 1024, 1024
COLOR_A  = (210, 40,  40)   # vermelho — objeto A / referência
COLOR_B  = (40,  80,  210)  # azul     — objeto B / candidatos
COLOR_DIM = (160, 160, 160) # cinza    — outros objetos

MODEL       = "gpt-4.1"
MAX_TOKENS  = 60
MAX_RETRIES = 3
RETRY_DELAY = 6   # segundos (base; multiplica por tentativa)

# ---------------------------------------------------------------------------
# Helpers de geometria
# ---------------------------------------------------------------------------

def load_ply_mesh(scene_id: str) -> pv.PolyData:
    scene_dir = SCANS_DIR / scene_id
    ply_files = sorted(scene_dir.glob("*_vh_clean_2.ply"))
    if not ply_files:
        raise FileNotFoundError(f"PLY não encontrado em {scene_dir}")

    ply = PlyData.read(str(ply_files[0]))
    v = ply["vertex"]
    pts = np.column_stack([
        np.asarray(v["x"], dtype=np.float32),
        np.asarray(v["y"], dtype=np.float32),
        np.asarray(v["z"], dtype=np.float32),
    ])
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
    if {"red", "green", "blue"}.issubset(names):
        mesh["rgb"] = np.column_stack([
            np.asarray(v["red"],   dtype=np.uint8),
            np.asarray(v["green"], dtype=np.uint8),
            np.asarray(v["blue"],  dtype=np.uint8),
        ])
    return mesh


def add_bbox_wireframe(
    plotter: pv.Plotter,
    row: pd.Series,
    color: tuple[int, int, int],
    line_width: int = 2,
) -> None:
    x0, y0, z0 = row["aabb_min_x"], row["aabb_min_y"], row["aabb_min_z"]
    x1, y1, z1 = row["aabb_max_x"], row["aabb_max_y"], row["aabb_max_z"]
    box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1))
    plotter.add_mesh(
        box, style="wireframe",
        color=[c / 255.0 for c in color],
        line_width=line_width,
        render_lines_as_tubes=False,
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_query_image(
    scene_id: str,
    manifest_df: pd.DataFrame,
    highlight: dict[str, tuple[tuple[int, int, int], str]],
    out_path: Path,
) -> None:
    """
    Renderiza vista top-down da cena com objetos destacados.

    highlight: {object_id: (color_rgb, short_label)}
    """
    pv.OFF_SCREEN = True

    mesh = load_ply_mesh(scene_id)
    scene_df = manifest_df[
        (manifest_df["scene_id"] == scene_id) &
        (manifest_df["is_valid_object"] == True)
    ].copy()

    if scene_df.empty:
        raise ValueError(f"Nenhum objeto válido para {scene_id} no manifesto")

    plotter = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H])
    plotter.set_background("white")

    # Mesh da cena com cores do sensor
    if "rgb" in mesh.array_names:
        mesh["colors"] = mesh["rgb"].astype(float) / 255.0
        plotter.add_mesh(mesh, scalars="colors", rgb=True, opacity=0.65)
    else:
        plotter.add_mesh(mesh, color="lightgray", opacity=0.65)

    # Bounding boxes de todos os objetos (cinza, fino)
    for _, row in scene_df.iterrows():
        if row["object_id"] not in highlight:
            add_bbox_wireframe(plotter, row, COLOR_DIM, line_width=1)

    # Bounding boxes dos objetos da query (coloridos, grossos)
    for oid, (color, _label) in highlight.items():
        rows = scene_df[scene_df["object_id"] == oid]
        if rows.empty:
            continue
        add_bbox_wireframe(plotter, rows.iloc[0], color, line_width=4)

    # Câmera top-down ortográfica centrada na cena
    cx = scene_df["centroid_x"].mean()
    cy = scene_df["centroid_y"].mean()
    z_top = float(scene_df["aabb_max_z"].max()) + 6.0

    x_extent = float(scene_df["aabb_max_x"].max() - scene_df["aabb_min_x"].min())
    y_extent = float(scene_df["aabb_max_y"].max() - scene_df["aabb_min_y"].min())
    scene_extent = max(x_extent, y_extent, 1.0) * 1.25  # margem 25%

    plotter.camera_position = [
        (cx, cy, z_top),
        (cx, cy, 0.0),
        (0.0, 1.0, 0.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = scene_extent / 2.0

    img_arr = plotter.screenshot(return_img=True)
    plotter.close()

    # Post-processamento: barra de escala + legenda de cores
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # Barra de escala de 1 metro
    px_per_m = IMG_W / scene_extent
    bar_len_px = max(int(px_per_m), 10)
    bx0, by = 30, IMG_H - 45
    draw.rectangle([bx0, by, bx0 + bar_len_px, by + 8], fill=(0, 0, 0))
    # Ticks nas extremidades
    draw.rectangle([bx0, by - 4, bx0 + 2, by + 12], fill=(0, 0, 0))
    draw.rectangle([bx0 + bar_len_px - 2, by - 4, bx0 + bar_len_px, by + 12], fill=(0, 0, 0))
    draw.text((bx0 + bar_len_px // 2 - 12, by + 12), "1 m", fill=(0, 0, 0), font=font_sm)

    # Legenda: objeto A e objeto B
    leg_x, leg_y = 20, 15
    for label, color in highlight.items():
        short = color[1]
        rgb = color[0]
        draw.rectangle([leg_x, leg_y, leg_x + 18, leg_y + 18], fill=rgb)
        draw.text((leg_x + 24, leg_y), short, fill=(0, 0, 0), font=font_sm)
        leg_y += 26

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=90)


# ---------------------------------------------------------------------------
# GPT-4.1 vision
# ---------------------------------------------------------------------------

def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gpt4_vision(api_key: str, prompt: str, img_path: Path) -> str:
    b64 = img_to_b64(img_path)
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def call_with_retry(api_key: str, prompt: str, img_path: Path) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return call_gpt4_vision(api_key, prompt, img_path)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {e.code}: {body[:200]}"
            print(f"  [tentativa {attempt+1}/{MAX_RETRIES}] {last_err}")
        except Exception as e:
            last_err = str(e)
            print(f"  [tentativa {attempt+1}/{MAX_RETRIES}] {last_err}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"API falhou após {MAX_RETRIES} tentativas: {last_err}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_DISTANCE = """\
You are looking at a top-down (overhead) render of a 3D indoor scene.
The RED bounding box is Object A: {label_a}
The BLUE bounding box is Object B: {label_b}
The scale bar at the bottom-left shows exactly 1 meter.

Estimate the physical distance in meters between the nearest surfaces of Object A and Object B.

Reply with ONLY a single decimal number in meters. Example: 1.35
No units, no words, no explanation."""

PROMPT_NEAREST = """\
You are looking at a top-down (overhead) render of a 3D indoor scene.
The RED bounding box is the reference object: {label_ref}
The BLUE bounding boxes are candidate objects of category "{target_category}":
{candidates_list}

Which candidate object is physically closest to the reference object?

Reply with ONLY the exact object ID from the list above. Example: {example_id}
No other words."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_distance_response(response: str) -> float | None:
    """Extrai o primeiro número válido da resposta do VLM."""
    # Normaliza vírgula decimal
    cleaned = response.replace(",", ".").strip()
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    for m in matches:
        val = float(m)
        if 0.0 < val <= 25.0:  # intervalo plausível para indoor
            return val
    return None


def parse_nearest_response(response: str, candidate_ids: list[str]) -> str | None:
    """Faz correspondência da resposta com os IDs candidatos conhecidos."""
    resp = response.strip()
    # Correspondência exata
    if resp in candidate_ids:
        return resp
    # Correspondência parcial (o VLM pode ter cortado o prefixo)
    for cid in candidate_ids:
        if cid in resp or resp in cid:
            return cid
    # Último recurso: verificar se algum sufixo (ex: "chair_003") aparece na resposta
    for cid in candidate_ids:
        suffix = cid.split("__")[-1] if "__" in cid else cid
        if suffix in resp:
            return cid
    return None


# ---------------------------------------------------------------------------
# Seleção de candidatos para nearest
# ---------------------------------------------------------------------------

def get_nearest_candidates(
    scene_id: str,
    reference_id: str,
    target_category: str,
    manifest_df: pd.DataFrame,
) -> list[str]:
    """Retorna todos os objetos válidos da categoria-alvo na cena, excluindo referência."""
    mask = (
        (manifest_df["scene_id"] == scene_id) &
        (manifest_df["is_valid_object"] == True) &
        (manifest_df["label_norm"] == target_category) &
        (manifest_df["object_id"] != reference_id)
    )
    return manifest_df.loc[mask, "object_id"].tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline VLM (GPT-4.1) para distance e nearest"
    )
    parser.add_argument(
        "--dry-run", type=int, default=0, metavar="N",
        help="Processar apenas as primeiras N queries (0 = todas)"
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="Reusar imagens já renderizadas (não re-renderiza)"
    )
    parser.add_argument(
        "--operator", choices=["distance", "nearest", "all"], default="all",
        help="Filtrar por operador (padrão: all)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("ERRO: variável OPENAI_API_KEY não definida.\n"
                 "Use: OPENAI_API_KEY=sk-... python scripts/81_vlm_baseline_distance_nearest.py")

    # --- Carrega dados ---
    print("Carregando dados...")
    gt_df      = pd.read_csv(GT_CSV)
    queries_df = pd.read_csv(QUERIES_CSV)
    manifest_df = pd.read_csv(MANIFEST_CSV)

    # Filtra apenas queries revisadas e aprovadas
    queries_df = queries_df[queries_df["review_keep"] == "yes"].copy()

    # Merge: GT (com query_id) + queries (com campos detalhados)
    extra_cols = [
        "scene_id", "operator", "structured_query",
        "label_a", "label_b",
        "reference_object", "reference_label", "target_category",
        "answer_object",
    ]
    # Garante que colunas existam (nearest pode ter NaN em label_a/b)
    for col in extra_cols:
        if col not in queries_df.columns:
            queries_df[col] = None

    merged = gt_df.merge(
        queries_df[extra_cols].drop_duplicates("structured_query"),
        on=["scene_id", "operator", "structured_query"],
        how="left",
    )

    # Filtra operador se solicitado
    if args.operator != "all":
        merged = merged[merged["operator"] == args.operator].copy()

    if args.dry_run > 0:
        merged = merged.head(args.dry_run).copy()

    print(f"Total de queries: {len(merged)} "
          f"(distance={len(merged[merged['operator']=='distance'])}, "
          f"nearest={len(merged[merged['operator']=='nearest'])})")

    # --- Resumo (pula queries já processadas) ---
    done_ids: set[str] = set()
    if OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["query_id"].tolist())
        print(f"Resumindo: {len(done_ids)} queries já processadas, pulando.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_err = 0

    for i, row in merged.iterrows():
        qid      = str(row["query_id"])
        scene_id = str(row["scene_id"])
        operator = str(row["operator"])

        if qid in done_ids:
            continue

        print(f"\n[{i+1}/{len(merged)}] {qid} | {operator} | {scene_id}")

        # ---- Monta highlight e prompt --------------------------------
        vlm_parsed  = None
        candidate_ids: list[str] = []

        if operator == "distance":
            obj_a   = str(row["gt_object_a"])
            obj_b   = str(row["gt_object_b"])
            label_a = str(row.get("label_a") or obj_a.split("__")[-1].rsplit("_", 1)[0])
            label_b = str(row.get("label_b") or obj_b.split("__")[-1].rsplit("_", 1)[0])
            gt_val  = float(row["gt_distance_m"])

            highlight = {
                obj_a: (COLOR_A, f"A: {label_a}"),
                obj_b: (COLOR_B, f"B: {label_b}"),
            }
            prompt = PROMPT_DISTANCE.format(label_a=label_a, label_b=label_b)

        elif operator == "nearest":
            ref_obj     = str(row["gt_object_a"])
            ref_label   = str(row.get("reference_label") or
                               ref_obj.split("__")[-1].rsplit("_", 1)[0])
            target_cat  = str(row.get("target_category") or "object")
            gt_val      = str(row.get("gt_answer_object", ""))

            candidate_ids = get_nearest_candidates(
                scene_id, ref_obj, target_cat, manifest_df
            )
            if not candidate_ids:
                print(f"  AVISO: nenhum candidato de '{target_cat}' em {scene_id}, pulando.")
                n_err += 1
                continue

            highlight = {ref_obj: (COLOR_A, f"REF: {ref_label}")}
            for cid in candidate_ids:
                short = cid.split("__")[-1] if "__" in cid else cid
                highlight[cid] = (COLOR_B, short)

            candidates_list = "\n".join(f"  - {cid}" for cid in candidate_ids)
            prompt = PROMPT_NEAREST.format(
                label_ref=ref_label,
                target_category=target_cat,
                candidates_list=candidates_list,
                example_id=candidate_ids[0],
            )
        else:
            print(f"  Operador desconhecido: {operator}, pulando.")
            continue

        # ---- Render --------------------------------------------------
        img_path = ARTIFACTS_DIR / f"{qid}.jpg"
        if args.skip_render and img_path.exists():
            print(f"  Render existente: {img_path.name}")
        else:
            try:
                render_query_image(scene_id, manifest_df, highlight, img_path)
                print(f"  Renderizado → {img_path.name}")
            except Exception as e:
                print(f"  ERRO no render: {e}")
                _save_row(OUTPUT_CSV, {
                    "query_id": qid, "scene_id": scene_id, "operator": operator,
                    "gt_value": gt_val, "vlm_response": "RENDER_ERROR",
                    "vlm_parsed": None, "is_correct": None, "error": str(e),
                })
                n_err += 1
                continue

        # ---- Chamada VLM --------------------------------------------
        try:
            response = call_with_retry(api_key, prompt, img_path)
            print(f"  VLM → {response!r}")
        except Exception as e:
            print(f"  ERRO na API: {e}")
            _save_row(OUTPUT_CSV, {
                "query_id": qid, "scene_id": scene_id, "operator": operator,
                "gt_value": gt_val, "vlm_response": "API_ERROR",
                "vlm_parsed": None, "is_correct": None, "error": str(e),
            })
            n_err += 1
            time.sleep(RETRY_DELAY)
            continue

        # ---- Parse + avaliação parcial ------------------------------
        if operator == "distance":
            parsed = parse_distance_response(response)
            is_correct = None  # avaliado por MAE no script 82
        else:
            parsed = parse_nearest_response(response, candidate_ids)
            is_correct = (parsed == gt_val) if parsed else False

        _save_row(OUTPUT_CSV, {
            "query_id":   qid,
            "scene_id":   scene_id,
            "operator":   operator,
            "gt_value":   gt_val,
            "vlm_response": response,
            "vlm_parsed": parsed,
            "is_correct": is_correct,
            "error":      None,
        })
        n_ok += 1
        time.sleep(0.8)  # delay conservador entre chamadas

    print(f"\n{'='*60}")
    print(f"Concluído: {n_ok} OK | {n_err} erros")
    print(f"Resultados: {OUTPUT_CSV}")
    print(f"Renders:    {ARTIFACTS_DIR}")


def _save_row(path: Path, data: dict) -> None:
    """Salva uma linha no CSV de resultados (cria ou appenda)."""
    df = pd.DataFrame([data])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


if __name__ == "__main__":
    main()