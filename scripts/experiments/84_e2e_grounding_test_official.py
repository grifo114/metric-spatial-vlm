#!/usr/bin/env python3
"""
83_e2e_grounding_test_official.py

Experimento de grounding ponta a ponta (E2E) no test_official_stage1.

Para cada query métrica (distance + nearest):
  1. Renderiza a cena com todos os objetos válidos numerados e coloridos
     por categoria (render neutro — sem destacar os objetos da query).
  2. Apresenta ao GPT-4.1 a imagem + lista numerada + query em linguagem
     natural SEM os object IDs (apenas labels de categoria).
  3. GPT-4.1 identifica quais objetos específicos estão referenciados.
  4. O motor geométrico executa o operador sobre os objetos identificados.
  5. Compara resultado ao GT para decompor E_total = E_grounding + E_geometric.

Uso:
    OPENAI_API_KEY=sk-... python scripts/83_e2e_grounding_test_official.py
    OPENAI_API_KEY=sk-... python scripts/83_e2e_grounding_test_official.py --dry-run 5
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
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
BENCHMARK    = ROOT / "benchmark"
SCANS_DIR    = ROOT / "data" / "scannet" / "scans"
POINTS_DIR   = ROOT / "artifacts" / "object_points_test_official_stage1"
RENDERS_DIR  = ROOT / "artifacts" / "e2e_grounding_renders"
RESULTS_DIR  = ROOT / "results" / "benchmark_v1"

GT_CSV       = BENCHMARK / "ground_truth_distance_nearest_test_official_stage1.csv"
QUERIES_CSV  = BENCHMARK / "queries_test_official_stage1_distance_nearest_final.csv"
MANIFEST_CSV = BENCHMARK / "objects_manifest_test_official_stage1.csv"
OUTPUT_CSV   = RESULTS_DIR / "e2e_grounding_test_official_raw.csv"

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 1024, 1024

# Cores por categoria (RGB)
CATEGORY_PALETTE = [
    (70,  130, 180),  # steel blue
    (34,  139,  34),  # green
    (210, 105,  30),  # chocolate
    (148,   0, 211),  # purple
    (220,  20,  60),  # crimson
    (255, 165,   0),  # orange
    (0,   139, 139),  # dark cyan
    (184, 134,  11),  # goldenrod
    (100, 149, 237),  # cornflower
    (85,  107,  47),  # olive
]

MODEL      = "gpt-4.1"
MAX_TOKENS = 100
MAX_RETRIES = 3
RETRY_DELAY = 6


# ---------------------------------------------------------------------------
# Utilidades geométricas
# ---------------------------------------------------------------------------

def load_points(obj_id: str, scene_id: str) -> np.ndarray | None:
    path = POINTS_DIR / scene_id / f"{obj_id}.npz"
    if not path.exists():
        return None
    return np.load(path)["points"]


def surface_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Distância mínima superfície-a-superfície via cKDTree."""
    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_a, k=1, workers=-1)
    return float(dists.min())


def centroid(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0)


def centroid_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    return float(np.linalg.norm(centroid(pts_a) - centroid(pts_b)))


def find_nearest_surface(
    ref_pts: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
) -> tuple[str, float]:
    """Retorna (object_id, dist) do candidato mais próximo por superfície."""
    best_id, best_dist = None, float("inf")
    tree_ref = cKDTree(ref_pts)
    for cid, cpts in candidates:
        dists, _ = tree_ref.query(cpts, k=1, workers=-1)
        d = float(dists.min())
        if d < best_dist:
            best_dist = d
            best_id = cid
    return best_id, best_dist


# ---------------------------------------------------------------------------
# Render neutro numerado (por cena, cacheado)
# ---------------------------------------------------------------------------

def _load_mesh(scene_id: str) -> pv.PolyData:
    scene_dir = SCANS_DIR / scene_id
    ply_files = sorted(scene_dir.glob("*_vh_clean_2.ply"))
    if not ply_files:
        raise FileNotFoundError(f"PLY não encontrado para {scene_id}")
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


def render_scene_numbered(
    scene_id: str,
    scene_df: pd.DataFrame,
    number_map: dict[int, str],  # number -> object_id
    category_colors: dict[str, tuple],
    out_path: Path,
) -> None:
    """
    Render top-down da cena com todos os objetos numerados.
    Cores por categoria para facilitar distinguir categorias visualmente.
    """
    pv.OFF_SCREEN = True
    mesh = _load_mesh(scene_id)

    plotter = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H])
    plotter.set_background("white")

    if "rgb" in mesh.array_names:
        mesh["colors"] = mesh["rgb"].astype(float) / 255.0
        plotter.add_mesh(mesh, scalars="colors", rgb=True, opacity=0.60)
    else:
        plotter.add_mesh(mesh, color="lightgray", opacity=0.60)

    # Bounding boxes de todos os objetos, coloridos por categoria
    num_to_row = {n: scene_df[scene_df["object_id"] == oid].iloc[0]
                  for n, oid in number_map.items()
                  if not scene_df[scene_df["object_id"] == oid].empty}

    for num, row in num_to_row.items():
        cat   = str(row["label_norm"])
        color = category_colors.get(cat, (120, 120, 120))
        cf    = [c / 255.0 for c in color]
        x0, y0, z0 = row["aabb_min_x"], row["aabb_min_y"], row["aabb_min_z"]
        x1, y1, z1 = row["aabb_max_x"], row["aabb_max_y"], row["aabb_max_z"]
        box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1))
        plotter.add_mesh(box, style="wireframe", color=cf, line_width=2)

    # Câmera ortográfica top-down
    cx = scene_df["centroid_x"].mean()
    cy = scene_df["centroid_y"].mean()
    z_top = float(scene_df["aabb_max_z"].max()) + 6.0
    x_ext = float(scene_df["aabb_max_x"].max() - scene_df["aabb_min_x"].min())
    y_ext = float(scene_df["aabb_max_y"].max() - scene_df["aabb_min_y"].min())
    extent = max(x_ext, y_ext, 1.0) * 1.25

    plotter.camera_position = [(cx, cy, z_top), (cx, cy, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = extent / 2.0

    img_arr = plotter.screenshot(return_img=True)
    plotter.close()

    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()

    px_per_m = IMG_W / extent

    # Projeção aproximada centroide → pixel (vista ortográfica top-down)
    def world_to_px(wx: float, wy: float) -> tuple[int, int]:
        px = int((wx - (cx - extent / 2)) * px_per_m)
        py = int(IMG_H - (wy - (cy - extent / 2)) * px_per_m)
        return px, py

    for num, row in num_to_row.items():
        cat   = str(row["label_norm"])
        color = category_colors.get(cat, (120, 120, 120))
        px, py = world_to_px(float(row["centroid_x"]), float(row["centroid_y"]))
        # Pequeno círculo colorido + número
        r = 8
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
        draw.text((px + r + 2, py - 7), str(num), fill=(0, 0, 0), font=font)

    # Escala de 1 m
    bar_px = max(int(px_per_m), 10)
    bx, by = 20, IMG_H - 40
    draw.rectangle([bx, by, bx + bar_px, by + 6], fill=(0, 0, 0))
    draw.text((bx + bar_px // 2 - 8, by + 9), "1 m", fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=88)


# ---------------------------------------------------------------------------
# Mapa de números para objetos da cena
# ---------------------------------------------------------------------------

def build_number_map(scene_df: pd.DataFrame) -> tuple[
    dict[int, str],          # number -> object_id
    dict[str, int],          # object_id -> number
    dict[str, tuple],        # category -> color
]:
    rows = scene_df.sort_values(["label_norm", "object_id"]).reset_index(drop=True)
    categories = rows["label_norm"].unique().tolist()
    cat_color   = {c: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
                   for i, c in enumerate(sorted(categories))}

    num_to_id = {}
    id_to_num = {}
    for i, row in rows.iterrows():
        n = i + 1
        oid = row["object_id"]
        num_to_id[n] = oid
        id_to_num[oid] = n

    return num_to_id, id_to_num, cat_color


# ---------------------------------------------------------------------------
# Prompts de grounding
# ---------------------------------------------------------------------------

PROMPT_DISTANCE = """\
Você está observando uma cena 3D de ambiente interno em vista superior.
Cada objeto está marcado com um número (círculo colorido).

Objetos presentes na cena:
{object_list}

Consulta de distância: "Qual a distância entre {label_a} e {label_b}?"

Identifique qual objeto específico da lista é o {label_a} \
e qual é o {label_b} referenciados.
Se houver múltiplas instâncias da mesma categoria, escolha a mais \
proeminente ou a que fizer mais sentido visualmente no contexto da cena.

Responda APENAS com dois IDs exatamente como aparecem na lista, \
separados por vírgula. Exemplo:
scene0008_00__monitor_029, scene0008_00__table_032"""

PROMPT_NEAREST = """\
Você está observando uma cena 3D de ambiente interno em vista superior.
Cada objeto está marcado com um número (círculo colorido).

Objetos presentes na cena:
{object_list}

Consulta de proximidade: "Qual {target_category} está mais próximo \
de {reference_label}?"

Identifique qual objeto específico da lista é o {reference_label} \
referenciado como ponto de referência.
Se houver múltiplas instâncias da categoria, escolha a mais proeminente.

Responda APENAS com um ID exatamente como aparece na lista. Exemplo:
scene0008_00__monitor_029"""


def _format_object_list(num_to_id: dict, scene_df: pd.DataFrame) -> str:
    lines = []
    id_to_cat = dict(zip(scene_df["object_id"], scene_df["label_norm"]))
    for num in sorted(num_to_id):
        oid = num_to_id[num]
        cat = id_to_cat.get(oid, "?")
        lines.append(f"  {num:3d}: {cat:15s} → {oid}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPT-4.1 call
# ---------------------------------------------------------------------------

def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gpt4(api_key: str, prompt: str, img_path: Path) -> str:
    b64 = img_to_b64(img_path)
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                               "detail": "high"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"].strip()


def call_with_retry(api_key: str, prompt: str, img_path: Path) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            return call_gpt4(api_key, prompt, img_path)
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}"
        except Exception as e:
            err = str(e)
        print(f"  [tentativa {attempt+1}/{MAX_RETRIES}] {err}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"API falhou após {MAX_RETRIES} tentativas")


# ---------------------------------------------------------------------------
# Parse de resposta
# ---------------------------------------------------------------------------

def extract_ids(response: str, valid_ids: set[str], n: int) -> list[str | None]:
    """Extrai até n IDs válidos da resposta do VLM."""
    found = []
    for token in re.split(r"[\s,;]+", response):
        token = token.strip().strip(".,;\"'")
        if token in valid_ids and token not in found:
            found.append(token)
        if len(found) == n:
            break
    # Pad com None se necessário
    while len(found) < n:
        found.append(None)
    return found[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--operator", choices=["distance","nearest","all"],
                        default="all")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENAI_API_KEY não definida.")

    # --- Carrega dados ---
    gt_df      = pd.read_csv(GT_CSV)
    queries_df = pd.read_csv(QUERIES_CSV)
    manifest_df = pd.read_csv(MANIFEST_CSV)

    queries_df = queries_df[queries_df["review_keep"] == "yes"].copy()

    extra = ["scene_id","operator","structured_query",
             "label_a","label_b",
             "reference_object","reference_label","target_category",
             "answer_object"]
    for c in extra:
        if c not in queries_df.columns:
            queries_df[c] = None

    merged = gt_df.merge(
        queries_df[extra].drop_duplicates("structured_query"),
        on=["scene_id","operator","structured_query"], how="left"
    )

    if args.operator != "all":
        merged = merged[merged["operator"] == args.operator].copy()
    if args.dry_run > 0:
        merged = merged.head(args.dry_run).copy()

    print(f"Queries: {len(merged)}  "
          f"(distance={len(merged[merged['operator']=='distance'])}, "
          f"nearest={len(merged[merged['operator']=='nearest'])})")

    # --- Resume ---
    done_ids: set[str] = set()
    if OUTPUT_CSV.exists():
        done_ids = set(pd.read_csv(OUTPUT_CSV)["query_id"].tolist())
        print(f"Resumindo: {len(done_ids)} já processadas")

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Cache de render e número-mapa por cena
    scene_cache: dict[str, tuple] = {}

    def get_scene_data(scene_id: str):
        if scene_id not in scene_cache:
            sdf = manifest_df[
                (manifest_df["scene_id"] == scene_id) &
                (manifest_df["is_valid_object"] == True)
            ].copy()
            num_to_id, id_to_num, cat_color = build_number_map(sdf)
            render_path = RENDERS_DIR / f"{scene_id}_numbered.jpg"
            if not args.skip_render or not render_path.exists():
                render_scene_numbered(scene_id, sdf, num_to_id, cat_color,
                                      render_path)
            scene_cache[scene_id] = (sdf, num_to_id, id_to_num, cat_color,
                                      render_path)
        return scene_cache[scene_id]

    n_ok, n_err = 0, 0

    for _, row in merged.iterrows():
        qid      = str(row["query_id"])
        scene_id = str(row["scene_id"])
        operator = str(row["operator"])

        if qid in done_ids:
            continue

        print(f"\n{qid} | {operator} | {scene_id}")

        try:
            sdf, num_to_id, id_to_num, cat_color, render_path = \
                get_scene_data(scene_id)
        except Exception as e:
            print(f"  ERRO render: {e}")
            _save(OUTPUT_CSV, _err_row(qid, scene_id, operator, str(e)))
            n_err += 1
            continue

        valid_ids = set(sdf["object_id"].tolist())
        obj_list  = _format_object_list(num_to_id, sdf)

        # ---- Monta prompt ------------------------------------------------
        if operator == "distance":
            label_a = str(row.get("label_a") or "")
            label_b = str(row.get("label_b") or "")
            gt_obj_a = str(row["gt_object_a"])
            gt_obj_b = str(row["gt_object_b"])
            gt_val   = float(row["gt_distance_m"])

            prompt = PROMPT_DISTANCE.format(
                object_list=obj_list, label_a=label_a, label_b=label_b
            )

        elif operator == "nearest":
            ref_label  = str(row.get("reference_label") or "")
            target_cat = str(row.get("target_category") or "")
            gt_ref     = str(row["gt_object_a"])
            gt_answer  = str(row.get("gt_answer_object", ""))
            gt_val     = gt_answer

            prompt = PROMPT_NEAREST.format(
                object_list=obj_list,
                target_category=target_cat,
                reference_label=ref_label,
            )
        else:
            continue

        # ---- Chamada VLM -------------------------------------------------
        try:
            response = call_with_retry(api_key, prompt, render_path)
            print(f"  GPT-4.1 → {response!r}")
        except Exception as e:
            print(f"  ERRO API: {e}")
            _save(OUTPUT_CSV, _err_row(qid, scene_id, operator, f"API_ERROR: {e}"))
            n_err += 1
            time.sleep(RETRY_DELAY)
            continue

        # ---- Parse + verificação grounding ------------------------------
        if operator == "distance":
            ids = extract_ids(response, valid_ids, 2)
            grounded_a, grounded_b = ids[0], ids[1]
            grounding_correct = (grounded_a == gt_obj_a and grounded_b == gt_obj_b) or \
                                (grounded_a == gt_obj_b and grounded_b == gt_obj_a)

            # Calcula distância com objetos identificados pelo VLM
            e_total_surface = None
            e_total_centroid = None
            if grounded_a and grounded_b:
                pts_a = load_points(grounded_a, scene_id)
                pts_b = load_points(grounded_b, scene_id)
                if pts_a is not None and pts_b is not None:
                    e_total_surface  = surface_distance(pts_a, pts_b)
                    e_total_centroid = centroid_distance(pts_a, pts_b)

            result = {
                "query_id": qid, "scene_id": scene_id, "operator": operator,
                "gt_value": gt_val,
                "gt_object_a": gt_obj_a, "gt_object_b": gt_obj_b,
                "grounded_a": grounded_a, "grounded_b": grounded_b,
                "grounding_correct": grounding_correct,
                "e_total_surface": e_total_surface,
                "e_total_centroid": e_total_centroid,
                "vlm_response": response, "error": None,
            }

        else:  # nearest
            ids = extract_ids(response, valid_ids, 1)
            grounded_ref = ids[0]
            grounding_correct = (grounded_ref == gt_ref)

            # Acha nearest com o objeto identificado pelo VLM
            nearest_vlm = None
            nearest_dist = None
            if grounded_ref:
                ref_pts = load_points(grounded_ref, scene_id)
                if ref_pts is not None:
                    candidates_df = sdf[
                        (sdf["label_norm"] == target_cat) &
                        (sdf["object_id"] != grounded_ref)
                    ]
                    cands = []
                    for _, crow in candidates_df.iterrows():
                        cpts = load_points(crow["object_id"], scene_id)
                        if cpts is not None:
                            cands.append((crow["object_id"], cpts))
                    if cands:
                        nearest_vlm, nearest_dist = find_nearest_surface(
                            ref_pts, cands
                        )

            result = {
                "query_id": qid, "scene_id": scene_id, "operator": operator,
                "gt_value": gt_val,
                "gt_object_a": gt_ref, "gt_object_b": None,
                "grounded_a": grounded_ref, "grounded_b": None,
                "grounding_correct": grounding_correct,
                "e_total_surface": int(nearest_vlm == gt_answer)
                    if nearest_vlm else None,   # Top-1 para nearest
                "e_total_centroid": None,
                "vlm_response": response, "error": None,
            }
            # Renomeia coluna para clareza
            result["nearest_vlm_answer"] = nearest_vlm
            result["nearest_vlm_dist"]   = nearest_dist

        _save(OUTPUT_CSV, result)
        n_ok += 1
        print(f"  grounding_correct={grounding_correct}")
        time.sleep(0.8)

    print(f"\n{'='*60}")
    print(f"Concluído: {n_ok} OK | {n_err} erros")
    print(f"Resultados: {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(path: Path, data: dict) -> None:
    df = pd.DataFrame([data])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True,  index=False)


def _err_row(qid, scene_id, operator, err) -> dict:
    return {"query_id": qid, "scene_id": scene_id, "operator": operator,
            "gt_value": None, "gt_object_a": None, "gt_object_b": None,
            "grounded_a": None, "grounded_b": None,
            "grounding_correct": None,
            "e_total_surface": None, "e_total_centroid": None,
            "vlm_response": "ERROR", "error": err}


if __name__ == "__main__":
    main()