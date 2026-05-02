#!/usr/bin/env python3
"""
83_e2e_grounding_test_official_v2.py

Versão estendida do script 83 com Spatial Context Injection.

==============================================================
Métrica usada (IMPORTANTE — Métrica A: grounding rate)
==============================================================
Este script mede a *taxa de grounding correto*, igual ao 83 original:

  - distance(label_a, label_b):
        grounding_correct = VLM identificou os dois objetos certos
                            (gt_object_a e gt_object_b).
        O motor geométrico calcula a distância depois.

  - nearest(target_category, reference_label):
        grounding_correct = VLM identificou o objeto de referência certo
                            (gt_ref).
        O motor geométrico calcula qual instância de target_category está
        mais próxima depois (find_nearest_surface).

Esta é a métrica que produziu os números 33% (distance) / 38.5% (nearest)
do pilot anterior. NÃO é "VLM acertou a resposta da query"; é "VLM
identificou a referência certa entre instâncias da mesma categoria".

==============================================================
Diferenças vs 83 original
==============================================================

  --prompt-mode {original,context}
      original : prompt funcionalmente idêntico ao 83 original.
      context  : a lista de objetos numerados ganha um descritor
                 scene-relative (sem vazar GT) entre parênteses.

  --descriptor-level {1,2,3}
      Profundidade do descritor (só usado em --prompt-mode context):
        1 : quadrante na cena
        2 : quadrante + ordenação entre objetos da mesma categoria (default)
        3 : quadrante + ordenação + categorias-vizinhas mais próximas

  --language {pt,en}
      Idioma dos descritores. Default = pt.

  --output-suffix STR
      Sufixo opcional do CSV de saída.

  --print-prompts
      Imprime o prompt antes de chamar a API. Use com --no-api para sanity
      check sem gastar token.

  --no-api
      Não chama API e não renderiza imagem. Apenas monta os prompts.

  --provider {openai,openrouter}
      Backend de API. 'openai' usa api.openai.com com OPENAI_API_KEY
      (default). 'openrouter' usa openrouter.ai/api/v1 com
      OPENROUTER_API_KEY. OpenRouter oferece API OpenAI-compatível para
      acesso unificado a múltiplos provedores (Qwen, Anthropic, Google,
      etc.) sem mudança no formato da requisição.

  --model NAME
      Identificador do modelo. Default = "gpt-4.1" (apropriado para
      provider=openai). Para OpenRouter, use os identificadores oficiais
      como "qwen/qwen2.5-vl-7b-instruct" ou
      "qwen/qwen2.5-vl-72b-instruct". Veja https://openrouter.ai/models
      para a lista completa.

==============================================================
Mudanças metodológicas vs versão anterior do v2
==============================================================

1. Exemplo no fim do prompt agora é ADAPTATIVO. O 83 original usava
   "scene0008_00__monitor_029, scene0008_00__table_032" hardcoded, o que
   citava IDs reais e potencialmente o próprio GT em scene0008_00. Aqui
   o exemplo é sorteado deterministicamente entre objetos que NÃO são GT
   e cujas categorias NÃO são as da query (quando possível).

2. Parser de resposta aceita tanto IDs completos quanto números da lista
   (extraído da edição anterior — mantido por ser melhoria genuína).

3. CSV de saída ganha colunas: prompt_mode, descriptor_level, language,
   example_a, example_b/example_ref. Permite auditoria post-hoc.

==============================================================
Uso típico
==============================================================

    # Sanity check seco (sem API):
    python scripts/83_e2e_grounding_test_official_v2.py \\
        --prompt-mode context --descriptor-level 2 \\
        --dry-run 5 --print-prompts --no-api

    # Reproduzir baseline com template corrigido (exemplo adaptativo):
    OPENAI_API_KEY=sk-... python scripts/83_e2e_grounding_test_official_v2.py \\
        --prompt-mode original --output-suffix _baseline_v2

    # Spatial context injection level 2:
    OPENAI_API_KEY=sk-... python scripts/83_e2e_grounding_test_official_v2.py \\
        --prompt-mode context --descriptor-level 2 \\
        --output-suffix _ctx_l2
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
# Path setup — make src.grounding importable regardless of where we run from
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.prompt_enrichment import (  # noqa: E402
    excluded_for_query,
    format_object_list,
    format_object_list_with_context,
)
from src.grounding.example_selection import pick_example_ids  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK    = ROOT / "benchmark"
SCANS_DIR    = ROOT / "data" / "scannet" / "scans"
POINTS_DIR   = ROOT / "artifacts" / "object_points_test_official_stage1"
RENDERS_DIR  = ROOT / "artifacts" / "e2e_grounding_renders"
RESULTS_DIR  = ROOT / "results" / "benchmark_v1"

GT_CSV       = BENCHMARK / "ground_truth_distance_nearest_test_official_stage1.csv"
QUERIES_CSV  = BENCHMARK / "queries_test_official_stage1_distance_nearest_final.csv"
MANIFEST_CSV = BENCHMARK / "objects_manifest_test_official_stage1.csv"

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 1024, 1024

CATEGORY_PALETTE = [
    (70,  130, 180), (34,  139,  34), (210, 105,  30), (148,   0, 211),
    (220,  20,  60), (255, 165,   0), (0,   139, 139), (184, 134,  11),
    (100, 149, 237), (85,  107,  47),
]

MAX_TOKENS  = 100
MAX_RETRIES = 3
RETRY_DELAY = 6

# API endpoints. Both providers expose an OpenAI-compatible /chat/completions
# interface, so the same payload format works for both. Provider is selected
# at runtime via --provider; model identifier is passed via --model.
API_ENDPOINTS = {
    "openai":     "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
API_KEY_ENV = {
    "openai":     "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# ---------------------------------------------------------------------------
# Geometric utilities (copied verbatim from 83 original)
# ---------------------------------------------------------------------------

def load_points(obj_id: str, scene_id: str) -> np.ndarray | None:
    path = POINTS_DIR / scene_id / f"{obj_id}.npz"
    if not path.exists():
        return None
    return np.load(path)["points"]


def surface_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
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
) -> tuple[str | None, float]:
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
# Render (copied verbatim from 83 original)
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
    number_map: dict[int, str],
    category_colors: dict[str, tuple],
    out_path: Path,
) -> None:
    pv.OFF_SCREEN = True
    mesh = _load_mesh(scene_id)

    plotter = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H])
    plotter.set_background("white")

    if "rgb" in mesh.array_names:
        mesh["colors"] = mesh["rgb"].astype(float) / 255.0
        plotter.add_mesh(mesh, scalars="colors", rgb=True, opacity=0.60)
    else:
        plotter.add_mesh(mesh, color="lightgray", opacity=0.60)

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

    def world_to_px(wx: float, wy: float) -> tuple[int, int]:
        px = int((wx - (cx - extent / 2)) * px_per_m)
        py = int(IMG_H - (wy - (cy - extent / 2)) * px_per_m)
        return px, py

    for num, row in num_to_row.items():
        cat   = str(row["label_norm"])
        color = category_colors.get(cat, (120, 120, 120))
        px, py = world_to_px(float(row["centroid_x"]), float(row["centroid_y"]))
        r = 8
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
        draw.text((px + r + 2, py - 7), str(num), fill=(0, 0, 0), font=font)

    bar_px = max(int(px_per_m), 10)
    bx, by = 20, IMG_H - 40
    draw.rectangle([bx, by, bx + bar_px, by + 6], fill=(0, 0, 0))
    draw.text((bx + bar_px // 2 - 8, by + 9), "1 m", fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=88)


# ---------------------------------------------------------------------------
# Number map (copied verbatim from 83 original)
# ---------------------------------------------------------------------------

def build_number_map(scene_df: pd.DataFrame):
    rows = scene_df.sort_values(["label_norm", "object_id"]).reset_index(drop=True)
    categories = rows["label_norm"].unique().tolist()
    cat_color = {c: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
                 for i, c in enumerate(sorted(categories))}
    num_to_id, id_to_num = {}, {}
    for i, row in rows.iterrows():
        n = i + 1
        oid = row["object_id"]
        num_to_id[n] = oid
        id_to_num[oid] = n
    return num_to_id, id_to_num, cat_color


# ---------------------------------------------------------------------------
# Prompts — bilingual. Selected by --language at runtime.
#
# EN templates use Option 2 (explicit role labeling): the prompt names each
# query category as "A" and "B" and tells the VLM exactly what to identify
# and in which order. This addresses the category-confusion failure mode
# observed in the PT pilot, where the VLM tended to answer in the target
# category instead of the reference category.
#
# Both templates use ADAPTIVE example IDs (pick_example_ids) — never the
# ground truth, never IDs from the query categories when possible.
# ---------------------------------------------------------------------------

PROMPTS_PT = {
    "distance": """\
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
separados por vírgula. Exemplo de formato:
{example_a}, {example_b}""",

    "nearest": """\
Você está observando uma cena 3D de ambiente interno em vista superior.
Cada objeto está marcado com um número (círculo colorido).

Objetos presentes na cena:
{object_list}

Consulta de proximidade: "Qual {target_category} está mais próximo \
de {reference_label}?"

Identifique qual objeto específico da lista é o {reference_label} \
referenciado como ponto de referência.
Se houver múltiplas instâncias da categoria, escolha a mais proeminente.

Responda APENAS com um ID exatamente como aparece na lista. \
Exemplo de formato:
{example_ref}""",
}

PROMPTS_EN = {
    "distance": """\
You are looking at a top-down view of a 3D indoor scene.
Each object is marked with a number (colored circle).

Objects in the scene:
{object_list}

The query asks for the distance between two objects:
- A: a {label_a}
- B: a {label_b}

Your task is to identify which specific object in the list is A (the {label_a}) \
and which is B (the {label_b}). If multiple instances of the same category \
exist in the scene, choose the most prominent or contextually meaningful one.

Respond ONLY with two object_ids from the list, separated by a comma. \
The first ID must be the {label_a} (A); the second must be the {label_b} (B). \
Example format:
{example_a}, {example_b}""",

    "nearest": """\
You are looking at a top-down view of a 3D indoor scene.
Each object is marked with a number (colored circle).

Objects in the scene:
{object_list}

The query asks: "Which {target_category} is closest to a {reference_label}?"

The query mentions two categories with different roles:
- TARGET: {target_category} (what we want to FIND — DO NOT pick this)
- REFERENCE: {reference_label} (the anchor — IDENTIFY this)

Your task is ONLY to identify which specific object in the list is the \
{reference_label} reference point. Do NOT pick the {target_category}; the \
geometric engine will compute the answer from the reference you provide. \
If multiple instances of {reference_label} exist, choose the most prominent \
or contextually meaningful one.

Respond ONLY with one object_id of category {reference_label} from the list. \
Example format:
{example_ref}""",
}


def get_prompt_template(language: str, operator: str) -> str:
    if language == "en":
        return PROMPTS_EN[operator]
    return PROMPTS_PT[operator]


# ---------------------------------------------------------------------------
# API call — provider-agnostic.
# ---------------------------------------------------------------------------
# Both OpenAI and OpenRouter accept the same chat/completions payload format.
# The only differences are (a) the URL, (b) the API key env var, and (c) the
# model identifier string. We pass all three explicitly so the script can be
# reproduced against either provider with identical request semantics.
#
# Note on rate limiting: OpenAI applies prompt caching automatically when the
# same image+prefix is reused across calls, which gave us ~75-90% discount on
# repeated calls within a scene during prior runs ($0.26 for 90 GPT-4.1 calls).
# OpenRouter forwards to the upstream provider; caching behavior depends on
# which model is selected.
# ---------------------------------------------------------------------------

def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(api_url: str, api_key: str, model: str,
             prompt: str, img_path: Path) -> str:
    """Send a multimodal chat completion request and return the raw text.

    Parameters
    ----------
    api_url : str
        Full URL of the chat/completions endpoint.
    api_key : str
        Bearer token for the provider.
    model : str
        Provider-specific model identifier (e.g. 'gpt-4.1' for OpenAI,
        'qwen/qwen2.5-vl-72b-instruct' for OpenRouter).
    prompt : str
        Textual prompt (assembled by the caller).
    img_path : Path
        Path to the rendered scene image (jpg/png).
    """
    b64 = img_to_b64(img_path)
    payload = {
        "model": model,
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
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"].strip()


def call_with_retry(api_url: str, api_key: str, model: str,
                    prompt: str, img_path: Path) -> str:
    """Wrapper around call_vlm with exponential-ish backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return call_vlm(api_url, api_key, model, prompt, img_path)
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}"
        except Exception as e:
            err = str(e)
        print(f"  [tentativa {attempt+1}/{MAX_RETRIES}] {err}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"API falhou após {MAX_RETRIES} tentativas")


# ---------------------------------------------------------------------------
# Response parsing
#
# Handles three response shapes:
#   - full id:    "scene0008_00__chair_011"
#   - number:     "10"
#   - mixed:      "scene0008_00__monitor_029, scene0008_00__table_032"
# ---------------------------------------------------------------------------

def extract_ids(
    response: str,
    valid_ids: set[str],
    n: int,
    num_to_id: dict[int, str] | None = None,
) -> list[str | None]:
    found: list[str] = []

    # 1) Full IDs
    for token in re.split(r"[\s,;]+", response):
        token = token.strip().strip(".,;\"'")
        if token in valid_ids and token not in found:
            found.append(token)
        if len(found) == n:
            break

    # 2) List numbers (e.g. "10" → num_to_id[10])
    if len(found) < n and num_to_id is not None:
        for token in re.split(r"[\s,;]+", response):
            token = token.strip().strip(".,;\"'")
            if token.isdigit():
                obj_id = num_to_id.get(int(token))
                if obj_id and obj_id in valid_ids and obj_id not in found:
                    found.append(obj_id)
            if len(found) == n:
                break

    while len(found) < n:
        found.append(None)
    return found[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Run on the first N queries only.")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip render generation if file already exists.")
    parser.add_argument("--operator", choices=["distance", "nearest", "all"],
                        default="all")
    parser.add_argument("--prompt-mode", choices=["original", "context"],
                        default="original",
                        help="original = baseline 83; context = inject "
                             "scene-relative descriptors into the object list.")
    parser.add_argument("--descriptor-level", type=int, choices=[1, 2, 3],
                        default=2,
                        help="Descriptor depth (only used in context mode).")
    parser.add_argument("--context-scope", choices=["all", "reference_only"],
                        default="all",
                        help="Where to inject context (only used in context "
                             "mode). 'all' adds descriptors to every object. "
                             "'reference_only' adds them only to candidates "
                             "of the reference category (nearest) or to both "
                             "query categories (distance).")
    parser.add_argument("--language", choices=["pt", "en"], default="pt",
                        help="Descriptor language. Defaults to PT to match the "
                             "language of the prompts. Use EN for Qwen runs.")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix appended to the output CSV name.")
    parser.add_argument("--print-prompts", action="store_true",
                        help="Print the assembled prompt for each query.")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip API calls and rendering. Useful with "
                             "--print-prompts and --dry-run.")
    parser.add_argument("--provider", choices=["openai", "openrouter"],
                        default="openai",
                        help="API backend. 'openai' uses api.openai.com with "
                             "OPENAI_API_KEY (default; matches all prior "
                             "GPT-4.1 runs). 'openrouter' uses "
                             "openrouter.ai/api/v1 with OPENROUTER_API_KEY.")
    parser.add_argument("--model", default="gpt-4.1",
                        help="Provider-specific model identifier. Default = "
                             "'gpt-4.1' (appropriate for --provider openai). "
                             "For --provider openrouter, use identifiers like "
                             "'qwen/qwen2.5-vl-7b-instruct' or "
                             "'qwen/qwen2.5-vl-72b-instruct'.")
    args = parser.parse_args()

    # --- Resolve API endpoint and key based on provider ----------------------
    api_url     = API_ENDPOINTS[args.provider]
    api_key_env = API_KEY_ENV[args.provider]
    api_key     = os.environ.get(api_key_env, "").strip()
    if not args.no_api and not api_key:
        sys.exit(f"{api_key_env} não definida (provider={args.provider}).")

    output_csv = RESULTS_DIR / f"e2e_grounding_test_official_raw{args.output_suffix}.csv"

    print(f"== Config ==")
    print(f"  provider         : {args.provider}")
    print(f"  model            : {args.model}")
    print(f"  prompt_mode      : {args.prompt_mode}")
    print(f"  descriptor_level : {args.descriptor_level}")
    print(f"  context_scope    : {args.context_scope}")
    print(f"  language         : {args.language}")
    print(f"  operator         : {args.operator}")
    print(f"  output           : {output_csv.name}")
    if args.no_api:
        print(f"  API CALLS        : DISABLED (--no-api)")
    print()

    # --- Load data ---
    gt_df       = pd.read_csv(GT_CSV)
    queries_df  = pd.read_csv(QUERIES_CSV)
    manifest_df = pd.read_csv(MANIFEST_CSV)

    queries_df = queries_df[queries_df["review_keep"] == "yes"].copy()

    extra = ["scene_id", "operator", "structured_query",
             "label_a", "label_b",
             "reference_object", "reference_label", "target_category",
             "answer_object"]
    for c in extra:
        if c not in queries_df.columns:
            queries_df[c] = None

    merged = gt_df.merge(
        queries_df[extra].drop_duplicates("structured_query"),
        on=["scene_id", "operator", "structured_query"], how="left",
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
    if output_csv.exists() and not args.no_api:
        done_ids = set(pd.read_csv(output_csv)["query_id"].tolist())
        print(f"Resumindo: {len(done_ids)} já processadas")

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scene_cache: dict[str, tuple] = {}

    def get_scene_data(scene_id: str):
        if scene_id not in scene_cache:
            sdf = manifest_df[
                (manifest_df["scene_id"] == scene_id) &
                (manifest_df["is_valid_object"] == True)
            ].copy()
            num_to_id, id_to_num, cat_color = build_number_map(sdf)
            render_path = RENDERS_DIR / f"{scene_id}_numbered.jpg"
            need_render = (not args.no_api) and (
                not args.skip_render or not render_path.exists()
            )
            if need_render:
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
            _save(output_csv, _err_row(qid, scene_id, operator, str(e)))
            n_err += 1
            continue

        valid_ids = set(sdf["object_id"].tolist())

        # ---- Build object list (the only place prompt_mode matters) ----
        if args.prompt_mode == "original":
            obj_list = format_object_list(num_to_id, sdf)
        else:
            excluded = excluded_for_query(
                operator,
                label_a=row.get("label_a"),
                label_b=row.get("label_b"),
                reference_label=row.get("reference_label"),
                target_category=row.get("target_category"),
            )

            if args.context_scope == "all":
                obj_list = format_object_list_with_context(
                    num_to_id, sdf,
                    level=args.descriptor_level,
                    excluded_categories=excluded,
                    language=args.language,
                    scope_category=None,
                )
            else:  # reference_only
                # For nearest: scope = reference_label (the thing the VLM
                #   needs to disambiguate).
                # For distance: scope is both query categories. We achieve
                #   this by calling the formatter twice and merging — but
                #   the simpler implementation is to call once with scope
                #   set to label_a then patch label_b's lines.
                if operator == "nearest":
                    scope = str(row.get("reference_label") or "")
                    obj_list = format_object_list_with_context(
                        num_to_id, sdf,
                        level=args.descriptor_level,
                        excluded_categories=excluded,
                        language=args.language,
                        scope_category=scope,
                    )
                elif operator == "distance":
                    # Build twice and merge: both label_a and label_b lines
                    # carry descriptors, others stay plain.
                    label_a_q = str(row.get("label_a") or "")
                    label_b_q = str(row.get("label_b") or "")
                    list_a = format_object_list_with_context(
                        num_to_id, sdf,
                        level=args.descriptor_level,
                        excluded_categories=excluded,
                        language=args.language,
                        scope_category=label_a_q,
                    ).splitlines()
                    list_b = format_object_list_with_context(
                        num_to_id, sdf,
                        level=args.descriptor_level,
                        excluded_categories=excluded,
                        language=args.language,
                        scope_category=label_b_q,
                    ).splitlines()
                    # Pick the version with a descriptor when either has one
                    merged_lines = []
                    for la, lb in zip(list_a, list_b):
                        merged_lines.append(la if "(" in la else lb)
                    obj_list = "\n".join(merged_lines)
                else:
                    obj_list = format_object_list(num_to_id, sdf)

        # ---- Assemble prompt ----
        if operator == "distance":
            label_a  = str(row.get("label_a") or "")
            label_b  = str(row.get("label_b") or "")
            gt_obj_a = str(row["gt_object_a"])
            gt_obj_b = str(row["gt_object_b"])
            gt_val   = float(row["gt_distance_m"])

            example_a, example_b = pick_example_ids(
                sdf, "distance",
                gt_object_a=gt_obj_a, gt_object_b=gt_obj_b,
                label_a=label_a, label_b=label_b,
                seed_key=qid,
            )
            if example_a is None or example_b is None:
                ids = sorted(valid_ids)
                example_a = ids[0] if ids else "OBJECT_A"
                example_b = ids[1] if len(ids) > 1 else "OBJECT_B"

            prompt = get_prompt_template(args.language, "distance").format(
                object_list=obj_list, label_a=label_a, label_b=label_b,
                example_a=example_a, example_b=example_b,
            )
            example_ref = None  # for unified result record below

        elif operator == "nearest":
            ref_label  = str(row.get("reference_label") or "")
            target_cat = str(row.get("target_category") or "")
            gt_ref     = str(row["gt_object_a"])
            gt_answer  = str(row.get("gt_answer_object", ""))
            gt_val     = gt_answer

            example_ref, _ = pick_example_ids(
                sdf, "nearest",
                gt_object_a=gt_ref,
                reference_label=ref_label, target_category=target_cat,
                seed_key=qid,
            )
            if example_ref is None:
                ids = sorted(valid_ids)
                example_ref = ids[0] if ids else "OBJECT_REF"

            prompt = get_prompt_template(args.language, "nearest").format(
                object_list=obj_list,
                target_category=target_cat,
                reference_label=ref_label,
                example_ref=example_ref,
            )
            example_a, example_b = None, None
        else:
            continue

        if args.print_prompts:
            print("  ----- PROMPT -----")
            for ln in prompt.splitlines():
                print(f"  | {ln}")
            print("  ----- END PROMPT -----")

        if args.no_api:
            n_ok += 1
            continue

        # ---- VLM call ----
        try:
            response = call_with_retry(api_url, api_key, args.model,
                                       prompt, render_path)
            print(f"  {args.model} → {response!r}")
        except Exception as e:
            print(f"  ERRO API: {e}")
            _save(output_csv, _err_row(qid, scene_id, operator, f"API_ERROR: {e}"))
            n_err += 1
            time.sleep(RETRY_DELAY)
            continue

        # ---- Parse + grounding check (Metric A) ----
        if operator == "distance":
            ids = extract_ids(response, valid_ids, 2, num_to_id=num_to_id)
            grounded_a, grounded_b = ids[0], ids[1]

            grounding_correct = (
                (grounded_a == gt_obj_a and grounded_b == gt_obj_b) or
                (grounded_a == gt_obj_b and grounded_b == gt_obj_a)
            )

            e_total_surface  = None
            e_total_centroid = None
            if grounded_a and grounded_b:
                pts_a = load_points(grounded_a, scene_id)
                pts_b = load_points(grounded_b, scene_id)
                if pts_a is not None and pts_b is not None:
                    e_total_surface  = surface_distance(pts_a, pts_b)
                    e_total_centroid = centroid_distance(pts_a, pts_b)

            result = {
                "query_id": qid, "scene_id": scene_id, "operator": operator,
                "provider": args.provider, "model": args.model,
                "prompt_mode": args.prompt_mode,
                "descriptor_level": args.descriptor_level if args.prompt_mode == "context" else None,
                "context_scope": args.context_scope if args.prompt_mode == "context" else None,
                "language": args.language,
                "example_a": example_a, "example_b": example_b,
                "example_ref": None,
                "gt_value": gt_val,
                "gt_object_a": gt_obj_a, "gt_object_b": gt_obj_b,
                "grounded_a": grounded_a, "grounded_b": grounded_b,
                "grounding_correct": grounding_correct,
                "e_total_surface": e_total_surface,
                "e_total_centroid": e_total_centroid,
                "nearest_vlm_answer": None,
                "nearest_vlm_dist":   None,
                "vlm_response": response, "error": None,
            }

        else:  # nearest — Metric A: VLM identifies the REFERENCE OBJECT.
            ids = extract_ids(response, valid_ids, 1, num_to_id=num_to_id)
            grounded_ref = ids[0]
            grounding_correct = (grounded_ref == gt_ref)

            # Geometric engine answers the actual nearest query, given the
            # VLM's grounding of the reference. If the VLM grounded the
            # wrong reference, the engine still computes nearest-from-that-
            # wrong-object, which is what e_total_surface compares to GT.
            nearest_vlm  = None
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
                            ref_pts, cands,
                        )

            result = {
                "query_id": qid, "scene_id": scene_id, "operator": operator,
                "provider": args.provider, "model": args.model,
                "prompt_mode": args.prompt_mode,
                "descriptor_level": args.descriptor_level if args.prompt_mode == "context" else None,
                "context_scope": args.context_scope if args.prompt_mode == "context" else None,
                "language": args.language,
                "example_a": None, "example_b": None,
                "example_ref": example_ref,
                "gt_value": gt_val,
                "gt_object_a": gt_ref, "gt_object_b": None,
                "grounded_a": grounded_ref, "grounded_b": None,
                "grounding_correct": grounding_correct,
                "e_total_surface": int(nearest_vlm == gt_answer)
                    if nearest_vlm else None,
                "e_total_centroid": None,
                "nearest_vlm_answer": nearest_vlm,
                "nearest_vlm_dist":   nearest_dist,
                "vlm_response": response, "error": None,
            }

        _save(output_csv, result)
        n_ok += 1
        print(f"  grounding_correct={grounding_correct}")
        time.sleep(0.8)

    print(f"\n{'='*60}")
    print(f"Concluído: {n_ok} OK | {n_err} erros")
    if not args.no_api:
        print(f"Resultados: {output_csv}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(path: Path, data: dict) -> None:
    df = pd.DataFrame([data])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def _err_row(qid, scene_id, operator, err) -> dict:
    return {
        "query_id": qid, "scene_id": scene_id, "operator": operator,
        "prompt_mode": None, "descriptor_level": None,
        "context_scope": None, "language": None,
        "example_a": None, "example_b": None, "example_ref": None,
        "gt_value": None, "gt_object_a": None, "gt_object_b": None,
        "grounded_a": None, "grounded_b": None,
        "grounding_correct": None,
        "e_total_surface": None, "e_total_centroid": None,
        "nearest_vlm_answer": None, "nearest_vlm_dist": None,
        "vlm_response": "ERROR", "error": err,
    }


if __name__ == "__main__":
    main()