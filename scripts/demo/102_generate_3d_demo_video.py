#!/usr/bin/env python3
"""
93_generate_3d_demo_video.py

Vídeo de demonstração 3D estilo SpatialLM:
  - Câmera voa pela cena scene0142_00 em trajetória suave
  - Bounding boxes 3D acendem nos objetos relevantes
  - Texto da query aparece em digitação progressiva
  - Resposta aparece com animação

Pipeline:
  PyVista (render offscreen) → PIL (overlay texto) → ffmpeg (MP4)

Queries:
  1. distance  : qual a distância entre chair1 e monitor1?       → 0.355 m
  2. nearest   : qual chair está mais próxima de table4?         → chair1
  3. between   : chair5 está entre cabinet3 e cabinet4?          → Não
  4. aligned   : cabinet1, table1 e desk1 estão alinhados?       → Sim

Uso:
    python scripts/93_generate_3d_demo_video.py
    python scripts/93_generate_3d_demo_video.py --fps 30
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
SCANS_DIR    = ROOT / "data" / "scannet" / "scans"
POINTS_DIR   = ROOT / "artifacts" / "object_points_test_official_stage1"
MANIFEST_CSV = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"
ALIAS_CSV    = ROOT / "benchmark" / "demo_alias_maps" / "scene0142_00_alias_map.csv"
OUT_DIR      = ROOT / "artifacts" / "demo_video"
SCENE_ID     = "scene0142_00"

# ---------------------------------------------------------------------------
# Vídeo
# ---------------------------------------------------------------------------
W, H = 1280, 720
FPS  = 24

# Cores
BG_COLOR  = (15, 15, 25)
WHITE     = (240, 240, 240)
GRAY      = (150, 150, 165)
ACCENT    = (0, 200, 140)
RED_C     = (220, 70, 70)
DIM_BOX   = (80, 80, 90)

# Cores das bboxes por papel na query
COLOR_A   = (220, 50,  50)   # objeto A / referência
COLOR_B   = (50,  100, 220)  # objeto B / candidatos
COLOR_C   = (220, 170, 30)   # objeto C (terceiro)
COLOR_ANS = (0,   210, 130)  # vencedor / resposta


# ---------------------------------------------------------------------------
# Definição das queries e trajetórias
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "label":    "DISTANCE",
        "text":     "Qual a distância entre chair1 e monitor1?",
        "answer":   "0,355 m",
        "ok":       True,
        "objects":  {
            "chair1":   COLOR_A,
            "monitor1": COLOR_B,
        },
        # Câmera: parte de visão geral, vai até o par de objetos
        "cam_path": [
            # (pos_xyz, target_xyz)
            ((4.0, -2.5, 4.5), (4.0, 3.5, 0.8)),   # overview geral
            ((4.5,  0.5, 2.8), (4.5, 3.5, 0.9)),   # aproximando
            ((5.0,  1.5, 1.8), (4.5, 4.5, 1.1)),   # zoom nos objetos
        ],
    },
    {
        "label":    "NEAREST",
        "text":     "Qual chair está mais próxima de table4?",
        "answer":   "chair1",
        "ok":       True,
        "objects":  {
            "table4": COLOR_A,
            "chair1": COLOR_ANS,
            "chair2": COLOR_B,
            "chair3": COLOR_B,
            "chair5": COLOR_B,
        },
        "cam_path": [
            ((7.0, -1.5, 4.0), (5.0, 3.0, 0.8)),
            ((6.5,  1.0, 2.5), (5.2, 2.8, 0.9)),
            ((6.0,  2.0, 1.5), (5.5, 2.5, 0.9)),
        ],
    },
    {
        "label":    "BETWEEN",
        "text":     "chair5 está entre cabinet3 e cabinet4?",
        "answer":   "Não",
        "ok":       False,
        "objects":  {
            "chair5":   COLOR_A,
            "cabinet3": COLOR_B,
            "cabinet4": COLOR_C,
        },
        "cam_path": [
            ((2.0, -2.0, 4.0), (3.5, 1.5, 0.6)),
            ((2.5,  0.0, 2.5), (3.8, 1.2, 0.6)),
            ((3.0,  0.5, 1.5), (3.8, 1.0, 0.7)),
        ],
    },
    {
        "label":    "ALIGNED",
        "text":     "cabinet1, table1 e desk1 estão alinhados?",
        "answer":   "Sim",
        "ok":       True,
        "objects":  {
            "cabinet1": COLOR_A,
            "table1":   COLOR_B,
            "desk1":    COLOR_C,
        },
        "cam_path": [
            ((0.0, -2.0, 4.5), (2.5, 3.5, 0.8)),
            ((0.5,  1.0, 3.0), (2.5, 3.5, 0.8)),
            ((1.0,  2.0, 1.8), (2.5, 3.5, 0.8)),
        ],
    },
]

# Frames por fase (ajuste para ritmo do vídeo)
T = {
    "fly_in":     FPS * 3,    # câmera voando até posição inicial
    "orbit":      FPS * 2,    # órbita enquanto bbox acende
    "type":       FPS * 2,    # digitação da query
    "hold":       FPS * 1,    # hold antes da resposta
    "answer":     FPS * 2,    # resposta aparece + hold
    "fly_next":   FPS * 1,    # transição
}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp_v(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * ease(t)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Carrega mesh PLY
# ---------------------------------------------------------------------------

def load_mesh(scene_id: str) -> pv.PolyData:
    ply_path = sorted((SCANS_DIR / scene_id).glob("*_vh_clean_2.ply"))[0]
    ply = PlyData.read(str(ply_path))
    v   = ply["vertex"]
    pts = np.column_stack([np.asarray(v["x"], np.float32),
                           np.asarray(v["y"], np.float32),
                           np.asarray(v["z"], np.float32)])
    mesh = pv.PolyData(pts)
    if "face" in ply:
        raw   = ply["face"].data["vertex_indices"]
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


# ---------------------------------------------------------------------------
# Renderiza frame 3D via PyVista
# ---------------------------------------------------------------------------

def render_frame(
    mesh: pv.PolyData,
    obj_df: pd.DataFrame,
    highlight: dict[str, tuple],   # alias → color_rgb
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    bbox_alpha: float = 1.0,       # 0→1 para acender as bbox
) -> np.ndarray:
    """Retorna array RGB (H, W, 3)."""
    pl = pv.Plotter(off_screen=True, window_size=[W, H])
    pl.set_background([c / 255 for c in BG_COLOR])

    # Mesh da cena
    if "rgb" in mesh.array_names:
        mesh["colors"] = mesh["rgb"].astype(float) / 255.0
        pl.add_mesh(mesh, scalars="colors", rgb=True, opacity=0.75,
                    show_scalar_bar=False)
    else:
        pl.add_mesh(mesh, color="lightgray", opacity=0.75)

    # Bounding boxes de todos os objetos (dimmer)
    for _, row in obj_df.iterrows():
        alias = str(row["alias"])
        x0, y0, z0 = row["aabb_min_x"], row["aabb_min_y"], row["aabb_min_z"]
        x1, y1, z1 = row["aabb_max_x"], row["aabb_max_y"], row["aabb_max_z"]
        box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1))

        if alias in highlight:
            color = highlight[alias]
            lw    = 4
            alpha = bbox_alpha
        else:
            color = DIM_BOX
            lw    = 1
            alpha = 0.35

        pl.add_mesh(box, style="wireframe",
                    color=[c / 255 for c in color],
                    line_width=lw, opacity=alpha)

    # Câmera
    pl.camera_position = [
        tuple(cam_pos.tolist()),
        tuple(cam_target.tolist()),
        (0.0, 0.0, 1.0),
    ]

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ---------------------------------------------------------------------------
# Overlay PIL (painel de texto)
# ---------------------------------------------------------------------------

FONT_OP   = None
FONT_Q    = None
FONT_ANS  = None
FONT_SM   = None

PANEL_H   = 140
PANEL_Y   = H - PANEL_H


def init_fonts() -> None:
    global FONT_OP, FONT_Q, FONT_ANS, FONT_SM
    FONT_OP  = _font(20)
    FONT_Q   = _font(26)
    FONT_ANS = _font(34)
    FONT_SM  = _font(15)


def overlay_text(
    img_arr: np.ndarray,
    operator: str,
    typed: str,
    answer: str | None,
    answer_ok: bool,
    answer_alpha: float,
    show_cursor: bool = True,
) -> Image.Image:
    img  = Image.fromarray(img_arr).convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw  = ImageDraw.Draw(over)

    # Painel semitransparente
    drw.rectangle([0, PANEL_Y, W, H], fill=(8, 8, 18, 215))
    drw.rectangle([0, PANEL_Y, W, PANEL_Y + 3], fill=(*ACCENT, 255))

    # Tag operador
    tag_text = f" {operator} "
    tw = drw.textlength(tag_text, font=FONT_OP) + 12
    drw.rectangle([20, PANEL_Y + 12, 20 + tw, PANEL_Y + 40],
                  fill=(*ACCENT, 255))
    drw.text((26, PANEL_Y + 13), tag_text, fill=(8, 8, 18), font=FONT_OP)

    # Query tipografada
    drw.text((20, PANEL_Y + 48), typed, fill=(*WHITE, 255), font=FONT_Q)

    # Cursor
    if show_cursor and answer is None:
        cx = 20 + int(drw.textlength(typed, font=FONT_Q))
        drw.rectangle([cx + 2, PANEL_Y + 50, cx + 12, PANEL_Y + 76],
                      fill=(*ACCENT, 255))

    # Resposta
    if answer and answer_alpha > 0:
        a_int = int(answer_alpha * 255)
        color = (*ACCENT, a_int) if answer_ok else (*RED_C, a_int)
        drw.text((20, PANEL_Y + 92),
                 f"→  {answer}", fill=color, font=FONT_ANS)

    result = Image.alpha_composite(img, over).convert("RGB")
    return result


# ---------------------------------------------------------------------------
# Gerador de trajetória suave (Catmull-Rom)
# ---------------------------------------------------------------------------

def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2*p0 - 5*p1 + 4*p2 - p3) * t2
        + (-p0 + 3*p1 - 3*p2 + p3) * t3
    )


def smooth_path(waypoints: list[tuple], n_frames: int) -> list[np.ndarray]:
    """Interpola waypoints com Catmull-Rom."""
    pts = [np.array(w, dtype=float) for w in waypoints]
    # Repete extremos para Catmull-Rom
    pts = [pts[0]] + pts + [pts[-1]]
    n_seg = len(pts) - 3
    frames = []
    for i in range(n_frames):
        t_global = i / max(n_frames - 1, 1) * n_seg
        seg = min(int(t_global), n_seg - 1)
        t   = t_global - seg
        p   = catmull_rom(pts[seg], pts[seg+1], pts[seg+2], pts[seg+3], t)
        frames.append(p)
    return frames


# ---------------------------------------------------------------------------
# Computa resposta geométrica real
# ---------------------------------------------------------------------------

def compute_surface_distance(alias_a: str, alias_b: str,
                              alias_df: pd.DataFrame) -> float:
    def load(alias):
        row  = alias_df[alias_df["alias"] == alias].iloc[0]
        path = ROOT / row["points_path"]
        return np.load(path)["points"]

    pts_a = load(alias_a)
    pts_b = load(alias_b)
    tree  = cKDTree(pts_b)
    d, _  = tree.query(pts_a, k=1, workers=-1)
    return float(d.min())


# ---------------------------------------------------------------------------
# Título e fim
# ---------------------------------------------------------------------------

def make_title_frame(text: str, sub: str) -> np.ndarray:
    img  = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_big = _font(46)
    font_sub = _font(20)

    tw  = draw.textlength(text, font=font_big)
    tw2 = draw.textlength(sub,  font=font_sub)
    draw.rectangle([(W - 500) // 2, H // 2 - 60,
                    (W + 500) // 2, H // 2 - 57], fill=ACCENT)
    draw.text(((W - tw) // 2,  H // 2 - 48), text, fill=WHITE, font=font_big)
    draw.text(((W - tw2) // 2, H // 2 + 18), sub,  fill=GRAY,  font=font_sub)
    return np.array(img)


def crossfade_arrays(a: np.ndarray, b: np.ndarray,
                     t: float) -> np.ndarray:
    return (a * (1 - t) + b * t).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--out", type=str,
                        default=str(OUT_DIR / "spatial_3d_demo.mp4"))
    args = parser.parse_args()

    init_fonts()

    print("Carregando dados ...")
    manifest = pd.read_csv(MANIFEST_CSV)
    alias_df = pd.read_csv(ALIAS_CSV)
    scene    = manifest[manifest["scene_id"] == SCENE_ID].copy()
    obj_df   = alias_df.merge(
        scene[["object_id", "aabb_min_x", "aabb_min_y", "aabb_min_z",
               "aabb_max_x", "aabb_max_y", "aabb_max_z"]],
        on="object_id"
    )

    print("Carregando mesh PLY ...")
    mesh = load_mesh(SCENE_ID)

    # Computa distância real para query 1
    d_real = compute_surface_distance("chair1", "monitor1", alias_df)
    QUERIES[0]["answer"] = f"{d_real:.3f} m".replace(".", ",")
    print(f"  chair1 ↔ monitor1: {d_real:.3f} m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="3d_demo_"))
    frame_idx = 0

    def save(arr_or_img):
        nonlocal frame_idx
        if isinstance(arr_or_img, Image.Image):
            arr_or_img.save(tmp / f"f{frame_idx:06d}.png")
        else:
            Image.fromarray(arr_or_img).save(tmp / f"f{frame_idx:06d}.png")
        frame_idx += 1

    # ------------------------------------------------------------------ #
    # TÍTULO
    # ------------------------------------------------------------------ #
    print("Gerando slide de título ...")
    title_arr = make_title_frame(
        "Raciocínio Espacial Métrico em Cenas 3D",
        "scene0142_00 · ScanNet · Motor Geométrico Explícito"
    )
    n_title = int(args.fps * 2.5)
    for i in range(n_title):
        save(title_arr)

    # ------------------------------------------------------------------ #
    # OVERVIEW INICIAL — câmera alta, visão geral da sala
    # ------------------------------------------------------------------ #
    print("Gerando overview inicial ...")
    overview_pos    = np.array([3.5, -3.0, 6.0])
    overview_target = np.array([3.5,  3.5, 0.7])

    n_overview = int(args.fps * 2)
    for i in range(n_overview):
        t     = i / n_overview
        # Fade-in (primeiros 0.5s)
        alpha = min(t * args.fps * 2, 1.0)
        arr   = render_frame(mesh, obj_df, {},
                             overview_pos, overview_target, bbox_alpha=0)
        if alpha < 1.0:
            arr = crossfade_arrays(title_arr, arr, alpha)
        img = overlay_text(arr, "", "", None, True, 0, False)
        save(img)

    # ------------------------------------------------------------------ #
    # QUERIES
    # ------------------------------------------------------------------ #
    prev_arr = render_frame(mesh, obj_df, {},
                            overview_pos, overview_target, bbox_alpha=0)

    for qi, q in enumerate(QUERIES):
        print(f"\n  Query {qi+1}: {q['label']}")
        operator  = q["label"]
        query_txt = q["text"]
        answer    = q["answer"]
        answer_ok = q["ok"]
        highlight = q["objects"]
        waypoints_pos    = [np.array(p[0]) for p in q["cam_path"]]
        waypoints_target = [np.array(p[1]) for p in q["cam_path"]]

        # 1. FLY-IN — câmera voa até posição inicial da query
        n_fly = T["fly_in"]
        pos_path    = smooth_path(
            [overview_pos.tolist()] + [waypoints_pos[0].tolist()], n_fly)
        target_path = smooth_path(
            [overview_target.tolist()] + [waypoints_target[0].tolist()], n_fly)

        for i in range(n_fly):
            arr = render_frame(mesh, obj_df, {},
                               pos_path[i], target_path[i], bbox_alpha=0)
            img = overlay_text(arr, operator, "", None, answer_ok, 0, False)
            save(img)
            prev_arr = arr

        # 2. ORBIT — câmera orbita, bbox acendem progressivamente
        n_orb = T["orbit"]
        pos_path    = smooth_path([p.tolist() for p in waypoints_pos],    n_orb)
        target_path = smooth_path([p.tolist() for p in waypoints_target], n_orb)

        for i in range(n_orb):
            ba  = ease(i / n_orb)
            arr = render_frame(mesh, obj_df, highlight,
                               pos_path[i], target_path[i], bbox_alpha=ba)
            img = overlay_text(arr, operator, "", None, answer_ok, 0, False)
            save(img)
            prev_arr = arr

        # 3. DIGITAÇÃO da query (câmera parada na última posição da órbita)
        n_type = T["type"]
        for i in range(n_type):
            n_chars = min(len(query_txt),
                          int(i * len(query_txt) / n_type) + 1)
            arr = render_frame(mesh, obj_df, highlight,
                               pos_path[-1], target_path[-1], bbox_alpha=1.0)
            img = overlay_text(arr, operator, query_txt[:n_chars],
                               None, answer_ok, 0,
                               show_cursor=True)
            save(img)
            prev_arr = arr

        # 4. HOLD
        for _ in range(T["hold"]):
            arr = render_frame(mesh, obj_df, highlight,
                               pos_path[-1], target_path[-1], bbox_alpha=1.0)
            img = overlay_text(arr, operator, query_txt,
                               None, answer_ok, 0, show_cursor=False)
            save(img)
            prev_arr = arr

        # 5. RESPOSTA aparece + hold
        n_ans = T["answer"]
        for i in range(n_ans):
            t_a = min(i / (n_ans * 0.4), 1.0)  # fade rápido
            arr = render_frame(mesh, obj_df, highlight,
                               pos_path[-1], target_path[-1], bbox_alpha=1.0)
            img = overlay_text(arr, operator, query_txt,
                               answer, answer_ok, ease(t_a),
                               show_cursor=False)
            save(img)
            prev_arr = arr

        # 6. TRANSIÇÃO para próxima query (fade para overview)
        if qi < len(QUERIES) - 1:
            n_tr = T["fly_next"]
            for i in range(n_tr):
                t = i / n_tr
                arr_next = render_frame(mesh, obj_df, {},
                                        overview_pos, overview_target,
                                        bbox_alpha=0)
                arr = crossfade_arrays(prev_arr, arr_next, ease(t))
                img = overlay_text(arr, "", "", None, True, 0, False)
                save(img)
            overview_pos    = waypoints_pos[-1].copy()
            overview_target = waypoints_target[-1].copy()

    # ------------------------------------------------------------------ #
    # SLIDE FINAL
    # ------------------------------------------------------------------ #
    print("\nGerando slide final ...")
    end_arr = make_title_frame(
        "PPGCOMP · UFBA · 2026",
        "github.com/grifo114/metric-spatial-vlm"
    )
    # Fade da última cena para o slide final
    n_fade_end = int(args.fps * 0.8)
    for i in range(n_fade_end):
        t = ease(i / n_fade_end)
        arr = crossfade_arrays(prev_arr, end_arr, t)
        save(arr)

    n_end = int(args.fps * 3)
    for _ in range(n_end):
        save(end_arr)

    # ------------------------------------------------------------------ #
    # FFMPEG
    # ------------------------------------------------------------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nMontando MP4: {frame_idx} frames @ {args.fps}fps "
          f"({frame_idx/args.fps:.1f}s) ...")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(tmp / "f%06d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp)

    if result.returncode != 0:
        print("ERRO ffmpeg:")
        print(result.stderr[-600:])
    else:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"Vídeo salvo: {out_path}")
        print(f"Tamanho: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()