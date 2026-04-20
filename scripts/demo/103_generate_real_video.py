#!/usr/bin/env python3
"""
94_generate_real_video.py

Vídeo de demonstração usando imagens reais do .sens da scene0142_00.

Pipeline:
  1. Varre o .sens para construir índice de offsets (sem carregar imagens)
  2. Lê só as poses (64 bytes por frame) para encontrar os melhores frames
  3. Para cada query, seleciona ~80 frames onde os objetos são mais visíveis
  4. Extrai esses frames, projeta as bboxes 3D em 2D, sobrepõe texto
  5. Monta MP4 via ffmpeg

Queries:
  1. DISTANCE  : qual a distância entre chair1 e monitor1?
  2. NEAREST   : qual chair está mais próxima de table4?
  3. BETWEEN   : chair5 está entre cabinet3 e cabinet4?
  4. ALIGNED   : cabinet1, table1 e desk1 estão alinhados?

Uso:
    python scripts/94_generate_real_video.py
    python scripts/94_generate_real_video.py --fps 24 --out demo_real.mp4
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
SENS_PATH    = ROOT / "data/scannet/scans/scene0142_00/scene0142_00.sens"
MANIFEST_CSV = ROOT / "benchmark/objects_manifest_test_official_stage1.csv"
ALIAS_CSV    = ROOT / "benchmark/demo_alias_maps/scene0142_00_alias_map.csv"
POINTS_DIR   = ROOT / "artifacts/object_points_test_official_stage1"
OUT_DIR      = ROOT / "artifacts/demo_video"
SCENE_ID     = "scene0142_00"

# ---------------------------------------------------------------------------
# Vídeo
# ---------------------------------------------------------------------------
W, H   = 1280, 720     # output (o .sens é 1296x968, redimensionamos)
FPS    = 24

BG     = (15, 15, 25)
WHITE  = (240, 240, 240)
GRAY   = (150, 150, 165)
ACCENT = (0, 200, 140)
RED_C  = (220, 70,  70)

COLOR_A   = (220, 50,  50)
COLOR_B   = (50, 100, 220)
COLOR_C   = (220, 170, 30)
COLOR_ANS = (0,  210, 130)
COLOR_DIM = (90, 90, 100)

PANEL_H = 145
PANEL_Y = H - PANEL_H

# Frames por fase
T_FLY   = FPS * 2      # frames de introdução (câmera se movendo)
T_BOX   = FPS * 1      # bbox acende
T_TYPE  = FPS * 2      # digitação
T_HOLD  = FPS * 1      # hold
T_ANS   = FPS * 2      # resposta + hold
T_TRANS = FPS * 1      # transição entre queries


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "label":   "DISTANCE",
        "text":    "Qual a distância entre chair1 e monitor1?",
        "answer":  None,      # calculado em runtime
        "ok":      True,
        "objects": {"chair1": COLOR_A, "monitor1": COLOR_B},
    },
    {
        "label":   "NEAREST",
        "text":    "Qual chair está mais próxima de table4?",
        "answer":  "chair1",
        "ok":      True,
        "objects": {
            "table4": COLOR_A,
            "chair1": COLOR_ANS,
            "chair2": COLOR_B,
            "chair3": COLOR_B,
            "chair5": COLOR_B,
        },
    },
    {
        "label":   "BETWEEN",
        "text":    "chair5 está entre cabinet3 e cabinet4?",
        "answer":  "Não",
        "ok":      False,
        "objects": {
            "chair5":   COLOR_A,
            "cabinet3": COLOR_B,
            "cabinet4": COLOR_C,
        },
    },
    {
        "label":   "ALIGNED",
        "text":    "cabinet1, table1 e desk1 estão alinhados?",
        "answer":  "Sim",
        "ok":      True,
        "objects": {
            "cabinet1": COLOR_A,
            "table1":   COLOR_B,
            "desk1":    COLOR_C,
        },
    },
]


# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for p in ["/System/Library/Fonts/Helvetica.ttc",
                  "/Library/Fonts/Arial.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if Path(p).exists():
                try:
                    _font_cache[size] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if size not in _font_cache:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


# ---------------------------------------------------------------------------
# Leitor .sens — índice de offsets
# ---------------------------------------------------------------------------

class SensIndex:
    """Lê o cabeçalho do .sens e indexa offsets de cada frame sem carregar imagens."""

    def __init__(self, path: Path):
        self.path = path
        self.num_frames     = 0
        self.color_w        = 0
        self.color_h        = 0
        self.intrinsic_color: np.ndarray = None
        self.frame_offsets: list[int] = []      # offset byte de cada frame
        self.color_sizes:   list[int] = []      # tamanho do JPEG de cada frame
        self.depth_sizes:   list[int] = []
        self._header_end    = 0
        self._build()

    def _build(self):
        print("  Construindo índice do .sens ...")
        with open(self.path, "rb") as f:
            # Cabeçalho
            f.read(4)                          # version
            slen = struct.unpack("Q", f.read(8))[0]
            f.read(slen)                       # sensor_name
            raw = f.read(16 * 4)               # intrinsic_color (4x4 float32)
            self.intrinsic_color = np.frombuffer(raw, np.float32).reshape(4, 4)
            f.read(16 * 4 * 3)                 # 3 outras matrizes
            f.read(4 + 4)                      # color_code, depth_code
            self.color_w = struct.unpack("I", f.read(4))[0]
            self.color_h = struct.unpack("I", f.read(4))[0]
            f.read(4 + 4)                      # depth_w, depth_h
            f.read(4)                          # depth_shift
            self.num_frames = struct.unpack("Q", f.read(8))[0]
            self._header_end = f.tell()

            # Índice: lê pose + tamanhos, pula dados
            for i in range(self.num_frames):
                self.frame_offsets.append(f.tell())
                f.read(64)                     # camera_to_world (4x4 float32)
                f.read(8 + 8)                  # timestamps
                csz = struct.unpack("Q", f.read(8))[0]
                dsz = struct.unpack("Q", f.read(8))[0]
                self.color_sizes.append(csz)
                self.depth_sizes.append(dsz)
                f.seek(csz + dsz, 1)           # pula imagens

                if i % 200 == 0:
                    print(f"    {i}/{self.num_frames} frames indexados ...",
                          end="\r", flush=True)

        print(f"  Índice pronto: {self.num_frames} frames.")

    def read_pose(self, idx: int) -> np.ndarray:
        with open(self.path, "rb") as f:
            f.seek(self.frame_offsets[idx])
            raw = f.read(64)
        return np.frombuffer(raw, np.float32).reshape(4, 4)

    def read_color(self, idx: int) -> np.ndarray:
        """Retorna array RGB (H, W, 3)."""
        with open(self.path, "rb") as f:
            f.seek(self.frame_offsets[idx] + 64 + 8 + 8 + 8 + 8)
            data = f.read(self.color_sizes[idx])
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)


# ---------------------------------------------------------------------------
# Geometria: projeção 3D → 2D
# ---------------------------------------------------------------------------

def project_point(p_world: np.ndarray,
                  cam_to_world: np.ndarray,
                  K: np.ndarray,
                  orig_w: int, orig_h: int,
                  out_w: int, out_h: int) -> tuple[int, int, float] | None:
    """Retorna (u, v, depth) em coordenadas da imagem redimensionada, ou None."""
    world_to_cam = np.linalg.inv(cam_to_world)
    p_h  = np.append(p_world, 1.0)
    p_c  = world_to_cam @ p_h
    if p_c[2] <= 0.1:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u_orig = fx * p_c[0] / p_c[2] + cx
    v_orig = fy * p_c[1] / p_c[2] + cy
    # Escala para resolução de saída
    u = int(u_orig * out_w / orig_w)
    v = int(v_orig * out_h / orig_h)
    return u, v, float(p_c[2])


def project_bbox(
    aabb_min: np.ndarray, aabb_max: np.ndarray,
    cam_to_world: np.ndarray, K: np.ndarray,
    orig_w: int, orig_h: int, out_w: int, out_h: int,
) -> list[tuple[int, int]] | None:
    """Projeta os 8 cantos do AABB. Retorna lista de (u, v) ou None."""
    corners = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ])
    projected = []
    for c in corners:
        r = project_point(c, cam_to_world, K, orig_w, orig_h, out_w, out_h)
        if r is None or r[2] > 15:
            return None
        projected.append((r[0], r[1]))
    return projected


BBOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),  # base
    (4,5),(5,6),(6,7),(7,4),  # topo
    (0,4),(1,5),(2,6),(3,7),  # laterais
]


def draw_bbox_3d(draw: ImageDraw.Draw,
                 pts: list[tuple[int, int]],
                 color: tuple, lw: int = 3) -> None:
    for i, j in BBOX_EDGES:
        x0, y0 = pts[i]
        x1, y1 = pts[j]
        draw.line([(x0, y0), (x1, y1)], fill=color, width=lw)


def draw_label_3d(draw: ImageDraw.Draw,
                  pts: list[tuple[int, int]],
                  label: str, color: tuple) -> None:
    """Label acima do ponto mais alto do bbox."""
    top_pt = min(pts, key=lambda p: p[1])
    tx, ty = top_pt[0] - 20, top_pt[1] - 28
    tx = max(4, min(W - 80, tx))
    ty = max(4, ty)
    fnt = font(18)
    tw  = draw.textlength(label, font=fnt) + 10
    draw.rectangle([tx - 2, ty - 2, tx + tw, ty + 22],
                   fill=(10, 10, 20, 200))
    draw.text((tx + 4, ty), label, fill=color, font=fnt)


# ---------------------------------------------------------------------------
# Scoring de frames por visibilidade dos objetos
# ---------------------------------------------------------------------------

def score_frame(pose: np.ndarray, K: np.ndarray,
                centroids: list[np.ndarray],
                orig_w: int, orig_h: int) -> float:
    """Score = nr de objetos visíveis e próximos do centro da imagem."""
    score = 0.0
    for c in centroids:
        r = project_point(c, pose, K, orig_w, orig_h, W, H)
        if r is None:
            continue
        u, v, depth = r
        if 0 < u < W and 0 < v < H and 0.3 < depth < 8.0:
            # Penaliza objetos na borda
            cx_dist = abs(u - W // 2) / (W // 2)
            cy_dist = abs(v - H // 2) / (H // 2)
            score  += 1.0 - 0.4 * (cx_dist + cy_dist) / 2
    return score


def find_best_window(index: SensIndex, K: np.ndarray,
                     centroids: list[np.ndarray],
                     n_frames: int,
                     stride: int = 3) -> list[int]:
    """
    Encontra a janela de n_frames consecutivos com maior visibilidade.
    Amostra a cada `stride` frames para eficiência.
    """
    print(f"    Avaliando visibilidade ({index.num_frames} frames, stride={stride}) ...")
    scores = []
    sampled = list(range(0, index.num_frames, stride))
    for i in sampled:
        pose = index.read_pose(i)
        s    = score_frame(pose, K, centroids, index.color_w, index.color_h)
        scores.append((i, s))

    # Janela deslizante sobre frames amostrados
    win = max(1, n_frames // stride)
    best_start, best_score = 0, -1.0
    for i in range(len(scores) - win + 1):
        s = sum(sc for _, sc in scores[i:i + win])
        if s > best_score:
            best_score = s
            best_start = scores[i][0]

    # Expande para stride=1 dentro da janela
    start = max(0, best_start)
    end   = min(index.num_frames - 1, best_start + n_frames)
    return list(range(start, end))


# ---------------------------------------------------------------------------
# Overlay de texto (painel inferior)
# ---------------------------------------------------------------------------

def overlay_text(img: Image.Image,
                 operator: str,
                 typed: str,
                 answer: str | None,
                 answer_ok: bool,
                 answer_alpha: float,
                 show_cursor: bool = True) -> Image.Image:
    img  = img.convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw  = ImageDraw.Draw(over)

    drw.rectangle([0, PANEL_Y, W, H], fill=(8, 8, 18, 220))
    drw.rectangle([0, PANEL_Y, W, PANEL_Y + 3], fill=(*ACCENT, 255))

    if operator:
        tag = f" {operator} "
        tw  = drw.textlength(tag, font=font(20)) + 12
        drw.rectangle([20, PANEL_Y + 12, 20 + tw, PANEL_Y + 40],
                      fill=(*ACCENT, 255))
        drw.text((26, PANEL_Y + 13), tag, fill=(8, 8, 18), font=font(20))

    if typed:
        drw.text((20, PANEL_Y + 48), typed,
                 fill=(*WHITE, 255), font=font(26))
        if show_cursor and answer is None:
            cx = 20 + int(drw.textlength(typed, font=font(26)))
            drw.rectangle([cx + 2, PANEL_Y + 50,
                           cx + 12, PANEL_Y + 76],
                          fill=(*ACCENT, 255))

    if answer and answer_alpha > 0:
        a_int = int(answer_alpha * 255)
        col   = (*ACCENT, a_int) if answer_ok else (*RED_C, a_int)
        drw.text((20, PANEL_Y + 88), f"→  {answer}",
                 fill=col, font=font(34))

    return Image.alpha_composite(img, over).convert("RGB")


# ---------------------------------------------------------------------------
# Distância de superfície
# ---------------------------------------------------------------------------

def surface_distance(alias_a: str, alias_b: str,
                     alias_df: pd.DataFrame) -> float:
    def pts(a):
        row  = alias_df[alias_df["alias"] == a].iloc[0]
        return np.load(ROOT / row["points_path"])["points"]
    pa, pb = pts(alias_a), pts(alias_b)
    t = CKDTree = cKDTree(pb)
    d, _ = t.query(pa, k=1, workers=-1)
    return float(d.min())


# ---------------------------------------------------------------------------
# Gera frames de um segmento de query
# ---------------------------------------------------------------------------

def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (a * (1 - t) + b * t).astype(np.uint8)


def frames_for_query(
    index: SensIndex,
    K: np.ndarray,
    obj_rows: pd.DataFrame,   # linhas do obj_df para esta query
    highlight: dict[str, tuple],
    query_text: str,
    answer: str,
    answer_ok: bool,
    operator: str,
) -> list[Image.Image]:
    """Retorna lista de imagens PIL para o segmento desta query."""

    centroids = [
        np.array([r["centroid_x"], r["centroid_y"], r["centroid_z"]])
        for _, r in obj_rows.iterrows()
    ]

    n_total = T_FLY + T_BOX + T_TYPE + T_HOLD + T_ANS
    frame_ids = find_best_window(index, K, centroids, n_total, stride=3)

    # Garante comprimento exato
    while len(frame_ids) < n_total:
        frame_ids.append(frame_ids[-1])
    frame_ids = frame_ids[:n_total]

    frames = []
    n_chars_total = len(query_text)

    for phase_i, fid in enumerate(frame_ids):
        color_arr = index.read_color(fid)
        pose      = index.read_pose(fid)

        # Redimensiona para (W, H)
        img = Image.fromarray(color_arr).resize((W, H), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Determina fase
        if phase_i < T_FLY:
            # Fase 1: câmera se movendo, sem bbox ainda
            bbox_alpha   = 0.0
            typed        = ""
            ans_alpha    = 0.0
            show_cursor  = False

        elif phase_i < T_FLY + T_BOX:
            # Fase 2: bbox acende
            t_b          = (phase_i - T_FLY) / T_BOX
            bbox_alpha   = ease(t_b)
            typed        = ""
            ans_alpha    = 0.0
            show_cursor  = False

        elif phase_i < T_FLY + T_BOX + T_TYPE:
            # Fase 3: digitação
            t_t          = (phase_i - T_FLY - T_BOX) / T_TYPE
            n_chars      = min(n_chars_total, int(t_t * n_chars_total) + 1)
            bbox_alpha   = 1.0
            typed        = query_text[:n_chars]
            ans_alpha    = 0.0
            show_cursor  = True

        elif phase_i < T_FLY + T_BOX + T_TYPE + T_HOLD:
            # Fase 4: hold
            bbox_alpha   = 1.0
            typed        = query_text
            ans_alpha    = 0.0
            show_cursor  = False

        else:
            # Fase 5: resposta
            t_a          = (phase_i - T_FLY - T_BOX - T_TYPE - T_HOLD) / T_ANS
            bbox_alpha   = 1.0
            typed        = query_text
            ans_alpha    = ease(min(t_a * 2.5, 1.0))
            show_cursor  = False

        # Desenha bboxes 3D
        if bbox_alpha > 0.05:
            for _, row in obj_rows.iterrows():
                alias = row["alias"]
                color = highlight.get(alias, COLOR_DIM)
                aabb_min = np.array([row["aabb_min_x"],
                                     row["aabb_min_y"],
                                     row["aabb_min_z"]])
                aabb_max = np.array([row["aabb_max_x"],
                                     row["aabb_max_y"],
                                     row["aabb_max_z"]])

                pts = project_bbox(aabb_min, aabb_max, pose, K,
                                   index.color_w, index.color_h, W, H)
                if pts is None:
                    continue

                lw = 4 if alias in highlight else 1
                c  = tuple(int(v * bbox_alpha +
                                COLOR_DIM[i] * (1 - bbox_alpha))
                            for i, v in enumerate(color[:3]))
                draw_bbox_3d(draw, pts, c, lw)

                if alias in highlight and bbox_alpha > 0.7:
                    draw_label_3d(draw, pts, alias,
                                  tuple(int(v * bbox_alpha) for v in color[:3]))

        # Overlay de texto
        img = overlay_text(img, operator, typed,
                           answer if ans_alpha > 0 else None,
                           answer_ok, ans_alpha, show_cursor)
        frames.append(img)

    return frames


# ---------------------------------------------------------------------------
# Transição
# ---------------------------------------------------------------------------

def make_transition(img_from: Image.Image,
                    img_to: Image.Image,
                    n: int) -> list[Image.Image]:
    a = np.array(img_from)
    b = np.array(img_to)
    result = []
    for i in range(n):
        t = ease(i / n)
        if t < 0.5:
            mid = crossfade(a, np.zeros_like(a), t * 2)
        else:
            mid = crossfade(np.zeros_like(b), b, (t - 0.5) * 2)
        result.append(Image.fromarray(mid.astype(np.uint8)))
    return result


# ---------------------------------------------------------------------------
# Slides de título / fim
# ---------------------------------------------------------------------------

def make_slide(line1: str, line2: str,
               n_frames: int, fade_in: bool = True) -> list[Image.Image]:
    slides = []
    for i in range(n_frames):
        t     = i / n_frames
        alpha = ease(min(t * 4, 1.0)) if fade_in else 1.0
        img   = Image.new("RGB", (W, H), BG)
        draw  = ImageDraw.Draw(img)
        # Linha decorativa
        lw = int(W * 0.5 * alpha)
        if lw > 1:
            draw.rectangle([(W - lw) // 2, H // 2 - 65,
                             (W + lw) // 2, H // 2 - 62],
                           fill=ACCENT)
        tw = draw.textlength(line1, font=font(42))
        draw.text(((W - tw) // 2, H // 2 - 52), line1,
                  fill=tuple(int(c * alpha) for c in WHITE), font=font(42))
        tw2 = draw.textlength(line2, font=font(20))
        draw.text(((W - tw2) // 2, H // 2 + 16), line2,
                  fill=tuple(int(c * alpha) for c in GRAY), font=font(20))
        slides.append(img)
    return slides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--out", type=str,
                        default=str(OUT_DIR / "spatial_real_demo.mp4"))
    args = parser.parse_args()

    print("Carregando dados ...")
    manifest = pd.read_csv(MANIFEST_CSV)
    alias_df = pd.read_csv(ALIAS_CSV)
    scene    = manifest[manifest["scene_id"] == SCENE_ID].copy()
    obj_df   = alias_df.merge(
        scene[["object_id", "aabb_min_x", "aabb_min_y", "aabb_min_z",
               "aabb_max_x", "aabb_max_y", "aabb_max_z"]],
        on="object_id"
    )

    # Calcula distância real para query 1
    d_real = surface_distance("chair1", "monitor1", alias_df)
    QUERIES[0]["answer"] = f"{d_real:.2f} m".replace(".", ",")
    print(f"  chair1 ↔ monitor1: {d_real:.3f} m")

    # Constrói índice do .sens
    print("Construindo índice .sens ...")
    index = SensIndex(SENS_PATH)
    K = index.intrinsic_color

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="real_demo_"))
    frame_idx = 0

    def save_frame(img: Image.Image) -> None:
        nonlocal frame_idx
        img.save(tmp / f"f{frame_idx:06d}.jpg", quality=92)
        frame_idx += 1

    # Título
    print("\nGerando título ...")
    for f in make_slide(
        "Raciocínio Espacial Métrico em Cenas 3D",
        "scene0142_00 · ScanNet · Motor Geométrico Explícito",
        int(args.fps * 2.5)
    ):
        save_frame(f)

    last_img = make_slide("", "", 1, fade_in=False)[0]

    # Queries
    for qi, q in enumerate(QUERIES):
        print(f"\nQuery {qi+1}/{len(QUERIES)}: {q['label']}")

        aliases = list(q["objects"].keys())
        q_rows  = obj_df[obj_df["alias"].isin(aliases)].copy()

        seg_frames = frames_for_query(
            index, K, q_rows,
            q["objects"],
            q["text"],
            q["answer"],
            q["ok"],
            q["label"],
        )

        # Transição do anterior
        if qi > 0:
            for f in make_transition(last_img, seg_frames[0], T_TRANS):
                save_frame(f)
        else:
            # Fade do slide de título para o primeiro frame
            blank = Image.new("RGB", (W, H), BG)
            for f in make_transition(blank, seg_frames[0], T_TRANS):
                save_frame(f)

        for f in seg_frames:
            save_frame(f)
            last_img = f

        print(f"  {len(seg_frames)} frames gerados")

    # Slide final
    print("\nGerando slide final ...")
    for f in make_transition(
        last_img,
        make_slide("PPGCOMP · UFBA · 2026",
                   "github.com/grifo114/metric-spatial-vlm",
                   1, fade_in=False)[0],
        T_TRANS
    ):
        save_frame(f)
    for f in make_slide(
        "PPGCOMP · UFBA · 2026",
        "github.com/grifo114/metric-spatial-vlm",
        int(args.fps * 3), fade_in=False
    ):
        save_frame(f)

    # Monta MP4
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = frame_idx / args.fps
    print(f"\nMontando MP4: {frame_idx} frames @ {args.fps}fps ({dur:.1f}s) ...")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(tmp / "f%06d.jpg"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "17",
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
        mb = out_path.stat().st_size / 1024 / 1024
        print(f"Vídeo salvo: {out_path}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()