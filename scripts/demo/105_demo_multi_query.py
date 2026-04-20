#!/usr/bin/env python3
"""
96_demo_multi_query.py

Vídeo de demonstração com 4 queries em partes diferentes da sala,
usando frames reais do .sens da scene0142_00.

Segmentos:
  1. DISTANCE  chair4 ↔ monitor1          @ frames 1113–1193
  2. NEAREST   qual chair mais próxima de cabinet2?  @ frames 897–977
  3. BETWEEN   chair5 entre cabinet3 e cabinet4?     @ frames 1569–1649
  4. ALIGNED   chair3, desk1 e monitor1 alinhados?   @ frames 1113–1193

Uso:
    python scripts/96_demo_multi_query.py
    python scripts/96_demo_multi_query.py --fps 24 --out demo.mp4
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
SENS      = ROOT / "data/scannet/scans/scene0142_00/scene0142_00.sens"
ALIAS_CSV = ROOT / "benchmark/demo_alias_maps/scene0142_00_alias_map.csv"
MANIFEST  = ROOT / "benchmark/objects_manifest_test_official_stage1.csv"
OUT_DIR   = ROOT / "artifacts/demo_video"

# ---------------------------------------------------------------------------
# Configuração de vídeo
# ---------------------------------------------------------------------------
W, H = 1280, 720
FPS  = 24

BG      = ( 15,  15,  25)
WHITE   = (240, 240, 240)
GRAY    = (150, 150, 165)
ACCENT  = (  0, 200, 140)
RED_C   = (220,  70,  70)
DIM_C   = ( 80,  80,  90)

COLOR_A   = (220,  55,  55)   # referência / objeto A
COLOR_B   = ( 50, 110, 220)   # objeto B / candidatos
COLOR_C   = (220, 170,  30)   # objeto C (terceiro)
COLOR_WIN = (  0, 210, 130)   # vencedor

PANEL_H = 145
PANEL_Y = H - PANEL_H

# Frames por fase em cada segmento
T_INTRO  = 14   # cena sem bbox (câmera se movendo)
T_BOX_IN = 12   # bbox acende
T_TYPE   = 22   # digitação
T_HOLD   = 10   # pausa
# T_ANSWER = restante dos frames do segmento

SEG_FRAMES = 80   # frames reais por segmento
T_FADE     = 18   # frames de crossfade entre segmentos

# ---------------------------------------------------------------------------
# Segmentos do vídeo
# ---------------------------------------------------------------------------
# Preenchidos depois de carregar dados (respostas calculadas em runtime)
SEGMENTS = [
    {
        "label":    "DISTANCE",
        "text":     "Qual a distância entre chair4 e monitor1?",
        "answer":   None,       # calculado em runtime
        "ok":       True,
        "objects":  {"chair4": COLOR_A, "monitor1": COLOR_B},
        "start":    1113,
    },
    {
        "label":    "NEAREST",
        "text":     "Qual chair está mais próxima de cabinet2?",
        "answer":   None,       # calculado em runtime
        "ok":       True,
        "objects":  {
            "cabinet2": COLOR_A,
            "chair1":   COLOR_B,
            "chair2":   COLOR_B,
            "chair3":   COLOR_B,
        },
        "start":    897,
        "winner":   None,       # alias do vencedor (calculado)
    },
    {
        "label":    "BETWEEN",
        "text":     "chair5 está entre cabinet3 e cabinet4?",
        "answer":   None,       # calculado em runtime
        "ok":       None,       # calculado em runtime
        "objects":  {
            "chair5":   COLOR_A,
            "cabinet3": COLOR_B,
            "cabinet4": COLOR_C,
        },
        "start":    1569,
    },
    {
        "label":    "ALIGNED",
        "text":     "chair3, desk1 e monitor1 estão alinhados?",
        "answer":   None,       # calculado em runtime
        "ok":       None,
        "objects":  {
            "chair3":  COLOR_A,
            "desk1":   COLOR_B,
            "monitor1": COLOR_C,
        },
        "start":    1113,
    },
]

# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------
_fonts: dict = {}

def fnt(size: int) -> ImageFont.FreeTypeFont:
    if size not in _fonts:
        for p in ["/System/Library/Fonts/Helvetica.ttc",
                  "/Library/Fonts/Arial.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if Path(p).exists():
                try:
                    _fonts[size] = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if size not in _fonts:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------

def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def project_corners(
    aabb_min: np.ndarray, aabb_max: np.ndarray,
    W2C: np.ndarray, K: np.ndarray,
    orig_w: int, orig_h: int,
) -> list[tuple[int, int]] | None:
    """Projeta 8 cantos do AABB. Retorna None se qualquer canto estiver atrás da câmera."""
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
    pts = []
    for c in corners:
        pc = W2C @ np.array([c[0], c[1], c[2], 1.0])
        if pc[2] < 0.05:
            return None
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        u_out = int(u * W / orig_w)
        v_out = int(v * H / orig_h)
        if not (-W < u_out < 2 * W and -H < v_out < 2 * H):
            return None
        pts.append((u_out, v_out))
    return pts


EDGES = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]


def draw_bbox(draw: ImageDraw.Draw, pts: list, color: tuple, lw: int = 3):
    for i, j in EDGES:
        draw.line([pts[i], pts[j]], fill=color, width=lw)


def draw_label(draw: ImageDraw.Draw, pts: list, text: str, color: tuple):
    top  = min(pts, key=lambda p: p[1])
    tx   = max(4, min(W - 110, top[0] - 30))
    ty   = max(4, top[1] - 30)
    tw   = draw.textlength(text, font=fnt(17)) + 10
    draw.rectangle([tx - 2, ty - 2, tx + tw, ty + 22], fill=(10, 10, 20))
    draw.text((tx + 4, ty), text, fill=color, font=fnt(17))


# ---------------------------------------------------------------------------
# Operadores geométricos
# ---------------------------------------------------------------------------

def surface_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    tree = cKDTree(pts_b)
    d, _ = tree.query(pts_a, k=1, workers=-1)
    return float(d.min())


def load_points(alias: str, alias_df: pd.DataFrame) -> np.ndarray:
    row = alias_df[alias_df["alias"] == alias].iloc[0]
    return np.load(ROOT / row["points_path"])["points"]


def centroid_xy(alias: str, obj_df: pd.DataFrame) -> np.ndarray:
    r = obj_df[obj_df["alias"] == alias].iloc[0]
    return np.array([(r["aabb_min_x"] + r["aabb_max_x"]) / 2,
                     (r["aabb_min_y"] + r["aabb_max_y"]) / 2])


def check_between(ox: str, oa: str, ob: str,
                  obj_df: pd.DataFrame, tau: float = 0.30) -> bool:
    """Critério between no plano XY."""
    cx = centroid_xy(ox, obj_df)
    ca = centroid_xy(oa, obj_df)
    cb = centroid_xy(ob, obj_df)
    seg = cb - ca
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6:
        return False
    t = np.dot(cx - ca, seg) / (seg_len ** 2)
    if not (0 <= t <= 1):
        return False
    proj  = ca + t * seg
    dist  = np.linalg.norm(cx - proj)
    return dist <= tau * seg_len


def check_aligned(oa: str, ob: str, oc: str,
                  obj_df: pd.DataFrame, tau: float = 0.25) -> bool:
    """Critério aligned no plano XY."""
    ca = centroid_xy(oa, obj_df)
    cb = centroid_xy(ob, obj_df)
    cc = centroid_xy(oc, obj_df)
    seg = cc - ca
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6:
        return False
    t    = np.dot(cb - ca, seg) / (seg_len ** 2)
    proj = ca + t * seg
    dist = np.linalg.norm(cb - proj)
    return dist <= tau * seg_len


# ---------------------------------------------------------------------------
# Leitura .sens (índice de offsets)
# ---------------------------------------------------------------------------

class SensIndex:
    def __init__(self, path: Path):
        self.path = path
        self._build()

    def _build(self):
        print("  Construindo índice .sens ...")
        with open(self.path, "rb") as f:
            f.read(4)
            slen = struct.unpack("Q", f.read(8))[0]; f.read(slen)
            raw  = f.read(64)
            self.K = np.frombuffer(raw, np.float32).reshape(4, 4)
            f.read(64 * 3); f.read(8)
            self.color_w = struct.unpack("I", f.read(4))[0]
            self.color_h = struct.unpack("I", f.read(4))[0]
            f.read(8); f.read(4)
            self.num_frames = struct.unpack("Q", f.read(8))[0]

            self.offsets     = []
            self.color_sizes = []
            self.depth_sizes = []
            for i in range(self.num_frames):
                self.offsets.append(f.tell())
                f.read(64 + 16)
                cs = struct.unpack("Q", f.read(8))[0]
                ds = struct.unpack("Q", f.read(8))[0]
                self.color_sizes.append(cs)
                self.depth_sizes.append(ds)
                f.seek(cs + ds, 1)
                if i % 500 == 0:
                    print(f"    {i}/{self.num_frames} ...", end="\r")
        print(f"  Índice pronto: {self.num_frames} frames.")

    def read_frame(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            pose = np.frombuffer(f.read(64), np.float32).reshape(4, 4)
            f.read(16)
            cs   = struct.unpack("Q", f.read(8))[0]
            f.read(8)
            data = f.read(cs)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return pose, np.array(img)


# ---------------------------------------------------------------------------
# Painel de texto
# ---------------------------------------------------------------------------

def overlay_panel(img: Image.Image, operator: str, typed: str,
                  answer: str | None, ok: bool,
                  ans_alpha: float) -> Image.Image:
    img  = img.convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d    = ImageDraw.Draw(over)

    d.rectangle([0, PANEL_Y, W, H], fill=(8, 8, 18, 215))
    d.rectangle([0, PANEL_Y, W, PANEL_Y + 3], fill=(*ACCENT, 255))

    tag = f" {operator} "
    tw  = d.textlength(tag, font=fnt(20)) + 12
    d.rectangle([20, PANEL_Y + 12, 20 + tw, PANEL_Y + 40],
                fill=(*ACCENT, 255))
    d.text((26, PANEL_Y + 13), tag, fill=(8, 8, 18), font=fnt(20))

    if typed:
        d.text((20, PANEL_Y + 48), typed, fill=(*WHITE, 255), font=fnt(26))
        if answer is None:
            cx = 20 + int(d.textlength(typed, font=fnt(26)))
            d.rectangle([cx + 2, PANEL_Y + 50, cx + 12, PANEL_Y + 76],
                        fill=(*ACCENT, 255))

    if answer and ans_alpha > 0:
        a_int = int(ans_alpha * 255)
        col   = (*ACCENT, a_int) if ok else (*RED_C, a_int)
        d.text((20, PANEL_Y + 88), f"→  {answer}", fill=col, font=fnt(34))

    return Image.alpha_composite(img, over).convert("RGB")


# ---------------------------------------------------------------------------
# Gera frames de um segmento
# ---------------------------------------------------------------------------

def gen_segment(
    index: SensIndex,
    obj_df: pd.DataFrame,
    seg: dict,
    n_frames: int,
) -> list[Image.Image]:
    frame_ids = list(range(seg["start"], seg["start"] + n_frames))
    operator  = seg["label"]
    query_txt = seg["text"]
    answer    = seg["answer"]
    ok        = seg["ok"]
    highlight = seg["objects"]
    winner    = seg.get("winner")

    # Atualiza cor do vencedor para NEAREST
    if winner and winner in highlight:
        highlight = dict(highlight)
        highlight[winner] = COLOR_WIN

    # Prepara aabb de cada objeto
    aabbs = {}
    for alias in highlight:
        r = obj_df[obj_df["alias"] == alias].iloc[0]
        aabbs[alias] = (
            np.array([r["aabb_min_x"], r["aabb_min_y"], r["aabb_min_z"]]),
            np.array([r["aabb_max_x"], r["aabb_max_y"], r["aabb_max_z"]]),
        )

    t_answer = max(4, n_frames - T_INTRO - T_BOX_IN - T_TYPE - T_HOLD)
    frames   = []

    for phase_i, fid in enumerate(frame_ids):
        if fid >= index.num_frames:
            fid = index.num_frames - 1
        pose, color_arr = index.read_frame(fid)
        W2C = np.linalg.inv(pose)

        img  = Image.fromarray(color_arr).resize((W, H), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Fase
        if phase_i < T_INTRO:
            bbox_a = 0.0; typed = ""; ans_a = 0.0
        elif phase_i < T_INTRO + T_BOX_IN:
            t = (phase_i - T_INTRO) / T_BOX_IN
            bbox_a = ease(t); typed = ""; ans_a = 0.0
        elif phase_i < T_INTRO + T_BOX_IN + T_TYPE:
            t = (phase_i - T_INTRO - T_BOX_IN) / T_TYPE
            n = min(len(query_txt), int(t * len(query_txt)) + 1)
            bbox_a = 1.0; typed = query_txt[:n]; ans_a = 0.0
        elif phase_i < T_INTRO + T_BOX_IN + T_TYPE + T_HOLD:
            bbox_a = 1.0; typed = query_txt; ans_a = 0.0
        else:
            t = (phase_i - T_INTRO - T_BOX_IN - T_TYPE - T_HOLD) / max(t_answer, 1)
            bbox_a = 1.0; typed = query_txt
            ans_a  = ease(min(t * 2.5, 1.0))

        # Bboxes
        if bbox_a > 0.02:
            for alias, color in highlight.items():
                if alias not in aabbs:
                    continue
                pts = project_corners(
                    aabbs[alias][0], aabbs[alias][1],
                    W2C, index.K, index.color_w, index.color_h
                )
                if pts is None:
                    continue
                lw = 4 if alias in highlight else 1
                c  = tuple(int(v * bbox_a) for v in color[:3])
                draw_bbox(draw, pts, c, lw)
                if bbox_a > 0.6:
                    draw_label(draw, pts, alias, c)

        img = overlay_panel(img, operator, typed,
                            answer if ans_a > 0 else None,
                            ok if ok is not None else True,
                            ans_a)
        frames.append(img)

    return frames


# ---------------------------------------------------------------------------
# Crossfade entre segmentos
# ---------------------------------------------------------------------------

def crossfade(imgs_a: list, imgs_b: list, n: int) -> list[Image.Image]:
    last  = np.array(imgs_a[-1])
    first = np.array(imgs_b[0])
    result = []
    for i in range(n):
        t = ease(i / n)
        if t < 0.5:
            arr = (last * (1 - t * 2) + np.zeros_like(last) * t * 2)
        else:
            arr = (np.zeros_like(first) * (1 - (t - 0.5) * 2) +
                   first * (t - 0.5) * 2)
        result.append(Image.fromarray(arr.astype(np.uint8)))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seg-frames", type=int, default=SEG_FRAMES,
                        help="Frames reais por segmento (default=80 ≈ 3.3s)")
    parser.add_argument("--out", default=str(OUT_DIR / "demo_multi_query.mp4"))
    args = parser.parse_args()

    # --- Dados ---
    alias_df = pd.read_csv(ALIAS_CSV)
    manifest = pd.read_csv(MANIFEST)
    scene    = manifest[manifest["scene_id"] == "scene0142_00"].copy()
    obj_df   = alias_df.merge(
        scene[["object_id", "aabb_min_x", "aabb_min_y", "aabb_min_z",
               "aabb_max_x", "aabb_max_y", "aabb_max_z"]],
        on="object_id"
    )

    # --- Calcula respostas geométricas ---
    print("Calculando respostas geométricas ...")

    # 1. DISTANCE: chair4 ↔ monitor1
    pts_chair4   = load_points("chair4",   alias_df)
    pts_monitor1 = load_points("monitor1", alias_df)
    d = surface_distance(pts_chair4, pts_monitor1)
    SEGMENTS[0]["answer"] = f"{d:.2f} m".replace(".", ",")
    SEGMENTS[0]["ok"]     = True
    print(f"  chair4 ↔ monitor1: {d:.3f} m")

    # 2. NEAREST: qual chair mais próxima de cabinet2?
    pts_cab2   = load_points("cabinet2", alias_df)
    chairs     = ["chair1", "chair2", "chair3"]
    dists      = {c: surface_distance(load_points(c, alias_df), pts_cab2)
                  for c in chairs}
    winner     = min(dists, key=dists.get)
    SEGMENTS[1]["answer"]  = f"{winner}  ({dists[winner]:.2f} m)".replace(".", ",")
    SEGMENTS[1]["ok"]      = True
    SEGMENTS[1]["winner"]  = winner
    print(f"  nearest to cabinet2: {winner} ({dists[winner]:.3f} m)")
    print(f"  dists: {dists}")

    # 3. BETWEEN: chair5 entre cabinet3 e cabinet4?
    result_between = check_between("chair5", "cabinet3", "cabinet4", obj_df)
    SEGMENTS[2]["answer"] = "Sim" if result_between else "Não"
    SEGMENTS[2]["ok"]     = result_between
    print(f"  chair5 between cabinet3/cabinet4: {result_between}")

    # 4. ALIGNED: chair3, desk1, monitor1 alinhados?
    result_aligned = check_aligned("chair3", "desk1", "monitor1", obj_df)
    SEGMENTS[3]["answer"] = "Sim" if result_aligned else "Não"
    SEGMENTS[3]["ok"]     = result_aligned
    print(f"  chair3/desk1/monitor1 aligned: {result_aligned}")

    # --- Índice .sens ---
    index = SensIndex(SENS)

    # --- Gera frames ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="demo96_"))
    frame_idx = 0

    def save(img: Image.Image):
        nonlocal frame_idx
        img.save(tmp / f"f{frame_idx:06d}.jpg", quality=93)
        frame_idx += 1

    all_segs = []
    for si, seg in enumerate(SEGMENTS):
        print(f"\nSegmento {si+1}/{len(SEGMENTS)}: {seg['label']} "
              f"(frames {seg['start']}–{seg['start']+args.seg_frames-1})")
        frames = gen_segment(index, obj_df, seg, args.seg_frames)
        all_segs.append(frames)
        print(f"  {len(frames)} frames gerados. Resposta: {seg['answer']}")

    # Salva com transições
    print("\nSalvando frames com transições ...")
    for si, seg_frames in enumerate(all_segs):
        for f in seg_frames:
            save(f)
        if si < len(all_segs) - 1:
            for f in crossfade(seg_frames, all_segs[si + 1], T_FADE):
                save(f)

    total_s = frame_idx / args.fps
    print(f"\nTotal: {frame_idx} frames  ({total_s:.1f}s @ {args.fps}fps)")

    # --- ffmpeg ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(tmp / "f%06d.jpg"),
        "-c:v", "libx264", "-preset", "slow",
        "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp)

    if result.returncode != 0:
        print("ERRO ffmpeg:\n" + result.stderr[-400:])
    else:
        mb = out_path.stat().st_size / 1024 / 1024
        print(f"Vídeo salvo: {out_path}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()