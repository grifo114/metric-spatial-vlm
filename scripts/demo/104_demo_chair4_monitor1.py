#!/usr/bin/env python3
"""
95_demo_chair4_monitor1.py

Vídeo de demonstração focado:
  - Frames reais 376–448 da scene0142_00 (chair4 + monitor1 visíveis)
  - Bboxes 3D projetadas com clipping robusto
  - Query DISTANCE animada: qual a distância entre chair4 e monitor1?
  - Sem slide de título nem final

Uso:
    python scripts/95_demo_chair4_monitor1.py
    python scripts/95_demo_chair4_monitor1.py --out meu_video.mp4
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
POINTS    = ROOT / "artifacts/object_points_test_official_stage1"
OUT_DIR   = ROOT / "artifacts/demo_video"

# Frames com chair4 + monitor1 visíveis (contíguos)
FRAME_IDS = list(range(376, 562))   # 186 frames ≈ 7.7s a 24fps

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
W, H   = 1280, 720
FPS    = 24

COLOR_CHAIR   = (220,  60,  60)   # vermelho — chair4
COLOR_MONITOR = ( 50, 120, 220)   # azul     — monitor1
PANEL_COLOR   = ( 8,   8,  18, 220)
ACCENT        = ( 0, 200, 140)
WHITE         = (240, 240, 240)
RED_C         = (220,  70,  70)

PANEL_H = 140
PANEL_Y = H - PANEL_H

# Fases (em frames)
N_TOTAL   = len(FRAME_IDS)          # 73 frames
T_INTRO   = 24   # câmera se movendo, sem bbox
T_BOX_IN  = 20   # bbox acende
T_TYPE    = 40   # digitação da query
T_HOLD    = 20    # hold
T_ANSWER  = N_TOTAL - T_INTRO - T_BOX_IN - T_TYPE - T_HOLD  # resto

QUERY_TEXT = "Qual a distância entre chair4 e monitor1?"


# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


F_OP  = None
F_Q   = None
F_ANS = None
F_LBL = None


def init_fonts():
    global F_OP, F_Q, F_ANS, F_LBL
    F_OP  = _font(20)
    F_Q   = _font(26)
    F_ANS = _font(34)
    F_LBL = _font(17)


# ---------------------------------------------------------------------------
# Projeção robusta (com clipping)
# ---------------------------------------------------------------------------

def project_corners(aabb_min: np.ndarray, aabb_max: np.ndarray,
                    W2C: np.ndarray, K: np.ndarray,
                    orig_w: int, orig_h: int) -> list[tuple[int, int]] | None:
    """
    Projeta os 8 cantos do AABB.
    Retorna None se qualquer canto estiver atrás da câmera OU se a
    projeção resultar em coordenadas absurdas (bbox voando).
    """
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

    pts_2d = []
    for c in corners:
        pc = W2C @ np.array([c[0], c[1], c[2], 1.0])
        if pc[2] < 0.05:          # atrás da câmera → descarta bbox inteira
            return None
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        # Escala para resolução de saída
        u_out = int(u * W / orig_w)
        v_out = int(v * H / orig_h)
        # Rejeita se coordenada absurda (projeção de objeto muito lateral)
        if not (-W < u_out < 2 * W and -H < v_out < 2 * H):
            return None
        pts_2d.append((u_out, v_out))

    return pts_2d


BBOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # base
    (4,5),(5,6),(6,7),(7,4),   # topo
    (0,4),(1,5),(2,6),(3,7),   # laterais
]


def draw_bbox(draw: ImageDraw.Draw,
              pts: list[tuple[int, int]],
              color: tuple, lw: int = 3) -> None:
    for i, j in BBOX_EDGES:
        x0, y0 = pts[i]
        x1, y1 = pts[j]
        draw.line([(x0, y0), (x1, y1)], fill=color, width=lw)


def draw_label(draw: ImageDraw.Draw,
               pts: list[tuple[int, int]],
               text: str, color: tuple) -> None:
    """Label acima do canto mais alto do bbox."""
    top  = min(pts, key=lambda p: p[1])
    tx   = max(4, min(W - 100, top[0] - 25))
    ty   = max(4, top[1] - 30)
    tw   = draw.textlength(text, font=F_LBL) + 10
    draw.rectangle([tx - 2, ty - 2, tx + tw, ty + 22],
                   fill=(10, 10, 20))
    draw.text((tx + 4, ty), text, fill=color, font=F_LBL)


# ---------------------------------------------------------------------------
# Painel de texto
# ---------------------------------------------------------------------------

def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def overlay_panel(img: Image.Image,
                  typed: str,
                  answer: str | None,
                  answer_alpha: float) -> Image.Image:
    img  = img.convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw  = ImageDraw.Draw(over)

    # Painel
    drw.rectangle([0, PANEL_Y, W, H], fill=PANEL_COLOR)
    drw.rectangle([0, PANEL_Y, W, PANEL_Y + 3], fill=(*ACCENT, 255))

    # Tag operador
    tag = " DISTANCE "
    tw  = drw.textlength(tag, font=F_OP) + 12
    drw.rectangle([20, PANEL_Y + 12, 20 + tw, PANEL_Y + 40],
                  fill=(*ACCENT, 255))
    drw.text((26, PANEL_Y + 13), tag, fill=(8, 8, 18), font=F_OP)

    # Query
    if typed:
        drw.text((20, PANEL_Y + 48), typed,
                 fill=(*WHITE, 255), font=F_Q)
        # Cursor
        if answer is None:
            cx = 20 + int(drw.textlength(typed, font=F_Q))
            drw.rectangle([cx + 2, PANEL_Y + 50,
                           cx + 12, PANEL_Y + 76],
                          fill=(*ACCENT, 255))

    # Resposta
    if answer and answer_alpha > 0:
        a_int = int(answer_alpha * 255)
        drw.text((20, PANEL_Y + 88),
                 f"→  {answer}",
                 fill=(*ACCENT, a_int), font=F_ANS)

    return Image.alpha_composite(img, over).convert("RGB")


# ---------------------------------------------------------------------------
# Leitura do .sens (streamer direto)
# ---------------------------------------------------------------------------

class SensStream:
    """Lê o .sens sequencialmente, acumulando offsets conforme avança."""

    def __init__(self, path: Path):
        self.path = path
        self._f   = open(path, "rb")
        self._read_header()
        self._build_offset_index()

    def _read_header(self):
        f = self._f
        f.read(4)                              # version
        slen = struct.unpack("Q", f.read(8))[0]
        f.read(slen)                           # sensor name
        raw = f.read(64)                       # intrinsic_color
        self.K = np.frombuffer(raw, np.float32).reshape(4, 4)
        f.read(64 * 3)                         # outras 3 matrizes
        f.read(8)                              # color/depth compression
        self.color_w = struct.unpack("I", f.read(4))[0]
        self.color_h = struct.unpack("I", f.read(4))[0]
        f.read(8)                              # depth w/h
        f.read(4)                              # depth_shift
        self.num_frames = struct.unpack("Q", f.read(8))[0]
        self._data_start = f.tell()

    def _build_offset_index(self):
        """Constrói índice de offsets sem carregar imagens."""
        print("  Construindo índice de offsets ...")
        f = self._f
        self.offsets     = []
        self.color_sizes = []
        self.depth_sizes = []
        for i in range(self.num_frames):
            self.offsets.append(f.tell())
            f.read(64 + 16)                    # pose + timestamps
            cs = struct.unpack("Q", f.read(8))[0]
            ds = struct.unpack("Q", f.read(8))[0]
            self.color_sizes.append(cs)
            self.depth_sizes.append(ds)
            f.seek(cs + ds, 1)
            if i % 500 == 0:
                print(f"    {i}/{self.num_frames} ...", end="\r")
        print(f"  Índice pronto: {self.num_frames} frames.")

    def read_frame(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (pose 4x4, color_arr HxWx3)."""
        self._f.seek(self.offsets[idx])
        pose = np.frombuffer(self._f.read(64), np.float32).reshape(4, 4)
        self._f.read(16)                       # timestamps
        cs   = struct.unpack("Q", self._f.read(8))[0]
        self._f.read(8)                        # depth_size
        data = self._f.read(cs)
        img  = Image.open(io.BytesIO(data)).convert("RGB")
        return pose, np.array(img)

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Distância de superfície
# ---------------------------------------------------------------------------

def surface_distance(alias_a: str, alias_b: str,
                     alias_df: pd.DataFrame) -> float:
    def load(a):
        row = alias_df[alias_df["alias"] == a].iloc[0]
        return np.load(ROOT / row["points_path"])["points"]
    pa, pb = load(alias_a), load(alias_b)
    tree   = cKDTree(pb)
    d, _   = tree.query(pa, k=1, workers=-1)
    return float(d.min())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--out", default=str(OUT_DIR / "demo_chair4_monitor1.mp4"))
    args = parser.parse_args()

    init_fonts()

    # Dados dos objetos
    alias_df = pd.read_csv(ALIAS_CSV)
    manifest = pd.read_csv(MANIFEST)
    scene    = manifest[manifest["scene_id"] == "scene0142_00"].copy()
    obj_df   = alias_df.merge(
        scene[["object_id", "aabb_min_x", "aabb_min_y", "aabb_min_z",
               "aabb_max_x", "aabb_max_y", "aabb_max_z"]],
        on="object_id"
    )

    def get_aabb(alias):
        r = obj_df[obj_df["alias"] == alias].iloc[0]
        return (np.array([r["aabb_min_x"], r["aabb_min_y"], r["aabb_min_z"]]),
                np.array([r["aabb_max_x"], r["aabb_max_y"], r["aabb_max_z"]]))

    aabb_chair   = get_aabb("chair4")
    aabb_monitor = get_aabb("monitor1")

    # Distância real
    d_real  = surface_distance("chair4", "monitor1", alias_df)
    answer  = f"{d_real:.2f} m".replace(".", ",")
    print(f"Distância chair4 ↔ monitor1: {d_real:.3f} m  → resposta: {answer}")

    # Índice do .sens
    stream = SensStream(SENS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="demo95_"))
    frame_idx = 0

    def save(img: Image.Image):
        nonlocal frame_idx
        img.save(tmp / f"f{frame_idx:06d}.jpg", quality=93)
        frame_idx += 1

    print(f"\nGerando {len(FRAME_IDS)} frames ...")

    for phase_i, fid in enumerate(FRAME_IDS):
        pose, color_arr = stream.read_frame(fid)
        W2C = np.linalg.inv(pose)

        # Redimensiona para (W, H)
        img  = Image.fromarray(color_arr).resize((W, H), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # --- Fase ---
        if phase_i < T_INTRO:
            bbox_alpha  = 0.0
            typed       = ""
            ans_alpha   = 0.0

        elif phase_i < T_INTRO + T_BOX_IN:
            t           = (phase_i - T_INTRO) / T_BOX_IN
            bbox_alpha  = ease(t)
            typed       = ""
            ans_alpha   = 0.0

        elif phase_i < T_INTRO + T_BOX_IN + T_TYPE:
            t           = (phase_i - T_INTRO - T_BOX_IN) / T_TYPE
            n_chars     = min(len(QUERY_TEXT), int(t * len(QUERY_TEXT)) + 1)
            bbox_alpha  = 1.0
            typed       = QUERY_TEXT[:n_chars]
            ans_alpha   = 0.0

        elif phase_i < T_INTRO + T_BOX_IN + T_TYPE + T_HOLD:
            bbox_alpha  = 1.0
            typed       = QUERY_TEXT
            ans_alpha   = 0.0

        else:
            t           = (phase_i - T_INTRO - T_BOX_IN - T_TYPE - T_HOLD) / max(T_ANSWER, 1)
            bbox_alpha  = 1.0
            typed       = QUERY_TEXT
            ans_alpha   = ease(min(t * 2.5, 1.0))

        # --- Bboxes ---
        if bbox_alpha > 0.02:
            for alias, color, aabb in [
                ("chair4",   COLOR_CHAIR,   aabb_chair),
                ("monitor1", COLOR_MONITOR, aabb_monitor),
            ]:
                pts = project_corners(aabb[0], aabb[1], W2C,
                                      stream.K, stream.color_w, stream.color_h)
                if pts is None:
                    continue

                # Aplica alpha na cor
                c = tuple(int(v * bbox_alpha) for v in color)
                draw_bbox(draw, pts, c, lw=4)

                if bbox_alpha > 0.6:
                    draw_label(draw, pts, alias, c)

        # --- Painel de texto ---
        img = overlay_panel(
            img,
            typed,
            answer if ans_alpha > 0 else None,
            ans_alpha,
        )

        save(img)

        if phase_i % 10 == 0:
            print(f"  {phase_i+1}/{len(FRAME_IDS)} frames", end="\r")

    stream.close()
    print(f"\n{frame_idx} frames gerados.")

    # Monta MP4
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Montando MP4 ({frame_idx/args.fps:.1f}s @ {args.fps}fps) ...")

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
        print("ERRO ffmpeg:\n" + result.stderr[-400:])
    else:
        mb = out_path.stat().st_size / 1024 / 1024
        print(f"Vídeo salvo: {out_path}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()