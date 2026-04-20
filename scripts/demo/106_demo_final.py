#!/usr/bin/env python3
"""
97_demo_final.py  (v2)

Ajustes em relação à versão anterior:
  - Labels das bboxes em português (cadeira, monitor, armário...)
  - Queries em português natural sem IDs técnicos
  - Digitação mais lenta, pausa mais longa antes da resposta
  - Fade-to-black suave ao final

Uso:
    python scripts/97_demo_final.py
    python scripts/97_demo_final.py --seg-frames 140 --out demo.mp4
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
ROOT      = Path(__file__).resolve().parents[1]
SENS      = ROOT / "data/scannet/scans/scene0142_00/scene0142_00.sens"
ALIAS_CSV = ROOT / "benchmark/demo_alias_maps/scene0142_00_alias_map.csv"
MANIFEST  = ROOT / "benchmark/objects_manifest_test_official_stage1.csv"
OUT_DIR   = ROOT / "artifacts/demo_video"

W, H = 1280, 720
FPS  = 24

ACCENT  = (  0, 200, 140)
WHITE   = (240, 240, 240)
RED_C   = (220,  70,  70)
COLOR_A = (220,  55,  55)
COLOR_B = ( 50, 110, 220)
COLOR_C = (220, 170,  30)
COLOR_W = (  0, 210, 130)

PANEL_Y = H - 145

# --- Timing (frames) ---
T_INTRO  = 16   # cena sem bbox
T_BOX_IN = 18   # bbox acende devagar
T_TYPE   = 52   # digitação lenta
T_HOLD   = 24   # pausa após query completa
T_ANSWER = 36   # resposta visível
T_FADE   = 24   # crossfade entre segmentos
T_BLACK  = 36   # fade-to-black final

# --- Nomes amigáveis PT-BR ---
LABELS_PT = {
    "chair1":   "cadeira 1",
    "chair2":   "cadeira 2",
    "chair3":   "cadeira 3",
    "chair4":   "cadeira",
    "chair5":   "cadeira",
    "monitor1": "monitor",
    "monitor2": "monitor 2",
    "cabinet2": "armário",
    "cabinet3": "armário 3",
    "cabinet4": "armário 4",
    "desk1":    "mesa",
}

SEGMENTS = [
    {
        "label":   "DISTANCE",
        "text":    "Qual a distância entre a cadeira e o monitor?",
        "objects": {"chair4": COLOR_A, "monitor1": COLOR_B},
        "start":   376,
        "answer":  None, "ok": True, "winner": None,
    },
    {
        "label":   "NEAREST",
        "text":    "Qual cadeira está mais próxima do armário?",
        "objects": {"cabinet2": COLOR_A, "chair1": COLOR_B,
                    "chair2": COLOR_B, "chair3": COLOR_B},
        "start":   897,
        "answer":  None, "ok": True, "winner": None,
    },
    {
        "label":   "BETWEEN",
        "text":    "A cadeira está entre os dois armários?",
        "objects": {"chair5": COLOR_A, "cabinet3": COLOR_B, "cabinet4": COLOR_C},
        "start":   1569,
        "answer":  None, "ok": None, "winner": None,
    },
]

# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------
_fc: dict = {}

def fnt(s: int) -> ImageFont.FreeTypeFont:
    if s not in _fc:
        for p in ["/System/Library/Fonts/Helvetica.ttc",
                  "/Library/Fonts/Arial.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if Path(p).exists():
                try:
                    _fc[s] = ImageFont.truetype(p, s); break
                except Exception:
                    pass
        if s not in _fc:
            _fc[s] = ImageFont.load_default()
    return _fc[s]

# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------

def ease(t: float) -> float:
    t = max(0., min(1., t))
    return t * t * (3 - 2 * t)


def project_corners(amin, amax, W2C, K, cw, ch):
    corners = np.array([
        [amin[0],amin[1],amin[2]], [amax[0],amin[1],amin[2]],
        [amax[0],amax[1],amin[2]], [amin[0],amax[1],amin[2]],
        [amin[0],amin[1],amax[2]], [amax[0],amin[1],amax[2]],
        [amax[0],amax[1],amax[2]], [amin[0],amax[1],amax[2]],
    ])
    pts = []
    for c in corners:
        pc = W2C @ np.array([c[0],c[1],c[2],1.])
        if pc[2] < 0.05: return None
        u = int(K[0,0]*pc[0]/pc[2]+K[0,2]) * W // cw
        v = int(K[1,1]*pc[1]/pc[2]+K[1,2]) * H // ch
        if not (-W < u < 2*W and -H < v < 2*H): return None
        pts.append((u, v))
    return pts


EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]


def draw_bbox(draw, pts, color, lw=4):
    for i, j in EDGES:
        draw.line([pts[i], pts[j]], fill=color, width=lw)


def draw_label(draw, pts, text, color):
    top = min(pts, key=lambda p: p[1])
    tx  = max(4, min(W-130, top[0]-35))
    ty  = max(4, top[1]-34)
    tw  = draw.textlength(text, font=fnt(18)) + 14
    draw.rectangle([tx-3, ty-3, tx+tw, ty+24], fill=(10, 10, 20))
    draw.text((tx+5, ty), text, fill=color, font=fnt(18))


def surface_dist(pa, pb):
    t = cKDTree(pb)
    d, _ = t.query(pa, k=1, workers=-1)
    return float(d.min())


def load_pts(alias, alias_df):
    row = alias_df[alias_df["alias"] == alias].iloc[0]
    return np.load(ROOT / row["points_path"])["points"]


def cxy(alias, obj_df):
    r = obj_df[obj_df["alias"] == alias].iloc[0]
    return np.array([(r["aabb_min_x"]+r["aabb_max_x"])/2,
                     (r["aabb_min_y"]+r["aabb_max_y"])/2])


def check_between(ox, oa, ob, obj_df, tau=0.30):
    cx, ca, cb = cxy(ox, obj_df), cxy(oa, obj_df), cxy(ob, obj_df)
    seg = cb - ca; L = np.linalg.norm(seg)
    if L < 1e-6: return False
    t = np.dot(cx-ca, seg) / L**2
    if not 0 <= t <= 1: return False
    return np.linalg.norm(cx-(ca+t*seg)) <= tau*L

# ---------------------------------------------------------------------------
# Índice .sens
# ---------------------------------------------------------------------------

class SensIndex:
    def __init__(self, path):
        print("  Construindo índice .sens ...")
        with open(path, "rb") as f:
            f.read(4)
            slen = struct.unpack("Q", f.read(8))[0]; f.read(slen)
            self.K = np.frombuffer(f.read(64), np.float32).reshape(4, 4)
            f.read(64*3); f.read(8)
            self.cw = struct.unpack("I", f.read(4))[0]
            self.ch = struct.unpack("I", f.read(4))[0]
            f.read(8); f.read(4)
            nf = struct.unpack("Q", f.read(8))[0]
            self.offsets = []; self.csizes = []; self.dsizes = []
            for i in range(nf):
                self.offsets.append(f.tell())
                f.read(80)
                cs = struct.unpack("Q", f.read(8))[0]
                ds = struct.unpack("Q", f.read(8))[0]
                self.csizes.append(cs); self.dsizes.append(ds)
                f.seek(cs+ds, 1)
                if i % 500 == 0:
                    print(f"    {i}/{nf}...", end="\r")
        self.path = path
        print(f"  Pronto: {nf} frames.")

    def read(self, idx):
        idx = min(idx, len(self.offsets)-1)
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            pose = np.frombuffer(f.read(64), np.float32).reshape(4, 4)
            f.read(16)
            cs = struct.unpack("Q", f.read(8))[0]; f.read(8)
            data = f.read(cs)
        return pose, np.array(Image.open(io.BytesIO(data)).convert("RGB"))

# ---------------------------------------------------------------------------
# Painel de texto
# ---------------------------------------------------------------------------

def make_panel(img, operator, typed, answer, ok, aa):
    img = img.convert("RGBA")
    ov  = Image.new("RGBA", (W, H), (0,0,0,0))
    d   = ImageDraw.Draw(ov)

    d.rectangle([0, PANEL_Y, W, H], fill=(8, 8, 18, 220))
    d.rectangle([0, PANEL_Y, W, PANEL_Y+3], fill=(*ACCENT, 255))

    tag = f" {operator} "
    tw  = d.textlength(tag, font=fnt(20)) + 12
    d.rectangle([20, PANEL_Y+12, 20+tw, PANEL_Y+40], fill=(*ACCENT, 255))
    d.text((26, PANEL_Y+13), tag, fill=(8, 8, 18), font=fnt(20))

    if typed:
        d.text((20, PANEL_Y+50), typed, fill=(*WHITE, 255), font=fnt(27))
        if aa == 0.:
            cx2 = 20 + int(d.textlength(typed, font=fnt(27)))
            d.rectangle([cx2+2, PANEL_Y+52, cx2+13, PANEL_Y+78],
                        fill=(*ACCENT, 255))

    if answer and aa > 0:
        ai  = int(aa * 255)
        col = (*ACCENT, ai) if ok else (*RED_C, ai)
        d.text((20, PANEL_Y+90), f"→  {answer}", fill=col, font=fnt(35))

    return Image.alpha_composite(img, ov).convert("RGB")

# ---------------------------------------------------------------------------
# Gera frames de um segmento
# ---------------------------------------------------------------------------

def gen_segment(index, obj_df, seg, n_frames):
    fids     = list(range(seg["start"], seg["start"] + n_frames))
    operator = seg["label"]
    qtxt     = seg["text"]
    answer   = seg["answer"]
    ok       = seg["ok"] if seg["ok"] is not None else True
    hl       = dict(seg["objects"])
    if seg["winner"] and seg["winner"] in hl:
        hl[seg["winner"]] = COLOR_W

    aabbs = {}
    for alias in hl:
        r = obj_df[obj_df["alias"] == alias].iloc[0]
        aabbs[alias] = (
            np.array([r["aabb_min_x"], r["aabb_min_y"], r["aabb_min_z"]]),
            np.array([r["aabb_max_x"], r["aabb_max_y"], r["aabb_max_z"]]),
        )

    # Marcos de fase fixos
    p0 = T_INTRO
    p1 = p0 + T_BOX_IN
    p2 = p1 + T_TYPE
    p3 = p2 + T_HOLD
    p4 = p3 + T_ANSWER
    # frames restantes vão para hold da resposta
    extra = max(0, n_frames - p4)

    frames = []
    for pi, fid in enumerate(fids):
        pose, carr = index.read(fid)
        W2C = np.linalg.inv(pose)
        img  = Image.fromarray(carr).resize((W, H), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Fase
        if pi < p0:
            ba, typed, aa = 0., "", 0.
        elif pi < p1:
            t = (pi - p0) / T_BOX_IN
            ba, typed, aa = ease(t), "", 0.
        elif pi < p2:
            t = (pi - p1) / T_TYPE
            n = min(len(qtxt), int(t * len(qtxt)) + 1)
            ba, typed, aa = 1., qtxt[:n], 0.
        elif pi < p3:
            ba, typed, aa = 1., qtxt, 0.
        else:
            t_tot = T_ANSWER + extra
            t = (pi - p3) / max(t_tot, 1)
            ba, typed, aa = 1., qtxt, ease(min(t * 2.0, 1.))

        # Bboxes
        if ba > 0.02:
            for alias, color in hl.items():
                if alias not in aabbs: continue
                pts = project_corners(aabbs[alias][0], aabbs[alias][1],
                                      W2C, index.K, index.cw, index.ch)
                if pts is None: continue
                c = tuple(int(v * ba) for v in color[:3])
                draw_bbox(draw, pts, c, 4)
                if ba > 0.55:
                    label_text = LABELS_PT.get(alias, alias)
                    draw_label(draw, pts, label_text, c)

        img = make_panel(img, operator, typed,
                         answer if aa > 0 else None, ok, aa)
        frames.append(img)

    return frames

# ---------------------------------------------------------------------------
# Transições
# ---------------------------------------------------------------------------

def crossfade_segs(fa, fb, n):
    a, b = np.array(fa[-1]), np.array(fb[0])
    out  = []
    for i in range(n):
        t   = ease(i / n)
        arr = (a * (1-t) + b * t).astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out


def fade_to_black(last_img, n):
    a   = np.array(last_img)
    blk = np.zeros_like(a)
    out = []
    for i in range(n):
        t   = ease(i / n)
        arr = (a * (1-t) + blk * t).astype(np.uint8)
        out.append(Image.fromarray(arr))
    # Alguns frames de preto puro
    for _ in range(12):
        out.append(Image.fromarray(blk.astype(np.uint8)))
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-frames", type=int, default=120,
                        help="Frames reais por segmento (default=120 ≈ 5s)")
    parser.add_argument("--fps",  type=int, default=FPS)
    parser.add_argument("--out",  default=str(OUT_DIR/"demo_final_v2.mp4"))
    args = parser.parse_args()

    alias_df = pd.read_csv(ALIAS_CSV)
    manifest = pd.read_csv(MANIFEST)
    scene    = manifest[manifest["scene_id"]=="scene0142_00"].copy()
    obj_df   = alias_df.merge(
        scene[["object_id","aabb_min_x","aabb_min_y","aabb_min_z",
               "aabb_max_x","aabb_max_y","aabb_max_z"]], on="object_id")

    print("Calculando respostas geométricas ...")

    # 1. DISTANCE
    d = surface_dist(load_pts("chair4", alias_df),
                     load_pts("monitor1", alias_df))
    SEGMENTS[0]["answer"] = f"{d:.2f} m".replace(".", ",")
    print(f"  cadeira ↔ monitor = {d:.3f} m")

    # 2. NEAREST — resposta em português
    pts_cab = load_pts("cabinet2", alias_df)
    chairs  = ["chair1","chair2","chair3"]
    dists   = {c: surface_dist(load_pts(c, alias_df), pts_cab) for c in chairs}
    winner  = min(dists, key=dists.get)
    label_w = LABELS_PT.get(winner, winner)
    SEGMENTS[1]["answer"] = f"{label_w}  ({dists[winner]:.2f} m)".replace(".", ",")
    SEGMENTS[1]["winner"] = winner
    print(f"  mais próxima do armário = {label_w} ({dists[winner]:.3f} m)")

    # 3. BETWEEN
    r = check_between("chair5","cabinet3","cabinet4", obj_df)
    SEGMENTS[2]["answer"] = "Sim" if r else "Não"
    SEGMENTS[2]["ok"]     = r
    print(f"  cadeira entre armários = {r}")

    index = SensIndex(SENS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="demo97_"))
    fi  = 0

    def save(img):
        nonlocal fi
        img.save(tmp / f"f{fi:06d}.jpg", quality=93)
        fi += 1

    all_segs = []
    for si, seg in enumerate(SEGMENTS):
        print(f"\nSegmento {si+1}: {seg['label']} "
              f"(frames {seg['start']}–{seg['start']+args.seg_frames-1})")
        frames = gen_segment(index, obj_df, seg, args.seg_frames)
        all_segs.append(frames)
        print(f"  {len(frames)} frames | resposta: {seg['answer']}")

    # Salva com crossfade entre segmentos
    for si, sf in enumerate(all_segs):
        for f in sf:
            save(f)
        if si < len(all_segs) - 1:
            for f in crossfade_segs(sf, all_segs[si+1], T_FADE):
                save(f)

    # Fade-to-black final
    print("\nAplicando fade-to-black ...")
    for f in fade_to_black(all_segs[-1][-1], T_BLACK):
        save(f)

    total_s = fi / args.fps
    print(f"\nTotal: {fi} frames  ({total_s:.1f}s @ {args.fps}fps)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y",
           "-framerate", str(args.fps),
           "-i", str(tmp / "f%06d.jpg"),
           "-c:v", "libx264", "-preset", "slow",
           "-crf", "17", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart",
           str(out)]
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp)

    if r2.returncode != 0:
        print("ERRO ffmpeg:\n" + r2.stderr[-300:])
    else:
        print(f"Vídeo: {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()