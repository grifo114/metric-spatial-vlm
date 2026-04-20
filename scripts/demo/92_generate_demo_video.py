#!/usr/bin/env python3
"""
92_generate_demo_video.py

Gera um vídeo de demonstração do motor de raciocínio espacial sobre a
cena scene0142_00, com quatro queries animadas em sequência:
  - distance : qual a distância entre cabinet2 e table4?
  - nearest  : qual chair está mais próximo de cabinet1?
  - between  : chair5 está entre desk1 e table4?
  - aligned  : cabinet1, chair2 e table3 estão alinhados?

Pipeline:
  1. Gera frames PNG via PIL (texto tipografado + fades + zoom)
  2. Monta MP4 via ffmpeg (sem dependência extra além do ffmpeg)

Uso:
    python scripts/92_generate_demo_video.py
    python scripts/92_generate_demo_video.py --fps 30 --out demo.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
DEMO_DIR  = ROOT / "artifacts" / "demo_visual"
OUT_DIR   = ROOT / "artifacts" / "demo_video"

BASE_IMG  = DEMO_DIR / "scene0142_00_base_render.png"

QUERY_IMGS = {
    "distance": DEMO_DIR / "scene0142_00_qual_a_distancia_entre_cabinet2_e_table4.png",
    "nearest" : DEMO_DIR / "scene0142_00_qual_chair_esta_mais_proximo_de_cabinet1.png",
    "between" : DEMO_DIR / "scene0142_00_chair5_esta_entre_desk1_e_table4.png",
    "aligned" : DEMO_DIR / "scene0142_00_cabinet1_chair2_e_table3_estao_alinhados.png",
}

# ---------------------------------------------------------------------------
# Configuração do vídeo
# ---------------------------------------------------------------------------
W, H   = 1280, 720
FPS    = 24
BG     = (15, 15, 25)          # fundo escuro
ACCENT = (0, 180, 120)         # verde (paleta da dissertação)
WHITE  = (240, 240, 240)
GRAY   = (140, 140, 155)
RED    = (210, 60, 60)

QUERIES = [
    {
        "operator": "DISTANCE",
        "query":    "Qual a distância entre cabinet2 e table4?",
        "answer":   "0,161 m",
        "answer_color": (0, 200, 140),
        "img":      "distance",
    },
    {
        "operator": "NEAREST",
        "query":    "Qual chair está mais próximo de cabinet1?",
        "answer":   "chair4  (1,936 m)",
        "answer_color": (0, 200, 140),
        "img":      "nearest",
    },
    {
        "operator": "BETWEEN",
        "query":    "chair5 está entre desk1 e table4?",
        "answer":   "Não",
        "answer_color": RED,
        "img":      "between",
    },
    {
        "operator": "ALIGNED",
        "query":    "cabinet1, chair2 e table3 estão alinhados?",
        "answer":   "Sim",
        "answer_color": (0, 200, 140),
        "img":      "aligned",
    },
]

# Durações em frames
T_TITLE      = int(FPS * 2.5)   # slide de título
T_SCENE_IN   = int(FPS * 0.8)   # fade-in da cena base
T_TYPE_DELAY = int(FPS * 0.5)   # pausa antes de digitar
T_HOLD_SCENE = int(FPS * 0.8)   # cena base antes do resultado
T_RESULT_IN  = int(FPS * 0.6)   # fade para imagem resultado
T_ANSWER_IN  = int(FPS * 0.4)   # resposta aparece
T_HOLD_END   = int(FPS * 1.8)   # pausa final por query
T_TRANSITION = int(FPS * 0.4)   # transição entre queries
T_END        = int(FPS * 3.0)   # slide final


# ---------------------------------------------------------------------------
# Utilitários de fontes
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_TITLE  = None
FONT_OP     = None
FONT_QUERY  = None
FONT_ANSWER = None
FONT_CREDIT = None
FONT_SMALL  = None


def load_fonts() -> None:
    global FONT_TITLE, FONT_OP, FONT_QUERY, FONT_ANSWER, FONT_CREDIT, FONT_SMALL
    FONT_TITLE  = _font(52, bold=True)
    FONT_OP     = _font(22, bold=True)
    FONT_QUERY  = _font(28)
    FONT_ANSWER = _font(38, bold=True)
    FONT_CREDIT = _font(18)
    FONT_SMALL  = _font(16)


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def ease_in_out(t: float) -> float:
    """t ∈ [0, 1] → suavizado."""
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Operações de imagem
# ---------------------------------------------------------------------------

def blank(color: tuple = BG) -> Image.Image:
    return Image.new("RGB", (W, H), color)


def fit_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Redimensiona mantendo aspecto e recorta ao centro."""
    iw, ih = img.size
    scale  = max(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img    = img.resize((nw, nh), Image.LANCZOS)
    x0     = (nw - target_w) // 2
    y0     = (nh - target_h) // 2
    return img.crop((x0, y0, x0 + target_w, y0 + target_h))


def zoom_crop(img: Image.Image, cx: float, cy: float,
              zoom: float) -> Image.Image:
    """Zoom centrado em (cx, cy) relativo (0–1)."""
    iw, ih = img.size
    nw = int(iw / zoom)
    nh = int(ih / zoom)
    x0 = max(0, min(int(cx * iw - nw // 2), iw - nw))
    y0 = max(0, min(int(cy * ih - nh // 2), ih - nh))
    cropped = img.crop((x0, y0, x0 + nw, y0 + nh))
    return cropped.resize((iw, ih), Image.LANCZOS)


def crossfade(img_a: Image.Image, img_b: Image.Image,
              alpha: float) -> Image.Image:
    return Image.blend(img_a, img_b, alpha)


def vignette(img: Image.Image, strength: float = 0.45) -> Image.Image:
    """Aplica vinheta escura nas bordas."""
    vig = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(vig)
    for i in range(min(W, H) // 2):
        t     = i / (min(W, H) // 2)
        val   = int(255 * ease_in_out(t) * strength + 255 * (1 - strength))
        val   = min(255, val)
        color = val
        draw.rectangle([i, i, W - i, H - i], outline=color)
    return Image.composite(img, Image.new("RGB", (W, H), (0, 0, 0)),
                           vig)


# ---------------------------------------------------------------------------
# Painel de texto inferior
# ---------------------------------------------------------------------------

PANEL_H = 160
PANEL_Y = H - PANEL_H


def draw_panel(frame: Image.Image, operator: str,
               typed_query: str, answer: str | None,
               answer_color: tuple, answer_alpha: float = 1.0) -> Image.Image:
    """Desenha painel semitransparente na base do frame."""
    frame = frame.copy()

    # Fundo do painel
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw     = ImageDraw.Draw(overlay)
    drw.rectangle([0, PANEL_Y, W, H], fill=(10, 10, 20, 210))
    # Linha superior colorida
    drw.rectangle([0, PANEL_Y, W, PANEL_Y + 3], fill=(*ACCENT, 255))

    frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    draw  = ImageDraw.Draw(frame)

    # Tag do operador
    tag_w = draw.textlength(f" {operator} ", font=FONT_OP) + 16
    draw.rectangle([24, PANEL_Y + 14, 24 + tag_w, PANEL_Y + 44],
                   fill=ACCENT)
    draw.text((32, PANEL_Y + 15), f" {operator} ",
              fill=(10, 10, 20), font=FONT_OP)

    # Texto da query
    draw.text((24, PANEL_Y + 54), typed_query,
              fill=WHITE, font=FONT_QUERY)

    # Cursor piscante (só quando ainda digitando)
    if answer is None and len(typed_query) > 0:
        tx = 24 + draw.textlength(typed_query, font=FONT_QUERY)
        draw.rectangle([tx + 2, PANEL_Y + 56,
                        tx + 14, PANEL_Y + 84],
                       fill=ACCENT)

    # Resposta
    if answer and answer_alpha > 0:
        ans_label = "→  "
        draw.text((24, PANEL_Y + 110),
                  ans_label + answer,
                  fill=tuple(int(c * answer_alpha + 10 * (1 - answer_alpha))
                              for c in answer_color),
                  font=FONT_ANSWER)

    return frame


# ---------------------------------------------------------------------------
# Geradores de segmentos
# ---------------------------------------------------------------------------

def frames_title(frames: list) -> None:
    """Slide de título com fade-in/out."""
    subtitle = "Raciocínio Espacial Métrico em Cenas 3D"
    line2    = "scene0142_00 · ScanNet · Motor Geométrico Explícito"

    for i in range(T_TITLE):
        t     = i / T_TITLE
        alpha = ease_in_out(min(t * 4, 1.0)) * ease_in_out(
            max(1 - (t - 0.75) * 4, 0.0))

        f    = blank(BG)
        draw = ImageDraw.Draw(f)

        # Linha decorativa
        lw = int(W * 0.55 * alpha)
        if lw > 1:
                draw.rectangle([(W - lw) // 2, H // 2 - 72,
                                (W + lw) // 2, H // 2 - 69],
                            fill=ACCENT)

        # Título
        tw = draw.textlength(subtitle, font=FONT_TITLE)
        col = tuple(int(c * alpha) for c in WHITE)
        draw.text(((W - tw) // 2, H // 2 - 60), subtitle,
                  fill=col, font=FONT_TITLE)

        # Subtítulo
        tw2 = draw.textlength(line2, font=FONT_CREDIT)
        col2 = tuple(int(c * alpha) for c in GRAY)
        draw.text(((W - tw2) // 2, H // 2 + 20), line2,
                  fill=col2, font=FONT_CREDIT)

        frames.append(f)


def frames_query(frames: list, scene_base: Image.Image,
                 scene_result: Image.Image,
                 q: dict, zoom_cx: float = 0.5,
                 zoom_cy: float = 0.5) -> None:
    """Segmento completo de uma query."""
    query_text   = q["query"]
    answer_text  = q["answer"]
    operator     = q["operator"]
    answer_color = q["answer_color"]

    # 1. Fade-in da cena base
    for i in range(T_SCENE_IN):
        alpha = ease_in_out(i / T_SCENE_IN)
        f     = crossfade(blank(BG), scene_base, alpha)
        f     = draw_panel(f, operator, "", None, answer_color)
        frames.append(f)

    # 2. Pausa breve
    for _ in range(T_TYPE_DELAY):
        f = scene_base.copy()
        f = draw_panel(f, operator, "", None, answer_color)
        frames.append(f)

    # 3. Efeito de digitação
    chars_per_frame = max(1, len(query_text) // (FPS * 1))  # ~1 segundo
    n_type_frames   = len(query_text) * max(1, FPS // 20)
    for i in range(n_type_frames):
        n_chars = min(len(query_text),
                      int(i * len(query_text) / n_type_frames) + 1)
        f = scene_base.copy()
        f = draw_panel(f, operator, query_text[:n_chars],
                       None, answer_color)
        frames.append(f)

    # 4. Hold com query completa
    for _ in range(T_HOLD_SCENE):
        f = scene_base.copy()
        f = draw_panel(f, operator, query_text, None, answer_color)
        frames.append(f)

    # 5. Zoom + crossfade para imagem resultado
    for i in range(T_RESULT_IN):
        t       = ease_in_out(i / T_RESULT_IN)
        zoom_v  = lerp(1.0, 1.45, t)
        zoomed  = zoom_crop(scene_base, zoom_cx, zoom_cy, zoom_v)
        f       = crossfade(zoomed, scene_result, t)
        f       = draw_panel(f, operator, query_text, None, answer_color)
        frames.append(f)

    # 6. Resposta aparece com fade
    for i in range(T_ANSWER_IN):
        alpha = ease_in_out(i / T_ANSWER_IN)
        f     = scene_result.copy()
        f     = draw_panel(f, operator, query_text,
                           answer_text, answer_color, alpha)
        frames.append(f)

    # 7. Hold final
    for _ in range(T_HOLD_END):
        f = scene_result.copy()
        f = draw_panel(f, operator, query_text,
                       answer_text, answer_color, 1.0)
        frames.append(f)


def frames_transition(frames: list,
                      img_from: Image.Image,
                      img_to: Image.Image) -> None:
    """Fade-to-black entre queries."""
    for i in range(T_TRANSITION):
        t = ease_in_out(i / T_TRANSITION)
        if t < 0.5:
            f = crossfade(img_from, blank(BG), t * 2)
        else:
            f = crossfade(blank(BG), img_to, (t - 0.5) * 2)
        frames.append(f)


def frames_end(frames: list) -> None:
    """Slide final."""
    line1 = "Motor Geométrico Explícito para Cenas 3D"
    line2 = "PPGCOMP · UFBA · 2026"
    line3 = "github.com/grifo114/metric-spatial-vlm"

    for i in range(T_END):
        t     = i / T_END
        alpha = ease_in_out(min(t * 3, 1.0))

        f    = blank(BG)
        draw = ImageDraw.Draw(f)

        lw = int(W * 0.4 * alpha)
        if lw > 1:
            draw.rectangle([(W - lw) // 2, H // 2 - 55,
                         (W + lw) // 2, H // 2 - 52], fill=ACCENT)

        for text, font, y, col in [
            (line1, FONT_QUERY,  H // 2 - 40, WHITE),
            (line2, FONT_CREDIT, H // 2 + 10,  GRAY),
            (line3, FONT_SMALL,  H // 2 + 40,  GRAY),
        ]:
            tw  = draw.textlength(text, font=font)
            c   = tuple(int(v * alpha) for v in col)
            draw.text(((W - tw) // 2, y), text, fill=c, font=font)

        frames.append(f)


# ---------------------------------------------------------------------------
# Montagem com ffmpeg
# ---------------------------------------------------------------------------

def save_video(frames: list, out_path: Path, fps: int) -> None:
    """Salva lista de frames PIL como MP4 via ffmpeg."""
    tmp = Path(tempfile.mkdtemp(prefix="demo_frames_"))
    print(f"  Salvando {len(frames)} frames em {tmp} ...")

    for i, frame in enumerate(frames):
        frame.save(tmp / f"frame_{i:06d}.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(tmp / "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"  Montando MP4: {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  ERRO ffmpeg:")
        print(result.stderr[-800:])
    else:
        print(f"  Vídeo salvo: {out_path}")
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Tamanho: {size_mb:.1f} MB  |  "
              f"Duração estimada: {len(frames)/fps:.1f}s")

    shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# Pontos de zoom por query (cx, cy relativos à imagem)
# Ajuste fino para centralizar nos objetos relevantes de cada query
# ---------------------------------------------------------------------------

ZOOM_CENTERS = {
    "distance": (0.62, 0.45),   # cabinet2 e table4
    "nearest" : (0.40, 0.40),   # cabinet1 + chairs
    "between" : (0.45, 0.55),   # chair5, desk1, table4
    "aligned" : (0.50, 0.50),   # cabinet1, chair2, table3
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",  type=int, default=FPS)
    parser.add_argument("--out",  type=str,
                        default=str(OUT_DIR / "spatial_reasoning_demo.mp4"))
    parser.add_argument("--no-vignette", action="store_true")
    args = parser.parse_args()

    load_fonts()

    # Verifica imagens necessárias
    missing = [p for p in [BASE_IMG, *QUERY_IMGS.values()] if not p.exists()]
    if missing:
        print("ERRO: imagens não encontradas:")
        for p in missing:
            print(f"  {p}")
        print("Execute o script 80 (make_demo_figure) primeiro.")
        return

    print("Carregando imagens ...")
    base_raw = Image.open(BASE_IMG).convert("RGB")
    scene_base = fit_image(base_raw, W, H)
    if not args.no_vignette:
        scene_base = vignette(scene_base, strength=0.30)

    results: dict[str, Image.Image] = {}
    for key, path in QUERY_IMGS.items():
        img = Image.open(path).convert("RGB")
        img = fit_image(img, W, H)
        if not args.no_vignette:
            img = vignette(img, strength=0.20)
        results[key] = img

    print("Gerando frames ...")
    all_frames: list[Image.Image] = []

    # Título
    frames_title(all_frames)
    print(f"  Título: {T_TITLE} frames")

    # Transição título → primeira query
    frames_transition(all_frames, blank(BG), scene_base)

    # Queries
    for qi, q in enumerate(QUERIES):
        cx, cy = ZOOM_CENTERS[q["img"]]
        frames_query(
            all_frames,
            scene_base,
            results[q["img"]],
            q,
            zoom_cx=cx,
            zoom_cy=cy,
        )
        print(f"  Query '{q['operator']}': {len(all_frames)} frames acumulados")

        # Transição entre queries (exceto após a última)
        if qi < len(QUERIES) - 1:
            frames_transition(all_frames, results[q["img"]], scene_base)

    # Slide final
    frames_transition(all_frames, results[QUERIES[-1]["img"]], blank(BG))
    frames_end(all_frames)

    total_s = len(all_frames) / args.fps
    print(f"\nTotal: {len(all_frames)} frames  ({total_s:.1f}s @ {args.fps}fps)")

    save_video(all_frames, Path(args.out), args.fps)


if __name__ == "__main__":
    main()