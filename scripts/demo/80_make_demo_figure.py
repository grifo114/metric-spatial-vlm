from __future__ import annotations

from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "artifacts" / "demo_visual"
OUT_DIR = ROOT / "artifacts" / "demo_visual"


def normalize_text(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("distância", "distancia")
        .replace("próximo", "proximo")
        .replace("está", "esta")
        .replace("estão", "estao")
    )


def slugify_query(text: str, max_len: int = 80) -> str:
    import re
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len] if text else "query"


def add_panel_title(img: Image.Image, title: str, font) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, _ = img.size

    draw.rectangle([0, 0, w, 30], fill=(255, 255, 255, 220))
    draw.text((10, 8), title, fill=(0, 0, 0, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", default="scene0142_00")
    args = parser.parse_args()

    scene_id = args.scene_id

    queries = [
        ("(a) distance", 'qual a distancia entre cabinet2 e table4?'),
        ("(b) nearest", 'qual chair esta mais proximo de cabinet1?'),
        ("(c) between", 'chair5 esta entre desk1 e table4?'),
        ("(d) aligned", 'cabinet1, chair2 e table3 estao alinhados?'),
    ]

    font = ImageFont.load_default()
    panels = []

    for title, query in queries:
        filename = f"{scene_id}_{slugify_query(query)}.png"
        path = IMG_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {path}")

        img = Image.open(path).convert("RGBA")
        img = add_panel_title(img, title, font)
        panels.append(img)

    # normaliza tamanho
    widths = [img.width for img in panels]
    heights = [img.height for img in panels]
    target_w = min(widths)
    target_h = min(heights)

    resized = [img.resize((target_w, target_h)) for img in panels]

    gap = 20
    margin = 20
    canvas_w = 2 * target_w + gap + 2 * margin
    canvas_h = 2 * target_h + gap + 2 * margin

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    positions = [
        (margin, margin),
        (margin + target_w + gap, margin),
        (margin, margin + target_h + gap),
        (margin + target_w + gap, margin + target_h + gap),
    ]

    for img, pos in zip(resized, positions):
        canvas.paste(img, pos)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{scene_id}_demo_figure.png"
    canvas.save(out_path)

    print(f"Figura composta salva em: {out_path}")
    print("Painéis usados:")
    for title, query in queries:
        print(f" - {title}: {query}")


if __name__ == "__main__":
    main()