from __future__ import annotations

from pathlib import Path
from math import ceil
from PIL import Image, ImageDraw

frames_dir = Path("/Users/jeffersonlopes/metric-spatial-vlm/artifacts/scene0142_00_video_demo_extract/color")
out_path = frames_dir.parent / "contact_sheet.jpg"

image_paths = sorted(frames_dir.glob("*.jpg"))
if not image_paths:
    raise SystemExit("Nenhum frame JPG encontrado.")

thumb_w, thumb_h = 240, 135
cols = 4
rows = ceil(len(image_paths) / cols)
label_h = 24
sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
draw = ImageDraw.Draw(sheet)

for idx, img_path in enumerate(image_paths):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h))
    x = (idx % cols) * thumb_w
    y = (idx // cols) * (thumb_h + label_h)

    # centraliza thumbnail
    paste_x = x + (thumb_w - img.width) // 2
    paste_y = y + (thumb_h - img.height) // 2
    sheet.paste(img, (paste_x, paste_y))

    label = img_path.stem
    draw.text((x + 8, y + thumb_h + 4), label, fill="black")

sheet.save(out_path, quality=95)
print(f"Salvo: {out_path}")