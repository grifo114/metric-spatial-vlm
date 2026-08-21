from pathlib import Path
import pandas as pd
import matplotlib

for backend in ["MacOSX", "TkAgg"]:
    try:
        matplotlib.use(backend)
        break
    except Exception:
        pass

import matplotlib.pyplot as plt
from PIL import Image


SCENE_ID = "scene0087_00"

OBJECTS = [
    "scene0087_00__sofa_003",
    "scene0087_00__table_006",
    "scene0087_00__table_012",
    "scene0087_00__chair_004",
]


def find_topdown(scene_id: str) -> Path:
    candidates = []

    for p in Path(".").rglob(f"{scene_id}_numbered.jpg"):
        if "demo_visuals" not in str(p):
            candidates.append(p)

    for p in Path(".").rglob(f"{scene_id}_numbered.png"):
        if "demo_visuals" not in str(p):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Could not find top-down numbered image for {scene_id}")

    candidates = sorted(candidates, key=lambda x: len(str(x)))
    return candidates[0]


def find_perspective(scene_id: str) -> Path:
    candidates = []

    folders = [
        Path("assets/perspective_scenes"),
        Path("assets/demo_perspective"),
        Path("artifacts/perspective_scenes"),
        Path("artifacts/demo_perspective"),
    ]

    names = [
        scene_id,
        f"{scene_id}_perspective",
        f"{scene_id}_3d",
        f"{scene_id}_view",
    ]

    exts = [".png", ".jpg", ".jpeg", ".webp"]

    for folder in folders:
        for name in names:
            for ext in exts:
                p = folder / f"{name}{ext}"
                if p.exists():
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find perspective image for {scene_id}. "
            f"Expected something like assets/perspective_scenes/{scene_id}.png"
        )

    return candidates[0]


def collect_points(image_path: Path, title: str):
    img = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

    points = []

    for obj in OBJECTS:
        print(f"\nClick the visual center of: {obj}")
        ax.set_title(f"{title}\nClick: {obj}")
        plt.draw()

        clicked = plt.ginput(1, timeout=0)

        if not clicked:
            raise RuntimeError(f"No click captured for {obj}")

        x, y = clicked[0]
        x, y = int(round(x)), int(round(y))

        print(f"{obj}: x={x}, y={y}")
        points.append({
            "scene_id": SCENE_ID,
            "object_id": obj,
            "x": x,
            "y": y,
        })

        ax.scatter([x], [y], s=140)
        ax.text(x + 8, y - 8, obj.split("__")[-1], fontsize=8)
        plt.draw()

    plt.close(fig)
    return points


def main():
    topdown_path = find_topdown(SCENE_ID)
    perspective_path = find_perspective(SCENE_ID)

    print(f"Top-down image: {topdown_path}")
    print(f"Perspective image: {perspective_path}")

    print("\nCalibrating TOP-DOWN markers...")
    topdown_points = collect_points(topdown_path, "Top-down marker calibration")

    print("\nCalibrating PERSPECTIVE markers...")
    perspective_points = collect_points(perspective_path, "Perspective marker calibration")

    Path("assets/topdown_scenes").mkdir(parents=True, exist_ok=True)
    Path("assets/perspective_scenes").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(topdown_points).to_csv("assets/topdown_scenes/markers.csv", index=False)
    pd.DataFrame(perspective_points).to_csv("assets/perspective_scenes/markers.csv", index=False)

    print("\nSaved:")
    print("assets/topdown_scenes/markers.csv")
    print("assets/perspective_scenes/markers.csv")


if __name__ == "__main__":
    main()
