from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _as_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _find_xyz_columns(df: pd.DataFrame):
    """
    Try to find centroid/center columns in the object manifest.
    The project has changed names across scripts, so this function is defensive.
    """
    candidates = [
        ("centroid_x", "centroid_y", "centroid_z"),
        ("center_x", "center_y", "center_z"),
        ("bbox_center_x", "bbox_center_y", "bbox_center_z"),
        ("cx", "cy", "cz"),
        ("x", "y", "z"),
    ]

    cols = set(df.columns)

    for triplet in candidates:
        if all(c in cols for c in triplet):
            return triplet

    return None


def _object_center_from_row(row: pd.Series, xyz_cols):
    if xyz_cols is None:
        return None

    x = _as_float(row.get(xyz_cols[0]))
    y = _as_float(row.get(xyz_cols[1]))
    z = _as_float(row.get(xyz_cols[2]))

    if x is None or y is None or z is None:
        return None

    return np.array([x, y, z], dtype=float)


def _object_center_from_points(points: np.ndarray | None):
    if points is None:
        return None

    arr = np.asarray(points)

    if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
        return None

    return np.nanmean(arr[:, :3], axis=0)


def _get_object_row(sdf: pd.DataFrame, object_id: str):
    if object_id is None:
        return None

    if "object_id" not in sdf.columns:
        return None

    m = sdf[sdf["object_id"].astype(str) == str(object_id)]

    if m.empty:
        return None

    return m.iloc[0]


def _short_id(object_id: str | None):
    if not object_id:
        return ""

    text = str(object_id)

    if "__" in text:
        return text.split("__")[-1]

    return text


def _safe_font(size: int = 16):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _compute_xy_bounds(sdf: pd.DataFrame, xyz_cols):
    centers = []

    if xyz_cols is None:
        return None

    for _, row in sdf.iterrows():
        c = _object_center_from_row(row, xyz_cols)
        if c is not None:
            centers.append(c)

    if not centers:
        return None

    arr = np.vstack(centers)

    min_x, max_x = float(np.nanmin(arr[:, 0])), float(np.nanmax(arr[:, 0]))
    min_y, max_y = float(np.nanmin(arr[:, 1])), float(np.nanmax(arr[:, 1]))

    if abs(max_x - min_x) < 1e-9:
        max_x = min_x + 1.0

    if abs(max_y - min_y) < 1e-9:
        max_y = min_y + 1.0

    return min_x, max_x, min_y, max_y


def _xy_to_pixel(center_xyz, bounds, image_size, margin_px: int = 30):
    """
    Project XY scene coordinates to image pixels.

    This assumes the top-down render follows the same canonical XY frame.
    If the render function uses a different orientation, we can adjust this later.
    """
    if center_xyz is None or bounds is None:
        return None

    w, h = image_size
    min_x, max_x, min_y, max_y = bounds

    x = float(center_xyz[0])
    y = float(center_xyz[1])

    px = margin_px + (x - min_x) / (max_x - min_x) * (w - 2 * margin_px)
    py = h - margin_px - (y - min_y) / (max_y - min_y) * (h - 2 * margin_px)

    return int(round(px)), int(round(py))


def draw_topdown_overlay(
    render_path: str | Path,
    scene_df: pd.DataFrame,
    selected_a: str | None = None,
    selected_b: str | None = None,
    distance_m: float | None = None,
    out_path: str | Path | None = None,
):
    """
    Draw selected objects and distance on top of the numbered top-down render.
    Uses neutral grayscale annotations.
    """
    render_path = Path(render_path)

    if out_path is None:
        out_path = render_path.with_name(render_path.stem + "_overlay.jpg")
    else:
        out_path = Path(out_path)

    img = Image.open(render_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = _safe_font(18)
    small_font = _safe_font(14)

    xyz_cols = _find_xyz_columns(scene_df)
    bounds = _compute_xy_bounds(scene_df, xyz_cols)

    points = {}

    for object_id, label in [(selected_a, "A"), (selected_b, "B")]:
        row = _get_object_row(scene_df, object_id)

        if row is None:
            points[label] = None
            continue

        center = _object_center_from_row(row, xyz_cols)
        points[label] = _xy_to_pixel(center, bounds, img.size)

    pa = points.get("A")
    pb = points.get("B")

    if pa is not None and pb is not None:
        #draw.line([pa, pb], fill=(0, 0, 0), width=5)

        if distance_m is not None:
            mx = int((pa[0] + pb[0]) / 2)
            my = int((pa[1] + pb[1]) / 2)
            text = f"{distance_m:.3f} m"
            pad = 5
            bbox = draw.textbbox((mx, my), text, font=small_font)
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=2,
            )
            draw.text((mx, my), text, fill=(0, 0, 0), font=small_font)

    for object_id, label, fill, text_fill in [
        (selected_a, "A", (0, 0, 0), (255, 255, 255)),
        (selected_b, "B", (255, 255, 255), (0, 0, 0)),
    ]:
        p = points.get(label)

        if p is None:
            continue

        x, y = p
        r = 18

        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=fill,
            outline=(0, 0, 0),
            width=4,
        )
        draw.text((x - 5, y - 9), label, fill=text_fill, font=font)

        id_text = f"{label}: {_short_id(object_id)}"
        tx, ty = x + 22, y - 12

        bbox = draw.textbbox((tx, ty), id_text, font=small_font)
        draw.rectangle(
            [bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=1,
        )
        draw.text((tx, ty), id_text, fill=(0, 0, 0), font=small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


def render_scene_3d_static(
    scene_id: str,
    scene_df: pd.DataFrame,
    load_points: Callable[[str, str], np.ndarray | None],
    selected_a: str | None = None,
    selected_b: str | None = None,
    distance_m: float | None = None,
    out_path: str | Path | None = None,
    max_points_other: int = 120,
    max_points_selected: int = 1200,
):
    """
    Render a static 3D view using available object point sets.
    This is not an interactive 3D viewer. It is a compact visual render for demos.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = Path("artifacts/demo_visuals") / f"{scene_id}_3d_overlay.png"
    else:
        out_path = Path(out_path)

    selected = {str(x) for x in [selected_a, selected_b] if x}

    fig = plt.figure(figsize=(7.2, 4.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    centers = {}

    if "object_id" not in scene_df.columns:
        ax.text2D(0.05, 0.5, "No object manifest available", transform=ax.transAxes)
    else:
        for _, row in scene_df.iterrows():
            oid = str(row["object_id"])
            pts = load_points(oid, scene_id)

            if pts is None:
                continue

            arr = np.asarray(pts)

            if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
                continue

            arr = arr[:, :3]
            centers[oid] = np.nanmean(arr, axis=0)

            if oid in selected:
                limit = min(max_points_selected, len(arr))
                alpha = 0.95
                size = 2.2
                shade = "black" if oid == str(selected_a) else "dimgray"
            else:
                limit = min(max_points_other, len(arr))
                alpha = 0.18
                size = 0.7
                shade = "lightgray"

            if len(arr) > limit:
                idx = np.linspace(0, len(arr) - 1, limit).astype(int)
                arr = arr[idx]

            ax.scatter(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                s=size,
                c=shade,
                alpha=alpha,
                depthshade=False,
            )

    ca = centers.get(str(selected_a))
    cb = centers.get(str(selected_b))

    if ca is not None and cb is not None:
        xs = [ca[0], cb[0]]
        ys = [ca[1], cb[1]]
        zs = [ca[2], cb[2]]

        ax.plot(xs, ys, zs, c="black", linewidth=2.5)

        mid = (ca + cb) / 2.0
        if distance_m is not None:
            ax.text(
                mid[0],
                mid[1],
                mid[2],
                f"{distance_m:.3f} m",
                fontsize=9,
                color="black",
            )

        ax.text(ca[0], ca[1], ca[2], "A", fontsize=10, color="black")
        ax.text(cb[0], cb[1], cb[2], "B", fontsize=10, color="black")

    ax.view_init(elev=28, azim=-55)
    ax.set_axis_off()

    try:
        ax.set_box_aspect((1.4, 1.0, 0.55))
    except Exception:
        pass

    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return out_path


# ============================================================
# Final demo overlays: point markers only
# ============================================================

def _load_marker_csv(root: str | Path, relative_path: str):
    path = Path(root) / relative_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _marker_center_from_csv(markers: pd.DataFrame | None, scene_id: str, object_id: str | None):
    if markers is None or markers.empty or object_id is None:
        return None

    if not {"scene_id", "object_id"}.issubset(set(markers.columns)):
        return None

    m = markers[
        (markers["scene_id"].astype(str) == str(scene_id)) &
        (markers["object_id"].astype(str) == str(object_id))
    ]

    if m.empty:
        return None

    row = m.iloc[0]
    cols = set(row.index)

    if {"x", "y"}.issubset(cols):
        return int(round(float(row["x"]))), int(round(float(row["y"])))

    if {"cx", "cy"}.issubset(cols):
        return int(round(float(row["cx"]))), int(round(float(row["cy"])))

    if {"x1", "y1", "x2", "y2"}.issubset(cols):
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))

    return None


def _draw_demo_point(draw, center, label, color, font, radius: int = 15):
    if center is None:
        return

    x, y = center
    r = radius

    draw.ellipse(
        [x - r - 4, y - r - 4, x + r + 4, y + r + 4],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=2,
    )

    draw.ellipse(
        [x - r, y - r, x + r, y + r],
        fill=color,
        outline=(0, 0, 0),
        width=2,
    )

    tx, ty = x + r + 6, y - r
    bbox = draw.textbbox((tx, ty), label, font=font)

    draw.rectangle(
        [bbox[0] - 5, bbox[1] - 4, bbox[2] + 5, bbox[3] + 4],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=1,
    )

    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)


def _object_center_from_manifest_or_points(
    scene_df: pd.DataFrame,
    object_id: str | None,
    scene_id: str | None,
    load_points: Callable[[str, str], np.ndarray | None] | None,
):
    if object_id is None:
        return None

    if "object_id" not in scene_df.columns:
        return None

    row = _get_object_row(scene_df, object_id)
    if row is None:
        return None

    xyz_cols = _find_xyz_columns(scene_df)
    center = _object_center_from_row(row, xyz_cols)

    if center is None and load_points is not None and scene_id is not None:
        pts = load_points(str(object_id), str(scene_id))
        center = _object_center_from_points(pts)

    return center


def draw_topdown_overlay(
    render_path: str | Path,
    scene_df: pd.DataFrame,
    selected_a: str | None = None,
    selected_b: str | None = None,
    distance_m: float | None = None,
    out_path: str | Path | None = None,
    scene_id: str | None = None,
    load_points: Callable[[str, str], np.ndarray | None] | None = None,
):
    """
    Draw point markers A/B on the top-down image.

    Priority:
    1. manual markers from assets/topdown_scenes/markers.csv
    2. fallback to centroid/point-set projection
    """
    render_path = Path(render_path)

    if out_path is None:
        out_path = render_path.with_name(render_path.stem + "_overlay.jpg")
    else:
        out_path = Path(out_path)

    img = Image.open(render_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _safe_font(16)

    root = Path.cwd()
    markers = _load_marker_csv(root, "assets/topdown_scenes/markers.csv")

    pa = _marker_center_from_csv(markers, scene_id, selected_a)
    pb = _marker_center_from_csv(markers, scene_id, selected_b)

    # Fallback if there is no manual marker.
    if pa is None or pb is None:
        centers = {}

        if "object_id" in scene_df.columns:
            all_centers = []

            for _, row in scene_df.iterrows():
                oid = str(row["object_id"])
                c = _object_center_from_manifest_or_points(scene_df, oid, scene_id, load_points)
                if c is not None:
                    centers[oid] = c
                    all_centers.append(c)

            if all_centers:
                arr = np.vstack(all_centers)
                min_x, max_x = float(np.nanmin(arr[:, 0])), float(np.nanmax(arr[:, 0]))
                min_y, max_y = float(np.nanmin(arr[:, 1])), float(np.nanmax(arr[:, 1]))

                if abs(max_x - min_x) < 1e-9:
                    max_x = min_x + 1.0
                if abs(max_y - min_y) < 1e-9:
                    max_y = min_y + 1.0

                def to_pixel(center_xyz, margin_px: int = 30):
                    if center_xyz is None:
                        return None

                    w, h = img.size
                    x = float(center_xyz[0])
                    y = float(center_xyz[1])

                    px = margin_px + (x - min_x) / (max_x - min_x) * (w - 2 * margin_px)
                    py = h - margin_px - (y - min_y) / (max_y - min_y) * (h - 2 * margin_px)

                    return int(round(px)), int(round(py))

                if pa is None:
                    pa = to_pixel(centers.get(str(selected_a)))
                if pb is None:
                    pb = to_pixel(centers.get(str(selected_b)))

    # No line. No distance label inside the image.
    _draw_demo_point(draw, pa, "A", (230, 0, 0), font, radius=18)
    _draw_demo_point(draw, pb, "B", (0, 85, 255), font, radius=18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


def find_perspective_image(root: str | Path, scene_id: str) -> Path | None:
    root = Path(root)

    folders = [
        root / "assets" / "perspective_scenes",
        root / "assets" / "demo_perspective",
        root / "artifacts" / "perspective_scenes",
        root / "artifacts" / "demo_perspective",
    ]

    names = [
        scene_id,
        f"{scene_id}_perspective",
        f"{scene_id}_3d",
        f"{scene_id}_view",
    ]

    extensions = [".png", ".jpg", ".jpeg", ".webp"]

    for folder in folders:
        for name in names:
            for ext in extensions:
                path = folder / f"{name}{ext}"
                if path.exists():
                    return path

    return None


def draw_perspective_overlay(
    image_path: str | Path,
    root: str | Path,
    scene_id: str,
    selected_a: str | None = None,
    selected_b: str | None = None,
    distance_m: float | None = None,
    out_path: str | Path | None = None,
):
    """
    Draw only calibrated A/B point markers on the perspective scene image.

    No bounding boxes.
    No connection line.
    No distance label inside the image.
    """
    image_path = Path(image_path)
    root = Path(root)

    if out_path is None:
        out_path = image_path.with_name(image_path.stem + "_points.png")
    else:
        out_path = Path(out_path)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _safe_font(16)

    markers = _load_marker_csv(root, "assets/perspective_scenes/markers.csv")

    pa = _marker_center_from_csv(markers, scene_id, selected_a)
    pb = _marker_center_from_csv(markers, scene_id, selected_b)

    _draw_demo_point(draw, pa, "A", (230, 0, 0), font, radius=18)
    _draw_demo_point(draw, pb, "B", (0, 85, 255), font, radius=18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)
    return out_path


def render_scene_3d_static(
    scene_id: str,
    scene_df: pd.DataFrame,
    load_points: Callable[[str, str], np.ndarray | None],
    selected_a: str | None = None,
    selected_b: str | None = None,
    distance_m: float | None = None,
    out_path: str | Path | None = None,
    max_points_other: int = 120,
    max_points_selected: int = 1200,
):
    """
    Fallback only. The demo should use a perspective image when available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = Path("artifacts/demo_visuals") / f"{scene_id}_3d_fallback.png"
    else:
        out_path = Path(out_path)

    fig = plt.figure(figsize=(7.0, 4.8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if "object_id" in scene_df.columns:
        for _, row in scene_df.iterrows():
            oid = str(row["object_id"])
            pts = load_points(oid, scene_id)

            if pts is None:
                continue

            arr = np.asarray(pts)
            if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
                continue

            arr = arr[:, :3]
            if len(arr) > max_points_other:
                idx = np.linspace(0, len(arr) - 1, max_points_other).astype(int)
                arr = arr[idx]

            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=0.8, c="gray", alpha=0.35, depthshade=False)

    ax.view_init(elev=22, azim=-60)
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    return out_path
