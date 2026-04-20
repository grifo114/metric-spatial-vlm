from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from plyfile import PlyData
except ImportError as e:
    raise SystemExit(
        "Este script precisa do pacote 'plyfile'. Instale com:\n"
        "pip install plyfile"
    ) from e


def load_vertices_xyz(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz


def load_intrinsics(intrinsic_txt: Path) -> tuple[float, float, float, float]:
    K = np.loadtxt(intrinsic_txt)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def load_pose(pose_txt: Path) -> np.ndarray:
    return np.loadtxt(pose_txt).astype(np.float32)


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def build_object_points(scene_dir: Path, alias_csv: Path) -> dict[str, dict]:
    scene_id = scene_dir.name
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.labels.ply"

    vertices = load_vertices_xyz(ply_path)

    with segs_path.open("r", encoding="utf-8") as f:
        segs_data = json.load(f)
    seg_indices = np.asarray(segs_data["segIndices"], dtype=np.int32)

    with agg_path.open("r", encoding="utf-8") as f:
        agg_data = json.load(f)

    rawid_to_segments: dict[int, list[int]] = {}
    for g in agg_data["segGroups"]:
        rawid_to_segments[int(g["objectId"])] = list(g.get("segments", []))

    alias_df = pd.read_csv(alias_csv)

    objects: dict[str, dict] = {}
    for _, row in alias_df.iterrows():
        alias = str(row["alias"])
        raw_object_id = int(row["raw_object_id"])
        label_norm = normalize_label(row["label_norm"])
        segs = set(rawid_to_segments[raw_object_id])
        mask = np.isin(seg_indices, list(segs))
        pts = vertices[mask]
        if len(pts) == 0:
            continue

        centroid = pts.mean(axis=0)

        objects[alias] = {
            "alias": alias,
            "label_norm": label_norm,
            "raw_object_id": raw_object_id,
            "points": pts,
            "centroid": centroid,
        }

    return objects


def project_points(
    pts_world: np.ndarray,
    world_to_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones((pts_world.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts_world, ones])
    pts_cam_h = (world_to_camera @ pts_h.T).T
    pts_cam = pts_cam_h[:, :3]

    z = pts_cam[:, 2]
    valid = z > 1e-6
    pts_cam = pts_cam[valid]
    z = z[valid]
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy
    uv = np.stack([u, v], axis=1)

    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < image_w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < image_h)
    )
    return uv[inside], pts_cam[inside]


def compute_bbox(uv: np.ndarray) -> tuple[int, int, int, int] | None:
    if uv.shape[0] < 20:
        return None
    x1 = int(np.floor(np.min(uv[:, 0])))
    y1 = int(np.floor(np.min(uv[:, 1])))
    x2 = int(np.ceil(np.max(uv[:, 0])))
    y2 = int(np.ceil(np.max(uv[:, 1])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def project_centroid(
    centroid_world: np.ndarray,
    world_to_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    p = np.hstack([centroid_world.astype(np.float32), [1.0]])
    cam = world_to_camera @ p
    if cam[2] <= 1e-6:
        return None
    u = fx * (cam[0] / cam[2]) + cx
    v = fy * (cam[1] / cam[2]) + cy
    return float(u), float(v)


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    pad = 4
    try:
        bb = draw.textbbox((x, y), text)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
    except Exception:
        tw = 8 * len(text)
        th = 14
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=(255, 255, 255), outline=color, width=2)
    draw.text((x + pad, y + pad), text, fill=color)


def surface_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    best = float("inf")
    for pa in points_a:
        d2 = np.sum((points_b - pa[None, :]) ** 2, axis=1)
        local = float(np.sqrt(d2.min()))
        if local < best:
            best = local
    for pb in points_b:
        d2 = np.sum((points_a - pb[None, :]) ** 2, axis=1)
        local = float(np.sqrt(d2.min()))
        if local < best:
            best = local
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay a single query on one extracted ScanNet frame.")
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--extract_dir", type=Path, required=True)
    parser.add_argument("--alias_csv", type=Path, required=True)
    parser.add_argument("--frame_id", type=str, required=True, help="Saved frame id, e.g. 000010")
    parser.add_argument("--mode", type=str, choices=["distance", "nearest"], required=True)
    parser.add_argument("--object_a", type=str, required=True, help="For distance: first alias. For nearest: reference alias.")
    parser.add_argument("--object_b", type=str, default="", help="For distance: second alias.")
    parser.add_argument("--target_category", type=str, default="", help="For nearest: category to search, e.g. chair")
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    extract_dir = args.extract_dir.resolve()
    alias_csv = args.alias_csv.resolve()
    out_path = args.out_path.resolve()

    intrinsic_txt = extract_dir / "meta" / "intrinsic_color.txt"
    image_path = extract_dir / "color" / f"{args.frame_id}.jpg"
    pose_path = extract_dir / "pose" / f"{args.frame_id}.txt"

    objects = build_object_points(scene_dir, alias_csv)
    if args.object_a not in objects:
        raise ValueError(f"Unknown alias: {args.object_a}")

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    draw = ImageDraw.Draw(image)

    fx, fy, cx, cy = load_intrinsics(intrinsic_txt)
    cam_to_world = load_pose(pose_path)
    world_to_camera = np.linalg.inv(cam_to_world)

    projected = {}
    for alias, obj in objects.items():
        uv, _ = project_points(obj["points"], world_to_camera, fx, fy, cx, cy, W, H)
        bbox = compute_bbox(uv)
        if bbox is None:
            continue
        projected[alias] = bbox

    highlight = []
    answer_text = ""

    if args.mode == "distance":
        if not args.object_b:
            raise ValueError("--object_b is required for distance")
        if args.object_b not in objects:
            raise ValueError(f"Unknown alias: {args.object_b}")

        a = objects[args.object_a]
        b = objects[args.object_b]
        d = surface_distance(a["points"], b["points"])
        answer_text = f"distance({args.object_a}, {args.object_b}) = {d:.3f} m"
        highlight = [args.object_a, args.object_b]

        pa = project_centroid(a["centroid"], world_to_camera, fx, fy, cx, cy)
        pb = project_centroid(b["centroid"], world_to_camera, fx, fy, cx, cy)
        if pa is not None and pb is not None:
            draw.line([pa, pb], fill=(220, 20, 60), width=4)

    elif args.mode == "nearest":
        if not args.target_category:
            raise ValueError("--target_category is required for nearest")

        ref = objects[args.object_a]
        candidates = [o for o in objects.values() if o["label_norm"] == normalize_label(args.target_category)]
        candidates = [o for o in candidates if o["alias"] != args.object_a]

        if not candidates:
            raise ValueError(f"No candidates for category: {args.target_category}")

        best_alias = None
        best_d = float("inf")
        for cand in candidates:
            d = surface_distance(ref["points"], cand["points"])
            if d < best_d:
                best_d = d
                best_alias = cand["alias"]

        answer_text = f"nearest({args.target_category}, {args.object_a}) = {best_alias} ({best_d:.3f} m)"
        highlight = [args.object_a, best_alias]

    # desenha boxes contextuais
    for alias, bbox in projected.items():
        x1, y1, x2, y2 = bbox
        if alias in highlight:
            color = (220, 20, 60)
            width = 4
        else:
            color = (70, 130, 180)
            width = 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw_label(draw, x1, max(0, y1 - 24), alias, color)

    draw_label(draw, 12, 12, f"frame {args.frame_id}", (0, 0, 0))
    draw_label(draw, 12, 44, answer_text, (0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Answer: {answer_text}")


if __name__ == "__main__":
    main()