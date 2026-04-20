from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from plyfile import PlyData
except ImportError as e:
    raise SystemExit("Instale plyfile: pip install plyfile") from e


def load_vertices_xyz(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


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
) -> np.ndarray:
    ones = np.ones((pts_world.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts_world, ones])
    pts_cam_h = (world_to_camera @ pts_h.T).T
    pts_cam = pts_cam_h[:, :3]

    z = pts_cam[:, 2]
    valid = z > 1e-6
    pts_cam = pts_cam[valid]
    z = z[valid]
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.float32)

    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy
    uv = np.stack([u, v], axis=1)

    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < image_w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < image_h)
    )
    return uv[inside]


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


def compute_bbox(uv: np.ndarray) -> tuple[float, float, float, float] | None:
    if uv.shape[0] < 20:
        return None
    x1 = float(np.min(uv[:, 0]))
    y1 = float(np.min(uv[:, 1]))
    x2 = float(np.max(uv[:, 0]))
    y2 = float(np.max(uv[:, 1]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def surface_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    best = float("inf")
    for pa in points_a:
        d2 = np.sum((points_b - pa[None, :]) ** 2, axis=1)
        best = min(best, float(np.sqrt(d2.min())))
    for pb in points_b:
        d2 = np.sum((points_a - pb[None, :]) ** 2, axis=1)
        best = min(best, float(np.sqrt(d2.min())))
    return best


def smooth_box(prev_box, new_box, alpha: float):
    if prev_box is None:
        return new_box
    if new_box is None:
        return prev_box
    return tuple(alpha * p + (1.0 - alpha) * n for p, n in zip(prev_box, new_box))


def smooth_point(prev_pt, new_pt, alpha: float):
    if prev_pt is None:
        return new_pt
    if new_pt is None:
        return prev_pt
    return (alpha * prev_pt[0] + (1.0 - alpha) * new_pt[0], alpha * prev_pt[1] + (1.0 - alpha) * new_pt[1])


def draw_box(draw: ImageDraw.ImageDraw, box, color, width):
    if box is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color):
    pad = 4
    bb = draw.textbbox((x, y), text)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    draw.rounded_rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], radius=6, fill=(255, 255, 255), outline=color, width=2)
    draw.text((x + pad, y + pad), text, fill=color)


def ffmpeg_make_video(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a stable minimal-overlay ScanNet query video.")
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--dense_extract_dir", type=Path, required=True)
    parser.add_argument("--alias_csv", type=Path, required=True)
    parser.add_argument("--mode", choices=["distance", "nearest"], required=True)
    parser.add_argument("--object_a", required=True)
    parser.add_argument("--object_b", default="")
    parser.add_argument("--target_category", default="")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--smooth_alpha", type=float, default=0.75)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    dense_dir = args.dense_extract_dir.resolve()
    alias_csv = args.alias_csv.resolve()
    out_dir = args.out_dir.resolve()

    objects = build_object_points(scene_dir, alias_csv)
    if args.object_a not in objects:
        raise ValueError(f"Unknown alias: {args.object_a}")

    if args.mode == "distance":
        if args.object_b not in objects:
            raise ValueError(f"Unknown alias: {args.object_b}")
        highlight_aliases = [args.object_a, args.object_b]
        answer_text = f"{args.object_a} ↔ {args.object_b}: {surface_distance(objects[args.object_a]['points'], objects[args.object_b]['points']):.3f} m"
        winner_alias = None
    else:
        candidates = [o for o in objects.values() if o["label_norm"] == normalize_label(args.target_category)]
        candidates = [o for o in candidates if o["alias"] != args.object_a]
        if not candidates:
            raise ValueError(f"No candidates for category: {args.target_category}")

        winner_alias = None
        winner_dist = float("inf")
        for cand in candidates:
            d = surface_distance(objects[args.object_a]["points"], cand["points"])
            if d < winner_dist:
                winner_dist = d
                winner_alias = cand["alias"]

        highlight_aliases = [args.object_a, winner_alias]
        answer_text = f"nearest {args.target_category} to {args.object_a}: {winner_alias} ({winner_dist:.3f} m)"

    frames_in = sorted((dense_dir / "color").glob("*.jpg"))
    if not frames_in:
        raise RuntimeError(f"No JPG frames found in {dense_dir / 'color'}")

    intrinsic_txt = dense_dir / "meta" / "intrinsic_color.txt"
    fx, fy, cx, cy = load_intrinsics(intrinsic_txt)

    clip_name = f"{args.mode}_{args.object_a}" + (f"_{args.object_b}" if args.mode == "distance" else f"_{winner_alias}")
    frames_out = out_dir / clip_name / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    prev_boxes = {alias: None for alias in highlight_aliases}
    prev_points = {alias: None for alias in highlight_aliases}

    for out_idx, image_path in enumerate(frames_in):
        frame_id = image_path.stem
        pose_path = dense_dir / "pose" / f"{frame_id}.txt"
        if not pose_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cam_to_world = load_pose(pose_path)
        world_to_camera = np.linalg.inv(cam_to_world)

        curr_boxes = {}
        curr_centroids = {}

        for alias in highlight_aliases:
            obj = objects[alias]
            uv = project_points(obj["points"], world_to_camera, fx, fy, cx, cy, W, H)
            box = compute_bbox(uv)
            curr_boxes[alias] = smooth_box(prev_boxes[alias], box, args.smooth_alpha)
            prev_boxes[alias] = curr_boxes[alias]

            cp = project_centroid(obj["centroid"], world_to_camera, fx, fy, cx, cy)
            curr_centroids[alias] = smooth_point(prev_points[alias], cp, args.smooth_alpha)
            prev_points[alias] = curr_centroids[alias]

        # header
        draw.rounded_rectangle([20, 20, W - 20, 90], radius=14, fill=(255, 255, 255, 220))
        draw.text((35, 32), answer_text, fill=(20, 20, 20, 255))

        # boxes
        colors = [(220, 20, 60, 255), (65, 105, 225, 255)]
        for alias, color in zip(highlight_aliases, colors):
            box = curr_boxes[alias]
            if box is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            draw.rounded_rectangle([x1, y1, x2, y2], radius=8, outline=color, width=5)
            draw_tag(draw, x1, max(5, y1 - 28), alias, color)

        # distance line
        if args.mode == "distance":
            pa = curr_centroids[args.object_a]
            pb = curr_centroids[args.object_b]
            if pa is not None and pb is not None:
                draw.line([pa, pb], fill=(255, 140, 0, 255), width=5)

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        composed.save(frames_out / f"{out_idx:06d}.png")

    out_mp4 = out_dir / clip_name / f"{clip_name}.mp4"
    ffmpeg_make_video(frames_out, out_mp4, args.fps)

    print(f"Video saved: {out_mp4}")
    print(f"Query summary: {answer_text}")


if __name__ == "__main__":
    main()