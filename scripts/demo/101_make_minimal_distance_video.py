from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit("Instale open3d: pip install open3d") from e

try:
    from plyfile import PlyData
except ImportError as e:
    raise SystemExit("Instale plyfile: pip install plyfile") from e


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


def resolve_ply_path(scene_dir: Path, scene_id: str) -> Path:
    candidates = [
        scene_dir / f"{scene_id}_vh_clean_2.labels.ply",
        scene_dir / f"{scene_id}_vh_clean_2.ply",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "PLY file not found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def compute_gravity_aligned_obb(
    points: np.ndarray,
    q_low: float = 5.0,
    q_high: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OBB alinhada à gravidade:
    - eixo Z fixo
    - rotação apenas no plano XY
    """
    lo = np.percentile(points, q_low, axis=0)
    hi = np.percentile(points, q_high, axis=0)

    mask = np.all((points >= lo) & (points <= hi), axis=1)
    pts = points[mask]
    if len(pts) < 20:
        pts = points

    center_xy = pts[:, :2].mean(axis=0)
    z_min = pts[:, 2].min()
    z_max = pts[:, 2].max()
    z_center = 0.5 * (z_min + z_max)

    xy = pts[:, :2] - center_xy[None, :]
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    x_axis_xy = eigvecs[:, 0]
    y_axis_xy = eigvecs[:, 1]

    x_axis = np.array([x_axis_xy[0], x_axis_xy[1], 0.0], dtype=np.float32)
    y_axis = np.array([y_axis_xy[0], y_axis_xy[1], 0.0], dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    center = np.array([center_xy[0], center_xy[1], z_center], dtype=np.float32)
    rel = pts - center[None, :]

    proj_x = rel @ x_axis
    proj_y = rel @ y_axis
    proj_z = rel @ z_axis

    x_min, x_max = proj_x.min(), proj_x.max()
    y_min, y_max = proj_y.min(), proj_y.max()
    z_min_l, z_max_l = proj_z.min(), proj_z.max()

    local_corners = np.array([
        [x_min, y_min, z_min_l],
        [x_max, y_min, z_min_l],
        [x_max, y_max, z_min_l],
        [x_min, y_max, z_min_l],
        [x_min, y_min, z_max_l],
        [x_max, y_min, z_max_l],
        [x_max, y_max, z_max_l],
        [x_min, y_max, z_max_l],
    ], dtype=np.float32)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    world_corners = center[None, :] + local_corners @ R.T

    return world_corners.astype(np.float32), center.astype(np.float32)


def build_object_points(scene_dir: Path, alias_csv: Path) -> dict[str, dict[str, Any]]:
    scene_id = scene_dir.name
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    ply_path = resolve_ply_path(scene_dir, scene_id)

    if not segs_path.exists():
        raise FileNotFoundError(f"Segs file not found: {segs_path}")
    if not agg_path.exists():
        raise FileNotFoundError(f"Aggregation file not found: {agg_path}")
    if not alias_csv.exists():
        raise FileNotFoundError(f"Alias CSV not found: {alias_csv}")

    vertices = load_vertices_xyz(ply_path)

    with segs_path.open("r", encoding="utf-8") as f:
        segs_data = json.load(f)
    seg_indices = np.asarray(segs_data["segIndices"], dtype=np.int32)

    if len(seg_indices) != len(vertices):
        raise ValueError(
            f"segIndices length ({len(seg_indices)}) != number of vertices ({len(vertices)})"
        )

    with agg_path.open("r", encoding="utf-8") as f:
        agg_data = json.load(f)

    rawid_to_segments: dict[int, list[int]] = {}
    for g in agg_data["segGroups"]:
        rawid_to_segments[int(g["objectId"])] = list(g.get("segments", []))

    alias_df = pd.read_csv(alias_csv)

    objects: dict[str, dict[str, Any]] = {}
    for _, row in alias_df.iterrows():
        alias = str(row["alias"])
        raw_object_id = int(row["raw_object_id"])
        label_norm = normalize_label(row["label_norm"])

        if raw_object_id not in rawid_to_segments:
            continue

        segs = set(rawid_to_segments[raw_object_id])
        mask = np.isin(seg_indices, list(segs))
        pts = vertices[mask]
        if len(pts) == 0:
            continue

        corners, visual_center = compute_gravity_aligned_obb(pts)

        objects[alias] = {
            "alias": alias,
            "label_norm": label_norm,
            "raw_object_id": raw_object_id,
            "points": pts,
            "centroid": visual_center,
            "bbox3d_corners": corners,
        }

    return objects


def project_points(
    pts_world: np.ndarray,
    world_to_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
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
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy
    uv = np.stack([u, v], axis=1)
    return uv, z


def visible_projected_corners(
    corners_world: np.ndarray,
    world_to_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    W: int,
    H: int,
) -> np.ndarray | None:
    ones = np.ones((corners_world.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([corners_world, ones])
    pts_cam_h = (world_to_camera @ pts_h.T).T
    pts_cam = pts_cam_h[:, :3]

    z = pts_cam[:, 2]
    valid = z > 1e-6

    if valid.sum() < 4:
        return None

    pts_cam = pts_cam[valid]
    z = z[valid]

    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy
    uv = np.stack([u, v], axis=1)

    return uv.astype(np.float32)


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


def surface_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    best = float("inf")
    for pa in points_a:
        d2 = np.sum((points_b - pa[None, :]) ** 2, axis=1)
        best = min(best, float(np.sqrt(d2.min())))
    for pb in points_b:
        d2 = np.sum((points_a - pb[None, :]) ** 2, axis=1)
        best = min(best, float(np.sqrt(d2.min())))
    return best


def draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color):
    pad = 4
    bb = draw.textbbox((x, y), text)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    draw.rounded_rectangle(
        [x, y, x + tw + 2 * pad, y + th + 2 * pad],
        radius=6,
        fill=(255, 255, 255, 220),
        outline=color,
        width=2,
    )
    draw.text((x + pad, y + pad), text, fill=color)


def smooth(prev, new, alpha: float):
    if prev is None:
        return new
    if new is None:
        return prev

    arr_prev = np.asarray(prev, dtype=np.float32)
    arr_new = np.asarray(new, dtype=np.float32)

    if arr_prev.shape != arr_new.shape:
        return arr_new

    return alpha * arr_prev + (1.0 - alpha) * arr_new


def draw_3d_bbox(draw: ImageDraw.ImageDraw, pts2d: np.ndarray, color, width: int, W: int, H: int):
    if pts2d is None or len(pts2d) < 4:
        return

    if len(pts2d) < 8:
        xs = pts2d[:, 0]
        ys = pts2d[:, 1]
        x1 = int(np.clip(np.min(xs), 0, W - 1))
        x2 = int(np.clip(np.max(xs), 0, W - 1))
        y1 = int(np.clip(np.min(ys), 0, H - 1))
        y2 = int(np.clip(np.max(ys), 0, H - 1))
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        return

    pts = [(int(round(p[0])), int(round(p[1]))) for p in pts2d[:8]]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for a, b in edges:
        draw.line([pts[a], pts[b]], fill=color, width=width)


def ffmpeg_make_video(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg não encontrado no sistema. Instale e tente novamente.")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def build_typed_overlay_text(frame_idx: int, total_frames: int, question: str, answer: str):
    intro_frames = max(6, int(0.12 * total_frames))
    typing_frames = max(12, int(0.28 * total_frames))
    answer_start = intro_frames + typing_frames

    if frame_idx < intro_frames:
        return "", ""

    if frame_idx < answer_start:
        typed_len = int((frame_idx - intro_frames + 1) / typing_frames * len(question))
        typed_len = max(0, min(len(question), typed_len))
        q = question[:typed_len]
        if frame_idx % 2 == 0:
            q += "|"
        return q, ""

    return question, answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a minimal stable distance video with only two gravity-aligned 3D boxes."
    )
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--dense_extract_dir", type=Path, required=True)
    parser.add_argument("--alias_csv", type=Path, required=True)
    parser.add_argument("--object_a", required=True)
    parser.add_argument("--object_b", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--smooth_alpha", type=float, default=0.80)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    dense_dir = args.dense_extract_dir.resolve()
    alias_csv = args.alias_csv.resolve()
    out_dir = args.out_dir.resolve()

    intrinsic_txt = dense_dir / "meta" / "intrinsic_color.txt"
    color_dir = dense_dir / "color"
    pose_dir = dense_dir / "pose"

    if not intrinsic_txt.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {intrinsic_txt}")
    if not color_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {color_dir}")
    if not pose_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {pose_dir}")

    objects = build_object_points(scene_dir, alias_csv)

    if args.object_a not in objects:
        raise ValueError(f"Alias desconhecido: {args.object_a}")
    if args.object_b not in objects:
        raise ValueError(f"Alias desconhecido: {args.object_b}")

    fx, fy, cx, cy = load_intrinsics(intrinsic_txt)

    frames_in = sorted(color_dir.glob("*.jpg"))
    if not frames_in:
        raise RuntimeError("Nenhum frame JPG encontrado na janela densa.")

    dist_m = surface_distance(objects[args.object_a]["points"], objects[args.object_b]["points"])
    question_text = f"Qual a distância entre {args.object_a} e {args.object_b}?"
    answer_text = f"{dist_m:.3f} m"

    clip_name = f"distance_{args.object_a}_{args.object_b}"
    frames_out = out_dir / clip_name / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    colors = {
        args.object_a: (220, 20, 60, 255),
        args.object_b: (65, 105, 225, 255),
    }

    prev_corners = {
        args.object_a: None,
        args.object_b: None,
    }
    prev_centroids = {
        args.object_a: None,
        args.object_b: None,
    }

    total_frames = len(frames_in)

    for out_idx, image_path in enumerate(frames_in):
        frame_id = image_path.stem
        pose_path = pose_dir / f"{frame_id}.txt"
        if not pose_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        base = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        cam_to_world = load_pose(pose_path)
        world_to_camera = np.linalg.inv(cam_to_world)

        current_centroids: dict[str, Any] = {}
        current_corners: dict[str, Any] = {}

        for alias in [args.object_a, args.object_b]:
            corners_world = objects[alias]["bbox3d_corners"]
            corners_uv = visible_projected_corners(
                corners_world,
                world_to_camera,
                fx, fy, cx, cy,
                W, H,
            )

            if corners_uv is not None:
                smoothed_corners = smooth(prev_corners[alias], corners_uv, args.smooth_alpha)
                prev_corners[alias] = smoothed_corners
                current_corners[alias] = smoothed_corners
            else:
                current_corners[alias] = prev_corners[alias]

            cp = project_centroid(
                objects[alias]["centroid"],
                world_to_camera,
                fx, fy, cx, cy,
            )
            if cp is not None:
                smoothed_cp = smooth(prev_centroids[alias], cp, args.smooth_alpha)
                prev_centroids[alias] = smoothed_cp
                current_centroids[alias] = smoothed_cp
            else:
                current_centroids[alias] = prev_centroids[alias]

        typed_question, typed_answer = build_typed_overlay_text(
            frame_idx=out_idx,
            total_frames=total_frames,
            question=question_text,
            answer=answer_text,
        )

        header_y1 = 24
        header_y2 = 120 if typed_answer else 90
        draw.rounded_rectangle(
            [24, header_y1, W - 24, header_y2],
            radius=14,
            fill=(255, 255, 255, 220),
        )

        if typed_question:
            draw.text((40, 36), typed_question, fill=(20, 20, 20, 255))

        if typed_answer:
            draw.text((40, 72), typed_answer, fill=(20, 20, 20, 255))

        for alias in [args.object_a, args.object_b]:
            pts2d = current_corners[alias]
            if pts2d is None:
                continue

            color = colors[alias]
            draw_3d_bbox(draw, pts2d, color=color, width=4, W=W, H=H)

            x = int(np.clip(np.min(pts2d[:, 0]), 0, W - 1))
            y = int(np.clip(np.min(pts2d[:, 1]) - 28, 5, H - 1))
            draw_tag(draw, x, y, alias, color)

        pa = current_centroids[args.object_a]
        pb = current_centroids[args.object_b]
        if pa is not None and pb is not None:
            if (0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H):
                draw.line([tuple(pa), tuple(pb)], fill=(255, 140, 0, 255), width=4)

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        composed.save(frames_out / f"{out_idx:06d}.png")

    out_mp4 = out_dir / clip_name / f"{clip_name}.mp4"
    ffmpeg_make_video(frames_out, out_mp4, args.fps)

    print(f"Frames rendered: {len(list(frames_out.glob('*.png')))}")
    print(f"Video saved: {out_mp4}")
    print(f"Query summary: {question_text} {answer_text}")


if __name__ == "__main__":
    main()