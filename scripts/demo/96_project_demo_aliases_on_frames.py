from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return fx, fy, cx, cy


def load_pose(pose_txt: Path) -> np.ndarray:
    pose = np.loadtxt(pose_txt).astype(np.float32)
    return pose


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def build_object_points(
    scene_dir: Path,
    alias_csv: Path,
) -> dict[str, dict]:
    scene_id = scene_dir.name

    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    
    if not segs_path.exists():
        raise FileNotFoundError(f"Segs file not found: {segs_path}")
    if not agg_path.exists():
        raise FileNotFoundError(f"Aggregation file not found: {agg_path}")
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
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

    seg_groups = agg_data["segGroups"]
    rawid_to_segments: dict[int, list[int]] = {}
    rawid_to_label: dict[int, str] = {}

    for g in seg_groups:
        raw_id = int(g["objectId"])
        rawid_to_segments[raw_id] = list(g.get("segments", []))
        rawid_to_label[raw_id] = normalize_label(g.get("label", ""))

    alias_df = pd.read_csv(alias_csv)

    objects: dict[str, dict] = {}
    for _, row in alias_df.iterrows():
        alias = str(row["alias"])
        raw_object_id = int(row["raw_object_id"])
        label_norm = normalize_label(row["label_norm"])

        if raw_object_id not in rawid_to_segments:
            print(f"[WARN] raw_object_id={raw_object_id} for alias={alias} not found in aggregation.")
            continue

        segs = set(rawid_to_segments[raw_object_id])
        mask = np.isin(seg_indices, list(segs))
        pts = vertices[mask]

        if pts.shape[0] == 0:
            print(f"[WARN] alias={alias} has zero points after segment lookup.")
            continue

        objects[alias] = {
            "alias": alias,
            "label_norm": label_norm,
            "raw_object_id": raw_object_id,
            "points": pts,
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
    pts_h = np.hstack([pts_world, ones])  # [N, 4]
    pts_cam_h = (world_to_camera @ pts_h.T).T  # [N, 4]
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


def compute_bbox_from_uv(uv: np.ndarray) -> tuple[int, int, int, int] | None:
    if uv.shape[0] < 20:
        return None

    x_min = int(np.floor(np.min(uv[:, 0])))
    y_min = int(np.floor(np.min(uv[:, 1])))
    x_max = int(np.ceil(np.max(uv[:, 0])))
    y_max = int(np.ceil(np.max(uv[:, 1])))

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    pad = 3
    try:
        bbox = draw.textbbox((x, y), text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw = 8 * len(text)
        th = 14

    bg = (255, 255, 255)
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=bg, outline=color, width=2)
    draw.text((x + pad, y + pad), text, fill=color)


def render_frame(
    image_path: Path,
    pose_path: Path,
    objects: dict[str, dict],
    intrinsic_txt: Path,
    out_path: Path,
    highlight_aliases: set[str] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    draw = ImageDraw.Draw(image)

    fx, fy, cx, cy = load_intrinsics(intrinsic_txt)
    cam_to_world = load_pose(pose_path)
    world_to_camera = np.linalg.inv(cam_to_world)

    if highlight_aliases is None:
        highlight_aliases = set()

    visible_count = 0
    for alias, obj in objects.items():
        uv = project_points(
            pts_world=obj["points"],
            world_to_camera=world_to_camera,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image_w=W,
            image_h=H,
        )
        bbox = compute_bbox_from_uv(uv)
        if bbox is None:
            continue

        visible_count += 1
        x1, y1, x2, y2 = bbox

        if alias in highlight_aliases:
            color = (220, 20, 60)   # crimson
            width = 4
        else:
            color = (70, 130, 180)  # steelblue
            width = 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw_label(draw, x1, max(0, y1 - 22), alias, color)

    draw_label(draw, 12, 12, f"visible projected objects: {visible_count}", (0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def parse_frame_ids(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Project filtered ScanNet objects onto selected extracted frames.")
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--extract_dir", type=Path, required=True)
    parser.add_argument("--alias_csv", type=Path, required=True)
    parser.add_argument("--frames", type=str, required=True, help="Comma-separated saved frame ids, e.g. 000006,000010,000013")
    parser.add_argument("--highlight_aliases", type=str, default="", help="Comma-separated aliases to highlight")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    extract_dir = args.extract_dir.resolve()
    alias_csv = args.alias_csv.resolve()
    out_dir = args.out_dir.resolve()

    intrinsic_txt = extract_dir / "meta" / "intrinsic_color.txt"
    color_dir = extract_dir / "color"
    pose_dir = extract_dir / "pose"

    objects = build_object_points(scene_dir=scene_dir, alias_csv=alias_csv)

    frame_ids = parse_frame_ids(args.frames)
    highlight_aliases = set(parse_frame_ids(args.highlight_aliases)) if args.highlight_aliases else set()

    print(f"Scene: {scene_dir.name}")
    print(f"Objects available from alias map: {len(objects)}")
    print(f"Frames to render: {frame_ids}")
    if highlight_aliases:
        print(f"Highlight aliases: {sorted(highlight_aliases)}")

    for frame_id in frame_ids:
        image_path = color_dir / f"{frame_id}.jpg"
        pose_path = pose_dir / f"{frame_id}.txt"
        out_path = out_dir / f"{frame_id}_projected.png"

        if not image_path.exists():
            print(f"[WARN] Missing frame image: {image_path}")
            continue
        if not pose_path.exists():
            print(f"[WARN] Missing pose file: {pose_path}")
            continue

        render_frame(
            image_path=image_path,
            pose_path=pose_path,
            objects=objects,
            intrinsic_txt=intrinsic_txt,
            out_path=out_path,
            highlight_aliases=highlight_aliases,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()