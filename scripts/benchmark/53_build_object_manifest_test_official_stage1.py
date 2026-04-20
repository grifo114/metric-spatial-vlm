from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
SCANS_DIR = ROOT / "data" / "scannet" / "scans"
LABEL_MAP_PATH = ROOT / "configs" / "label_map.yaml"
SCENES_TEST_PATH = ROOT / "benchmark" / "scenes_test_official_stage1.txt"

OUT_CSV = ROOT / "benchmark" / "objects_manifest_test_official_stage1.csv"
OUT_PARQUET = ROOT / "benchmark" / "objects_manifest_test_official_stage1.parquet"
POINTS_OUT_DIR = ROOT / "artifacts" / "object_points_test_official_stage1"

MIN_POINTS_DEFAULT = 150
MIN_POINTS_MONITOR = 80


def load_label_config(path: Path) -> Tuple[set[str], Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    allowed = set(data.get("allowed_categories", []))
    label_map = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in data.get("label_map", {}).items()
    }
    return allowed, label_map


def normalize_label(raw_label: str, allowed: set[str], label_map: Dict[str, str]) -> Optional[str]:
    raw = raw_label.strip().lower()
    if raw in label_map:
        return label_map[raw]
    if raw in allowed:
        return raw
    for k, v in label_map.items():
        if k in raw:
            return v
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_single_file(scene_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(scene_dir.glob(pattern))
    return matches[0] if matches else None


def load_scene_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_vertices_xyz(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def object_is_valid(label_norm: Optional[str], n_points: int, extent: np.ndarray) -> Tuple[bool, str]:
    if label_norm is None:
        return False, "label_not_in_benchmark"

    min_points = MIN_POINTS_MONITOR if label_norm == "monitor" else MIN_POINTS_DEFAULT
    if n_points < min_points:
        return False, "too_few_points"

    if np.any(extent <= 1e-4):
        return False, "degenerate_bbox"

    if np.linalg.norm(extent) < 0.10:
        return False, "too_small_extent"

    return True, ""


def main():
    allowed, label_map = load_label_config(LABEL_MAP_PATH)
    scene_ids = load_scene_ids(SCENES_TEST_PATH)

    rows = []
    POINTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scene_id in scene_ids:
        scene_dir = SCANS_DIR / scene_id

        aggregation_path = find_single_file(scene_dir, "*.aggregation.json")
        segs_path = find_single_file(scene_dir, "*segs.json")
        ply_path = find_single_file(scene_dir, "*vh_clean_2.ply")

        if aggregation_path is None or segs_path is None or ply_path is None:
            print(f"[WARN] Missing files for {scene_id}")
            continue

        aggregation = load_json(aggregation_path)
        segs = load_json(segs_path)
        xyz = load_vertices_xyz(ply_path)

        seg_indices = segs["segIndices"]

        scene_out_dir = POINTS_OUT_DIR / scene_id
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        for idx, obj in enumerate(aggregation["segGroups"]):
            instance_id_raw = obj.get("id", idx)
            raw_label = str(obj.get("label", "")).strip().lower()
            label_norm = normalize_label(raw_label, allowed, label_map)

            object_segments = set(obj.get("segments", []))
            point_mask = np.array([seg_id in object_segments for seg_id in seg_indices], dtype=bool)
            obj_points = xyz[point_mask]

            n_points = int(obj_points.shape[0])
            if n_points == 0:
                continue

            centroid = obj_points.mean(axis=0)
            aabb_min = obj_points.min(axis=0)
            aabb_max = obj_points.max(axis=0)
            extent = aabb_max - aabb_min
            diag_3d = float(np.linalg.norm(extent))

            is_valid_object, invalid_reason = object_is_valid(label_norm, n_points, extent)

            object_id = f"{scene_id}__{label_norm or 'unknown'}_{int(instance_id_raw):03d}"
            points_path = scene_out_dir / f"{object_id}.npz"
            np.savez_compressed(points_path, points=obj_points)

            quality_score = 0.0
            quality_score += 1.0 if label_norm is not None else 0.0
            quality_score += 1.0 if n_points >= (MIN_POINTS_MONITOR if label_norm == "monitor" else MIN_POINTS_DEFAULT) else 0.0
            quality_score += 1.0 if np.all(extent > 1e-4) else 0.0
            quality_score /= 3.0

            rows.append({
                "scene_id": scene_id,
                "split": "test_official_stage1",
                "source_dataset": "scannet",
                "object_id": object_id,
                "instance_id_raw": instance_id_raw,
                "label_raw": raw_label,
                "label_norm": label_norm,
                "is_valid_object": is_valid_object,
                "invalid_reason": invalid_reason,
                "n_points": n_points,
                "quality_score": quality_score,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "aabb_min_x": float(aabb_min[0]),
                "aabb_min_y": float(aabb_min[1]),
                "aabb_min_z": float(aabb_min[2]),
                "aabb_max_x": float(aabb_max[0]),
                "aabb_max_y": float(aabb_max[1]),
                "aabb_max_z": float(aabb_max[2]),
                "extent_x": float(extent[0]),
                "extent_y": float(extent[1]),
                "extent_z": float(extent[2]),
                "diag_3d": diag_3d,
                "centroid_xy_x": float(centroid[0]),
                "centroid_xy_y": float(centroid[1]),
                "points_path": str(points_path.relative_to(ROOT)),
            })

    df = pd.DataFrame(rows).sort_values(["scene_id", "label_norm", "object_id"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"Saved parquet: {OUT_PARQUET}")
    except Exception as e:
        print(f"Parquet not saved: {e}")

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Total objects: {len(df)}")
    print(f"Valid objects: {int(df['is_valid_object'].sum())}")

    if len(df) > 0:
        summary = (
            df[df["is_valid_object"]]
            .groupby(["scene_id", "label_norm"])
            .size()
            .unstack(fill_value=0)
        )
        print("\nValid objects by scene/category:")
        print(summary.to_string())


if __name__ == "__main__":
    main()