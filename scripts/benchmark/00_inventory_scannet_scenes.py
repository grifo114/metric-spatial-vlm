from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCANS_DIR = ROOT / "data" / "scannet" / "scans"
LABEL_MAP_PATH = ROOT / "configs" / "label_map.yaml"
OUT_CSV = ROOT / "benchmark" / "scene_inventory.csv"
OUT_PARQUET = ROOT / "benchmark" / "scene_inventory.parquet"


BENCHMARK_CATEGORIES = [
    "chair",
    "table",
    "desk",
    "sofa",
    "bed",
    "door",
    "cabinet",
    "monitor",
]

MIN_POINTS_DEFAULT = 150
MIN_POINTS_MONITOR = 80


def load_label_config(path: Path) -> Tuple[set[str], Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    allowed = set(data.get("allowed_categories", []))
    label_map = {str(k).strip().lower(): str(v).strip().lower() for k, v in data.get("label_map", {}).items()}
    return allowed, label_map


def normalize_label(raw_label: str, allowed: set[str], label_map: Dict[str, str]) -> Optional[str]:
    raw = raw_label.strip().lower()

    if raw in label_map:
        return label_map[raw]

    if raw in allowed:
        return raw

    # fallback leve por substring, sem exagero
    for k, v in label_map.items():
        if k in raw:
            return v

    return None


def find_single_file(scene_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(scene_dir.glob(pattern))
    if not matches:
        return None
    return matches[0]


def load_scene_type(scene_dir: Path, scene_id: str) -> str:
    txt_file = scene_dir / f"{scene_id}.txt"
    if not txt_file.exists():
        return "unknown"

    try:
        content = txt_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "unknown"

    for line in content.splitlines():
        line = line.strip()
        if line.lower().startswith("scenetype"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().lower()

    return "unknown"


def guess_room_group(scene_type: str) -> str:
    s = scene_type.lower()

    if any(k in s for k in ["bedroom", "guest room", "hotel room"]):
        return "bedroom"

    if any(k in s for k in ["living room", "lounge", "family room"]):
        return "living_room"

    if any(k in s for k in ["office", "study", "conference room", "meeting room", "computer lab"]):
        return "office_study"

    return "mixed_other"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_segment_point_counts(seg_indices: List[int]) -> Counter:
    return Counter(seg_indices)


def instance_point_count(segments: List[int], seg_point_counts: Counter) -> int:
    return sum(seg_point_counts.get(seg_id, 0) for seg_id in segments)


def object_is_valid(label_norm: Optional[str], n_points: int) -> bool:
    if label_norm is None:
        return False

    if label_norm == "monitor":
        return n_points >= MIN_POINTS_MONITOR

    return n_points >= MIN_POINTS_DEFAULT


def scene_candidate_flags(valid_counts: Dict[str, int], n_valid_objects: int, n_valid_categories: int) -> Dict[str, bool]:
    has_surface = (valid_counts.get("table", 0) + valid_counts.get("desk", 0)) >= 1
    has_furniture = (valid_counts.get("chair", 0) + valid_counts.get("sofa", 0) + valid_counts.get("bed", 0)) >= 1
    has_reference = (valid_counts.get("door", 0) + valid_counts.get("cabinet", 0) + valid_counts.get("monitor", 0)) >= 1

    has_required_categories = has_surface and has_furniture and has_reference
    is_candidate = (n_valid_objects >= 8) and (n_valid_categories >= 3) and has_required_categories

    return {
        "has_surface": has_surface,
        "has_furniture": has_furniture,
        "has_reference": has_reference,
        "has_required_categories": has_required_categories,
        "is_candidate": is_candidate,
    }


def inventory_scene(scene_dir: Path, allowed: set[str], label_map: Dict[str, str]) -> Dict[str, Any]:
    scene_id = scene_dir.name

    aggregation_path = find_single_file(scene_dir, "*.aggregation.json")
    segs_path = find_single_file(scene_dir, "*segs.json")

    if aggregation_path is None or segs_path is None:
        return {
            "scene_id": scene_id,
            "status": "missing_files",
            "scene_type": "unknown",
            "room_type_guess": "mixed_other",
            "n_objects_total": 0,
            "n_valid_objects": 0,
            "n_categories_total": 0,
            "n_valid_categories": 0,
            **{f"n_{c}": 0 for c in BENCHMARK_CATEGORIES},
            **{f"n_valid_{c}": 0 for c in BENCHMARK_CATEGORIES},
            "has_surface": False,
            "has_furniture": False,
            "has_reference": False,
            "has_required_categories": False,
            "is_candidate": False,
            "notes": "aggregation or segs file missing",
        }

    aggregation = load_json(aggregation_path)
    segs = load_json(segs_path)

    seg_indices = segs.get("segIndices", [])
    seg_point_counts = build_segment_point_counts(seg_indices)

    seg_groups = aggregation.get("segGroups", [])

    scene_type = load_scene_type(scene_dir, scene_id)
    room_type_guess = guess_room_group(scene_type)

    raw_label_counter = Counter()
    valid_label_counter = Counter()

    n_objects_total = 0
    n_valid_objects = 0

    for obj in seg_groups:
        raw_label = str(obj.get("label", "")).strip().lower()
        segments = obj.get("segments", [])

        n_points = instance_point_count(segments, seg_point_counts)
        label_norm = normalize_label(raw_label, allowed, label_map)

        n_objects_total += 1

        if label_norm is not None:
            raw_label_counter[label_norm] += 1

        if object_is_valid(label_norm, n_points):
            n_valid_objects += 1
            valid_label_counter[label_norm] += 1

    n_categories_total = sum(1 for c in BENCHMARK_CATEGORIES if raw_label_counter.get(c, 0) > 0)
    n_valid_categories = sum(1 for c in BENCHMARK_CATEGORIES if valid_label_counter.get(c, 0) > 0)

    flags = scene_candidate_flags(valid_label_counter, n_valid_objects, n_valid_categories)

    row = {
        "scene_id": scene_id,
        "status": "ok",
        "scene_type": scene_type,
        "room_type_guess": room_type_guess,
        "n_objects_total": n_objects_total,
        "n_valid_objects": n_valid_objects,
        "n_categories_total": n_categories_total,
        "n_valid_categories": n_valid_categories,
        **{f"n_{c}": raw_label_counter.get(c, 0) for c in BENCHMARK_CATEGORIES},
        **{f"n_valid_{c}": valid_label_counter.get(c, 0) for c in BENCHMARK_CATEGORIES},
        **flags,
        "notes": "",
    }

    return row


def main() -> None:
    if not SCANS_DIR.exists():
        raise FileNotFoundError(f"ScanNet scans directory not found: {SCANS_DIR}")

    allowed, label_map = load_label_config(LABEL_MAP_PATH)

    scene_dirs = sorted([p for p in SCANS_DIR.iterdir() if p.is_dir() and p.name.startswith("scene")])
    if not scene_dirs:
        raise RuntimeError(f"No ScanNet scene directories found in {SCANS_DIR}")

    rows = []
    for scene_dir in scene_dirs:
        try:
            row = inventory_scene(scene_dir, allowed, label_map)
        except Exception as e:
            row = {
                "scene_id": scene_dir.name,
                "status": "error",
                "scene_type": "unknown",
                "room_type_guess": "mixed_other",
                "n_objects_total": 0,
                "n_valid_objects": 0,
                "n_categories_total": 0,
                "n_valid_categories": 0,
                **{f"n_{c}": 0 for c in BENCHMARK_CATEGORIES},
                **{f"n_valid_{c}": 0 for c in BENCHMARK_CATEGORIES},
                "has_surface": False,
                "has_furniture": False,
                "has_reference": False,
                "has_required_categories": False,
                "is_candidate": False,
                "notes": f"{type(e).__name__}: {e}",
            }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["is_candidate", "n_valid_objects", "n_valid_categories"], ascending=[False, False, False])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    try:
        df.to_parquet(OUT_PARQUET, index=False)
        parquet_msg = f"Saved parquet: {OUT_PARQUET}"
    except Exception as e:
        parquet_msg = f"Parquet not saved ({type(e).__name__}: {e})"

    print(f"Total scenes scanned: {len(df)}")
    print(f"Candidate scenes: {int(df['is_candidate'].sum())}")
    print(f"CSV saved: {OUT_CSV}")
    print(parquet_msg)

    print("\nTop 15 candidate scenes:")
    cols = [
        "scene_id",
        "room_type_guess",
        "n_valid_objects",
        "n_valid_categories",
        "n_valid_chair",
        "n_valid_table",
        "n_valid_desk",
        "n_valid_sofa",
        "n_valid_bed",
        "n_valid_door",
        "n_valid_cabinet",
        "n_valid_monitor",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    print(df[df["is_candidate"]][existing_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()