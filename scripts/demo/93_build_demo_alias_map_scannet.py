from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES = ["chair", "table", "desk", "cabinet", "door", "monitor"]


def load_aggregation(agg_path: Path) -> dict[str, Any]:
    if not agg_path.exists():
        raise FileNotFoundError(f"Aggregation file not found: {agg_path}")

    with agg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "segGroups" not in data:
        raise ValueError(f"Invalid aggregation file: missing 'segGroups' in {agg_path}")

    return data


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def build_alias_rows(scene_id: str, seg_groups: list[dict[str, Any]], allowed_labels: set[str]) -> list[dict[str, Any]]:
    counters: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []

    # Ordena por objectId para deixar estável e reproduzível
    sorted_groups = sorted(seg_groups, key=lambda g: int(g.get("objectId", 10**9)))

    for group in sorted_groups:
        raw_label = str(group.get("label", "")).strip()
        label_norm = normalize_label(raw_label)

        if label_norm not in allowed_labels:
            continue

        object_id = int(group.get("objectId", -1))
        n_segments = len(group.get("segments", []))

        counters[label_norm] += 1
        alias = f"{label_norm.replace(' ', '_')}{counters[label_norm]}"

        rows.append(
            {
                "scene_id": scene_id,
                "alias": alias,
                "label_norm": label_norm,
                "raw_label": raw_label,
                "raw_object_id": object_id,
                "scannet_object_key": f"{scene_id}__{label_norm.replace(' ', '_')}_{object_id:03d}",
                "n_segments": n_segments,
            }
        )

    return rows


def save_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scene_id",
        "alias",
        "label_norm",
        "raw_label",
        "raw_object_id",
        "scannet_object_key",
        "n_segments",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filtered alias map from a ScanNet aggregation file.")
    parser.add_argument(
        "--scene_dir",
        type=Path,
        required=True,
        help="Path to the ScanNet scene directory, e.g. data/scannet/scans/scene0207_00",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="List of categories to keep. Default: chair table desk cabinet door monitor",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, saves inside the scene directory.",
    )
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    scene_id = scene_dir.name
    agg_path = scene_dir / f"{scene_id}.aggregation.json"

    allowed_labels = {normalize_label(x) for x in args.categories}
    data = load_aggregation(agg_path)
    seg_groups = data["segGroups"]

    rows = build_alias_rows(scene_id, seg_groups, allowed_labels)

    if args.out_csv is None:
        out_csv = scene_dir / f"{scene_id}_demo_alias_map.csv"
    else:
        out_csv = args.out_csv.resolve()

    save_csv(rows, out_csv)

    print(f"Scene: {scene_id}")
    print(f"Aggregation: {agg_path}")
    print(f"Saved alias map: {out_csv}")
    print(f"Filtered categories: {', '.join(sorted(allowed_labels))}")
    print(f"Rows saved: {len(rows)}")

    counts = Counter(row["label_norm"] for row in rows)
    if counts:
        print("\nCounts by label:")
        for label, count in counts.most_common():
            print(f"  {label}: {count}")

    if rows:
        print("\nPreview:")
        preview_rows = rows[:20]
        for row in preview_rows:
            print(
                f"  {row['alias']:10s} | "
                f"{row['label_norm']:8s} | "
                f"raw_object_id={row['raw_object_id']:3d} | "
                f"segments={row['n_segments']:3d}"
            )
    else:
        print("\nNo objects matched the selected categories.")


if __name__ == "__main__":
    main()