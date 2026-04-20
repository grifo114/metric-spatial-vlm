from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import is_between_xy, is_aligned_xy

MANIFEST = ROOT / "benchmark" / "objects_manifest_dev.csv"
CONFIG = ROOT / "configs" / "benchmark_config.yaml"
OUT_PATH = ROOT / "benchmark" / "queries_dev_between_aligned_candidates.jsonl"

FURNITURE = {"chair", "sofa", "bed"}
SURFACES = {"table", "desk"}
REFERENCES = {"door", "cabinet", "monitor"}
ALL_ALLOWED = FURNITURE | SURFACES | REFERENCES


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def centroid_xyz(row: pd.Series) -> np.ndarray:
    return row[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)


def make_between_candidates(scene_id: str, df_scene: pd.DataFrame, cfg: dict) -> List[dict]:
    tau_between = float(cfg["between"]["tau_between_m"])
    max_q = int(cfg["query_generation"]["max_queries_per_scene_per_operator"])

    furniture_rows = df_scene[df_scene["label_norm"].isin(FURNITURE)].to_dict(orient="records")
    surface_rows = df_scene[df_scene["label_norm"].isin(SURFACES)].to_dict(orient="records")

    candidates = []

    for x in furniture_rows:
        cx = centroid_xyz(pd.Series(x))

        for a, b in combinations(surface_rows, 2):
            ca = centroid_xyz(pd.Series(a))
            cb = centroid_xyz(pd.Series(b))

            if is_between_xy(cx, ca, cb, tau_between=tau_between):
                candidates.append({
                    "scene_id": scene_id,
                    "operator": "between",
                    "object_x": x["object_id"],
                    "label_x": x["label_norm"],
                    "object_a": a["object_id"],
                    "label_a": a["label_norm"],
                    "object_b": b["object_id"],
                    "label_b": b["label_norm"],
                    "tau_between_m": tau_between,
                    "structured_query": f'between("{x["object_id"]}", "{a["object_id"]}", "{b["object_id"]}")',
                    "natural_query": f'{x["object_id"]} está entre {a["object_id"]} e {b["object_id"]}?',
                    "status": "candidate",
                })

    # preferir diversidade de objeto X
    selected = []
    used_x = set()
    for c in candidates:
        if c["object_x"] not in used_x:
            selected.append(c)
            used_x.add(c["object_x"])
        if len(selected) >= max_q:
            break

    if len(selected) < max_q:
        used_queries = {c["structured_query"] for c in selected}
        for c in candidates:
            if c["structured_query"] not in used_queries:
                selected.append(c)
                used_queries.add(c["structured_query"])
            if len(selected) >= max_q:
                break

    return selected


def make_aligned_candidates(scene_id: str, df_scene: pd.DataFrame, cfg: dict) -> List[dict]:
    tau_align = float(cfg["aligned"]["tau_align_m"])
    max_q = int(cfg["query_generation"]["max_queries_per_scene_per_operator"])

    rows = df_scene[df_scene["label_norm"].isin(ALL_ALLOWED)].to_dict(orient="records")
    candidates = []

    for a, b, c in combinations(rows, 3):
        ca = centroid_xyz(pd.Series(a))
        cb = centroid_xyz(pd.Series(b))
        cc = centroid_xyz(pd.Series(c))

        if is_aligned_xy(ca, cb, cc, tau_align=tau_align):
            labels = {a["label_norm"], b["label_norm"], c["label_norm"]}
            mixed_bonus = 1 if len(labels) == 3 else 0

            candidates.append({
                "scene_id": scene_id,
                "operator": "aligned",
                "object_a": a["object_id"],
                "label_a": a["label_norm"],
                "object_b": b["object_id"],
                "label_b": b["label_norm"],
                "object_c": c["object_id"],
                "label_c": c["label_norm"],
                "tau_align_m": tau_align,
                "mixed_bonus": mixed_bonus,
                "structured_query": f'aligned("{a["object_id"]}", "{b["object_id"]}", "{c["object_id"]}")',
                "natural_query": f'{a["object_id"]}, {b["object_id"]} e {c["object_id"]} estão alinhados?',
                "status": "candidate",
            })

    # preferir triplas com categorias diferentes
    candidates = sorted(candidates, key=lambda x: x["mixed_bonus"], reverse=True)

    selected = []
    used_signatures = set()
    for c in candidates:
        sig = tuple(sorted([c["label_a"], c["label_b"], c["label_c"]]))
        if sig not in used_signatures:
            selected.append(c)
            used_signatures.add(sig)
        if len(selected) >= max_q:
            break

    if len(selected) < max_q:
        used_queries = {c["structured_query"] for c in selected}
        for c in candidates:
            if c["structured_query"] not in used_queries:
                selected.append(c)
                used_queries.add(c["structured_query"])
            if len(selected) >= max_q:
                break

    return selected


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["is_valid_object"] == True].copy()
    cfg = load_config(CONFIG)

    all_queries = []

    for scene_id, df_scene in df.groupby("scene_id"):
        between_queries = make_between_candidates(scene_id, df_scene, cfg)
        aligned_queries = make_aligned_candidates(scene_id, df_scene, cfg)

        all_queries.extend(between_queries)
        all_queries.extend(aligned_queries)

        print(f"{scene_id}: between={len(between_queries)}, aligned={len(aligned_queries)}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved: {OUT_PATH}")
    print(f"Total candidate queries: {len(all_queries)}")


if __name__ == "__main__":
    main()