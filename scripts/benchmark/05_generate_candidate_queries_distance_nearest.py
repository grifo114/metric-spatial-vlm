from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry.geometry_ops import (
    load_points_npz,
    centroid_distance,
    surface_distance,
    nearest_object_by_surface,
)

MANIFEST = ROOT / "benchmark" / "objects_manifest_dev.csv"
CONFIG = ROOT / "configs" / "benchmark_config.yaml"
OUT_PATH = ROOT / "benchmark" / "queries_dev_distance_nearest_candidates.jsonl"


FURNITURE = {"chair", "sofa", "bed"}
SURFACES = {"table", "desk"}
REFERENCES = {"door", "cabinet", "monitor"}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_points_cache(df_scene: pd.DataFrame) -> Dict[str, object]:
    cache = {}
    for _, row in df_scene.iterrows():
        object_id = row["object_id"]
        points_path = ROOT / row["points_path"]
        cache[object_id] = load_points_npz(points_path)
    return cache


def make_distance_candidates(
    scene_id: str,
    df_scene: pd.DataFrame,
    points_cache: Dict[str, object],
    cfg: dict,
) -> List[dict]:
    min_d = float(cfg["distance"]["min_surface_distance_m"])
    max_d = float(cfg["distance"]["max_surface_distance_m"])
    max_q = int(cfg["query_generation"]["max_queries_per_scene_per_operator"])

    rows = df_scene.to_dict(orient="records")
    candidates = []

    for a, b in combinations(rows, 2):
        label_a = a["label_norm"]
        label_b = b["label_norm"]

        # evitar pares da mesma categoria no piloto inicial
        if label_a == label_b:
            continue

        # preferir pares semanticamente mais úteis
        useful_pair = (
            (label_a in FURNITURE and label_b in SURFACES.union(REFERENCES))
            or
            (label_b in FURNITURE and label_a in SURFACES.union(REFERENCES))
            or
            (label_a in SURFACES and label_b in REFERENCES)
            or
            (label_b in SURFACES and label_a in REFERENCES)
        )

        if not useful_pair:
            continue

        pts_a = points_cache[a["object_id"]]
        pts_b = points_cache[b["object_id"]]

        d_surf = surface_distance(pts_a, pts_b)
        if d_surf < min_d or d_surf > max_d:
            continue

        d_cent = centroid_distance(pts_a, pts_b)

        candidates.append({
            "scene_id": scene_id,
            "operator": "distance",
            "object_a": a["object_id"],
            "label_a": label_a,
            "object_b": b["object_id"],
            "label_b": label_b,
            "surface_distance_m": round(float(d_surf), 6),
            "centroid_distance_m": round(float(d_cent), 6),
            "structured_query": f'distance("{a["object_id"]}", "{b["object_id"]}")',
            "natural_query": f'qual a distância entre {a["object_id"]} e {b["object_id"]}?',
            "status": "candidate",
        })

    # ordenar por menor distância de superfície
    candidates = sorted(candidates, key=lambda x: x["surface_distance_m"])

    # pegar até max_q, tentando diversidade de labels
    selected = []
    used_label_pairs = set()
    for c in candidates:
        pair = tuple(sorted([c["label_a"], c["label_b"]]))
        if pair not in used_label_pairs:
            selected.append(c)
            used_label_pairs.add(pair)
        if len(selected) >= max_q:
            break

    # fallback se a diversidade não bastar
    if len(selected) < max_q:
        used_ids = {(c["object_a"], c["object_b"]) for c in selected}
        for c in candidates:
            pair_ids = (c["object_a"], c["object_b"])
            if pair_ids not in used_ids:
                selected.append(c)
                used_ids.add(pair_ids)
            if len(selected) >= max_q:
                break

    return selected


def make_nearest_candidates(
    scene_id: str,
    df_scene: pd.DataFrame,
    points_cache: Dict[str, object],
    cfg: dict,
) -> List[dict]:
    margin = float(cfg["nearest"]["ambiguity_margin_m"])
    max_q = int(cfg["query_generation"]["max_queries_per_scene_per_operator"])

    rows = df_scene.to_dict(orient="records")
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label_norm"], []).append(r)

    candidates = []

    # referências úteis para nearest
    ref_pool = [r for r in rows if r["label_norm"] in SURFACES.union(REFERENCES)]
    target_categories = ["chair", "sofa", "bed"]

    for ref in ref_pool:
        ref_points = points_cache[ref["object_id"]]

        for target_label in target_categories:
            target_rows = by_label.get(target_label, [])
            if len(target_rows) < 2:
                continue

            candidate_dict = {
                r["object_id"]: points_cache[r["object_id"]]
                for r in target_rows
                if r["object_id"] != ref["object_id"]
            }

            if len(candidate_dict) < 2:
                continue

            # distâncias para todos os candidatos
            dists = []
            for object_id, pts in candidate_dict.items():
                d = surface_distance(ref_points, pts)
                dists.append((object_id, float(d)))

            dists = sorted(dists, key=lambda x: x[1])

            best_id, best_d = dists[0]
            second_id, second_d = dists[1]

            # excluir casos ambíguos
            if abs(second_d - best_d) < margin:
                continue

            candidates.append({
                "scene_id": scene_id,
                "operator": "nearest",
                "reference_object": ref["object_id"],
                "reference_label": ref["label_norm"],
                "target_category": target_label,
                "answer_object": best_id,
                "answer_distance_m": round(best_d, 6),
                "second_best_object": second_id,
                "second_best_distance_m": round(second_d, 6),
                "margin_to_second_m": round(second_d - best_d, 6),
                "structured_query": f'nearest("{ref["object_id"]}", "{target_label}")',
                "natural_query": f'qual objeto da categoria {target_label} está mais próximo de {ref["object_id"]}?',
                "status": "candidate",
            })

    # ordenar por margem decrescente: queries menos ambíguas primeiro
    candidates = sorted(candidates, key=lambda x: x["margin_to_second_m"], reverse=True)

    selected = []
    used_ref_target = set()
    for c in candidates:
        key = (c["reference_object"], c["target_category"])
        if key not in used_ref_target:
            selected.append(c)
            used_ref_target.add(key)
        if len(selected) >= max_q:
            break

    return selected


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["is_valid_object"] == True].copy()
    cfg = load_config(CONFIG)

    all_queries = []

    for scene_id, df_scene in df.groupby("scene_id"):
        points_cache = load_points_cache(df_scene)

        distance_queries = make_distance_candidates(scene_id, df_scene, points_cache, cfg)
        nearest_queries = make_nearest_candidates(scene_id, df_scene, points_cache, cfg)

        all_queries.extend(distance_queries)
        all_queries.extend(nearest_queries)

        print(f"{scene_id}: distance={len(distance_queries)}, nearest={len(nearest_queries)}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved: {OUT_PATH}")
    print(f"Total candidate queries: {len(all_queries)}")


if __name__ == "__main__":
    main()