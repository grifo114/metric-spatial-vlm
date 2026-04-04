"""
02_select_pairs.py — v2

Selects object pairs for evaluation from gt_centroids.json.

Changes from v1:
  - Same-label pairs now allowed (e.g. chair vs chair)
  - MIN_POINTS lowered from 50 to 20
  - Falls back across ranges if a range is exhausted
"""

import os, json, random, itertools
import numpy as np
import pandas as pd

INPUT_FILE  = "results/gt_centroids.json"
OUTPUT_DIR  = "results"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "selected_pairs.json")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "selected_pairs.csv")

SCENES = [
    "scene0000_00", "scene0001_00", "scene0010_00", "scene0011_00",
    "scene0015_00", "scene0019_00", "scene0030_00", "scene0045_00",
    "scene0050_00", "scene0062_00", "scene0077_00", "scene0086_00",
    "scene0100_00", "scene0114_00", "scene0139_00", "scene0153_00",
    "scene0164_00", "scene0181_00", "scene0207_00", "scene0222_00",
]

N_SHORT  = 5
N_MEDIUM = 3
N_LONG   = 2
MIN_POINTS = 20

random.seed(42)

def euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def range_label(d):
    if d < 1.5:   return "short"
    if d < 3.0:   return "medium"
    return "long"

def select_pairs_for_scene(scene_id, objects):
    valid = [o for o in objects
             if o["in_target"] and o["n_points"] >= MIN_POINTS]

    if len(valid) < 2:
        return []

    candidates = {"short": [], "medium": [], "long": []}

    for a, b in itertools.combinations(valid, 2):
        d   = euclidean(a["centroid"], b["centroid"])
        rng = range_label(d)
        candidates[rng].append((a, b, d))

    for rng in candidates:
        random.shuffle(candidates[rng])

    # Track used object pairs to avoid duplicates
    used = set()
    selected = []

    def pick(rng, quota):
        count = 0
        for a, b, d in candidates[rng]:
            if count >= quota:
                break
            key = (min(a["object_id"], b["object_id"]),
                   max(a["object_id"], b["object_id"]))
            if key in used:
                continue
            used.add(key)
            selected.append({
                "scene_id":      scene_id,
                "range":         rng,
                "gt_distance_m": round(d, 4),
                "obj_a": {
                    "object_id": a["object_id"],
                    "label":     a["label"],
                    "centroid":  a["centroid"],
                    "n_points":  a["n_points"],
                },
                "obj_b": {
                    "object_id": b["object_id"],
                    "label":     b["label"],
                    "centroid":  b["centroid"],
                    "n_points":  b["n_points"],
                },
                "query": (
                    f"What is the distance between the {a['label']} "
                    f"(object {a['object_id']}) and the {b['label']} "
                    f"(object {b['object_id']})?"
                ),
            })
            count += 1

    pick("short",  N_SHORT)
    pick("medium", N_MEDIUM)
    pick("long",   N_LONG)

    return selected

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r") as f:
        gt_data = json.load(f)

    all_pairs   = {}
    total_pairs = 0
    skipped     = []

    print(f"{'Scene':<16} {'Short':>6} {'Medium':>7} {'Long':>5} {'Total':>6}")
    print("─" * 46)

    for scene_id in SCENES:
        objects = gt_data.get(scene_id, [])
        pairs   = select_pairs_for_scene(scene_id, objects)
        all_pairs[scene_id] = pairs
        total_pairs += len(pairs)

        n_s = sum(1 for p in pairs if p["range"] == "short")
        n_m = sum(1 for p in pairs if p["range"] == "medium")
        n_l = sum(1 for p in pairs if p["range"] == "long")

        target = N_SHORT + N_MEDIUM + N_LONG
        flag = "" if len(pairs) == target else " ⚠"
        print(f"{scene_id:<16} {n_s:>6} {n_m:>7} {n_l:>5} {len(pairs):>6}{flag}")

        if len(pairs) < target:
            skipped.append(scene_id)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_pairs, f, indent=2)

    rows = []
    for scene_id, pairs in all_pairs.items():
        for p in pairs:
            rows.append({
                "scene_id":      p["scene_id"],
                "range":         p["range"],
                "gt_distance_m": p["gt_distance_m"],
                "label_a":       p["obj_a"]["label"],
                "object_id_a":   p["obj_a"]["object_id"],
                "label_b":       p["obj_b"]["label"],
                "object_id_b":   p["obj_b"]["object_id"],
                "cx_a": p["obj_a"]["centroid"][0],
                "cy_a": p["obj_a"]["centroid"][1],
                "cz_a": p["obj_a"]["centroid"][2],
                "cx_b": p["obj_b"]["centroid"][0],
                "cy_b": p["obj_b"]["centroid"][1],
                "cz_b": p["obj_b"]["centroid"][2],
                "query": p["query"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("─" * 46)
    print(f"{'TOTAL':<16} "
          f"{df[df.range=='short'].shape[0]:>6} "
          f"{df[df.range=='medium'].shape[0]:>7} "
          f"{df[df.range=='long'].shape[0]:>5} "
          f"{total_pairs:>6}")

    if skipped:
        print(f"\n⚠  Scenes with fewer pairs than ideal: {skipped}")

    print(f"\nOutput: {OUTPUT_JSON}")
    print(f"        {OUTPUT_CSV}")
    print(f"\nDistance distribution:")
    print(df.groupby("range")["gt_distance_m"].describe().round(3))

if __name__ == "__main__":
    main()
