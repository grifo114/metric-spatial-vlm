"""
01_extract_gt_centroids.py

Extracts ground-truth 3D centroids for each object instance in each
ScanNet scene, using the mesh (.ply), segmentation (.segs.json) and
aggregation (.aggregation.json) files.

Output: results/gt_centroids.json
"""

import os
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────
SCANNET_DIR = "data/scannet/scans"
OUTPUT_DIR  = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gt_centroids.json")

SCENES = [
    "scene0000_00", "scene0001_00", "scene0010_00", "scene0011_00",
    "scene0015_00", "scene0019_00", "scene0030_00", "scene0045_00",
    "scene0050_00", "scene0062_00", "scene0077_00", "scene0086_00",
    "scene0100_00", "scene0114_00", "scene0139_00", "scene0153_00",
    "scene0164_00", "scene0181_00", "scene0207_00", "scene0222_00",
]

TARGET_CATEGORIES = {
    "chair", "table", "desk", "sofa", "bed", "monitor", "door",
    "window", "bookshelf", "lamp", "cabinet", "sink", "toilet",
    "bathtub", "counter", "refrigerator", "television", "curtain",
    "pillow", "nightstand"
}

def load_mesh_vertices(ply_path):
    mesh = o3d.io.read_point_cloud(ply_path)
    return np.asarray(mesh.points)

def load_segmentation(segs_path):
    with open(segs_path, "r") as f:
        data = json.load(f)
    seg_to_verts = {}
    for vertex_idx, seg_id in enumerate(data["segIndices"]):
        seg_to_verts.setdefault(seg_id, []).append(vertex_idx)
    return seg_to_verts

def load_aggregation(agg_path):
    with open(agg_path, "r") as f:
        data = json.load(f)
    return [
        {"object_id": g["objectId"], "label": g["label"], "segments": g["segments"]}
        for g in data["segGroups"]
    ]

def compute_centroid(vertices, seg_to_verts, segments):
    all_points = []
    for seg_id in segments:
        if seg_id in seg_to_verts:
            all_points.append(vertices[seg_to_verts[seg_id]])
    if not all_points:
        return None
    return np.vstack(all_points).mean(axis=0)

def process_scene(scene_id):
    scene_dir = os.path.join(SCANNET_DIR, scene_id)
    ply_path  = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")
    segs_path = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    agg_path  = os.path.join(scene_dir, f"{scene_id}.aggregation.json")

    for path in [ply_path, segs_path, agg_path]:
        if not os.path.isfile(path):
            print(f"  [WARN] Missing: {path}")
            return []

    vertices     = load_mesh_vertices(ply_path)
    seg_to_verts = load_segmentation(segs_path)
    instances    = load_aggregation(agg_path)

    objects = []
    for inst in instances:
        label    = inst["label"].lower().strip()
        n_points = sum(len(seg_to_verts.get(s, [])) for s in inst["segments"])
        centroid = compute_centroid(vertices, seg_to_verts, inst["segments"])
        if centroid is None:
            continue
        objects.append({
            "object_id": inst["object_id"],
            "label":     label,
            "centroid":  centroid.tolist(),
            "n_points":  n_points,
            "in_target": label in TARGET_CATEGORIES,
        })
    return objects

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    total_objects = 0
    total_target  = 0

    print(f"Processing {len(SCENES)} scenes...\n")
    for scene_id in tqdm(SCENES, desc="Scenes"):
        objects = process_scene(scene_id)
        results[scene_id] = objects
        total_objects += len(objects)
        total_target  += sum(1 for o in objects if o["in_target"])
        tqdm.write(
            f"  {scene_id}: {len(objects)} objects "
            f"({sum(1 for o in objects if o['in_target'])} in target categories)"
        )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"Scenes processed : {len(SCENES)}")
    print(f"Total objects    : {total_objects}")
    print(f"Target objects   : {total_target}")
    print(f"Output saved to  : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
