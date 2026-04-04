"""
06_proposed_method.py  v2

Proposed method: 3D centroid-based distance estimation.

Two modes:
  GT   — centroids from annotated mesh (theoretical upper bound)
  RGBD — centroids from RGBD point cloud fusion via Open3D TSDF

Fix: resize color frame to depth resolution before TSDF integration.
"""

import os, json, struct, zlib, io
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
from tqdm import tqdm

INPUT_CSV   = "results/selected_pairs.csv"
GT_JSON     = "results/gt_centroids.json"
POSES_DIR   = "data/scannet/poses"
SCANS_DIR   = "data/scannet/scans"
OUTPUT_GT   = "results/predictions_proposed_gt.csv"
OUTPUT_RGBD = "results/predictions_proposed_rgbd.csv"

VOXEL_LENGTH  = 0.02
SDF_TRUNC     = 0.08
DEPTH_MAX     = 6.0
N_FUSE_FRAMES = 30
BBOX_MARGIN   = 0.3

# ── SensReader ────────────────────────────────────────────────────

class SensReader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self._read_header()
        self._index_frames()

    def _read_header(self):
        version = struct.unpack("I", self.f.read(4))[0]
        assert version == 4
        strlen = struct.unpack("Q", self.f.read(8))[0]
        self.f.read(strlen)
        self.f.read(64 * 4)
        self.color_compression = struct.unpack("i", self.f.read(4))[0]
        self.depth_compression = struct.unpack("i", self.f.read(4))[0]
        self.color_width  = struct.unpack("I", self.f.read(4))[0]
        self.color_height = struct.unpack("I", self.f.read(4))[0]
        self.depth_width  = struct.unpack("I", self.f.read(4))[0]
        self.depth_height = struct.unpack("I", self.f.read(4))[0]
        self.depth_shift  = struct.unpack("f", self.f.read(4))[0]
        self.n_frames     = struct.unpack("Q", self.f.read(8))[0]
        self.data_start   = self.f.tell()

    def _index_frames(self):
        self.offsets = []
        self.f.seek(self.data_start)
        for _ in range(self.n_frames):
            self.offsets.append(self.f.tell())
            self.f.read(64 + 16)
            color_size = struct.unpack("Q", self.f.read(8))[0]
            depth_size = struct.unpack("Q", self.f.read(8))[0]
            self.f.read(color_size)
            self.f.read(depth_size)

    def read_frame(self, idx):
        """Returns (color resized to depth res, depth in metres)."""
        self.f.seek(self.offsets[idx])
        self.f.read(64 + 16)
        color_size = struct.unpack("Q", self.f.read(8))[0]
        depth_size = struct.unpack("Q", self.f.read(8))[0]
        color_data = self.f.read(color_size)
        depth_data = self.f.read(depth_size)

        # Color — resize to depth resolution so Open3D sizes match
        color_full = Image.open(io.BytesIO(color_data)).convert("RGB")
        color = np.array(
            color_full.resize(
                (self.depth_width, self.depth_height),
                Image.BILINEAR
            )
        )

        raw      = zlib.decompress(depth_data)
        depth_mm = np.frombuffer(raw, dtype=np.uint16).reshape(
            self.depth_height, self.depth_width
        ).astype(np.float32)
        depth_m = depth_mm / self.depth_shift

        return color, depth_m

    def close(self):
        self.f.close()

# ── Helpers ───────────────────────────────────────────────────────

def load_intrinsics(scene_id):
    with open(os.path.join(POSES_DIR, scene_id, "intrinsics.json")) as f:
        return json.load(f)

def load_all_poses(scene_id):
    pose_dir = os.path.join(POSES_DIR, scene_id)
    poses = {}
    for fname in sorted(os.listdir(pose_dir)):
        if not fname.endswith(".txt"):
            continue
        idx  = int(fname.replace(".txt", ""))
        pose = np.loadtxt(os.path.join(pose_dir, fname), dtype=np.float64)
        if np.isfinite(pose).all() and not np.allclose(pose, 0):
            poses[idx] = pose
    return poses

def project(pt_world, intr, pose_c2w):
    w2c     = np.linalg.inv(pose_c2w)
    pt      = w2c @ np.array([*pt_world, 1.0])
    x, y, z = pt[:3]
    if z <= 0.1:
        return None, None
    u = intr["fx"] * x / z + intr["cx"]
    v = intr["fy"] * y / z + intr["cy"]
    return (u, v), float(z)

def find_best_frame_idx(ca, cb, intr, poses):
    W, H = intr["width"], intr["height"]
    margin = 10
    best_idx   = None
    best_score = -np.inf

    for idx, pose in poses.items():
        (pa, _) = project(ca, intr, pose)
        (pb, _) = project(cb, intr, pose)
        if pa is None or pb is None:
            continue
        ua, va = pa; ub, vb = pb
        if not (margin < ua < W-margin and margin < va < H-margin):
            continue
        if not (margin < ub < W-margin and margin < vb < H-margin):
            continue
        score = -(abs(ua-W/2)+abs(va-H/2)+abs(ub-W/2)+abs(vb-H/2))
        if score > best_score:
            best_score = score
            best_idx   = idx

    return best_idx

def fuse_rgbd_frames(reader, poses, center_idx, intr):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # Use depth resolution for intrinsics (color is resized to match)
    cam_intr = o3d.camera.PinholeCameraIntrinsic(
        width=intr["width"], height=intr["height"],
        fx=intr["fx"], fy=intr["fy"],
        cx=intr["cx"], cy=intr["cy"],
    )

    all_idxs = sorted(poses.keys())
    try:
        pos = all_idxs.index(center_idx)
    except ValueError:
        pos = 0
    half     = N_FUSE_FRAMES // 2
    selected = all_idxs[max(0, pos-half): pos+half+1]

    fused = 0
    for fidx in selected:
        if fidx not in poses:
            continue
        try:
            color, depth = reader.read_frame(fidx)
        except Exception:
            continue

        depth = np.where((depth > 0.1) & (depth < DEPTH_MAX), depth, 0.0)

        color_o3d = o3d.geometry.Image(color.astype(np.uint8))
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=DEPTH_MAX,
            convert_rgb_to_intensity=False,
        )

        extrinsic = np.linalg.inv(poses[fidx])
        volume.integrate(
            rgbd, cam_intr,
            extrinsic
        )
        fused += 1

    if fused == 0:
        return None

    return volume.extract_point_cloud()

def centroid_from_pcd_near_gt(pcd, gt_centroid, margin=BBOX_MARGIN):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return None
    c    = np.array(gt_centroid)
    mask = np.all((pts >= c-margin) & (pts <= c+margin), axis=1)
    crop = pts[mask]
    return crop.mean(axis=0) if len(crop) >= 5 else None

# ── GT Mode ───────────────────────────────────────────────────────

def run_gt_mode(df, gt_data):
    results = []
    for _, row in df.iterrows():
        scene_id = row["scene_id"]
        objs     = {o["object_id"]: o for o in gt_data.get(scene_id, [])}
        a_id     = int(row["object_id_a"])
        b_id     = int(row["object_id_b"])

        if a_id not in objs or b_id not in objs:
            results.append({
                "scene_id": scene_id, "range": row["range"],
                "gt_distance_m": row["gt_distance_m"],
                "pred_distance_m": None,
                "label_a": row["label_a"], "label_b": row["label_b"],
                "method": "proposed_gt", "status": "object_not_found",
            })
            continue

        ca   = np.array(objs[a_id]["centroid"])
        cb   = np.array(objs[b_id]["centroid"])
        pred = round(float(np.linalg.norm(ca - cb)), 4)

        results.append({
            "scene_id": scene_id, "range": row["range"],
            "gt_distance_m": row["gt_distance_m"],
            "pred_distance_m": pred,
            "label_a": row["label_a"], "label_b": row["label_b"],
            "method": "proposed_gt", "status": "ok",
        })
    return pd.DataFrame(results)

# ── RGBD Mode ─────────────────────────────────────────────────────

def run_rgbd_mode(df):
    results      = []
    poses_cache  = {}
    intr_cache   = {}
    reader_cache = {}
    pcd_cache    = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="RGBD pairs"):
        scene_id = row["scene_id"]

        if scene_id not in poses_cache:
            poses_cache[scene_id]  = load_all_poses(scene_id)
            intr_cache[scene_id]   = load_intrinsics(scene_id)
            sens_path = os.path.join(SCANS_DIR, scene_id, f"{scene_id}.sens")
            reader_cache[scene_id] = SensReader(sens_path)

        poses  = poses_cache[scene_id]
        intr   = intr_cache[scene_id]
        reader = reader_cache[scene_id]

        ca_gt = [row["cx_a"], row["cy_a"], row["cz_a"]]
        cb_gt = [row["cx_b"], row["cy_b"], row["cz_b"]]

        best_idx = find_best_frame_idx(ca_gt, cb_gt, intr, poses)

        if best_idx is None:
            results.append({
                "scene_id": scene_id, "range": row["range"],
                "gt_distance_m": row["gt_distance_m"],
                "pred_distance_m": None,
                "label_a": row["label_a"], "label_b": row["label_b"],
                "method": "proposed_rgbd", "status": "no_valid_frame",
            })
            continue

        cache_key = (scene_id, best_idx)
        if cache_key not in pcd_cache:
            pcd = fuse_rgbd_frames(reader, poses, best_idx, intr)
            pcd_cache[cache_key] = pcd
        else:
            pcd = pcd_cache[cache_key]

        if pcd is None or len(np.asarray(pcd.points)) == 0:
            results.append({
                "scene_id": scene_id, "range": row["range"],
                "gt_distance_m": row["gt_distance_m"],
                "pred_distance_m": None,
                "label_a": row["label_a"], "label_b": row["label_b"],
                "method": "proposed_rgbd", "status": "fusion_failed",
            })
            continue

        ca_rgbd = centroid_from_pcd_near_gt(pcd, ca_gt)
        cb_rgbd = centroid_from_pcd_near_gt(pcd, cb_gt)

        if ca_rgbd is None or cb_rgbd is None:
            results.append({
                "scene_id": scene_id, "range": row["range"],
                "gt_distance_m": row["gt_distance_m"],
                "pred_distance_m": None,
                "label_a": row["label_a"], "label_b": row["label_b"],
                "method": "proposed_rgbd", "status": "centroid_failed",
            })
            continue

        pred = round(float(np.linalg.norm(ca_rgbd - cb_rgbd)), 4)

        results.append({
            "scene_id":        scene_id,
            "range":           row["range"],
            "gt_distance_m":   row["gt_distance_m"],
            "pred_distance_m": pred,
            "label_a":         row["label_a"],
            "label_b":         row["label_b"],
            "method":          "proposed_rgbd",
            "status":          "ok",
        })

    for r in reader_cache.values():
        r.close()

    return pd.DataFrame(results)

# ── Summary ───────────────────────────────────────────────────────

def print_summary(out_df, label):
    valid = out_df[out_df["status"] == "ok"].copy()
    valid["abs_error"] = (valid["pred_distance_m"] - valid["gt_distance_m"]).abs()
    valid["rel_error"] = valid["abs_error"] / valid["gt_distance_m"] * 100

    print(f"\n{'─'*50}")
    print(f"{label}")
    print(f"{'─'*50}")
    print(f"Pairs processed : {len(out_df)}")
    print(f"Valid           : {len(valid)}")
    print(f"Failed          : {len(out_df)-len(valid)}")

    if len(valid) > 0:
        print(f"\nMAE             : {valid['abs_error'].mean():.4f} m")
        print(f"Mean Rel. Error : {valid['rel_error'].mean():.2f} %")
        print(f"\nBy range:")
        for rng in ["short", "medium", "long"]:
            sub = valid[valid["range"] == rng]
            if len(sub) == 0:
                continue
            print(f"  {rng:<8}  n={len(sub):>3}  "
                  f"MAE={sub['abs_error'].mean():.3f} m  "
                  f"RelErr={sub['rel_error'].mean():.1f}%")

    print(f"\nStatus breakdown:")
    print(out_df["status"].value_counts().to_string())

# ── Main ─────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    with open(GT_JSON) as f:
        gt_data = json.load(f)

    if "object_id_a" not in df.columns:
        print("ERROR: run script 02 first.")
        return

    print("=" * 50)
    print("Proposed Method — GT Mode")
    print("=" * 50)
    gt_df = run_gt_mode(df, gt_data)
    gt_df.to_csv(OUTPUT_GT, index=False)
    print_summary(gt_df, "Proposed — GT centroids")
    print(f"\nOutput: {OUTPUT_GT}")

    print("\n" + "=" * 50)
    print("Proposed Method — RGBD Mode")
    print("=" * 50)
    rgbd_df = run_rgbd_mode(df)
    rgbd_df.to_csv(OUTPUT_RGBD, index=False)
    print_summary(rgbd_df, "Proposed — RGBD fusion")
    print(f"\nOutput: {OUTPUT_RGBD}")

if __name__ == "__main__":
    main()
