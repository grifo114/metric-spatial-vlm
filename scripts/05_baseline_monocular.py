"""
05_baseline_monocular.py  v4

Baseline B: monocular depth (DPT) with scale calibration from
sensor depth.

Scale calibration strategy:
  - Read the actual depth frame from .sens (metric, millimetres)
  - Sample N random pixels with valid sensor depth
  - Fit linear scale: DPT_pred = scale * sensor_depth_m
  - Apply scale to predict depth at object projected pixels
  - This simulates a monocular system with access to a few sparse
    depth measurements for calibration — but NOT at the object locations

The 54 no_valid_frame failures from Baseline A carry over here.

Input:  results/selected_pairs.csv
        data/scannet/poses/<scene>/
        data/scannet/scans/<scene>/<scene>.sens
Output: results/predictions_baseline_b.csv
"""

import os, json, struct, zlib, io, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import DPTForDepthEstimation, DPTImageProcessor

INPUT_CSV  = "results/selected_pairs.csv"
POSES_DIR  = "data/scannet/poses"
SCANS_DIR  = "data/scannet/scans"
OUTPUT_CSV = "results/predictions_baseline_b.csv"

N_CALIB_PIXELS = 200   # pixels used for scale calibration
random.seed(42)

# ── Device ───────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

MODEL_NAME = "Intel/dpt-hybrid-midas"
print(f"Loading {MODEL_NAME}...")
processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
model     = DPTForDepthEstimation.from_pretrained(MODEL_NAME)
model.to(DEVICE).eval()
print("Model loaded.\n")

# ── SensFrameReader (RGB + Depth) ─────────────────────────────────

class SensFrameReader:
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

    def read_color(self, frame_idx):
        self.f.seek(self.offsets[frame_idx])
        self.f.read(64 + 16)
        color_size = struct.unpack("Q", self.f.read(8))[0]
        self.f.read(8)
        color_data = self.f.read(color_size)
        return np.array(Image.open(io.BytesIO(color_data)).convert("RGB"))

    def read_color_and_depth(self, frame_idx):
        """Returns (color H×W×3 uint8, depth H×W float32 metres)."""
        self.f.seek(self.offsets[frame_idx])
        self.f.read(64 + 16)
        color_size = struct.unpack("Q", self.f.read(8))[0]
        depth_size = struct.unpack("Q", self.f.read(8))[0]
        color_data = self.f.read(color_size)
        depth_data = self.f.read(depth_size)

        color = np.array(Image.open(io.BytesIO(color_data)).convert("RGB"))

        raw   = zlib.decompress(depth_data)
        depth_mm = np.frombuffer(raw, dtype=np.uint16).reshape(
            self.depth_height, self.depth_width
        ).astype(np.float32)
        depth_m = depth_mm / self.depth_shift   # convert to metres

        return color, depth_m

    def close(self):
        self.f.close()

# ── Geometry ─────────────────────────────────────────────────────

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

def project(centroid_world, intr, pose_c2w):
    w2c     = np.linalg.inv(pose_c2w)
    pt      = w2c @ np.array([*centroid_world, 1.0])
    x, y, z = pt[:3]
    if z <= 0.1:
        return None, None
    u = intr["fx"] * x / z + intr["cx"]
    v = intr["fy"] * y / z + intr["cy"]
    return (u, v), float(z)

def find_best_frame(ca, cb, intr, poses):
    W, H   = intr["width"], intr["height"]
    margin = 10
    best   = None
    best_score = -np.inf

    for idx, pose in poses.items():
        (pa, za) = project(ca, intr, pose)
        (pb, zb) = project(cb, intr, pose)
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
            best = (idx, pose, pa, za, pb, zb)

    return best

def run_dpt(color_rgb, target_h, target_w):
    pil_img = Image.fromarray(color_rgb)
    inputs  = processor(images=pil_img, return_tensors="pt")
    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return pred

def sample_patch(arr, u, v, patch=4):
    H, W = arr.shape
    u = int(round(np.clip(u, 0, W-1)))
    v = int(round(np.clip(v, 0, H-1)))
    return float(arr[
        max(0,v-patch):min(H,v+patch),
        max(0,u-patch):min(W,u+patch)
    ].mean())

def fit_scale_from_sensor(dpt_map, sensor_depth_m,
                          exclude_pixels, n_samples=N_CALIB_PIXELS):
    """
    Fit DPT scale using random pixels with valid sensor depth,
    excluding pixels near the object locations (to avoid data leakage).

    Fit model: dpt = scale * sensor_depth  (no intercept, anchored at 0)

    Returns scale (float) or None.
    """
    H, W = dpt_map.shape
    exclude_set = set()
    for (u, v) in exclude_pixels:
        for du in range(-20, 21):
            for dv in range(-20, 21):
                eu = int(round(u)) + du
                ev = int(round(v)) + dv
                if 0 <= eu < W and 0 <= ev < H:
                    exclude_set.add((eu, ev))

    # Collect valid calibration pixels
    candidates = []
    all_pixels = [(u, v) for v in range(H) for u in range(W)]
    random.shuffle(all_pixels)

    for (u, v) in all_pixels:
        if (u, v) in exclude_set:
            continue
        sd = sensor_depth_m[v, u]
        if sd <= 0.1 or sd > 10.0 or not np.isfinite(sd):
            continue
        dd = dpt_map[v, u]
        if dd <= 0 or not np.isfinite(dd):
            continue
        candidates.append((dd, sd))
        if len(candidates) >= n_samples:
            break

    if len(candidates) < 10:
        return None

    dpt_vals    = np.array([c[0] for c in candidates])
    sensor_vals = np.array([c[1] for c in candidates])

    # Least squares: dpt = scale * sensor
    scale = float(np.dot(dpt_vals, sensor_vals) / np.dot(sensor_vals, sensor_vals))
    return scale if scale > 0 else None

def backproject(px, py, depth_m, intr):
    x = (px - intr["cx"]) * depth_m / intr["fx"]
    y = (py - intr["cy"]) * depth_m / intr["fy"]
    return np.array([x, y, depth_m])

# ── Main ─────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Processing {len(df)} pairs...\n")

    poses_cache  = {}
    intr_cache   = {}
    reader_cache = {}
    results      = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pairs"):
        scene_id = row["scene_id"]

        if scene_id not in poses_cache:
            poses_cache[scene_id]  = load_all_poses(scene_id)
            intr_cache[scene_id]   = load_intrinsics(scene_id)
            sens_path = os.path.join(SCANS_DIR, scene_id, f"{scene_id}.sens")
            reader_cache[scene_id] = SensFrameReader(sens_path)

        poses  = poses_cache[scene_id]
        intr   = intr_cache[scene_id]
        reader = reader_cache[scene_id]

        ca = [row["cx_a"], row["cy_a"], row["cz_a"]]
        cb = [row["cx_b"], row["cy_b"], row["cz_b"]]

        best = find_best_frame(ca, cb, intr, poses)

        if best is None:
            results.append({
                "scene_id": scene_id, "range": row["range"],
                "gt_distance_m": row["gt_distance_m"],
                "pred_distance_m": None, "label_a": row["label_a"],
                "label_b": row["label_b"],
                "method": "baseline_b_dpt", "status": "no_valid_frame",
            })
            continue

        frame_idx, pose, pa, za, pb, zb = best

        # Read RGB + sensor depth from .sens
        color, sensor_depth = reader.read_color_and_depth(frame_idx)

        # Resize sensor depth to match color resolution if needed
        if sensor_depth.shape != (intr["height"], intr["width"]):
            sd_pil  = Image.fromarray(sensor_depth).resize(
                (intr["width"], intr["height"]), Image.NEAREST
            )
            sensor_depth = np.array(sd_pil, dtype=np.float32)

        # Run DPT on RGB
        dpt_map = run_dpt(color, intr["height"], intr["width"])

        # Calibrate scale using random pixels, excluding object locations
        exclude = [pa, pb]
        scale   = fit_scale_from_sensor(dpt_map, sensor_depth, exclude)

        if scale is None:
            results.append({
                "scene_id": scene_id, "range": row["range"],
                "gt_distance_m": row["gt_distance_m"],
                "pred_distance_m": None, "label_a": row["label_a"],
                "label_b": row["label_b"],
                "method": "baseline_b_dpt", "status": "scale_error",
            })
            continue

        # Sample DPT depth at object locations → convert to metric
        id_a = sample_patch(dpt_map, pa[0], pa[1])
        id_b = sample_patch(dpt_map, pb[0], pb[1])

        dm_a = id_a / scale if scale > 0 else None
        dm_b = id_b / scale if scale > 0 else None

        if dm_a is None or dm_b is None or dm_a < 0.05 or dm_b < 0.05:
            status = "metric_error"
            pred   = None
        else:
            pt_a   = backproject(pa[0], pa[1], dm_a, intr)
            pt_b   = backproject(pb[0], pb[1], dm_b, intr)
            pred   = round(float(np.linalg.norm(pt_a - pt_b)), 4)
            status = "ok"

        results.append({
            "scene_id":        scene_id,
            "range":           row["range"],
            "gt_distance_m":   row["gt_distance_m"],
            "pred_distance_m": pred,
            "label_a":         row["label_a"],
            "label_b":         row["label_b"],
            "method":          "baseline_b_dpt",
            "status":          status,
        })

    for r in reader_cache.values():
        r.close()

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    valid = out_df[out_df["status"] == "ok"].copy()
    valid["abs_error"] = (valid["pred_distance_m"] - valid["gt_distance_m"]).abs()
    valid["rel_error"] = valid["abs_error"] / valid["gt_distance_m"] * 100

    print(f"\n{'─'*50}")
    print(f"Baseline B — DPT Monocular Depth")
    print(f"{'─'*50}")
    print(f"Pairs processed : {len(out_df)}")
    print(f"Valid           : {len(valid)}")
    print(f"Failed          : {len(out_df) - len(valid)}")

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
    print(f"\nOutput: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
