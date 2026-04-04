"""
03_extract_sens_frames.py

Reads ScanNet .sens files and extracts RGB frames, depth maps and poses.
Header order matches the official ScanNet SensReader exactly.
"""

import os, sys, json, struct, zlib
import numpy as np
from PIL import Image
from tqdm import tqdm

SCANNET_DIR = "data/scannet/scans"
OUTPUT_DIR  = "data/scannet/frames"
FRAME_STEP  = 10
MAX_FRAMES  = 100

SCENES = [
    "scene0000_00", "scene0001_00", "scene0010_00", "scene0011_00",
    "scene0015_00", "scene0019_00", "scene0030_00", "scene0045_00",
    "scene0050_00", "scene0062_00", "scene0077_00", "scene0086_00",
    "scene0100_00", "scene0114_00", "scene0139_00", "scene0153_00",
    "scene0164_00", "scene0181_00", "scene0207_00", "scene0222_00",
]

# ── SensReader ───────────────────────────────────────────────────

class RGBDFrame:
    def load(self, f):
        self.camera_to_world = np.frombuffer(
            f.read(64), dtype=np.float32).reshape(4, 4)
        self.timestamp_color  = struct.unpack("Q", f.read(8))[0]
        self.timestamp_depth  = struct.unpack("Q", f.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", f.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", f.read(8))[0]
        self.color_data = f.read(self.color_size_bytes)
        self.depth_data = f.read(self.depth_size_bytes)

    def decompress_color(self):
        import io
        return np.array(Image.open(io.BytesIO(self.color_data)).convert("RGB"))

    def decompress_depth(self, w, h):
        raw = zlib.decompress(self.depth_data)
        return np.frombuffer(raw, dtype=np.uint16).reshape(h, w)


class SensReader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self._read_header()

    def _read_header(self):
        # Official ScanNet header order:
        # version → sensor_name → intrinsics×4 → compression → dims → shift → n_frames
        version = struct.unpack("I", self.f.read(4))[0]
        assert version == 4, f"Unsupported version {version}"

        strlen = struct.unpack("Q", self.f.read(8))[0]
        self.sensor_name = self.f.read(strlen).decode("utf-8")

        # Intrinsic / extrinsic matrices (4×4 float32 each)
        self.intrinsic_color = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
        self.extrinsic_color = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
        self.intrinsic_depth = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
        self.extrinsic_depth = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)

        # Compression types
        self.color_compression_type = struct.unpack("i", self.f.read(4))[0]
        self.depth_compression_type = struct.unpack("i", self.f.read(4))[0]

        # Image dimensions
        self.color_width  = struct.unpack("I", self.f.read(4))[0]
        self.color_height = struct.unpack("I", self.f.read(4))[0]
        self.depth_width  = struct.unpack("I", self.f.read(4))[0]
        self.depth_height = struct.unpack("I", self.f.read(4))[0]
        self.depth_shift  = struct.unpack("f", self.f.read(4))[0]
        self.n_frames     = struct.unpack("Q", self.f.read(8))[0]

    def intrinsics_dict(self):
        K = self.intrinsic_depth
        return {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width":  int(self.depth_width),
            "height": int(self.depth_height),
            "depth_shift": float(self.depth_shift),
        }

    def read_frame(self):
        frame = RGBDFrame()
        frame.load(self.f)
        return frame

    def close(self):
        self.f.close()

# ── Processing ───────────────────────────────────────────────────

def process_scene(scene_id):
    sens_path = os.path.join(SCANNET_DIR, scene_id, f"{scene_id}.sens")
    out_dir   = os.path.join(OUTPUT_DIR, scene_id)
    done_flag = os.path.join(out_dir, "done.flag")

    if os.path.isfile(done_flag):
        with open(done_flag) as f:
            n = f.read().strip()
        print(f"  {scene_id}: already done ({n}), skipping.")
        return int(n.split()[0])

    color_dir = os.path.join(out_dir, "color")
    depth_dir = os.path.join(out_dir, "depth")
    pose_dir  = os.path.join(out_dir, "pose")
    for d in [color_dir, depth_dir, pose_dir]:
        os.makedirs(d, exist_ok=True)

    reader = SensReader(sens_path)

    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(reader.intrinsics_dict(), f, indent=2)

    saved = 0
    for idx in tqdm(range(reader.n_frames),
                    desc=f"  {scene_id}", leave=False):
        frame = reader.read_frame()

        if idx % FRAME_STEP != 0:
            continue
        if saved >= MAX_FRAMES:
            continue

        name = f"{idx:06d}"

        Image.fromarray(frame.decompress_color()).save(
            os.path.join(color_dir, f"{name}.jpg"), quality=90)

        depth = frame.decompress_depth(reader.depth_width, reader.depth_height)
        Image.fromarray(depth).save(os.path.join(depth_dir, f"{name}.png"))

        np.savetxt(os.path.join(pose_dir, f"{name}.txt"),
                   frame.camera_to_world, fmt="%.8f")

        saved += 1

    reader.close()

    with open(done_flag, "w") as f:
        f.write(f"{saved} frames\n")

    return saved

# ── Main ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Extracting frames | step={FRAME_STEP} | max={MAX_FRAMES}/scene\n")

    total = 0
    for scene_id in SCENES:
        sens = os.path.join(SCANNET_DIR, scene_id, f"{scene_id}.sens")
        if not os.path.isfile(sens):
            print(f"  [WARN] Missing .sens: {scene_id}")
            continue
        n = process_scene(scene_id)
        print(f"  {scene_id}: {n} frames")
        total += n

    print(f"\nTotal frames extracted: {total}")
    print(f"Output: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
