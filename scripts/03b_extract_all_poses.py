"""
03b_extract_all_poses.py

Extracts ALL camera poses from each .sens file (no images).
Poses are tiny (4x4 float32) so all frames fit easily on disk.
This allows finding which frame each object pair is visible in.

Output: data/scannet/poses/<scene_id>/
          000000.txt, 000001.txt, ...
          intrinsics.json
"""

import os, json, struct, zlib
import numpy as np
from tqdm import tqdm

SCANNET_DIR = "data/scannet/scans"
OUTPUT_DIR  = "data/scannet/poses"

SCENES = [
    "scene0000_00", "scene0001_00", "scene0010_00", "scene0011_00",
    "scene0015_00", "scene0019_00", "scene0030_00", "scene0045_00",
    "scene0050_00", "scene0062_00", "scene0077_00", "scene0086_00",
    "scene0100_00", "scene0114_00", "scene0139_00", "scene0153_00",
    "scene0164_00", "scene0181_00", "scene0207_00", "scene0222_00",
]

class SensHeaderOnly:
    """Reads header and skips frame data — fast pose extraction."""
    def __init__(self, path):
        self.f = open(path, "rb")
        self._read_header()

    def _read_header(self):
        version = struct.unpack("I", self.f.read(4))[0]
        assert version == 4
        strlen = struct.unpack("Q", self.f.read(8))[0]
        self.sensor_name = self.f.read(strlen).decode("utf-8")
        self.intrinsic_color = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
        self.extrinsic_color = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
        self.intrinsic_depth = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
        self.extrinsic_depth = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
        self.color_compression_type = struct.unpack("i", self.f.read(4))[0]
        self.depth_compression_type = struct.unpack("i", self.f.read(4))[0]
        self.color_width  = struct.unpack("I", self.f.read(4))[0]
        self.color_height = struct.unpack("I", self.f.read(4))[0]
        self.depth_width  = struct.unpack("I", self.f.read(4))[0]
        self.depth_height = struct.unpack("I", self.f.read(4))[0]
        self.depth_shift  = struct.unpack("f", self.f.read(4))[0]
        self.n_frames     = struct.unpack("Q", self.f.read(8))[0]

    def intrinsics_dict(self):
        K = self.intrinsic_depth
        return {
            "fx": float(K[0,0]), "fy": float(K[1,1]),
            "cx": float(K[0,2]), "cy": float(K[1,2]),
            "width":  int(self.depth_width),
            "height": int(self.depth_height),
            "depth_shift": float(self.depth_shift),
        }

    def read_all_poses(self):
        """Read pose + skip image data for every frame. Returns list of 4x4 arrays."""
        poses = []
        for _ in range(self.n_frames):
            pose = np.frombuffer(self.f.read(64), dtype=np.float32).reshape(4,4)
            # timestamps
            self.f.read(16)
            # color size + data
            color_size = struct.unpack("Q", self.f.read(8))[0]
            # depth size + data
            depth_size = struct.unpack("Q", self.f.read(8))[0]
            self.f.read(color_size)
            self.f.read(depth_size)
            poses.append(pose)
        return poses

    def close(self):
        self.f.close()


def process_scene(scene_id):
    sens_path = os.path.join(SCANNET_DIR, scene_id, f"{scene_id}.sens")
    out_dir   = os.path.join(OUTPUT_DIR, scene_id)
    done_flag = os.path.join(out_dir, "done.flag")

    if os.path.isfile(done_flag):
        with open(done_flag) as f:
            n = int(f.read().split()[0])
        print(f"  {scene_id}: already done ({n} poses), skipping.")
        return n

    os.makedirs(out_dir, exist_ok=True)

    reader = SensHeaderOnly(sens_path)

    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump(reader.intrinsics_dict(), f, indent=2)

    poses = reader.read_all_poses()
    reader.close()

    for idx, pose in enumerate(poses):
        np.savetxt(os.path.join(out_dir, f"{idx:06d}.txt"),
                   pose, fmt="%.8f")

    with open(done_flag, "w") as f:
        f.write(f"{len(poses)} poses\n")

    return len(poses)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Extracting all poses from {len(SCENES)} scenes...\n")

    for scene_id in tqdm(SCENES, desc="Scenes"):
        n = process_scene(scene_id)
        tqdm.write(f"  {scene_id}: {n} poses")

    print("\nDone. Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
