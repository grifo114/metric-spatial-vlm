from __future__ import annotations

import argparse
import csv
import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


COLOR_COMPRESSION = {
    0: "raw",
    1: "png",
    2: "jpeg",
}

DEPTH_COMPRESSION = {
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",  # not handled here
}


@dataclass
class RGBDFrame:
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_data: bytes
    depth_data: bytes


class ScanNetSensReader:
    def __init__(self, sens_path: Path):
        self.sens_path = sens_path
        self.version: int | None = None
        self.sensor_name: str | None = None
        self.intrinsic_color: np.ndarray | None = None
        self.extrinsic_color: np.ndarray | None = None
        self.intrinsic_depth: np.ndarray | None = None
        self.extrinsic_depth: np.ndarray | None = None
        self.color_compression: str | None = None
        self.depth_compression: str | None = None
        self.color_width: int | None = None
        self.color_height: int | None = None
        self.depth_width: int | None = None
        self.depth_height: int | None = None
        self.depth_shift: float | None = None
        self.num_frames: int | None = None
        self.frames: list[RGBDFrame] = []

    @staticmethod
    def _read(f: BinaryIO, fmt: str):
        size = struct.calcsize(fmt)
        data = f.read(size)
        if len(data) != size:
            raise EOFError(f"Unexpected EOF while reading format {fmt}")
        return struct.unpack(fmt, data)

    @staticmethod
    def _read_matrix4x4(f: BinaryIO) -> np.ndarray:
        vals = ScanNetSensReader._read(f, "f" * 16)
        return np.asarray(vals, dtype=np.float32).reshape(4, 4)

    def load(self, load_frames: bool = True) -> None:
        if not self.sens_path.exists():
            raise FileNotFoundError(f".sens not found: {self.sens_path}")

        with self.sens_path.open("rb") as f:
            self.version = self._read(f, "I")[0]

            strlen = self._read(f, "Q")[0]
            sensor_name_bytes = f.read(strlen)
            self.sensor_name = sensor_name_bytes.decode("utf-8", errors="ignore")

            self.intrinsic_color = self._read_matrix4x4(f)
            self.extrinsic_color = self._read_matrix4x4(f)
            self.intrinsic_depth = self._read_matrix4x4(f)
            self.extrinsic_depth = self._read_matrix4x4(f)

            color_compression_code = self._read(f, "i")[0]
            depth_compression_code = self._read(f, "i")[0]
            self.color_compression = COLOR_COMPRESSION.get(color_compression_code, f"unknown_{color_compression_code}")
            self.depth_compression = DEPTH_COMPRESSION.get(depth_compression_code, f"unknown_{depth_compression_code}")

            self.color_width = self._read(f, "I")[0]
            self.color_height = self._read(f, "I")[0]
            self.depth_width = self._read(f, "I")[0]
            self.depth_height = self._read(f, "I")[0]
            self.depth_shift = self._read(f, "f")[0]

            self.num_frames = self._read(f, "Q")[0]

            if not load_frames:
                return

            for _ in range(self.num_frames):
                camera_to_world = self._read_matrix4x4(f)
                timestamp_color = self._read(f, "Q")[0]
                timestamp_depth = self._read(f, "Q")[0]
                color_size = self._read(f, "Q")[0]
                depth_size = self._read(f, "Q")[0]

                color_data = f.read(color_size)
                depth_data = f.read(depth_size)

                self.frames.append(
                    RGBDFrame(
                        camera_to_world=camera_to_world,
                        timestamp_color=timestamp_color,
                        timestamp_depth=timestamp_depth,
                        color_data=color_data,
                        depth_data=depth_data,
                    )
                )


def save_matrix_txt(path: Path, mat: np.ndarray) -> None:
    np.savetxt(path, mat, fmt="%.8f")


def save_metadata(reader: ScanNetSensReader, out_dir: Path) -> None:
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_txt(meta_dir / "intrinsic_color.txt", reader.intrinsic_color)
    save_matrix_txt(meta_dir / "extrinsic_color.txt", reader.extrinsic_color)
    save_matrix_txt(meta_dir / "intrinsic_depth.txt", reader.intrinsic_depth)
    save_matrix_txt(meta_dir / "extrinsic_depth.txt", reader.extrinsic_depth)

    with (meta_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"version: {reader.version}\n")
        f.write(f"sensor_name: {reader.sensor_name}\n")
        f.write(f"color_compression: {reader.color_compression}\n")
        f.write(f"depth_compression: {reader.depth_compression}\n")
        f.write(f"color_size: {reader.color_width}x{reader.color_height}\n")
        f.write(f"depth_size: {reader.depth_width}x{reader.depth_height}\n")
        f.write(f"depth_shift: {reader.depth_shift}\n")
        f.write(f"num_frames: {reader.num_frames}\n")


def save_color_frame(reader: ScanNetSensReader, frame: RGBDFrame, out_path: Path) -> None:
    if reader.color_compression == "jpeg":
        out_path.write_bytes(frame.color_data)
        return

    if reader.color_compression == "png":
        out_path.write_bytes(frame.color_data)
        return

    if reader.color_compression == "raw":
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError("Pillow is required to save raw color frames. Install with: pip install pillow") from e

        arr = np.frombuffer(frame.color_data, dtype=np.uint8)
        arr = arr.reshape(reader.color_height, reader.color_width, 3)
        Image.fromarray(arr).save(out_path)
        return

    raise NotImplementedError(f"Unsupported color compression: {reader.color_compression}")


def save_depth_frame(reader: ScanNetSensReader, frame: RGBDFrame, out_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow is required to save depth frames. Install with: pip install pillow") from e

    if reader.depth_compression == "zlib_ushort":
        raw = zlib.decompress(frame.depth_data)
        depth = np.frombuffer(raw, dtype=np.uint16).reshape(reader.depth_height, reader.depth_width)
        Image.fromarray(depth).save(out_path)
        return

    if reader.depth_compression == "raw_ushort":
        depth = np.frombuffer(frame.depth_data, dtype=np.uint16).reshape(reader.depth_height, reader.depth_width)
        Image.fromarray(depth).save(out_path)
        return

    raise NotImplementedError(f"Unsupported depth compression: {reader.depth_compression}")


def extract(
    sens_path: Path,
    out_dir: Path,
    frame_stride: int,
    max_frames: int | None,
    save_depth: bool,
) -> None:
    reader = ScanNetSensReader(sens_path)
    reader.load(load_frames=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    color_dir = out_dir / "color"
    pose_dir = out_dir / "pose"
    depth_dir = out_dir / "depth"

    color_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    if save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(reader, out_dir)

    csv_path = out_dir / "frames_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "frame_idx",
                "saved_idx",
                "timestamp_color",
                "timestamp_depth",
                "color_path",
                "pose_path",
                "depth_path",
            ],
        )
        writer.writeheader()

        saved = 0
        for i, frame in enumerate(reader.frames):
            if i % frame_stride != 0:
                continue
            if max_frames is not None and saved >= max_frames:
                break

            saved_idx = saved
            color_path = color_dir / f"{saved_idx:06d}.jpg"
            pose_path = pose_dir / f"{saved_idx:06d}.txt"
            depth_path = depth_dir / f"{saved_idx:06d}.png" if save_depth else None

            save_color_frame(reader, frame, color_path)
            save_matrix_txt(pose_path, frame.camera_to_world)

            if save_depth:
                save_depth_frame(reader, frame, depth_path)

            writer.writerow(
                {
                    "frame_idx": i,
                    "saved_idx": saved_idx,
                    "timestamp_color": frame.timestamp_color,
                    "timestamp_depth": frame.timestamp_depth,
                    "color_path": str(color_path),
                    "pose_path": str(pose_path),
                    "depth_path": str(depth_path) if depth_path else "",
                }
            )

            saved += 1

    print(f"Sens file: {sens_path}")
    print(f"Output dir: {out_dir}")
    print(f"Frames in file: {reader.num_frames}")
    print(f"Saved frames: {saved}")
    print(f"Frame stride: {frame_stride}")
    print(f"Color compression: {reader.color_compression}")
    print(f"Depth compression: {reader.depth_compression}")
    print(f"Index CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB frames, poses, and optional depth from a ScanNet .sens file.")
    parser.add_argument("--sens_path", type=Path, required=True, help="Path to sceneXXXX_YY.sens")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--frame_stride", type=int, default=30, help="Save one frame every N frames (default: 30)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to save")
    parser.add_argument("--save_depth", action="store_true", help="Also save depth PNGs")
    args = parser.parse_args()

    extract(
        sens_path=args.sens_path.resolve(),
        out_dir=args.out_dir.resolve(),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        save_depth=args.save_depth,
    )


if __name__ == "__main__":
    main()