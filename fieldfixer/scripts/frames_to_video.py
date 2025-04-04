"""Convert a directory of frames into an mp4 video using PyAV."""

from __future__ import annotations

import argparse
from pathlib import Path
from fractions import Fraction

import av
import imageio.v3 as iio
import numpy as np


def _to_uint8(arr):
    if arr.dtype == np.uint8:
        out = arr
    elif arr.dtype == np.uint16:
        out = (arr >> 8).astype(np.uint8)
    else:
        out = np.clip(arr, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    return out


def frames_to_video(frame_dir: Path, out_path: Path, fps: float) -> None:
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.png")) or sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames found under {frame_dir}")

    sample = _to_uint8(iio.imread(frames[0]))
    height, width = sample.shape[:2]

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for frame_path in frames:
        rgb = _to_uint8(iio.imread(frame_path))
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frames -> mp4 converter")
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    frames_to_video(args.frame_dir, args.out, args.fps)
    print(f"Wrote {args.out}")
