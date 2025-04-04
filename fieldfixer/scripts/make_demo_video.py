"""Generate a small demo video so FieldFixer can run end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np


def make_demo_video(out_path: Path, width: int = 640, height: int = 360, fps: int = 24, frames: int = 48) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    xs = np.linspace(0, 1, width, dtype=np.float32)
    ys = np.linspace(0, 1, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for i in range(frames):
        phase = i / max(frames - 1, 1)
        base = np.stack([grid_x, grid_y, np.full_like(grid_x, phase)], axis=-1)
        rgb = np.clip(base * 255.0, 0, 255).astype(np.uint8)

        # Add a moving square so motion is visible
        size = height // 5
        cx = int((0.1 + 0.8 * phase) * (width - size))
        cy = int((0.8 - 0.6 * phase) * (height - size))
        rgb[cy:cy + size, cx:cx + size] = [255, 255, 255]

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a synthetic demo clip for FieldFixer")
    parser.add_argument("--out", type=Path, default=Path("samples/alley_night/input.mp4"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()

    make_demo_video(args.out, width=args.width, height=args.height, fps=args.fps, frames=args.frames)
    print(f"Demo video written to {args.out}")
