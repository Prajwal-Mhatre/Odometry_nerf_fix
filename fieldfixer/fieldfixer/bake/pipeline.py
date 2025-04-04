from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fieldfixer.io.video import VideoReader


def run_bake(inp: Path, out: Path, profile: str, modules: list[str]) -> None:
    """Stub bake pipeline that emits identity sidecars for quick testing."""

    out = Path(out)
    (out / "W").mkdir(parents=True, exist_ok=True)
    (out / "M").mkdir(parents=True, exist_ok=True)
    (out / "LUT").mkdir(parents=True, exist_ok=True)

    from imageio.v3 import imwrite

    vr = VideoReader(str(inp))
    frames_written = 0
    width = vr.width
    height = vr.height

    for idx, frame in enumerate(vr):
        h, w = frame.shape[:2]
        np.savez_compressed(
            out / "W" / f"{idx:06d}.npz",
            du=np.zeros((h, w), np.float16),
            dv=np.zeros((h, w), np.float16),
        )
        imwrite(out / "M" / f"{idx:06d}.png", np.full((h, w), 255, dtype=np.uint8))
        frames_written = idx + 1
        width, height = w, h

    if frames_written == 0:
        np.savez_compressed(
            out / "W" / "000000.npz",
            du=np.zeros((height, width), np.float16),
            dv=np.zeros((height, width), np.float16),
        )
        imwrite(out / "M" / "000000.png", np.full((height, width), 255, dtype=np.uint8))
        frames_written = 1

    vr.close()

    (out / "LUT" / "scene.cube").write_text(_identity_cube_lut())
    (out / "curves.json").write_text(json.dumps({"global": {"exposure": 1.0, "gamma": 1.0}}, indent=2))

    meta = {
        "version": 1,
        "mapping": "displacement",
        "modules": modules,
        "profile": profile,
        "input": str(inp),
        "width": width,
        "height": height,
        "frames": frames_written,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def _identity_cube_lut(size: int = 33) -> str:
    lines = [f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r / (size - 1):.6f} {g / (size - 1):.6f} {b / (size - 1):.6f}")
    return "\n".join(lines)
