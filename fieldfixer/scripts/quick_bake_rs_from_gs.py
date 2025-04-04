"""Quick RS->GS bake using Farneback flow."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


def _sorted(glob_pattern: str) -> list[str]:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    return paths


def _ensure_dirs(out_root: Path) -> None:
    for sub in ("W", "M", "LUT"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)


def _save_flow(path: Path, du: np.ndarray, dv: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, du=du.astype(np.float16), dv=dv.astype(np.float16))


def _confidence_mask(rs_rgb: np.ndarray, gs_rgb: np.ndarray, du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    h, w = rs_rgb.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + du
    map_y = ys + dv
    warped = cv2.remap(
        rs_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if warped.ndim == 2:
        diff_gray = cv2.absdiff(warped, gs_rgb)
    else:
        diff = cv2.absdiff(warped, gs_rgb)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    conf = 255 - np.clip(diff_gray * 2, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(conf, (5, 5), 0)


def bake(rs_glob: str, gs_glob: str, out_dir: str, fps: float = 20.0) -> None:
    out_root = Path(out_dir)
    _ensure_dirs(out_root)

    rs_paths = _sorted(rs_glob)
    gs_paths = _sorted(gs_glob)
    n = min(len(rs_paths), len(gs_paths))

    sample = iio.imread(rs_paths[0])
    height, width = sample.shape[:2]

    (out_root / "curves.json").write_text(json.dumps({"global": {"exposure": 1.0, "gamma": 1.0}}, indent=2))
    (out_root / "LUT" / "scene.cube").write_text(_identity_cube_lut())

    for idx, (rs_path, gs_path) in enumerate(zip(rs_paths[:n], gs_paths[:n])):
        print(f"[Bake] Processing frame {idx + 1}/{n}")
        rs = iio.imread(rs_path)
        gs = iio.imread(gs_path)
        if rs.ndim == 3:
            rs_gray = cv2.cvtColor(rs, cv2.COLOR_RGB2GRAY)
        else:
            rs_gray = rs
        if gs.ndim == 3:
            gs_gray = cv2.cvtColor(gs, cv2.COLOR_RGB2GRAY)
        else:
            gs_gray = gs
        flow = cv2.calcOpticalFlowFarneback(
            rs_gray,
            gs_gray,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=5,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        du = flow[..., 0].astype(np.float32)
        dv = flow[..., 1].astype(np.float32)
        _save_flow(out_root / "W" / f"{idx:06d}.npz", du, dv)
        # For mask confidence, ensure RGB for visual difference
        rs_rgb = rs if rs.ndim == 3 else cv2.cvtColor(rs, cv2.COLOR_GRAY2RGB)
        gs_rgb = gs if gs.ndim == 3 else cv2.cvtColor(gs, cv2.COLOR_GRAY2RGB)
        mask = _confidence_mask(rs_rgb, gs_rgb, du, dv)
        iio.imwrite(out_root / "M" / f"{idx:06d}.png", mask)

    meta = {
        "version": 1,
        "mapping": "displacement",
        "modules": ["rs_from_gs"],
        "profile": "quick",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": n,
        "sources": {"dataset": "TUM RS-GS seq4"},
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))


def _identity_cube_lut(size: int = 33) -> str:
    lines = [f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r / (size - 1):.6f} {g / (size - 1):.6f} {b / (size - 1):.6f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bake RS->GS displacement sidecars using Farneback flow")
    parser.add_argument("--rs", required=True, help="Glob for rolling-shutter frames (cam1)")
    parser.add_argument("--gs", required=True, help="Glob for global-shutter frames (cam0)")
    parser.add_argument("--out", required=True, help="Output bake directory")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    bake(args.rs, args.gs, args.out, fps=args.fps)
