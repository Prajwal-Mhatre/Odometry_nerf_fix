from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SidecarBundle:
    root: Path
    meta: dict

    @classmethod
    def load(cls, root: Path) -> "SidecarBundle":
        root = Path(root)
        meta_path = root / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(root=root, meta=meta)

    def load_warp(self, idx: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / "W" / f"{idx:06d}.npz"
        if not path.exists():
            return (
                np.zeros(shape, dtype=np.float16),
                np.zeros(shape, dtype=np.float16),
            )
        with np.load(path) as z:
            du = z["du"].astype(np.float32)
            dv = z["dv"].astype(np.float32)
        return du, dv

    def load_mask(self, idx: int, shape: tuple[int, int]) -> np.ndarray:
        path = self.root / "M" / f"{idx:06d}.png"
        if not path.exists():
            return np.full(shape, 255, dtype=np.uint8)
        import imageio.v3 as iio

        m = iio.imread(path)
        return m if m.ndim == 2 else m[..., 0]

    def load_curves(self, idx: int) -> dict:
        path = self.root / "curves.json"
        if not path.exists():
            return {"exposure": 1.0, "gamma": 1.0}
        data = json.loads(path.read_text())
        key = str(idx)
        return data.get(key, data.get("global", {"exposure": 1.0, "gamma": 1.0}))
