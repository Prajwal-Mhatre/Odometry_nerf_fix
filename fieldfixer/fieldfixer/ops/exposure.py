from __future__ import annotations

import numpy as np


def apply_curves(img: np.ndarray, curves: dict) -> np.ndarray:
    """Apply exposure, gamma, and white-balance adjustments."""

    exposure = float(curves.get("exposure", 1.0))
    gamma = float(curves.get("gamma", 1.0))
    wb = np.array(curves.get("white_balance", [1.0, 1.0, 1.0]), dtype=np.float32)

    arr = img.astype(np.float32) / 255.0
    arr = np.clip(arr * wb[None, None, :], 0, 10)
    arr = np.clip(arr * exposure, 0, 10)
    arr = arr ** (1.0 / max(gamma, 1e-6))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
