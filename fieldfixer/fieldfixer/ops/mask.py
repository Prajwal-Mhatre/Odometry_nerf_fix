from __future__ import annotations

import numpy as np


def composite_with_mask(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend foreground/background frames using a uint8 mask."""

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = fg.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
