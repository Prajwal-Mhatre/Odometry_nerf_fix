from __future__ import annotations

import numpy as np
from numba import njit

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

# TODO(codex): Add SSE/AVX-accelerated path via Numba prange if OpenCV absent.
# TODO(codex): Add edge-handling modes (replicate, reflect, constant).


def apply_displacement(img: np.ndarray, du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Warp an RGB frame using displacement maps."""

    h, w = img.shape[:2]
    if _HAS_CV2:
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = xs + du
        map_y = ys + dv
        return cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return _bilinear_sample(img, du, dv)


@njit(cache=True, fastmath=True)
def _bilinear_sample(img: np.ndarray, du: np.ndarray, dv: np.ndarray) -> np.ndarray:  # pragma: no cover - numba compiled
    h, w, c = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            xf = x + du[y, x]
            yf = y + dv[y, x]
            x0 = int(np.floor(xf))
            x1 = x0 + 1
            y0 = int(np.floor(yf))
            y1 = y0 + 1
            dx = xf - x0
            dy = yf - y0
            x0 = 0 if x0 < 0 else (w - 1 if x0 >= w else x0)
            x1 = 0 if x1 < 0 else (w - 1 if x1 >= w else x1)
            y0 = 0 if y0 < 0 else (h - 1 if y0 >= h else y0)
            y1 = 0 if y1 < 0 else (h - 1 if y1 >= h else y1)
            for ch in range(c):
                v00 = img[y0, x0, ch]
                v01 = img[y0, x1, ch]
                v10 = img[y1, x0, ch]
                v11 = img[y1, x1, ch]
                out[y, x, ch] = (
                    (1 - dx) * (1 - dy) * v00
                    + dx * (1 - dy) * v01
                    + (1 - dx) * dy * v10
                    + dx * dy * v11
                )
    return out
