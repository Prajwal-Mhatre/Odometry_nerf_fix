from __future__ import annotations

from pathlib import Path

import numpy as np


def load_cube_lut(path: Path) -> dict:
    """Load a .cube LUT file into a lookup table dictionary."""

    txt = Path(path).read_text().splitlines()
    size: int | None = None
    table: list[list[float]] = []
    for line in txt:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if line[0].isdigit() or line[0] == "-":
            table.append([float(x) for x in line.split()])
    if size is None:
        raise ValueError("Invalid .cube (missing LUT_3D_SIZE)")
    arr = np.array(table, dtype=np.float32).reshape(size, size, size, 3)
    return {"size": size, "table": arr}


def apply_lut(img: np.ndarray, lut: dict) -> np.ndarray:
    size = lut["size"]
    table = lut["table"]
    rgb = img.astype(np.float32) / 255.0
    coords = rgb * (size - 1)

    x0 = np.floor(coords[..., 0]).astype(np.int32)
    y0 = np.floor(coords[..., 1]).astype(np.int32)
    z0 = np.floor(coords[..., 2]).astype(np.int32)
    dx = coords[..., 0] - x0
    dy = coords[..., 1] - y0
    dz = coords[..., 2] - z0

    x1 = np.clip(x0 + 1, 0, size - 1)
    y1 = np.clip(y0 + 1, 0, size - 1)
    z1 = np.clip(z0 + 1, 0, size - 1)

    def sample(ix, iy, iz):  # noqa: ANN001 - helper for readability
        return table[ix, iy, iz]

    c000 = sample(x0, y0, z0)
    c100 = sample(x1, y0, z0)
    c010 = sample(x0, y1, z0)
    c110 = sample(x1, y1, z0)
    c001 = sample(x0, y0, z1)
    c101 = sample(x1, y0, z1)
    c011 = sample(x0, y1, z1)
    c111 = sample(x1, y1, z1)

    c00 = c000 * (1 - dx)[..., None] + c100 * dx[..., None]
    c01 = c001 * (1 - dx)[..., None] + c101 * dx[..., None]
    c10 = c010 * (1 - dx)[..., None] + c110 * dx[..., None]
    c11 = c011 * (1 - dx)[..., None] + c111 * dx[..., None]

    c0 = c00 * (1 - dy)[..., None] + c10 * dy[..., None]
    c1 = c01 * (1 - dy)[..., None] + c11 * dy[..., None]

    out = c0 * (1 - dz)[..., None] + c1 * dz[..., None]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
