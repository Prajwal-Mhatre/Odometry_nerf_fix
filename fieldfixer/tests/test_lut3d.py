from pathlib import Path

import numpy as np

from fieldfixer.ops.lut3d import apply_lut, load_cube_lut


def _write_test_lut(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "LUT_3D_SIZE 2",
                "0.0 0.0 0.0",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "1.0 1.0 0.0",
                "0.0 0.0 1.0",
                "1.0 0.0 1.0",
                "0.0 1.0 1.0",
                "1.0 1.0 1.0",
            ]
        )
    )


def test_load_and_apply_lut(tmp_path: Path) -> None:
    lut_path = tmp_path / "scene.cube"
    _write_test_lut(lut_path)
    lut = load_cube_lut(lut_path)
    img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    out = apply_lut(img, lut)
    assert out.shape == img.shape
    assert out[0, 0, 0] <= out[0, 1, 0]
