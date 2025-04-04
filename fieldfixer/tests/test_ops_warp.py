import numpy as np

from fieldfixer.ops.warp import apply_displacement


def test_apply_displacement_identity() -> None:
    img = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    du = np.zeros((3, 3), dtype=np.float32)
    dv = np.zeros((3, 3), dtype=np.float32)
    warped = apply_displacement(img, du, dv)
    assert np.array_equal(warped, img)


def test_apply_displacement_shift() -> None:
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[1, 1] = [255, 0, 0]
    du = np.zeros((3, 3), dtype=np.float32)
    dv = np.zeros((3, 3), dtype=np.float32)
    du[:] = 1.0
    warped = apply_displacement(img, du, dv)
    assert warped[1, 2, 0] >= warped[1, 1, 0]
