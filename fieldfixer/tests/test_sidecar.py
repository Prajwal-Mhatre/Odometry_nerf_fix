import json
from pathlib import Path

import numpy as np

from fieldfixer.io.sidecar import SidecarBundle


def test_sidecar_identity(tmp_path: Path) -> None:
    bundle = SidecarBundle.load(tmp_path)
    du, dv = bundle.load_warp(0, shape=(4, 4))
    mask = bundle.load_mask(0, shape=(4, 4))
    curves = bundle.load_curves(0)

    assert du.shape == (4, 4)
    assert dv.shape == (4, 4)
    assert np.allclose(du, 0.0)
    assert np.allclose(dv, 0.0)
    assert mask.dtype == np.uint8
    assert mask.min() == 255 and mask.max() == 255
    assert curves["exposure"] == 1.0
    assert curves["gamma"] == 1.0


def test_sidecar_reads_meta(tmp_path: Path) -> None:
    meta = {"version": 1}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    bundle = SidecarBundle.load(tmp_path)
    assert bundle.meta == meta
