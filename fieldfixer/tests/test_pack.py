import json
from pathlib import Path

import numpy as np
import pytest

from fieldfixer.bake.exporters.pack import pack_sidecars

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


@pytest.mark.skipif(iio is None, reason="imageio.v3 required for mask writes")
def test_pack_sidecars_writes_warp_and_mask(tmp_path: Path) -> None:
    flow_dir = tmp_path / "module" / "flows"
    conf_dir = flow_dir.parent / "conf"
    flow_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)

    flow = np.zeros((2, 2, 2), dtype=np.float32)
    flow[..., 0] = 1.0
    np.save(flow_dir / "000000.npy", flow)
    np.save(conf_dir / "000000.npy", np.ones((2, 2), dtype=np.float32))

    out_dir = tmp_path / "out"
    pack_sidecars([], [flow_dir], out_dir)

    warp_path = out_dir / "W" / "000000.npz"
    mask_path = out_dir / "M" / "000000.png"
    meta_path = out_dir / "meta.json"

    assert warp_path.exists()
    with np.load(warp_path) as data:
        assert data["du"].shape == (2, 2)
        assert data["dv"].shape == (2, 2)
        assert np.allclose(data["du"], 1.0)
        assert np.allclose(data["dv"], 0.0)

    assert mask_path.exists()
    mask = iio.imread(mask_path)
    assert mask.dtype == np.uint8
    assert mask.shape == (2, 2)
    assert int(mask.min()) == 255
    assert int(mask.max()) == 255

    meta = json.loads(meta_path.read_text())
    assert meta["modules"] == ["module"]
    assert meta["frame_count"] == 1
    assert meta["frame_start"] == 0
    assert meta["frame_end"] == 0


@pytest.mark.skipif(iio is None, reason="imageio.v3 required for mask writes")
def test_pack_sidecars_merges_multiple_flows(tmp_path: Path) -> None:
    module_a = tmp_path / "A" / "flows"
    module_b = tmp_path / "B" / "flows"
    conf_a = module_a.parent / "conf"
    conf_b = module_b.parent / "conf"
    for path in (module_a, module_b, conf_a, conf_b):
        path.mkdir(parents=True)

    flow_a = np.zeros((1, 1, 2), dtype=np.float32)
    flow_a[..., 0] = 2.0
    flow_b = np.zeros((1, 1, 2), dtype=np.float32)
    flow_b[..., 0] = 0.0
    flow_b[..., 1] = 4.0

    np.save(module_a / "000000.npy", flow_a)
    np.save(module_b / "000000.npy", flow_b)
    np.save(conf_a / "000000.npy", np.ones((1, 1), dtype=np.float32))
    np.save(conf_b / "000000.npy", 0.5 * np.ones((1, 1), dtype=np.float32))

    out_dir = tmp_path / "out"
    pack_sidecars([], [module_a, module_b], out_dir)

    with np.load(out_dir / "W" / "000000.npz") as data:
        du = data["du"].astype(np.float32)
        dv = data["dv"].astype(np.float32)
        # Weighted average: (2 * 1 + 0 * 0.5) / (1 + 0.5) = 4/3
        assert np.allclose(du, 4.0 / 3.0)
        # Weighted average: (0 * 1 + 4 * 0.5) / (1 + 0.5) = 2/1.5 ~= 1.3333
        assert np.allclose(dv, 4.0 / 3.0)

    mask = iio.imread(out_dir / "M" / "000000.png")
    # Expected mean confidence = (1 + 0.5) / 2 = 0.75 -> 191 after scaling
    assert int(mask[0, 0]) in {190, 191}
