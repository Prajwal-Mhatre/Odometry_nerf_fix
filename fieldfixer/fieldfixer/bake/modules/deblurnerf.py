"""Deblur-NeRF wrapper stub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# TODO(codex): Call upstream Deblur-NeRF training with frames extracted from --in.
# TODO(codex): Render sharp targets at input poses to work_dir/targets/{frame}.png.
# TODO(codex): Compute dense flow (orig -> target) using RAFT if available, else OpenCV DIS.
# TODO(codex): Save flow to work_dir/flows/{frame}.npy, confidence to work_dir/conf/{frame}.npy.
# TODO(codex): Return dict(meta=...) with commit hash and params.


def run_deblurnerf(config: dict[str, Any], workspace: Path) -> None:
    """Train Deblur-NeRF and export sharp targets for downstream packing."""

    raise NotImplementedError("Deblur-NeRF integration pending")
