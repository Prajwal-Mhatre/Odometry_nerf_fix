"""RawNeRF wrapper stub."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_rawnerf(config: dict[str, Any], workspace: Path) -> None:
    """Train RawNeRF and produce denoised/HDR renders and exposure curves."""

    raise NotImplementedError("RawNeRF integration pending")
