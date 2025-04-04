"""RS-NeRF / URS-NeRF wrapper stubs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# TODO(codex): Launch RS-NeRF/URS-NeRF to estimate rolling-shutter poses.
# TODO(codex): Produce GS-corrected target renders per frame.
# TODO(codex): Compute orig->target flow; write flows + conf.


def run_rsnerf(config: dict[str, Any], workspace: Path) -> None:
    """Train RS-NeRF/URS-NeRF and drop corrected renders into workspace."""

    raise NotImplementedError("RS-NeRF integration pending")
