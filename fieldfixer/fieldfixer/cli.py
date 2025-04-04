from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm

from fieldfixer.io.sidecar import SidecarBundle
from fieldfixer.io.video import VideoReader, VideoWriter
from fieldfixer.ops.exposure import apply_curves
from fieldfixer.ops.lut3d import apply_lut, load_cube_lut
from fieldfixer.ops.mask import composite_with_mask
from fieldfixer.ops.warp import apply_displacement

app = typer.Typer(help="FieldFixer CLI")


@app.command("apply")
def apply_cli(
    inp: Path = typer.Option(..., "--in", help="Input video"),
    bake: Path = typer.Option(..., "--bake", help="Bake directory with sidecars"),
    out: Path = typer.Option(..., "--out", help="Output video path"),
    crf: int = typer.Option(18, help="H264 CRF"),
):
    bundle = SidecarBundle.load(bake)
    lut_path = bake / "LUT" / "scene.cube"
    lut = load_cube_lut(lut_path) if lut_path.exists() else None

    vr = VideoReader(inp)
    vw = VideoWriter(out, width=vr.width, height=vr.height, fps=vr.fps, crf=crf)

    for i, frame in enumerate(tqdm(vr, total=vr.nframes or None)):
        du, dv = bundle.load_warp(i, shape=frame.shape[:2])
        mask = bundle.load_mask(i, shape=frame.shape[:2])
        curves = bundle.load_curves(i)

        warped = apply_displacement(frame, du, dv)
        composited = composite_with_mask(warped, frame, mask)
        tonemapped = apply_curves(composited, curves)

        if lut is not None:
            tonemapped = apply_lut(tonemapped, lut)

        vw.write(tonemapped)

    vw.close()
    vr.close()


@app.command("bake")
def bake_cli(
    inp: Path = typer.Option(..., "--in"),
    out: Path = typer.Option(..., "--out"),
    profile: str = typer.Option("quality", "--profile"),
    modules: list[str] = typer.Option(["rsnerf", "deblurnerf"], "--modules"),
):
    """Run the offline bake pipeline using selected modules."""

    from fieldfixer.bake.pipeline import run_bake

    run_bake(inp, out, profile, modules)


if __name__ == "__main__":
    app()
