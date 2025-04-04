"""Pack module outputs into FieldFixer sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # Optional dependency for image output; imported lazily elsewhere too.
    import imageio.v3 as iio
except Exception:  # pragma: no cover - tests stub in tmp envs that may lack imageio
    iio = None


def pack_sidecars(target_dirs: Iterable[Path], flow_dirs: Iterable[Path], out_dir: Path) -> None:
    """Convert rendered targets and flow fields into runtime sidecars."""

    target_dirs = [Path(p) for p in target_dirs]
    flow_dirs = [Path(p) for p in flow_dirs]
    out_dir = Path(out_dir)

    w_dir = out_dir / "W"
    m_dir = out_dir / "M"
    lut_dir = out_dir / "LUT"
    for folder in (w_dir, m_dir, lut_dir):
        folder.mkdir(parents=True, exist_ok=True)

    frame_indices = _collect_frame_indices(flow_dirs)
    if not frame_indices:
        return

    written_frames: list[int] = []
    shape_hint: tuple[int, int] | None = None

    for frame_idx in frame_indices:
        flows, confidences = _load_flows_for_frame(flow_dirs, frame_idx)
        if not flows:
            continue
        height, width = flows[0].shape[:2]
        shape_hint = (height, width)

        fused_flow, fused_conf = _fuse_flows(flows, confidences)
        _write_warp(w_dir / f"{frame_idx:06d}.npz", fused_flow)
        _write_mask(m_dir / f"{frame_idx:06d}.png", fused_conf)
        written_frames.append(frame_idx)

    if not written_frames:
        return

    _maybe_copy_curves(target_dirs, out_dir)
    _maybe_copy_lut(target_dirs, lut_dir)
    _write_meta(out_dir, flow_dirs, written_frames, shape_hint)


def _collect_frame_indices(flow_dirs: Sequence[Path]) -> list[int]:
    indices: set[int] = set()
    for flow_dir in flow_dirs:
        if not flow_dir.exists():
            continue
        for file in flow_dir.glob("*.npy"):
            idx = _parse_frame_index(file.stem)
            if idx is not None:
                indices.add(idx)
    return sorted(indices)


def _parse_frame_index(stem: str) -> int | None:
    try:
        return int(stem)
    except ValueError:
        return None


def _load_flows_for_frame(flow_dirs: Sequence[Path], frame_idx: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    flows: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for flow_dir in flow_dirs:
        flow_path = flow_dir / f"{frame_idx:06d}.npy"
        if not flow_path.exists():
            continue
        flow = np.load(flow_path)
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(f"Invalid flow shape {flow.shape} at {flow_path}")
        flow = flow.astype(np.float32, copy=False)
        conf = _load_confidence_map(flow_dir, frame_idx, flow.shape[:2])
        flows.append(flow)
        confidences.append(conf)
    return flows, confidences


def _load_confidence_map(flow_dir: Path, frame_idx: int, shape: tuple[int, int]) -> np.ndarray:
    candidates = [
        flow_dir.parent / "conf" / f"{frame_idx:06d}.npy",
        flow_dir.parent / "conf" / flow_dir.name / f"{frame_idx:06d}.npy",
        flow_dir / "conf" / f"{frame_idx:06d}.npy",
    ]
    for cand in candidates:
        if cand.exists():
            conf = np.load(cand)
            if conf.ndim == 3 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            if conf.shape != shape:
                raise ValueError(f"Confidence shape {conf.shape} mismatch with flow shape {shape} at {cand}")
            return conf.astype(np.float32, copy=False)
    return np.ones(shape, dtype=np.float32)


def _fuse_flows(flows: Sequence[np.ndarray], confidences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not flows:
        raise ValueError("No flows to fuse")
    accum_flow = np.zeros_like(flows[0], dtype=np.float32)
    accum_conf = np.zeros(flows[0].shape[:2], dtype=np.float32)
    for flow, conf in zip(flows, confidences):
        if flow.shape != flows[0].shape:
            raise ValueError("Flow shapes must match for fusion")
        if conf.shape != flows[0].shape[:2]:
            raise ValueError("Confidence map dimensions must align with flow")
        accum_flow += flow * conf[..., None]
        accum_conf += conf
    with np.errstate(divide="ignore", invalid="ignore"):
        fused_flow = np.divide(
            accum_flow,
            accum_conf[..., None],
            out=np.zeros_like(accum_flow),
            where=accum_conf[..., None] > 1e-6,
        )
    max_weight = max(len(flows), 1)
    fused_conf = np.clip(accum_conf / max_weight, 0.0, 1.0)
    return fused_flow, fused_conf


def _write_warp(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    du = flow[..., 0].astype(np.float16)
    dv = flow[..., 1].astype(np.float16)
    np.savez_compressed(path, du=du, dv=dv)


def _write_mask(path: Path, confidence: np.ndarray) -> None:
    if iio is None:
        raise RuntimeError("imageio.v3 is required to write mask PNGs")
    mask = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, mask)


def _maybe_copy_curves(target_dirs: Sequence[Path], out_dir: Path) -> None:
    for directory in target_dirs:
        candidate = directory / "curves.json"
        if candidate.exists():
            data = candidate.read_text()
            (out_dir / "curves.json").write_text(data)
            break


def _maybe_copy_lut(target_dirs: Sequence[Path], lut_dir: Path) -> None:
    for directory in target_dirs:
        for name in ("scene.cube", "lut.cube"):
            candidate = directory / name
            if candidate.exists():
                lut_dir.mkdir(parents=True, exist_ok=True)
                (lut_dir / "scene.cube").write_text(candidate.read_text())
                return


def _write_meta(out_dir: Path, flow_dirs: Sequence[Path], frames: Sequence[int], shape_hint: tuple[int, int] | None) -> None:
    modules = sorted({
        flow_dir.parent.name if flow_dir.name.lower() == "flows" else flow_dir.name
        for flow_dir in flow_dirs
        if flow_dir.exists()
    })
    meta: dict[str, object] = {
        "version": 1,
        "mapping": "displacement",
        "modules": modules,
        "frame_start": frames[0],
        "frame_end": frames[-1],
        "frame_count": len(frames),
    }
    if shape_hint is not None:
        height, width = shape_hint
        meta.update({"height": int(height), "width": int(width)})

    sources: dict[str, object] = {}
    for flow_dir in flow_dirs:
        module_name = flow_dir.parent.name if flow_dir.name.lower() == "flows" else flow_dir.name
        meta_path = flow_dir.parent / "meta.json"
        if meta_path.exists():
            try:
                sources[module_name] = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
    if sources:
        meta["sources"] = sources

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
