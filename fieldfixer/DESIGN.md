# DESIGN.md - FieldFixer Developer Definition

## 1) Project Charter

**Goal:** Deliver a robust, open-source video repair pipeline that leverages NeRF-era research offline and applies corrections online on CPU using simple sidecar files.

**Non-Goals:** Real-time neural inference; shipping 3D viewers; requiring GPUs on the end user's machine.

## 2) Personas & Top User Stories

**Creator Casey (no GPU):**
- *Story:* As a creator, I want to fix rolling shutter and noise on my clip so that I can publish professional-looking footage from my phone.
- *Acceptance:* Running `fieldfixer apply` with provided sidecars produces a stabilized, de-jelloed, cleaner video in < 3x input length on a typical laptop CPU.

**Engineer Eli (has GPU offline):**
- *Story:* As a maintainer, I want to mix and match research modules for bake so that I can improve quality over time without changing the runtime.
- *Acceptance:* New bake modules emit the same sidecar contract; runtime code does not change.

## 3) Architecture Overview

    +----------------+      Bake (offline, GPU)       +----------------------+ 
    | input video(s) | ---> [modules: RS, Deblur, RAW]| -> targets/flows/etc | 
    +----------------+                                 +----------------------+ 
              |                                                   |           
              | pack sidecars                                     v           
              |                                        +----------------------+ 
              +--------------------------------------> |   bake/ (sidecars)   | 
                                                       +----------------------+ 

                                                       Apply (online, CPU)    
                                                       +----------------------+ 
                                                       |  fieldfixer apply    | 
    +----------------+                                 |  warp->mask->curves->LUT| 
    | input video    | ------------------------------> |  output video        | 
    +----------------+                                 +----------------------+ 

Key contract: New modules must output targets (desired render) and flows (orig -> target). The packer converts them into displacement maps plus masks; color work becomes curves/LUT.

## 4) Repository Layout (high level)

```
fieldfixer/
  fieldfixer/
    cli.py             # Typer CLI
    io/                # video + sidecar I/O
    ops/               # image operations (NumPy/Numba/OpenCV)
    bake/              # offline pipeline + research wrappers
    config/            # Hydra profiles
  third_party/         # git submodules for research repos
  samples/             # small demo clip + baked sidecars
  tests/               # unit tests for ops + schema
```

## 5) Language & Libraries

- Python 3.11 only (no C++ to write).
- Runtime: PyAV, NumPy, Numba, OpenCV-Python, imageio, pydantic.
- Bake orchestration: Hydra, PyTorch (used inside research repos).

## 6) Detailed Interfaces

### 6.1 CLI (Typer)

`fieldfixer bake`
- Inputs: `--in` path, `--out` dir, `--profile`, list of `--modules`.
- Outputs: sidecars under `--out` (W/, M/, LUT/, curves.json, meta.json).
- Exit codes: 0 success, non-zero on missing inputs or module failure.

`fieldfixer apply`
- Inputs: video file plus sidecars dir.
- Outputs: repaired video.
- Exit codes: 0 success, non-zero on sidecar/video mismatch.

### 6.2 Python module boundaries (runtime)

```
# fieldfixer/io/video.py
class VideoReader:  # iterator of np.ndarray[H,W,3] (uint8, RGB)
    ...

class VideoWriter:  # accepts np.ndarray[H,W,3] (uint8, RGB)
    ...

# fieldfixer/io/sidecar.py
class SidecarBundle:
    @classmethod
    def load(root: Path) -> SidecarBundle: ...
    def load_warp(idx: int, shape: tuple[int,int]) -> tuple[np.ndarray, np.ndarray]: ...
    def load_mask(idx: int) -> np.ndarray: ...
    def load_curves(idx: int) -> dict: ...

# fieldfixer/ops/warp.py
def apply_displacement(img: np.ndarray, du: np.ndarray, dv: np.ndarray) -> np.ndarray: ...

# fieldfixer/ops/mask.py
def composite_with_mask(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

# fieldfixer/ops/exposure.py
def apply_curves(img: np.ndarray, curves: dict) -> np.ndarray: ...

# fieldfixer/ops/lut3d.py
def load_cube_lut(path: Path) -> dict: ...
def apply_lut(img: np.ndarray, lut: dict) -> np.ndarray: ...
```

### 6.3 Bake module contract (plug-in pattern)

Each `bake/modules/*.py` must implement:

```
def run(cfg: DictConfig, in_path: Path, work_dir: Path) -> dict:
    """
    Produces:
      work_dir/targets/{frame:06d}.png   # target render for frame t
      work_dir/flows/{frame:06d}.npy     # optical flow (u,v) from I_t -> target (float32,H,W,2)
      work_dir/conf/{frame:06d}.npy      # optional confidence map (float32,H,W)
    Returns a dict with metadata to be merged into meta.json.
    """
```

The packer (`bake/exporters/pack.py`) will:
- fuse or choose flows across modules;
- convert flow -> displacement (du,dv) float16;
- write masks (uint8);
- compute global LUT or curves if supplied.

## 7) Data Contracts (Schemas)

### 7.1 `meta.json` (pydantic, simplified)

```
{
  "version": 1,
  "width": 1920,
  "height": 1080,
  "fps": 29.97,
  "mapping": "displacement",
  "modules": ["rsnerf", "deblurnerf", "rawnerf", "nerfw"],
  "profile": "quality",
  "sources": {
    "rsnerf": {"commit": "abc123", "params": {"train_steps": 30000}},
    "deblurnerf": {"commit": "def456"},
    "rawnerf": {"commit": "ghi789"},
    "nerfw": {"commit": "jkl012"}
  },
    "ranges": {
    "frame_start": 0,
    "frame_end": 1432
  },
  "hash": "sha256:..."  // bake config hash for reproducibility
}
```

### 7.2 `curves.json` (two valid shapes)

Global (applies to all frames):

```
{"global": {"exposure": 1.15, "gamma": 0.95, "white_balance": [1.02, 1.0, 0.98]}}
```

Per-frame (stringified frame indices):

```
{
  "0": {"exposure": 1.10, "gamma": 1.0, "white_balance": [1.0, 1.0, 1.0]},
  "1": {"exposure": 1.12, "gamma": 0.98, "white_balance": [1.03, 1.0, 0.97]}
}
```

### 7.3 Flow -> Displacement conversion (deterministic)

Flow `F_t` in `(u, v)` from orig -> target. Store displacement as `du := F_t[..., 0]` and `dv := F_t[..., 1]`. Quantize to float16 for storage (keep float32 in RAM during apply).

## 8) Performance Targets (NFRs)

- Runtime (1080p, laptop CPU, 8 cores): warp + mask + curves + LUT <= 30 ms/frame target (about real-time).
- Memory <= 500 MB peak.
- Determinism: identical sidecars and input must yield bit-exact output.
- Reproducibility: `meta.json.hash` must change if bake config or module commits change.

## 9) Testing Strategy

- Unit tests (`tests/`):
  - `test_ops_warp.py`: identity displacement -> same frame; random displacement -> matches OpenCV remap baseline.
  - `test_lut3d.py`: identity LUT leaves frame unchanged; roundtrip color bounds.
  - `test_sidecar.py`: missing files fall back safely; shape mismatches raise.
- Golden tests: sample 10-frame input with baked sidecars -> compare output SSIM/PSNR to stored gold (tolerances configurable).

## 10) Error Handling Rules

- Missing `W/`: use identity displacement; log warning (once).
- Missing `M/`: use full-confidence mask (255).
- Missing `curves.json`: use `{ "exposure":1.0, "gamma":1.0 }`.
- Invalid `.cube`: skip LUT step and continue.
- Frame count mismatches: fail fast with exit code 2 and checklist message.

## 11) Telemetry (optional, off by default)

- `FFX_VERBOSE=1` -> log per-stage timings.
- `FFX_PROFILE_PATH=<file>` -> write CSV timings per frame.

## 12) Roadmap (modules you will integrate)

- v0.1: identity bake, full runtime (done)
- v0.2: Deblur module (train -> render -> flow)
- v0.3: Rolling-shutter module (GS re-renders -> flow)
- v0.4: RAW/HDR curves module
- v0.5: Appearance/LUT harmonization
- v0.6: Stabilization path (smooth virtual camera plus crop guards)

## Codex-friendly TODOs

```
/fieldfixer/bake/modules/deblurnerf.py

# TODO(codex): Call upstream Deblur-NeRF training with frames extracted from `--in`.
# TODO(codex): Render sharp targets at input poses to work_dir/targets/{frame}.png.
# TODO(codex): Compute dense flow (orig -> target) using RAFT if available, else OpenCV DIS.
# TODO(codex): Save flow to work_dir/flows/{frame}.npy, confidence to work_dir/conf/{frame}.npy.
# TODO(codex): Return dict(meta=...) with commit hash and params.

/fieldfixer/bake/modules/rsnerf.py

# TODO(codex): Launch RS-NeRF/URS-NeRF to estimate rolling-shutter poses.
# TODO(codex): Produce GS-corrected target renders per frame.
# TODO(codex): Compute orig->target flow; write flows plus conf.

/fieldfixer/bake/exporters/pack.py

# TODO(codex): Read all module flows/conf; fuse (for example, weighted by confidence).
# TODO(codex): Convert fused flow to du/dv float16; write W/{frame}.npz.
# TODO(codex): Build mask M/{frame}.png from conf (threshold plus blur edges).
# TODO(codex): Fit global .cube LUT if nerfw provided per-frame appearance deltas.
# TODO(codex): Write curves.json from rawnerf module if present.
# TODO(codex): Write meta.json with bake profile, module list, and hash.

/fieldfixer/ops/warp.py

# TODO(codex): Add SSE/AVX-accelerated path via Numba prange if OpenCV absent.
# TODO(codex): Add edge-handling modes (replicate, reflect, constant).

/tests/

# TODO(codex): Implement golden test using samples/alley_night with identity sidecars.
# TODO(codex): Add property-based tests for LUT tri-linear interpolation.
```

## Glossary (quick)

- **Target render:** what each frame should look like after a given correction (sharp, GS-corrected, denoised, etc.).
- **Flow:** dense 2D vector field mapping original frame -> target render.
- **Displacement (du, dv):** stored flow as two float16 images.
- **Mask:** confidence/occlusion map to composite warped pixels.
- **Curves/LUT:** global color corrections (exposure/gamma/WB and a 3D LUT).

## Definition of Done (for v0.1 runtime)

- `pip install -e .` works on macOS/Linux/Windows (Python 3.11).
- `fieldfixer apply` consumes identity sidecars and writes an output identical to the input (bit-exact).
