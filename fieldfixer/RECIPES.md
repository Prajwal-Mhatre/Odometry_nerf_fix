# RECIPES.md — FieldFixer End-to-End Runs

Three example pipelines show how to bake/apply FieldFixer modules, what artifacts appear, and how to sanity-check the results.

## A) Rolling-Shutter correction (TUM seq4)

**Bake**
```bash
fieldfixer bake \
  --in data/tum_rs/seq4.mp4 \
  --out runs/tum_seq4/bake \
  --profile quality \
  --modules rsnerf
```

**Apply**
```bash
fieldfixer apply \
  --in data/tum_rs/seq4.mp4 \
  --bake runs/tum_seq4/bake \
  --out runs/tum_seq4/fixed.mp4
```

**Expected outputs**
```
runs/tum_seq4/bake/
  W/000000.npz ...  # per-frame (du,dv)
  M/000000.png ...  # per-frame confidence
  LUT/scene.cube    # optional color LUT (module-dependent)
  curves.json       # optional exposure/gamma
  meta.json         # bake provenance
runs/tum_seq4/fixed.mp4
```

**Qualitative result**
- Reduced “jello” wobbles during fast pans; straighter verticals.
- Warps vary smoothly row-wise; masks favor confident regions.

**Quick checks**
- Inspect a mid-frame mask: bright areas mark confident warped pixels.
- Load a warp field: row-wise displacement should correlate with RS skew.
- If GS reference exists, measure PSNR/SSIM vs GS frames.

## B) Deblur (Deblur-NeRF blurball)

**Bake**
```bash
fieldfixer bake \
  --in data/deblurnerf/real_camera_motion_blur/blurball \
  --out runs/blurball/bake \
  --profile quality \
  --modules deblurnerf
```

**Apply**
```bash
fieldfixer apply \
  --in data/deblurnerf/real_camera_motion_blur/blurball \
  --bake runs/blurball/bake \
  --out runs/blurball/fixed.mp4
```

**Qualitative result**
- Sharper moving objects and background textures.
- Reduced ghosting; logos/text regain clarity.

**Quick checks**
- Compare Laplacian variance (sharpness proxy) before/after.
- Zoom to 200%: edges and text should look crisper.

## C) RAW low-light / HDR (RawNeRF “refraw360”)

**Bake**
```bash
fieldfixer bake \
  --in data/rawnerf/raw/<scene_folder> \
  --out runs/rawscene/bake \
  --profile quality \
  --modules rawnerf
```

**Apply**
```bash
fieldfixer apply \
  --in runs/rawscene/bake/ldr_preview.mp4 \
  --bake runs/rawscene/bake \
  --out runs/rawscene/fixed.mp4
```

**Qualitative result**
- Cleaner noise in dark regions; highlight recovery; consistent tone/white balance.

**Quick checks**
- `curves.json` should show exposure > 1.0 or non-trivial WB.
- Histogram of outputs: shadows lifted, highlights preserved.

---

## Common outputs

For any bake/apply cycle you should see:

- `fixed.mp4` (or .mov) matching input resolution/FPS unless modules crop.
- Sidecars under `bake/`:
  - `W/{frame:06d}.npz` — displacement maps (du, dv) in pixel units.
  - `M/{frame:06d}.png` — confidence masks (0–255).
  - `curves.json` — exposure/gamma/WB metadata (global or per-frame).
  - `LUT/scene.cube` — optional 3D LUT (33³).
  - `meta.json` — modules, profile, size/fps, source commits.

Qualitative improvements depend on modules enabled (RS, Deblur, RAW/HDR). Use these recipes as baselines before integrating additional research tracks or custom datasets.
