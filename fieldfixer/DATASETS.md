# DATASETS.md — FieldFixer Inputs

FieldFixer’s bake phase has four core tracks. Each section lists a primary dataset with direct download commands, optional alternatives, and tips on preparing the data into a single `--in` path (either a folder of frames or a video file).

## 1) Rolling-Shutter (RS) correction

**Primary dataset — TUM Rolling-Shutter Visual-Inertial**  
Time-synchronized rolling-shutter + global-shutter image streams with IMU and ground-truth poses for 10 sequences. GS frames provide a reference for RS correction quality.

```bash
mkdir -p data/tum_rs && cd data/tum_rs
wget https://cdn3.vision.in.tum.de/rolling/dataset-seq4.tar
tar -xf dataset-seq4.tar
```

The page lists all ten archives and calibration files. Choose any `dataset-seqX.tar` you need.  
Alternative RS imagery: Linköping Rolling Shutter Rectification Dataset; Neural Global Shutter (RSGR) dataset if available.

**Prep → one `--in` path**

- **Option A:** use image folder directly: `fieldfixer bake --in data/tum_rs/dataset-seq4/images/ ...`
- **Option B:** convert frames to an mp4 (30 fps):
  ```bash
  ffmpeg -y -pattern_type glob -i 'data/tum_rs/dataset-seq4/images/*.png' \
    -r 30 -pix_fmt yuv420p data/tum_rs/seq4.mp4
  ```

## 2) Motion/Defocus Deblur (multi-view blur)

**Primary dataset — Deblur-NeRF (official)**  
31 scenes with real camera motion blur, defocus blur, and some object motion. The authors recommend `real_camera_motion_blur/blurball` as a starter.

```bash
pip install gdown
mkdir -p data/deblurnerf && cd data/deblurnerf
gdown --folder https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_ -O .
```

Alternative blur datasets: check the Deblur-NeRF project page for more scenes or synthetic motion-blur sets.

**Prep → one `--in` path**

Scenes are already in LLFF-style multi-view format; pass the directory directly or render to mp4 as in the RS section.

## 3) RAW low-light / HDR (denoise + exposure fusion)

**Primary dataset — RawNeRF “refraw360”**  
Official RAW bursts ideal for rawnerf module testing.

```bash
mkdir -p data/rawnerf && cd data/rawnerf
wget https://storage.googleapis.com/gresearch/refraw360/raw.zip
unzip -q raw.zip
```

**Alternative:** SID (See-in-the-Dark, Sony subset ~25 GB) for low-light exposure/denoise training.

```bash
mkdir -p data/sid && cd data/sid
wget https://storage.googleapis.com/isl-datasets/SID/Sony.zip
unzip -q Sony.zip
```

**Prep → one `--in` path**

Feed the folder of RAW frames to your bake module. RawNeRF wrapper should ingest RAW (e.g., DNG), run the model, and emit tone/exposure curves plus target renders.

## 4) Generic video for smoke testing (identity bake)

Grab any short mp4 (e.g., from DAVIS). Useful to verify runtime plumbing before integrating research repos.

---

## FAQ

- **Phone videos?** Yes. Convert to frames or supply the mp4 directly; bake produces the same sidecar format.
- **Ground truth required?** No for usage; yes if you want quantitative validation (TUM’s GS, SID’s long exposure references, etc.).
- **Licensing?** Check each dataset’s license before redistributing results.

## Citations / dataset pages

- TUM Rolling-Shutter Visual-Inertial Dataset — Computer Vision Group
- Deblur-NeRF dataset — GitHub project page / Google Drive
- RawNeRF project page — Google Research
- SID dataset — CVPR 2018 (See-in-the-Dark)
- Linköping Rolling-Shutter Rectification Dataset
