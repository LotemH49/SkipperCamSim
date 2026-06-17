# SkipperCamSim

Simulate SkipperCam survey images from SMASH catalog sources: magnitude-to-charge calibration, Gaussian PSF deposition, sky background, Poisson photon noise, and readout noise.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place SMASH catalog CSV files in the project root (e.g. `lmc_smash_g99_v2.csv`). These are not tracked in git due to size.

## Main script

```bash
python skippercam_smash.py
```

Tune patch size via `width_px` / `height_px`, exposure time, PSF sigma, and other constants at the top of `skippercam_smash.py`.

## Project layout

- `skippercam_smash.py` — main image simulation pipeline
- `skippercam_pixelate` — earlier pixelation prototype
- `SMASH_readout.py` — SMASH readout utilities
- `simular_curvitas/` — related variability / segmentation experiments
- `INFO.md`, `NOTES.md` — survey parameters and open questions
