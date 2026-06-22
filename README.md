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

To export all 20 CCDs, set `write_mef = True` (writes MEF + auto-stitches `skippercam_focal_plane.fits` as abutting 4×5 grid with no gaps).

View in SAOImage DS9:

**Focal-plane array** (default — abutting 4×5, no gaps):

```bash
python skippercam_smash.py --stitch skippercam_sim.fits
/Applications/SAOImageDS9.app/Contents/MacOS/ds9 skippercam_focal_plane.fits
```

**With physical gaps** (set `focal_plane_gaps = True` or use `--stitch-gaps`):

```bash
python skippercam_smash.py --stitch-gaps skippercam_sim.fits
```

Or MEF with IRAF mosaic (after `--patch-mef`): **File → Open As → Mosaic IRAF**

**Sky-aligned mosaic** (chips placed by RA/Dec, gaps between CCDs — not a flat array):

```bash
/Applications/SAOImageDS9.app/Contents/MacOS/ds9 -mosaicimage wcs skippercam_test.fits
```

To fix headers on an existing MEF without re-simulating:

```bash
python skippercam_smash.py --patch-mef skippercam_test.fits
```

## Project layout

- `skippercam_smash.py` — main image simulation pipeline
- `skippercam_pixelate` — earlier pixelation prototype
- `SMASH_readout.py` — SMASH readout utilities
- `simular_curvitas/` — related variability / segmentation experiments
- `INFO.md`, `NOTES.md` — survey parameters and open questions
