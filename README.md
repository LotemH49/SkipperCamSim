# SkipperCamSim

Simulate SkipperCam survey images from SMASH catalog sources: magnitude-to-charge calibration, Gaussian PSF deposition, sky background, Poisson photon noise, and readout noise.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the main SMASH catalog at the project root as `lmc_smash_g99_v3.csv` (not tracked in git due to size). Older or test catalogs live in `data/catalogs/`.

## Main script

```bash
python skippercam_smash.py
```

Tune exposure time, PSF sigma, and other constants at the top of `skippercam_smash.py`.

To export all 20 CCDs, set `write_mef = True` (writes MEF under `FITS/` and can auto-stitch the focal plane).

## Star counting

```bash
python simular_curvitas/star_count.py map 12
```

## More

See [docs/USAGE.md](docs/USAGE.md) for DS9 viewing, full project layout, and auxiliary scripts.
