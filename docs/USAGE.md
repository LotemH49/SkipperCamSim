# SkipperCamSim — usage and layout

## DS9 viewing

**Focal-plane array** (default — abutting 4×5, no gaps):

```bash
python skippercam_smash.py --stitch skippercam_sim.fits
/Applications/SAOImageDS9.app/Contents/MacOS/ds9 FITS/skippercam_focal_plane.fits
```

**With physical gaps** (set `focal_plane_gaps = True` or use `--stitch-gaps`):

```bash
python skippercam_smash.py --stitch-gaps skippercam_sim.fits
```

Or MEF with IRAF mosaic (after `--patch-mef`): **File → Open As → Mosaic IRAF**

**Sky-aligned mosaic** (chips placed by RA/Dec, gaps between CCDs — not a flat array):

```bash
/Applications/SAOImageDS9.app/Contents/MacOS/ds9 -mosaicimage wcs FITS/skippercam_test.fits
```

To fix headers on an existing MEF without re-simulating:

```bash
python skippercam_smash.py --patch-mef FITS/skippercam_test.fits
```

## Project layout

| Path | Purpose |
|------|---------|
| `skippercam_smash.py` | Main image simulation pipeline |
| `simular_curvitas/star_count.py` | Star detection, maps, histograms |
| `scripts/` | `cluster.py`, `basic_cluster.py`, `SMASH_readout.py` |
| `experiments/curvitas/` | Legacy variability / segmentation experiments |
| `FITS/` | Simulated MEF and focal-plane FITS |
| `data/catalogs/` | Secondary SMASH CSVs and test catalogs |
| `assets/` | Array layout figures |
| `INFO.md`, `docs/NOTES.md` | Survey parameters and open questions |

## Other scripts

```bash
python scripts/SMASH_readout.py          # download SMASH → lmc_smash_g99_v3.csv (root)
python scripts/cluster.py                # multi-threshold connected-component census
python simular_curvitas/star_count.py map 12
python experiments/curvitas/get_lumi_curves_skippercam.py
```
