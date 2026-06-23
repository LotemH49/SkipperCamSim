"""SEP source extraction on SkipperCam MEF (skippercam_sim_v4.fits).

Mirrors simular_curvitas/get_lumi_curves.py: background subtraction, SEP
extract, Kron-based aperture photometry, flux matrix, plot, FITS output.
Each CCD extension is treated like one image in the original time-series loop.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sep
import tqdm
from astropy.io import fits
from matplotlib import rcParams

rcParams["figure.figsize"] = [10.0, 8.0]

_CURVITAS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURVITAS_DIR.parents[1]
FITS_PATH = _PROJECT_ROOT / "FITS" / "skippercam_sim_v4.fits"
OUTPUT_FITS = _CURVITAS_DIR / "object_luminosity_skippercam.fits"

SEP_THRESH_SIGMA = 3.0
KRON_FACTOR = 6.0
APERTURE_KRON_MULT = 2.5
BG_BW, BG_BH, BG_FW, BG_FH = 64, 64, 3, 3


def ccd_extensions(hdul: fits.HDUList) -> list[tuple[int, str]]:
    """Return (HDU index, name) for extensions that carry image data."""
    out: list[tuple[int, str]] = []
    for i, hdu in enumerate(hdul):
        if hdu.data is None:
            continue
        name = hdu.name or f"CCD{i}"
        out.append((i, name))
    return out


def sep_detect_and_phot(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Background, extract, Kron aperture flux on one CCD."""
    img = np.asarray(data, dtype=np.float32)
    bkg = sep.Background(img, bw=BG_BW, bh=BG_BH, fw=BG_FW, fh=BG_FH)
    img_sub = img - bkg.back()

    objects = sep.extract(img_sub, SEP_THRESH_SIGMA, err=bkg.globalrms)
    n_objects = len(objects)
    if n_objects == 0:
        return np.array([]), np.array([]), 0

    kron_radii, _ = sep.kron_radius(
        img_sub,
        objects["x"],
        objects["y"],
        objects["a"],
        objects["b"],
        objects["theta"],
        KRON_FACTOR,
    )
    r_apertures = APERTURE_KRON_MULT * np.array(kron_radii)
    flux, _, _ = sep.sum_circle(
        img_sub, objects["x"], objects["y"], r=r_apertures
    )
    return flux, objects, n_objects


def main():
    with fits.open(FITS_PATH) as hdul:
        ccds = ccd_extensions(hdul)

        # Reference CCD (first extension with data) — same role as lista_archivos[0]
        ref_idx, ref_name = ccds[0]
        ref_data = hdul[ref_idx].data
        ref_flux, ref_objects, n_ref = sep_detect_and_phot(ref_data)
        print(f"Reference CCD {ref_name} (HDU {ref_idx}): {n_ref} objects")

        n_ccds = len(ccds)
        flux_rows: list[np.ndarray] = []
        star_counts: list[int] = []
        ccd_labels: list[str] = []

        for hdu_idx, ccd_name in tqdm.tqdm(ccds, desc="CCDs"):
            data = hdul[hdu_idx].data
            flux, _, n_objects = sep_detect_and_phot(data)
            flux_rows.append(flux)
            star_counts.append(n_objects)
            ccd_labels.append(ccd_name)

    total_stars = int(sum(star_counts))
    print(f"Star count per CCD: {dict(zip(ccd_labels, star_counts))}")
    print(f"Total stars (all CCDs, SEP): {total_stars}")

    max_objects = max(len(row) for row in flux_rows) if flux_rows else 0
    flux_matrix = np.full((n_ccds, max_objects), np.nan, dtype=np.float32)
    for i, row in enumerate(flux_rows):
        if len(row):
            flux_matrix[i, : len(row)] = row

    # Plot: sample object charges on reference CCD (analogous to per-object curves)
    if n_ref > 0:
        fig, ax = plt.subplots()
        n_plot = min(25, n_ref)
        start = min(100, max(0, n_ref - n_plot))
        obj_idx = np.arange(start, start + n_plot)
        ax.plot(obj_idx, ref_flux[start : start + n_plot], "o-")
        ax.set_ylabel("Total charge [electrons]")
        ax.set_xlabel("Object index (reference CCD)")
        ax.set_title(f"SEP flux on {ref_name} (objects {start}–{start + n_plot - 1})")
        plt.tight_layout()
        plt.show()

    # Histogram of all detected charges (focal plane)
    all_flux = np.concatenate([row for row in flux_rows if len(row)])
    if len(all_flux):
        fig, ax = plt.subplots()
        ax.hist(all_flux, bins=100, histtype="step")
        ax.set_xlabel("Total charge [electrons]")
        ax.set_ylabel("Count")
        ax.set_title(f"SEP charge distribution ({total_stars} detections)")
        plt.tight_layout()
        plt.show()

    # Bar chart of star counts per CCD
    ccd_indices = np.arange(n_ccds)
    fig, ax = plt.subplots()
    ax.bar(ccd_indices, star_counts)
    ax.set_xlabel("CCD index")
    ax.set_ylabel("Object count")
    ax.set_title(f"Stars per CCD (total {total_stars})")
    plt.tight_layout()
    plt.show()

    OUTPUT_FITS.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU(flux_matrix)
    primary.header["NCCD"] = (n_ccds, "Number of CCD rows")
    primary.header["NSTARS"] = (total_stars, "Sum of per-CCD SEP counts")
    primary.header["SRCFILE"] = (str(FITS_PATH.name), "Input MEF")
    counts_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ccd_index", format="J", array=ccd_indices),
            fits.Column(name="ccd_name", format="20A", array=ccd_labels),
            fits.Column(name="n_stars", format="J", array=np.array(star_counts)),
        ],
        name="STAR_COUNTS",
    )
    fits.HDUList([primary, counts_hdu]).writeto(OUTPUT_FITS, overwrite=True)
    print(f"Wrote {OUTPUT_FITS}")


if __name__ == "__main__":
    main()
