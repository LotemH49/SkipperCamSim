"""Count stars in the SkipperCam MEF (skippercam_sim_v4.fits).

The LMC field is heavily crowded (median background ~45 e-/pix is mostly
overlapping PSF wings), so the choice of method matters a lot. We run four and
compare them across all 20 CCDs:

  1. MATCHED-FILTER PEAKS  (PRIMARY)
        Correlate each CCD with the PSF to form an *integrated* S/N map, then
        take local maxima above SN_THRESHOLD. Local maxima separate blended
        stars (unlike connected components, which merge touching stars), so
        this is the most appropriate counter for a crowded field and ties
        directly to the S/N knob. It also yields the faintest confidently
        detected star in electrons.

  2. SEP                    (the get_lumi_curves.py approach)
        Background subtract -> extract -> aperture photometry -> keep sources
        with integrated S/N >= SN_THRESHOLD. SExtractor-style; tends to
        under-count in the densest regions because its cleaning merges blends.

  3. GLOBAL THRESHOLD       (the cluster.py approach)
        Connected components above sky + SN_THRESHOLD * sigma.

  4. LOCAL THRESHOLD        (adaptive)
        skimage local (adaptive) threshold + connected components.

Tune SN_THRESHOLD (and the knobs below) at the top of the file.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import sep
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------
# TUNABLES
# ----------------------------------------------------------------------------
SN_THRESHOLD = 5          # integrated signal-to-noise to "confidently" detect

# Detector / PSF (from skippercam_sim_v4.fits header + skippercam_smash.py)
READ_NOISE_E = 7.0          # readout noise [e- RMS/pix]   (RDNOISE)
PSF_SIGMA_PIX = 1.5        # Gaussian PSF sigma [pix]      (PSFSIG)
GAIN = 1.0                  # data already in electrons

# Matched-filter peak detector (primary)
MIN_PEAK_DISTANCE = 3        # min separation between peaks [pix] (FWHM ~ 3.5)

# Clump refinement: split blended groups using shape + local iterative subtraction
USE_CLUMP_REFINE = True
CLUMP_SN_FRAC = 0.50         # clump mask = S/N above this fraction of SN_THRESHOLD
CLUMP_MIN_AREA = 10          # min labeled area [pix] to consider
CLUMP_AREA_FACTOR = 2      # area > factor * pi*(2*sigma)^2 -> suspect blend
CLUMP_ECCENTRICITY = .55    # elongated blobs likely hold >1 star
CLUMP_EXTENT_MAX = 0.45      # area/bbox; compact clumps = single star
CLUMP_PAD_PIX = 7            # ROI padding around clump bbox
CLUMP_ITER_ROUNDS = 3           # local subtract + re-detect passes per clump
CLUMP_MAX_PEAKS_PER_ROUND = 24  # peak_local_max cap inside each clump per round
CLUMP_MAX_AREA = 2500           # skip huge connected regions [pix]

# Star map / run mode
SHOW_PLOT = True            # True: plot CCD star map
SHOW_HIST = False           # True: g-band histogram (detected vs SMASH catalog)
CCD_TO_MAP = 17              # CCD to plot when SHOW_PLOT=True (1-20, or "CCD05")
HOVER_RADIUS_PIX = 4        # show charge when cursor is within this many px of a star
SN_COLORMAP = LinearSegmentedColormap.from_list(
    "teal_orange", ["#008080", "#FF8C00"], N=256
)  # faint stars = teal, bright = orange
SN_COLOR_VMAX_PCT = 99      # color scale vmax = this percentile of detected S/N
STAR_MARKER_RADIUS = 0      # 0 = single center pixel per star

# g-band histogram (SHOW_HIST or: python star_count.py hist)
HIST_BINS = 100
HIST_GMAG_MIN = 12.0        # brighter (lower g) ←——→ fainter (higher g)
HIST_GMAG_MAX = 24.0
HIST_LOG_Y = True           # log scale helps ~10M catalog sources vs ~200k detections
HIST_SAVE = False           # write PNG alongside interactive window
HIST_SKY_DEDUP_ARCSEC = 0.01  # dedupe SMASH rows by sky position (id is NOT unique)

# Photometry zero-point (skippercam_smash.py: ZP_g integrated e- at g=0, one exposure)
EXPOSURE_TIME_S = 8.0
ZP_G_E = 1.64e9
M_REF = 2.5 * np.log10(ZP_G_E)   # reference magnitude tied to ZP_g
FAINT_GMAG_MAX = 99.0            # SMASH sentinel for valid gmag

# SEP (comparison)
SEP_DETECT_SIGMA = 1.5      # per-pixel detection floor (x robust RMS)
SEP_MIN_AREA = 3
SEP_DEBLEND_NTHRESH = 32
SEP_DEBLEND_CONT = 5e-3
FIXED_APERTURE_PIX = 3.0    # point-source aperture radius (~2 PSF sigma)

# Background mesh (SEP)
BG_BW, BG_BH, BG_FW, BG_FH = 64, 64, 3, 3

# Local-threshold method
LOCAL_BLOCK_SIZE = 51       # odd; adaptive-threshold neighborhood [pix]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FITS_PATH = _PROJECT_ROOT / "FITS" / "skippercam_sim_v4.fits"
MAPS_DIR = _PROJECT_ROOT / "simular_curvitas" / "star_maps"
SMASH_CSV = _PROJECT_ROOT / "lmc_smash_g99_v3.csv"
HIST_PATH = MAPS_DIR / f"gmag_hist_sn{int(SN_THRESHOLD)}.png"

LABEL_STRUCTURE = np.ones((3, 3), dtype=int)

# Rough per-CCD runtime (seconds) for ETA lines — measured on M1-class hardware.
_SEC_PER_CCD_MATCHED_REFINE = 6.0
_SEC_PER_CCD_MATCHED_FAST = 0.6
_SEC_PER_CCD_COUNTS_EXTRA = 2.5   # SEP + threshold methods in run_counts()


def _seconds_per_ccd(matched_only: bool = True) -> float:
    base = _SEC_PER_CCD_MATCHED_REFINE if USE_CLUMP_REFINE else _SEC_PER_CCD_MATCHED_FAST
    if matched_only:
        return base
    return base + _SEC_PER_CCD_COUNTS_EXTRA


def _iter_ccds(hdul: fits.HDUList, desc: str, *, matched_only: bool = True):
    """Yield (hdu_idx, name) with a tqdm bar and printed ETA."""
    ccds = ccd_extensions(hdul)
    est = len(ccds) * _seconds_per_ccd(matched_only)
    print(
        f"{desc}: {len(ccds)} CCDs, ~{est:.0f}s estimated "
        f"(clump refine={'on' if USE_CLUMP_REFINE else 'off'})",
        flush=True,
    )
    return tqdm(ccds, desc=desc, unit="CCD", total=len(ccds))

# Crowded field: raise SEP internal limits so deep extraction doesn't overflow.
sep.set_extract_pixstack(10_000_000)
sep.set_sub_object_limit(4096)


def gaussian_kernel(sigma: float, trunc: float = 4.0) -> np.ndarray:
    """Normalized 2D Gaussian (sum = 1)."""
    radius = int(max(1, round(trunc * sigma)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    k = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float64)


PSF_KERNEL = gaussian_kernel(PSF_SIGMA_PIX)
KERNEL_SQ_SUM = float(np.sum(PSF_KERNEL ** 2))          # = 1 / (4 pi sigma^2)
SEP_KERNEL = PSF_KERNEL.astype(np.float32)
PSF_AREA_PIX = float(np.pi * (2.0 * PSF_SIGMA_PIX) ** 2)


def _matched_filter_maps(
    img_sub: np.ndarray, var: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (corr, noise, snr_map) from background-subtracted image."""
    corr = ndi.convolve(img_sub.astype(np.float64), PSF_KERNEL, mode="nearest")
    noise = np.sqrt(ndi.convolve(var, PSF_KERNEL ** 2, mode="nearest"))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_map = np.where(noise > 0, corr / noise, 0.0)
    return corr, noise, snr_map


def _subtract_psf_stamp(
    work: np.ndarray, cy: int, cx: int, flux: float
) -> None:
    """Subtract flux * normalized PSF centered at (cy, cx) from work in-place."""
    kh, kw = PSF_KERNEL.shape
    r = kh // 2
    h, w = work.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    ky0 = r - (cy - y0)
    kx0 = r - (cx - x0)
    ky1 = ky0 + (y1 - y0)
    kx1 = kx0 + (x1 - x0)
    work[y0:y1, x0:x1] -= flux * PSF_KERNEL[ky0:ky1, kx0:kx1]


def _min_peak_distance(min_dist: float | None = None) -> float:
    return MIN_PEAK_DISTANCE if min_dist is None else min_dist


def _peaks_too_close(
    py: int,
    px: int,
    peaks: np.ndarray,
    min_dist: float | None = None,
) -> bool:
    """True if (py, px) is strictly inside the min-distance exclusion zone of any peak."""
    if len(peaks) == 0:
        return False
    r = _min_peak_distance(min_dist)
    pt = np.array([py, px], dtype=np.float64)
    if len(peaks) < 48:
        d2 = (peaks[:, 0] - py) ** 2 + (peaks[:, 1] - px) ** 2
        return bool(np.min(d2) < r * r)
    dist, _ = cKDTree(peaks.astype(np.float64, copy=False)).query(pt, k=1)
    return float(dist) < r


def check_no_double_count(
    peaks: np.ndarray,
    min_dist: float | None = None,
) -> tuple[int, float]:
    """Return (n_close_pairs, nearest_neighbour_separation_px).

    A close pair is two peaks with separation < min_dist. Uses query_pairs so
    each violating pair is counted once (not once per peak). With
    peak_local_max enforcing min_dist, violations should be 0 on every CCD.
    """
    if len(peaks) < 2:
        return 0, float("inf")
    r = _min_peak_distance(min_dist)
    pts = peaks.astype(np.float64, copy=False)
    tree = cKDTree(pts)
    nn_dist, _ = tree.query(pts, k=2)
    min_sep = float(nn_dist[:, 1].min())
    violations = len(tree.query_pairs(r - 1e-9))
    return violations, min_sep


def _iterative_peaks_in_roi(
    work: np.ndarray,
    var: np.ndarray,
    sn_threshold: float,
    global_peaks: np.ndarray,
    region_mask: np.ndarray,
    y0: int,
    x0: int,
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    """Find extra peaks in a clump by subtracting PSFs and re-detecting locally."""
    found_yx: list[tuple[int, int]] = []
    found_flux: list[float] = []
    found_snr: list[float] = []

    for _ in range(CLUMP_ITER_ROUNDS):
        corr, _, snr_local = _matched_filter_maps(work, var)
        snr_local = np.where(region_mask, snr_local, 0.0)

        local_peaks = peak_local_max(
            snr_local,
            min_distance=MIN_PEAK_DISTANCE,
            threshold_abs=sn_threshold,
            num_peaks=CLUMP_MAX_PEAKS_PER_ROUND,
        )
        if len(local_peaks) == 0:
            break

        added = 0
        for py, px in local_peaks:
            gy, gx = y0 + py, x0 + px
            if _peaks_too_close(gy, gx, global_peaks, MIN_PEAK_DISTANCE):
                continue
            if found_yx and _peaks_too_close(
                gy, gx, np.array(found_yx), MIN_PEAK_DISTANCE
            ):
                continue

            snr_val = float(snr_local[py, px])
            corr_peak = float(
                _parabolic_peak_value(corr, np.array([py]), np.array([px]))[0]
            )
            flux = corr_peak / KERNEL_SQ_SUM
            found_yx.append((gy, gx))
            found_flux.append(flux)
            found_snr.append(snr_val)
            _subtract_psf_stamp(work, py, px, flux)
            added += 1

        if added == 0:
            break

    return found_yx, found_flux, found_snr


def _clump_is_blended(prop) -> bool:
    """Heuristic: blob shape/size inconsistent with a single Gaussian PSF."""
    if prop.area < CLUMP_MIN_AREA or prop.area > CLUMP_MAX_AREA:
        return False
    score = 0
    if prop.area > CLUMP_AREA_FACTOR * PSF_AREA_PIX:
        score += 1
    if prop.eccentricity > CLUMP_ECCENTRICITY:
        score += 1
    if prop.extent < CLUMP_EXTENT_MAX:
        score += 1
    return score >= 2


def _refine_clump_peaks(
    img_sub: np.ndarray,
    var: np.ndarray,
    snr_map: np.ndarray,
    peaks: np.ndarray,
    snr_at_peaks: np.ndarray,
    flux_at_peaks: np.ndarray,
    sn_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add stars inside blended clumps using shape analysis + local deconvolution."""
    if not USE_CLUMP_REFINE or len(peaks) == 0:
        return peaks, snr_at_peaks, flux_at_peaks

    clump_thresh = max(2.0, sn_threshold * CLUMP_SN_FRAC)
    labels, _ = ndi.label(snr_map > clump_thresh, structure=LABEL_STRUCTURE)

    extra_peaks: list[list[int]] = []
    extra_snr: list[float] = []
    extra_flux: list[float] = []

    for prop in measure.regionprops(labels, intensity_image=snr_map):
        if not _clump_is_blended(prop):
            continue

        ymin, xmin, ymax, xmax = prop.bbox
        y0 = max(0, ymin - CLUMP_PAD_PIX)
        x0 = max(0, xmin - CLUMP_PAD_PIX)
        y1 = min(img_sub.shape[0], ymax + CLUMP_PAD_PIX)
        x1 = min(img_sub.shape[1], xmax + CLUMP_PAD_PIX)

        region_mask = labels[y0:y1, x0:x1] == prop.label
        peaks_in = []
        for i, (py, px) in enumerate(peaks):
            if y0 <= py < y1 and x0 <= px < x1 and region_mask[py - y0, px - x0]:
                peaks_in.append(i)

        # flux-based expectation: clump S/N sum suggests multiple sources
        clump_snr_sum = float(np.sum(snr_map[y0:y1, x0:x1] * region_mask))
        n_expected = max(1, int(round(clump_snr_sum / max(sn_threshold, 1.0))))
        n_have = len(peaks_in)
        if n_have >= n_expected:
            continue

        max_new = max(1, n_expected - n_have)
        work = img_sub[y0:y1, x0:x1].astype(np.float64, copy=True)
        var_crop = var[y0:y1, x0:x1]
        for idx in peaks_in:
            py, px = peaks[idx]
            _subtract_psf_stamp(work, py - y0, px - x0, flux_at_peaks[idx])

        new_yx, new_flux, new_snr = _iterative_peaks_in_roi(
            work,
            var_crop,
            sn_threshold,
            peaks,
            region_mask,
            y0,
            x0,
        )

        n_added_clump = 0
        for (gy, gx), f, s in zip(new_yx, new_flux, new_snr):
            if _peaks_too_close(gy, gx, peaks, MIN_PEAK_DISTANCE):
                continue
            if extra_peaks and _peaks_too_close(
                gy, gx, np.array(extra_peaks), MIN_PEAK_DISTANCE
            ):
                continue
            extra_peaks.append([gy, gx])
            extra_flux.append(f)
            extra_snr.append(s)
            n_added_clump += 1
            if n_added_clump >= max_new:
                break

    if not extra_peaks:
        return peaks, snr_at_peaks, flux_at_peaks

    peaks = np.vstack([peaks, np.array(extra_peaks, dtype=np.intp)])
    snr_at_peaks = np.concatenate([snr_at_peaks, np.array(extra_snr)])
    flux_at_peaks = np.concatenate([flux_at_peaks, np.array(extra_flux)])
    return peaks, snr_at_peaks, flux_at_peaks


def ccd_extensions(hdul: fits.HDUList) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i, hdu in enumerate(hdul):
        if hdu.data is None:
            continue
        out.append((i, hdu.name or f"CCD{i}"))
    return out


def estimate_background(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (background, img_minus_background) using SEP's mesh estimator."""
    img = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(img, bw=BG_BW, bh=BG_BH, fw=BG_FW, fh=BG_FH)
    back = bkg.back()
    return back, img - back


# ----------------------------------------------------------------------------
# Method 1: matched-filter peak detection (PRIMARY)
# ----------------------------------------------------------------------------
def detect_matched_filter(data: np.ndarray, sn_threshold: float) -> dict:
    """Detect star peaks via a PSF matched filter.

    Returns a dict with:
      peaks    : (N, 2) array of (y, x) integer peak positions
      flux     : (N,) total electrons per detected star
      snr      : (N,) integrated S/N at each peak
      snr_map  : 2D matched-filter S/N image
      img      : raw CCD image (float)
      med_back : median background [e-/pix]
    """
    img = np.asarray(data, dtype=np.float64)
    back, img_sub = estimate_background(data)
    img_sub = np.ascontiguousarray(img_sub, dtype=np.float32)
    med_back = float(np.median(back))

    # Physical per-pixel variance: Poisson(background) + readout^2.
    var = np.maximum(back, 0.0).astype(np.float64) + READ_NOISE_E ** 2

    # Matched filter: correlation with the PSF and its propagated noise.
    corr, noise, snr_map = _matched_filter_maps(img_sub, var)

    peaks = peak_local_max(
        snr_map,
        min_distance=MIN_PEAK_DISTANCE,
        threshold_abs=sn_threshold,
    )
    if len(peaks) == 0:
        empty = np.array([])
        return {
            "peaks": peaks, "bright_pix": peaks, "flux": empty, "snr": empty,
            "snr_map": snr_map, "img": img, "med_back": med_back,
        }

    snr = snr_map[peaks[:, 0], peaks[:, 1]]
    corr_peak = _parabolic_peak_value(corr, peaks[:, 0], peaks[:, 1])
    flux = corr_peak / KERNEL_SQ_SUM

    # Split blended clumps using blob shape + local iterative PSF subtraction.
    peaks, snr, flux = _refine_clump_peaks(
        img_sub, var, snr_map, peaks, snr, flux, sn_threshold
    )

    # Snap each detection to the brightest RAW pixel in a small window so the
    # plotted marker sits on the star's true peak pixel (not the smoothed peak).
    by, bx = _brightest_pixels(img, peaks, radius=MIN_PEAK_DISTANCE)

    # Two peaks may snap onto the same brightest pixel -> keep one (no dup).
    bright = np.stack([by, bx], axis=1)
    _, keep = np.unique(bright, axis=0, return_index=True)
    keep.sort()
    by, bx = by[keep], bx[keep]
    peaks, snr = peaks[keep], snr[keep]
    flux = flux[keep]

    return {
        "peaks": peaks,                       # S/N-map peak (y, x)
        "bright_pix": np.stack([by, bx], 1),  # brightest raw pixel (y, x)
        "flux": flux, "snr": snr,
        "snr_map": snr_map, "img": img, "med_back": med_back,
    }


def _parabolic_peak_value(
    corr: np.ndarray, ys: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    """Sub-pixel parabolic estimate of the correlation peak amplitude."""
    h, w = corr.shape
    c = corr[ys, xs]
    lx = corr[ys, np.clip(xs - 1, 0, w - 1)]
    rx = corr[ys, np.clip(xs + 1, 0, w - 1)]
    ly = corr[np.clip(ys - 1, 0, h - 1), xs]
    ry = corr[np.clip(ys + 1, 0, h - 1), xs]
    den_x, den_y = (lx - 2 * c + rx), (ly - 2 * c + ry)
    dx = np.where(den_x != 0, 0.5 * (lx - rx) / den_x, 0.0)
    dy = np.where(den_y != 0, 0.5 * (ly - ry) / den_y, 0.0)
    dx, dy = np.clip(dx, -0.5, 0.5), np.clip(dy, -0.5, 0.5)
    return c - 0.25 * ((lx - rx) * dx + (ly - ry) * dy)


def _brightest_pixels(
    img: np.ndarray, peaks: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each peak, find the brightest raw pixel within +/- radius."""
    h, w = img.shape
    by = np.empty(len(peaks), dtype=np.intp)
    bx = np.empty(len(peaks), dtype=np.intp)
    for i in range(len(peaks)):
        py, px = peaks[i]
        y0, y1 = max(0, py - radius), min(h, py + radius + 1)
        x0, x1 = max(0, px - radius), min(w, px + radius + 1)
        win = img[y0:y1, x0:x1]
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        by[i], bx[i] = y0 + dy, x0 + dx
    return by, bx


def count_matched_filter(
    data: np.ndarray, sn_threshold: float
) -> tuple[int, np.ndarray, float]:
    """Return (n_peaks, total_fluxes_e, median_background)."""
    res = detect_matched_filter(data, sn_threshold)
    return len(res["peaks"]), res["flux"], res["med_back"]


# ----------------------------------------------------------------------------
# Method 2: SEP with integrated S/N cut  (get_lumi_curves.py style)
# ----------------------------------------------------------------------------
def count_sep(data: np.ndarray, sn_threshold: float) -> int:
    back, img_sub = estimate_background(data)
    bkg = sep.Background(
        np.ascontiguousarray(data, dtype=np.float32),
        bw=BG_BW, bh=BG_BH, fw=BG_FW, fh=BG_FH,
    )
    err_detect = bkg.rms()
    err_phot = np.sqrt(np.maximum(back, 0.0) + READ_NOISE_E ** 2).astype(np.float32)

    try:
        objects = sep.extract(
            img_sub,
            SEP_DETECT_SIGMA,
            err=err_detect,
            minarea=SEP_MIN_AREA,
            filter_kernel=SEP_KERNEL,
            deblend_nthresh=SEP_DEBLEND_NTHRESH,
            deblend_cont=SEP_DEBLEND_CONT,
            clean=True,
        )
    except Exception as exc:  # deblend/pixstack overflow on densest CCDs
        print(f"    [SEP warning] {exc}")
        return 0
    if len(objects) == 0:
        return 0

    flux, fluxerr, _ = sep.sum_circle(
        img_sub, objects["x"], objects["y"],
        r=FIXED_APERTURE_PIX, err=err_phot, gain=GAIN,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = np.where(fluxerr > 0, flux / fluxerr, 0.0)
    return int(((sn >= sn_threshold) & (flux > 0)).sum())


# ----------------------------------------------------------------------------
# Method 3: global threshold connected components (cluster.py style)
# ----------------------------------------------------------------------------
def count_global_threshold(data: np.ndarray, sn_threshold: float) -> int:
    img = np.asarray(data, dtype=np.float64)
    med = np.median(img)
    mad = np.median(np.abs(img - med))
    sigma = 1.4826 * mad if mad > 0 else img.std()
    _, nlabels = ndi.label(img > med + sn_threshold * sigma, structure=LABEL_STRUCTURE)
    return int(nlabels)


# ----------------------------------------------------------------------------
# Method 4: local (adaptive) threshold connected components
# ----------------------------------------------------------------------------
def count_local_threshold(data: np.ndarray, sn_threshold: float) -> int:
    img = np.asarray(data, dtype=np.float64)
    med = np.median(img)
    mad = np.median(np.abs(img - med))
    sigma = 1.4826 * mad if mad > 0 else img.std()
    local_bg = threshold_local(img, block_size=LOCAL_BLOCK_SIZE, method="gaussian")
    _, nlabels = ndi.label(img > local_bg + sn_threshold * sigma, structure=LABEL_STRUCTURE)
    return int(nlabels)


def theoretical_limit(sn_threshold: float, background_e: float) -> float:
    """S/N-limiting total flux for an isolated Gaussian PSF on a given background.

    noise/pix = sqrt(background + RN^2); noise-equivalent area = 4*pi*sigma^2.
    """
    sigma_pix_sq = background_e + READ_NOISE_E * READ_NOISE_E
    n_pix = 4.0 * np.pi * PSF_SIGMA_PIX * PSF_SIGMA_PIX
    a, b, c = 1.0, -(sn_threshold ** 2), -(sn_threshold ** 2) * n_pix * sigma_pix_sq
    return float((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a))


def resolve_ccd(hdul: fits.HDUList, selector) -> tuple[int, str]:
    """Map a selector (1-based index, or name like 'CCD05') to (hdu_idx, name)."""
    ccds = ccd_extensions(hdul)
    if isinstance(selector, str) and selector.isdigit():
        selector = int(selector)
    if isinstance(selector, int):
        if not 1 <= selector <= len(ccds):
            raise ValueError(
                f"CCD {selector} out of range 1..{len(ccds)}"
            )
        return ccds[selector - 1]
    for hdu_idx, name in ccds:  # match by name
        if name.lower() == str(selector).lower():
            return hdu_idx, name
    raise ValueError(f"CCD '{selector}' not found")


def _paint_star_centers(
    rgb: np.ndarray,
    bright: np.ndarray,
    snr: np.ndarray,
    snr_vmin: float,
    snr_vmax: float,
    radius: int = STAR_MARKER_RADIUS,
) -> None:
    """Paint saturated S/N colors onto star center pixels (visible on gray)."""
    cmap = plt.colormaps[SN_COLORMAP] if isinstance(SN_COLORMAP, str) else SN_COLORMAP
    norm = Normalize(vmin=snr_vmin, vmax=snr_vmax)
    h, w = rgb.shape[:2]
    colors = cmap(norm(snr))[:, :3]
    for i in range(len(bright)):
        y, x = int(bright[i, 0]), int(bright[i, 1])
        c = colors[i]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    rgb[yy, xx] = c


def _attach_status_bar(
    ax: plt.Axes,
    bright_pix: np.ndarray,
    flux: np.ndarray,
    snr: np.ndarray,
    img: np.ndarray,
    radius_pix: float = HOVER_RADIUS_PIX,
) -> None:
    """Show charge in the bottom status bar only (no floating label)."""
    h, w = img.shape
    tree = cKDTree(bright_pix.astype(np.float64)) if len(flux) else None

    def format_coord(x: float, y: float) -> str:
        if x is None or y is None:
            return ""
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return ""
        pix = float(img[yi, xi])
        msg = f"({x:.1f}, {y:.1f}) = [{pix:.1f} e-]"
        if tree is not None:
            dist, idx = tree.query([yi, xi])
            if dist <= radius_pix:
                msg += f"  star [{flux[idx]:.1f} e-  S/N {snr[idx]:.1f}]"
        return msg

    ax.format_coord = format_coord


def make_star_map(selector, show: bool = True, save: bool = True) -> int:
    """Render a map of all detected stars for one CCD; return the star count."""
    est = _seconds_per_ccd(matched_only=True)
    t0 = time.time()
    with fits.open(FITS_PATH) as hdul:
        hdu_idx, name = resolve_ccd(hdul, selector)
        print(f"Detecting on {name} (~{est:.0f}s)...", flush=True)
        data = hdul[hdu_idx].data

    res = detect_matched_filter(data, SN_THRESHOLD)
    peaks = res["peaks"]
    bright = res["bright_pix"]
    flux = res["flux"]
    snr = res["snr"]
    n = len(peaks)
    violations, min_sep = check_no_double_count(peaks)

    # Grayscale percentile stretch over positive pixels, matching
    # skippercam_smash.py display_image().
    img = res["img"]
    finite = img[np.isfinite(img)]
    sample = finite[finite > 0] if np.any(finite > 0) else finite
    vmin, vmax = np.percentile(sample, [1.0, 99.0])
    if vmin == vmax:
        vmax = vmin + 1.0

    gray = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    snr_vmax = SN_THRESHOLD
    if n:
        snr_vmax = float(np.percentile(snr, SN_COLOR_VMAX_PCT))
        if snr_vmax <= SN_THRESHOLD:
            snr_vmax = float(snr.max())
        _paint_star_centers(rgb, bright, snr, SN_THRESHOLD, snr_vmax)

    height, width = img.shape
    fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    ax.imshow(rgb, origin="lower", interpolation="nearest")

    if n:
        sm = cm.ScalarMappable(
            norm=Normalize(vmin=SN_THRESHOLD, vmax=snr_vmax),
            cmap=SN_COLORMAP,
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("S/N")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(
        f"{name}: {n:,} stars (S/N >= {SN_THRESHOLD:g})  |  "
        f"min sep = {min_sep:.2f} px, duplicate pairs = {violations}"
    )
    fig.tight_layout()

    if show:
        _attach_status_bar(ax, bright, flux, snr, img)

    if save:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        out = MAPS_DIR / f"star_map_{name}_sn{SN_THRESHOLD:g}.png"
        fig.savefig(out, dpi=150)
        print(f"  saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(
        f"{name}: {n:,} stars | nearest-neighbour min sep {min_sep:.2f} px "
        f"(>= {MIN_PEAK_DISTANCE}) | duplicate pairs {violations} "
        f"-> {'NO double counting' if violations == 0 else 'DUPLICATES FOUND'}"
    )
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return n


# ----------------------------------------------------------------------------
# g-band histogram: detected stars vs SMASH input catalog
# ----------------------------------------------------------------------------
def phot_calib() -> tuple[float, float]:
    """Return (exposure_time_s, m_ref) from FITS primary header when present."""
    with fits.open(FITS_PATH) as hdul:
        hdr = hdul[0].header
        exp = float(hdr.get("EXPTIME", EXPOSURE_TIME_S))
        m_ref = float(hdr.get("ZPREF", M_REF))
    return exp, m_ref


def flux_to_gmag(
    flux_e: np.ndarray,
    exposure_time_s: float | None = None,
    m_ref_mag: float | None = None,
) -> np.ndarray:
    """Invert skippercam_smash.get_charge (deterministic; no Poisson draw).

    rate = 10^(-0.4 * (g - m_ref))  =>  g = m_ref - 2.5 * log10(flux / exposure)
    """
    if exposure_time_s is None or m_ref_mag is None:
        exp_hdr, m_ref_hdr = phot_calib()
        exposure_time_s = exposure_time_s if exposure_time_s is not None else exp_hdr
        m_ref_mag = m_ref_mag if m_ref_mag is not None else m_ref_hdr

    flux_e = np.asarray(flux_e, dtype=np.float64)
    rate = np.maximum(flux_e / exposure_time_s, 1e-300)
    return m_ref_mag - 2.5 * np.log10(rate)


def collect_detected_fluxes(sn_threshold: float | None = None) -> np.ndarray:
    """Run matched-filter detection on all CCDs; return concatenated flux [e-]."""
    sn_threshold = SN_THRESHOLD if sn_threshold is None else sn_threshold
    chunks: list[np.ndarray] = []
    with fits.open(FITS_PATH) as hdul:
        bar = _iter_ccds(hdul, "Detecting for histogram", matched_only=True)
        for hdu_idx, _name in bar:
            data = hdul[hdu_idx].data
            res = detect_matched_filter(data, sn_threshold)
            if len(res["flux"]):
                chunks.append(res["flux"])
            bar.set_postfix_str(f"{len(res['flux']):,} stars", refresh=False)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def _import_skippercam_smash():
    """Import skippercam_smash with project root on path (loads SMASH CSV once)."""
    import os
    import sys

    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    prev = os.getcwd()
    os.chdir(root)
    try:
        import skippercam_smash as scs  # noqa: WPS433 — lazy import by design
    finally:
        os.chdir(prev)
    return scs


def catalog_gmag_focal_plane() -> np.ndarray:
    """SMASH gmag for sources deposited on the 20-CCD focal plane.

    SMASH ``id`` is NOT unique (many rows share an id at different RA/Dec).
    We dedupe by sky position (same rule as unique sources on the chips).
    Each row with gmag < 99 and center strictly on a CCD is one deposited source.
    """
    scs = _import_skippercam_smash()
    width, height = scs.ccd_width_px, scs.ccd_height_px
    round_deg = HIST_SKY_DEDUP_ARCSEC / 3600.0

    ra_chunks: list[np.ndarray] = []
    dec_chunks: list[np.ndarray] = []
    g_chunks: list[np.ndarray] = []

    print(f"Loading SMASH catalog from {SMASH_CSV.name}...", flush=True)
    for ccd_id in tqdm(range(1, 21), desc="SMASH catalog", unit="CCD"):
        ccd_ra, ccd_dec = scs.ccd_sky_center(ccd_id)
        ra_r, dec_r, pad_ra, pad_dec = scs.ccd_sky_extents(ccd_dec)
        patch = scs.get_smash_data(
            ccd_ra, ccd_dec, ra_r + pad_ra, dec_r + pad_dec
        )
        stars = patch[patch["gmag"] < FAINT_GMAG_MAX]
        if stars.empty:
            continue
        x, y = scs.sky_to_pixel(
            stars["ra"].to_numpy(),
            stars["dec"].to_numpy(),
            ccd_ra,
            ccd_dec,
            scs.pix_size,
            width,
            height,
        )
        on_chip = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        sub = stars.iloc[on_chip]
        if sub.empty:
            continue
        ra_chunks.append(np.round(sub["ra"].to_numpy() / round_deg) * round_deg)
        dec_chunks.append(np.round(sub["dec"].to_numpy() / round_deg) * round_deg)
        g_chunks.append(sub["gmag"].to_numpy(dtype=np.float64))

    if not ra_chunks:
        return np.array([], dtype=np.float64)

    ra_all = np.concatenate(ra_chunks)
    dec_all = np.concatenate(dec_chunks)
    g_all = np.concatenate(g_chunks)
    _, keep = np.unique(np.column_stack([ra_all, dec_all]), axis=0, return_index=True)
    keep.sort()
    return g_all[keep]


def make_gmag_histogram(show: bool = True, save: bool | None = None) -> None:
    """Side-by-side g-band histograms: SMASH input vs matched-filter detections."""
    if save is None:
        save = HIST_SAVE
    exp_s, m_ref = phot_calib()
    t0 = time.time()

    catalog_g_all = catalog_gmag_focal_plane()
    flux_e = collect_detected_fluxes()
    detected_g_all = flux_to_gmag(flux_e, exp_s, m_ref)

    in_range = lambda g: (g >= HIST_GMAG_MIN) & (g <= HIST_GMAG_MAX)
    catalog_g = catalog_g_all[in_range(catalog_g_all)]
    detected_g = detected_g_all[in_range(detected_g_all)]

    n_cat_g20 = int((catalog_g_all < 20).sum())
    print(
        f"\nCatalog context: {len(catalog_g_all):,} sources on focal plane "
        f"({n_cat_g20:,} with g < 20). "
        f"Full CSV has ~3.1M with g < 20 over the whole LMC field — "
        f"not the same as this 20-CCD patch.",
        flush=True,
    )

    bins = np.linspace(HIST_GMAG_MIN, HIST_GMAG_MAX, HIST_BINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=HIST_LOG_Y)

    cat_label = (
        f"SMASH deposited on 20 CCDs\n"
        f"{len(catalog_g):,} in [{HIST_GMAG_MIN:g}, {HIST_GMAG_MAX:g}] "
        f"({len(catalog_g_all):,} total)"
    )
    det_label = (
        f"Detected (matched filter, S/N ≥ {SN_THRESHOLD:g})\n"
        f"{len(detected_g):,} in range ({len(detected_g_all):,} total)"
    )

    for ax, data, title, color in zip(
        axes,
        [catalog_g, detected_g],
        [cat_label, det_label],
        ["#4477AA", "#EE6677"],
    ):
        ax.hist(
            data,
            bins=bins,
            histtype="stepfilled",
            alpha=0.55,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_xlabel("g-band magnitude")
        ax.set_title(title, fontsize=10)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        if HIST_LOG_Y:
            ax.set_yscale("log")

    axes[0].set_ylabel("count" + (" (log)" if HIST_LOG_Y else ""))
    fig.suptitle(
        f"SkipperCam focal plane  |  {FITS_PATH.name}  |  "
        f"EXPTIME={exp_s:g}s  m_ref={m_ref:.3f}",
        fontsize=12,
    )
    fig.tight_layout()

    if save:
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(HIST_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved {HIST_PATH}")

    print(
        f"Histogram bins [{HIST_GMAG_MIN:g}, {HIST_GMAG_MAX:g}]: "
        f"catalog {len(catalog_g):,}  |  detected {len(detected_g):,}  |  "
        f"detected/catalog in range: "
        f"{100.0 * len(detected_g) / max(len(catalog_g), 1):.2f}%  |  "
        f"elapsed {time.time() - t0:.1f}s"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_total():
    """Count stars on all CCDs (matched filter) and print focal-plane totals."""
    t0 = time.time()
    mf_counts: dict[str, int] = {}
    faint_fluxes: list[np.ndarray] = []
    backgrounds: list[float] = []

    with fits.open(FITS_PATH) as hdul:
        bar = _iter_ccds(hdul, "Counting stars", matched_only=True)
        for hdu_idx, name in bar:
            data = hdul[hdu_idx].data
            n_mf, fluxes, med_back = count_matched_filter(data, SN_THRESHOLD)
            mf_counts[name] = n_mf
            backgrounds.append(med_back)
            if len(fluxes):
                faint_fluxes.append(fluxes)
            bar.set_postfix_str(f"{name} {n_mf:,}", refresh=False)

    elapsed = time.time() - t0
    total_mf = sum(mf_counts.values())
    all_flux = np.concatenate(faint_fluxes) if faint_fluxes else np.array([])
    mean_back = float(np.mean(backgrounds)) if backgrounds else 0.0

    print(f"\nS/N threshold: {SN_THRESHOLD:g}  |  input: {FITS_PATH.name}")
    print(f"Total stars (20 CCDs, matched filter): {total_mf:,}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / max(len(mf_counts), 1):.1f}s / CCD)")

    if len(all_flux):
        print(
            f"Faintest confident detection: {all_flux.min():.1f} e-  "
            f"(1st pct {np.percentile(all_flux, 1):.1f} e-, "
            f"median {np.median(all_flux):.1f} e-)"
        )
        print(
            f"Theoretical {SN_THRESHOLD:g}-S/N limit: "
            f"{theoretical_limit(SN_THRESHOLD, mean_back):.1f} e-"
        )

    return total_mf


def run_counts():
    print(f"S/N threshold: {SN_THRESHOLD}")
    print(f"Input: {FITS_PATH.name}\n")
    t0 = time.time()

    mf_counts: dict[str, int] = {}
    sep_counts: dict[str, int] = {}
    global_counts: dict[str, int] = {}
    local_counts: dict[str, int] = {}
    faint_fluxes: list[np.ndarray] = []
    backgrounds: list[float] = []

    with fits.open(FITS_PATH) as hdul:
        bar = _iter_ccds(hdul, "All methods", matched_only=False)
        for hdu_idx, name in bar:
            data = hdul[hdu_idx].data
            n_mf, fluxes, med_back = count_matched_filter(data, SN_THRESHOLD)
            mf_counts[name] = n_mf
            backgrounds.append(med_back)
            if len(fluxes):
                faint_fluxes.append(fluxes)
            sep_counts[name] = count_sep(data, SN_THRESHOLD)
            global_counts[name] = count_global_threshold(data, SN_THRESHOLD)
            local_counts[name] = count_local_threshold(data, SN_THRESHOLD)
            bar.set_postfix_str(f"{name} mf={n_mf:,}", refresh=False)
            tqdm.write(
                f"  {name}: matched-filter={n_mf:6d}  SEP={sep_counts[name]:6d}  "
                f"global={global_counts[name]:6d}  local={local_counts[name]:6d}"
            )

    elapsed = time.time() - t0

    total_mf = sum(mf_counts.values())
    total_sep = sum(sep_counts.values())
    total_global = sum(global_counts.values())
    total_local = sum(local_counts.values())
    all_flux = np.concatenate(faint_fluxes) if faint_fluxes else np.array([])
    mean_back = float(np.mean(backgrounds)) if backgrounds else 0.0

    print("\n" + "=" * 64)
    print(f"FOCAL-PLANE TOTALS (20 CCDs)  -  S/N >= {SN_THRESHOLD:g}")
    print("=" * 64)
    print(f"  Matched-filter peaks (PRIMARY) : {total_mf:>9,}")
    print(f"  SEP (get_lumi_curves style)    : {total_sep:>9,}")
    print(f"  Global threshold (cluster.py)  : {total_global:>9,}")
    print(f"  Local adaptive threshold       : {total_local:>9,}")

    if len(all_flux):
        print("\nFaintest confidently detected star "
              f"(matched filter, S/N >= {SN_THRESHOLD:g}):")
        print(f"  lowest total electrons : {all_flux.min():.1f} e-")
        print(f"  1st percentile         : {np.percentile(all_flux, 1):.1f} e-")
        print(f"  median detected        : {np.median(all_flux):.1f} e-")

    print(f"\nMean focal-plane background : {mean_back:.1f} e-/pix "
          f"(sky + diffuse stellar light)")
    print(f"Theoretical isolated-PSF {SN_THRESHOLD:g}-S/N limit : "
          f"{theoretical_limit(SN_THRESHOLD, mean_back):.1f} e- total")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / 20:.1f}s / CCD)")


def main():
    args = sys.argv[1:]
    if args and args[0] == "map":
        selector = args[1] if len(args) > 1 else CCD_TO_MAP
        if str(selector).lower() == "all":
            with fits.open(FITS_PATH) as hdul:
                names = [name for _, name in ccd_extensions(hdul)]
            est = len(names) * _seconds_per_ccd(matched_only=True)
            print(f"Rendering {len(names)} maps (~{est:.0f}s estimated)", flush=True)
            t0 = time.time()
            for name in tqdm(names, desc="Star maps", unit="CCD"):
                make_star_map(name, show=False, save=True)
            print(f"All maps done in {time.time() - t0:.1f}s")
        else:
            make_star_map(selector, show=True, save=True)
    elif args and args[0] == "counts":
        run_counts()
    elif args and args[0] == "hist":
        make_gmag_histogram(show=True, save=HIST_SAVE)
    elif SHOW_HIST:
        make_gmag_histogram(show=True, save=HIST_SAVE)
    elif SHOW_PLOT:
        make_star_map(CCD_TO_MAP, show=True, save=True)
    else:
        run_total()


if __name__ == "__main__":
    main()
