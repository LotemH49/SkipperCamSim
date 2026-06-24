"""Fast star counting and maps — same logic as star_count.py, parallel over CCDs.

Usage (.venv/bin/python simular_curvitas/star_count_fast.py):
  (no args)       focal-plane count (parallel)
  map [CCD]       single-CCD map, interactive
  map all         all 20 PNGs, parallel detect
  verify [CCD]    confirm fast == star_count.detect_matched_filter
  profile [CCD]   per-CCD detect timing
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any

import matplotlib

if len(sys.argv) > 2 and sys.argv[1] == "map" and sys.argv[2].lower() == "all":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm

import star_count as sc

SN_THRESHOLD = sc.SN_THRESHOLD
FITS_PATH = sc.FITS_PATH
MAPS_DIR = sc.MAPS_DIR
CCD_TO_MAP = sc.CCD_TO_MAP

N_WORKERS = 0  # 0 -> min(20, cpu_count())


def _worker_count() -> int:
    n = N_WORKERS if N_WORKERS > 0 else min(20, cpu_count() or 1)
    return max(1, n)


def detect_matched_filter_fast(
    data: np.ndarray,
    sn_threshold: float,
) -> dict[str, Any]:
    """Identical to star_count.detect_matched_filter (parity guaranteed)."""
    return sc.detect_matched_filter(data, sn_threshold)


def _mp_detect_one(payload: tuple[int, str, np.ndarray, float]) -> tuple:
    _hdu_idx, name, data, sn_thr = payload
    res = detect_matched_filter_fast(data, sn_thr)
    return name, len(res["peaks"]), res["flux"], res["med_back"], res


def _mp_map_one(payload: tuple[int, str, np.ndarray, float]) -> tuple:
    _hdu_idx, name, data, sn_thr = payload
    res = detect_matched_filter_fast(data, sn_thr)
    return name, res


def _load_ccd_payloads() -> list[tuple[int, str, np.ndarray, float]]:
    out: list[tuple[int, str, np.ndarray, float]] = []
    with fits.open(FITS_PATH) as hdul:
        for hdu_idx, name in sc.ccd_extensions(hdul):
            data = np.ascontiguousarray(hdul[hdu_idx].data, dtype=np.float64).copy()
            out.append((hdu_idx, name, data, SN_THRESHOLD))
    return out


def render_star_map(
    name: str,
    res: dict[str, Any],
    *,
    show: bool = True,
    save: bool = True,
) -> int:
    peaks = res["peaks"]
    bright = res["bright_pix"]
    flux = res["flux"]
    snr = res["snr"]
    n = len(peaks)
    violations, min_sep = sc.check_no_double_count(bright)

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
        snr_vmax = float(np.percentile(snr, sc.SN_COLOR_VMAX_PCT))
        if snr_vmax <= SN_THRESHOLD:
            snr_vmax = float(snr.max())
        sc._paint_star_centers(rgb, bright, snr, SN_THRESHOLD, snr_vmax)

    height, width = img.shape
    fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    ax.imshow(rgb, origin="lower", interpolation="nearest")

    if n:
        sm = cm.ScalarMappable(
            norm=Normalize(vmin=SN_THRESHOLD, vmax=snr_vmax),
            cmap=sc.SN_COLORMAP,
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
        sc._attach_status_bar(ax, bright, flux, snr, img)

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
        f"(>= {sc.MIN_PEAK_DISTANCE}) | duplicate pairs {violations} "
        f"-> {'NO double counting' if violations == 0 else 'DUPLICATES FOUND'}"
    )
    return n


def make_star_map_fast(selector, show: bool = True, save: bool = True) -> int:
    est = sc._seconds_per_ccd(matched_only=True)
    t0 = time.time()
    with fits.open(FITS_PATH) as hdul:
        _hdu_idx, name = sc.resolve_ccd(hdul, selector)
        print(f"Detecting on {name} (~{est:.0f}s)...", flush=True)
        data = hdul[_hdu_idx].data

    res = detect_matched_filter_fast(data, SN_THRESHOLD)
    render_star_map(name, res, show=show, save=save)
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return len(res["peaks"])


def run_total_fast() -> int:
    t0 = time.time()
    payloads = _load_ccd_payloads()
    n_workers = _worker_count()
    est = len(payloads) * sc._seconds_per_ccd(True) / n_workers
    print(
        f"Counting stars (fast): {len(payloads)} CCDs, "
        f"{n_workers} workers, ~{est:.0f}s est",
        flush=True,
    )

    mf_counts: dict[str, int] = {}
    faint_fluxes: list[np.ndarray] = []
    backgrounds: list[float] = []

    if n_workers == 1:
        bar = tqdm(payloads, desc="Counting", unit="CCD")
        for item in bar:
            name, n_mf, fluxes, med_back, _res = _mp_detect_one(item)
            mf_counts[name] = n_mf
            backgrounds.append(med_back)
            if len(fluxes):
                faint_fluxes.append(fluxes)
            bar.set_postfix_str(f"{name} {n_mf:,}", refresh=False)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_mp_detect_one, p): p[1] for p in payloads}
            bar = tqdm(total=len(futures), desc="Counting", unit="CCD")
            for fut in as_completed(futures):
                name, n_mf, fluxes, med_back, _res = fut.result()
                mf_counts[name] = n_mf
                backgrounds.append(med_back)
                if len(fluxes):
                    faint_fluxes.append(fluxes)
                bar.set_postfix_str(f"{name} {n_mf:,}", refresh=False)
                bar.update(1)
            bar.close()

    elapsed = time.time() - t0
    total_mf = sum(mf_counts.values())
    all_flux = np.concatenate(faint_fluxes) if faint_fluxes else np.array([])
    mean_back = float(np.mean(backgrounds)) if backgrounds else 0.0

    print(f"\nS/N threshold: {SN_THRESHOLD:g}  |  input: {FITS_PATH.name}")
    print(f"Total stars (20 CCDs, matched filter): {total_mf:,}")
    print(
        f"Elapsed: {elapsed:.1f}s ({elapsed / max(len(mf_counts), 1):.1f}s / CCD)  "
        f"[{n_workers} workers]"
    )

    if len(all_flux):
        print(
            f"Faintest confident detection (integrated): {all_flux.min():.1f} e-  "
            f"(1st pct {np.percentile(all_flux, 1):.1f} e-, "
            f"median {np.median(all_flux):.1f} e-)"
        )
        print(
            f"Theoretical {SN_THRESHOLD:g}-S/N limit: "
            f"{sc.theoretical_limit(SN_THRESHOLD, mean_back):.1f} e-"
        )

    return total_mf


def map_all_fast() -> None:
    t0 = time.time()
    payloads = _load_ccd_payloads()
    n_workers = _worker_count()
    est = len(payloads) * sc._seconds_per_ccd(True) / n_workers
    print(
        f"Rendering {len(payloads)} maps (~{est:.0f}s est), {n_workers} workers",
        flush=True,
    )

    results: dict[str, dict] = {}
    if n_workers == 1:
        for item in tqdm(payloads, desc="Detect", unit="CCD"):
            name, res = _mp_map_one(item)
            results[name] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_mp_map_one, p): p[1] for p in payloads}
            bar = tqdm(total=len(futures), desc="Detect", unit="CCD")
            for fut in as_completed(futures):
                name, res = fut.result()
                results[name] = res
                bar.update(1)
            bar.close()

    for name in tqdm(sorted(results), desc="Save PNG", unit="CCD"):
        render_star_map(name, results[name], show=False, save=True)

    print(f"All maps done in {time.time() - t0:.1f}s")


def verify_detection(selector) -> bool:
    with fits.open(FITS_PATH) as hdul:
        _hdu_idx, name = sc.resolve_ccd(hdul, selector)
        data = hdul[_hdu_idx].data

    print(f"Verifying {name} vs star_count.detect_matched_filter ...", flush=True)
    t0 = time.perf_counter()
    ref = sc.detect_matched_filter(data, SN_THRESHOLD)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = detect_matched_filter_fast(data, SN_THRESHOLD)
    t_fast = time.perf_counter() - t0

    rp, rf, rs = ref["peaks"], ref["flux"], ref["snr"]
    fp, ff, fs = fast["peaks"], fast["flux"], fast["snr"]

    ok = True
    if len(rp) != len(fp):
        print(f"FAIL count: ref={len(rp)} fast={len(fp)}")
        ok = False
    else:
        ridx = np.lexsort((rp[:, 1], rp[:, 0]))
        fidx = np.lexsort((fp[:, 1], fp[:, 0]))
        if not np.array_equal(rp[ridx], fp[fidx]):
            print("FAIL positions differ")
            ok = False
        if not np.allclose(rf[ridx], ff[fidx], rtol=1e-9, atol=1e-6):
            print(f"FAIL flux: max diff {np.max(np.abs(rf - rf)):.4g}")
            ok = False
        if not np.allclose(rs[ridx], fs[fidx], rtol=1e-9, atol=1e-6):
            print(f"FAIL snr: max diff {np.max(np.abs(rs - fs)):.4g}")
            ok = False

    if ok:
        print(
            f"PASS  {len(rp):,} stars identical  |  "
            f"detect {t_ref:.1f}s (same code path)"
        )
    return ok


def profile_detect(data: np.ndarray, sn_threshold: float) -> None:
    t0 = time.perf_counter()
    res = detect_matched_filter_fast(data, sn_threshold)
    elapsed = time.perf_counter() - t0
    print(f"detect_matched_filter: {len(res['peaks']):,} stars in {elapsed:.2f}s")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "map":
        selector = args[1] if len(args) > 1 else CCD_TO_MAP
        if str(selector).lower() == "all":
            map_all_fast()
        else:
            make_star_map_fast(selector, show=True, save=True)
    elif args and args[0] == "verify":
        selector = args[1] if len(args) > 1 else CCD_TO_MAP
        sys.exit(0 if verify_detection(selector) else 1)
    elif args and args[0] == "profile":
        selector = args[1] if len(args) > 1 else CCD_TO_MAP
        with fits.open(FITS_PATH) as hdul:
            hdu_idx, name = sc.resolve_ccd(hdul, selector)
            data = hdul[hdu_idx].data
        print(f"Profiling {name} ...", flush=True)
        profile_detect(data, SN_THRESHOLD)
    else:
        run_total_fast()


if __name__ == "__main__":
    main()
