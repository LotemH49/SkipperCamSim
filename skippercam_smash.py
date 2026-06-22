import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

#tunables
ccd_number = 12  # 1-20
write_mef = True
_PROJECT_ROOT = Path(__file__).resolve().parent
mef_path = "FITS/skippercam_sim_v2.fits"
focal_plane_gaps = False  # False = abutting 4x5 grid; True = physical mm gaps
write_focal_plane = False  # auto-stitch after MEF export
focal_plane_path = "FITS/skippercam_focal_plane_sim_v2.fits"
exposure_time = 8  # seconds
readout_noise = 7.0  # e- RMS per pixel (fast survey mode)
psf_sigma_pix = 1.5  # Gaussian PSF σ [pix]; sole PSF tunable (FWHM ≈ 2.355σ)
psf_trunc_sigma = 5  # truncate kernel at this many σ 
psf_pad_pix = math.ceil(psf_trunc_sigma * psf_sigma_pix)  # catalog pad matches kernel
catalog_extra_pad_pix = 2  # extra catalog margin for edge overlap
# Define the LMC coordinates
lmc_ra = 80.89
lmc_dec = -69.76
pix_size = 1.43  # arcsec per pix
pixel_pitch_mm = 15e-3  # 15 µm detector pitch
factor = 1 # for testing
ccd_width_px = 1278
ccd_height_px = 1058
ccd_width_mm = ccd_width_px * pixel_pitch_mm
ccd_height_mm = ccd_height_px * pixel_pitch_mm


_pix_scale_deg = pix_size / 3600.0
_arcsec_per_mm = pix_size / pixel_pitch_mm
_cos_dec = math.cos(math.radians(lmc_dec))
ccd_dec_read_radius = 0.5 * ccd_height_px * _pix_scale_deg

ZP_g = 1.64E+09  # integrated e- for a 0-mag source in one exposure
m_ref = 2.5 * np.log10(ZP_g)  # reference magnitude tied to ZP_g
sky_e_rate = 3.11E+01  # e- / s / pixel (SkipperCam paper Sec. 2.4)[changes with 0 point]
g_lim = 20  # monitored sources: g < g_lim (INFO.md / paper)

psf_pad_deg = psf_pad_pix * pix_size / 3600.0
catalog_pad_deg = (psf_pad_pix + catalog_extra_pad_pix) * pix_size / 3600.0
psf_deposit_batch_size = 50_000  # stars per PSF stamp batch (limits RAM)
faint_gmag_max = 99  # SMASH sentinel for valid magnitudes

# Load SMASH DR2 sample data
smash_df = pd.read_csv("lmc_smash_g99_v3.csv")
#smash_df = pd.read_csv("lmc_onestar_tester.csv")

# CCD center offset from array center (mm). +x = +RA, +y = +Dec. (0, 0) = lmc_ra/lmc_dec.
x_offset = 25.95876
y_offset = 20.2744 
ccd_1_x = -38.93833
ccd_1_y = 40.5488
ccd_location: dict[int, tuple[float, float]] = {
    1: (ccd_1_x , ccd_1_y),
    2: (ccd_1_x + x_offset, ccd_1_y ),
    3: (ccd_1_x + x_offset*2, ccd_1_y ),
    4: (ccd_1_x + x_offset*3, ccd_1_y ),
    5: (ccd_1_x , ccd_1_y - y_offset),
    6: (ccd_1_x + x_offset, ccd_1_y - y_offset),
    7: (ccd_1_x + x_offset*2, ccd_1_y - y_offset),
    8: (ccd_1_x + x_offset*3, ccd_1_y - y_offset),
    9: (ccd_1_x , ccd_1_y - y_offset*2),
    10: (ccd_1_x + x_offset, ccd_1_y - y_offset*2),
    11: (ccd_1_x + x_offset*2, ccd_1_y - y_offset*2),
    12: (ccd_1_x + x_offset*3, ccd_1_y - y_offset*2),
    13: (ccd_1_x , ccd_1_y - y_offset*3),
    14: (ccd_1_x + x_offset, ccd_1_y - y_offset*3),
    15: (ccd_1_x + x_offset*2, ccd_1_y - y_offset*3),
    16: (ccd_1_x + x_offset*3, ccd_1_y - y_offset*3),
    17: (ccd_1_x , ccd_1_y - y_offset*4),
    18: (ccd_1_x + x_offset, ccd_1_y - y_offset*4),
    19: (ccd_1_x + x_offset*2, ccd_1_y - y_offset*4),
    20: (ccd_1_x + x_offset*3, ccd_1_y - y_offset*4),
}

def get_ccd_location(ccd_number):
    return ccd_location[ccd_number]


def resolve_fits_path(path: str | Path) -> Path:
    """Resolve FITS paths under the project; map /FITS/... → project/FITS/..."""
    p = Path(path).expanduser()
    if p.is_absolute() and not p.exists():
        parts = p.parts
        if len(parts) >= 3 and parts[1] == "FITS":
            candidate = _PROJECT_ROOT / "FITS" / Path(*parts[2:])
            if candidate.is_file():
                return candidate.resolve()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()


def ensure_fits_parent(path: str | Path) -> Path:
    """Ensure parent directory exists for a FITS output path."""
    p = resolve_fits_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# Select SMASH catalog rows inside the RA/Dec read box.
def get_smash_data(ra, dec, ra_read_radius, dec_read_radius):
    mask = (
        (smash_df["ra"] >= ra - ra_read_radius)
        & (smash_df["ra"] <= ra + ra_read_radius)
        & (smash_df["dec"] >= dec - dec_read_radius)
        & (smash_df["dec"] <= dec + dec_read_radius)
    )
    return smash_df[mask]


# Convert g-band magnitude to photoelectrons for one exposure.
def get_charge(m_target, m_ref_mag=m_ref, exposure_time_s=exposure_time):
    m_target = np.asarray(m_target, dtype=np.float64)
    rate =  np.power(10.0, -0.4 * (m_target - m_ref_mag))
    return rate * exposure_time_s


# Map sky RA/Dec (deg) to floating-point pixel coordinates on the image.
def sky_to_pixel(ra, dec, center_ra, center_dec, pix_size_arcsec, width, height):
    pix_scale_deg = pix_size_arcsec / 3600.0
    cos_dec = math.cos(math.radians(center_dec))
    dx = (ra - center_ra) * cos_dec
    dy = dec - center_dec
    x = dx / pix_scale_deg + width / 2.0
    y = dy / pix_scale_deg + height / 2.0
    return x, y


def ccd_sky_center(ccd_id):
    """Sky RA/Dec at CCD center; ccd_location is (dx, dy) mm from array center."""
    dx_mm, dy_mm = get_ccd_location(ccd_id)
    ddec_deg = dy_mm * _arcsec_per_mm / 3600.0
    dec = lmc_dec + ddec_deg
    cos_dec = math.cos(math.radians(dec))
    dra_deg = dx_mm * _arcsec_per_mm / 3600.0 / max(cos_dec, 1e-12)
    return lmc_ra + dra_deg, dec


def ccd_sky_extents(center_dec):
    """Half-width of CCD footprint and catalog pad in RA/Dec [deg] at center_dec."""
    cos_dec = math.cos(math.radians(center_dec))
    ra_radius = 0.5 * ccd_width_px * _pix_scale_deg / max(cos_dec, 1e-12)
    dec_radius = 0.5 * ccd_height_px * _pix_scale_deg
    pad_ra = catalog_pad_deg / max(cos_dec, 1e-12)
    pad_dec = catalog_pad_deg
    return ra_radius, dec_radius, pad_ra, pad_dec


def make_ccd_wcs(ra_center, dec_center, width_px, height_px, pix_size_arcsec=pix_size):
    """WCS inverse of sky_to_pixel (+x = +RA, +y = +Dec, origin lower)."""
    pix_scale_deg = pix_size_arcsec / 3600.0
    cos_dec = math.cos(math.radians(dec_center))
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.crpix = [width_px / 2 + 0.5, height_px / 2 + 0.5]
    wcs.wcs.cd = [[pix_scale_deg / cos_dec, 0.0], [0.0, pix_scale_deg]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000.0
    return wcs


def _focal_plane_bounds_mm():
    """Bounding box of all CCDs in mm (+x=+RA, +y=+Dec)."""
    half_w_mm = ccd_width_mm / 2
    half_h_mm = ccd_height_mm / 2
    x_mm: list[float] = []
    y_mm: list[float] = []
    for ccd_id in range(1, 21):
        dx, dy = get_ccd_location(ccd_id)
        x_mm.extend([dx - half_w_mm, dx + half_w_mm])
        y_mm.extend([dy - half_h_mm, dy + half_h_mm])
    return min(x_mm), max(x_mm), min(y_mm), max(y_mm)


def _focal_plane_canvas_shape():
    min_x, max_x, min_y, max_y = _focal_plane_bounds_mm()
    mosaic_w = int(math.ceil((max_x - min_x) / pixel_pitch_mm))
    mosaic_h = int(math.ceil((max_y - min_y) / pixel_pitch_mm))
    return mosaic_w, mosaic_h, min_x, min_y


def _ccd_pixel_origin(ccd_id, min_x_mm, min_y_mm):
    """Lower-left chip corner in focal-plane canvas pixels (origin lower)."""
    dx, dy = get_ccd_location(ccd_id)
    px_per_mm = 1.0 / pixel_pitch_mm
    half_w_mm = ccd_width_mm / 2
    half_h_mm = ccd_height_mm / 2
    x0 = int(round((dx - half_w_mm - min_x_mm) * px_per_mm))
    y0 = int(round((dy - half_h_mm - min_y_mm) * px_per_mm))
    return x0, y0


def _ccd_det_keywords_abut(ccd_id, width_px, height_px):
    """IRAF mosaic section: abutting 4x5 grid (no gaps)."""
    col = (ccd_id - 1) % 4
    row = 4 - (ccd_id - 1) // 4
    x1 = col * width_px + 1
    x2 = (col + 1) * width_px
    y1 = row * height_px + 1
    y2 = (row + 1) * height_px
    mosaic_w = 4 * width_px
    mosaic_h = 5 * height_px
    return {
        "DETSIZE": (f"[1:{mosaic_w},1:{mosaic_h}]", "Full focal-plane detector extent"),
        "DETSEC": (
            f"[{x1}:{x2},{y1}:{y2}]",
            "CCD section on focal plane [1-based]",
        ),
        "CCDSEC": (f"[1:{width_px},1:{height_px}]", "Valid data section"),
    }


def _ccd_det_keywords_gaps(ccd_id, width_px, height_px):
    """IRAF mosaic section from true focal-plane mm layout (includes gaps)."""
    mosaic_w, mosaic_h, min_x, min_y = _focal_plane_canvas_shape()
    x0, y0 = _ccd_pixel_origin(ccd_id, min_x, min_y)
    x1 = x0 + 1
    x2 = x0 + width_px
    y1 = y0 + 1
    y2 = y0 + height_px
    return {
        "DETSIZE": (f"[1:{mosaic_w},1:{mosaic_h}]", "Full focal-plane detector extent"),
        "DETSEC": (
            f"[{x1}:{x2},{y1}:{y2}]",
            "CCD section on focal plane [1-based]",
        ),
        "CCDSEC": (f"[1:{width_px},1:{height_px}]", "Valid data section"),
    }


def _ccd_det_keywords(ccd_id, width_px, height_px, gaps=None):
    use_gaps = focal_plane_gaps if gaps is None else gaps
    if use_gaps:
        return _ccd_det_keywords_gaps(ccd_id, width_px, height_px)
    return _ccd_det_keywords_abut(ccd_id, width_px, height_px)


def _stitch_focal_plane_abut(mef_path, out_path):
    """Stitch MEF into abutting 4x5 grid (no gaps between chips)."""
    mosaic_w = 4 * ccd_width_px
    mosaic_h = 5 * ccd_height_px
    canvas = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
    with fits.open(mef_path) as hdul:
        for ccd_id in range(1, len(hdul)):
            data = hdul[ccd_id].data
            col = (ccd_id - 1) % 4
            row = 4 - (ccd_id - 1) // 4
            y0 = row * ccd_height_px
            x0 = col * ccd_width_px
            canvas[y0 : y0 + ccd_height_px, x0 : x0 + ccd_width_px] = data

    header = fits.Header()
    header["ARRAY_RA"] = lmc_ra
    header["ARRAY_DEC"] = lmc_dec
    header["NCCD"] = 20
    header["BUNIT"] = "electron"
    header["COMMENT"] = "Abutting 4x5 stitch (no gaps)"
    fits.PrimaryHDU(canvas, header=header).writeto(out_path, overwrite=True)
    print(f"Wrote focal-plane stitch: {out_path} ({mosaic_w}x{mosaic_h})", flush=True)


def _stitch_focal_plane_gaps(mef_path, out_path):
    """Stitch MEF using true mm chip positions; gaps are zero pixels."""
    mosaic_w, mosaic_h, min_x, min_y = _focal_plane_canvas_shape()
    canvas = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
    with fits.open(mef_path) as hdul:
        for ccd_id in range(1, len(hdul)):
            data = hdul[ccd_id].data
            height, width = data.shape
            x0, y0 = _ccd_pixel_origin(ccd_id, min_x, min_y)
            canvas[y0 : y0 + height, x0 : x0 + width] = data

    wcs = make_ccd_wcs(lmc_ra, lmc_dec, mosaic_w, mosaic_h)
    header = wcs.to_header(relax=True)
    for key in list(header):
        if key.startswith(("PC", "CDELT")):
            del header[key]
    cd = wcs.wcs.cd
    header["CD1_1"] = cd[0, 0]
    header["CD1_2"] = cd[0, 1]
    header["CD2_1"] = cd[1, 0]
    header["CD2_2"] = cd[1, 1]
    header["RADESYS"] = "ICRS"
    header["EQUINOX"] = 2000.0
    header["NCCD"] = 20
    header["BUNIT"] = "electron"
    header["COMMENT"] = "Focal-plane stitch with physical gaps (mm layout)"
    fits.PrimaryHDU(canvas.astype(np.float32), header=header).writeto(
        out_path, overwrite=True
    )
    gap_x = int(round((x_offset - ccd_width_mm) / pixel_pitch_mm))
    gap_y = int(round((y_offset - ccd_height_mm) / pixel_pitch_mm))
    print(
        f"Wrote focal-plane stitch: {out_path} ({mosaic_w}x{mosaic_h}), "
        f"gap ~{gap_x}x{gap_y} px",
        flush=True,
    )


def stitch_focal_plane(mef_path=mef_path, out_path=focal_plane_path, gaps=None):
    """Stitch MEF to one focal-plane image (default: abutting 4x5, no gaps)."""
    mef_path = resolve_fits_path(mef_path)
    out_path = ensure_fits_parent(out_path)
    if not mef_path.is_file():
        raise FileNotFoundError(f"MEF not found: {mef_path}")
    use_gaps = focal_plane_gaps if gaps is None else gaps
    if use_gaps:
        _stitch_focal_plane_gaps(mef_path, out_path)
    else:
        _stitch_focal_plane_abut(mef_path, out_path)


def _ccd_fits_header(wcs, ccd_id, width_px, height_px, dx_mm, dy_mm, gaps=None):
    """FITS header for one CCD extension (DS9 Mosaic WCS + IRAF DETSEC)."""
    header = wcs.to_header(relax=True)
    for key in list(header):
        if key.startswith(("PC", "CDELT")):
            del header[key]
    cd = wcs.wcs.cd
    header["CD1_1"] = cd[0, 0]
    header["CD1_2"] = cd[0, 1]
    header["CD2_1"] = cd[1, 0]
    header["CD2_2"] = cd[1, 1]
    header["WCSAXES"] = 2
    header["RADESYS"] = "ICRS"
    header["EQUINOX"] = 2000.0
    header["EXTNAME"] = f"CCD{ccd_id:02d}"
    header["CCDID"] = ccd_id
    header["HPUID"] = ccd_id
    header["FPOSX"] = (dx_mm, "mm from array center (+RA)")
    header["FPOSY"] = (dy_mm, "mm from array center (+Dec)")
    header["EXPTIME"] = exposure_time
    header["RDNOISE"] = readout_noise
    header["BUNIT"] = "electron"
    for key, value in _ccd_det_keywords(ccd_id, width_px, height_px, gaps=gaps).items():
        header[key] = value
    return header


def patch_mef_headers(path=mef_path, gaps=None):
    """Rewrite WCS/DETSEC headers on an existing MEF (no re-simulation)."""
    path = resolve_fits_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MEF not found: {path}")
    use_gaps = focal_plane_gaps if gaps is None else gaps
    with fits.open(path, mode="readonly") as hdul:
        primary = fits.PrimaryHDU(data=None, header=hdul[0].header)
        out = fits.HDUList([primary])
        for ccd_id in range(1, len(hdul)):
            data = hdul[ccd_id].data
            height, width = data.shape
            ra, dec = ccd_sky_center(ccd_id)
            dx_mm, dy_mm = get_ccd_location(ccd_id)
            wcs = make_ccd_wcs(ra, dec, width, height)
            header = _ccd_fits_header(
                wcs, ccd_id, width, height, dx_mm, dy_mm, gaps=use_gaps
            )
            out.append(fits.ImageHDU(data.astype(np.float32), header=header))
    out.writeto(path, overwrite=True)
    mode = "gaps" if use_gaps else "abut"
    print(f"Patched headers ({mode}) in {path}", flush=True)


def simulate_ccd(ccd_id):
    """Pixelate and noise-simulate one CCD; returns images, sky center, focal-plane offset, stars."""
    ccd_ra, ccd_dec = ccd_sky_center(ccd_id)
    ra_radius, dec_radius, pad_ra, pad_dec = ccd_sky_extents(ccd_dec)
    smash_data = get_smash_data(
        ccd_ra,
        ccd_dec,
        ra_radius + pad_ra,
        dec_radius + pad_dec,
    )
    ideal_image, stars_on_patch = pixelate_smash_data(
        smash_data,
        ccd_ra,
        ccd_dec,
        ra_radius,
        dec_radius,
    )
    noisy_image = simulate_microchip(ideal_image)
    dx_mm, dy_mm = get_ccd_location(ccd_id)
    return ideal_image, noisy_image, ccd_ra, ccd_dec, dx_mm, dy_mm, stars_on_patch


def _primary_mef_header():
    header = fits.Header()
    header["ARRAY_RA"] = (lmc_ra, "Array center RA [deg]")
    header["ARRAY_DEC"] = (lmc_dec, "Array center Dec [deg]")
    header["NCCD"] = (20, "Number of CCD extensions")
    header["EXPTIME"] = (exposure_time, "Exposure time [s]")
    header["RDNOISE"] = (readout_noise, "Readout noise sigma [e-]")
    header["PIXSIZE"] = (pix_size, "Pixel scale [arcsec/pix]")
    header["BUNIT"] = "electron"
    header["ZPREF"] = (m_ref, "Reference magnitude for ZP_g")
    header["SKYRATE"] = (sky_e_rate, "Sky rate [e-/s/pix]")
    header["PSFSIG"] = (psf_sigma_pix, "Gaussian PSF sigma [pix]")
    return header


def _log_wcs_check(ccd_id, wcs, stars):
    star = stars.iloc[0]
    world = wcs.all_pix2world(star["x_pix"], star["y_pix"], 0)
    cos_star = math.cos(math.radians(star["dec"]))
    dra_arcsec = (world[0] - star["ra"]) * cos_star * 3600.0
    ddec_arcsec = (world[1] - star["dec"]) * 3600.0
    residual = math.hypot(dra_arcsec, ddec_arcsec)
    print(
        f"WCS check (CCD {ccd_id}): residual {residual:.3f} arcsec",
        flush=True,
    )


def write_mosaic_mef(path=mef_path):
    """Write 20-CCD visit as MEF: empty primary + one noisy ImageHDU per CCD."""
    path = ensure_fits_parent(path)
    primary = fits.PrimaryHDU(data=None, header=_primary_mef_header())
    hdul = fits.HDUList([primary])
    wcs_check_done = False

    for ccd_id in range(1, 21):
        _ideal, noisy, ra, dec, dx_mm, dy_mm, stars_on_patch = simulate_ccd(ccd_id)
        height, width = noisy.shape
        wcs = make_ccd_wcs(ra, dec, width, height)
        header = _ccd_fits_header(wcs, ccd_id, width, height, dx_mm, dy_mm)
        hdul.append(fits.ImageHDU(noisy.astype(np.float32), header=header))
        print(f"CCD {ccd_id}/20 written to MEF", flush=True)

        if not wcs_check_done and len(stars_on_patch) > 0:
            _log_wcs_check(ccd_id, wcs, stars_on_patch)
            wcs_check_done = True

    hdul.writeto(path, overwrite=True)
    if write_focal_plane:
        stitch_focal_plane(path, focal_plane_path)


# Build a normalized circular 2D Gaussian PSF kernel (truncated at psf_trunc_sigma * σ).
def make_psf_kernel():
    radius_int = psf_pad_pix
    dy, dx = np.mgrid[-radius_int : radius_int + 1, -radius_int : radius_int + 1]
    r2 = dx * dx + dy * dy
    weights = np.exp(-0.5 * r2 / (psf_sigma_pix * psf_sigma_pix))
    weights /= weights.sum()
    offsets = np.column_stack([dy.ravel(), dx.ravel()])
    return offsets, weights.ravel().astype(np.float32)


# Stamp each star's charge onto the image using a Gaussian PSF kernel.
def _deposit_psf(image, x, y, charge, offsets, weights, batch_size=psf_deposit_batch_size):
    height, width = image.shape
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    off_y = offsets[:, 0]
    off_x = offsets[:, 1]

    for start in range(0, len(xi), batch_size):
        end = min(start + batch_size, len(xi))
        yi_b = yi[start:end]
        xi_b = xi[start:end]
        ch_b = charge[start:end]

        all_y = yi_b[:, None] + off_y[None, :]
        all_x = xi_b[:, None] + off_x[None, :]
        valid = (
            (all_y >= 0)
            & (all_y < height)
            & (all_x >= 0)
            & (all_x < width)
        )
        w_on_chip = weights[None, :] * valid
        # No edge renormalization: clipped PSF flux leaves the chip.
        all_c = ch_b[:, None] * w_on_chip

        flat_y = all_y.ravel()
        flat_x = all_x.ravel()
        flat_c = all_c.ravel()
        flat_valid = valid.ravel()

        np.add.at(image, (flat_y[flat_valid], flat_x[flat_valid]), flat_c[flat_valid])


# Deposit all stars onto a 2D image; tag g < g_lim as monitored sources.
def pixelate_smash_data(
    smash_data,
    center_ra,
    center_dec,
    ra_read_radius,
    dec_read_radius,
    pix_size_arcsec=pix_size,
    exposure_time_s=exposure_time,
    m_ref_mag=m_ref,
    g_lim_mag=g_lim,
    width_px=ccd_width_px,
    height_px=ccd_height_px,
):
    width, height = width_px, height_px
    stars = smash_data[smash_data["gmag"] < faint_gmag_max].copy()

    
    if stars.empty:
        return np.zeros((height, width), dtype=np.float32), stars

    stars["monitored"] = stars["gmag"] < faint_gmag_max
    x, y = sky_to_pixel(
        stars["ra"].to_numpy(),
        stars["dec"].to_numpy(),
        center_ra,
        center_dec,
        pix_size_arcsec,
        width,
        height,
    )

    stars["x_pix"] = x
    stars["y_pix"] = y
    stars["charge"] = get_charge(
        stars["gmag"].to_numpy(),
        m_ref_mag=m_ref_mag,
        exposure_time_s=exposure_time_s,
    )
    offsets, weights = make_psf_kernel()

    image = np.zeros((height, width), dtype=np.float32)
    print(
        f"Depositing {len(stars):,} stars "
        f"(PSF σ={psf_sigma_pix} pix, kernel={len(weights)} px)...",
        flush=True,
    )
    _deposit_psf(
        image,
        stars["x_pix"].to_numpy(),
        stars["y_pix"].to_numpy(),
        stars["charge"].to_numpy(),
        offsets,
        weights,
    )

    in_image = (
        (stars["x_pix"] >= -psf_pad_pix)
        & (stars["x_pix"] < width + psf_pad_pix)
        & (stars["y_pix"] >= -psf_pad_pix)
        & (stars["y_pix"] < height + psf_pad_pix)
    )
    stars_for_stats = stars.loc[in_image].copy()
    return image, stars_for_stats


# Add independent Gaussian readout noise to every pixel.
def add_readout_noise(image, readout_sigma=readout_noise):
    noise = np.random.normal(0.0, readout_sigma, size=image.shape)
    return image + noise


# One exposure: sky + Poisson photon noise + readout noise on the charge image.
def simulate_microchip(
    signal_image,
    exposure_time_s=exposure_time,
    sky_rate=sky_e_rate,
    readout_sigma=readout_noise,
):
    sky_e = sky_rate * exposure_time_s
    expected = np.maximum(signal_image + sky_e, 0.0)
    poisson_image = np.random.poisson(expected).astype(np.float32)
    return add_readout_noise(poisson_image, readout_sigma)


# Scatter plot of source positions in the sky patch.
def display_smash_data(smash_data):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        smash_data["ra"], 
        smash_data["dec"], 
        c=smash_data["gmag"], 
        cmap="viridis", 
        edgecolor="k"
    )
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    cbar = plt.colorbar(scatter)
    cbar.set_label("gmag")
    plt.title("SMASH Sources (color: gmag)")
    plt.show()


# Show a single grayscale image with percentile stretch.
def display_image(image, vmin_pct=1.0, vmax_pct=99.0, title=None):
    finite = image[np.isfinite(image)]
    if finite.size == 0 or np.all(finite == 0):
        print("Image is empty.")
        return

    sample = finite[finite > 0] if np.any(finite > 0) else finite
    vmin, vmax = np.percentile(sample, [vmin_pct, vmax_pct])
    if vmin == vmax:
        vmax = vmin + 1.0

    if title is None:
        title = f"Charge [e-] ({pix_size}\" / pix, t={exposure_time}s)"

    fig, ax = plt.subplots(figsize=(10, 10 * image.shape[0] / image.shape[1]))
    ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# Side-by-side ideal expected image and simulated noisy microchip frame.
def display_ideal_vs_noisy(ideal_image, noisy_image, vmin_pct=1.0, vmax_pct=99.0):
    combined = np.concatenate([ideal_image.ravel(), noisy_image.ravel()])
    sample = combined[combined > 0] if np.any(combined > 0) else combined
    vmin, vmax = np.percentile(sample, [vmin_pct, vmax_pct])
    if vmin == vmax:
        vmax = vmin + 1.0

    height, width = ideal_image.shape
    extent = (0, width, 0, height)
    panel_w = 6.0
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2 * panel_w, panel_w * height / width),
        sharex=True,
        sharey=True,
    )

    for ax, img, title in zip(
        axes,
        [ideal_image, noisy_image],
        ["Ideal expected [e-]", "Simulated microchip [e-]"],
    ):
        ax.imshow(
            img,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x [pix]")

    axes[0].set_ylabel("y [pix]")
    axes[1].tick_params(labelleft=False)
    fig.tight_layout()
    plt.show()


# Load SMASH data, pixelate, simulate noise, print survey stats, and display.
def main():
    if write_mef:
        print(f"Writing 20-CCD MEF to {mef_path}...", flush=True)
        write_mosaic_mef(mef_path)
        print("Done.", flush=True)
        return
    ideal_image, noisy_image, ccd_ra, ccd_dec, dx_mm, dy_mm, stars_on_patch = (
        simulate_ccd(ccd_number)
    )
    ra_radius, dec_radius, pad_ra, pad_dec = ccd_sky_extents(ccd_dec)
    smash_data = get_smash_data(
        ccd_ra,
        ccd_dec,
        ra_radius + pad_ra,
        dec_radius + pad_dec,
    )
    print(f"CCD {ccd_number} offset from array center: ({dx_mm:.3f}, {dy_mm:.3f}) mm")
    print(f"CCD sky center: RA={ccd_ra:.5f}, Dec={ccd_dec:.5f}")
    print(f"Patch size: {ideal_image.shape[1]} x {ideal_image.shape[0]} pix")
    print(f"Charge at g={g_lim}: {get_charge(g_lim):.2f} e-")
    n_deposited = len(smash_data[smash_data["gmag"] < faint_gmag_max])
    print(f"Deposited stars (incl. PSF pad): {n_deposited:,}")
    print(f"Sources in patch (centers only): {len(stars_on_patch):,}")
    if len(stars_on_patch) > 0:
        n_mon = int(stars_on_patch["monitored"].sum())
        n_faint = len(stars_on_patch) - n_mon
        print(f"  Monitored (g < {g_lim}): {n_mon:,}")
        print(f"  Faint contributors (g >= {g_lim}): {n_faint:,}")
    print(f"Sky per pixel: {sky_e_rate * exposure_time:.2f} e-")
    print(f"Readout noise sigma: {readout_noise} e-")

    if len(stars_on_patch) > 0:
        mon = stars_on_patch[stars_on_patch["monitored"]]
        if len(mon) > 0:
            print(f"Median monitored star charge [e-]: {mon['charge'].median():.2f}")

    display_ideal_vs_noisy(ideal_image, noisy_image)
    


if __name__ == "__main__":
    import sys

    def _cli_fits_path(default: str) -> str:
        paths = [a for a in sys.argv[2:] if not a.startswith("--")]
        return paths[0] if paths else default

    if len(sys.argv) > 1 and sys.argv[1] == "--patch-mef":
        patch_mef_headers(
            _cli_fits_path(mef_path),
            gaps=True if "--gaps" in sys.argv else None,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--stitch":
        stitch_focal_plane(
            _cli_fits_path(mef_path),
            sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else focal_plane_path,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--stitch-gaps":
        stitch_focal_plane(
            _cli_fits_path(mef_path),
            sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else focal_plane_path,
            gaps=True,
        )
    else:
        main()
