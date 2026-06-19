import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#tunables
exposure_time = 8  # seconds
readout_noise = 7.0  # e- RMS per pixel (fast survey mode)
psf_sigma_pix = 1  # Gaussian PSF σ [pix]; sole PSF tunable (FWHM ≈ 2.355σ)
psf_trunc_sigma = 5  # truncate kernel at this many σ (~99.7% flux enclosed)
psf_pad_pix = math.ceil(psf_trunc_sigma * psf_sigma_pix)  # catalog pad matches kernel
# Define the LMC coordinates
lmc_ra = 80.89
lmc_dec = -69.76
pix_size = 1.43  # arcsec per pix
pixel_pitch_mm = 15e-3  # 15 µm detector pitch
ccd_width_px = 1278
ccd_height_px = 1058
ccd_width_mm = ccd_width_px * pixel_pitch_mm
ccd_height_mm = ccd_height_px * pixel_pitch_mm


_pix_scale_deg = pix_size / 3600.0
_arcsec_per_mm = pix_size / pixel_pitch_mm
_cos_dec = math.cos(math.radians(lmc_dec))
ccd_ra_read_radius = 0.5 * ccd_width_px * _pix_scale_deg / max(_cos_dec, 1e-12)
ccd_dec_read_radius = 0.5 * ccd_height_px * _pix_scale_deg

ZP_g = 1.64E+09  # integrated e- for a 0-mag source in one exposure
m_ref = 2.5 * np.log10(ZP_g)  # reference magnitude tied to ZP_g
sky_e_rate = 3.11E+01  # e- / s / pixel (SkipperCam paper Sec. 2.4)[changes with 0 point]
g_lim = 20  # monitored sources: g < g_lim (INFO.md / paper)
ccd_number = 20  # 1-20
psf_pad_deg = psf_pad_pix * pix_size / 3600.0
psf_pad_deg_ra = psf_pad_deg / max(_cos_dec, 1e-12)
psf_deposit_batch_size = 50_000  # stars per PSF stamp batch (limits RAM)
faint_gmag_max = 99  # SMASH sentinel for valid magnitudes

# Load SMASH DR2 sample data
smash_df = pd.read_csv("lmc_smash_g99_v2.csv")
#smash_df = pd.read_csv("lmc_onestar_tester.csv")

# CCD center offset from array center (mm). +x = +RA, +y = +Dec. (0, 0) = lmc_ra/lmc_dec.
ccd_location: dict[int, tuple[float, float]] = {
    1: (-38.93833 , 40.5488),
    2: (0.0, 0.0),
    3: (0.0, 0.0),
    4: (38.93833 , 40.5488),
    5: (0.0, 0.0),
    6: (0.0, 0.0),
    7: (0.0, 0.0),
    8: (0.0, 0.0),
    9: (0.0, 0.0),
    10: (0.0, 0.0),
    11: (0.0, 0.0),
    12: (0.0, 0.0),
    13: (0.0, 0.0),
    14: (0.0, 0.0),
    15: (0.0, 0.0),
    16: (0 , 0),
    17: (-38.93833 , -40.5488),
    18: (0.0, 0.0),
    19: (0.0, 0.0),
    20: (38.93833 , -40.5488),
}

def get_ccd_location(ccd_number):
    return ccd_location[ccd_number]


# Select SMASH catalog rows inside the RA/Dec read box.
def get_smash_data(ra, dec, ra_read_radius, dec_read_radius):
    mask = (
        (smash_df["ra"] > ra - ra_read_radius)
        & (smash_df["ra"] < ra + ra_read_radius)
        & (smash_df["dec"] > dec - dec_read_radius)
        & (smash_df["dec"] < dec + dec_read_radius)
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
    dra_deg = dx_mm * _arcsec_per_mm / 3600.0 / max(_cos_dec, 1e-12)
    ddec_deg = dy_mm * _arcsec_per_mm / 3600.0
    return lmc_ra + dra_deg, lmc_dec + ddec_deg


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
        all_c = ch_b[:, None] * weights[None, :]

        flat_y = all_y.ravel()
        flat_x = all_x.ravel()
        flat_c = all_c.ravel()

        valid = (
            (flat_y >= 0)
            & (flat_y < height)
            & (flat_x >= 0)
            & (flat_x < width)
        )
        np.add.at(image, (flat_y[valid], flat_x[valid]), flat_c[valid])


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
        (stars["x_pix"] >= 0)
        & (stars["x_pix"] < width)
        & (stars["y_pix"] >= 0)
        & (stars["y_pix"] < height)
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
    ccd_ra, ccd_dec = ccd_sky_center(ccd_number)

    smash_data = get_smash_data(
        ccd_ra,
        ccd_dec,
        ccd_ra_read_radius + psf_pad_deg_ra,
        ccd_dec_read_radius + psf_pad_deg,
    )
    ideal_image, stars_on_patch = pixelate_smash_data(
        smash_data,
        ccd_ra,
        ccd_dec,
        ccd_ra_read_radius,
        ccd_dec_read_radius,
    )
    noisy_image = simulate_microchip(ideal_image)

    dx_mm, dy_mm = get_ccd_location(ccd_number)
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
    main()
