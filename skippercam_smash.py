import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#tunables
exposure_time = 8  # seconds
readout_noise = 7.0  # e- RMS per pixel (fast survey mode)

# Define the LMC coordinates
lmc_ra = 80.89
lmc_dec = -69.76
pix_size = 1.43  # arcsec per pix

# Pick the simulated patch size directly in pixels.
# This drives the RA/Dec box via the local plate scale.
width_px = 1278
height_px = 1058

_pix_scale_deg = pix_size / 3600.0
_cos_dec = math.cos(math.radians(lmc_dec))
lmc_ra_read_radius = 0.5 * width_px * _pix_scale_deg / max(_cos_dec, 1e-12)
lmc_dec_read_radius = 0.5 * height_px * _pix_scale_deg

ZP_g = 13149669667  # integrated e- for a 0-mag source in one exposure
m_ref = 2.5 * np.log10(ZP_g)  # reference magnitude tied to ZP_g
n_ref = 1.0 / exposure_time  # e-/s at m_ref; get_charge(0) == ZP_g
sky_e_rate = 6.3  # e- / s / pixel (SkipperCam paper Sec. 2.4)
g_lim = 20  # monitored sources: g < g_lim (INFO.md / paper)
psf_sigma_pix = 2  # Gaussian PSF sigma
psf_pad_pix = math.ceil(3.0 * psf_sigma_pix)  # matches make_psf_kernel radius
psf_pad_deg = psf_pad_pix * pix_size / 3600.0
psf_pad_deg_ra = psf_pad_deg / max(_cos_dec, 1e-12)
psf_deposit_batch_size = 50_000  # stars per PSF stamp batch (limits RAM)
faint_gmag_max = 99  # SMASH sentinel for valid magnitudes

# Load SMASH DR2 sample data
smash_df = pd.read_csv("lmc_smash_g99_v2.csv")


# Pixel width/height for the sky patch at the given plate scale.
def patch_size_pixels(ra_read_radius, dec_read_radius, center_dec, pix_size_arcsec):
    pix_scale_deg = pix_size_arcsec / 3600.0
    cos_dec = math.cos(math.radians(center_dec))
    width = int(np.ceil(2.0 * ra_read_radius * cos_dec / pix_scale_deg))
    height = int(np.ceil(2.0 * dec_read_radius / pix_scale_deg))
    return max(width, 1), max(height, 1)


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
    rate = n_ref * np.power(10.0, -0.4 * (m_target - m_ref_mag))
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


# Build a normalized 2D Gaussian PSF kernel for stamping star flux.
def make_psf_kernel(sigma_pix, radius_sigma=3.0):
    radius_int = int(math.ceil(radius_sigma * sigma_pix))
    dy, dx = np.mgrid[-radius_int : radius_int + 1, -radius_int : radius_int + 1]
    weights = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma_pix * sigma_pix))
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
    sigma_pix=psf_sigma_pix,
):
    width, height = patch_size_pixels(
        ra_read_radius, dec_read_radius, center_dec, pix_size_arcsec
    )
    stars = smash_data[smash_data["gmag"] < faint_gmag_max].copy()
    if stars.empty:
        return np.zeros((height, width), dtype=np.float32), stars

    stars["monitored"] = stars["gmag"] < g_lim_mag
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

    offsets, weights = make_psf_kernel(sigma_pix)

    image = np.zeros((height, width), dtype=np.float32)
    print(
        f"Depositing {len(stars):,} stars "
        f"(PSF sigma={sigma_pix} pix, kernel={len(weights)} px)...",
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
    smash_data = get_smash_data(
        lmc_ra,
        lmc_dec,
        lmc_ra_read_radius + psf_pad_deg_ra,
        lmc_dec_read_radius + psf_pad_deg,
    )
    ideal_image, stars_on_patch = pixelate_smash_data(
        smash_data,
        lmc_ra,
        lmc_dec,
        lmc_ra_read_radius,
        lmc_dec_read_radius,
    )
    noisy_image = simulate_microchip(ideal_image)

    print(f"Patch size: {ideal_image.shape[1]} x {ideal_image.shape[0]} pix")
    print(f"m_ref={m_ref:.4f}, n_ref={n_ref:.6f} e-/s, charge(0)={get_charge(0):.4e} e-")
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
