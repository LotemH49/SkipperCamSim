from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
fits_file = _PROJECT_ROOT / "FITS" / "skippercam_sim_v4.fits"
MATCH_RADIUS_PIX = 5.0
LABEL_STRUCTURE = np.ones((3, 3), dtype=int)


def detect_at_threshold(hdul, threshold: float) -> list[dict]:
    """Return one detection dict per connected component above threshold."""
    detections: list[dict] = []

    for ccd_index, hdu in enumerate(hdul):
        data = hdu.data
        if data is None:
            continue

        arr = np.asarray(data, dtype=np.float64)
        thr_image = arr > threshold
        labels, nlabels = ndi.label(thr_image, structure=LABEL_STRUCTURE)
        if nlabels == 0:
            continue

        indices = np.arange(1, nlabels + 1, dtype=np.int32)
        charges = ndi.sum(arr, labels=labels, index=indices)
        centroids = ndi.center_of_mass(thr_image, labels=labels, index=indices)
        slices = ndi.find_objects(labels)

        for label_idx, slc in enumerate(slices):
            if slc is None:
                continue
            cy, cx = centroids[label_idx]
            y_slice, x_slice = slc
            detections.append(
                {
                    "ccd_index": ccd_index,
                    "threshold": float(threshold),
                    "centroid_y": float(cy),
                    "centroid_x": float(cx),
                    "charge": float(charges[label_idx]),
                    "bbox": (y_slice.start, x_slice.start, y_slice.stop, x_slice.stop),
                }
            )

    return detections


def _centroid_distance(det_a: dict, det_b: dict) -> float:
    dy = det_a["centroid_y"] - det_b["centroid_y"]
    dx = det_a["centroid_x"] - det_b["centroid_x"]
    return float(np.hypot(dy, dx))


def _bucket_key(centroid_y: float, centroid_x: float, cell_size: float) -> tuple[int, int]:
    return (int(centroid_y // cell_size), int(centroid_x // cell_size))


def _find_matching_source(
    detection: dict,
    buckets: dict[tuple[int, int, int], list[dict]],
    match_radius_pix: float,
) -> dict | None:
    ccd_index = detection["ccd_index"]
    cell_size = match_radius_pix
    by, bx = _bucket_key(detection["centroid_y"], detection["centroid_x"], cell_size)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for source in buckets.get((ccd_index, by + dy, bx + dx), []):
                if _centroid_distance(detection, source) < match_radius_pix:
                    return source
    return None


def _add_to_buckets(
    source: dict,
    buckets: dict[tuple[int, int, int], list[dict]],
    match_radius_pix: float,
) -> None:
    ccd_index = source["ccd_index"]
    by, bx = _bucket_key(source["centroid_y"], source["centroid_x"], match_radius_pix)
    buckets.setdefault((ccd_index, by, bx), []).append(source)


def unique_census(
    all_detections: list[dict], match_radius_pix: float = MATCH_RADIUS_PIX
) -> list[dict]:
    """Deduplicate detections across thresholds; highest threshold processed first."""
    sorted_dets = sorted(
        all_detections, key=lambda d: d["threshold"], reverse=True
    )

    unique_sources: list[dict] = []
    buckets: dict[tuple[int, int, int], list[dict]] = {}

    for detection in sorted_dets:
        match = _find_matching_source(detection, buckets, match_radius_pix)
        if match is not None:
            match["best_charge"] = max(match["best_charge"], detection["charge"])
            match["thresholds_seen"].add(detection["threshold"])
            continue

        source = {
            "ccd_index": detection["ccd_index"],
            "centroid_y": detection["centroid_y"],
            "centroid_x": detection["centroid_x"],
            "best_charge": detection["charge"],
            "thresholds_seen": {detection["threshold"]},
            "bbox": detection["bbox"],
        }
        unique_sources.append(source)
        _add_to_buckets(source, buckets, match_radius_pix)

    return unique_sources


def basic_thresholding(threshold: float, path: str = fits_file) -> list[dict]:
    """Run detection at a single threshold (opens FITS file)."""
    with fits.open(path) as hdul:
        detections = detect_at_threshold(hdul, threshold)
    print(f"threshold={threshold}: {len(detections)} raw detections")
    return detections


def multiple_thresholding(
    threshold_min: float,
    threshold_max: float,
    step: float,
    path: str = fits_file,
    match_radius_pix: float = MATCH_RADIUS_PIX,
) -> list[dict]:
    """Sweep thresholds, deduplicate detections, and return unique source census."""
    thresholds = np.arange(threshold_min, threshold_max, step)
    all_detections: list[dict] = []
    raw_counts: dict[float, int] = {}

    with fits.open(path) as hdul:
        for threshold in thresholds:
            detections = detect_at_threshold(hdul, float(threshold))
            all_detections.extend(detections)
            raw_counts[float(threshold)] = len(detections)
    
    unique_sources = unique_census(all_detections, match_radius_pix)

    print(
        f"Threshold sweep: {threshold_min}–{threshold_max} step {step} "
        f"({len(thresholds)} levels)"
    )
    print(f"Raw detections per threshold: {raw_counts}")
    print(f"Unique sources (deduplicated): {len(unique_sources)}")
    print(f"Match radius: {match_radius_pix} pix")

    return unique_sources


def main():
    multiple_thresholding(130, 1000, 10)


if __name__ == "__main__":
    main()
