import math
import time

import pandas as pd
import pyvo as vo
import requests
from pyvo.dal.exceptions import DALServiceError

# LMC center (same as skippercam_smash.py)
LMC_RA = 80.89
LMC_DEC = -69.76

# 4.13 deg^2 on sky; aspect ratio from sensor (W x H, arbitrary units)
PATCH_AREA_DEG2 = 4.13
SENSOR_WIDTH_DEG = 3.89
SENSOR_HEIGHT_DEG = 3.89

_cos_dec = math.cos(math.radians(LMC_DEC))
_patch_w = math.sqrt(PATCH_AREA_DEG2 * SENSOR_WIDTH_DEG / SENSOR_HEIGHT_DEG)
_patch_h = math.sqrt(PATCH_AREA_DEG2 * SENSOR_HEIGHT_DEG / SENSOR_WIDTH_DEG)
LMC_RA_READ_RADIUS = _patch_w / (2.0 * _cos_dec)
LMC_DEC_READ_RADIUS = _patch_h / 2.0

TERMINAL_PHASES = frozenset({"COMPLETED", "ERROR", "ABORTED", "UNKNOWN"})
ACTIVE_PHASES = frozenset({"QUEUED", "EXECUTING", "RUN", "COMPLETED", "ERROR", "UNKNOWN"})
JOB_WAIT_TIMEOUT_S = 3600.0
STATUS_POLL_TIMEOUT_S = 60.0


def log(message: str) -> None:
    print(message, flush=True)


def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, DALServiceError):
        reason = str(exc).lower()
        return "timed out" in reason or "timeout" in reason or "connection" in reason
    return False


def current_phase(job) -> str:
    return job._job.phase


def wait_with_progress(job, timeout_s: float = JOB_WAIT_TIMEOUT_S) -> None:
    """Wait for TAP job completion with progress logs and transient-error retries."""
    start = time.monotonic()
    last_phase = None
    fallback_interval_s = 1.0
    deadline = start + timeout_s

    try:
        job._update(timeout=STATUS_POLL_TIMEOUT_S)
    except DALServiceError as exc:
        if not is_transient_error(exc):
            raise

    while time.monotonic() < deadline:
        elapsed = time.monotonic() - start
        phase = current_phase(job)
        if phase != last_phase:
            log(f"[{elapsed:6.0f}s] job phase: {phase}")
            last_phase = phase

        if phase in TERMINAL_PHASES:
            log(f"[{elapsed:6.0f}s] job phase: {phase} (done)")
            job.raise_if_error()
            return

        if phase not in ACTIVE_PHASES:
            raise DALServiceError(f"Job is not active (phase={phase})")

        remaining = deadline - time.monotonic()
        poll_timeout = min(STATUS_POLL_TIMEOUT_S, remaining)
        if poll_timeout <= 0:
            break

        try:
            job._update(wait_for_statechange=True, timeout=poll_timeout)
        except DALServiceError as exc:
            elapsed = time.monotonic() - start
            if is_transient_error(exc):
                log(f"[{elapsed:6.0f}s] status check failed ({exc}); retrying...")
                time.sleep(min(fallback_interval_s, 10.0))
                fallback_interval_s = min(120.0, fallback_interval_s * 1.2)
                continue
            raise

        phase = current_phase(job)
        elapsed = time.monotonic() - start
        if phase == last_phase:
            log(f"[{elapsed:6.0f}s] still {phase}...")
        else:
            log(f"[{elapsed:6.0f}s] job phase: {phase}")
            last_phase = phase

        if phase in TERMINAL_PHASES:
            log(f"[{elapsed:6.0f}s] job phase: {phase} (done)")
            job.raise_if_error()
            return

        time.sleep(fallback_interval_s)
        fallback_interval_s = min(120.0, fallback_interval_s * 1.2)

    raise TimeoutError(f"TAP job did not finish within {timeout_s:.0f}s")


log(
    f"Patch center: RA={LMC_RA}, Dec={LMC_DEC} | "
    f"radii RA±{LMC_RA_READ_RADIUS:.4f}°, Dec±{LMC_DEC_READ_RADIUS:.4f}° | "
    f"{PATCH_AREA_DEG2} deg²"
)

tap = vo.dal.TAPService("https://datalab.noirlab.edu/tap")
log("Connected to NOIRLab TAP")

query = f"""
SELECT id, ra, dec, umag, gmag, rmag, imag, zmag
FROM smash_dr2.object
WHERE gmag < 99
  AND ra  BETWEEN {LMC_RA} - {LMC_RA_READ_RADIUS} AND {LMC_RA} + {LMC_RA_READ_RADIUS}
  AND dec BETWEEN {LMC_DEC} - {LMC_DEC_READ_RADIUS} AND {LMC_DEC} + {LMC_DEC_READ_RADIUS}
"""

log("Submitting query...")
job = tap.submit_job(query)
log(f"Job id: {job.job_id}")

log("Starting query execution...")
job.run()
wait_with_progress(job)

log("Fetching results...")
t0 = time.monotonic()
df = job.fetch_result(max_retries=5).to_table().to_pandas()
log(f"Fetched {len(df):,} rows in {time.monotonic() - t0:.1f}s")

out_path = "lmc_smash_g99_v2.csv"
log(f"Writing {out_path}...")
t0 = time.monotonic()
df.to_csv(out_path, index=False)
log(f"Saved {out_path} ({len(df):,} rows) in {time.monotonic() - t0:.1f}s")

job.delete()
log("Job deleted. Done.")
