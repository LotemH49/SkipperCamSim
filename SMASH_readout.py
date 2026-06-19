import threading
import time
from contextlib import contextmanager

import pandas as pd
import pyvo as vo

# LMC center (same as skippercam_smash.py)
LMC_RA = 80.89
LMC_DEC = -69.76

# Catalog box for full 20-CCD SkipperCam mosaic (center + half-chip + PSF pad)
LMC_RA_READ_RADIUS = 4
LMC_DEC_READ_RADIUS = 1.4


@contextmanager
def step_timer(label: str):
    stop = threading.Event()
    start = time.monotonic()

    def tick() -> None:
        while not stop.is_set():
            elapsed = time.monotonic() - start
            print(f"\r{label} {elapsed:.0f}s", end="", flush=True)
            if stop.wait(1.0):
                break

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        elapsed = time.monotonic() - start
        print(f"\r{label} {elapsed:.1f}s")


query = f"""
SELECT id, ra, dec, umag, gmag, rmag, imag, zmag
FROM smash_dr2.object
WHERE gmag < 99
  AND ra  BETWEEN {LMC_RA} - {LMC_RA_READ_RADIUS} AND {LMC_RA} + {LMC_RA_READ_RADIUS}
  AND dec BETWEEN {LMC_DEC} - {LMC_DEC_READ_RADIUS} AND {LMC_DEC} + {LMC_DEC_READ_RADIUS}
"""

print(
    f"Querying SMASH: RA {LMC_RA}±{LMC_RA_READ_RADIUS}°, "
    f"Dec {LMC_DEC}±{LMC_DEC_READ_RADIUS}°"
)

with step_timer("Connecting to TAP..."):
    tap = vo.dal.TAPService("https://datalab.noirlab.edu/tap")

with step_timer("Waiting for TAP..."):
    result = tap.search(query)

with step_timer("Converting to pandas..."):
    df = result.to_table().to_pandas()

print(f"Got {len(df):,} rows")

out_path = "lmc_smash_g99_v3.csv"
with step_timer(f"Writing {out_path}..."):
    df.to_csv(out_path, index=False)

print("Done.")
