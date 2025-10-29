#!/usr/bin/env python3
"""
Plot centroid (and optional two-stage) CFO time series + histogram
from the filtered high-quality subset produced by cfo_filter_and_report.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python3 plot_cfo_time_hist.py filtered.csv")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
if 'cfo_centroid_hz' not in df.columns:
    print("No cfo_centroid_hz column found.")
    sys.exit(1)

# Choose estimator
cfo = df['cfo_centroid_hz'].astype(float)
time = df.get('t_start_s', pd.Series(range(len(cfo)), dtype=float))

# Remove NaNs
mask = np.isfinite(cfo)
cfo = cfo[mask]
time = time[mask]

# Basic stats
median = np.median(cfo)
iqr = np.percentile(cfo, 75) - np.percentile(cfo, 25)
print(f"Centroid CFO: median={median:+.1f} Hz, IQR={iqr:.1f} Hz, N={len(cfo)}")

# --- Plot time series ---
plt.figure(figsize=(8, 3))
plt.plot(time, cfo, "o-", ms=4, alpha=0.8)
plt.axhline(median, color="r", lw=1, ls="--", label=f"median {median:+.0f} Hz")
plt.xlabel("Time [s]")
plt.ylabel("CFO (Hz)")
plt.title("CFO vs. Time (centroid)")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.savefig("plot_cfo_time.png", dpi=200)

# --- Plot histogram ---
plt.figure(figsize=(4, 3))
plt.hist(cfo, bins=15, color="gray", edgecolor="black")
plt.axvline(median, color="r", lw=1, ls="--")
plt.xlabel("CFO (Hz)")
plt.ylabel("Count")
plt.title(f"CFO histogram (IQR ≈ {iqr/1e3:.2f} kHz)")
plt.tight_layout()
plt.savefig("plot_cfo_hist.png", dpi=200)

print("Saved plots: plot_cfo_time.png, plot_cfo_hist.png")