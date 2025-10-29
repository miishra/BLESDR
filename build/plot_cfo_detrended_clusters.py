#!/usr/bin/env python3
"""
De-trend CFO (centroid) by removing global slope and visualize clustering.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python3 plot_cfo_detrended_clusters.py filtered.csv")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
time = df.get("t_start_s", pd.Series(range(len(df)), dtype=float))
cfo = df["cfo_centroid_hz"].astype(float)

mask = np.isfinite(time) & np.isfinite(cfo)
time, cfo = time[mask], cfo[mask]

# global detrend
coeffs = np.polyfit(time, cfo, 1)
trend = np.polyval(coeffs, time)
cfo_dt = cfo - trend
slope = coeffs[0]
print(f"Slope removed: {slope:+.3f} Hz/s")

# histogram of detrended
plt.figure(figsize=(4,3))
plt.hist(cfo_dt, bins=20, color='gray', edgecolor='black')
plt.axvline(np.median(cfo_dt), color='r', ls='--')
plt.title("De-trended CFO histogram")
plt.xlabel("CFO (Hz)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot_cfo_detrended_hist.png", dpi=200)

# scatter for clustering view
plt.figure(figsize=(8,3))
plt.scatter(time, cfo_dt, c=cfo_dt, cmap="coolwarm", s=30)
plt.axhline(0, color='k', lw=1)
plt.xlabel("Time [s]")
plt.ylabel("CFO (Hz, de-trended)")
plt.title("De-trended CFO vs Time (centroid)")
plt.colorbar(label="CFO (Hz)")
plt.grid(True, ls=":")
plt.tight_layout()
plt.savefig("plot_cfo_detrended_time.png", dpi=200)