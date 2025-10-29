#!/usr/bin/env python3
"""
posthoc_cfo_sign_autofix.py

Post-hoc fix for bimodal CFO sign flips:
- Loads a CSV with per-burst features.
- (Optional) linear de-trend of CFO vs time.
- Finds two CFO lobes (1-D k-means, no sklearn).
- Flips the *minority* lobe to match the dominant lobe’s sign.
- Writes a new CSV with an added column '<cfo_col>_signfixed'.
- Saves before/after histograms and time-series plots.

Usage (common):
  python3 posthoc_cfo_sign_autofix.py filtered.csv

With options:
  python3 posthoc_cfo_sign_autofix.py filtered.csv \
    --cfo-col cfo_centroid_hz \
    --time-col pcap_ts \
    --detrend \
    --out fixed.csv \
    --plots-outdir signfix_plots
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True); return p

def linear_detrend(t, y):
    """Return y - (a*t + b). Ignores non-finite points in fit."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 2:
        return y, 0.0, 0.0
    # robust-ish: center time to reduce collinearity
    tc = t[m] - np.median(t[m])
    A = np.vstack([tc, np.ones_like(tc)]).T
    a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
    y_dt = y.copy()
    y_dt[m] = y[m] - (a * (t[m] - np.median(t[m])) + b)
    return y_dt, float(a), float(b)

def kmeans_1d(x, k=2, iters=50, seed=0):
    """Tiny 1-D k-means. Returns (centers, labels)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < k:
        # degenerate
        centers = np.array([np.nanmean(x)]*k)
        labels = np.zeros_like(x, dtype=int)
        return centers, labels
    # smart-ish init: percentiles
    qs = np.linspace(0.2, 0.8, k)
    centers = np.quantile(x, qs)
    # fallback if identical
    if np.allclose(np.diff(centers), 0):
        centers = rng.choice(x, size=k, replace=False)
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(iters):
        # assign
        d = np.abs(x[:, None] - centers[None, :])
        labels_new = np.argmin(d, axis=1)
        if np.all(labels_new == labels):
            break
        labels = labels_new
        # update
        for j in range(k):
            m = labels == j
            if np.any(m):
                centers[j] = np.mean(x[m])
    return centers, labels

def sign_autofix(y, time=None, detrend=False, plots_dir=None, tag="", deadband_hz=200.0):
    """
    Core fixer:
      - optional detrend
      - k-means 1D (k=2)
      - identify minority cluster; if its mean has opposite sign to the majority cluster
        and |means| are comparable, flip that cluster.
      - return fixed values + diagnostics
    """
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    yv = y[mask].copy()

    # De-trend (optional)
    slope = 0.0
    if detrend and time is not None:
        yv_dt, slope, _ = linear_detrend(time[mask], yv)
    else:
        yv_dt = yv

    # Cluster (on de-trended values, so lobes are clean)
    centers, labels = kmeans_1d(yv_dt, k=2, iters=100, seed=0)
    c0, c1 = float(centers[0]), float(centers[1])
    n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))

    # Decide majority/minority
    majority = 0 if n0 >= n1 else 1
    minority = 1 - majority

    mean_major, mean_minor = (c0, c1) if majority == 0 else (c1, c0)

    # Heuristics: flip minority if (1) opposite sign, and (2) magnitudes "comparable"
    opp_sign = (np.sign(mean_major) * np.sign(mean_minor) == -1)
    mag_ratio = min(abs(mean_major), abs(mean_minor)) / (max(abs(mean_major), abs(mean_minor)) + 1e-12)

    # You can make this stricter/looser. 0.5 means within 2x magnitude.
    do_flip = opp_sign and (mag_ratio >= 0.5)

    y_fixed_dt = yv_dt.copy()
    if do_flip:
        y_fixed_dt[labels == minority] = - y_fixed_dt[labels == minority]

    # Undo de-trend (put back original slope) so outputs remain in original units
    if detrend and time is not None:
        # we had: yv_dt = yv - (a*(t-t0) + b). We want y_fixed in original axis:
        # y_fixed = y_fixed_dt + (a*(t-t0) + b). But b cancels if we want same absolute axis.
        # For "sign-fix" we care about sign consistency; keep original DC by adding back the fit.
        tloc = time[mask]
        t0 = np.median(tloc)
        A_back = slope * (tloc - t0)  # ignore intercept to preserve original median
        y_fixed = yv.copy()
        y_fixed[:] = y_fixed_dt + A_back
    else:
        y_fixed = yv_fixed = y_fixed_dt

    # Compose full vector (preserve NaNs where present)
    y_out = y.copy()
    y_out[mask] = y_fixed

    diag = {
        "detrended": bool(detrend and time is not None),
        "slope_hz_per_s": float(slope),
        "n_total": int(y.size),
        "n_valid": int(mask.sum()),
        "centers_hz": (c0, c1),
        "counts": (n0, n1),
        "major_center_hz": float(mean_major),
        "minor_center_hz": float(mean_minor),
        "opposite_sign": bool(opp_sign),
        "mag_ratio": float(mag_ratio),
        "flipped": bool(do_flip),
    }

    # Plots (optional)
    if plots_dir:
        ensure_dir(plots_dir)
        # Before
        fig = plt.figure(figsize=(6,4))
        plt.hist(yv, bins=15)
        med = np.median(yv[np.isfinite(yv)])
        plt.axvline(med, linestyle="--")
        plt.xlabel("CFO (Hz)"); plt.ylabel("Count")
        plt.title(f"Before sign-fix (median ≈ {med:.0f} Hz)")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"hist_before{tag}.png"), dpi=150)
        plt.close(fig)

        # After
        fig = plt.figure(figsize=(6,4))
        plt.hist(y_out[mask], bins=15)
        med2 = np.median(y_out[mask])
        q25, q75 = np.percentile(y_out[mask], [25,75])
        iqr = q75-q25
        plt.axvline(med2, linestyle="--")
        plt.xlabel("CFO (Hz)"); plt.ylabel("Count")
        plt.title(f"After sign-fix (median ≈ {med2:.0f} Hz, IQR ≈ {iqr/1e3:.2f} kHz)")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"hist_after{tag}.png"), dpi=150)
        plt.close(fig)

        # Time series before/after (if time)
        if time is not None:
            tloc = np.asarray(time, float)
            m2 = np.isfinite(tloc) & np.isfinite(y)
            fig = plt.figure(figsize=(10,3.6))
            plt.plot(tloc[m2]-tloc[m2].min(), y[m2], marker="o")
            plt.axhline(np.median(y[m2]), linestyle="--")
            plt.xlabel("Time (s, relative)")
            plt.ylabel("CFO (Hz)")
            plt.title("CFO vs Time (before)")
            plt.grid(True, linestyle=":", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"time_before{tag}.png"), dpi=150)
            plt.close(fig)

            fig = plt.figure(figsize=(10,3.6))
            plt.plot(tloc[mask]-tloc[m2].min(), y_out[mask], marker="o")
            plt.axhline(np.median(y_out[mask]), linestyle="--")
            plt.xlabel("Time (s, relative)")
            plt.ylabel("CFO (Hz)")
            plt.title("CFO vs Time (after sign-fix)")
            plt.grid(True, linestyle=":", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"time_after{tag}.png"), dpi=150)
            plt.close(fig)

    return y_out, diag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in", help="Input CSV (e.g., filtered.csv)")
    ap.add_argument("--out", default=None, help="Output CSV (default: <in> with _signfixed suffix)")
    ap.add_argument("--cfo-col", default="cfo_centroid_hz",
                    help="Which CFO column to fix (default: cfo_centroid_hz)")
    ap.add_argument("--time-col", default="pcap_ts",
                    help="Time column for optional de-trend (default: pcap_ts)")
    ap.add_argument("--detrend", action="store_true",
                    help="Apply linear de-trend before clustering")
    ap.add_argument("--plots-outdir", default="signfix_plots",
                    help="Directory to write before/after plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    if args.cfo_col not in df.columns:
        raise SystemExit(f"Column '{args.cfo_col}' not in CSV.")
    time_vec = df[args.time_col].values if args.time_col in df.columns else None

    y = df[args.cfo_col].values.astype(float)
    y_fixed, diag = sign_autofix(
        y,
        time=time_vec,
        detrend=args.detrend,
        plots_dir=args.plots_outdir,
        tag=""
    )

    out_col = f"{args.cfo_col}_signfixed"
    df[out_col] = y_fixed

    out_path = args.out or (str(Path(args.csv_in).with_suffix("")) + "_signfixed.csv")
    df.to_csv(out_path, index=False)

    # Report
    med = np.nanmedian(y)
    q25, q75 = np.nanpercentile(y[np.isfinite(y)], [25,75])
    iqr = q75-q25
    med2 = np.nanmedian(y_fixed)
    q25b, q75b = np.nanpercentile(y_fixed[np.isfinite(y_fixed)], [25,75])
    iqr2 = q75b-q25b

    print("\n=== POST-HOC CFO SIGN AUTO-FIX ===")
    print(f"input file: {args.csv_in}")
    print(f"output    : {out_path}")
    print(f"cfo_col   : {args.cfo_col}  → new col: {out_col}")
    print(f"detrended : {diag['detrended']}  (slope {diag['slope_hz_per_s']:.3f} Hz/s)")
    print(f"clusters  : centers {diag['centers_hz']}  counts {diag['counts']}")
    print(f"flip?     : {diag['flipped']}  (opp_sign={diag['opposite_sign']}, mag_ratio={diag['mag_ratio']:.2f})")
    print(f"before    : median={med:+.1f} Hz, IQR={iqr/1e3:.2f} kHz")
    print(f"after     : median={med2:+.1f} Hz, IQR={iqr2/1e3:.2f} kHz")
    print(f"plots     : {args.plots_outdir}")

if __name__ == "__main__":
    main()