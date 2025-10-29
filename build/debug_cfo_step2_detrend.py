#!/usr/bin/env python3
import sys, csv, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv)>1 else "features_fixed_overwrite.csv"
OUT = Path("debug_step2_out"); OUT.mkdir(exist_ok=True, parents=True)

def f(x):
    try: return float(x)
    except: return np.nan

rows = list(csv.DictReader(open(CSV, newline="")))
# Pick an estimator with best handoff: two-stage; also compute a consensus if present
t     = np.array([f(r.get("pcap_ts")) for r in rows], float)
c_2   = np.array([f(r.get("cfo_two_stage_hz")) for r in rows], float)
c_q   = np.array([f(r.get("cfo_quick_hz")) for r in rows], float)
c_c   = np.array([f(r.get("cfo_centroid_hz")) for r in rows], float)
c_cs  = np.array([f(r.get("cfo_consensus_hz")) for r in rows], float)
pdu   = np.array([int(r.get("pdu_type") or -1) for r in rows], int)
adv_mask = np.isfinite(t) & np.isfinite(c_2) & (pdu == 0)  # ADV_IND only

if np.sum(adv_mask) < 10:
    print("Not enough rows after mask.")
    sys.exit(1)

# Choose series: consensus if available else two-stage
y_raw = np.where(np.isfinite(c_cs), c_cs, c_2).copy()
t0 = np.nanmin(t[adv_mask])
tt = t[adv_mask] - t0
yy = y_raw[adv_mask]

# Robust linear fit via iterative reweighting
w = np.ones_like(yy)
for _ in range(5):
    A = np.vstack([tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(A * w[:,None], yy * w, rcond=None)
    resid = yy - (coef[0]*tt + coef[1])
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    w = 1.0 / np.clip(np.abs(resid)/(6*mad), 1.0, 5.0)  # downweight large outliers

slope_hz_per_s, intercept = float(coef[0]), float(coef[1])
slope_hz_per_min = slope_hz_per_s * 60.0
ppm = (slope_hz_per_s / 2.402e9) * 1e6

# De-trend
y_detr = yy - (slope_hz_per_s*tt + intercept)
q25, med, q75 = np.percentile(y_detr[np.isfinite(y_detr)], [25, 50, 75])
iqr = q75 - q25
sd  = np.std(y_detr[np.isfinite(y_detr)])

print("\n=== DRIFT FIT (consensus or two-stage) ===")
print(f"slope: {slope_hz_per_s:+.3f} Hz/s  ({slope_hz_per_min:+.2f} Hz/min),  {ppm:+.3f} ppm @ 2.402 GHz")
print(f"after de-trend: median={med:+.2f} Hz, IQR={iqr:.2f} Hz, std={sd:.2f} Hz")
print(f"rows used: {np.sum(adv_mask)}  (file → {CSV})")
print(f"plots → {OUT}")

# Plots
plt.figure()
plt.plot(tt, yy, '.', ms=3, label='raw CFO')
plt.plot(tt, slope_hz_per_s*tt + intercept, '-', lw=2, label='fit')
plt.xlabel('time since start (s)'); plt.ylabel('CFO (Hz)'); plt.title('CFO vs time (ADV only)')
plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"cfo_time_with_fit.png", dpi=150)

plt.figure()
plt.hist(y_detr, bins=60)
plt.xlabel('CFO residual (Hz)'); plt.ylabel('count'); plt.title('Residual CFO after de-trend')
plt.grid(True, linestyle=':'); plt.tight_layout()
plt.savefig(OUT/"cfo_residual_hist.png", dpi=150)