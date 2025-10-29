#!/usr/bin/env python3
import sys, csv, math
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ----- config -----
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "features_airtag.csv"
OUTDIR = Path("debug_step1_out"); OUTDIR.mkdir(parents=True, exist_ok=True)

USE_COLS = [
    "pkt_idx","pcap_ts","pdu_type","gated_len_us","psd_pnr_db",
    "cfo_quick_hz","cfo_centroid_hz","cfo_two_stage_hz","cfo_two_stage_coarse_hz"
]

def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            keep = {}
            for k in USE_COLS:
                v = row.get(k, "")
                if k in ("pkt_idx","pdu_type"):
                    try: keep[k] = int(v)
                    except: keep[k] = None
                else:
                    try: keep[k] = float(v)
                    except: keep[k] = float("nan")
            rows.append(keep)
    return rows

def fin(a): 
    a = np.asarray(a, float)
    return a[np.isfinite(a)]

def sign_agree(a, b, eps=200.0):
    """
    Return an array: +1 if signs agree (or both |x|<eps), -1 if disagree.
    eps: small deadband where values are treated as ~zero to avoid noise flips.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = np.zeros_like(a, int)
    for i,(x,y) in enumerate(zip(a,b)):
        sx =  0 if abs(x) < eps else (1 if x>0 else -1)
        sy =  0 if abs(y) < eps else (1 if y>0 else -1)
        s[i] = 1 if (sx==sy) else -1
    return s

def robust_width(a):
    """IQR width: q75-q25 (less sensitive than max-min)."""
    a = fin(a)
    if a.size==0: return float("nan")
    q25,q50,q75 = np.percentile(a,[25,50,75])
    return q75-q25, q50

def make_hist(x, title, fname, bins=50):
    x = fin(x)
    if x.size==0: return
    plt.figure()
    plt.hist(x, bins=bins)
    plt.grid(True, ls=":", lw=0.5)
    plt.title(title); plt.xlabel("Hz"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(OUTDIR/fname, dpi=150); plt.close()

def make_time(t, y, title, fname, thin=1):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if not np.any(m): return
    idx = np.where(m)[0][::max(1,thin)]
    plt.figure()
    plt.plot(t[idx], y[idx], linewidth=1)
    plt.grid(True, ls=":", lw=0.5)
    plt.title(title); plt.xlabel("time (s)"); plt.ylabel("Hz")
    plt.tight_layout(); plt.savefig(OUTDIR/fname, dpi=150); plt.close()

rows = read_csv(CSV_PATH)
if not rows:
    print("No rows read; check CSV path.", file=sys.stderr); sys.exit(1)

# Columns
ts   = np.array([r["pcap_ts"] for r in rows], float)
gl   = np.array([r["gated_len_us"] for r in rows], float)
pnr  = np.array([r["psd_pnr_db"] for r in rows], float)
c_q  = np.array([r["cfo_quick_hz"] for r in rows], float)
c_c  = np.array([r["cfo_centroid_hz"] for r in rows], float)
c_2  = np.array([r["cfo_two_stage_hz"] for r in rows], float)
c_2c = np.array([r["cfo_two_stage_coarse_hz"] for r in rows], float)

# 1) SIGN AGREEMENT
agree_q_c  = sign_agree(c_q, c_c)
agree_q_2  = sign_agree(c_q, c_2)
agree_c_2  = sign_agree(c_c, c_2)
agree_2c_2 = sign_agree(c_2c, c_2)

def pct_pos(a): 
    a = np.asarray(a, int)
    return 100.0*np.sum(a==1)/max(1,len(a))

print("\n=== SIGN AGREEMENT (deadband ±200 Hz) ===")
print(f"quick vs centroid : {pct_pos(agree_q_c):6.2f}% agree")
print(f"quick vs two-stage: {pct_pos(agree_q_2):6.2f}% agree")
print(f"centroid vs two   : {pct_pos(agree_c_2):6.2f}% agree")
print(f"coarse  vs two    : {pct_pos(agree_2c_2):6.2f}% agree (should be ≳95–99%)")

# 2) COARSE↔FINE HANDOFF (two-stage vs coarse)
diff_2_minus_2c = c_2 - c_2c
w_iqr, w_med = robust_width(diff_2_minus_2c)
print("\n=== TWO-STAGE HANDOFF CHECK ===")
print(f"(two_stage - coarse): median={w_med:9.2f} Hz, IQR={w_iqr:9.2f} Hz")
print("Expected: |median| small (~0) and IQR ≲ a few kHz. Large values imply bad handoff / polarity mix.")

# 3) WINDOW/GATING BIAS (CFO vs gated_len_us / psd_pnr_db)
def corr(a,b):
    a,b = fin(a), fin(b)
    if a.size<3 or b.size<3: return float("nan")
    a = (a - np.mean(a)) / (np.std(a)+1e-12)
    b = (b - np.mean(b)) / (np.std(b)+1e-12)
    return float(np.mean(a*b))

for name, vec in [("cfo_centroid_hz", c_c), ("cfo_two_stage_hz", c_2), ("cfo_quick_hz", c_q)]:
    print(f"\n=== GATING CORR for {name} ===")
    print(f"corr({name}, gated_len_us): {corr(vec, gl): .3f}   (strong |.| suggests window-length bias)")
    print(f"corr({name}, psd_pnr_db)  : {corr(vec, pnr): .3f}   (strong |.| suggests SNR-dependent bias)")

# 4) PLOTS
make_hist(c_q, "cfo_quick_hz (hist)", "hist_cfo_quick.png")
make_hist(c_c, "cfo_centroid_hz (hist)", "hist_cfo_centroid.png")
make_hist(c_2, "cfo_two_stage_hz (hist)", "hist_cfo_two_stage.png")
make_hist(c_2c,"cfo_two_stage_coarse_hz (hist)", "hist_cfo_two_stage_coarse.png")

thin = max(1, len(ts)//2000)
make_time(ts, c_q, "cfo_quick_hz vs time", "time_quick.png", thin=thin)
make_time(ts, c_c, "cfo_centroid_hz vs time", "time_centroid.png", thin=thin)
make_time(ts, c_2, "cfo_two_stage_hz vs time", "time_two_stage.png", thin=thin)
make_time(ts, c_2c,"cfo_two_stage_coarse_hz vs time","time_two_stage_coarse.png", thin=thin)

print(f"\nWrote plots and logs to: {OUTDIR.resolve()}")