#!/usr/bin/env python3
import argparse, csv, os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------- Feature helpers (same math you had) ----------------
def rm_dc(x): return x - np.mean(x)
def norm_pow(x): p = np.sqrt(np.mean(np.abs(x)**2) + 1e-12); return x / p
def estimate_cfo(x, fs): ph = np.angle(x[1:] * np.conj(x[:-1])); return (fs/(2*np.pi)) * np.median(ph)
def iq_imbalance(x):
    I,Q = x.real, x.imag
    mII, mQQ, mIQ = np.mean(I*I), np.mean(Q*Q), np.mean(I*Q)
    alpha = np.sqrt(max(mII,1e-16)/max(mQQ,1e-16))
    phi = 0.5*np.arctan2(2*mIQ, (mII-mQQ+1e-16))
    return float(alpha), float(np.degrees(phi))
def rise_time_us(x, fs, tail=200):
    env = np.abs(x); steady = np.mean(env[-tail:]) if len(env)>tail else np.mean(env)
    if steady<=0: return 0.0
    n10 = int(np.argmax(env >= 0.1*steady)); n90 = int(np.argmax(env >= 0.9*steady))
    return (n90-n10)*1e6/fs if n90>n10 else 0.0
def welch_psd(x, fs, nfft=4096):
    L = min(len(x), nfft); w = np.hanning(L)
    X = np.fft.rfft(w * x[:L], n=L); S = (np.abs(X)**2) / (np.sum(w**2))
    f = np.fft.rfftfreq(L, d=1/fs); return f, S
def spectral_stats(x, fs):
    f,S = welch_psd(x, fs); S = np.real(S)+1e-18
    centroid = float(np.sum(f*S)/np.sum(S))
    pnr = float(10*np.log10(np.max(S)/(np.median(S)+1e-18)))
    bw_mask = S > (np.max(S)*10**(-3/10)); bw = float((f[bw_mask][-1]-f[bw_mask][0]) if np.any(bw_mask) else 0.0)
    return centroid, pnr, bw

def packet_features(x, fs):
    x = norm_pow(rm_dc(x))
    cfo = estimate_cfo(x, fs)
    alpha, phi_deg = iq_imbalance(x)
    rt_us = rise_time_us(x, fs)
    fcent, pnr, bw = spectral_stats(x, fs)
    cfo_inst = (fs/(2*np.pi))*np.angle(x[1:]*np.conj(x[:-1]))
    return {
        "cfo_hz": float(cfo),
        "cfo_std_hz": float(np.std(cfo_inst)),
        "iq_gain_alpha": float(alpha),
        "iq_phase_deg": float(phi_deg),
        "rise_time_us": float(rt_us),
        "psd_centroid_hz": float(fcent),
        "psd_pnr_db": float(pnr),
        "bw_3db_hz": float(bw),
        "mag_mean": float(np.mean(np.abs(x))),
        "i_std": float(np.std(np.real(x))),
        "q_std": float(np.std(np.imag(x))),
    }

# ---------------- Plot helpers ----------------
def ecdf(vals):
    a = np.asarray(vals)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(a)
    n = xs.size
    F = np.arange(1, n+1, dtype=float) / n
    return xs, F

def save_cdf(feature_name, values, outdir):
    xs, F = ecdf(values)
    if xs.size == 0:
        return None
    fig = plt.figure()
    plt.plot(xs, F, drawstyle="steps-post")
    plt.xlabel(feature_name)
    plt.ylabel("F(x)")
    plt.title(f"CDF: {feature_name}  (N={len(values)})")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    path = os.path.join(outdir, f"cdf_{sanitize(feature_name)}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path

def save_hist(feature_name, values, outdir, bins=60):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    fig = plt.figure()
    plt.hist(a, bins=bins)
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(f"Histogram: {feature_name}  (N={len(values)})")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    path = os.path.join(outdir, f"hist_{sanitize(feature_name)}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path

def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with pkt_*.fc32 files (float32 interleaved I,Q)")
    ap.add_argument("--fs", type=float, required=True, help="Complex sample rate AFTER decim (e.g., 2e6)")
    ap.add_argument("--out-csv", default="features.csv")
    ap.add_argument("--plots-dir", default="plots", help="Directory to save plots (created if missing)")
    ap.add_argument("--make-cdfs", action="store_true", default=True, help="Make one CDF per feature (default on)")
    ap.add_argument("--no-cdfs", action="store_true", help="Disable CDF plots")
    ap.add_argument("--make-hists", action="store_true", help="Also save histograms per feature")
    ap.add_argument("--max-files", type=int, default=0, help="Process at most this many packets (0 = all)")
    args = ap.parse_args()

    # Handle cdf toggles
    make_cdfs = args.make_cdfs and not args.no_cdfs

    files = sorted(Path(args.dir).glob("pkt_*.fc32"))
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    rows = []
    for i, fn in enumerate(files):
        iq = np.fromfile(fn, dtype=np.float32)
        if iq.size < 4:
            continue
        # float32 interleaved (I,Q) => complex64 view is okay
        x = iq.view(np.complex64)
        feats = packet_features(x, args.fs)
        feats["file"] = fn.name
        rows.append(feats)

    if not rows:
        print("No feature rows produced")
        return

    # Write CSV
    keys = ["file"] + [k for k in rows[0].keys() if k!="file"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows → {args.out_csv}")

    # Aggregate per-feature arrays
    feat_arrays = {k: [] for k in keys if k != "file"}
    for r in rows:
        for k in feat_arrays.keys():
            feat_arrays[k].append(r[k])

    # Make plot directory
    Path(args.plots-dir if hasattr(args, 'plots-dir') else args.plots_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir

    # Save plots
    saved = []
    for feat_name, vals in feat_arrays.items():
        if make_cdfs:
            p = save_cdf(feat_name, vals, plots_dir)
            if p: saved.append(p)
        if args.make_hists:
            p = save_hist(feat_name, vals, plots_dir)
            if p: saved.append(p)
    if saved:
        print(f"Saved {len(saved)} plot(s) → {plots_dir}")

    # Tiny text summary
    def robust_stats(a):
        a = np.asarray(a); a = a[np.isfinite(a)]
        if a.size == 0: return None
        q25, q50, q75 = np.percentile(a, [25, 50, 75])
        iqr = q75 - q25
        return dict(median=float(q50), iqr=float(iqr), min=float(np.min(a)), max=float(np.max(a)))

    print("\nFeature summaries:")
    for k, vals in feat_arrays.items():
        s = robust_stats(vals)
        if s:
            print(f"  {k:>18}: median={s['median']:.4g}  IQR={s['iqr']:.4g}  min={s['min']:.4g}  max={s['max']:.4g}")

if __name__ == "__main__":
    main()
