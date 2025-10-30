#!/usr/bin/env python3
"""
Fast analyzer for BLE fingerprints by advertiser MAC (AdvA), reading the
integrated feature file produced by the new C++ pipeline.

Expected CSV header (minimum):
  pkt_idx, pcap_ts, rf_channel, pdu_type, adv_addr, access_address,
  cfo_quick_hz, cfo_centroid_hz, cfo_two_stage_hz, cfo_std_hz, cfo_std_sym_hz,
  iq_gain_alpha, iq_phase_deg_deg, rise_time_us, psd_centroid_hz, psd_pnr_db,
  bw_3db_hz, gated_len_us, cfo_two_stage_coarse_hz

Outputs:
  - agg/agg_per_mac.csv : per-MAC medians & IQRs (CFO stats use ADV_* only)
  - optional PNGs under plots/ (overall, per-MAC, agg bar/IQR, time-series)
  - optional clustering on per-MAC medians

Usage (fast defaults):
  python3 analyze_by_mac_from_features.py \
      --features-csv features_integrated.csv \
      --outdir mac_analysis_out_fast

Optional:
  --plots --per-mac-plots --top-macs 6 --agg-plots --time-plots --cluster
"""

import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# sklearn is optional; clustering is disabled by default
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# ---------------- small utils ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True); return p
def sanitize(s): return str(s).replace("/", "_").replace(" ", "_").replace(":", "")
def parse_float(v):
    try: return float(v)
    except Exception: return np.nan

def read_csv_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        hdr = r.fieldnames or []
    return rows, hdr

def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def is_cfo_col(name: str) -> bool:
    return name.startswith("cfo_")

# ---------------- plotting (histograms + bar/IQR + time) ----------------
def save_hist(vals, title, xlabel, outpath, bins=50):
    a = np.asarray(vals, float); a = a[np.isfinite(a)]
    if a.size == 0: return False
    fig = plt.figure()
    plt.hist(a, bins=bins)
    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True

def save_timeseries(t, y, title, ylabel, outpath, thin=1):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if not np.any(m): return False
    if thin > 1:
        idx = np.arange(m.sum())[::thin]
        tt = t[m][idx]; yy = y[m][idx]
    else:
        tt = t[m]; yy = y[m]
    fig = plt.figure()
    plt.plot(tt, yy, linewidth=1)
    plt.xlabel("time (s)"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True

def save_bar_iqr(labels, medians, iqrs, title, ylabel, outpath, rotate=45):
    """Bar plot of medians with IQR/2 error (±IQR/2 around median)."""
    L = list(labels)
    m = np.asarray(medians, float)
    q = np.asarray(iqrs, float)
    good = np.isfinite(m) & np.isfinite(q)
    if not np.any(good): return False
    L = [L[i] for i in range(len(L)) if good[i]]
    m = m[good]; q = q[good]
    yerr = q/2.0
    x = np.arange(len(L))
    fig = plt.figure()
    plt.bar(x, m, yerr=yerr, capsize=3)
    plt.xticks(x, L, rotation=rotate, ha="right")
    plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True

# ---------------- clustering (optional) ----------------
def auto_k_cluster(X, kmin=2, kmax=8, seed=0):
    if not HAVE_SK:  # lightweight fallback
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-12
        Xz = (X-mu)/sd
        from math import inf
        best = None; best_sc = -inf; best_k = None
        for k in range(kmin, kmax+1):
            rng = np.random.default_rng(seed)
            n = Xz.shape[0]
            C = Xz[rng.choice(n, size=k, replace=False)].copy()
            labels = np.zeros(n, dtype=int)
            for _ in range(25):
                d2 = np.sum((Xz[:,None,:]-C[None,:,:])**2, axis=2)
                labels = np.argmin(d2, axis=1)
                for j in range(k):
                    m = labels==j
                    if np.any(m): C[j] = Xz[m].mean(axis=0)
            inertia = float(np.sum((Xz - C[labels])**2))
            sc = -inertia
            if sc > best_sc: best_sc, best, best_k = sc, (labels, C), k
        return best[0], best[1], best_k, (mu, sd)

    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    best = None; best_sc = -1e9; best_k = None
    for k in range(kmin, kmax+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(Xz)
        try: sc = silhouette_score(Xz, labels)
        except Exception: sc = -1e9
        if sc > best_sc:
            best_sc, best, best_k = sc, (labels, km.cluster_centers_), k
    return best[0], best[1], best_k, scaler

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--outdir", default="mac_analysis_out_fast")
    # FAST defaults: everything off unless asked
    ap.add_argument("--plots", action="store_true", help="Make overall HISTOGRAMS for selected features")
    ap.add_argument("--per-mac-plots", action="store_true", help="Also make per-MAC HISTOGRAMS for top-N MACs")
    ap.add_argument("--agg-plots", action="store_true", help="Per-MAC median+IQR bar plots for selected features")
    ap.add_argument("--top-macs", type=int, default=8, help="How many MACs to plot when --per-mac-plots/--agg-plots")
    ap.add_argument("--time-plots", action="store_true", help="Per-feature time-series plots (uses pcap_ts from CSV)")
    ap.add_argument("--cluster", action="store_true", help="Per-MAC clustering on aggregated medians")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    # Include ALL CFO methods by default in the plotting list (match integrated names)
    ap.add_argument("--plot-features",
        default="cfo_two_stage_hz,cfo_quick_hz,cfo_centroid_hz,cfo_std_hz,cfo_std_sym_hz,cfo_centroid_hz_signfixed,cfo_joint_hz,"
                "iq_gain_alpha,iq_phase_deg_deg,psd_pnr_db,bw_3db_hz",
        help="Comma list of feature names to plot"
    )
    # Which PDUs are considered advertiser-originated for CFO usage
    ap.add_argument("--pdu-allow", default="0x00,0x02,0x04,0x06",
                    help="Comma list of PDU types to treat as advertiser-originated (default ADV_*).")
    args = ap.parse_args()

    out_plots = ensure_dir(Path(args.outdir)/"plots")
    out_agg   = ensure_dir(Path(args.outdir)/"agg")
    out_clu   = ensure_dir(Path(args.outdir)/"cluster")

    # 1) Load features (integrated CSV)
    rows, hdr = read_csv_rows(args.features_csv)
    if not rows:
        print("No features found", file=sys.stderr); sys.exit(1)

    required_meta = {"adv_addr", "pdu_type", "pcap_ts"}
    missing = [c for c in required_meta if c not in hdr]
    if missing:
        print(f"[ERROR] Missing required columns in features CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    # Meta
    macs = np.array([ (r.get("adv_addr") or "UNK").upper() for r in rows ], dtype=object)
    # pdu_type may be "0xN" or decimal
    def _pdu(v):
        try:
            s = str(v).strip()
            return int(s, 0) if s != "" and s.lower() != "none" else None
        except Exception:
            return None
    pdu_types = np.array([ _pdu(r.get("pdu_type")) for r in rows ], dtype=object)
    t = np.array([ parse_float(r.get("pcap_ts", np.nan)) for r in rows ], float)

    # 2) Feature columns = everything numeric except known meta
    drop_cols = {
        "pkt_idx","pcap_ts","rf_channel","pdu_type","adv_addr","access_address"
    }
    feat_cols = [c for c in hdr if c not in drop_cols]

    # numeric matrix (same order as feat_cols)
    X = np.array([[parse_float(r.get(c, np.nan)) for c in feat_cols] for r in rows], float)

    # 3) CFO-only mask: restrict to advertiser-originated PDUs
    allow_set = set(int(s.strip(), 0) for s in args.pdu_allow.split(",") if s.strip())
    pdu_is_adv = np.array([ (pt in allow_set) if (pt is not None) else False for pt in pdu_types ], bool)

    # 4) Group indices per MAC (vectorized aggregation downstream)
    uniq_macs, inv = np.unique(macs, return_inverse=True)
    groups = {mac: np.where(inv == i)[0] for i, mac in enumerate(uniq_macs)}
    counts = {mac: len(groups[mac]) for mac in uniq_macs}

    # 5) Aggregate per MAC (medians & IQRs)
    # Choose a focused, compact set; include ALL CFO variants we have
    preferred = [
        "cfo_two_stage_hz", "cfo_quick_hz", "cfo_centroid_hz", "cfo_centroid_hz_signfixed", "cfo_joint_hz",
        "cfo_std_hz", "cfo_std_sym_hz",
        "cfo_two_stage_coarse_hz",
        "iq_gain_alpha", "iq_phase_deg_deg", "psd_pnr_db",
        "bw_3db_hz", "psd_centroid_hz", "rise_time_us", "gated_len_us"
    ]
    agg_src = [c for c in preferred if c in feat_cols]

    agg_rows = []
    for mac in uniq_macs:
        idx_all = groups[mac]
        row = {"mac": mac, "count": int(idx_all.size)}

        # Build per-feature stats; CFO columns use advertiser-only subset
        for j, c in enumerate(feat_cols):
            if is_cfo_col(c):
                idx = idx_all[np.isin(idx_all, np.where(pdu_is_adv)[0])]
            else:
                idx = idx_all

            if idx.size == 0:
                continue
            col = X[idx, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            q25, q50, q75 = np.percentile(col, [25, 50, 75])
            row[f"{c}_median"] = f"{q50:.6g}"
            row[f"{c}_iqr"]    = f"{(q75-q25):.6g}"
        agg_rows.append(row)

    # Keep focused columns in output
    keep = ["mac","count"]
    for c in agg_src:
        keep += [f"{c}_median", f"{c}_iqr"]
    norm_rows = [{k: r.get(k, "") for k in keep} for r in agg_rows]
    agg_path = Path(out_agg)/"agg_per_mac.csv"
    write_csv(agg_path, norm_rows, keep)
    print(f"[OK] Wrote per-MAC agg → {agg_path}")

    # 6) Optional plots (overall + per-MAC) with CFO-only advertiser gating
    if args.plots or args.per_mac_plots or args.agg_plots or args.time_plots:
        plot_feats = [s.strip() for s in args.plot_features.split(",") if s.strip()]
        plot_feats = [c for c in plot_feats if c in feat_cols]
        if not plot_feats:
            plot_feats = [c for c in ("cfo_two_stage_hz","cfo_quick_hz","cfo_centroid_hz","cfo_centroid_hz_signfixed","cfo_joint_hz",) if c in feat_cols]

    if args.plots:
        for c in plot_feats:
            j = feat_cols.index(c)
            if is_cfo_col(c):
                mask = pdu_is_adv & np.isfinite(X[:, j])
                data = X[mask, j]
            else:
                data = X[:, j]
            save_hist(data, f"Histogram (overall): {c}", c,
                      str(Path(out_plots)/f"hist_overall_{sanitize(c)}.png"))

    if args.per_mac_plots:
        top = sorted(uniq_macs, key=lambda m: -counts[m])[:max(1, args.top_macs)]
        for c in plot_feats:
            j = feat_cols.index(c)
            for mac in top:
                idx = groups[mac]
                if is_cfo_col(c):
                    idx = idx[np.isin(idx, np.where(pdu_is_adv)[0])]
                col = X[idx, j] if idx.size else np.array([])
                if np.isfinite(col).sum() == 0: continue
                save_hist(col, f"Histogram: {c} (MAC={mac})", c,
                          str(Path(out_plots)/f"hist_{sanitize(c)}__MAC_{sanitize(mac)}.png"))

    # 7) Optional per-MAC median+IQR bar plots (error bars) – based on agg CSV (CFOs adv-only)
    if args.agg_plots:
        agg_rows2, hdr2 = read_csv_rows(agg_path)
        mac_list = [r["mac"] for r in agg_rows2]
        counts_map = {m:int(r.get("count", "0") or 0) for m,r in zip(mac_list, agg_rows2)}
        top = sorted(mac_list, key=lambda m: -counts_map.get(m,0))[:max(1, args.top_macs)]
        for feat in plot_feats:
            med_col = f"{feat}_median"
            iqr_col = f"{feat}_iqr"
            labels, meds, iqrs = [], [], []
            for m in top:
                r = next((rr for rr in agg_rows2 if rr["mac"] == m), None)
                if not r: continue
                try:
                    med = float(r.get(med_col, ""))
                    iqr = float(r.get(iqr_col, ""))
                except Exception:
                    continue
                labels.append(m); meds.append(med); iqrs.append(iqr)
            if labels:
                save_bar_iqr(labels, meds, iqrs,
                             f"Per-MAC median±IQR: {feat} (top {len(labels)})",
                             feat,
                             str(Path(out_plots)/f"agg_bar_{sanitize(feat)}__top{len(labels)}.png"))

    # 8) Optional time-series (CFOs use advertiser-only packets); timestamps from CSV
    if args.time_plots:
        THIN = max(1, len(t)//2000)  # cap at ~2000 points per plot
        for c in plot_feats:
            j = feat_cols.index(c)
            if is_cfo_col(c):
                mask = pdu_is_adv
                save_timeseries(t[mask], X[mask, j], f"{c} over time (ADV only)", c,
                                str(Path(out_plots)/f"time_{sanitize(c)}_advonly.png"), thin=THIN)
            else:
                save_timeseries(t, X[:, j], f"{c} over time", c,
                                str(Path(out_plots)/f"time_{sanitize(c)}.png"), thin=THIN)

    # 9) Optional per-MAC clustering on aggregated medians (CFOs already adv-only in agg)
    if args.cluster:
        agg_rows2, hdr2 = read_csv_rows(agg_path)
        mac_list = [r["mac"] for r in agg_rows2]
        clu_cols = [h for h in hdr2 if h.endswith("_median") and h not in ("mac","count")]
        def _f(v):
            try: return float(v)
            except Exception: return np.nan
        X_mac = np.array([[_f(r.get(c, "")) for c in clu_cols] for r in agg_rows2], float)
        good = np.all(np.isfinite(X_mac), axis=1)
        mac_c = [m for m,g in zip(mac_list, good) if g]
        X_mac_c = X_mac[good]
        if X_mac_c.shape[0] >= 5 and X_mac_c.shape[1] >= 2:
            labels, centers, kbest, scaler = auto_k_cluster(X_mac_c, kmin=args.k_min, kmax=args.k_max, seed=0)
            out_mac_clu = Path(out_clu)/"cluster_assignments_per_mac.csv"
            with open(out_mac_clu, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["mac","cluster"])
                for m, c in zip(mac_c, labels): w.writerow([m, int(c)])
            print(f"[OK] Wrote per-MAC cluster assignments → {out_mac_clu}")
        else:
            print("[INFO] Clustering skipped (insufficient clean rows/dims).")

    print(f"Done. Outputs under: {args.outdir}")

if __name__ == "__main__":
    main()