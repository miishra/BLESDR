#!/usr/bin/env python3
"""
Fast analyzer for BLE fingerprints by advertiser MAC (AdvA).

- Loads per-packet features CSV produced by features_from_iqdir.py.
- If CSV lacks 'adv_addr'/'pdu_type'/'pcap_ts', parses PCAP (DLT 256/251) and aligns by index.
- Writes per-MAC aggregation CSV (medians & IQRs for a compact feature set including ALL CFO variants).
- **CFO features are computed/aggregated only from advertiser-originated PDUs** (ADV_*), others use all rows.

Optional (off by default for speed):
    * Overall HISTOGRAMS for selected features (--plots)
    * Per-MAC HISTOGRAMS for top-N MACs (--per-mac-plots --top-macs N)
    * Per-MAC median + IQR BAR plots (--agg-plots)  ← error bars show IQR/2 above/below median
    * Time-series plots using PCAP timestamps (--time-plots)
    * Per-MAC clustering (--cluster)

Usage (fast defaults):
  python3 analyze_by_mac_fast.py \
    --features-csv features_ch37.csv \
    --pcap out_ch37.pcap \
    --outdir mac_analysis_out_fast

Enable extras as needed, e.g.:
  --plots --per-mac-plots --top-macs 6 --agg-plots --time-plots --cluster
"""

import argparse, csv, sys, struct, re
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

# ---------------- PCAP parsing (DLT 256 or 251) ----------------
def parse_pcap_meta(pcap_path):
    """
    Returns (mac_list, pdu_type_list, ts_list). Handles:
      - DLT 256: [10B RF hdr][4B AA][LL hdr..CRC]
      - DLT 251: [4B AA][LL hdr..CRC]
    AdvA offsets:
      ADV_IND(0), ADV_NONCONN_IND(2), ADV_SCAN_IND(6), SCAN_RSP(4): payload[0:6]
      SCAN_REQ(3), CONNECT_REQ(5): payload[6:12]
    """
    macs, pdus, tss = [], [], []
    with open(pcap_path, "rb") as f:
        gh = f.read(24)
        if len(gh) != 24: raise RuntimeError("PCAP: short global header")
        magic_le = struct.unpack("<I", gh[:4])[0]
        magic_be = struct.unpack(">I", gh[:4])[0]
        if magic_le == 0xA1B2C3D4:
            endian = "<"
            _,_,_,_,_,_, linktype = struct.unpack("<IHHIIII", gh)
        elif magic_be == 0xA1B2C3D4:
            endian = ">"
            _,_,_,_,_,_, linktype = struct.unpack(">IHHIIII", gh)
        else:
            raise RuntimeError("PCAP: unsupported magic")
        rec_hdr = endian + "IIII"
        LL_OFF = 10+4 if linktype == 256 else (4 if linktype == 251 else 0)

        while True:
            rh = f.read(16)
            if len(rh) != 16: break
            ts_sec, ts_usec, incl, _ = struct.unpack(rec_hdr, rh)
            buf = f.read(incl)
            if len(buf) != incl: break
            ts = ts_sec + ts_usec*1e-6

            mac = None
            pdu = None
            if incl >= LL_OFF + 2 + 3:
                h0 = buf[LL_OFF+0]
                h1 = buf[LL_OFF+1]
                pdu = h0 & 0x0F
                ln  = h1 & 0x3F
                if incl >= LL_OFF + 2 + ln + 3:
                    pl = buf[LL_OFF+2 : LL_OFF+2+ln]
                    if pdu in (0,2,6,4):
                        if len(pl) >= 6: mac = ":".join(f"{b:02X}" for b in pl[0:6][::-1])
                    elif pdu in (3,5):
                        if len(pl) >= 12: mac = ":".join(f"{b:02X}" for b in pl[6:12][::-1])

            macs.append(mac)
            pdus.append(pdu)
            tss.append(ts)
    return macs, pdus, tss

# ---------------- plotting (histograms + bar/IQR) ----------------
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
    if not HAVE_SK:  # super fast fallback: single k
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
    ap.add_argument("--pcap", help="PCAP (DLT 256/251) for timestamps or when CSV lacks adv_addr/pdu_type/pcap_ts")
    ap.add_argument("--outdir", default="mac_analysis_out_fast")
    # FAST defaults: everything off unless asked
    ap.add_argument("--plots", action="store_true", help="Make overall HISTOGRAMS for selected features")
    ap.add_argument("--per-mac-plots", action="store_true", help="Also make per-MAC HISTOGRAMS for top-N MACs")
    ap.add_argument("--agg-plots", action="store_true", help="Per-MAC median+IQR bar plots for selected features")
    ap.add_argument("--top-macs", type=int, default=8, help="How many MACs to plot when --per-mac-plots/--agg-plots")
    ap.add_argument("--time-plots", action="store_true", help="Per-feature time-series plots")
    ap.add_argument("--cluster", action="store_true", help="Per-MAC clustering on aggregated medians")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    # Include ALL CFO methods by default in the plotting list
    ap.add_argument("--plot-features",
                    default="cfo_two_stage_hz,cfo_hz,cfo_preaa_hz,cfo_centroid_hz,cfo_std_hz,cfo_std_hz_sym,iq_gain_alpha,psd_pnr_db,bw_3db_hz",
                    help="Comma list of feature names to plot")
    # Which PDUs are considered "advertiser-originated" for CFO usage
    ap.add_argument("--pdu-allow", default="0x00,0x02,0x04,0x06",
                    help="Comma list of PDU types to treat as advertiser-originated (default ADV_*).")
    args = ap.parse_args()

    out_plots = ensure_dir(Path(args.outdir)/"plots")
    out_agg   = ensure_dir(Path(args.outdir)/"agg")
    out_clu   = ensure_dir(Path(args.outdir)/"cluster")

    # 1) Load features
    rows, hdr = read_csv_rows(args.features_csv)
    if not rows:
        print("No features found", file=sys.stderr); sys.exit(1)

    have_adv = ("adv_addr" in (hdr or []))
    have_pdu = ("pdu_type" in (hdr or []))
    have_ts  = ("pcap_ts"  in (hdr or []))
    files = [r.get("file","") for r in rows]

    # Feature columns = everything numeric except known meta
    drop_cols = {"file","access_address","ts","timestamp","time","adv_addr","pdu_type","pcap_ts"}
    feat_cols = [c for c in hdr if c not in drop_cols]
    X = np.array([[parse_float(r.get(c, np.nan)) for c in feat_cols] for r in rows], float)

    # 2) MACs + PDU types + timestamps
    # From CSV if available; else parse PCAP
    if have_adv:
        macs = np.array([ (r.get("adv_addr") or "UNK").upper() for r in rows ], dtype=object)
    else:
        if not args.pcap:
            print("features CSV has no 'adv_addr'; please provide --pcap", file=sys.stderr)
            sys.exit(1)
        macs_pcap, _, _ = parse_pcap_meta(args.pcap)
        n = min(len(files), len(macs_pcap))
        macs = np.array([ (m if m else "UNK") for m in macs_pcap[:n] ], dtype=object)
        X = X[:n]; files = files[:n]

    if have_pdu:
        pdu_types = np.array([ None if (r.get("pdu_type") in (None,"")) else int(str(r.get("pdu_type")), 0)
                               for r in rows[:len(macs)] ], dtype=object)
    else:
        if not args.pcap:
            print("features CSV has no 'pdu_type'; please provide --pcap", file=sys.stderr)
            sys.exit(1)
        _, pdus_pcap, _ = parse_pcap_meta(args.pcap)
        pdu_types = np.array(pdus_pcap[:len(macs)], dtype=object)

    if have_ts:
        t = np.array([parse_float(r.get("pcap_ts", np.nan)) for r in rows[:len(macs)]], float)
    else:
        if args.pcap:
            _, _, tpcap = parse_pcap_meta(args.pcap)
            t = np.asarray(tpcap[:len(macs)], float)
        else:
            t = np.arange(len(macs), dtype=float)

    # Truncate all to same length if needed
    n = min(len(macs), X.shape[0], len(pdu_types), len(t))
    macs = macs[:n]; X = X[:n]; pdu_types = pdu_types[:n]; t = t[:n]

    # CFO-only mask: restrict to advertiser-originated PDUs
    allow_set = set(int(s.strip(), 0) for s in args.pdu_allow.split(",") if s.strip())
    pdu_is_adv = np.array([ (pt in allow_set) if (pt is not None) else False for pt in pdu_types ], bool)

    # 3) Group indices per MAC (vectorized aggregation downstream)
    uniq_macs, inv = np.unique(macs, return_inverse=True)
    groups = {mac: np.where(inv == i)[0] for i, mac in enumerate(uniq_macs)}
    counts = {mac: len(groups[mac]) for mac in uniq_macs}

    # 4) Aggregate per MAC (medians & IQRs)
    pref = [
        "cfo_two_stage_hz", "cfo_hz", "cfo_preaa_hz", "cfo_centroid_hz",
        "cfo_std_hz", "cfo_std_hz_sym",
        "iq_gain_alpha", "psd_pnr_db", "bw_3db_hz", "psd_centroid_hz", "iq_phase_deg", "rise_time_us"
    ]
    agg_src = [c for c in pref if c in feat_cols]

    agg_rows = []
    for mac in uniq_macs:
        idx_all = groups[mac]
        row = {"mac": mac, "count": int(idx_all.size)}

        # Build per-feature stats; for CFO columns use advertiser-only subset
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

    # 5) Optional plots (overall + per-MAC) with CFO-only advertiser gating
    if args.plots or args.per_mac_plots or args.agg_plots or args.time_plots:
        plot_feats = [s.strip() for s in args.plot_features.split(",") if s.strip()]
        plot_feats = [c for c in plot_feats if c in feat_cols]
        if not plot_feats:
            plot_feats = [c for c in ("cfo_two_stage_hz","cfo_hz","cfo_preaa_hz","cfo_centroid_hz") if c in feat_cols]

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

    # 6) Optional per-MAC median+IQR bar plots (error bars) – already based on agg CSV (CFOs adv-only)
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

    # 7) Optional time-series (CFOs use advertiser-only packets)
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

    # 8) Optional per-MAC clustering on aggregated medians (CFOs already adv-only in agg)
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