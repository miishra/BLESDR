#!/usr/bin/env python3
"""
Analyze BLE fingerprints by advertiser MAC (AdvA) using features + PCAP:

- Load per-packet feature CSV (from features_from_iqdir.py).
- Parse PCAP (DLT 251) produced by iq2pcap to extract AdvA and timestamps.
- Align by packet index (PCAP write order == IQ dump order).
- Aggregate per MAC; save CSV with medians and IQRs.
- Plot per-feature distributions (overall + per-MAC): CDFs (and optional histograms).
- Unsupervised clustering with auto-k (silhouette; elbow fallback).
- Time-stability plots for individual features and full fingerprint vector.
- Outputs are PNGs suitable for IEEE S&P (no fancy themes, single chart per figure).

Usage:
  python3 analyze_by_mac.py \
    --features-csv features_ch37.csv \
    --pcap out_ch37.pcap \
    --outdir mac_analysis_out \
    --make-hists \
    --k-min 2 --k-max 12
"""

import argparse, csv, os, sys, struct, math
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- optional sklearn ---
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# ---------------- I/O helpers ----------------
def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_").replace(":", "")

# ---------------- PCAP (DLT 251) parser ----------------
def parse_pcap_adv_macs(pcap_path):
    """
    Return (mac_list, ts_sec_list)
      mac_list[i]  = 'AA:BB:CC:DD:EE:FF' (advertiser MAC where applicable; None if unknown)
      ts_sec_list[i] = float seconds (ts_sec + ts_usec*1e-6)
    Assumes each packet payload = BLE LL PDU: header(2) + payload + CRC(3)
    Header: byte0: PDU type (low 4 bits), Tx/RxAdd bits; byte1: length (lower 6 bits)
    AdvA offset by type:
      ADV_IND(0), ADV_NONCONN_IND(2), ADV_SCAN_IND(6), SCAN_RSP(4): AdvA at payload[0:6]
      SCAN_REQ(3), CONNECT_REQ(5): AdvA at payload[6:12] (after ScanA/InitA)
    """
    macs = []
    tss  = []

    with open(pcap_path, "rb") as f:
        ghdr = f.read(24)
        if len(ghdr) != 24:
            raise RuntimeError("PCAP: short global header")
        magic = struct.unpack("<I", ghdr[:4])[0]
        if magic != 0xA1B2C3D4:
            # try swapped
            magic_be = struct.unpack(">I", ghdr[:4])[0]
            if magic_be == 0xA1B2C3D4:
                endian = ">"
            else:
                raise RuntimeError("PCAP: unsupported magic")
        else:
            endian = "<"

        rec_hdr_fmt = endian + "IIII"
        while True:
            rh = f.read(16)
            if not rh:
                break
            if len(rh) != 16:
                break
            ts_sec, ts_usec, incl, orig = struct.unpack(rec_hdr_fmt, rh)
            payload = f.read(incl)
            if len(payload) != incl:
                break

            mac_str = None
            ts = ts_sec + ts_usec * 1e-6

            if incl >= 2+3:  # need at least header+CRC
                hdr0 = payload[0]
                hdr1 = payload[1]
                pdu_type = hdr0 & 0x0F
                length = hdr1 & 0x3F
                # payload bytes (excluding 2 header) start at payload[2]
                if incl >= 2 + length + 3:  # CRC present
                    pl = payload[2:2+length]  # payload w/o CRC
                    # AdvA placement by type:
                    if pdu_type in (0, 2, 6, 4):  # ADV_IND, ADV_NONCONN_IND, ADV_SCAN_IND, SCAN_RSP
                        if len(pl) >= 6:
                            advA = pl[0:6]
                            mac_str = ":".join(f"{b:02X}" for b in advA[::-1])  # BLE broadcast addresses are LSB-first on air
                    elif pdu_type in (3, 5):  # SCAN_REQ, CONNECT_REQ: [ScanA/InitA(6)] + AdvA(6) + ...
                        if len(pl) >= 12:
                            advA = pl[6:12]
                            mac_str = ":".join(f"{b:02X}" for b in advA[::-1])
                    else:
                        # Other PDUs (rare on adv channels)
                        mac_str = None

            macs.append(mac_str)
            tss.append(ts)

    return macs, tss

# ---------------- small utils ----------------
def parse_float(v):
    try: return float(v)
    except Exception: return np.nan

def ecdf(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.array([]), np.array([])
    xs = np.sort(a)
    F = np.arange(1, xs.size+1) / xs.size
    return xs, F

# ---------------- plotting ----------------
def save_cdf(vals, title, xlabel, outpath):
    xs, F = ecdf(vals)
    if xs.size == 0: return False
    fig = plt.figure()
    plt.plot(xs, F, drawstyle="steps-post")
    plt.xlabel(xlabel); plt.ylabel("F(x)"); plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.savefig(outpath, dpi=180, bbox_inches="tight"); plt.close(fig)
    return True

def save_hist(vals, title, xlabel, outpath, bins=60):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0: return False
    fig = plt.figure()
    plt.hist(a, bins=bins)
    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.savefig(outpath, dpi=180, bbox_inches="tight"); plt.close(fig)
    return True

def save_timeseries(t, y, title, ylabel, outpath):
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    if not np.any(m): return False
    fig = plt.figure()
    plt.plot(t[m], y[m], linewidth=1)
    plt.xlabel("time (s)"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.savefig(outpath, dpi=180, bbox_inches="tight"); plt.close(fig)
    return True

def rolling_median(y, win=21):
    a = np.asarray(y, dtype=float)
    if a.size == 0 or win <= 1: return a
    r = win//2
    out = np.copy(a)
    for i in range(a.size):
        lo = max(0, i-r); hi = min(a.size, i+r+1)
        out[i] = np.median(a[lo:hi])
    return out

# ---------------- clustering ----------------
def kmeans_fallback(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = np.sum((X[:,None,:]-C[None,:,:])**2, axis=2)
        labels = np.argmin(d2, axis=1)
        for j in range(k):
            m = labels==j
            if np.any(m): C[j] = X[m].mean(axis=0)
            else: C[j] = X[rng.integers(0,n)]
    inertia = float(np.sum((X - C[labels])**2))
    return labels, C, inertia

def silhouette_fallback(X, labels):
    n = X.shape[0]
    if n > 2500:
        idx = np.random.default_rng(0).choice(n, size=1500, replace=False)
        X = X[idx]; labels = labels[idx]
    D = np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))
    s = []
    for i in range(X.shape[0]):
        same = labels==labels[i]; other = labels!=labels[i]
        a = np.mean(D[i, same]) if np.sum(same)>1 else 0.0
        b = np.inf
        for c in np.unique(labels[other]):
            b = min(b, np.mean(D[i, labels==c]))
        s.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
    return float(np.mean(s)) if s else 0.0

def auto_k_cluster(X, kmin=2, kmax=10, seed=0):
    if HAVE_SK:
        scaler = StandardScaler().fit(X)
        Xz = scaler.transform(X)
    else:
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)+1e-12
        scaler = (mu, sd); Xz = (X-mu)/sd

    best_k, best_score, best = None, -1e9, None
    inertias = []
    for k in range(kmin, kmax+1):
        if HAVE_SK:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            labels = km.fit_predict(Xz); inertia = float(km.inertia_)
            inertias.append((k, inertia))
            try: score = silhouette_score(Xz, labels)
            except Exception: score = -1e9
            if score > best_score: best_k, best_score, best = k, score, (labels, km.cluster_centers_)
        else:
            labels, C, inertia = kmeans_fallback(Xz, k, seed=seed)
            inertias.append((k, inertia))
            score = silhouette_fallback(Xz, labels)
            if score > best_score: best_k, best_score, best = k, score, (labels, C)

    if best_k is None or not np.isfinite(best_score) or best_score < 0:
        ks = np.array([k for k,_ in inertias]); I = np.array([i for _,i in inertias], float)
        drops = np.diff(I, prepend=I[0]); frac = np.abs(drops)/(I[0]+1e-12)
        idx = np.argmax(frac < 0.05); best_k = int(ks[max(1, idx)])
        if HAVE_SK:
            km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Xz)
            best = (km.labels_, km.cluster_centers_)
        else:
            best = kmeans_fallback(Xz, best_k, seed=seed)[:2]
        best_score = float("nan")

    labels, centers = best
    return labels, centers, best_k, scaler

# ---------------- time vector stability ----------------
def vector_stability_over_time(X, t):
    X = np.asarray(X, float); t = np.asarray(t, float)
    idx = np.argsort(t); X = X[idx]; t = t[idx]
    W = min(25, max(5, X.shape[0]//20))
    diffs = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        lo = max(0, i-W); hi = i
        if lo >= hi: diffs[i] = 0.0
        else:
            med = np.median(X[lo:hi], axis=0)
            diffs[i] = float(np.linalg.norm(X[i]-med))
    return t, diffs

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--pcap", required=True, help="PCAP (DLT 251) from iq2pcap")
    ap.add_argument("--outdir", default="mac_analysis_out")
    ap.add_argument("--make-hists", action="store_true")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    args = ap.parse_args()

    out_plots = ensure_dir(Path(args.outdir)/"plots")
    out_agg   = ensure_dir(Path(args.outdir)/"agg")
    out_clu   = ensure_dir(Path(args.outdir)/"cluster")

    # 1) Load feature rows
    feat_rows = read_csv(args.features_csv)
    if not feat_rows:
        print("No features found", file=sys.stderr); sys.exit(1)

    # Feature columns
    ignore = {"file", "access_address", "ts", "timestamp", "time"}
    feat_cols = [k for k in feat_rows[0].keys() if k not in ignore]

    files = [r.get("file","") for r in feat_rows]
    X = np.array([[parse_float(r[k]) for k in feat_cols] for r in feat_rows], float)

    # 2) Parse PCAP for MACs + timestamps
    macs, tpcap = parse_pcap_adv_macs(args.pcap)

    # Align by index (callback writes PCAP and IQ dumps in the same order)
    n = min(len(files), len(macs))
    if len(files) != len(macs):
        print(f"Warning: feature rows ({len(files)}) != pcap packets ({len(macs)}); truncating to {n}", file=sys.stderr)

    files = files[:n]; X = X[:n]; macs = macs[:n]; tpcap = np.asarray(tpcap[:n], float)

    # Some PDUs may not carry AdvA -> macs[i] could be None; label as "UNK"
    macs = [m if m else "UNK" for m in macs]

    # 3) Aggregate per MAC
    uniq_macs = sorted(set(macs))
    agg_rows = []
    for mac in uniq_macs:
        m = np.array([mm == mac for mm in macs], bool)
        sub = X[m]
        row = {"mac": mac, "count": int(np.sum(m))}
        for j, k in enumerate(feat_cols):
            col = sub[:, j]; col = col[np.isfinite(col)]
            if col.size == 0:
                row[f"{k}_median"] = ""; row[f"{k}_iqr"] = ""
            else:
                q25, q50, q75 = np.percentile(col, [25, 50, 75])
                row[f"{k}_median"] = f"{q50:.6g}"
                row[f"{k}_iqr"]    = f"{(q75-q25):.6g}"
        agg_rows.append(row)

    write_csv(Path(out_agg)/"agg_per_mac.csv",
              agg_rows,
              ["mac","count"] + [f"{k}_{suf}" for k in feat_cols for suf in ("median","iqr")])

    # 4) Distributions (overall + per-MAC)
    for j, k in enumerate(feat_cols):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            save_cdf(col, f"CDF (overall): {k}", k, str(Path(out_plots)/f"cdf_overall_{sanitize(k)}.png"))
            if args.make_hists:
                save_hist(col, f"Histogram (overall): {k}", k, str(Path(out_plots)/f"hist_overall_{sanitize(k)}.png"))
        for mac in uniq_macs:
            m = np.array([mm == mac for mm in macs], bool)
            colm = X[m, j]; colm = colm[np.isfinite(colm)]
            if colm.size:
                save_cdf(colm, f"CDF: {k} (MAC={mac})", k,
                         str(Path(out_plots)/f"cdf_{sanitize(k)}__MAC_{sanitize(mac)}.png"))
                if args.make_hists:
                    save_hist(colm, f"Histogram: {k} (MAC={mac})", k,
                              str(Path(out_plots)/f"hist_{sanitize(k)}__MAC_{sanitize(mac)}.png"))

    # 5) Clustering (auto-k)
    good = np.all(np.isfinite(X), axis=1)
    Xc = X[good]; mac_c = [macs[i] for i,g in enumerate(good) if g]; t_c = tpcap[good]
    if Xc.shape[0] >= 5 and Xc.shape[1] >= 2:
        labels, centers, kbest, scaler = auto_k_cluster(Xc, kmin=args.k_min, kmax=args.k_max, seed=0)

        # Save assignments
        rows = []
        for i in range(Xc.shape[0]):
            rows.append({"idx": i, "mac": mac_c[i], "cluster": int(labels[i]), "t": f"{t_c[i]:.6f}"})
        write_csv(Path(out_clu)/"cluster_assignments.csv", rows, ["idx","mac","cluster","t"])

        # PCA scatter colored by cluster
        if HAVE_SK:
            pca = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(n_components=2, random_state=0).fit(Xc)
            Z = pca.transform(Xc)
        else:
            mu = Xc.mean(axis=0, keepdims=True)
            U,S,Vt = np.linalg.svd(Xc-mu, full_matrices=False)
            Z = (Xc-mu) @ Vt[:2].T

        fig = plt.figure()
        for c in sorted(set(labels)):
            m = labels == c
            plt.plot(Z[m,0], Z[m,1], linestyle="", marker="o", label=f"cluster {c}")
        plt.xlabel("PC 1"); plt.ylabel("PC 2")
        plt.title(f"PCA of fingerprints (k={kbest})")
        plt.legend()
        plt.grid(True, linestyle=":", linewidth=0.5)
        fig.savefig(str(Path(out_plots)/"clusters_pca2.png"), dpi=180, bbox_inches="tight"); plt.close(fig)

        # Cluster sizes
        fig = plt.figure()
        sizes = [int(np.sum(labels==c)) for c in sorted(set(labels))]
        plt.bar([str(c) for c in sorted(set(labels))], sizes)
        plt.xlabel("cluster"); plt.ylabel("count")
        plt.title(f"Cluster sizes (k={kbest})")
        plt.grid(True, axis="y", linestyle=":", linewidth=0.5)
        fig.savefig(str(Path(out_plots)/"cluster_sizes.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
    else:
        print("Clustering skipped (insufficient clean data).")

    # 6) Time stability
    # Per-feature timeseries (overall)
    for j, k in enumerate(feat_cols):
        y = X[:, j]
        save_timeseries(tpcap, y, f"{k} over time", k, str(Path(out_plots)/f"time_{sanitize(k)}.png"))
        ym = rolling_median(y, win=min(101, max(5, len(y)//20)))
        fig = plt.figure()
        plt.plot(tpcap, ym, linewidth=1.5)
        plt.xlabel("time (s)"); plt.ylabel(k); plt.title(f"{k} (rolling median)")
        plt.grid(True, linestyle=":", linewidth=0.5)
        fig.savefig(str(Path(out_plots)/f"time_{sanitize(k)}_median.png"), dpi=180, bbox_inches="tight"); plt.close(fig)

    # Vector stability overall
    good2 = np.all(np.isfinite(X), axis=1) & np.isfinite(tpcap)
    if np.any(good2):
        t_vec, d_vec = vector_stability_over_time(X[good2], tpcap[good2])
        save_timeseries(t_vec, d_vec, "Fingerprint vector stability vs time", "||x - median(prev)||2",
                        str(Path(out_plots)/"time_vector_stability.png"))

    # Per-MAC vector stability
    if len(set(macs)) > 1:
        for mac in uniq_macs:
            m = (np.array(macs) == mac) & np.all(np.isfinite(X), axis=1) & np.isfinite(tpcap)
            if np.sum(m) < 5: continue
            t_vec, d_vec = vector_stability_over_time(X[m], tpcap[m])
            save_timeseries(t_vec, d_vec, f"Vector stability vs time (MAC={mac})", "||x - median(prev)||2",
                            str(Path(out_plots)/f"time_vector_stability__MAC_{sanitize(mac)}.png"))

    print(f"Done. Outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
