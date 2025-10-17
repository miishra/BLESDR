#!/usr/bin/env python3
import csv, numpy as np, matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

# ------------ Config ------------
ASSIGN = "mac_analysis_out/cluster/cluster_assignments.csv"
FEATS  = "features_ch37.csv"
OUT    = "mac_analysis_out/cluster_plots"
TOP_MACS_FOR_HEATMAP = 20     # heatmap width
MAX_MACS_FOR_INDIV_PLOTS = 24 # per-MAC bar + PCA plots (avoid exploding #figures)
FIG_DPI = 180

# --- DBSCAN settings (kept inline; only clustering logic is enhanced) ---
USE_DBSCAN = True
BASE_MIN_SAMPLES = 10           # floor; effective min_samples becomes max(BASE_MIN_SAMPLES, 2*#PCs)

Path(OUT).mkdir(parents=True, exist_ok=True)

# ------------ Load clustering assignments ------------
rows = []
with open(ASSIGN) as f:
    r = csv.DictReader(f)
    for q in r:
        rows.append(q)

labels = np.array([int(q["cluster"]) for q in rows])
macs   = np.array([q["mac"] for q in rows])

# ------------ Load features (to rebuild PCA) ------------
with open(FEATS) as f:
    r = csv.DictReader(f)
    feats = list(r)

# # choose numeric feature columns
# skip_cols = {"file", "access_address", "mac", "ts", "timestamp", "time"}
# feat_cols = [k for k in feats[0].keys() if k not in skip_cols]

# X = np.array([[float(q[k]) if q[k] not in ("", "nan", "NaN") else np.nan for k in feat_cols]
#               for q in feats], dtype=float)

# choose ONLY the six aggregated, robust features
wanted_feats = [
    "cfo_hz_median",
    "cfo_hz_iqr",
    "iq_gain_alpha_median",
    "iq_gain_alpha_iqr",
    "psd_pnr_db_median",
    "bw_3db_hz_median",
]

# Verify all exist; fail fast with a clear message if not
missing = [c for c in wanted_feats if c not in feats[0].keys()]
if missing:
    raise RuntimeError(
        f"Requested features not found in {FEATS}: {missing}\n"
        "Make sure you're pointing to the aggregated-per-MAC CSV."
    )

feat_cols = wanted_feats

X = np.array([[float(q[k]) if q[k] not in ("", "nan", "NaN") else np.nan for k in feat_cols]
              for q in feats], dtype=float)

# Align indices (defensive)
n = min(len(rows), X.shape[0])
X = X[:n]
labels = labels[:n]
macs = macs[:n]

# Drop rows with NaNs
good = np.all(np.isfinite(X), axis=1)
X = X[good]
labels = labels[good]
macs = macs[good]

# ------------ (ENHANCED) Optional: Re-cluster with DBSCAN ------------
if USE_DBSCAN:
    try:
        from sklearn.preprocessing import RobustScaler, QuantileTransformer
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        print(f"[WARN] scikit-learn not available ({e}); falling back to provided labels.")
    else:
        # 1) Robust scale (median/IQR) to limit influence of outliers
        Xs = RobustScaler().fit_transform(X)

        # 2) Gaussianize marginals to reduce skew and curved manifolds
        n_q = min(1000, X.shape[0])
        Xg = QuantileTransformer(n_quantiles=n_q, output_distribution="normal", random_state=0).fit_transform(Xs)

        # 3) PCA to a compact, whitened subspace (retain >=90% variance, at least 2 PCs)
        pca = PCA(n_components=min(12, Xg.shape[1]), whiten=True, random_state=0)
        Xp_full = pca.fit_transform(Xg)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, 0.90) + 1)
        k = max(2, min(k, Xp_full.shape[1]))
        Xp = Xp_full[:, :k]

        # 4) Adaptive min_samples
        min_samples = max(BASE_MIN_SAMPLES, 2*k)

        # 5) Auto-tune eps via kNN distances; pick setting that avoids one giant cluster
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(Xp)
        dists, _ = nn.kneighbors(Xp)
        kth = np.sort(dists[:, -1])  # distance to k-th neighbor

        percentiles = list(range(90, 59, -2))  # 90,88,...,60
        best = {"score": (-1, 1.0), "labels": None, "eps": None}  # (n_clusters, largest_frac) with lexicographic preference

        for p in percentiles:
            eps = float(np.percentile(kth, p))
            db = DBSCAN(eps=eps, min_samples=min_samples)
            lab = db.fit_predict(Xp)
            uniq = set(lab)
            n_noise = np.sum(lab == -1)
            n_clusters = len([c for c in uniq if c != -1])

            if n_clusters == 0:
                # everything noise or a single blob labelled -1
                cand = (0, 1.0)
            else:
                # fraction of largest non-noise cluster
                counts = Counter(lab[lab != -1])
                largest_frac = max(counts.values()) / float(np.sum(lab != -1))
                cand = (n_clusters, 1.0 - largest_frac)  # prefer more clusters, then smaller dominance

            # keep the best (more clusters, less dominance)
            if cand > best["score"]:
                best = {"score": cand, "labels": lab, "eps": eps}

        labels_db = best["labels"]
        if labels_db is None:
            # Fallback: single try at 80th percentile
            eps = float(np.percentile(kth, 80))
            labels_db = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xp)
            chosen_eps = eps
        else:
            chosen_eps = best["eps"]

        # overwrite labels
        labels = labels_db

        # persist assignments (same schema)
        out_assign = Path(ASSIGN)
        out_assign.parent.mkdir(parents=True, exist_ok=True)
        with out_assign.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx","mac","cluster","t"])
            for i,(m,c) in enumerate(zip(macs, labels)):
                w.writerow([i, m, int(c), float(i)])

        uniq = sorted(set(labels))
        n_clusters = len([c for c in uniq if c != -1])
        n_noise = int(np.sum(labels == -1))
        print(f"[DBSCAN] PCs={k}, min_samples={min_samples}, eps~{chosen_eps:.4g} | "
              f"clusters={n_clusters}, noise={n_noise}")

# ------------ PCA (SVD) ------------
mu = X.mean(axis=0, keepdims=True)
X0 = X - mu
U, S, Vt = np.linalg.svd(X0, full_matrices=False)  # X0 = U S Vt
# 2D and 3D scores
Z2 = X0 @ Vt[:2].T
Z3 = X0 @ Vt[:3].T

# ------------ Helper: color cycle by cluster ------------
def cluster_colors(unique_clusters: Iterable[int]) -> Dict[int, str]:
    # use matplotlib default color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    uniq = sorted(set(unique_clusters))
    return {c: prop_cycle[i % len(prop_cycle)] for i, c in enumerate(uniq)}

colors = cluster_colors(labels)

# ------------ 1) PCA 2D scatter (colored by cluster) ------------
plt.figure()
for c in sorted(set(labels)):
    m = (labels == c)
    plt.plot(Z2[m, 0], Z2[m, 1], ls="", marker="o", ms=3, label=f"cluster {c}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (2D) colored by cluster")
plt.grid(True, ls=":", lw=0.5)
plt.legend(markerscale=2, fontsize=8)
plt.savefig(f"{OUT}/clusters_pca2_replot.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# ------------ 2) PCA 3D scatter (colored by cluster) ------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for c in sorted(set(labels)):
    m = (labels == c)
    ax.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2], s=6, depthshade=False, label=f"cluster {c}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (3D) colored by cluster")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/clusters_pca3.png", dpi=FIG_DPI)
plt.close(fig)

# ------------ 3) Cluster × MAC counts (overall heatmap of top MACs) ------------
ct = defaultdict(Counter)  # ct[cluster][mac] = count
for m, c in zip(macs, labels):
    ct[c][m] += 1

top_macs = [m for m, _ in Counter(macs).most_common(TOP_MACS_FOR_HEATMAP)]
clusters_sorted = sorted(ct)
data = np.array([[ct[c][m] for m in top_macs] for c in clusters_sorted], dtype=float)

plt.figure(figsize=(max(6, 0.45*len(top_macs)), 2.6 + 0.25*len(clusters_sorted)))
plt.imshow(data, aspect="auto")
plt.xticks(range(len(top_macs)), top_macs, rotation=90)
plt.yticks(range(len(clusters_sorted)), [f"cluster {c}" for c in clusters_sorted])
plt.colorbar(label="count")
plt.title("Cluster × MAC counts (top MACs)")
plt.tight_layout()
plt.savefig(f"{OUT}/cluster_mac_heatmap.png", dpi=FIG_DPI)
plt.close()

# Save cluster×MAC table as CSV for traceability
out_csv = Path(OUT) / "cluster_mac_counts.csv"
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    hdr = ["cluster", "mac", "count"]
    w.writerow(hdr)
    for c in clusters_sorted:
        for m, k in ct[c].most_common():
            w.writerow([c, m, k])

# ------------ i) Histogram: #unique MACs per cluster ------------
unique_macs_per_cluster = {c: len(set(ct[c].keys())) for c in clusters_sorted}
plt.figure()
plt.bar([str(c) for c in clusters_sorted],
        [unique_macs_per_cluster[c] for c in clusters_sorted])
plt.xlabel("Cluster")
plt.ylabel("# Unique MACs")
plt.title("Unique MACs per cluster")
plt.grid(axis="y", ls=":", lw=0.5)
plt.savefig(f"{OUT}/unique_macs_per_cluster_hist.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# ------------ ii) Per-MAC clusters associated ------------
# Make per-MAC bar plots for the most frequent MACs and save a CSV summary.
mac_total = Counter(macs)
mac_ranked = [m for m, _ in mac_total.most_common(MAX_MACS_FOR_INDIV_PLOTS)]

# CSV summary: for each MAC, counts per cluster
mac_summary_csv = Path(OUT) / "per_mac_cluster_counts.csv"
with mac_summary_csv.open("w", newline="") as f:
    w = csv.writer(f)
    hdr = ["mac"] + [f"cluster_{c}" for c in clusters_sorted] + ["total"]
    w.writerow(hdr)
    for m in sorted(set(macs), key=lambda x: (-mac_total[x], x)):
        row = [m] + [ct[c][m] for c in clusters_sorted] + [mac_total[m]]
        w.writerow(row)

# Per-MAC bar plots (top N)
for m in mac_ranked:
    counts = [ct[c][m] for c in clusters_sorted]
    plt.figure()
    plt.bar([str(c) for c in clusters_sorted], counts, color=[colors[c] for c in clusters_sorted])
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(f"Cluster membership for MAC {m}")
    for i, v in enumerate(counts):
        if v > 0:
            plt.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.grid(axis="y", ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUT}/per_mac_clusters__{m.replace(':','')}.png", dpi=FIG_DPI)
    plt.close()

# ------------ iii) (already done) PCA 3D above ------------

# ------------ iv) Per-MAC PCA plots ------------
# We’ll highlight each MAC’s points in the global PCA space (PC1/PC2),
# plotting all other points faintly for context.
for m in mac_ranked:
    mask = (macs == m)
    if not np.any(mask):
        continue
    plt.figure()
    # background
    plt.plot(Z2[~mask, 0], Z2[~mask, 1], ls="", marker=".", ms=2, alpha=0.25, color="#999999", label="others")
    # this MAC, colored by cluster
    for c in sorted(set(labels[mask])):
        mm = mask & (labels == c)
        plt.plot(Z2[mm, 0], Z2[mm, 1], ls="", marker="o", ms=4, label=f"{m} (cluster {c})", color=colors[c])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA (2D): points for MAC {m}")
    plt.grid(True, ls=":", lw=0.5)
    plt.legend(fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(f"{OUT}/per_mac_pca2__{m.replace(':','')}.png", dpi=FIG_DPI)
    plt.close()

# ------------ Console status ------------
print(f"Saved plots to: {OUT}")
print(f"- clusters_pca2_replot.png")
print(f"- clusters_pca3.png")
print(f"- cluster_mac_heatmap.png")
print(f"- unique_macs_per_cluster_hist.png")
print(f"- per_mac_clusters__<MAC>.png (top {MAX_MACS_FOR_INDIV_PLOTS})")
print(f"- per_mac_pca2__<MAC>.png (top {MAX_MACS_FOR_INDIV_PLOTS})")
print(f"And CSVs:\n- {out_csv}\n- {mac_summary_csv}")