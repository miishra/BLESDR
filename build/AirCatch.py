#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CFO-based MAC-agnostic adversary detection (final version)

Adds:
  i)   3D PCA + saved plot
  ii)  Long window size (15 min)
  iii) Automatic cluster-count exploration
  iv)  Adversary MAC percentage per cluster
  v)   Duration threshold for adversary validity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mpl_toolkits.mplot3d import Axes3D  # noqa


# =========================
# Configuration
# =========================

ADV_CSV = "ble_dump_jan15_1_Adv_out.csv"

PAYLOAD_TAG = "4c001219"
ADV_PAYLOAD_TAG = "4c001219ff"

WINDOW_S = 120          # 2 minutes
K_RANGE = range(3, 20)  # clusters to test
OVERALL_CFO_WEIGHT = 1.5

MIN_DURATION_S = 100  # 10 minutes

CFO_COLS_RAW = [
    "CFO_Hz",
    "CFO_00_Hz",
    "CFO_11_Hz",
    "CFO_10_Hz",
    "CFO_01_Hz",
]

CFO_COLS_SEG = [
    "CFO_Hz",
    "CFO_00",
    "CFO_11",
    "CFO_10",
    "CFO_01",
]


# =========================
# Utilities
# =========================

def prepare_segments(df, window_s):
    df = df[df["payload"].str.contains(PAYLOAD_TAG, na=False)]
    df = df.dropna(subset=["timestamp", "AdvA"] + CFO_COLS_RAW)
    df = df.sort_values("timestamp")

    df["seg_id"] = (df["timestamp"] // window_s).astype(int)

    seg = (
        df.groupby("seg_id")
          .agg(
              t_start=("timestamp", "min"),
              t_end=("timestamp", "max"),
              n_packets=("AdvA", "count"),
              n_macs=("AdvA", "nunique"),
              CFO_Hz=("CFO_Hz", "mean"),
              CFO_00=("CFO_00_Hz", "mean"),
              CFO_11=("CFO_11_Hz", "mean"),
              CFO_10=("CFO_10_Hz", "mean"),
              CFO_01=("CFO_01_Hz", "mean"),
              adv_macs=("payload",
                        lambda x: x.str.contains(ADV_PAYLOAD_TAG).sum()),
          )
          .reset_index()
    )

    seg["duration"] = seg["t_end"] - seg["t_start"]
    seg["churn"] = seg["n_macs"] / seg["n_packets"]
    return seg


def cfo_feature_matrix(seg):
    X = StandardScaler().fit_transform(seg[CFO_COLS_SEG].values)
    X[:, 0] *= OVERALL_CFO_WEIGHT
    return X


# =========================
# Load & segment
# =========================

adv = pd.read_csv(ADV_CSV)
seg = prepare_segments(adv, WINDOW_S)

X = cfo_feature_matrix(seg)


print("\n=== Cluster count exploration ===")

n_segments = len(seg)
K_MAX_ALLOWED = max(2, n_segments - 1)

print(f"Segments available: {n_segments}")
print(f"Max clusters allowed: {K_MAX_ALLOWED}")

results = []

for k in K_RANGE:
    if k > K_MAX_ALLOWED:
        print(f"K={k:2d} | skipped (k > n_segments-1)")
        continue

    labels = AgglomerativeClustering(
        n_clusters=k, linkage="ward"
    ).fit_predict(X)

    seg["cluster"] = labels

    # ---- Safe silhouette handling ----
    n_labels = len(np.unique(labels))
    if 2 <= n_labels < len(X):
        sil = silhouette_score(X, labels)
    else:
        sil = np.nan

    cluster_stats = []
    for cid, g in seg.groupby("cluster"):
        cluster_stats.append({
            "cluster": cid,
            "mean_churn": g["churn"].mean(),
            "adv_mac_pct": g["adv_macs"].sum() / max(g["n_packets"].sum(), 1),
            "duration": len(g) * WINDOW_S,   # FIXED duration
        })

    dfc = pd.DataFrame(cluster_stats)
    max_row = dfc.loc[dfc["mean_churn"].idxmax()]

    results.append({
        "K": k,
        "silhouette": sil,
        "max_churn": max_row["mean_churn"],
        "adv_mac_pct": max_row["adv_mac_pct"],
        "duration": max_row["duration"],
        "n_labels": n_labels,
    })

    sil_str = f"{sil:.3f}" if not np.isnan(sil) else "NA"

    print(
        f"K={k:2d} | labels={n_labels:2d} | "
        f"sil={sil_str:>5} | "
        f"max_churn={max_row['mean_churn']:.3f} | "
        f"adv%={100*max_row['adv_mac_pct']:.1f}% | "
        f"dur={max_row['duration']:.1f}s"
    )

res_df = pd.DataFrame(results)


# =========================
# Choose K (stable & interpretable)
# =========================

K_FINAL = (
    res_df
    .assign(sil_filled=lambda d: d["silhouette"].fillna(-1))
    .sort_values(
        by=["adv_mac_pct", "duration", "sil_filled"],
        ascending=False
    )
    .iloc[0]["K"]
)

print(f"\n>>> Selected K = {int(K_FINAL)} <<<\n")


# =========================
# Final clustering
# =========================

labels = AgglomerativeClustering(
    n_clusters=int(K_FINAL), linkage="ward"
).fit_predict(X)

seg["cluster"] = labels

cluster_summary = []
for cid, g in seg.groupby("cluster"):
    cluster_summary.append({
        "cluster": cid,
        "mean_churn": g["churn"].mean(),
        "adv_mac_pct": g["adv_macs"].sum() / max(g["n_packets"].sum(), 1),
        "duration": g["duration"].sum(),
        "segments": len(g),
    })

summary_df = pd.DataFrame(cluster_summary)
print(summary_df.sort_values("mean_churn", ascending=False))


# =========================
# 3D PCA visualization
# =========================

pca = PCA(n_components=3)
X3 = pca.fit_transform(X)

adv_cluster = summary_df.loc[
    summary_df["mean_churn"].idxmax(), "cluster"
]

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for cid in summary_df["cluster"]:
    mask = seg["cluster"] == cid
    ax.scatter(
        X3[mask, 0],
        X3[mask, 1],
        X3[mask, 2],
        s=40,
        label=f"C{cid}",
        alpha=0.8,
    )

mask = seg["cluster"] == adv_cluster
ax.scatter(
    X3[mask, 0],
    X3[mask, 1],
    X3[mask, 2],
    s=120,
    facecolors="none",
    edgecolors="red",
    linewidths=2,
    label="Adversary",
)

ax.set_title("3D PCA – CFO segment clustering")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()

plt.tight_layout()
plt.savefig("cfo_3d_pca_adversary.png", dpi=300)
plt.show()


# =========================
# Final adversary decision
# =========================

ADV_PCT_MIN = 0.2     # 20% adversary packets (tunable)
CHURN_MIN   = 0.8     # high churn
DUR_MIN     = MIN_DURATION_S

adv_row = summary_df.loc[summary_df["cluster"] == adv_cluster].iloc[0]

print("\n=== Final adversary decision ===")
print(adv_row)

if (
    adv_row["mean_churn"] >= CHURN_MIN and
    adv_row["adv_mac_pct"] >= ADV_PCT_MIN and
    adv_row["duration"]   >= DUR_MIN
):
    print("\n>>> ADVERSARY CONFIRMED <<<")
else:
    print("\n>>> NOT CONFIRMED (criteria not met) <<<")