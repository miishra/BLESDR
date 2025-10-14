#!/usr/bin/env python3
import csv, numpy as np, matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

ASSIGN="mac_analysis_out/cluster/cluster_assignments.csv"
FEATS ="features_ch37.csv"
OUT   ="mac_analysis_out/plots"

Path(OUT).mkdir(parents=True, exist_ok=True)

# load assignments
rows=[]
with open(ASSIGN) as f:
    r=csv.DictReader(f)
    for q in r: rows.append(q)
labels=np.array([int(q["cluster"]) for q in rows])
macs  =np.array([q["mac"] for q in rows])

# load features to rebuild PCA (same columns as before)
with open(FEATS) as f:
    r=csv.DictReader(f)
    feats=list(r)
feat_cols=[k for k in feats[0].keys() if k not in {"file","access_address","ts","timestamp","time"}]
X=np.array([[float(q[k]) if q[k] not in ("","nan") else np.nan for k in feat_cols] for q in feats], float)

# align by index
n=min(len(rows), X.shape[0])
X=X[:n]; labels=labels[:n]; macs=macs[:n]
good=np.all(np.isfinite(X),axis=1)
X=X[good]; labels=labels[good]; macs=macs[good]

# PCA (SVD)
mu=X.mean(axis=0, keepdims=True)
U,S,Vt=np.linalg.svd(X-mu, full_matrices=False)
Z=(X-mu)@Vt[:2].T

# 1) PCA scatter color by cluster
plt.figure()
for c in sorted(set(labels)):
    m=(labels==c)
    plt.plot(Z[m,0], Z[m,1], ls="", marker="o", ms=3, label=f"cluster {c}")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA colored by cluster")
plt.grid(True, ls=":", lw=0.5); plt.legend()
plt.savefig(f"{OUT}/clusters_pca2_replot.png", dpi=180, bbox_inches="tight"); plt.close()

# 2) Cluster x MAC table (top MACs)
ct=defaultdict(Counter)
for m,c in zip(macs, labels): ct[c][m]+=1
top_macs = [m for m,_ in Counter(macs).most_common(12)]
data=np.array([[ct[c][m] for m in top_macs] for c in sorted(ct)])
plt.figure()
plt.imshow(data, aspect="auto")
plt.xticks(range(len(top_macs)), top_macs, rotation=90)
plt.yticks(range(len(sorted(ct))), [f"cluster {c}" for c in sorted(ct)])
plt.colorbar(label="count")
plt.title("Cluster × MAC counts (top 12 MACs)")
plt.tight_layout()
plt.savefig(f"{OUT}/cluster_mac_heatmap.png", dpi=180); plt.close()
print(f"Saved PCA and heatmap to {OUT}")
