#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirCatch.py — CFO-based adversary detection (key-aware, ecosystem-aware, per-MAC segmentation)

This version includes:
  1) CORE DENSITY (computed only on densest core, robust to outliers)
  2) IMPROVED MAC-CHURN METRICS (to capture highly changing MACs better than unique/segments)

Key churn metrics per cluster:
  - singleton_mac_frac: fraction of unique MACs that appear exactly once
  - singleton_seg_frac: fraction of segments that belong to singleton MACs
  - top_mac_share: share of segments belonging to top MAC (dominance)
  - turnover: mean Jaccard distance of MAC-sets between consecutive windows (seg_id)
  - n_eff_macs: effective number of MACs = exp(H), where H is Shannon entropy of MAC frequencies
  - mac_rate_per_min: unique MACs per minute of window coverage
  - n_eff_rate_per_min: effective MACs per minute

We define:
  churn_score = turnover if available else singleton_mac_frac

Candidate ranking uses:
  churn_score desc, duration_cov desc, core_mac_density_scaled desc

Strict decision uses:
  duration, core density
"""

import re
import hashlib
from collections import Counter
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  # kept (optional exploration)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import argparse
import multiprocessing as mp


# =========================
# Configuration
# =========================

ADV_CSV = "controlled/Car_Adv/scenarios_car__adv0_apple0_google0_samsung0_tile0__20260125_033106/background_only.csv"
# ADV_CSV = "car.csv"

# Fixed dataset folder (no prompt)
CONTROLLED_ROOT = "controlled"
CONTROLLED_SUBFOLDER = "Benign"  # <-- change this to another subfolder name under controlled/

PAYLOAD_TAG = "4c001219"        # optional, NOT used as filter by default
ADV_PAYLOAD_TAG = "4c001219ff"  # ADV marker used for GT and adv_mac_pct

WINDOW_S = 1800                  # seconds per time bucket (e.g., 300=5min, 120=2min, 900=15min)
K_RANGE = range(3, 20)
OVERALL_CFO_WEIGHT = 1

MIN_DURATION_S = 1500           # strict decision min support (seconds)

# --- key / type logic ---
KEY_SIM_THR = 0.99              # merge clusters if keys are extremely similar (same ecosystem)
TYPE_SEP_WEIGHT = 1.0           # strong separation between ecosystem types

# Density safety
S_MIN_DENSITY = 3               # compute density only if cluster has >=3 segments
R_MIN = 0.15                    # clamp radius in z-space for density
EPS = 1e-9

CORE_FRAC_Q = 0.2               # fraction of points to keep in core for density
CORE_MIN_PTS = 3                # min points in core for density
CORE_RADIUS_PCTL = 0.90         # robust clamp: use pctl of core distances as radius (reduces spikes)

CORE_DENSITY_VERSION = "v2_full_support_pctl_radius"

b210 = False                     # set True if B210 CRC filtering desired

CFO_COLS_RAW = ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"]
CFO_COLS_SEG = ["CFO", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]

# Final strict decision thresholds (ALL must pass)
DUR_MIN = MIN_DURATION_S
UNIQUE_MACS_MIN = 15

_HEX_DIGITS = "0123456789abcdefABCDEF"

# Density threshold applied on CORE density scaled
DENSITY_MIN = 0.9 #1.8

# Periodic mode (process each CSV in successive time blocks)
PERIODIC_MODE = True
PERIODIC_BLOCK_S = 1800   # 30 minutes
PERIODIC_STEP_S = 1800    # non-overlapping by default

# Periodic persistence confirmation (simple)
# Require the same cluster to persist long enough within the file (seconds).
# This uses the cluster's per-block persistence_s (t_end.max - t_start.min) and aggregates across blocks.
PERIODIC_MIN_PERSISTENCE_S = 1500


# =========================
# No-passer-by strict gates (persistence + real CFO density)
# =========================
# Keep these generic and defensible:
#  1) persistence: reject short passer-bys
#  2) temporal uniformity: reject bursty/non-stationary clusters that are not consistent over time

STRICT_MIN_PERSISTENCE_S = 1500

# Normalized window-entropy gate (1 = perfectly uniform, 0 = all mass in one window)
STRICT_WIN_ENTROPY_NORM_MIN = 0.60

def robust_stats(x: np.ndarray) -> dict:
    """Compute small, robust statistics for a 1D array."""
    if x is None or len(x) == 0:
        return {"median": 0.0, "iqr": 0.0, "p10": 0.0, "p90": 0.0, "mad": 0.0}

    x = np.asarray(x, dtype=float)
    med = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return {
        "median": float(med),
        "iqr": float(q3 - q1),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "mad": float(np.median(np.abs(x - med))),
    }

def singleton_mac_frac(macs: list) -> float:
    """Fraction of MACs that appear exactly once."""
    if not macs:
        return 0.0
    c = Counter(macs)
    return sum(1 for _, cnt in c.items() if int(cnt) == 1) / max(len(c), 1)


def _collect_csv_files(path: Path) -> list[Path]:
    """Collect CSV files from a file or directory (recursive)."""
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() == ".csv" else []
    if p.is_dir():
        return sorted([x for x in p.rglob("*.csv") if x.is_file()])
    return []


def _is_hex(s: str) -> bool:
    return bool(s) and (_HEX_RE.match(s) is not None)


def _clean_hex(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"[^0-9a-f]", "", t)
    if len(t) % 2 == 1:
        t = t[:-1]
    return t


def _hex_to_bytes(hexstr: str) -> bytes:
    try:
        return bytes.fromhex(hexstr)
    except Exception:
        return b""


def add_packet_support_feature(X: np.ndarray, seg: pd.DataFrame, weight: float = 3.0) -> np.ndarray:
    # Robust: log-compress; prevents a 97-packet MAC from dominating
    v = np.log1p(seg["n_packets"].astype(float).values).reshape(-1, 1)
    v = StandardScaler().fit_transform(v) * weight
    return np.hstack([X, v])


def add_cfo_robust_spread_features(X: np.ndarray, seg: pd.DataFrame, weight: float = 1.0) -> np.ndarray:
    """Append robust CFO-spread features for clustering.

    Uses robust_stats() to summarize how each segment deviates from the cluster's
    median CFO vector. This helps separate stable transmitters vs mixed/noisy
    segments.

    Per segment we compute:
      - l2_dev: L2 distance to median CFO vector
      - l1_dev: L1 distance to median CFO vector

    Then we add two *robust* scalars derived from these deviations (computed within
    the current segment table):
      - mad(l2_dev), iqr(l2_dev)

    Finally, we append standardized (z-scored) l2_dev and l1_dev per row plus the
    global robust stats (broadcast as constants) so clusters formed in different
    files/blocks remain comparable.
    """
    if seg is None or len(seg) == 0:
        return X

    cols = [c for c in CFO_COLS_SEG if c in seg.columns]
    if not cols:
        return X

    A = seg[cols].astype(float).values
    med = np.nanmedian(A, axis=0)
    dif = A - med[None, :]

    l2 = np.linalg.norm(dif, axis=1)
    l1 = np.sum(np.abs(dif), axis=1)

    rs = robust_stats(l2)
    mad_l2 = float(rs.get("mad", 0.0))
    iqr_l2 = float(rs.get("iqr", 0.0))

    # Per-row features (standardized)
    F_row = np.vstack([l2, l1]).T
    F_row = StandardScaler().fit_transform(F_row)

    # Broadcast robust scalars as additional features
    F_const = np.tile(np.array([[mad_l2, iqr_l2]], dtype=float), (len(seg), 1))
    F = np.hstack([F_row, F_const]) * float(weight)

    return np.hstack([X, F])


def classify_tag_ecosystem_from_payload(payload_hex: str) -> str:
    """
    Parse AD structures to classify ecosystem:
      - APPLE: Manufacturer Specific (0xFF), CompanyID=0x004C, prefix bytes 0x12 0x19
      - GOOGLE: Service Data 16-bit (0x16), UUID=0xFEAA
      - TILE:   Service Data 16-bit (0x16), UUID=0xFEED
      - SAMSUNG:Service Data 16-bit (0x16), UUID=0xFD5A
      - UNKNOWN otherwise

    We try two layouts:
      A) payload = AdvA(6) + AD...
      B) payload = Header(2) + AdvA(6) + AD...
    """
    hx = _clean_hex(payload_hex)
    if len(hx) < 16:
        return "UNKNOWN"

    b = _hex_to_bytes(hx)
    if len(b) < 8:
        return "UNKNOWN"

    def parse_ad_structures(ad_data: bytes) -> str:
        pos = 0
        ad_len = len(ad_data)

        while pos + 1 < ad_len:
            length = ad_data[pos]
            if length == 0:
                break
            if pos + 1 + length > ad_len:
                break

            ad_type = ad_data[pos + 1]
            data = ad_data[pos + 2: pos + 1 + length]  # excludes length byte

            # Manufacturer Specific (0xFF): [CompanyID LE (2)][...]
            if ad_type == 0xFF and len(data) >= 4:
                company_id = data[0] | (data[1] << 8)
                if company_id == 0x004C:
                    # FindMy prefix bytes: 0x12 0x19
                    if data[2] == 0x12 and data[3] == 0x19:
                        return "APPLE"

            # Service Data - 16-bit UUID (0x16): [UUID LE (2)][...]
            if ad_type == 0x16 and len(data) >= 2:
                svc_uuid = data[0] | (data[1] << 8)
                if svc_uuid == 0xFEAA:
                    return "GOOGLE"
                if svc_uuid == 0xFEED:
                    return "TILE"
                if svc_uuid == 0xFD5A:
                    return "SAMSUNG"

            pos += 1 + length

        return "UNKNOWN"

    # Layout A: AdvA(6) + AD...
    if len(b) >= 7:
        eco = parse_ad_structures(b[6:])
        if eco != "UNKNOWN":
            return eco

    # Layout B: Header(2) + AdvA(6) + AD...
    if len(b) >= 9:
        eco = parse_ad_structures(b[8:])
        if eco != "UNKNOWN":
            return eco

    return "UNKNOWN"


# =========================
# Key handling
# =========================

def normalize_key_str(x):
    """Normalize a key-like field into a list of hex strings."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (bytes, bytearray)):
        x = x.hex()
    s = str(x).strip()

    keys = re.findall(r"k:([0-9a-fA-F]+)", s)
    if keys:
        return [k.lower() for k in keys if _is_hex(k)]

    parts = re.split(r"[,\s;|]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if p.startswith("0x"):
            p = p[2:]
        if len(p) >= 16 and _is_hex(p):
            out.append(p.lower())
    return out


def extract_pubkey_from_payload(payload: str):
    """Key = first half of the cleaned payload hex."""
    hx = _clean_hex(payload)
    if not hx:
        return None
    half = len(hx) // 2
    if half < 16:
        return None
    return hx[:half]


def key_char_similarity(a: str, b: str) -> float:
    """Fraction of matching hex characters over min length."""
    if not a or not b:
        return 0.0
    a = a.lower()
    b = b.lower()
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / float(n)


def keys_max_similarity(keys_a, keys_b) -> float:
    """Max char-similarity between any key in A and any key in B."""
    if not keys_a or not keys_b:
        return 0.0
    best = 0.0
    for ka in keys_a:
        for kb in keys_b:
            sim = key_char_similarity(ka, kb)
            if sim > best:
                best = sim
                if best >= 0.999:
                    return best
    return best


# =========================
# Union-Find for cluster merging
# =========================

class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# =========================
# Segmentation + features
# =========================

def collect_pubkeys(df: pd.DataFrame) -> pd.Series:
    """
    Series of list-of-keys per row. Priority:
      1) explicit key-like columns if present
      2) payload extraction first-half
    """
    key_cols = ["public_key", "pubkey", "pub_key", "key", "keys", "Keys", "PublicKey", "Public_Key"]
    existing = [c for c in key_cols if c in df.columns]

    if existing:
        def row_keys(r):
            out = []
            for c in existing:
                out.extend(normalize_key_str(r.get(c)))
            if not out:
                k = extract_pubkey_from_payload(r.get("payload"))
                if k:
                    out = [k]
            # dedup
            seen, uniq = set(), []
            for k in out:
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(k)
            return uniq
        return df.apply(row_keys, axis=1)

    def payload_keys(payload):
        k = extract_pubkey_from_payload(payload)
        return [k] if k else []

    return df["payload"].apply(payload_keys)

# ---------------------------
# Hex helpers + Samsung PRIVID extraction
# ---------------------------

def _clean_hexpayload(payload_hex: str) -> str:
    if payload_hex is None:
        return ""
    s = str(payload_hex).strip()
    if not s:
        return ""
    s = s.replace(" ", "").replace(":", "").replace("-", "").replace("\n", "").replace("\r", "").replace("\t", "")
    s = "".join(ch for ch in s if ch in _HEX_DIGITS)
    if len(s) < 2:
        return ""
    if len(s) % 2 == 1:
        s = s[:-1]
    return s.lower()


def _hex_to_bytes(hx: str) -> bytes:
    try:
        return bytes.fromhex(hx)
    except Exception:
        return b""

def get_samsung_privid_from_payload(payload_hex: str) -> str:
    """
    Extract Samsung PRIVID from payload if present.
    PRIVID lives inside a Service Data (AD type 0x16) with UUID=0xFD5A.
    PRIVID is 8 bytes at data[4:12] where data[0:2] is UUID.
    Returns PRIVID hex (lowercase) or "".
    """
    hx = _clean_hexpayload(payload_hex)
    if not hx:
        return ""
    b = _hex_to_bytes(hx)
    if not b:
        return ""

    def extract_from_ad(ad_data: bytes) -> str:
        pos = 0
        n = len(ad_data)
        while pos + 1 < n:
            length = ad_data[pos]
            if length == 0:
                break
            if pos + 1 + length > n:
                break
            ad_type = ad_data[pos + 1]
            data = ad_data[pos + 2: pos + 1 + length]
            if ad_type == 0x16 and len(data) >= 12:
                svc_uuid = data[0] | (data[1] << 8)
                if svc_uuid == 0xFD5A:
                    return data[4:12].hex()
            pos += 1 + length
        return ""

    # Layout A: AdvA(6) + AD...
    if len(b) >= 7:
        v = extract_from_ad(b[6:])
        if v:
            return v
    # Layout B: Header(2) + AdvA(6) + AD...
    if len(b) >= 9:
        v = extract_from_ad(b[8:])
        if v:
            return v
    return ""


def prepare_segments(df: pd.DataFrame, window_s: int) -> tuple[pd.DataFrame, set[str]]:
    """
    PER-DEVICE segmentation (mostly per-MAC, but Samsung uses PRIVID):

      - For non-Samsung: seg_key = (seg_id, eco, AdvA)  [per-MAC]
      - For Samsung:     seg_key = (seg_id, eco, PRIVID) [per-PRIVID]

    Returns:
      seg: segment dataframe
      adv_mac_set: ground-truth set of MACs that appear in packets whose payload contains ADV_PAYLOAD_TAG
    """
    df = df.copy()

    # normalize / required columns
    df["payload"] = df["payload"].astype(str).str.lower()
    df["AdvA"] = df["AdvA"].astype(str)

    if not b210 and "crc_ok" in df.columns:
        df = df[df["crc_ok"] == 1]  # filter valid CRC only
        # print("\n[DEBUG] Filtered CRC-OK packets; remaining:", len(df))

    df = df.dropna(subset=["timestamp", "AdvA"] + CFO_COLS_RAW)
    df = df.sort_values("timestamp")

    # Ground-truth ADV MAC set (packet-level marker)
    adv_mask = df["payload"].str.contains(ADV_PAYLOAD_TAG, na=False)
    adv_mac_set = set(df.loc[adv_mask, "AdvA"].astype(str).tolist())

    # Drop packets that contain "aafe40" in payload (Connected state of GOOGLE)
    google_mask = df["payload"].str.contains("aafe40", na=False)
    if google_mask.any():
        df = df[~google_mask]
        print("\n[DEBUG] Dropped GOOGLE connected-state packets; remaining:", len(df))        

    # Ecosystem per packet
    df["eco"] = df["payload"].apply(classify_tag_ecosystem_from_payload)
    df["dev_type"] = df["eco"].apply(lambda e: f"TAG_{e}")

    # --- NEW: Samsung PRIVID per packet (empty string if not found) ---
    # Requires you to have get_samsung_privid_from_payload(payload_hex: str) -> str defined.
    df["privid"] = ""
    samsung_mask = (df["eco"] == "SAMSUNG")
    if samsung_mask.any():
        df.loc[samsung_mask, "privid"] = df.loc[samsung_mask, "payload"].apply(get_samsung_privid_from_payload)

        # If PRIVID missing, fall back to AdvA so we don't drop those packets
        missing_priv = samsung_mask & (df["privid"].astype(str) == "")
        if missing_priv.any():
            df.loc[missing_priv, "privid"] = df.loc[missing_priv, "AdvA"].astype(str)

    # --- NEW: per-packet device identifier used for segmentation ---
    # Non-Samsung -> AdvA; Samsung -> PRIVID (or fallback AdvA)
    df["dev_id"] = df["AdvA"].astype(str)
    df.loc[samsung_mask, "dev_id"] = df.loc[samsung_mask, "privid"].astype(str)

    # Keys per packet
    df["pubkeys"] = collect_pubkeys(df)

    # Window index
    df["seg_id"] = (df["timestamp"] // window_s).astype(int)

    def agg_segment(g: pd.DataFrame, seg_id: int, eco: str, dev_id: str) -> pd.Series:
        t_start = float(g["timestamp"].min())
        t_end = float(g["timestamp"].max())
        n_packets = int(g["AdvA"].count())

        dev_id = str(dev_id)

        # In this mode, each segment corresponds to exactly one device-id (MAC or PRIVID)
        mac_set = [dev_id]
        n_macs = 1

        keys_flat = []
        for lst in g["pubkeys"].tolist():
            if isinstance(lst, list):
                keys_flat.extend(lst)
        key_set = sorted(set([k for k in keys_flat if k]))

        adv_macs = int(g["payload"].str.contains(ADV_PAYLOAD_TAG, na=False).sum())
        # NOTE: gt_adv remains MAC-based ground truth; for Samsung using PRIVID, this will almost always be False.
        # If you want gt_adv for Samsung/PRIVID too, you should add a separate ground-truth set keyed by PRIVID.
        gt_adv = (g["AdvA"].astype(str).isin(adv_mac_set).any())

        # CFO mean + robust stats
        cfo_stats = {}
        for col_raw, col_base in zip(
            ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"],
            ["CFO", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]
        ):
            vals = g[col_raw].astype(float).values
            cfo_stats[col_base] = float(np.mean(vals))

        return pd.Series({
            # Use dev_id in the seg_key; for Samsung this is PRIVID
            "seg_key": f"{int(seg_id)}:{eco}:{dev_id}",
            "seg_id": int(seg_id),
            "eco": eco,
            "dev_type": f"TAG_{eco}",

            # Keep existing "mac" column for compatibility, but it now holds MAC or PRIVID.
            # If you want both, you can also store "advA" + "privid".
            "mac": dev_id,

            # Optional: expose original fields for debugging / downstream rules
            "advA": str(g["AdvA"].iloc[0]) if "AdvA" in g.columns and len(g) else "",
            "privid": str(g["privid"].iloc[0]) if eco == "SAMSUNG" and "privid" in g.columns and len(g) else "",

            "t_start": t_start,
            "t_end": t_end,
            "duration_est": float(window_s),
            "duration_obs": float(t_end - t_start),

            "n_packets": n_packets,
            "n_macs": n_macs,
            "mac_set": mac_set,
            "key_set": key_set,

            **cfo_stats,

            "adv_macs": adv_macs,
            "gt_adv": bool(gt_adv),
        })

    rows = []
    # --- UPDATED grouping: use dev_id instead of AdvA ---
    for (sid, eco, dev_id), g in df.groupby(["seg_id", "eco", "dev_id"], sort=True):
        rows.append(agg_segment(g, sid, eco, dev_id))

    # #filter out segment which are greater than 900 seconds
    # rows = [r for r in rows if r["duration_obs"] < 900]

    seg = pd.DataFrame(rows)
    if len(seg) == 0:
        raise RuntimeError("No segments produced. Check input CSV columns and filters.")

    # Per-device segments => churn not meaningful; keep only for debugging.
    seg["churn"] = seg["n_macs"] / seg["n_packets"].clip(lower=1)
    return seg, adv_mac_set


def cfo_feature_matrix(seg: pd.DataFrame) -> np.ndarray:
    X = StandardScaler().fit_transform(seg[CFO_COLS_SEG].values)
    X[:, 0] *= OVERALL_CFO_WEIGHT
    return X


def add_type_feature(X: np.ndarray, seg: pd.DataFrame, weight: float = TYPE_SEP_WEIGHT) -> np.ndarray:
    types = seg["dev_type"].astype(str).fillna("UNKNOWN").values
    uniq = sorted(set(types))
    mapping = {t: i for i, t in enumerate(uniq)}
    t_id = np.array([mapping[t] for t in types], dtype=float).reshape(-1, 1)
    t_id = StandardScaler().fit_transform(t_id) * weight
    return np.hstack([X, t_id])


def merge_clusters_by_key_similarity(seg: pd.DataFrame, labels: np.ndarray, key_sim_thr: float = KEY_SIM_THR) -> np.ndarray:
    """
    Post-clustering merge:
      - only merges clusters of the SAME dev_type
      - merges if keys are highly similar (>= key_sim_thr)
    """
    seg2 = seg.copy()
    seg2["cluster"] = labels
    clusters = sorted(seg2["cluster"].unique().tolist())

    cluster_keys = {}
    cluster_type = {}
    for cid, g in seg2.groupby("cluster"):
        keys = []
        for ks in g["key_set"].tolist():
            if isinstance(ks, list):
                keys.extend(ks)
        cluster_keys[cid] = sorted(set([k for k in keys if k]))
        tm = g["dev_type"].astype(str).mode()
        cluster_type[cid] = str(tm.iloc[0]) if len(tm) else "UNKNOWN"

    uf = UnionFind(clusters)

    for i in range(len(clusters)):
        a = clusters[i]
        for j in range(i + 1, len(clusters)):
            b = clusters[j]
            if cluster_type[a] != cluster_type[b]:
                continue
            ka = cluster_keys.get(a, [])
            kb = cluster_keys.get(b, [])
            if not ka or not kb:
                continue
            if keys_max_similarity(ka, kb) >= key_sim_thr:
                uf.union(a, b)

    root_to_new = {}
    new_labels = np.zeros_like(labels)
    nxt = 0
    for idx, old in enumerate(labels):
        r = uf.find(int(old))
        if r not in root_to_new:
            root_to_new[r] = nxt
            nxt += 1
        new_labels[idx] = root_to_new[r]
    return new_labels


# =========================
# Cluster stats (GLOBAL geometry + CORE density + NEW churn)
# =========================

def compute_cluster_geometry(X: np.ndarray, seg: pd.DataFrame) -> pd.DataFrame:
    """Global geometry (kept for reference): radius based on dispersion from centroid."""
    out = []
    for cid, idxs in seg.groupby("cluster").groups.items():
        idxs = list(idxs)
        Xc = X[idxs, :]
        centroid = Xc.mean(axis=0)
        d = np.linalg.norm(Xc - centroid, axis=1)
        out.append({
            "cluster": int(cid),
            "radius": float(np.std(d)) if len(d) > 1 else 0.0,
            "mean_dist": float(np.mean(d)) if len(d) else 0.0,
        })
    return pd.DataFrame(out)


def compute_core_density(X_space: np.ndarray, seg_with_cluster: pd.DataFrame,
                         q: float = CORE_FRAC_Q, min_core: int = CORE_MIN_PTS) -> pd.DataFrame:
    """Compute core density around densest region.

    v2 changes:
      - MAC diversity uses FULL cluster support (unique_macs / total_segments)
        instead of (core_unique_macs / core_segments), avoiding small-core inflation.
      - core_radius uses a robust percentile of distances within the chosen core.

    Safety:
      - Prevent trivial single-MAC / tiny-support clusters from producing inflated density.
    """
    rows = []
    for cid, idxs in seg_with_cluster.groupby("cluster").groups.items():
        idxs = list(idxs)
        Xc = X_space[idxs, :]
        n = len(idxs)
        if n == 0:
            continue

        # pairwise distances (n x n)
        D = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
        medoid_local = int(np.argmin(D.sum(axis=1)))
        d = D[medoid_local, :]

        core_k = max(int(min_core), int(np.ceil(q * n)))
        core_k = min(core_k, n)
        core_local = np.argsort(d)[:core_k]
        core_global = [idxs[i] for i in core_local]

        # robust radius: percentile over core distances (less sensitive than max)
        if core_k > 1:
            try:
                core_radius = float(np.percentile(d[core_local], CORE_RADIUS_PCTL * 100.0))
            except Exception:
                core_radius = float(np.max(d[core_local]))
        else:
            core_radius = 0.0

        core_radius_clamped = max(float(core_radius), float(R_MIN))

        # FULL-cluster MAC diversity
        all_macs = seg_with_cluster.loc[idxs, "mac"].astype(str).tolist()
        all_macs = [m for m in all_macs if m]
        unique_macs = len(set(all_macs))

        # For reporting only
        core_macs = seg_with_cluster.loc[core_global, "mac"].astype(str).tolist()
        core_macs = [m for m in core_macs if m]
        core_unique_macs = len(set(core_macs))

        total_segments = int(n)
        core_segments = int(core_k)

        # Base density
        mac_div_full = unique_macs / max(total_segments, 1)
        core_mac_density = mac_div_full / (core_radius_clamped + EPS)
        core_mac_density_scaled = core_mac_density

        # Safety clamp: trivial clusters (single MAC or insufficient support) should not look "dense"
        if unique_macs < 2 or total_segments < int(min_core):
            core_mac_density = 0.0
            core_mac_density_scaled = 0.0

        rows.append({
            "cluster": int(cid),
            "core_segments": int(core_segments),
            "core_unique_macs": int(core_unique_macs),
            "core_mac_div": float(mac_div_full),
            "core_radius": float(core_radius),
            "core_radius_clamped": float(core_radius_clamped),
            "core_mac_density": float(core_mac_density),
            "core_mac_density_scaled": float(core_mac_density_scaled),
            "core_density_version": CORE_DENSITY_VERSION,
        })

    return pd.DataFrame(rows)


def is_homogeneous_cluster(g: pd.DataFrame) -> dict:
    # removed homogeneity logic; keep stub for compatibility with any downstream merges
    return {
        "homogeneous": False,
        "homo_top_mac": "",
        "homo_mac_share": 0.0,
        "homo_top_key": "",
        "homo_key_share": 0.0,
        "duration_cov": float(g["seg_id"].nunique() * WINDOW_S) if (g is not None and "seg_id" in g.columns) else float("nan"),
        "duration_span": float("nan"),
        "unique_windows": int(g["seg_id"].nunique()) if (g is not None and "seg_id" in g.columns) else int(len(g)) if g is not None else 0,
    }


def summarize_clusters(seg: pd.DataFrame, X_for_geom: np.ndarray, X_density_space: np.ndarray) -> pd.DataFrame:
    """
    X_for_geom: full feature space used for clustering (optional for global geometry)
    X_density_space: CFO-only (recommended) space for core density computation
    """
    geom = compute_cluster_geometry(X_for_geom, seg)

    rows = []
    for cid, g in seg.groupby("cluster"):
        dt_mode = g["dev_type"].astype(str).mode()
        dev_type = str(dt_mode.iloc[0]) if len(dt_mode) else "UNKNOWN"

        # Window coverage (how many buckets the cluster appears in)
        unique_windows = int(g["seg_id"].nunique()) if "seg_id" in g.columns else len(g)
        duration_cov = float(unique_windows * WINDOW_S)

        # True persistence: time span between first and last packet in the cluster
        # (segment-level t_start/t_end already reflect per-window packet min/max)
        if "t_start" in g.columns and "t_end" in g.columns:
            persistence_s = float(g["t_end"].max() - g["t_start"].min())
        else:
            persistence_s = float("nan")

        # Keep previous name for compatibility but prefer persistence_s for decisions
        duration_span = persistence_s if np.isfinite(persistence_s) else duration_cov

        # NEW: temporal activity uniformity across covered windows
        win_counts = g.groupby("seg_id").size().astype(float) if "seg_id" in g.columns else pd.Series([float(len(g))])
        if len(win_counts) > 1 and win_counts.sum() > 0:
            pwin = (win_counts / win_counts.sum()).values
            Hwin = float(-(pwin * np.log(pwin + 1e-12)).sum())
            Hwin_norm = Hwin / float(np.log(len(pwin)))
        else:
            Hwin_norm = 1.0

        gt_any = bool(g["gt_adv"].any()) if "gt_adv" in g.columns else False
        gt_frac = float(g["gt_adv"].mean()) if "gt_adv" in g.columns else 0.0

        # ---- MAC distribution in this cluster ----
        macs = [m for m in g["mac"].astype(str).tolist() if m]
        c = Counter(macs)
        seg_count = len(macs)

        unique_macs = len(c)
        top_mac_share = (c.most_common(1)[0][1] / seg_count) if seg_count else 0.0

        # Old "mac_diversity" kept for reference (bounded, can dilute)
        mac_diversity = (unique_macs / max(seg_count, 1))

        # Singleton MAC fractions
        singleton_mac_frac = (sum(1 for _, cnt in c.items() if int(cnt) == 1) / max(unique_macs, 1)) if unique_macs else 0.0
        singleton_seg_frac = (sum(cnt for _, cnt in c.items() if int(cnt) == 1) / max(seg_count, 1)) if seg_count else 0.0

        # Entropy -> effective MAC count
        if seg_count > 0:
            p = np.array([cnt / seg_count for cnt in c.values()], dtype=float)
            H = float(-(p * np.log(p + 1e-12)).sum())
            n_eff = float(np.exp(H))
        else:
            H = 0.0
            n_eff = 0.0

        # MAC rates per minute
        dur_min = max(duration_cov / 60.0, 1e-9)
        mac_rate_per_min = float(unique_macs / dur_min)
        n_eff_rate_per_min = float(n_eff / dur_min)

        total_packets = int(g["n_packets"].sum())
        n_packets_avg = float(g["n_packets"].mean())
        pkt_per_mac = float(total_packets / max(unique_macs, 1))
        singleton_pkt_frac = float((g["n_packets"] <= 1).mean())

        row = {
            "cluster": int(cid),
            "dev_type": dev_type,
            "segments": int(len(g)),

            "unique_windows": unique_windows,
            "duration_cov": duration_cov,
            "persistence_s": float(persistence_s) if np.isfinite(persistence_s) else np.nan,
            "duration_span": float(duration_span),
            "win_entropy_norm": float(Hwin_norm),

            "unique_macs": int(unique_macs),

            # old metric (kept)
            "mac_diversity": float(mac_diversity),

            "adv_mac_pct": float(g["adv_macs"].sum() / max(g["n_packets"].sum(), 1)),
            "n_packets_avg": n_packets_avg,
            "total_packets": total_packets,

            "gt_any": gt_any,
            "gt_frac": gt_frac,
        }

        row.update(is_homogeneous_cluster(g))
        rows.append(row)

    df = pd.DataFrame(rows).merge(geom, on="cluster", how="left")

    # Global density (optional / diagnostic)
    df["radius_clamped"] = df["radius"].fillna(0.0)
    mask = df["segments"] >= S_MIN_DENSITY
    df.loc[mask, "radius_clamped"] = df.loc[mask, "radius_clamped"].clip(lower=R_MIN)

    df["mac_density"] = np.nan
    df.loc[mask, "mac_density"] = df.loc[mask, "mac_diversity"] / (df.loc[mask, "radius_clamped"] + EPS)

    df["mac_density_scaled"] = np.nan
    df.loc[mask, "mac_density_scaled"] = (
        df.loc[mask, "mac_density"] #* np.log2(1.0 + df.loc[mask, "unique_macs"])
    )

    # CORE density (the real discriminator)
    core_df = compute_core_density(X_density_space, seg, q=CORE_FRAC_Q, min_core=CORE_MIN_PTS)
    df = df.merge(core_df, on="cluster", how="left")

    if "core_density_version" not in df.columns:
        df["core_density_version"] = CORE_DENSITY_VERSION

    return df


def rank_adversary_clusters(summary_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank candidate clusters.

    Cleaned: rank by core density, but drop non-uniform (bursty) clusters.
    """
    df = summary_df.copy()
    if len(df) == 0:
        return df

    # Uniformity filter: keep only clusters that are reasonably uniform over time
    if "win_entropy_norm" in df.columns:
        df["uniform_ok"] = df["win_entropy_norm"].fillna(1.0) >= float(STRICT_WIN_ENTROPY_NORM_MIN)
        df = df[df["uniform_ok"]].copy()
    else:
        df["uniform_ok"] = True

    if len(df) == 0:
        return df

    df["dens_rank"] = df.get("core_mac_density_scaled", np.nan).fillna(-1e18)
    df = df.sort_values(by=["dens_rank"], ascending=[False])
    return df.head(top_n)


def _strict_decision(row: pd.Series) -> tuple[bool, dict]:
    """Return (ok, components) for strict confirmation.

    Paper-friendly rule:
      confirm := (persistence >= STRICT_MIN_PERSISTENCE_S) AND
                 (core density >= DENSITY_MIN) AND
                 (win_entropy_norm >= STRICT_WIN_ENTROPY_NORM_MIN)

    Notes:
      - persistence rejects transient passer-bys.
      - density ensures a tight CFO-based cluster.
      - uniformity rejects bursty / on-off clusters that can have high density but are not stable.
    """
    dur_val = row.get("persistence_s", np.nan)
    if dur_val is None or (isinstance(dur_val, float) and np.isnan(dur_val)):
        dur_val = row.get("duration_span", np.nan)
    if dur_val is None or (isinstance(dur_val, float) and np.isnan(dur_val)):
        dur_val = row.get("duration_cov", 0.0)

    dur_ok = bool(float(dur_val) >= float(STRICT_MIN_PERSISTENCE_S))

    dens = float(row.get("core_mac_density_scaled", np.nan))
    dens_ok = bool(np.isfinite(dens) and (dens >= float(DENSITY_MIN))) if DENSITY_MIN is not None else True

    # Never allow a 1-MAC (or 1-segment) cluster to pass density gating
    if int(row.get("unique_macs", 0) or 0) < 2 or int(row.get("segments", 0) or 0) < int(CORE_MIN_PTS):
        dens_ok = False

    wen = float(row.get("win_entropy_norm", 1.0))
    uniform_ok = bool(wen >= float(STRICT_WIN_ENTROPY_NORM_MIN))

    ok = bool(dur_ok and dens_ok)

    return ok, {
        "dur_ok": dur_ok,
        "unique_ok": bool(float(row.get("unique_macs", 0.0)) >= UNIQUE_MACS_MIN),
        "dens_ok": dens_ok,
        "decision_ok": ok,
    }


def _list_csvs_in_controlled_subfolder(root: str = CONTROLLED_ROOT, subfolder: str = CONTROLLED_SUBFOLDER) -> list[Path]:
    base = Path(root) / subfolder
    return _collect_csv_files(base)


def _fit_agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    """Fit agglomerative clustering with a stable configuration and return labels."""
    k = int(max(2, k))
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(X)


def _choose_k_by_silhouette(X: np.ndarray, k_candidates: list[int]) -> int:
    """Pick k that maximizes silhouette_score over candidate k values.

    Falls back to a reasonable k if silhouette cannot be computed.
    """
    n = int(X.shape[0])
    if n < 3:
        return 3

    # sanitize and bound by sample count
    ks = sorted({int(k) for k in k_candidates if int(k) >= 3})
    ks = [k for k in ks if k < n]  # silhouette needs at least 2 clusters and k < n
    if not ks:
        return max(3, min(10, n - 1))

    best_k = ks[0]
    best_score = -1e18

    for k in ks:
        try:
            labels = _fit_agglomerative(X, k)
            # Need at least 3 clusters and not all singleton
            if len(set(labels.tolist())) < 3:
                continue
            sc = float(silhouette_score(X, labels, metric="euclidean"))
            if sc > best_score:
                best_score = sc
                best_k = k
        except Exception:
            continue

    return int(best_k)


def _iter_time_blocks(df: pd.DataFrame, t_col: str = "timestamp"):
    """Yield (block_start, block_end, df_block)."""
    if df is None or len(df) == 0 or t_col not in df.columns:
        return
    t = pd.to_numeric(df[t_col], errors="coerce")
    if t.isna().all():
        return
    tmin = float(np.nanmin(t.values))
    tmax = float(np.nanmax(t.values))
    start = tmin
    while start <= tmax:
        end = start + float(PERIODIC_BLOCK_S)
        m = (t >= start) & (t < end)
        dfb = df.loc[m].copy()
        yield float(start), float(end), dfb
        start += float(PERIODIC_STEP_S)


def _pca_outdir_for_label(label: str) -> Path:
    """Folder to save PCA plots."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))
    d = Path(f"{safe}PCA")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_pca3d_plot(seg_t: pd.DataFrame, X_t: np.ndarray, src_file: str, dev_type: str, out_png: str) -> None:
    """Save a 3D PCA plot for a per-dev_type segment table."""
    try:
        if seg_t is None or len(seg_t) < 3:
            return
        if X_t is None or getattr(X_t, "shape", (0,))[0] < 3:
            return

        X = np.asarray(X_t, dtype=float)
        pca = PCA(n_components=3, random_state=0)
        Z = pca.fit_transform(X)

        labels = seg_t.get("cluster", pd.Series([-1] * len(seg_t))).astype(int).values

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        uniq = np.unique(labels)
        cmap = plt.get_cmap("tab20")
        for i, lab in enumerate(uniq):
            m = labels == lab
            ax.scatter(
                Z[m, 0], Z[m, 1], Z[m, 2],
                s=18,
                alpha=0.85,
                color=cmap(int(i) % 20),
                label=f"c{int(lab)}" if lab >= 0 else "noise",
            )

        ax.set_title(
            f"3D PCA: {Path(src_file).name} | {dev_type}\n"
            f"explained var: {pca.explained_variance_ratio_.sum():.2f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend(loc="best", fontsize=8)

        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
    except Exception:
        # Plotting must never break the pipeline
        try:
            plt.close("all")
        except Exception:
            pass


def _run_one_csv_once(csv_path: Path, adv: pd.DataFrame, *, block_start: Optional[float] = None, block_end: Optional[float] = None):
    """Core per-file (or per-block) run that assumes adv data already loaded/sliced."""
    adv = adv.copy()
    adv["_src_file"] = str(csv_path)

    seg, adv_mac_set = prepare_segments(adv, WINDOW_S)

    # --- NEW: per-dev_type clustering ---
    all_checks_rows = []
    all_summary_parts = []
    confirmed_any = False
    n_clusters_total = 0

    # base name for PCA plot output (one per dev_type, per block if periodic)
    block_tag = ""
    if block_start is not None and block_end is not None:
        block_tag = f"__b{int(block_start)}_{int(block_end)}"

    # Use subfolder-specific output dir for PCA
    pca_dir = _pca_outdir_for_label(CONTROLLED_SUBFOLDER)

    for dev_type, seg_t in seg.groupby("dev_type", sort=True):
        seg_t = seg_t.copy().reset_index(drop=True)
        if len(seg_t) < 3:
            continue

        X_cfo_t = cfo_feature_matrix(seg_t)
        X_t = add_packet_support_feature(X_cfo_t.copy(), seg_t, weight=3.0)
        X_t = add_cfo_robust_spread_features(X_t, seg_t, weight=1.0)

        # Defensive: ensure row alignment
        n = int(len(seg_t))
        if getattr(X_t, "shape", (0,))[0] != n:
            m = min(int(getattr(X_t, "shape", (0,))[0]), n)
            seg_t = seg_t.iloc[:m].copy().reset_index(drop=True)
            X_cfo_t = X_cfo_t[:m]
            X_t = X_t[:m]
            n = m
        if n < 3:
            continue

        n = int(X_t.shape[0])
        k_grid = list(range(2, min(16, n)))
        k_best = _choose_k_by_silhouette(X_t, k_grid)
        labels0 = _fit_agglomerative(X_t, int(k_best))
        seg_t["cluster"] = labels0

        # Save PCA plot per CSV/dev_type (and per block if periodic)
        try:
            safe_src = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(csv_path).stem)[:80]
            safe_type = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dev_type))[:40]
            pca_png = pca_dir / f"aircatch_pca3d__{safe_src}__{safe_type}{block_tag}.png"
            _save_pca3d_plot(seg_t, X_t, str(csv_path), str(dev_type), str(pca_png))
        except Exception:
            pass

        summary_t = summarize_clusters(seg_t, X_for_geom=X_t, X_density_space=X_cfo_t)

        summary_t["src_file"] = str(csv_path)
        summary_t["k_chosen"] = int(k_best)
        summary_t["dev_type"] = str(dev_type)

        # Ensure cluster ids are unique across dev_types for this file
        summary_t = summary_t.copy()
        summary_t["cluster"] = summary_t["cluster"].astype(int) + int(n_clusters_total)
        seg_t["cluster"] = seg_t["cluster"].astype(int) + int(n_clusters_total)

        n_clusters_total += int(seg_t["cluster"].nunique())

        ranked_t = rank_adversary_clusters(summary_t, top_n=10)

        if len(ranked_t) > 0:
            for _, row in ranked_t.iterrows():
                ok, comps = _strict_decision(row)
                confirmed_any = confirmed_any or bool(ok)

                all_checks_rows.append({
                    "src_file": str(csv_path),
                    "cluster": int(row.get("cluster", -1)),
                    "dev_type": str(dev_type),
                    "segments": int(row.get("segments", 0)),
                    "duration_cov": float(row.get("duration_cov", np.nan)),
                    "duration_span": float(row.get("duration_span", np.nan)),
                    "persistence_s": float(row.get("persistence_s", np.nan)),
                    "unique_macs": int(row.get("unique_macs", 0)),
                    "core_mac_density_scaled": float(row.get("core_mac_density_scaled", np.nan)),
                    "adv_mac_pct": float(row.get("adv_mac_pct", np.nan)),
                    "gt_frac": float(row.get("gt_frac", np.nan)),
                    "dur_ok": bool(comps.get("dur_ok", False)),
                    "unique_ok": bool(comps.get("unique_ok", False)),
                    "dens_ok": bool(comps.get("dens_ok", False)),
                    "uniform_ok": bool(comps.get("uniform_ok", False)),
                    "decision_ok": bool(ok),
                    "k_chosen": int(k_best),
                })

        all_summary_parts.append(summary_t)

    summary_df = pd.concat(all_summary_parts, ignore_index=True) if all_summary_parts else pd.DataFrame()

    meta = {
        "src_file": str(csv_path),
        "n_segments": int(len(seg)),
        "gt_adv_mac_count": int(len(adv_mac_set)),
        "n_clusters": int(n_clusters_total),
        "n_ranked": int(len(all_checks_rows)),
        # block/file strict decision flag
        "strict_confirmed": bool(confirmed_any),
        "confirmed_any": bool(confirmed_any),
        "density_min": float(DENSITY_MIN) if DENSITY_MIN is not None else np.nan,
    }

    # include block bounds in meta when periodic
    if block_start is not None and block_end is not None:
        meta["block_start"] = float(block_start)
        meta["block_end"] = float(block_end)

    return pd.DataFrame(all_checks_rows), summary_df, meta


def _run_one_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    adv = pd.read_csv(csv_path)

    # Always run through the same code path so we can reuse strict decision flagging.
    if not PERIODIC_MODE:
        cdf, sdf, m = _run_one_csv_once(csv_path, adv, block_start=None, block_end=None)
        meta = m.copy()
        meta["periodic"] = False
        meta["confirmed_any"] = bool(m.get("strict_confirmed", False))
        return cdf, sdf, meta

    all_checks = []
    all_summaries = []
    all_meta = []

    strict_blocks = []

    for b0, b1, adv_b in _iter_time_blocks(adv, t_col="timestamp"):
        if len(adv_b) == 0:
            continue
        cdf, sdf, m = _run_one_csv_once(csv_path, adv_b, block_start=b0, block_end=b1)
        if cdf is not None and len(cdf) > 0:
            all_checks.append(cdf)
        if sdf is not None and len(sdf) > 0:
            sdf = sdf.copy()
            sdf["block_start"] = float(b0)
            sdf["block_end"] = float(b1)
            all_summaries.append(sdf)
        all_meta.append(m)

        strict_blocks.append(bool(m.get("strict_confirmed", False)))

    check_df = pd.concat(all_checks, ignore_index=True) if all_checks else pd.DataFrame()
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    # Final decision: strict-only (no periodic aggregation)
    strict_any = bool(any(strict_blocks))

    meta = {
        "src_file": str(csv_path),
        "periodic": True,
        "block_s": float(PERIODIC_BLOCK_S),
        "step_s": float(PERIODIC_STEP_S),
        "n_blocks": int(len(all_meta)),
        "strict_any_block": bool(strict_any),
        "confirmed_any": bool(strict_any),
    }
    meta["blocks"] = all_meta

    return check_df, summary_df, meta


def _write_core_density_plot(all_candidates: pd.DataFrame, out_png: str) -> None:
    """Save a simple distribution plot of core_mac_density_scaled across scenarios."""
    if all_candidates is None or len(all_candidates) == 0:
        return

    d = all_candidates.copy()
    d = d[np.isfinite(d["core_mac_density_scaled"])].copy()
    if len(d) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.hist(d["core_mac_density_scaled"].values, bins=40, alpha=0.8, color="steelblue")
    if DENSITY_MIN is not None:
        plt.axvline(float(DENSITY_MIN), color="red", linestyle="--", linewidth=2, label=f"DENSITY_MIN={DENSITY_MIN}")
        plt.legend(loc="best")
    plt.title("core_mac_density_scaled distribution (ranked candidates across scenarios)")
    plt.xlabel("core_mac_density_scaled")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)


# =========================
# Analysis helpers: per-scenario adversary-max cluster + DENSITY_MIN grid
# =========================

# Grid to evaluate (paper-facing tuning support)
DENSITY_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.8, 2.0, 2.5, 3.0]


def _parse_s_to_seconds(token: str) -> float:
    """Parse tokens like '2s','10s','30s','1min','5min','15min' into seconds."""
    if not token:
        return float("nan")
    t = str(token).strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)(s|min)$", t)
    if not m:
        return float("nan")
    v = float(m.group(1))
    unit = m.group(2)
    return v if unit == "s" else v * 60.0


def _parse_tx_rot_from_filename(p: Path) -> tuple[float, float, str, str]:
    """Extract tx and rot periods from names like scenario_tx-10s_rot-5min.csv."""
    name = p.name
    m = re.search(r"tx-([0-9]+(?:\.[0-9]+)?(?:s|min))_rot-([0-9]+(?:\.[0-9]+)?(?:s|min))", name)
    if not m:
        return (float("nan"), float("nan"), "", "")
    tx_tok = m.group(1)
    rot_tok = m.group(2)
    return (_parse_s_to_seconds(tx_tok), _parse_s_to_seconds(rot_tok), tx_tok, rot_tok)


def _scenario_advmax_row(summary_df: pd.DataFrame, src_file: str) -> dict:
    """Return dict describing the cluster with maximum adv_mac_pct in a scenario."""
    if summary_df is None or len(summary_df) == 0:
        return {"src_file": src_file}

    d = summary_df.copy()
    if "adv_mac_pct" not in d.columns:
        d["adv_mac_pct"] = 0.0
    # if all NaN, treat as 0
    d["adv_mac_pct"] = d["adv_mac_pct"].fillna(0.0)

    # pick max adv presence; ties broken by higher core density
    d["core_mac_density_scaled"] = d.get("core_mac_density_scaled", np.nan)
    d["core_mac_density_scaled"] = d["core_mac_density_scaled"].fillna(-1e18)

    best = d.sort_values(by=["adv_mac_pct", "core_mac_density_scaled"], ascending=[False, False]).iloc[0]

    return {
        "src_file": src_file,
        "cluster": int(best.get("cluster", -1)),
        "dev_type": str(best.get("dev_type", "")),
        "segments": int(best.get("segments", 0)),
        "duration_cov": float(best.get("duration_cov", np.nan)),
        "unique_macs": int(best.get("unique_macs", 0)),
        "turnover": float(best.get("turnover", np.nan)) if ("turnover" in best.index) else np.nan,
        "singleton_mac_frac": float(best.get("singleton_mac_frac", np.nan)),
        "mac_rate_per_min": float(best.get("mac_rate_per_min", np.nan)),
        "core_mac_density_scaled": float(best.get("core_mac_density_scaled", np.nan)) if best.get("core_mac_density_scaled", -1e18) > -1e17 else np.nan,
        "adv_mac_pct": float(best.get("adv_mac_pct", 0.0)),
        "gt_frac": float(best.get("gt_frac", np.nan)),
    }


def _write_advmax_density_plot(df_advmax: pd.DataFrame, out_png: str) -> None:
    """Scatter plot: adversary-max core density vs tx/rot behavior."""
    if df_advmax is None or len(df_advmax) == 0:
        return

    # tolerate error-only rows
    if "core_mac_density_scaled" not in df_advmax.columns:
        return
    if "tx_s" not in df_advmax.columns or "rot_s" not in df_advmax.columns:
        return

    d = df_advmax.copy()
    d = d[np.isfinite(d["core_mac_density_scaled"])].copy()
    if len(d) == 0:
        return

    # Use rot period as color, tx period as x; density as y
    x = d["tx_s"].astype(float).values
    y = d["core_mac_density_scaled"].astype(float).values
    c = d["rot_s"].astype(float).values

    plt.figure(figsize=(10, 5))
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=60, alpha=0.9, edgecolors="k", linewidths=0.3)
    cb = plt.colorbar(sc)
    cb.set_label("rot period (s)")

    # reference density line
    if DENSITY_MIN is not None:
        plt.axhline(float(DENSITY_MIN), color="red", linestyle="--", linewidth=2, label=f"DENSITY_MIN={DENSITY_MIN}")
        plt.legend(loc="best")

    # If we try to log-scale an axis with non-positive values, matplotlib will throw.
    # Decide scaling based on the actual plotted values.
    try:
        yv = pd.to_numeric(d.get("core_mac_density_scaled", pd.Series([], dtype=float)), errors="coerce")
        has_pos_y = bool((yv.dropna() > 0).any())
    except Exception:
        has_pos_y = False

    try:
        xv = pd.to_numeric(d.get("tx_s", pd.Series([], dtype=float)), errors="coerce")
        has_pos_x = bool((xv.dropna() > 0).any())
    except Exception:
        has_pos_x = False

    if has_pos_y:
        plt.yscale("log")
    else:
        plt.yscale("linear")

    if has_pos_x:
        plt.xscale("log")
    else:
        plt.xscale("linear")

    plt.xlabel("tx period (s) [log]")
    plt.ylabel("core_mac_density_scaled (adv-max cluster)")
    plt.title("Adversary-max core density vs tx/rot behavior")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)


def _run_csv_list(csvs: list[Path], label: str) -> None:
    """Run pipeline over a list of CSV paths and write consolidated outputs."""
    if not csvs:
        raise FileNotFoundError(f"No CSVs to run for: {label}")

    print(f"\n=== Run mode: {label} ===")
    print(f"CSV files: {len(csvs)}")

    all_checks = []
    meta_rows = []
    advmax_rows = []

    for i, p in enumerate(csvs, start=1):
        print(f"\n[{i}/{len(csvs)}] Running: {p}")
        try:
            cand_df, summary_df, meta = _run_one_csv(p)
            if cand_df is not None and len(cand_df) > 0:
                all_checks.append(cand_df)
            meta_rows.append(meta)

            # per-scenario adversary-max cluster stats
            row = _scenario_advmax_row(summary_df, src_file=str(p))
            tx_s, rot_s, tx_tok, rot_tok = _parse_tx_rot_from_filename(p)
            row.update({"tx_s": float(tx_s), "rot_s": float(rot_s), "tx_tok": tx_tok, "rot_tok": rot_tok})

            # grid pass/fail for this adv-max cluster under different DENSITY_MIN
            core_val = row.get("core_mac_density_scaled", np.nan)
            for thr in DENSITY_GRID:
                row[f"pass_dens_ge_{thr}"] = bool(np.isfinite(core_val) and (float(core_val) >= float(thr)))

            advmax_rows.append(row)

        except Exception as e:
            meta_rows.append({
                "src_file": str(p),
                "error": str(e),
                "confirmed_any": False,
            })
            print(f"[ERROR] Failed on {p}: {e}")

            tx_s, rot_s, tx_tok, rot_tok = _parse_tx_rot_from_filename(p)
            advmax_rows.append({
                "src_file": str(p),
                "error": str(e),
                "tx_s": float(tx_s),
                "rot_s": float(rot_s),
                "tx_tok": tx_tok,
                "rot_tok": rot_tok,
            })

    meta_df = pd.DataFrame(meta_rows)
    checks_df = pd.concat(all_checks, ignore_index=True) if all_checks else pd.DataFrame()
    advmax_df = pd.DataFrame(advmax_rows)

    # ensure columns exist for downstream plotting even if all rows errored
    if "core_mac_density_scaled" not in advmax_df.columns:
        advmax_df["core_mac_density_scaled"] = np.nan
    if "tx_s" not in advmax_df.columns:
        advmax_df["tx_s"] = np.nan
    if "rot_s" not in advmax_df.columns:
        advmax_df["rot_s"] = np.nan

    tag = f"dens{DENSITY_MIN}"
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:80]

    out_csv = f"aircatch_{safe_label}_candidate_checks__{tag}.csv"
    out_meta = f"aircatch_{safe_label}_meta__{tag}.csv"
    out_png = f"aircatch_{safe_label}_core_density_dist__{tag}.png"

    out_advmax = f"aircatch_{safe_label}_advmax_per_scenario__densgrid.csv"
    out_advmax_png = f"aircatch_{safe_label}_advmax_core_density_vs_tx_rot__{tag}.png"

    meta_df.to_csv(out_meta, index=False)
    if len(checks_df) > 0:
        # Ensure stable columns for analysis (even if some are missing for a run)
        for col in [
            "k_chosen",
            "win_entropy_norm",
            "uniform_ok",
        ]:
            if col not in checks_df.columns:
                checks_df[col] = np.nan

        checks_df.to_csv(out_csv, index=False)
        _write_core_density_plot(checks_df, out_png)

    if len(advmax_df) > 0:
        advmax_df.to_csv(out_advmax, index=False)
        _write_advmax_density_plot(advmax_df, out_advmax_png)

    print("\n=== Outputs ===")
    print(f"Wrote: {out_meta}")
    if len(checks_df) > 0:
        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_png}")
    if len(advmax_df) > 0:
        print(f"Wrote: {out_advmax}")
        print(f"Wrote: {out_advmax_png}")


def main():
    ap = argparse.ArgumentParser(description="AirCatch batch/single runner")
    ap.add_argument(
        "--input",
        default=None,
        help="Optional: path to a single CSV file or a folder (recursively loads *.csv). If omitted, runs batch on controlled/<CONTROLLED_SUBFOLDER>.",
    )
    ap.add_argument(
        "--density-min",
        default=None,
        type=float,
        help="Override DENSITY_MIN for this run.",
    )
    args = ap.parse_args()

    global DENSITY_MIN
    if args.density_min is not None:
        DENSITY_MIN = float(args.density_min)

    if args.input:
        p = Path(args.input).expanduser()
        csvs = _collect_csv_files(p)
        if not csvs:
            raise FileNotFoundError(f"No CSV files found under: {p}")
        label = f"single_{p.name}" if p.is_file() else f"folder_{p.name}"
        _run_csv_list(csvs, label=label)
        return

    base = Path(CONTROLLED_ROOT) / CONTROLLED_SUBFOLDER
    csvs = _list_csvs_in_controlled_subfolder(CONTROLLED_ROOT, CONTROLLED_SUBFOLDER)
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under {base}")

    _run_csv_list(csvs, label=f"batch_{CONTROLLED_SUBFOLDER}")


if __name__ == "__main__":
    main()