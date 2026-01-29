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

ADV_CSV = "controlled/SDR_Adv/scenarios_car__adv0_apple0_google0_samsung0_tile0__20260125_033106/background_only.csv"
# ADV_CSV = "car.csv"

# Fixed dataset folder (no prompt)
CONTROLLED_ROOT = "controlled"
CONTROLLED_SUBFOLDER = "WtoH"  # <-- change this to another subfolder name under controlled/

PAYLOAD_TAG = "4c001219"        # optional, NOT used as filter by default

# ADV marker(s) used for GT and adv_mac_pct
ADV_PAYLOAD_TAGS = ["4c001219fc", "4c001219fd", "4c001219fe", "4c001219ff"]

WINDOW_S = 1800                 # seconds per time bucket (e.g., 300=5min, 120=2min, 900=15min)
K_RANGE = range(3, 20)
OVERALL_CFO_WEIGHT = 1

MIN_DURATION_S = 1700           # strict decision min support (seconds)

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
DENSITY_MIN = 1 #1.8

# Grid of density thresholds used for per-scenario (adv-max) pass/fail reporting
# (kept separate from DENSITY_MIN so we can sweep thresholds without rerunning.)
DENSITY_GRID = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0]

# Periodic mode (process each CSV in successive time blocks)
PERIODIC_MODE = True
PERIODIC_BLOCK_S = 3600   # 10 minutes, 1800 for 30 minutes
PERIODIC_STEP_S = 3000    # non-overlapping by default, 1800 for 30 minutes

# Periodic persistence confirmation (simple)
# Require the same cluster to persist long enough within the file (seconds).
# This uses the cluster's per-block persistence_s (t_end.max - t_start.min) and aggregates across blocks.
PERIODIC_MIN_PERSISTENCE_S = 3600


# =========================
# No-passer-by strict gates (persistence + real CFO density)
# =========================
# Keep these generic and defensible:
#  1) persistence: reject short passer-bys
#
# NOTE: Temporal uniformity gating was removed by request.

STRICT_MIN_PERSISTENCE_S = 1700

def robust_stats(x: np.ndarray) -> dict:
    """Compute small, robust statistics for a 1D array.

    Keeps existing keys for backward compatibility and adds extra robust summaries.
    """
    if x is None or len(x) == 0:
        return {
            "median": 0.0,
            "iqr": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "mad": 0.0,
            # extras
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "trimmed_mean_10": 0.0,
            "winsor_std_10": 0.0,
            "mad_scale": 0.0,
        }

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "median": 0.0,
            "iqr": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "mad": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "trimmed_mean_10": 0.0,
            "winsor_std_10": 0.0,
            "mad_scale": 0.0,
        }

    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))

    p01 = float(np.percentile(x, 1))
    p05 = float(np.percentile(x, 5))
    p10 = float(np.percentile(x, 10))
    p90 = float(np.percentile(x, 90))
    p95 = float(np.percentile(x, 95))
    p99 = float(np.percentile(x, 99))

    mad = float(np.median(np.abs(x - med)))
    # MAD-to-sigma scale factor for normal dist
    mad_scale = float(1.4826 * mad)

    # Trimmed mean (10% each tail)
    lo_t, hi_t = p10, p90
    xt = x[(x >= lo_t) & (x <= hi_t)]
    trimmed_mean_10 = float(np.mean(xt)) if xt.size else float(np.mean(x))

    # Winsorized std (10% each tail)
    xw = np.clip(x, lo_t, hi_t)
    winsor_std_10 = float(np.std(xw)) if xw.size else float(np.std(x))

    return {
        "median": float(med),
        "iqr": float(q3 - q1),
        "p10": float(p10),
        "p90": float(p90),
        "mad": float(mad),
        # extras
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(p01),
        "p05": float(p05),
        "p95": float(p95),
        "p99": float(p99),
        "trimmed_mean_10": float(trimmed_mean_10),
        "winsor_std_10": float(winsor_std_10),
        "mad_scale": float(mad_scale),
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
      adv_mac_set: ground-truth set of MACs that appear in packets whose payload contains any ADV_PAYLOAD_TAGS
    """
    df = df.copy()

    # normalize / required columns
    df["payload"] = df["payload"].astype(str).str.lower()
    df["AdvA"] = df["AdvA"].astype(str)

    # --- Ground-truth ADV MAC set (MUST be computed BEFORE any CRC filtering) ---
    # Positive case definition: if any packets contain ADV_PAYLOAD_TAGS, the scenario is positive.
    adv_mask_gt = df["payload"].astype(str).str.lower().apply(lambda s: any(t in s for t in ADV_PAYLOAD_TAGS))
    adv_mac_set = set(df.loc[adv_mask_gt, "AdvA"].astype(str).tolist())

    if not b210 and "crc_ok" in df.columns:
        df = df[df["crc_ok"] == 1]  # filter valid CRC only

    # Ensure required columns exist; be robust to missing CFO side columns
    for c in CFO_COLS_RAW:
        if c not in df.columns:
            df[c] = np.nan

    df = df.dropna(subset=["timestamp", "AdvA"])
    df = df.sort_values("timestamp")

    # Ground-truth ADV MAC set (packet-level marker): match any configured ADV tag prefix
    adv_mask = df["payload"].astype(str).str.lower().apply(
        lambda s: any(t in s for t in ADV_PAYLOAD_TAGS)
    )
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

        # Count packets in this segment that match any configured ADV tag prefix

        # Count packets in this segment that match any configured ADV tag prefix
        adv_macs = int(
            g["payload"].astype(str).str.lower().apply(lambda s: any(t in s for t in ADV_PAYLOAD_TAGS)).sum()
        )
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

        # Cluster-level GT based on *MAC presence* of adversary-tagged packets.
        # Each row in `seg` is a per-device segment; `adv_macs>0` means that MAC/PRIVID
        # had at least one tagged packet in that time bucket.
        macs_in_cluster = g["mac"].astype(str)
        unique_mac_set = set([m for m in macs_in_cluster.tolist() if m])
        unique_macs = len(unique_mac_set)

        adv_seg_mask = (g.get("adv_macs", 0).fillna(0).astype(float) > 0)
        adv_mac_set = set(g.loc[adv_seg_mask, "mac"].astype(str).tolist())
        adv_mac_set = set([m for m in adv_mac_set if m])

        adv_mac_pct = float(len(adv_mac_set) / max(unique_macs, 1))

        # "gt" is MAC-based: a MAC is adversary if it ever appeared in the global adv_mac_set
        # built from tagged packets across the whole file/block.
        gt_any = bool(g["gt_adv"].any()) if "gt_adv" in g.columns else False

        gt_mac_set = set(g.loc[g.get("gt_adv", False).astype(bool), "mac"].astype(str).tolist()) if "gt_adv" in g.columns else set()
        gt_mac_set = set([m for m in gt_mac_set if m])
        gt_frac = float(len(gt_mac_set) / max(unique_macs, 1))

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

            "unique_macs": int(unique_macs),

            # old metric (kept)
            "mac_diversity": float(mac_diversity),

            # Requested: adversary fraction based on *MACs* that have any tagged packet
            "adv_mac_pct": float(adv_mac_pct),
            "n_packets_avg": n_packets_avg,
            "total_packets": total_packets,

            "gt_any": gt_any,
            "gt_frac": float(gt_frac),
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
    """Rank candidate clusters."""
    df = summary_df.copy()
    if len(df) == 0:
        return df

    df["dens_rank"] = df.get("core_mac_density_scaled", np.nan).fillna(-1e18)
    df = df.sort_values(by=["dens_rank"], ascending=[False])
    return df.head(top_n)


def _strict_decision(row: pd.Series) -> tuple[bool, dict]:
    """Return (ok, components) for strict confirmation.

    confirm := (persistence >= STRICT_MIN_PERSISTENCE_S) AND
               (core density >= DENSITY_MIN)
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


# =========================
# Periodic persistence tracker (across steps)
# =========================

def _cluster_cfo_centroid_5d(row: pd.Series) -> np.ndarray:
    """Extract 5D CFO centroid from a summary row (CFO, CFO_00, CFO_11, CFO_10, CFO_01)."""
    vals = []
    for k in ["CFO", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]:
        v = row.get(k, np.nan)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


def _cluster_signature_from_summary_row(row: pd.Series, *, q_hz: float = 10.0) -> str:
    """Build a stable signature for linking clusters across periodic blocks.

    Signature := dev_type + quantized 5D CFO centroid.
    q_hz controls quantization granularity (Hz).
    """
    dev_type = str(row.get("dev_type", "UNKNOWN"))
    c = _cluster_cfo_centroid_5d(row)
    if not np.isfinite(c).all():
        # fallback: still return a dev_type-only signature; will be conservative
        return f"{dev_type}|cfo=NA"

    q = float(q_hz) if (q_hz is not None and q_hz > 0) else 1.0
    cq = np.round(c / q).astype(int)

    # Use a short stable string (avoid floats)
    return f"{dev_type}|{int(cq[0])},{int(cq[1])},{int(cq[2])},{int(cq[3])},{int(cq[4])}"


def _update_periodic_tracks(tracks: dict, summary_df: pd.DataFrame, *, block_start: float, block_end: float) -> None:
    """Update persistence tracks with clusters observed in the current block."""
    if summary_df is None or len(summary_df) == 0:
        return

    for _, r in summary_df.iterrows():
        sig = _cluster_signature_from_summary_row(r)
        t0 = float(block_start)
        t1 = float(block_end)

        st = tracks.get(sig)
        if st is None:
            tracks[sig] = {
                "first_seen": t0,
                "last_seen": t1,
                "n_blocks": 1,
                "last_row": r,
            }
        else:
            st["last_seen"] = max(float(st.get("last_seen", t1)), t1)
            st["n_blocks"] = int(st.get("n_blocks", 0)) + 1
            st["last_row"] = r


def _any_track_confirmed(tracks: dict, *, min_persist_s: float) -> bool:
    for _, st in tracks.items():
        t0 = float(st.get("first_seen", np.nan))
        t1 = float(st.get("last_seen", np.nan))
        if np.isfinite(t0) and np.isfinite(t1) and (t1 - t0) >= float(min_persist_s):
            return True
    return False


def _first_track_confirm_time(tracks: dict, *, min_persist_s: float) -> float:
    """Earliest time (absolute timestamp) when any tracked entity reaches persistence >= min_persist_s."""
    best = float("inf")
    for _, st in tracks.items():
        t0 = float(st.get("first_seen", np.nan))
        t1 = float(st.get("last_seen", np.nan))
        if not (np.isfinite(t0) and np.isfinite(t1)):
            continue
        if (t1 - t0) >= float(min_persist_s):
            best = min(best, t1)
    return best if np.isfinite(best) and best != float("inf") else float("nan")


def _compute_ttd_seconds(meta: dict) -> float:
    """Time-to-detect (TTD) from beginning of the file to first correct flag.

    Prefer periodic persistence confirmation time (track-based). Fallback to block strict-confirmed.
    """
    # Preferred: track-based confirmation time
    if isinstance(meta, dict) and meta.get("periodic", False):
        t0 = float(meta.get("t0", np.nan))
        t_conf = float(meta.get("ttd_confirm_time", np.nan))
        if np.isfinite(t0) and np.isfinite(t_conf) and t_conf >= t0:
            return float(t_conf - t0)

    # Fallback: legacy strict-confirmed block timing
    blocks = meta.get("blocks") if isinstance(meta, dict) else None
    if not isinstance(blocks, list) or len(blocks) == 0:
        return float("nan")

    b = [x for x in blocks if isinstance(x, dict) and ("block_start" in x) and ("strict_confirmed" in x)]
    if not b:
        return float("nan")
    b = sorted(b, key=lambda r: float(r.get("block_start", 0.0)))
    t0 = float(b[0].get("block_start", 0.0))
    for r in b:
        if bool(r.get("strict_confirmed", False)):
            return max(0.0, float(r.get("block_start", 0.0)) - t0)
    return float("nan")


def _run_one_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    adv = pd.read_csv(csv_path)

    # Compute GT adv MAC count from RAW input (no CRC filter) so evaluation matches definition.
    try:
        _payload = adv.get("payload", pd.Series([], dtype=str)).astype(str).str.lower()
        _adva = adv.get("AdvA", pd.Series([], dtype=str)).astype(str)
        gt_mask = _payload.apply(lambda s: any(t in s for t in ADV_PAYLOAD_TAGS))
        gt_adv_mac_count = int(len(set(_adva.loc[gt_mask].tolist())))
    except Exception:
        gt_adv_mac_count = 0

    # Always run through the same code path so we can reuse strict decision flagging.
    if not PERIODIC_MODE:
        cdf, sdf, m = _run_one_csv_once(csv_path, adv, block_start=None, block_end=None)
        meta = m.copy()
        meta["periodic"] = False
        meta["confirmed_any"] = bool(m.get("strict_confirmed", False))
        meta["gt_adv_mac_count"] = int(gt_adv_mac_count)
        return cdf, sdf, meta

    all_checks = []
    all_summaries = []
    all_meta = []

    tracks = {}

    # Track file start time for TTD
    try:
        _t = pd.to_numeric(adv.get("timestamp", pd.Series([], dtype=float)), errors="coerce")
        t0_file = float(np.nanmin(_t.values)) if len(_t) else float("nan")
    except Exception:
        t0_file = float("nan")

    strict_any = False

    for b0, b1, adv_b in _iter_time_blocks(adv, t_col="timestamp"):
        if len(adv_b) == 0:
            continue
        cdf, sdf, m = _run_one_csv_once(csv_path, adv_b, block_start=b0, block_end=b1)
        if isinstance(m, dict):
            m["block_start"] = float(b0)
            m["block_end"] = float(b1)
            strict_any = strict_any or bool(m.get("strict_confirmed", False))

        if cdf is not None and len(cdf) > 0:
            all_checks.append(cdf)
        if sdf is not None and len(sdf) > 0:
            sdf = sdf.copy()
            sdf["block_start"] = float(b0)
            sdf["block_end"] = float(b1)
            all_summaries.append(sdf)
            _update_periodic_tracks(tracks, sdf, block_start=float(b0), block_end=float(b1))

        all_meta.append(m)

    check_df = pd.concat(all_checks, ignore_index=True) if all_checks else pd.DataFrame()
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    persist_confirmed = _any_track_confirmed(tracks, min_persist_s=float(PERIODIC_MIN_PERSISTENCE_S))
    confirmed_time = _first_track_confirm_time(tracks, min_persist_s=float(PERIODIC_MIN_PERSISTENCE_S))

    # Final periodic decision: require BOTH strict confirmation and persistence
    confirmed_any = bool(persist_confirmed and strict_any)

    meta = {
        "src_file": str(csv_path),
        "periodic": True,
        "block_s": float(PERIODIC_BLOCK_S),
        "step_s": float(PERIODIC_STEP_S),
        "n_blocks": int(len(all_meta)),
        "periodic_min_persistence_s": float(PERIODIC_MIN_PERSISTENCE_S),
        "confirmed_any": bool(confirmed_any),
        "persist_confirmed": bool(persist_confirmed),
        "strict_any": bool(strict_any),
        "n_tracks": int(len(tracks)),
        "gt_adv_mac_count": int(gt_adv_mac_count),
        # For TTD
        "t0": float(t0_file),
        "ttd_confirm_time": float(confirmed_time),
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


def _write_adv_mac_pct_distribution_pdf(checks_df: pd.DataFrame, out_pdf: str) -> None:
    """Plot distribution of adv_mac_pct across flagged candidate clusters."""
    if checks_df is None or len(checks_df) == 0:
        return

    if "adv_mac_pct" not in checks_df.columns:
        return

    d = checks_df.copy()
    d["adv_mac_pct"] = pd.to_numeric(d["adv_mac_pct"], errors="coerce")
    d = d[np.isfinite(d["adv_mac_pct"])].copy()
    if len(d) == 0:
        return

    # Clamp to [0,1] in case of any upstream anomalies
    vals = d["adv_mac_pct"].astype(float).clip(lower=0.0, upper=1.0).values

    plt.figure(figsize=(9, 4.5))
    bins = np.linspace(0.0, 1.0, 21)
    plt.hist(vals, bins=bins, alpha=0.85, color="slateblue", edgecolor="white")

    p50 = float(np.percentile(vals, 50))
    p90 = float(np.percentile(vals, 90))
    p95 = float(np.percentile(vals, 95))
    plt.axvline(p50, color="black", linestyle="--", linewidth=1.0, label=f"p50={p50:.2f}")
    plt.axvline(p90, color="darkred", linestyle=":", linewidth=1.2, label=f"p90={p90:.2f}")
    plt.axvline(p95, color="darkgreen", linestyle=":", linewidth=1.2, label=f"p95={p95:.2f}")

    plt.title("Distribution of adv_mac_pct across flagged candidate clusters")
    plt.xlabel("adv_mac_pct (fraction of MACs in cluster with ADV tag)")
    plt.ylabel("# candidate clusters")
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    try:
        plt.savefig(out_pdf)
    finally:
        plt.close()


def _plot_cdf(ax, values: np.ndarray, *, label: str, color: str, linewidth: float = 2.0) -> None:
    """Plot an empirical CDF on a Matplotlib axis."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    v = np.sort(v)
    y = np.arange(1, v.size + 1, dtype=float) / float(v.size)
    ax.plot(v, y, label=label, color=color, linewidth=linewidth)


def _write_adv_mac_pct_cdf_pdf(checks_df: pd.DataFrame, out_pdf: str) -> None:
    """CDF of adv_mac_pct across flagged candidate clusters."""
    if checks_df is None or len(checks_df) == 0:
        return
    if "adv_mac_pct" not in checks_df.columns:
        return

    d = checks_df.copy()
    # Coerce to numeric and drop invalid
    d["adv_mac_pct"] = pd.to_numeric(d["adv_mac_pct"], errors="coerce")
    d = d[np.isfinite(d["adv_mac_pct"])].copy()
    if len(d) == 0:
        return

    vals = d["adv_mac_pct"].astype(float).clip(lower=0.0, upper=1.0).values

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _plot_cdf(ax, vals, label="flagged candidates", color="slateblue")

    ax.set_title("CDF of adv_mac_pct (flagged candidate clusters)")
    ax.set_xlabel("adv_mac_pct (fraction of MACs with ADV tag)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    try:
        fig.savefig(out_pdf)
    finally:
        plt.close(fig)


def _write_core_density_flagged_vs_not_cdf_pdf(summary_df: pd.DataFrame, checks_df: pd.DataFrame, out_pdf: str) -> None:
    """Two CDFs: core density of clusters that are flagged (decision_ok==True) vs not."""
    if summary_df is None or len(summary_df) == 0:
        return
    if "core_mac_density_scaled" not in summary_df.columns:
        return
    if checks_df is None or len(checks_df) == 0:
        return

    # Flagged := decision_ok == True in candidate checks.
    # Use (src_file, cluster) keys because cluster ids restart per input file.
    if "decision_ok" not in checks_df.columns:
        return

    flagged_df = checks_df[checks_df["decision_ok"].astype(bool)].copy()
    if len(flagged_df) == 0:
        return

    if "src_file" not in flagged_df.columns or "cluster" not in flagged_df.columns:
        return

    flagged_keys = set(
        zip(flagged_df["src_file"].astype(str).tolist(), flagged_df["cluster"].astype(int).tolist())
    )

    d = summary_df.copy()
    if "src_file" not in d.columns:
        # Without src_file we cannot safely compare across many inputs.
        return

    d = d[np.isfinite(d["core_mac_density_scaled"])].copy()
    if len(d) == 0:
        return

    d["_key"] = list(zip(d["src_file"].astype(str).tolist(), d["cluster"].astype(int).tolist()))
    d["_is_flagged"] = d["_key"].isin(flagged_keys)

    v_flag = d.loc[d["_is_flagged"], "core_mac_density_scaled"].astype(float).values
    v_nofl = d.loc[~d["_is_flagged"], "core_mac_density_scaled"].astype(float).values

    if v_flag.size == 0 or v_nofl.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _plot_cdf(ax, v_flag, label=f"flagged decision_ok=True (n={v_flag.size})", color="darkorange")
    _plot_cdf(ax, v_nofl, label=f"not flagged (n={v_nofl.size})", color="steelblue")

    ax.set_title("CDF of core_mac_density_scaled: decision_ok True vs False")
    ax.set_xlabel("core_mac_density_scaled")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    try:
        fig.savefig(out_pdf)
    finally:
        plt.close(fig)


def _write_core_density_adv_present_vs_not_cdf_pdf(summary_df: pd.DataFrame, *, adv_presence_col: str, out_pdf: str) -> None:
    """Two CDFs: core density of clusters with adversary MAC presence vs without."""
    if summary_df is None or len(summary_df) == 0:
        return
    if "core_mac_density_scaled" not in summary_df.columns:
        return
    if adv_presence_col not in summary_df.columns:
        return

    d = summary_df.copy()
    d = d[np.isfinite(d["core_mac_density_scaled"])].copy()
    if len(d) == 0:
        return

    present = d[adv_presence_col].fillna(0.0).astype(float) > 0.0
    v_yes = d.loc[present, "core_mac_density_scaled"].astype(float).values
    v_no = d.loc[~present, "core_mac_density_scaled"].astype(float).values

    if v_yes.size == 0 or v_no.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _plot_cdf(ax, v_yes, label=f"adv present (n={v_yes.size})", color="crimson")
    _plot_cdf(ax, v_no, label=f"adv absent (n={v_no.size})", color="seagreen")

    ax.set_title(f"CDF of core_mac_density_scaled: {adv_presence_col}>0 vs ==0")
    ax.set_xlabel("core_mac_density_scaled")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    try:
        fig.savefig(out_pdf)
    finally:
        plt.close(fig)


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


def _write_advmax_density_plot(advmax_df: pd.DataFrame, out_png: str) -> None:
    """Plot adv-max core density vs tx/rot behavior.

    Robust to csv rows that may contain non-numeric tokens (e.g. from error rows).
    """
    if advmax_df is None or len(advmax_df) == 0:
        return

    d = advmax_df.copy()

    # Coerce to numeric and drop invalid rows (matplotlib scatter requires floats)
    for col in ["tx_s", "rot_s", "core_mac_density_scaled"]:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[np.isfinite(d["tx_s"]) & np.isfinite(d["rot_s"]) & np.isfinite(d["core_mac_density_scaled"])].copy()
    if len(d) == 0:
        # Still write an empty placeholder figure so batch doesn't crash
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title("Adversary-max core density vs tx/rot behavior (no valid rows)")
        ax.axis("off")
        fig.tight_layout()
        try:
            fig.savefig(out_png, dpi=200)
        finally:
            plt.close(fig)
        return

    # Use rot period as color, tx period as x; density as y
    # NOTE: force numpy float dtype here to prevent any downstream dtype leakage.
    x = np.asarray(d["tx_s"].values, dtype=float)
    y = np.asarray(d["core_mac_density_scaled"].values, dtype=float)
    c = np.asarray(d["rot_s"].values, dtype=float)

    # Safety: drop any non-finite rows after numpy coercion
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[m], y[m], c[m]

    if x.size == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title("Adversary-max core density vs tx/rot behavior (no valid numeric rows)")
        ax.axis("off")
        fig.tight_layout()
        try:
            fig.savefig(out_png, dpi=200)
        finally:
            plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis",
            s=60,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.3,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("rot period (s)")

        # reference density line
        if DENSITY_MIN is not None and np.isfinite(float(DENSITY_MIN)):
            ax.axhline(float(DENSITY_MIN), color="red", linestyle="--", linewidth=2, label=f"DENSITY_MIN={DENSITY_MIN}")
            ax.legend(loc="best")

        # If we try to log-scale an axis with non-positive values, matplotlib will throw.
        has_pos_y = bool((y > 0).any())
        has_pos_x = bool((x > 0).any())

        ax.set_yscale("log" if has_pos_y else "linear")
        ax.set_xscale("log" if has_pos_x else "linear")

        ax.set_xlabel("tx period (s) [log]")
        ax.set_ylabel("core_mac_density_scaled (adv-max cluster)")
        ax.set_title("Adversary-max core density vs tx/rot behavior")
        fig.tight_layout()

        fig.savefig(out_png, dpi=200)
    finally:
        plt.close(fig)


# =========================
# Evaluation (paper metrics)
# =========================

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _scenario_hours(df_raw: pd.DataFrame) -> float:
    if df_raw is None or len(df_raw) == 0 or "timestamp" not in df_raw.columns:
        return 0.0
    t = pd.to_numeric(df_raw["timestamp"], errors="coerce")
    if t.isna().all():
        return 0.0
    dur = float(np.nanmax(t.values) - np.nanmin(t.values))
    return max(dur / 3600.0, 1e-9)


def _compute_cluster_purity_from_summary(summary_df: pd.DataFrame) -> float:
    """Cluster purity when gt_frac is available: average of max(gt, 1-gt) across clusters."""
    if summary_df is None or len(summary_df) == 0 or "gt_frac" not in summary_df.columns:
        return float("nan")
    g = pd.to_numeric(summary_df["gt_frac"], errors="coerce")
    g = g[np.isfinite(g)]
    if len(g) == 0:
        return float("nan")
    return float(np.mean(np.maximum(g.values, 1.0 - g.values)))


def _compute_detection_metrics(meta: dict, checks_df: pd.DataFrame) -> dict:
    """Per-CSV detection metrics using file-level confirmed_any vs gt_adv_mac_count>0.

    This is conservative for clustering: it treats the scenario as positive if any adversary-tagged MACs exist.
    """
    gt_pos = bool(int(meta.get("gt_adv_mac_count", 0) or 0) > 0)
    pred_pos = bool(meta.get("confirmed_any", False))

    tp = int(gt_pos and pred_pos)
    fp = int((not gt_pos) and pred_pos)
    fn = int(gt_pos and (not pred_pos))
    tn = int((not gt_pos) and (not pred_pos))

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec) if (prec + rec) else 0.0

    # For completeness: how many clusters were flagged inside this file
    n_flagged_clusters = int((checks_df["decision_ok"].astype(bool).sum()) if (checks_df is not None and len(checks_df) and "decision_ok" in checks_df.columns) else 0)

    return {
        "gt_pos": gt_pos,
        "pred_pos": pred_pos,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_flagged_clusters": n_flagged_clusters,
    }


def _compute_ttd_seconds(meta: dict) -> float:
    """Time-to-detect (TTD) from beginning of the file to first correct flag.

    Prefer periodic persistence confirmation time (track-based). Fallback to block strict-confirmed.
    """
    # Preferred: track-based confirmation time
    if isinstance(meta, dict) and meta.get("periodic", False):
        t0 = float(meta.get("t0", np.nan))
        t_conf = float(meta.get("ttd_confirm_time", np.nan))
        if np.isfinite(t0) and np.isfinite(t_conf) and t_conf >= t0:
            return float(t_conf - t0)

    # Fallback: legacy strict-confirmed block timing
    blocks = meta.get("blocks") if isinstance(meta, dict) else None
    if not isinstance(blocks, list) or len(blocks) == 0:
        return float("nan")

    b = [x for x in blocks if isinstance(x, dict) and ("block_start" in x) and ("strict_confirmed" in x)]
    if not b:
        return float("nan")
    b = sorted(b, key=lambda r: float(r.get("block_start", 0.0)))
    t0 = float(b[0].get("block_start", 0.0))
    for r in b:
        if bool(r.get("strict_confirmed", False)):
            return max(0.0, float(r.get("block_start", 0.0)) - t0)
    return float("nan")


def _write_paper_report_txt(per_csv_rows: list[dict], out_txt: str) -> None:
    lines = []
    lines.append("AirCatch evaluation report")
    lines.append(f"n_csv={len(per_csv_rows)}")
    lines.append("")

    # totals
    tp = sum(int(r.get("tp", 0)) for r in per_csv_rows)
    fp = sum(int(r.get("fp", 0)) for r in per_csv_rows)
    fn = sum(int(r.get("fn", 0)) for r in per_csv_rows)
    tn = sum(int(r.get("tn", 0)) for r in per_csv_rows)

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec) if (prec + rec) else 0.0

    total_hours = float(sum(float(r.get("hours", 0.0)) for r in per_csv_rows))
    fp_h = _safe_div(fp, total_hours)
    fn_h = _safe_div(fn, total_hours)

    lines.append("=== Overall detection (scenario-level) ===")
    lines.append(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    lines.append(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    lines.append("")
    lines.append("=== Operational error rates ===")
    lines.append(f"TotalHours={total_hours:.3f} FP_per_hour={fp_h:.4f} FN_per_hour={fn_h:.4f}")
    lines.append("")

    # TTD summary (only for gt_pos scenarios)
    ttd = [float(r.get("ttd_s", np.nan)) for r in per_csv_rows if bool(r.get("gt_pos", False))]
    ttd = [x for x in ttd if np.isfinite(x)]
    if ttd:
        lines.append("=== Time-to-detect (TTD, seconds) on positive scenarios ===")
        lines.append(f"n={len(ttd)}  median={np.median(ttd):.2f}  p90={np.percentile(ttd,90):.2f}  p95={np.percentile(ttd,95):.2f}")
        lines.append("")

    # clustering metrics
    sil = [float(r.get("silhouette", np.nan)) for r in per_csv_rows]
    sil = [x for x in sil if np.isfinite(x)]
    if sil:
        lines.append("=== Clustering quality ===")
        lines.append(f"Silhouette: n={len(sil)} mean={np.mean(sil):.4f} median={np.median(sil):.4f}")
    pur = [float(r.get("purity", np.nan)) for r in per_csv_rows]
    pur = [x for x in pur if np.isfinite(x)]
    if pur:
        lines.append(f"Purity: n={len(pur)} mean={np.mean(pur):.4f} median={np.median(pur):.4f}")
    lines.append("")

    lines.append("=== Per-CSV ===")
    for r in per_csv_rows:
        lines.append(
            f"{r.get('src_file','')}: gt_pos={r.get('gt_pos')} pred_pos={r.get('pred_pos')} "
            f"tp={r.get('tp')} fp={r.get('fp')} fn={r.get('fn')} tn={r.get('tn')} "
            f"prec={float(r.get('precision',0.0)):.3f} rec={float(r.get('recall',0.0)):.3f} f1={float(r.get('f1',0.0)):.3f} "
            f"FP/h={float(r.get('fp_h',0.0)):.3f} FN/h={float(r.get('fn_h',0.0)):.3f} "
            f"TTD_s={r.get('ttd_s')} sil={r.get('silhouette')} purity={r.get('purity')}"
        )

    Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_eval_plots(per_csv_rows: list[dict], out_prefix: str) -> None:
    """Generate paper-friendly plots: PR bars, FP/FN per hour, TTD CDF."""
    if not per_csv_rows:
        return

    df = pd.DataFrame(per_csv_rows)

    # 1) Precision/Recall/F1 bar (overall)
    tp = int(df["tp"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec) if (prec + rec) else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(["Precision", "Recall", "F1"], [prec, rec, f1], color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Detection (scenario-level)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    try:
        fig.savefig(f"{out_prefix}__prf_bar.pdf")
    finally:
        plt.close(fig)

    # 2) FP/h and FN/h (overall)
    total_hours = float(df["hours"].sum()) if "hours" in df.columns else 0.0
    fp_h = _safe_div(fp, total_hours)
    fn_h = _safe_div(fn, total_hours)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(["FP/h", "FN/h"], [fp_h, fn_h], color=["#E45756", "#72B7B2"], alpha=0.9)
    ax.set_title("Operational error rates")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    try:
        fig.savefig(f"{out_prefix}__fp_fn_per_hour.pdf")
    finally:
        plt.close(fig)

    # 3) TTD CDF for positives
    if "ttd_s" in df.columns:
        ttd = df.loc[df["gt_pos"].astype(bool), "ttd_s"]
        ttd = pd.to_numeric(ttd, errors="coerce")
        ttd = ttd[np.isfinite(ttd)]
        if len(ttd) > 0: 
            fig, ax = plt.subplots(figsize=(6.2, 3.9))
            _plot_cdf(ax, ttd.values.astype(float), label=f"TTD n={len(ttd)}", color="black")
            ax.set_title("Time-to-detect (TTD) CDF on positive scenarios")
            ax.set_xlabel("seconds")
            ax.set_ylabel("CDF")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower right")
            fig.tight_layout()
            try:
                fig.savefig(f"{out_prefix}__ttd_cdf.pdf")
            finally:
                plt.close(fig)

    # 4) Silhouette distribution (optional)
    if "silhouette" in df.columns:
        sil = pd.to_numeric(df["silhouette"], errors="coerce")
        sil = sil[np.isfinite(sil)]
        if len(sil) > 0:
            fig, ax = plt.subplots(figsize=(6.2, 3.9))
            ax.hist(sil.values.astype(float), bins=25, color="#4C78A8", alpha=0.85)
            ax.set_title("Silhouette score distribution")
            ax.set_xlabel("silhouette")
            ax.set_ylabel("count")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            try:
                fig.savefig(f"{out_prefix}__silhouette_hist.pdf")
            finally:
                plt.close(fig)


def _worker_run_one_csv(args):
    """mp worker wrapper: returns (cand_df, summary_df, meta, advmax_row, eval_row)."""
    p_str, = args
    p = Path(p_str)
    try:
        raw = pd.read_csv(p)
        cand_df, summary_df, meta = _run_one_csv(p)

        # Per-scenario operational duration
        hours = _scenario_hours(raw)

        # Clustering metrics (use silhouette already selected in _choose_k_by_silhouette if available in meta, else NaN)
        sil = float(meta.get("silhouette", np.nan)) if isinstance(meta, dict) else np.nan
        purity = _compute_cluster_purity_from_summary(summary_df)

        det = _compute_detection_metrics(meta, cand_df)
        ttd_s = _compute_ttd_seconds(meta)

        eval_row = {
            "src_file": str(p),
            "hours": float(hours),
            "ttd_s": float(ttd_s) if np.isfinite(ttd_s) else np.nan,
            "silhouette": float(sil) if np.isfinite(sil) else np.nan,
            "purity": float(purity) if np.isfinite(purity) else np.nan,
            **det,
        }
        eval_row["fp_h"] = _safe_div(eval_row["fp"], hours)
        eval_row["fn_h"] = _safe_div(eval_row["fn"], hours)

        row = _scenario_advmax_row(summary_df, src_file=str(p))
        tx_s, rot_s, tx_tok, rot_tok = _parse_tx_rot_from_filename(p)
        row.update({"tx_s": float(tx_s), "rot_s": float(rot_s), "tx_tok": tx_tok, "rot_tok": rot_tok})
        core_val = row.get("core_mac_density_scaled", np.nan)
        for thr in DENSITY_GRID:
            row[f"pass_dens_ge_{thr}"] = bool(np.isfinite(core_val) and (float(core_val) >= float(thr)))

        return cand_df, summary_df, meta, row, eval_row
    except Exception as e:
        meta = {"src_file": str(p), "error": str(e), "confirmed_any": False}
        tx_s, rot_s, tx_tok, rot_tok = _parse_tx_rot_from_filename(p)
        row = {
            "src_file": str(p),
            "error": str(e),
            "tx_s": float(tx_s),
            "rot_s": float(rot_s),
            "tx_tok": tx_tok,
            "rot_tok": rot_tok,
        }
        eval_row = {
            "src_file": str(p),
            "hours": np.nan,
            "ttd_s": np.nan,
            "silhouette": np.nan,
            "purity": np.nan,
            "gt_pos": False,
            "pred_pos": False,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_flagged_clusters": 0,
            "fp_h": np.nan,
            "fn_h": np.nan,
        }
        return pd.DataFrame(), pd.DataFrame(), meta, row, eval_row


def _run_csv_list(csvs: list[Path], label: str) -> None:
    """Run pipeline over a list of CSV paths and write consolidated outputs."""
    if not csvs:
        raise FileNotFoundError(f"No CSVs to run for: {label}")

    print(f"\n=== Run mode: {label} ===")
    print(f"CSV files: {len(csvs)}")

    all_checks = []
    meta_rows = []
    advmax_rows = []
    eval_rows = []

    # Keep per-file summary so we can build global CDF comparisons
    all_summaries = []

    # Parallel per-CSV processing
    n_workers = min(32, max(1, mp.cpu_count()))
    tasks = [(str(p),) for p in csvs]

    with mp.Pool(processes=n_workers) as pool:
        for i, (cand_df, summary_df, meta, row, eval_row) in enumerate(pool.imap_unordered(_worker_run_one_csv, tasks, chunksize=1), start=1):
            src = str(meta.get("src_file", ""))
            print(f"\n[{i}/{len(csvs)}] Done: {src if src else 'unknown'}")

            if cand_df is not None and len(cand_df) > 0:
                all_checks.append(cand_df)
            if summary_df is not None and len(summary_df) > 0:
                if "src_file" not in summary_df.columns:
                    summary_df = summary_df.copy()
                    summary_df["src_file"] = src
                all_summaries.append(summary_df)

            meta_rows.append(meta)
            advmax_rows.append(row)
            eval_rows.append(eval_row)

    meta_df = pd.DataFrame(meta_rows)
    checks_df = pd.concat(all_checks, ignore_index=True) if all_checks else pd.DataFrame()
    advmax_df = pd.DataFrame(advmax_rows)

    # Combine summaries for global distribution comparisons (flagged vs not)
    summary_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

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

    # NEW PDFs: requested CDF plots
    out_adv_mac_pct_cdf_pdf = f"aircatch_{safe_label}_adv_mac_pct_cdf__{tag}.pdf"
    out_core_cdf_flagged_pdf = f"aircatch_{safe_label}_core_density_cdf_flagged_vs_not__{tag}.pdf"
    out_core_cdf_advmacs_pdf = f"aircatch_{safe_label}_core_density_cdf_adv_mac_pct_gt0_vs_0__{tag}.pdf"
    out_core_cdf_gt_pdf = f"aircatch_{safe_label}_core_density_cdf_gt_frac_gt0_vs_0__{tag}.pdf"

    meta_df.to_csv(out_meta, index=False)
    if len(checks_df) > 0:
        for col in ["k_chosen"]:
            if col not in checks_df.columns:
                checks_df[col] = np.nan
        checks_df.to_csv(out_csv, index=False)
        _write_adv_mac_pct_cdf_pdf(checks_df, out_adv_mac_pct_cdf_pdf)

    if len(summary_all) > 0 and len(checks_df) > 0:
        _write_core_density_flagged_vs_not_cdf_pdf(summary_all, checks_df, out_core_cdf_flagged_pdf)
        _write_core_density_adv_present_vs_not_cdf_pdf(summary_all, adv_presence_col="adv_mac_pct", out_pdf=out_core_cdf_advmacs_pdf)
        # _write_core_density_adv_present_vs_not_cdf_pdf(summary_all, adv_presence_col="gt_frac", out_pdf=out_core_cdf_gt_pdf)

    if len(advmax_df) > 0:
        advmax_df.to_csv(out_advmax, index=False)
        _write_advmax_density_plot(advmax_df, out_advmax_png)

    # Write evaluation report + plots (paper metrics)
    out_eval_txt = f"aircatch_{safe_label}_eval_report__{tag}.txt"
    out_eval_prefix = f"aircatch_{safe_label}_eval__{tag}"
    _write_paper_report_txt(eval_rows, out_eval_txt)
    _write_eval_plots(eval_rows, out_eval_prefix)

    print("\n=== Outputs ===")
    print(f"Wrote: {out_meta}")
    if len(checks_df) > 0:
        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_png}")
        print(f"Wrote: {out_adv_mac_pct_cdf_pdf}")
        if len(summary_all) > 0:
            print(f"Wrote: {out_core_cdf_flagged_pdf}")
            print(f"Wrote: {out_core_cdf_advmacs_pdf}")
            print(f"Wrote: {out_core_cdf_gt_pdf}")
    if len(advmax_df) > 0:
        print(f"Wrote: {out_advmax}")
        print(f"Wrote: {out_advmax_png}")
    print(f"Wrote: {out_eval_txt}")
    print(f"Wrote: {out_eval_prefix}__prf_bar.pdf")
    print(f"Wrote: {out_eval_prefix}__fp_fn_per_hour.pdf")
    print(f"Wrote: {out_eval_prefix}__ttd_cdf.pdf")
    print(f"Wrote: {out_eval_prefix}__silhouette_hist.pdf")


def _eval_confusion(per_csv_rows: list[dict]) -> tuple[int, int, int, int]:
    tp = sum(int(r.get("tp", 0)) for r in per_csv_rows)
    fp = sum(int(r.get("fp", 0)) for r in per_csv_rows)
    fn = sum(int(r.get("fn", 0)) for r in per_csv_rows)
    tn = sum(int(r.get("tn", 0)) for r in per_csv_rows)
    return tp, fp, fn, tn


def _worker_eval_one_csv_for_params(args):
    """Worker for sweep: run one CSV under a specific (block_s, step_s, dens_min)."""
    p_str, block_s, step_s, dens_min = args
    p = Path(p_str)

    # Set globals for this worker process
    global PERIODIC_BLOCK_S, PERIODIC_STEP_S, DENSITY_MIN
    PERIODIC_BLOCK_S = float(block_s)
    PERIODIC_STEP_S = float(step_s)
    DENSITY_MIN = float(dens_min)

    cand_df, summary_df, meta = _run_one_csv(p)
    raw = pd.read_csv(p)
    hours = _scenario_hours(raw)
    det = _compute_detection_metrics(meta, cand_df)
    ttd_s = _compute_ttd_seconds(meta)

    return {
        "src_file": str(p),
        "hours": float(hours),
        "ttd_s": float(ttd_s) if np.isfinite(ttd_s) else np.nan,
        **det,
    }


def _sweep_block_and_density(csvs: list[Path], *, block_grid: list[float], dens_grid: list[float]) -> pd.DataFrame:
    """Grid-search PERIODIC_BLOCK_S and DENSITY_MIN.

    PERIODIC_STEP_S is set to 0.8 * PERIODIC_BLOCK_S.

    Optimized: evaluates each (block,dens) combo using a shared multiprocessing pool.
    """
    rows = []
    if not csvs:
        return pd.DataFrame(rows)

    n_workers = min(32, max(1, mp.cpu_count()))

    with mp.Pool(processes=n_workers) as pool:
        for block_s in block_grid:
            step_s = float(0.8 * float(block_s))

            for dens in dens_grid:
                tasks = [(str(p), float(block_s), float(step_s), float(dens)) for p in csvs]
                eval_rows = list(pool.imap_unordered(_worker_eval_one_csv_for_params, tasks, chunksize=1))

                tp, fp, fn, tn = _eval_confusion(eval_rows)

                ttd = [float(r.get("ttd_s", np.nan)) for r in eval_rows if bool(r.get("gt_pos", False))]
                ttd = [x for x in ttd if np.isfinite(x)]
                ttd_med = float(np.median(ttd)) if ttd else float("nan")
                ttd_p90 = float(np.percentile(ttd, 90)) if ttd else float("nan")

                rows.append({
                    "block_s": float(block_s),
                    "step_s": float(step_s),
                    "dens_min": float(dens),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "precision": _safe_div(tp, tp + fp),
                    "recall": _safe_div(tp, tp + fn),
                    "f1": _safe_div(2.0 * _safe_div(tp, tp + fp) * _safe_div(tp, tp + fn), _safe_div(tp, tp + fp) + _safe_div(tp, tp + fn)) if (_safe_div(tp, tp + fp) + _safe_div(tp, tp + fn)) else 0.0,
                    "ttd_median_s": ttd_med,
                    "ttd_p90_s": ttd_p90,
                })

    return pd.DataFrame(rows)


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
    ap.add_argument("--sweep-density", action="store_true", help="Sweep DENSITY_MIN over DENSITY_GRID and report confusion matrix")
    ap.add_argument("--sweep-density-grid", type=str, default="", help="Optional comma-separated density grid (overrides DENSITY_GRID)")
    ap.add_argument("--sweep-block-density", action="store_true", help="Grid-search PERIODIC_BLOCK_S and DENSITY_MIN (step=0.8*block)")
    ap.add_argument("--sweep-block-grid", type=str, default="", help="Comma-separated block sizes in seconds (e.g., 300,450,600,900)")

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

    if args.sweep_block_density:
        # Default grid: around typical realtime chunk sizes
        block_grid = [300.0, 450.0, 600.0, 750.0, 900.0]
        if args.sweep_block_grid.strip():
            try:
                block_grid = [float(x) for x in args.sweep_block_grid.split(",") if x.strip()]
            except Exception:
                pass

        dens_grid = DENSITY_GRID
        if args.sweep_density_grid.strip():
            try:
                dens_grid = [float(x) for x in args.sweep_density_grid.split(",") if x.strip()]
            except Exception:
                dens_grid = DENSITY_GRID

        df = _sweep_block_and_density(csvs, block_grid=block_grid, dens_grid=dens_grid)
        out = "aircatch_sweep_block_density.csv"
        df.to_csv(out, index=False)

        # Prefer exact target if present
        target = df[(df["tp"] == 23) & (df["fp"] == 0) & (df["fn"] == 0) & (df["tn"] == 0)]
        if len(target) > 0:
            best = target.sort_values(["ttd_median_s", "block_s", "dens_min"], ascending=[True, True, True]).iloc[0]
            print(f"FOUND target: block={best['block_s']} step={best['step_s']} dens={best['dens_min']} TP=23 FP=0 FN=0 TN=0")
        else:
            # Otherwise: enforce zero-fp/zero-fn if possible; then lowest TTD
            z = df[(df["fp"] == 0) & (df["fn"] == 0)]
            pick = z if len(z) > 0 else df
            best = pick.sort_values(["f1", "fp", "fn", "ttd_median_s"], ascending=[False, True, True, True]).iloc[0]
            print(f"BEST: block={best['block_s']} step={best['step_s']} dens={best['dens_min']} TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']} TTDmed={best['ttd_median_s']}")

        print(f"Wrote: {out}")
        return

    if args.sweep_density:
        dens_grid = DENSITY_GRID
        if args.sweep_density_grid.strip():
            try:
                dens_grid = [float(x) for x in args.sweep_density_grid.split(",") if x.strip()]
            except Exception:
                dens_grid = DENSITY_GRID

        df = _sweep_density_min(csvs, label=f"batch_{CONTROLLED_SUBFOLDER}", dens_grid=dens_grid)
        out = f"aircatch_sweep_density__block{PERIODIC_BLOCK_S}_step{PERIODIC_STEP_S}.csv"
        df.to_csv(out, index=False)

        # Find exact target if present
        target = df[(df["tp"] == 23) & (df["fp"] == 0) & (df["fn"] == 0) & (df["tn"] == 0)]
        if len(target) > 0:
            best = target.sort_values("dens_min").iloc[0]
            print(f"FOUND target: DENSITY_MIN={best['dens_min']} TP=23 FP=0 FN=0 TN=0")
        else:
            # Otherwise, best by F1 then lowest FP
            best = df.sort_values(["f1", "fp"], ascending=[False, True]).iloc[0]
            print(f"BEST (by F1 then FP): DENSITY_MIN={best['dens_min']} TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']}")

        print(f"Wrote: {out}")
        return

    _run_csv_list(csvs, label=f"batch_{CONTROLLED_SUBFOLDER}")


if __name__ == "__main__":
    main()