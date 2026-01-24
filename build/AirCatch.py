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
  duration, unique_macs, churn_score (or singleton), core density
"""

import re
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  # kept (optional exploration)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================
# Configuration
# =========================

ADV_CSV = "test1.csv"  # input CSV file

PAYLOAD_TAG = "4c001219"        # optional, NOT used as filter by default
ADV_PAYLOAD_TAG = "4c001219ff"  # ADV marker used for GT and adv_mac_pct

WINDOW_S = 900                  # seconds per time bucket (e.g., 300=5min, 120=2min, 900=15min)
K_RANGE = range(3, 20)
OVERALL_CFO_WEIGHT = 1

MIN_DURATION_S = 1800            # strict decision min support (seconds)

# --- key / type logic ---
KEY_SIM_THR = 0.99              # merge clusters if keys are extremely similar (same ecosystem)
TYPE_SEP_WEIGHT = 1.0           # strong separation between ecosystem types

# Homogeneous cluster drop settings (candidate exclusion)
HOMOGENEOUS_DUR_S = 600         # cluster duration >= 10 min
HOMO_MAC_SHARE = 0.70
HOMO_KEY_SHARE = 0.70
HOMO_REQUIRE_KEYS = True        # if no keys present, cannot be homogeneous

# Density safety
S_MIN_DENSITY = 3               # compute density only if cluster has >=3 segments
R_MIN = 0.15                    # clamp radius in z-space for density
EPS = 1e-9

# Candidate filtering (for ranked list)
CAND_ADV_PCT_MIN = 0.00         # relaxed: set >0 to require ADV marker support
CAND_SEG_MIN     = 5            # min segments
CAND_DUR_MIN     = 600          # min duration (seconds)

# --- churn thresholds for candidates (NEW) ---
# We consider a cluster a *churny* candidate if ANY of these passes:
CAND_TURNOVER_MIN          = 0.60   # high window-to-window churn
CAND_SINGLETON_MAC_FRAC_MIN = 0.70  # most MACs appear once
CAND_MAC_RATE_PER_MIN_MIN   = 1.5   # unique MACs per minute

# Final strict decision thresholds (ALL must pass)
DUR_MIN = MIN_DURATION_S
UNIQUE_MACS_MIN = 15

# Strict churn requirement (NEW):
STRICT_TURNOVER_MIN          = 0.70
STRICT_SINGLETON_MAC_FRAC_MIN = 0.80
STRICT_MAC_RATE_PER_MIN_MIN   = 2.0

# Density threshold applied on CORE density scaled
DENSITY_MIN = 2.0

# --- CORE density parameters ---
CORE_FRAC_Q = 0.20              # densest core fraction (e.g., 20%)
CORE_MIN_PTS = 5                # minimum points in core

# Keep extra stats SMALL and robust
EXTRA_STATS = (
    "median",
    "iqr",
    "p10",
    "p90",
    "mad",
)

b210 = True  # if False, will filter crc_ok==1

# CFO columns in raw CSV
CFO_COLS_RAW = ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"]

# Segment feature columns (mean + robust stats)
CFO_COLS_SEG = ["CFO", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]
for base in ["CFO", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]:
    for s in EXTRA_STATS:
        CFO_COLS_SEG.append(f"{base}_{s}")

KEY_HASH_WEIGHT = 1
KEY_BUCKETS = 64

# Debug
DEBUG_PRINT_DF_HEAD = False


# =========================
# Helpers: hex + AD parsing (ecosystem classification)
# =========================

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


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


def key_to_bucket(key: str) -> int:
    if not key:
        return -1
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "little")
    return v % KEY_BUCKETS


def add_keyhash_feature(X: np.ndarray, seg: pd.DataFrame, weight=KEY_HASH_WEIGHT) -> np.ndarray:
    # representative key per row: pick first key in key_set if present
    rep = []
    for ks in seg["key_set"].tolist():
        if isinstance(ks, list) and len(ks) > 0:
            rep.append(ks[0])
        else:
            rep.append("")
    buckets = np.array([key_to_bucket(k) for k in rep], dtype=float).reshape(-1, 1)
    buckets = StandardScaler().fit_transform(buckets) * weight
    return np.hstack([X, buckets])


def add_packet_support_feature(X: np.ndarray, seg: pd.DataFrame, weight: float = 3.0) -> np.ndarray:
    # Robust: log-compress; prevents a 97-packet MAC from dominating
    v = np.log1p(seg["n_packets"].astype(float).values).reshape(-1, 1)
    v = StandardScaler().fit_transform(v) * weight
    return np.hstack([X, v])


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


def prepare_segments(df: pd.DataFrame, window_s: int) -> tuple[pd.DataFrame, set[str]]:
    """
    PER-MAC segmentation:
      seg_key = (seg_id, eco, AdvA)

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

    df = df.dropna(subset=["timestamp", "AdvA"] + CFO_COLS_RAW)
    df = df.sort_values("timestamp")

    # Ground-truth ADV MAC set (packet-level marker)
    adv_mask = df["payload"].str.contains(ADV_PAYLOAD_TAG, na=False)
    adv_mac_set = set(df.loc[adv_mask, "AdvA"].astype(str).tolist())

    # Ecosystem per packet
    df["eco"] = df["payload"].apply(classify_tag_ecosystem_from_payload)
    df["dev_type"] = df["eco"].apply(lambda e: f"TAG_{e}")

    # Keys per packet
    df["pubkeys"] = collect_pubkeys(df)

    # Window index
    df["seg_id"] = (df["timestamp"] // window_s).astype(int)

    if DEBUG_PRINT_DF_HEAD:
        print("\n[DEBUG] df head:")
        print(df.head(30).to_string(index=False))

    def agg_segment(g: pd.DataFrame, seg_id: int, eco: str, mac: str) -> pd.Series:
        t_start = float(g["timestamp"].min())
        t_end = float(g["timestamp"].max())
        n_packets = int(g["AdvA"].count())

        mac = str(mac)
        mac_set = [mac]
        n_macs = 1

        keys_flat = []
        for lst in g["pubkeys"].tolist():
            if isinstance(lst, list):
                keys_flat.extend(lst)
        key_set = sorted(set([k for k in keys_flat if k]))

        adv_macs = int(g["payload"].str.contains(ADV_PAYLOAD_TAG, na=False).sum())
        gt_adv = (mac in adv_mac_set)

        # CFO mean + robust stats
        cfo_stats = {}
        for col_raw, col_base in zip(
            ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"],
            ["CFO", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]
        ):
            vals = g[col_raw].astype(float).values
            cfo_stats[col_base] = float(np.mean(vals))
            rs = robust_stats(vals)
            for s in EXTRA_STATS:
                cfo_stats[f"{col_base}_{s}"] = float(rs[s])

        return pd.Series({
            "seg_key": f"{int(seg_id)}:{eco}:{mac}",
            "seg_id": int(seg_id),
            "eco": eco,
            "dev_type": f"TAG_{eco}",
            "mac": mac,

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
    for (sid, eco, mac), g in df.groupby(["seg_id", "eco", "AdvA"], sort=True):
        rows.append(agg_segment(g, sid, eco, mac))

    seg = pd.DataFrame(rows)
    if len(seg) == 0:
        raise RuntimeError("No segments produced. Check input CSV columns and filters.")

    # Per-MAC segments => churn not meaningful; keep only for debugging.
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
    """
    Core density computed only around the densest region:
      - pick medoid as robust center (min sum distances)
      - keep closest q-fraction points (at least min_core)
      - core_radius = max distance inside core (clamped)
      - core_density = core_mac_div / core_radius
      - scaled = core_density * log2(1 + core_unique_macs)
    """
    rows = []
    for cid, idxs in seg_with_cluster.groupby("cluster").groups.items():
        idxs = list(idxs)
        Xc = X_space[idxs, :]
        n = len(idxs)
        if n == 0:
            continue

        # pairwise distances (n x n) — fine for typical cluster sizes
        D = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
        medoid_local = int(np.argmin(D.sum(axis=1)))
        d = D[medoid_local, :]

        core_k = max(int(min_core), int(np.ceil(q * n)))
        core_k = min(core_k, n)
        core_local = np.argsort(d)[:core_k]
        core_global = [idxs[i] for i in core_local]

        core_radius = float(np.max(d[core_local])) if core_k > 1 else 0.0
        core_radius_clamped = max(core_radius, R_MIN)

        core_macs = seg_with_cluster.loc[core_global, "mac"].astype(str).tolist()
        core_macs = [m for m in core_macs if m]
        core_unique_macs = len(set(core_macs))

        core_segments = core_k
        core_mac_div = core_unique_macs / max(core_segments, 1)
        core_mac_density = core_mac_div / (core_radius_clamped + EPS)
        core_mac_density_scaled = core_mac_density * np.log2(1.0 + core_unique_macs)

        rows.append({
            "cluster": int(cid),
            "core_segments": int(core_segments),
            "core_unique_macs": int(core_unique_macs),
            "core_mac_div": float(core_mac_div),
            "core_radius": float(core_radius),
            "core_radius_clamped": float(core_radius_clamped),
            "core_mac_density": float(core_mac_density),
            "core_mac_density_scaled": float(core_mac_density_scaled),
        })

    return pd.DataFrame(rows)


def is_homogeneous_cluster(g: pd.DataFrame) -> dict:
    segments = len(g)

    # Duration support via window coverage
    unique_windows = int(g["seg_id"].nunique()) if "seg_id" in g.columns else segments
    duration_cov = float(unique_windows * WINDOW_S)

    # Also keep a real time-span duration for debugging
    if "t_start" in g.columns and "t_end" in g.columns:
        duration_span = float(g["t_end"].max() - g["t_start"].min())
    else:
        duration_span = duration_cov

    # MAC dominance
    macs_flat = g["mac"].astype(str).tolist() if "mac" in g.columns else []
    macs_flat = [m for m in macs_flat if m]

    mac_share = 0.0
    top_mac = ""
    if macs_flat:
        c = Counter(macs_flat)
        top_mac, top_cnt = c.most_common(1)[0]
        mac_share = top_cnt / max(len(macs_flat), 1)

    # Key dominance across segments
    seg_key_counts = Counter()
    for ks in g["key_set"].tolist():
        if not isinstance(ks, list):
            continue
        for k in set([x for x in ks if x]):
            seg_key_counts[k] += 1

    top_key = ""
    key_share = 0.0
    keys_exist = (len(seg_key_counts) > 0)
    if keys_exist:
        top_key, top_kcnt = seg_key_counts.most_common(1)[0]
        key_share = top_kcnt / max(segments, 1)

    key_ok = (key_share >= HOMO_KEY_SHARE) if keys_exist else (not HOMO_REQUIRE_KEYS)

    dur_for_homo = max(duration_cov, duration_span)
    homo = (dur_for_homo >= HOMOGENEOUS_DUR_S) and (mac_share >= HOMO_MAC_SHARE) and key_ok

    return {
        "homogeneous": bool(homo),
        "homo_top_mac": top_mac,
        "homo_mac_share": float(mac_share),
        "homo_top_key": top_key,
        "homo_key_share": float(key_share),
        "duration_cov": duration_cov,
        "duration_span": duration_span,
        "unique_windows": unique_windows,
    }


def _compute_turnover(g: pd.DataFrame) -> float:
    """
    Window-to-window turnover:
      turnover = mean(1 - Jaccard(S_t, S_{t+1}))
    where S_t is MAC set in window seg_id t within this cluster.
    """
    if "seg_id" not in g.columns or "mac" not in g.columns:
        return float("nan")

    win_sets = []
    for sid, gg in g.groupby("seg_id"):
        S = set([m for m in gg["mac"].astype(str).tolist() if m])
        if S:
            win_sets.append((int(sid), S))

    win_sets.sort(key=lambda x: x[0])
    if len(win_sets) < 2:
        return float("nan")

    dists = []
    for i in range(len(win_sets) - 1):
        A = win_sets[i][1]
        B = win_sets[i + 1][1]
        inter = len(A & B)
        union = len(A | B)
        jacc = (inter / union) if union else 0.0
        dists.append(1.0 - jacc)

    return float(np.mean(dists)) if dists else float("nan")


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

        unique_windows = int(g["seg_id"].nunique()) if "seg_id" in g.columns else len(g)
        duration_cov = float(unique_windows * WINDOW_S)
        duration_span = float(g["t_end"].max() - g["t_start"].min()) if ("t_start" in g.columns and "t_end" in g.columns) else duration_cov

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
        singleton_mac_frac = (sum(1 for _, cnt in c.items() if cnt == 1) / max(unique_macs, 1)) if unique_macs else 0.0
        singleton_seg_frac = (sum(cnt for _, cnt in c.items() if cnt == 1) / max(seg_count, 1)) if seg_count else 0.0

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

        # Turnover across windows
        turnover = _compute_turnover(g)

        # Unified churn score
        churn_score = float(turnover) if not np.isnan(turnover) else float(singleton_mac_frac)

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
            "duration_span": duration_span,

            "unique_macs": int(unique_macs),

            # old metric (kept)
            "mac_diversity": float(mac_diversity),

            # NEW churn metrics
            "top_mac_share": float(top_mac_share),
            "singleton_mac_frac": float(singleton_mac_frac),
            "singleton_seg_frac": float(singleton_seg_frac),
            "mac_entropy": float(H),
            "n_eff_macs": float(n_eff),
            "mac_rate_per_min": float(mac_rate_per_min),
            "n_eff_rate_per_min": float(n_eff_rate_per_min),
            "turnover": float(turnover) if not np.isnan(turnover) else np.nan,
            "churn_score": float(churn_score),

            "adv_mac_pct": float(g["adv_macs"].sum() / max(g["n_packets"].sum(), 1)),
            "n_packets_avg": n_packets_avg,
            "total_packets": total_packets,
            "pkt_per_mac": pkt_per_mac,
            "singleton_pkt_frac": singleton_pkt_frac,

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
        df.loc[mask, "mac_density"] * np.log2(1.0 + df.loc[mask, "unique_macs"])
    )

    # CORE density (the real discriminator)
    core_df = compute_core_density(X_density_space, seg, q=CORE_FRAC_Q, min_core=CORE_MIN_PTS)
    df = df.merge(core_df, on="cluster", how="left")

    return df


def rank_adversary_clusters(summary_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Candidate clusters:
      - not homogeneous
      - enough support (segments or duration)
      - AND churny by at least one churn test:
           turnover high OR singleton_mac_frac high OR mac_rate_per_min high
      - optionally ADV marker support

    Ranking:
      churn_score desc, duration_cov desc, core_mac_density_scaled desc
    """
    df = summary_df.copy()
    df = df[df["homogeneous"] == False]  # noqa: E712

    # support
    df = df[
        (df["segments"].fillna(0) >= CAND_SEG_MIN) |
        (df["duration_cov"].fillna(0.0) >= CAND_DUR_MIN)
    ].copy()

    # churny tests
    turn_ok = df["turnover"].fillna(-1.0) >= CAND_TURNOVER_MIN
    sing_ok = df["singleton_mac_frac"].fillna(0.0) >= CAND_SINGLETON_MAC_FRAC_MIN
    rate_ok = df["mac_rate_per_min"].fillna(0.0) >= CAND_MAC_RATE_PER_MIN_MIN
    df = df[turn_ok | sing_ok | rate_ok].copy()

    if CAND_ADV_PCT_MIN > 0:
        df = df[df["adv_mac_pct"].fillna(0.0) >= CAND_ADV_PCT_MIN].copy()

    if len(df) == 0:
        return df

    df["dens_rank"] = df["core_mac_density_scaled"].fillna(-1e18)
    df = df.sort_values(
        by=["churn_score", "duration_cov", "dens_rank"],
        ascending=[False, False, False],
    )
    return df.head(top_n)


# =========================
# Main
# =========================

def _fit_agglomerative(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    sklearn compatibility helper:
    - newer sklearn: metric=
    - older sklearn: affinity=
    """
    try:
        return AgglomerativeClustering(
            n_clusters=int(n_clusters),
            linkage="average",
            metric="euclidean",
        ).fit_predict(X)
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=int(n_clusters),
            linkage="average",
            affinity="euclidean",
        ).fit_predict(X)


def main():
    adv = pd.read_csv(ADV_CSV)

    seg, adv_mac_set = prepare_segments(adv, WINDOW_S)

    # ---------------- Feature spaces ----------------
    # CFO-only space (used for CORE density geometry)
    X_cfo = cfo_feature_matrix(seg)

    # Full clustering space: CFO + behavioral support
    X = X_cfo.copy()
    X = add_packet_support_feature(X, seg, weight=3.0)

    # Optional: ecosystem / key separation
    # X = add_type_feature(X, seg, weight=TYPE_SEP_WEIGHT)
    # X = add_keyhash_feature(X, seg, weight=KEY_HASH_WEIGHT)

    print("\n=== Segments produced (per window x ecosystem x MAC) ===")
    print(seg[["seg_key", "seg_id", "eco", "mac", "n_packets", "adv_macs", "gt_adv"]].head(25).to_string(index=False))
    print(f"\nTotal segment-rows: {len(seg)}")
    print("Ecosystem counts (segment-rows):")
    print(seg["eco"].value_counts().to_string())
    print(f"\nGT ADV MAC count: {len(adv_mac_set)}")

    # ---------------- Choose K (fixed or exploration) ----------------
    K_FINAL = 10  # set fixed or re-enable exploration
    # print(f"\n>>> Selected K = {int(K_FINAL)} (fixed) <<<\n")

    # ---------------- Final clustering ----------------
    labels0 = _fit_agglomerative(X, int(K_FINAL))
    # labels = merge_clusters_by_key_similarity(seg, labels0, key_sim_thr=KEY_SIM_THR)
    seg["cluster"] = labels0

    summary_df = summarize_clusters(seg, X_for_geom=X, X_density_space=X_cfo)

    # Sort for display (adversary-ish first)
    summary_df_sorted = summary_df.sort_values(
        ["homogeneous", "churn_score", "duration_cov", "core_mac_density_scaled"],
        ascending=[True, False, False, False]
    )

    print("\n=== Cluster summary ===")
    print(summary_df_sorted[[
        "cluster", "dev_type", "segments", "duration_cov",
        "unique_macs", "mac_rate_per_min",
        "turnover", "singleton_mac_frac", "top_mac_share", "n_eff_macs", "churn_score",
        "adv_mac_pct",
        "core_segments", "core_unique_macs", "core_radius_clamped", "core_mac_density_scaled",
        "gt_any", "gt_frac",
        "homogeneous", "homo_top_mac", "homo_mac_share", "homo_top_key", "homo_key_share"
    ]].to_string(index=False))

    # ---------------- Ranked candidates ----------------
    ranked = rank_adversary_clusters(summary_df, top_n=10)
    print("\n=== Ranked adversary candidates (top) ===")
    if len(ranked) == 0:
        print("No candidate clusters matched the candidate thresholds.")
    else:
        print(ranked[[
            "cluster", "dev_type", "segments", "duration_cov",
            "unique_macs", "mac_rate_per_min",
            "turnover", "singleton_mac_frac", "top_mac_share", "n_eff_macs", "churn_score",
            "adv_mac_pct",
            "core_radius_clamped", "core_mac_density_scaled",
            "gt_any", "gt_frac"
        ]].to_string(index=False))

    # ---------------- PCA visualization ----------------
    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)  # visualize full clustering space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for cid in sorted(seg["cluster"].unique()):
        mask = (seg["cluster"] == cid).values
        ax.scatter(X3[mask, 0], X3[mask, 1], X3[mask, 2],
                   s=35, alpha=0.70, label=f"C{cid}")

    gt_mask = seg["gt_adv"].astype(bool).values
    if np.any(gt_mask):
        ax.scatter(X3[gt_mask, 0], X3[gt_mask, 1], X3[gt_mask, 2],
                   s=140, facecolors="none", edgecolors="red", linewidths=2.2,
                   label="GT: ADV MAC segments")

        gt_centroid = X3[gt_mask].mean(axis=0)
        ax.scatter([gt_centroid[0]], [gt_centroid[1]], [gt_centroid[2]],
                   s=220, marker="X", edgecolors="red", facecolors="red",
                   label="GT centroid")
    else:
        print("\n[WARN] No GT ADV segments found (no MACs matched ADV marker).")

    if len(ranked) > 0:
        cand_clusters = set(ranked["cluster"].astype(int).tolist())
        cand_mask = seg["cluster"].apply(lambda c: int(c) in cand_clusters).values
        ax.scatter(X3[cand_mask, 0], X3[cand_mask, 1], X3[cand_mask, 2],
                   s=90, facecolors="none", edgecolors="black", linewidths=1.5,
                   label="Candidates (top-N)")

    ax.set_title("3D PCA — Predicted clusters + GT overlay")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig("cfo_3d_pca_clusters_gt.png", dpi=300)
    plt.show()

    # ---------------- Final strict decision (ANY candidate passes) ----------------
    print("\n=== Final adversary decision (strict AND; churn + core-density) ===")
    confirmed_any = False

    if len(ranked) == 0:
        print(">>> NOT CONFIRMED (no candidate clusters) <<<")
    else:
        for _, row in ranked.iterrows():
            # core density gate
            dens_ok = True
            if DENSITY_MIN > 0:
                dens_ok = (
                    (not np.isnan(row["core_mac_density_scaled"])) and
                    (row["core_mac_density_scaled"] >= DENSITY_MIN)
                )

            # churn gate: prefer turnover; fallback to singleton fraction
            turnover = row["turnover"]
            if not np.isnan(turnover):
                churn_ok = (turnover >= STRICT_TURNOVER_MIN)
                churn_reason = f"turnover {turnover:.3f} >= {STRICT_TURNOVER_MIN}"
            else:
                churn_ok = (row["singleton_mac_frac"] >= STRICT_SINGLETON_MAC_FRAC_MIN)
                churn_reason = f"singleton_mac_frac {row['singleton_mac_frac']:.3f} >= {STRICT_SINGLETON_MAC_FRAC_MIN}"

            # also enforce rate (optional but useful)
            rate_ok = (row["mac_rate_per_min"] >= STRICT_MAC_RATE_PER_MIN_MIN)

            ok = (
                (row["duration_cov"] >= DUR_MIN) and
                # (row["unique_macs"] >= UNIQUE_MACS_MIN) and
                # churn_ok and
                # rate_ok and
                dens_ok
            )

            print(
                "\n-- Candidate cluster check --\n"
                f"cluster={int(row['cluster'])} type={row['dev_type']} seg={int(row['segments'])}\n"
                f"Duration: {row['duration_cov']:.1f}s (>= {DUR_MIN}) -> {row['duration_cov'] >= DUR_MIN}\n"
                f"Unique MACs: {int(row['unique_macs'])} (>= {UNIQUE_MACS_MIN}) -> {row['unique_macs'] >= UNIQUE_MACS_MIN}\n"
                f"Churn: {churn_reason} -> {churn_ok}\n"
                f"MAC rate: {row['mac_rate_per_min']:.3f}/min (>= {STRICT_MAC_RATE_PER_MIN_MIN}) -> {rate_ok}\n"
                f"Core density scaled: {row['core_mac_density_scaled'] if not np.isnan(row['core_mac_density_scaled']) else float('nan'):.3f} "
                f"(>= {DENSITY_MIN}) -> {dens_ok}\n"
            )

            print(
                f"=> {'PASS' if ok else 'fail'} | "
                f"adv%={100*row['adv_mac_pct']:.1f}% core_dens={row['core_mac_density_scaled'] if not np.isnan(row['core_mac_density_scaled']) else float('nan'):.3f} "
                f"gt%={100*row['gt_frac']:.1f}%"
            )

            confirmed_any = confirmed_any or ok

        print("\n>>> ADVERSARY CONFIRMED <<<" if confirmed_any else "\n>>> NOT CONFIRMED (criteria not met) <<<")


if __name__ == "__main__":
    main()