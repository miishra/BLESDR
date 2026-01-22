#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirCatch.py — CFO-based adversary detection (key-aware, ecosystem-aware, per-MAC segmentation)

What this version changes (per your latest request):
- Segments are PER-MAC inside each time window:
    segment key = (seg_id, ecosystem, AdvA)
  This avoids mixing dozens of MACs into one row (bias / contamination).

- Still “stop collapsing ecosystems”:
    ecosystem = APPLE / GOOGLE / TILE / SAMSUNG / UNKNOWN
  You get per-MAC-per-ecosystem-per-window samples.

- Density / radius safety:
    * density only computed when cluster has >= S_MIN_DENSITY segments
    * radius is clamped: radius_clamped = max(radius, R_MIN)

- Ranked candidate clusters:
    returns top-N candidate clusters (not just one).

Important consequence of per-MAC segmentation:
- segment-level churn = n_macs / n_packets is no longer meaningful (n_macs=1 always)
- Instead we use CLUSTER-LEVEL MAC diversity:
    mac_diversity = unique_macs / segments
  (proxy for “MAC churn inside cluster”)

PCA plot:
- Points = segments in CFO PCA space
- Colors = predicted clusters
- Ground truth overlay = ALL segments whose MAC is in the GT ADV MAC set
  (GT MACs = MACs that appear in packets where payload contains ADV_PAYLOAD_TAG)
- Also plots GT centroid marker for visual “GT CFO cluster center”.

Input CSV must contain:
  timestamp, AdvA, payload, CFO_Hz, CFO_00_Hz, CFO_11_Hz, CFO_10_Hz, CFO_01_Hz
"""

import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import hashlib


# =========================
# Configuration
# =========================

ADV_CSV = "ble_dump_jan18.csv"  # input CSV file

PAYLOAD_TAG = "4c001219"        # optional, NOT used as filter by default
ADV_PAYLOAD_TAG = "4c001219ff"  # ADV marker used for GT and adv_mac_pct

WINDOW_S = 600                  # seconds per time bucket (e.g., 300=5min, 120=2min, 900=15min)
K_RANGE = range(3, 20)
OVERALL_CFO_WEIGHT = 1

MIN_DURATION_S = 600            # strict decision min support (seconds)

# --- key / type logic ---
KEY_SIM_THR = 0.99              # merge clusters if keys are extremely similar (same ecosystem)
TYPE_SEP_WEIGHT = 1.0           # strong separation between ecosystem types

# Homogeneous cluster drop settings (candidate exclusion)
HOMOGENEOUS_DUR_S = 600         # cluster duration >= 10 min
HOMO_MAC_SHARE = 0.90
HOMO_KEY_SHARE = 0.90
HOMO_REQUIRE_KEYS = True

# Density safety
S_MIN_DENSITY = 3               # compute density only if cluster has >=3 segments
R_MIN = 0.15                    # clamp radius in z-space for density
EPS = 1e-9

# Candidate filtering (for ranked list)
CAND_ADV_PCT_MIN = 0.00         # relaxed: set >0 to require ADV marker support
CAND_MACDIV_MIN  = 0.20         # mac_diversity threshold for candidates (tune)
CAND_SEG_MIN     = 5              # min segments for candidates
CAND_DUR_MIN     = 360.0

# Final strict decision thresholds (ALL must pass)
ADV_PCT_MIN = 0.20
MACDIV_MIN  = 0.80              # replaces CHURN_MIN in per-MAC segmentation regime
DUR_MIN     = MIN_DURATION_S
DENSITY_MIN = 0.0               # if >0, additional AND constraint (never alone)

# Keep extra stats SMALL and robust
EXTRA_STATS = (
    "median",
    "iqr",
    "p10",
    "p90",
    "mad",
)

# CFO columns
CFO_COLS_RAW = ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"]
# CFO_COLS_SEG = ["CFO_Hz", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]

CFO_COLS_SEG = [
    "CFO",
    "CFO_00",
    "CFO_11",
    "CFO_10",
    "CFO_01",
]

# Automatically expand robust stats
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
    """
    Compute small, robust statistics for a 1D array.
    Returns zeros if insufficient data.
    """
    if x is None or len(x) == 0:
        return {
            "median": 0.0,
            "iqr": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "mad": 0.0,
        }

    x = np.asarray(x, dtype=float)

    med = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return {
        "median": med,
        "iqr": q3 - q1,
        "p10": np.percentile(x, 10),
        "p90": np.percentile(x, 90),
        "mad": np.median(np.abs(x - med)),
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
    # unknown bucket (-1) => keep it as separate value
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
    """
    Key = first half of the cleaned payload hex.
    """
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

    df = df[df["crc_ok"] == 1]  # filter valid CRC only

    df = df.dropna(subset=["timestamp", "AdvA"] + CFO_COLS_RAW)
    df = df.sort_values("timestamp")

    # Ground-truth ADV MAC set (packet-level marker)
    adv_mask = df["payload"].str.contains(ADV_PAYLOAD_TAG, na=False)
    adv_mac_set = set(df.loc[adv_mask, "AdvA"].astype(str).tolist())

    # Ecosystem per packet
    df["eco"] = df["payload"].apply(classify_tag_ecosystem_from_payload)  # APPLE/GOOGLE/...
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

        # This group is per-MAC already
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

        # ---------------- CFO mean + robust stats ----------------
        cfo_stats = {}

        for col_raw, col_base in zip(
            ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"],
            ["CFO", "CFO_00", "CFO_11", "CFO_10", "CFO_01"]
        ):
            vals = g[col_raw].astype(float).values

            # Mean (existing behavior)
            cfo_stats[col_base] = float(np.mean(vals))

            # Extra robust stats
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
            "n_macs": n_macs,       # always 1 here
            "mac_set": mac_set,
            "key_set": key_set,

            **cfo_stats,   # <-- NEW: inject mean + robust stats

            "adv_macs": adv_macs,
            "gt_adv": bool(gt_adv),
        })

    rows = []
    for (sid, eco, mac), g in df.groupby(["seg_id", "eco", "AdvA"], sort=True):
        rows.append(agg_segment(g, sid, eco, mac))

    seg = pd.DataFrame(rows)
    if len(seg) == 0:
        raise RuntimeError("No segments produced. Check input CSV columns and filters.")

    # Per-MAC segments => this churn is not meaningful; keep only for debugging.
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
# Cluster stats
# =========================

def compute_cluster_geometry(X: np.ndarray, seg: pd.DataFrame) -> pd.DataFrame:
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


def is_homogeneous_cluster(g: pd.DataFrame) -> dict:
    segments = len(g)

    # ---- FIX: duration should not be sum(duration_est) for per-MAC segmentation ----
    # Coverage by unique windows (seg_id) is the right support measure here.
    unique_windows = int(g["seg_id"].nunique()) if "seg_id" in g.columns else segments
    duration_cov = float(unique_windows * WINDOW_S)

    # Also keep a real time-span duration for debugging
    if "t_start" in g.columns and "t_end" in g.columns:
        duration_span = float(g["t_end"].max() - g["t_start"].min())
    else:
        duration_span = duration_cov

    # ---- MAC dominance ----
    if "mac" in g.columns:
        macs_flat = g["mac"].astype(str).tolist()
    else:
        macs_flat = []
        for ms in g["mac_set"].tolist():
            if isinstance(ms, list):
                macs_flat.extend(ms)
    macs_flat = [m for m in macs_flat if m]

    mac_share = 0.0
    top_mac = ""
    if macs_flat:
        c = Counter(macs_flat)
        top_mac, top_cnt = c.most_common(1)[0]
        mac_share = top_cnt / max(len(macs_flat), 1)

    # ---- Key dominance across segments (presence across rows) ----
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

    # ---- FIX: use duration_cov (or duration_span) but NOT sum(duration_est) ----
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


def summarize_clusters(seg: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    geom = compute_cluster_geometry(X, seg)

    rows = []
    for cid, g in seg.groupby("cluster"):
        dt_mode = g["dev_type"].astype(str).mode()
        dev_type = str(dt_mode.iloc[0]) if len(dt_mode) else "UNKNOWN"

        # ---- FIX: duration support should be window coverage / time span ----
        unique_windows = int(g["seg_id"].nunique()) if "seg_id" in g.columns else len(g)
        duration_cov = float(unique_windows * WINDOW_S)
        duration_span = float(g["t_end"].max() - g["t_start"].min()) if ("t_start" in g.columns and "t_end" in g.columns) else duration_cov

        # cluster GT stats
        gt_any = bool(g["gt_adv"].any()) if "gt_adv" in g.columns else False
        gt_frac = float(g["gt_adv"].mean()) if "gt_adv" in g.columns else 0.0

        # per-MAC segmentation => churn proxy is MAC diversity in cluster
        macs_in_cluster = g["mac"].astype(str).tolist() if "mac" in g.columns else []
        uniq_macs = len(set([m for m in macs_in_cluster if m]))
        mac_diversity = uniq_macs / max(len(g), 1)

        total_packets = int(g["n_packets"].sum())
        singleton_frac = float((g["n_packets"] <= 1).mean())
        mac_rate = float(uniq_macs / max(duration_cov, 1.0))   # MACs per second
        pkt_per_mac = float(total_packets / max(uniq_macs, 1)) # avg packets per unique MAC

        row = {
            "cluster": int(cid),
            "dev_type": dev_type,
            "segments": int(len(g)),

            # new duration fields
            "unique_windows": unique_windows,
            "duration_cov": duration_cov,
            "duration_span": duration_span,

            "unique_macs": int(uniq_macs),
            "mac_diversity": float(mac_diversity),

            "adv_mac_pct": float(g["adv_macs"].sum() / max(g["n_packets"].sum(), 1)),
            "n_packets_avg": float(g["n_packets"].mean()),
            "gt_any": gt_any,
            "gt_frac": gt_frac,
            "total_packets": total_packets,
            "singleton_frac": singleton_frac,
            "mac_rate": mac_rate,
            "pkt_per_mac": pkt_per_mac,
        }

        row.update(is_homogeneous_cluster(g))
        rows.append(row)

    df = pd.DataFrame(rows).merge(geom, on="cluster", how="left")

    # density safety (based on mac_diversity)
    df["radius_clamped"] = df["radius"].fillna(0.0)
    mask = df["segments"] >= S_MIN_DENSITY
    df.loc[mask, "radius_clamped"] = df.loc[mask, "radius_clamped"].clip(lower=R_MIN)

    df["mac_density"] = np.nan
    df.loc[mask, "mac_density"] = df.loc[mask, "mac_diversity"] / (df.loc[mask, "radius_clamped"] + EPS)

    # --- NEW: scale-aware MAC density ---
    df["mac_density_scaled"] = np.nan
    mask = df["segments"] >= S_MIN_DENSITY

    df.loc[mask, "mac_density_scaled"] = (
        df.loc[mask, "mac_density"] *
        np.log2(1.0 + df.loc[mask, "unique_macs"])
    )
    return df

def rank_adversary_clusters(summary_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Candidate clusters:
      - not homogeneous
      - mac_diversity >= CAND_MACDIV_MIN
      - and (segments>=CAND_SEG_MIN OR duration>=CAND_DUR_MIN)

    Ranking:
      - mac_diversity desc
      - duration desc
      - mac_density desc (only meaningful for segments>=S_MIN_DENSITY)
    """
    df = summary_df.copy()
    df = df[df["homogeneous"] == False]  # noqa: E712

    df = df[
        (df["mac_diversity"].fillna(0.0) >= CAND_MACDIV_MIN) &
        ((df["segments"].fillna(0) >= CAND_SEG_MIN) | ((df["segments"] >= CAND_SEG_MIN) | (df["duration_cov"] >= CAND_DUR_MIN)))
    ].copy()

    if len(df) == 0:
        return df

    df["dens_rank"] = df["mac_density_scaled"].fillna(-1e18)
    df = df.sort_values(
        by=["mac_diversity", "duration_cov", "dens_rank"],
        ascending=[False, False, False],
    )
    return df.head(top_n)


# =========================
# Main
# =========================

def main():
    adv = pd.read_csv(ADV_CSV)

    seg, adv_mac_set = prepare_segments(adv, WINDOW_S)

    # Base CFO features
    X = cfo_feature_matrix(seg)

    # Behavioral support (VERY important)
    X = add_packet_support_feature(X, seg, weight=3.0)

    # Ecosystem separation
    # X = add_type_feature(X, seg, weight=TYPE_SEP_WEIGHT)

    # Key hash separation
    # X = add_keyhash_feature(X, seg, weight=KEY_HASH_WEIGHT)

    print("\n=== Segments produced (per window x ecosystem x MAC) ===")
    print(seg[["seg_key", "seg_id", "eco", "mac", "n_packets", "adv_macs", "gt_adv"]].head(25).to_string(index=False))
    print(f"\nTotal segment-rows: {len(seg)}")
    print("Ecosystem counts (segment-rows):")
    print(seg["eco"].value_counts().to_string())
    print(f"\nGT ADV MAC count: {len(adv_mac_set)}")

    # ---------------- Cluster count exploration ----------------
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

        labels0 = AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(X)
        labels = merge_clusters_by_key_similarity(seg, labels0, key_sim_thr=KEY_SIM_THR)

        seg_tmp = seg.copy()
        seg_tmp["cluster"] = labels

        n_labels0 = len(np.unique(labels0))
        sil = silhouette_score(X, labels0) if (2 <= n_labels0 < len(X)) else np.nan

        summary = summarize_clusters(seg_tmp, X)
        ranked1 = rank_adversary_clusters(summary, top_n=1)

        if len(ranked1) > 0:
            best = ranked1.iloc[0]
            best_type = str(best["dev_type"])
            best_adv = float(best["adv_mac_pct"])
            best_macdiv = float(best["mac_diversity"])
            best_dur = float(best["duration_cov"])
            best_rad = float(best["radius"])
            best_den = float(best["mac_density"]) if not np.isnan(best["mac_density"]) else np.nan
            best_gt = float(best["gt_frac"])
        else:
            best_type, best_adv, best_macdiv, best_dur, best_rad, best_den, best_gt = "NONE", 0.0, 0.0, 0.0, np.nan, np.nan, 0.0

        results.append({
            "K": int(k),
            "silhouette": float(sil) if not np.isnan(sil) else np.nan,
            "best_adv_mac_pct": best_adv,
            "best_duration": best_dur,
            "best_mac_div": best_macdiv,
            "best_radius": best_rad if not np.isnan(best_rad) else np.nan,
            "best_mac_density": best_den if not np.isnan(best_den) else np.nan,
            "labels_premerge": int(n_labels0),
            "labels_postmerge": int(summary["cluster"].nunique()),
            "best_type": best_type,
            "best_gt_frac": best_gt,
        })

        sil_str = f"{sil:.3f}" if not np.isnan(sil) else "NA"
        print(
            f"K={k:2d} | labels(pre)={n_labels0:2d} | labels(post)={int(summary['cluster'].nunique()):2d} | "
            f"sil={sil_str:>5} | best_type={best_type:>10} | "
            f"best_macdiv={best_macdiv:.3f} | best_adv%={100*best_adv:.1f}% | "
            f"best_dur={best_dur:.1f}s | best_rad={(best_rad if not np.isnan(best_rad) else 0.0):.3f} | "
            f"best_dens={(best_den if not np.isnan(best_den) else 0.0):.3f} | gt={100*best_gt:.1f}%"
        )

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        raise RuntimeError("No valid K tested. Reduce K_RANGE or ensure enough segments.")

    # ---------------- Choose K ----------------
    # Prefer duration, then mac_div, then silhouette
    K_FINAL = (
        res_df
        .assign(sil_filled=lambda d: d["silhouette"].fillna(-1))
        .sort_values(by=["sil_filled"], ascending=False)
        .iloc[0]["K"]
    )
    print(f"\n>>> Selected K = {int(K_FINAL)} <<<\n")

    # K_FINAL = 20  # or set to a fixed value based on exploration
    # print(f"\n>>> Selected K = {int(K_FINAL)} (fixed) <<<\n")

    # ---------------- Final clustering ----------------
    labels0 = AgglomerativeClustering(
        n_clusters=int(K_FINAL),
        linkage="average",
        metric="euclidean"
    ).fit_predict(X)
    labels = merge_clusters_by_key_similarity(seg, labels0, key_sim_thr=KEY_SIM_THR)
    seg["cluster"] = labels

    summary_df = summarize_clusters(seg, X)
    summary_df_sorted = summary_df.sort_values(
        ["homogeneous", "adv_mac_pct", "mac_diversity", "duration_cov"],
        ascending=[True, False, False, False]
    )

    print("=== Cluster summary ===")
    print(summary_df_sorted[[
        "cluster", "dev_type", "segments", "duration_cov",
        "unique_macs", "mac_diversity", "adv_mac_pct",
        "radius", "radius_clamped", "mac_density",
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
            "unique_macs", "mac_diversity", "adv_mac_pct",
            "radius", "radius_clamped", "mac_density",
            "gt_any", "gt_frac"
        ]].to_string(index=False))

    # ---------------- PCA visualization (CFO-only) + Ground Truth overlay ----------------
    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # plot predicted clusters
    for cid in sorted(seg["cluster"].unique()):
        mask = (seg["cluster"] == cid).values
        ax.scatter(X3[mask, 0], X3[mask, 1], X3[mask, 2],
                   s=35, alpha=0.70, label=f"C{cid}")

    # Ground truth: all segments whose MAC is in adv_mac_set
    gt_mask = seg["gt_adv"].astype(bool).values
    if np.any(gt_mask):
        ax.scatter(X3[gt_mask, 0], X3[gt_mask, 1], X3[gt_mask, 2],
                   s=140, facecolors="none", edgecolors="red", linewidths=2.2,
                   label="GT: ADV MAC segments")

        # GT centroid marker (visual “GT CFO cluster center”)
        gt_centroid = X3[gt_mask].mean(axis=0)
        ax.scatter([gt_centroid[0]], [gt_centroid[1]], [gt_centroid[2]],
                   s=220, marker="X", edgecolors="red", facecolors="red",
                   label="GT centroid")
    else:
        print("\n[WARN] No GT ADV segments found (no MACs matched ADV marker).")

    # Highlight candidate clusters (top-N) as black rings
    if len(ranked) > 0:
        cand_clusters = set(ranked["cluster"].astype(int).tolist())
        cand_mask = seg["cluster"].apply(lambda c: int(c) in cand_clusters).values
        ax.scatter(X3[cand_mask, 0], X3[cand_mask, 1], X3[cand_mask, 2],
                   s=90, facecolors="none", edgecolors="black", linewidths=1.5,
                   label="Candidates (top-N)")

    ax.set_title("3D PCA (CFO features) — Predicted clusters + GT ADV MAC overlay")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig("cfo_3d_pca_clusters_gt.png", dpi=300)
    plt.show()

    # ---------------- Final strict decision (ANY candidate passes) ----------------
    print("\n=== Final adversary decision (strict AND; no density-only) ===")
    confirmed_any = False

    if len(ranked) == 0:
        print(">>> NOT CONFIRMED (no candidate clusters) <<<")
    else:
        for _, row in ranked.iterrows():
            dens_ok = True
            if DENSITY_MIN > 0:
                dens_ok = (
                    not np.isnan(row["mac_density_scaled"]) and
                    row["mac_density_scaled"] >= DENSITY_MIN
                )

            UNIQUE_MACS_MIN = 15
            PKT_PER_MAC_MAX = 2.0
            SINGLETON_FRAC_MIN = 0.70

            ok = (
                (row["mac_diversity"] >= 0.80) and
                (row["duration_cov"] >= 600) and
                (row["unique_macs"] >= UNIQUE_MACS_MIN) and
                (row["pkt_per_mac"] <= PKT_PER_MAC_MAX) and
                (row["singleton_frac"] >= SINGLETON_FRAC_MIN) and
                dens_ok
            )

            print(
                f"cluster={int(row['cluster'])} type={row['dev_type']} "
                f"seg={int(row['segments'])} dur={row['duration_cov']:.1f}s "
                f"mac_div={row['mac_diversity']:.3f} adv%={100*row['adv_mac_pct']:.1f}% "
                f"dens={(row['mac_density_scaled'] if not np.isnan(row['mac_density_scaled']) else float('nan')):.3f} "
                f"gt%={100*row['gt_frac']:.1f}% "
                f"=> {'PASS' if ok else 'fail'}"
            )
            confirmed_any = confirmed_any or ok

        print("\n>>> ADVERSARY CONFIRMED <<<" if confirmed_any else "\n>>> NOT CONFIRMED (criteria not met) <<<")


if __name__ == "__main__":
    main()