#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirCatch — Streaming BLE tracker detection via online CFO clustering.

OPTION B (one-stage, persistent clusters):
  - Buffers packets per MAC until p packets -> finalize segment
  - Assign each segment to the best matching EXISTING CLUSTER among ALL clusters (no Δt window)
  - If no cluster passes γ, spawn a new cluster
  - Clusters are persistent (no pruning)

IMPORTANT FIXES (based on your debug summary):
  1) z-space warmup: avoid clustering while the scaler is still "cold"
  2) optional auto-gamma: learn γ from observed match distances
  3) raw-mode made usable: new clusters can start with a diagonal covariance prior (Hz² scale)

Also keeps:
  i)  Boxplot of CFO features per cluster on ONE concise plot
  ii) 3-D PCA scatter of segments colored by cluster
  iii) Bar plot: number of MACs per cluster

Saves plots in high quality (PDF + PNG) into OUTPUT_DIR.
"""

import os
import sys
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- config ---------------------------

FNAME = "cfo_samples_rail.csv"
OUTPUT_DIR = "all_devices_static_rail"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_TRACKLETS_JSONL = os.path.join(OUTPUT_DIR, "aircatch_tracklets.jsonl")
OUT_FLAGGED_JSONL = os.path.join(OUTPUT_DIR, "aircatch_flagged.jsonl")

# Plot outputs
OUT_BOX_PDF = os.path.join(OUTPUT_DIR, "aircatch_clusters_cfo_boxplot.pdf")
OUT_BOX_PNG = os.path.join(OUTPUT_DIR, "aircatch_clusters_cfo_boxplot.png")
OUT_PCA3D_PDF = os.path.join(OUTPUT_DIR, "aircatch_clusters_pca3d.pdf")
OUT_PCA3D_PNG = os.path.join(OUTPUT_DIR, "aircatch_clusters_pca3d.png")
OUT_MACBAR_PDF = os.path.join(OUTPUT_DIR, "aircatch_clusters_mac_counts.pdf")
OUT_MACBAR_PNG = os.path.join(OUTPUT_DIR, "aircatch_clusters_mac_counts.png")

# --- AirCatch params (Algorithm) ---
SEGMENT_SIZE_P = 50              # p
ASSOC_GAP_DT = 2.0               # kept for compatibility, NOT USED in option B
T_MIN = 30.0                    # T_min (seconds)
N_MIN = 1                        # N_min (segments)
K_MIN = 1                        # K_min (unique keys) (NOTE: keys are placeholder unless you parse real keys)
THETA = 4                        # θ (score threshold)  <-- CHANGED (was 3). With T_MIN=300, this stops "flag everything".
EPS = 1e-6                       # ε

# --------------------------- feature space choice ---------------------------

# If True: use standardized z-space for clustering. If False: use raw CFO Hz space.
USE_ZSPACE = True

# Initial gamma (used until AUTO_GAMMA learns a better one).
# In z-space, squared Mahalanobis typically sits around O(d) for matches.
GATE_GAMMA_Z_INIT = 0.5
GATE_GAMMA_RAW_INIT = 1.0e10  # raw is scale-dependent; recommended to keep AUTO_GAMMA on, or tune for your data.

# Scatter stability threshold differs by space
ETA_TRACE_MAX_Z = 8.0
ETA_TRACE_MAX_RAW = 5.0e9

# Require tag ecosystem filter?
REQUIRE_LOSTMODE_PREFIX = True

# Global normalization update cadence (only used if USE_ZSPACE=True)
GLOBAL_NORM_UPDATE_EVERY = 1
GLOBAL_NORM_MIN_STD = 1e-3

# Optional: freeze global scaler after N segment updates to prevent z-drift (recommended for z-space)
GLOBAL_NORM_FREEZE_AFTER = 120  # set None to never freeze

# OPTION B: clusters must remain persistent -> disable pruning
PRUNE_INACTIVE_AFTER = None      # MUST be None for persistent clusters

# --------------------------- NEW: warmup + auto-gamma ---------------------------

WARMUP_SEGS = 10                 # warm up scaler for first N segments (z-space)
AUTO_GAMMA = True                # learn gamma from match distances
AUTO_GAMMA_Q = 0.995             # quantile used as gamma
AUTO_GAMMA_MIN_MATCHES = 40      # start learning once we have enough matches
AUTO_GAMMA_MULT = 1.10           # safety margin

# Raw-mode diag covariance prior floor (Hz^2)
RAW_DIAG_FLOOR_HZ = 200.0        # min std per feature ~200 Hz

# CFO feature columns likely present
OPTIONAL_CFO_COLS = [
    "CFO_Hz", "cfo_hz", "cfo_quick_hz",
    # "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz",
    # "cfo_equal_00_hz", "cfo_equal_11_hz",
    # "cfo_jump_10_hz", "cfo_jump_01_hz",
]

# Prefer these 5 for plotting if present
PLOT_CFO_PREFERRED = ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"]

# --------------------------- utils: column resolution ---------------------------

def _col_lookup_case_insensitive(df: pd.DataFrame, name: str) -> str:
    target = str(name).lower()
    for c in df.columns:
        if str(c).lower() == target:
            return str(c)
    return ""

def resolve_first_present(df: pd.DataFrame, candidates: List[str]) -> str:
    for cand in candidates:
        hit = _col_lookup_case_insensitive(df, cand)
        if hit:
            return hit
    return ""

def resolve_time_column(df: pd.DataFrame) -> str:
    candidates = ["timestamp", "pcap_ts", "ts", "time", "t", "epoch", "epoch_s", "unix", "unix_ts"]
    cols = list(map(str, df.columns))
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(
        "Could not resolve timestamp column. "
        f"Available columns: {', '.join(cols[:50])}"
    )

def resolve_mac_column(df: pd.DataFrame) -> str:
    hit = resolve_first_present(df, ["AdvA", "adv_addr", "adva", "advA", "adv_address", "advaddr"])
    if hit:
        return hit
    for c in df.columns:
        lc = str(c).lower()
        if "adv" in lc and ("addr" in lc or "adva" in lc):
            return str(c)
    raise ValueError("Could not resolve MAC/AdvA column (try: AdvA/adv_addr).")

def resolve_payload_column(df: pd.DataFrame) -> str:
    return resolve_first_present(df, ["payload_hex", "payload", "pdu_hex", "pdu", "adv_data_hex", "adv_data"])

def resolve_cfo_feature_columns(df: pd.DataFrame) -> List[str]:
    present = []
    for cand in OPTIONAL_CFO_COLS:
        hit = _col_lookup_case_insensitive(df, cand)
        if hit and hit not in present:
            present.append(hit)
    if not present:
        for c in df.columns:
            lc = str(c).lower()
            if "cfo" in lc and "hz" in lc:
                present.append(str(c))
    if not present:
        raise ValueError("No CFO feature columns found (expected columns with CFO and Hz).")
    return present

# --------------------------- AirCatch: subroutines ---------------------------

def has_lostmode_prefix(payload_hex: str) -> bool:
    """
    Filters 'all kinds of Tags' by parsing AD structures inside the advertising payload.

    True if:
      - Apple Find My: AD type 0xFF, Apple company 0x004C, then 0x12 0x19
      - Tag ecosystems: AD type 0x16 (Service Data 16-bit UUID), UUID in {FEAA, FEED, FD5A, FD59}
    """
    if not isinstance(payload_hex, str):
        return False

    s = payload_hex.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    s = "".join(ch for ch in s if ch in "0123456789abcdef")
    if len(s) < 2:
        return False
    try:
        b = bytes.fromhex(s)
    except ValueError:
        return False

    TAG_SERVICE_UUIDS = {0xFEAA, 0xFEED, 0xFD5A, 0xFD59}

    def parse_ad_structures(ad: bytes) -> bool:
        pos = 0
        n = len(ad)
        while pos + 1 <= n:
            length = ad[pos]
            if length == 0:
                break
            end = pos + 1 + length
            if end > n:
                break
            ad_type = ad[pos + 1]
            data = ad[pos + 2:end]

            if ad_type == 0xFF and len(data) >= 4:
                company_id = data[0] | (data[1] << 8)
                if company_id == 0x004C:
                    if data[2] == 0x12 and data[3] == 0x19:
                        return True

            if ad_type == 0x16 and len(data) >= 2:
                svc_uuid = data[0] | (data[1] << 8)
                if svc_uuid in TAG_SERVICE_UUIDS:
                    return True

            pos = end
        return False

    # Try both [AdvA(6)+AD] and [AD only]
    if len(b) >= 8:
        if len(b) >= 6 and parse_ad_structures(b[6:]):
            return True
        if parse_ad_structures(b):
            return True
    else:
        return parse_ad_structures(b)

    return False

def extract_public_key(payload_hex: str, mac: str) -> Optional[str]:
    """
    Placeholder "key". Replace this with real key extraction if you have it.
    """
    if not isinstance(payload_hex, str):
        return None
    s = payload_hex.strip().lower()
    s = s[2:] if s.startswith("0x") else s
    if len(s) < 8:
        return None
    core = s[:64] if len(s) >= 64 else s
    return f"k:{core}"

def estimate_cfo_features(row: pd.Series, cfo_cols: List[str]) -> np.ndarray:
    vals = []
    for c in cfo_cols:
        v = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        vals.append(float(v) if np.isfinite(v) else np.nan)
    return np.asarray(vals, dtype=float)

def finalize_segment(buffer_rows: List[Tuple[float, np.ndarray, Optional[str]]]) -> Tuple[float, float, np.ndarray, Set[str]]:
    ts = np.array([x[0] for x in buffer_rows], dtype=float)
    t_s = float(np.nanmin(ts))
    t_e = float(np.nanmax(ts))
    F = np.vstack([x[1] for x in buffer_rows])
    fbar = np.nanmean(F, axis=0)
    K_s = set([k for (_, _, k) in buffer_rows if isinstance(k, str) and k != ""])
    return t_s, t_e, fbar, K_s

# --------------------------- Online global standardizer ---------------------------

class OnlineGlobalScaler:
    def __init__(self, d: int, eps: float = EPS):
        self.d = int(d)
        self.eps = float(eps)
        self.n = 0
        self.mean = np.zeros(self.d, dtype=float)
        self.M2 = np.zeros(self.d, dtype=float)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,):
            raise ValueError("OnlineGlobalScaler.update: shape mismatch")
        if not np.all(np.isfinite(x)):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.d, dtype=float)
        v = self.M2 / max(1, (self.n - 1))
        return np.maximum(v, GLOBAL_NORM_MIN_STD**2)

    def std(self) -> np.ndarray:
        return np.sqrt(self.var())

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.mean) / (self.std() + self.eps)

# --------------------------- Cluster (Tracklet) ---------------------------

def _inv_psd(mat: np.ndarray, eps: float = EPS) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    m = 0.5 * (m + m.T)
    m = m + eps * np.eye(m.shape[0], dtype=float)
    return np.linalg.inv(m)

@dataclass
class Tracklet:
    """
    In Option B: Tracklet == persistent cluster state.
    """
    id: int
    d: int

    n: int = 0
    mu: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    M2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    t_min: float = math.inf
    t_max: float = -math.inf

    macs: Set[str] = field(default_factory=set)
    keys: Set[str] = field(default_factory=set)

    # NEW: raw-mode diagonal covariance prior (Hz^2 per feature)
    prior_diag: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mu.size == 0:
            self.mu = np.zeros(self.d, dtype=float)
        if self.M2.size == 0:
            self.M2 = np.zeros((self.d, self.d), dtype=float)

    def cov(self) -> np.ndarray:
        if self.n < 2:
            if (not USE_ZSPACE) and (self.prior_diag is not None):
                diag = np.asarray(self.prior_diag, dtype=float)
                diag = np.maximum(diag, (RAW_DIAG_FLOOR_HZ**2) * np.ones_like(diag))
                return np.diag(diag)
            return np.eye(self.d, dtype=float)
        C = self.M2 / max(1, (self.n - 1))
        return 0.5 * (C + C.T)

    def trace(self) -> float:
        return float(np.trace(self.cov()))

    def mahalanobis(self, x: np.ndarray, eps: float = EPS) -> float:
        x = np.asarray(x, dtype=float)
        diff = (x - self.mu)
        inv = _inv_psd(self.cov(), eps=eps)
        return float(diff.T @ inv @ diff)

    def update_stats_welford(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,):
            raise ValueError("Tracklet.update_stats_welford: shape mismatch")
        if not np.all(np.isfinite(x)):
            return
        self.n += 1
        delta = x - self.mu
        self.mu += delta / self.n
        delta2 = x - self.mu
        self.M2 += np.outer(delta, delta2)

    def merge_segment(self, t_s: float, t_e: float, mac: str, keys_in_seg: Set[str]) -> None:
        self.t_min = min(self.t_min, float(t_s))
        self.t_max = max(self.t_max, float(t_e))
        if isinstance(mac, str) and mac != "":
            self.macs.add(mac)
        for k in keys_in_seg:
            self.keys.add(k)

    def score(self) -> int:
        """
        Score uses space-dependent trace threshold.
        NOTE: key term only affects scoring/flagging; merging remains CFO-only.
        """
        T = self.t_max - self.t_min
        s = 0
        s += int(T >= T_MIN)
        s += int(self.n >= N_MIN)
        s += int(len(self.keys) >= K_MIN)
        thr = ETA_TRACE_MAX_Z if USE_ZSPACE else ETA_TRACE_MAX_RAW
        s += int(self.trace() <= thr)
        return s

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "n_segments": int(self.n),
            "t_min": float(self.t_min) if np.isfinite(self.t_min) else None,
            "t_max": float(self.t_max) if np.isfinite(self.t_max) else None,
            "duration": float(self.t_max - self.t_min) if (np.isfinite(self.t_min) and np.isfinite(self.t_max)) else None,
            "macs": sorted(list(self.macs)),
            "keys": sorted(list(self.keys)),
            "mu": self.mu.tolist(),           # mu in the *cluster space* (z-space if USE_ZSPACE else raw)
            "cov_trace": float(self.trace()), # trace in the *cluster space*
            "space": "z" if USE_ZSPACE else "raw",
            "score": int(self.score()),
        }

# --------------------------- AirCatch core ---------------------------

@dataclass
class AirCatchParams:
    p: int = SEGMENT_SIZE_P
    dt: float = ASSOC_GAP_DT     # not used in Option B
    gamma: float = 0.0
    theta: int = THETA
    eps: float = EPS

def aircatch_stream(df: pd.DataFrame,
                    time_col: str,
                    mac_col: str,
                    payload_col: str,
                    cfo_cols: List[str],
                    params: AirCatchParams) -> Tuple[List[Tracklet], List[Tracklet], pd.DataFrame]:
    """
    Runs AirCatch over the dataframe in timestamp order.
    Option B: candidates are ALL clusters (persistent).
    Returns: (all_clusters, flagged_clusters, segments_df)
    """
    tvals = pd.to_numeric(df[time_col], errors="coerce")
    df = df.assign(_t=tvals).replace([np.inf, -np.inf], np.nan).dropna(subset=["_t"])
    df = df.sort_values("_t").drop(columns=["_t"])

    d = len(cfo_cols)
    scaler = OnlineGlobalScaler(d=d, eps=params.eps)

    buffers: Dict[str, List[Tuple[float, np.ndarray, Optional[str]]]] = {}
    clusters: List[Tracklet] = []
    flagged_ids: Set[int] = set()

    seg_records: List[Dict[str, Any]] = []

    next_id = 1
    seg_count = 0

    # ---------------- debug knobs ----------------
    DEBUG = True
    DEBUG_EVERY_SEG = 1
    DEBUG_TOPK = 5
    DEBUG_SCALER_EVERY = 25
    # ---------------------------------------------

    # Debug counters
    n_rows = 0
    skip_tag = 0
    skip_bad_cfo_pkt = 0
    n_segments_total = 0
    n_spawn = 0
    n_match = 0
    d_best_list: List[float] = []
    spawned_d_list: List[float] = []
    matched_d_list: List[float] = []
    per_mac_segments: Dict[str, int] = {}

    # Live gamma (AUTO_GAMMA can raise it)
    gamma_live = float(params.gamma)
    match_d_hist: List[float] = []

    # Raw-mode diag variance estimate for priors
    raw_diag_var = np.ones(d, dtype=float)
    raw_diag_scaler = OnlineGlobalScaler(d=d, eps=params.eps)  # used only for var()

    # OPTION B: all clusters are always candidates
    def _candidates() -> List[Tracklet]:
        return clusters

    for _, row in df.iterrows():
        n_rows += 1

        t_k = float(pd.to_numeric(row[time_col], errors="coerce"))
        m_k = str(row[mac_col])

        p_k = row.get(payload_col, None) if payload_col else None

        # (filter) tag ecosystems
        if REQUIRE_LOSTMODE_PREFIX:
            if not (isinstance(p_k, str) and has_lostmode_prefix(p_k)):
                skip_tag += 1
                continue

        # Extract key (placeholder)
        K_k = extract_public_key(p_k, m_k) if isinstance(p_k, str) else None

        # CFO features per packet
        f_k = estimate_cfo_features(row, cfo_cols)
        if not np.all(np.isfinite(f_k)):
            skip_bad_cfo_pkt += 1
            continue

        # append into B[m_k]
        buffers.setdefault(m_k, []).append((t_k, f_k, K_k))
        if len(buffers[m_k]) < params.p:
            continue

        # finalize segment
        t_s, t_e, f_bar, K_s = finalize_segment(buffers[m_k])
        buffers[m_k].clear()

        if not np.all(np.isfinite(f_bar)):
            continue

        # ---------------- warmup / feature vector ----------------
        if USE_ZSPACE:
            # Warmup scaler on first WARMUP_SEGS segments: do not cluster yet
            if scaler.n < int(WARMUP_SEGS):
                scaler.update(f_bar)
                seg_count += 1
                continue
            x = scaler.transform(f_bar)  # z-space
        else:
            # Maintain a raw diag variance estimate from segment means
            raw_diag_scaler.update(f_bar)
            raw_diag_var = raw_diag_scaler.var()
            x = f_bar.copy()
        # ---------------------------------------------------------

        # candidate set (ALL clusters)
        C = _candidates()

        # choose best match + collect top-K distances for debug
        best = None
        d_best = float("inf")
        dist_list: List[Tuple[float, int]] = []

        if C:
            for tau in C:
                dist = tau.mahalanobis(x, eps=params.eps)
                if dist < d_best:
                    d_best = dist
                    best = tau
                if DEBUG:
                    dist_list.append((dist, tau.id))

        # gate/spawn
        spawned = False
        if (best is None) or (d_best > gamma_live):
            tau = Tracklet(id=next_id, d=d)
            next_id += 1
            if not USE_ZSPACE:
                tau.prior_diag = raw_diag_var.copy()
            clusters.append(tau)
            spawned = True
            n_spawn += 1
            spawned_d_list.append(d_best if np.isfinite(d_best) else float("inf"))
        else:
            tau = best
            n_match += 1
            matched_d_list.append(d_best)

            # --- auto gamma update from match distances ---
            if AUTO_GAMMA and np.isfinite(d_best):
                match_d_hist.append(float(d_best))
                if len(match_d_hist) >= int(AUTO_GAMMA_MIN_MATCHES):
                    q = float(np.quantile(np.asarray(match_d_hist, dtype=float), float(AUTO_GAMMA_Q)))
                    gamma_live = max(gamma_live, float(AUTO_GAMMA_MULT) * q)
            # ---------------------------------------------

        # update cluster stats (in chosen space)
        tau.update_stats_welford(x)
        tau.merge_segment(t_s=t_s, t_e=t_e, mac=m_k, keys_in_seg=K_s)

        # record per-segment assignment for plotting (raw CFO Hz segment means)
        rec = {
            "cluster_id": int(tau.id),
            "t_s": float(t_s),
            "t_e": float(t_e),
            "mac": m_k,
            "n_keys": int(len(K_s)),
        }
        for j, colname in enumerate(cfo_cols):
            rec[f"seg_{colname}"] = float(f_bar[j])
        seg_records.append(rec)

        # counters
        seg_count += 1
        n_segments_total += 1
        d_best_list.append(d_best)
        per_mac_segments[m_k] = per_mac_segments.get(m_k, 0) + 1

        # global scaler update (only if z-space)
        if USE_ZSPACE and (seg_count % GLOBAL_NORM_UPDATE_EVERY == 0):
            if (GLOBAL_NORM_FREEZE_AFTER is None) or (scaler.n < int(GLOBAL_NORM_FREEZE_AFTER)):
                scaler.update(f_bar)

        # pruning disabled for Option B (guarded)
        if PRUNE_INACTIVE_AFTER is not None:
            t_now = t_s
            keep: List[Tracklet] = []
            for tt in clusters:
                if tt.id in flagged_ids:
                    keep.append(tt)
                else:
                    if (t_now - tt.t_max) <= float(PRUNE_INACTIVE_AFTER):
                        keep.append(tt)
            clusters = keep

        # decision
        if tau.score() >= params.theta:
            flagged_ids.add(tau.id)

        # ---------------- debug prints ----------------
        if DEBUG and (n_segments_total % DEBUG_EVERY_SEG == 0):
            dur = float(t_e - t_s)
            print("-" * 90)
            print(f"[SEG {n_segments_total:>4}] mac={m_k}  seg_dur={dur:.3f}s  |K_s|={len(K_s)}  "
                  f"clusters_now={len(clusters)}  gamma={gamma_live:.3f}  space={'z' if USE_ZSPACE else 'raw'}")
            print(f"          decision={'SPAWN' if spawned else 'MATCH'} -> cluster_id={tau.id}  d_best={d_best:.3f}")

            if dist_list:
                dist_list.sort(key=lambda x: x[0])
                topk = dist_list[:max(0, int(DEBUG_TOPK))]
                topk_str = ", ".join([f"(id={cid}, d={dist:.2f})" for (dist, cid) in topk])
                print(f"          top{len(topk)} nearest: {topk_str}")
            else:
                print("          (no existing clusters yet)")

        if DEBUG and USE_ZSPACE and (seg_count % DEBUG_SCALER_EVERY == 0):
            frozen = (GLOBAL_NORM_FREEZE_AFTER is not None and scaler.n >= int(GLOBAL_NORM_FREEZE_AFTER))
            print(f"[scaler] n={scaler.n} mean={np.array2string(scaler.mean, precision=3)} "
                  f"std={np.array2string(scaler.std(), precision=3)} "
                  f"{'(FROZEN)' if frozen else ''}")
        # -----------------------------------------------

    flagged = [tau for tau in clusters if tau.id in flagged_ids]
    seg_df = pd.DataFrame(seg_records)

    # ---------------- final debug summary ----------------
    if DEBUG:
        print("\n" + "=" * 90)
        print("[debug summary]")
        print(f"rows_total                 : {n_rows}")
        print(f"skip_tag_filter            : {skip_tag}")
        print(f"skip_bad_cfo_packet         : {skip_bad_cfo_pkt}")
        print(f"segments_total             : {n_segments_total}")
        print(f"clusters_final             : {len(clusters)}")
        print(f"spawned_clusters           : {n_spawn}")
        print(f"matched_segments           : {n_match}")
        if n_segments_total > 0:
            print(f"spawn_rate                 : {100.0 * n_spawn / n_segments_total:.1f}%")

        if d_best_list:
            arr = np.array([x for x in d_best_list if np.isfinite(x)], dtype=float)
            if arr.size > 0:
                qs = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
                print("d_best quantiles           : "
                      f"p05={qs[0]:.2f} p25={qs[1]:.2f} p50={qs[2]:.2f} p75={qs[3]:.2f} p95={qs[4]:.2f}")
        if matched_d_list:
            arrm = np.array([x for x in matched_d_list if np.isfinite(x)], dtype=float)
            if arrm.size > 0:
                qs = np.quantile(arrm, [0.05, 0.50, 0.95])
                print("d_best (matches) quantiles : "
                      f"p05={qs[0]:.2f} p50={qs[1]:.2f} p95={qs[2]:.2f}")
        if spawned_d_list:
            arrs = np.array([x for x in spawned_d_list if np.isfinite(x)], dtype=float)
            if arrs.size > 0:
                qs = np.quantile(arrs, [0.05, 0.50, 0.95])
                print("d_best (spawns) quantiles  : "
                      f"p05={qs[0]:.2f} p50={qs[1]:.2f} p95={qs[2]:.2f}")

        if per_mac_segments:
            top = sorted(per_mac_segments.items(), key=lambda kv: -kv[1])[:10]
            print("top MACs by segments       : " + ", ".join([f"{m}:{c}" for m, c in top]))
        print("=" * 90 + "\n")
    # ------------------------------------------------------

    return clusters, flagged, seg_df

# --------------------------- plotting ---------------------------

def _resolve_plot_cfo_cols(seg_df: pd.DataFrame, cfo_cols: List[str]) -> List[str]:
    lower_to_actual = {c.lower(): c for c in cfo_cols}
    chosen_actual: List[str] = []
    for pref in PLOT_CFO_PREFERRED:
        hit = lower_to_actual.get(pref.lower(), "")
        if hit and hit not in chosen_actual:
            chosen_actual.append(hit)
        if len(chosen_actual) >= 5:
            break
    if not chosen_actual:
        chosen_actual = cfo_cols[:5]
    return [f"seg_{c}" for c in chosen_actual]

def plot_cluster_cfo_boxplot(seg_df: pd.DataFrame, cfo_cols: List[str], out_pdf: str, out_png: str) -> None:
    if seg_df is None or seg_df.empty:
        print("[!] No segments collected; skipping boxplot.")
        return

    plot_cols = _resolve_plot_cfo_cols(seg_df, cfo_cols)
    plot_cols = [c for c in plot_cols if c in seg_df.columns]
    if not plot_cols:
        print("[!] No segment CFO columns available for plotting; skipping boxplot.")
        return

    counts = seg_df["cluster_id"].value_counts().sort_index()
    cluster_ids = [int(cid) for cid in counts.index.tolist()]
    if len(cluster_ids) == 0:
        print("[!] No clusters; skipping boxplot.")
        return

    n_feat = len(plot_cols)
    fig_w = max(10.0, min(24.0, 0.5 * len(cluster_ids) + 6.0))
    fig_h = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    base_positions = np.arange(len(cluster_ids), dtype=float)
    offsets = np.linspace(-0.25, 0.25, n_feat) if n_feat > 1 else np.array([0.0])

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]

    handles, labels = [], []

    for fi, col in enumerate(plot_cols):
        data, positions = [], []
        for i, cid in enumerate(cluster_ids):
            vals = pd.to_numeric(seg_df.loc[seg_df["cluster_id"] == cid, col], errors="coerce").dropna().values
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(base_positions[i] + offsets[fi])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.12 if n_feat > 1 else 0.35,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=1.2),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        color = prop_cycle[fi % len(prop_cycle)]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)

        handles.append(bp["boxes"][0])
        labels.append(col.replace("seg_", ""))

    ax.set_title("Cluster-wise CFO feature distributions (segment means)")
    ax.set_ylabel("CFO (Hz)")
    ax.set_xlabel("Cluster ID")
    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(cid) for cid in cluster_ids], rotation=90)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    if handles:
        ax.legend(handles, labels, title="CFO feature", loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] saved boxplot: {out_pdf} and {out_png}")

def plot_clusters_pca3d(seg_df: pd.DataFrame, cfo_cols: List[str], out_pdf: str, out_png: str) -> None:
    if seg_df is None or seg_df.empty:
        print("[!] No segments collected; skipping PCA plot.")
        return

    seg_feature_cols = [f"seg_{c}" for c in cfo_cols if f"seg_{c}" in seg_df.columns]
    if len(seg_feature_cols) < 2:
        print("[!] Not enough features for PCA; skipping PCA plot.")
        return

    X = seg_df[seg_feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y = seg_df["cluster_id"].values.astype(int)

    ok = np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    if X.shape[0] < 5:
        print("[!] Too few finite segments for PCA; skipping PCA plot.")
        return

    # offline standardization for visualization only
    mu = np.mean(X, axis=0)
    sig = np.std(X, axis=0, ddof=1)
    sig = np.maximum(sig, GLOBAL_NORM_MIN_STD)
    Xz = (X - mu) / sig

    pca = PCA(n_components=3, random_state=0)
    Xp = pca.fit_transform(Xz)

    counts = pd.Series(y).value_counts()
    top_clusters = set(counts.head(12).index.tolist())

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("tab20")
    unique_clusters = np.unique(y)

    for cid in unique_clusters:
        mask = (y == cid)
        pts = Xp[mask]
        if pts.shape[0] == 0:
            continue
        color = cmap(int(cid) % 20)
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=10, alpha=0.7,
            color=color,
            label=str(cid) if cid in top_clusters else None
        )

    evr = pca.explained_variance_ratio_
    ax.set_title(f"Clusters in 3-D PCA space (EVR: {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    if len(top_clusters) > 0:
        ax.legend(title="Top clusters", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] saved PCA 3D: {out_pdf} and {out_png}")

def plot_cluster_mac_counts(tracklets: List[Tracklet], out_pdf: str, out_png: str) -> None:
    if not tracklets:
        print("[!] No clusters; skipping MAC-count bar plot.")
        return

    ids = [t.id for t in sorted(tracklets, key=lambda x: x.id)]
    mac_counts = [len(t.macs) for t in sorted(tracklets, key=lambda x: x.id)]

    fig_w = max(10.0, min(24.0, 0.45 * len(ids) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax.bar([str(i) for i in ids], mac_counts)

    ax.set_title("Number of unique MACs per cluster")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("# MACs")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(axis="x", labelrotation=90)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] saved MAC-count bar: {out_pdf} and {out_png}")

def print_cluster_stats(tracklets: List[Tracklet], max_macs: int = 20, max_keys: int = 20) -> None:
    """
    Prints:
      - number of clusters
      - per-cluster: n_segments, duration, score, cov_trace
      - mean feature vector (mu) in z-space (standardized space)
      - MACs + keys contained (truncated)
    """
    if not tracklets:
        print("[stats] No clusters/tracklets to report.")
        return

    # Sort: highest score, then longest duration, then most segments
    tracklets_sorted = sorted(
        tracklets,
        key=lambda t: (-t.score(), -(t.t_max - t.t_min), -t.n, t.id),
    )

    print("\n" + "=" * 80)
    print(f"[stats] Clusters total: {len(tracklets_sorted)}")
    print("=" * 80)

    for tau in tracklets_sorted:
        info = tau.to_jsonable()

        macs = info["macs"] or []
        keys = info["keys"] or []

        macs_show = macs[:max_macs]
        keys_show = keys[:max_keys]

        macs_more = "" if len(macs) <= max_macs else f" ... (+{len(macs)-max_macs} more)"
        keys_more = "" if len(keys) <= max_keys else f" ... (+{len(keys)-max_keys} more)"

        mu_str = ", ".join([f"{v:+.3f}" for v in info["mu"]])

        print(f"\n--- Cluster id={info['id']} ---")
        print(f"  score      : {info['score']}  (T>={T_MIN}s, N>={N_MIN}, |K|>={K_MIN}, tr<={ETA_TRACE_MAX})")
        print(f"  n_segments : {info['n_segments']}")
        print(f"  duration   : {info['duration']:.2f}s" if info["duration"] is not None else "  duration   : None")
        print(f"  cov_trace  : {info['cov_trace']:.3f}")
        print(f"  mu(z)      : [{mu_str}]")
        print(f"  |MACs|     : {len(macs)}")
        print(f"  |Keys|     : {len(keys)}")

        print(f"  MACs       : {', '.join(macs_show) if macs_show else '(none)'}{macs_more}")
        print(f"  Keys       : {', '.join(keys_show) if keys_show else '(none)'}{keys_more}")

# --------------------------- CLI / reporting ---------------------------

def write_jsonl(path: str, items: List[Tracklet]) -> None:
    with open(path, "w") as f:
        for tau in items:
            f.write(json.dumps(tau.to_jsonable()) + "\n")
    print(f"[✓] wrote: {path} ({len(items)} items)")

def main() -> None:
    print("=" * 80)
    print("AirCatch — Streaming BLE tracker detection via online CFO clustering (Option B)")
    print("=" * 80)

    if not os.path.exists(FNAME):
        print(f"ERROR: Missing input file: {FNAME}")
        sys.exit(1)

    df = pd.read_csv(FNAME)

    time_col = resolve_time_column(df)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df[np.isfinite(df[time_col])].copy()

    mac_col = resolve_mac_column(df)
    payload_col = resolve_payload_column(df)
    cfo_cols = resolve_cfo_feature_columns(df)

    print(f"[i] time_col   : {time_col}")
    print(f"[i] mac_col    : {mac_col}")
    print(f"[i] payload_col: {payload_col if payload_col else '(none)'}")
    print(f"[i] CFO cols   : {cfo_cols}")
    print(f"[i] USE_ZSPACE : {USE_ZSPACE}")
    print(f"[i] WARMUP_SEGS: {WARMUP_SEGS if USE_ZSPACE else '(n/a)'}")
    print(f"[i] AUTO_GAMMA : {AUTO_GAMMA} (q={AUTO_GAMMA_Q}, min_matches={AUTO_GAMMA_MIN_MATCHES}, mult={AUTO_GAMMA_MULT})")
    print(f"[i] THETA      : {THETA}  (T_MIN={T_MIN}, N_MIN={N_MIN}, K_MIN={K_MIN})")

    if REQUIRE_LOSTMODE_PREFIX and not payload_col:
        print("ERROR: REQUIRE_LOSTMODE_PREFIX=True but no payload column found in CSV.")
        print("       Either add a payload column or set REQUIRE_LOSTMODE_PREFIX=False.")
        sys.exit(1)

    gamma_init = GATE_GAMMA_Z_INIT if USE_ZSPACE else GATE_GAMMA_RAW_INIT

    params = AirCatchParams(
        p=SEGMENT_SIZE_P,
        dt=ASSOC_GAP_DT,     # not used in Option B
        gamma=gamma_init,
        theta=THETA,
        eps=EPS
    )

    clusters, flagged, seg_df = aircatch_stream(
        df=df,
        time_col=time_col,
        mac_col=mac_col,
        payload_col=payload_col,
        cfo_cols=cfo_cols,
        params=params
    )

    print_cluster_stats(clusters, max_macs=30, max_keys=30)

    # Save outputs
    write_jsonl(OUT_TRACKLETS_JSONL, clusters)
    write_jsonl(OUT_FLAGGED_JSONL, flagged)

    # Plots
    plot_cluster_cfo_boxplot(seg_df, cfo_cols, OUT_BOX_PDF, OUT_BOX_PNG)
    plot_clusters_pca3d(seg_df, cfo_cols, OUT_PCA3D_PDF, OUT_PCA3D_PNG)
    plot_cluster_mac_counts(clusters, OUT_MACBAR_PDF, OUT_MACBAR_PNG)

    # Summary
    print("\n" + "=" * 80)
    print(f"Clusters total: {len(clusters)} | Flagged: {len(flagged)} | Segments: {len(seg_df) if seg_df is not None else 0}")
    print("=" * 80)

    if flagged:
        flagged_sorted = sorted(flagged, key=lambda t: (-t.score(), -(t.t_max - t.t_min), -t.n, t.id))
        for tau in flagged_sorted[:20]:
            info = tau.to_jsonable()
            print(
                f"- id={info['id']:>3} score={info['score']} "
                f"nSeg={info['n_segments']:<4} dur={info['duration']:.1f}s "
                f"|K|={len(info['keys']):<3} |M|={len(info['macs']):<3} tr={info['cov_trace']:.2f}"
            )
    else:
        print("No clusters flagged. (With THETA=4 and T_MIN=300, this is normal unless clusters persist ≥ 300s.)")

if __name__ == "__main__":
    main()