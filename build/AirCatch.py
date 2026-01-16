#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirCatch — Streaming BLE tracker detection via online CFO clustering.

OPTION B (one-stage, persistent clusters):
  - Buffers packets per MAC until p packets -> finalize segment
  - Assign each segment to the best matching EXISTING CLUSTER among ALL clusters (no Δt window)
  - If no cluster passes γ, spawn a new cluster
  - Clusters are persistent (no pruning)

This version adds (MINIMAL structural changes):
  A) Top-K candidate evaluation (K=3 default)
  B) Cluster MAC-homogeneity (purity/entropy) tracking
  C) Dynamic gating (gamma) based on:
       - same-MAC reassociation (looser)
       - homogeneous clusters (stricter for NEW MACs)
       - heterogeneous clusters (looser for NEW MACs)
  D) Quarantine "benign" clusters (highly homogeneous & stable):
       - quarantined clusters accept only SAME MAC segments
       - quarantined clusters are excluded as candidates for NEW MACs
  E) Batch runner: pass --input (file or dir), save per-scenario outputs + summary plots (PDF)

Plots:
  - Per-scenario: MAC-level boxplot & PCA3D (optional) and cluster plots (boxplot/PCA3D/MAC-count)
  - Batch-level: purity vs scenario, flagged vs scenario (PDF)

NOTE:
  - This is still UNSUPERVISED in the sense it does not require ground-truth labels.
  - It uses observed MACs only to measure cluster homogeneity and protect benign clusters.
"""

import os
import sys
import json
import math
import re
import glob
import argparse
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AirCatch batch evaluation over CSVs")
    p.add_argument("--input", required=True, help="Input CSV file OR directory containing CSVs")
    p.add_argument("--out", default="aircatch_batch_results", help="Output directory for all results")
    p.add_argument("--no_mac_plots", action="store_true", help="Disable MAC-level pre-clustering plots (faster)")
    p.add_argument("--no_cluster_plots", action="store_true", help="Disable cluster-level plots (faster)")
    return p.parse_args()

# --------------------------- config ---------------------------

BASIC_CLUSTERING = True  # if True, only apply basic AirCatch rules (T_min, N_min, K_min, trace max) for scoring; if False, also apply new heterogeneity reward in scoring

# Keep plots readable when there are tons of MACs
MAX_MACS_FOR_PLOTS = 30   # top-N MACs by packet count

# --- AirCatch params (Algorithm) ---
SEGMENT_SIZE_P = 1               # p
ASSOC_GAP_DT = 2.0               # kept for compatibility, NOT USED in option B
T_MIN = 30.0                     # T_min (seconds)
N_MIN = 1                        # N_min (segments)
K_MIN = 1                        # K_min (unique keys) (NOTE: keys are placeholder unless you parse real keys)
THETA = 4                        # θ (base score threshold)
EPS = 1e-6                       # ε

# --------------------------- feature space choice ---------------------------

USE_ZSPACE = False

# In z-space, squared Mahalanobis typically sits around O(d) for matches.
GATE_GAMMA_Z_INIT = 12

# Raw-space init (defined to avoid NameError even if unused)
GATE_GAMMA_RAW_INIT = 50

# Scatter stability threshold differs by space
ETA_TRACE_MAX_Z = 8.0
ETA_TRACE_MAX_RAW = 50

# Require tag ecosystem filter?
REQUIRE_LOSTMODE_PREFIX = True

# Global normalization update cadence (only used if USE_ZSPACE=True)
GLOBAL_NORM_UPDATE_EVERY = 1
GLOBAL_NORM_MIN_STD = 1e-3

# --- covariance regularization (z-space) ---
COV_MIN_N = 20
Z_COV_DIAG_FLOOR = 0.25
COV_SHRINK_TO_I = 0.15
INV_EPS = 1e-3

# freeze scaler (None = never freeze)
GLOBAL_NORM_FREEZE_AFTER = None

# OPTION B: clusters persistent -> disable pruning
PRUNE_INACTIVE_AFTER = None

# warmup
WARMUP_SEGS = 50

# Raw-mode diag covariance prior floor (Hz^2)
RAW_DIAG_FLOOR_HZ = 7000

MIN_CLUSTER_SIZE_FOR_MATCH = 0

# --------------------------- NEW: Top-K + homogeneity policy ---------------------------

TOPK_CANDIDATES = 10

# homogeneity estimation (trust only after enough segments)
N_HOMO_MIN = 5
P_HOMO = 0.90     # purity >= 0.90 => homogeneous
P_HET = 0.55      # purity <= 0.55 and macs>=3 => heterogeneous-like

# quarantine benign clusters
N_QUAR = 15
P_QUAR = 0.90

# dynamic gamma multipliers
GAMMA_SAME_MAC_MULT = 2.5
GAMMA_HOMO_NEW_MULT = 0.25
GAMMA_HET_NEW_MULT = 3.0
GAMMA_UNC_NEW_MULT = 1.25

# --------------------------- NEW: dominant-MAC lock ---------------------------

# --- NEW: early guard (pre-lock) ---
DOM_GUARD_MIN_DOMCOUNT = 8    # once a MAC has repeated this many segments inside a cluster...
DOM_GUARD_PURITY = 0.70       # ...and dominates the cluster this much...
DOM_GUARD_BLOCK_NEW_MAC = True  # ...block any NEW MAC from entering it (even before lock)
DOM_LOCK_SINGLETON_FRAC_MAX = 0.25  # don't lock if too many singletons already

# --------------------------- dominant-MAC lock ---------------------------

DOM_LOCK_MIN_N = 20                # lock only after enough segments
DOM_LOCK_PURITY = 0.80             # dominant MAC fraction to consider it benign-anchored

# optional mild bias against entering a cluster with a new MAC (kept from your old code)
NEW_MAC_DIST_PENALTY = 1

# CFO feature columns likely present
OPTIONAL_CFO_COLS = [
    "CFO_Hz", "cfo_hz", "cfo_quick_hz",
    "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz",
    "cfo_equal_00_hz", "cfo_equal_11_hz",
    "cfo_jump_10_hz", "cfo_jump_01_hz",
]

# Prefer these 5 for plotting if present
PLOT_CFO_PREFERRED = ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"]

# --------------------------- NEW: feature weighting for distance ---------------------------

# Give MUCH higher weight to overall CFO than transition CFOs.
# Effect: distance is dominated by CFO_Hz; transitions only influence when overall is close.
OVERALL_CFO_WEIGHT = 600000000
TRANSITION_CFO_WEIGHT = 0.5

def build_feature_weights(cfo_cols: List[str]) -> np.ndarray:
    """
    Return per-feature multiplicative weights aligned with `cfo_cols`.
    We apply these weights directly in the clustering space (z-space or raw),
    meaning Mahalanobis distance and covariance are computed in the weighted space.
    """
    w = np.ones(len(cfo_cols), dtype=float)
    for i, c in enumerate(cfo_cols):
        lc = str(c).lower()

        # Overall CFO (primary)
        if lc in ("cfo_hz", "cfo_quick_hz"):
            w[i] = OVERALL_CFO_WEIGHT
            continue

        # Transition CFOs (noisy)
        if lc in ("cfo_00_hz", "cfo_11_hz", "cfo_10_hz", "cfo_01_hz"):
            w[i] = TRANSITION_CFO_WEIGHT
            continue

        # Any other CFO-like feature: keep default (1.0) unless you want to downweight it too.
        w[i] = 1.0

    return w

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
    True if:
      - Apple Find My: AD type 0xFF, Apple company 0x004C, then 0x12 0x19
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

def _inv_psd(mat: np.ndarray, eps: float = INV_EPS) -> np.ndarray:
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

    # NEW: MAC composition tracking
    mac_counts: Dict[str, int] = field(default_factory=dict)

    # NEW: quarantine state
    is_quarantined: bool = False

    # raw-mode diagonal covariance prior (Hz^2 per feature)
    prior_diag: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mu.size == 0:
            self.mu = np.zeros(self.d, dtype=float)
        if self.M2.size == 0:
            self.M2 = np.zeros((self.d, self.d), dtype=float)

    # ---------- homogeneity metrics ----------

    # def purity(self) -> float:
    #     total = sum(self.mac_counts.values())
    #     if total <= 0:
    #         return 1.0
    #     mmax = max(self.mac_counts.values()) if self.mac_counts else 0
    #     return float(mmax) / float(total)

    def entropy_norm(self) -> float:
        total = sum(self.mac_counts.values())
        if total <= 0 or len(self.mac_counts) <= 1:
            return 0.0
        ps = np.array([c / total for c in self.mac_counts.values()], dtype=float)
        H = -np.sum(ps * np.log(ps + 1e-12))
        Hmax = np.log(float(len(self.mac_counts)))
        return float(H / (Hmax + 1e-12))

    def is_homogeneous(self) -> bool:
        return (self.n >= int(N_HOMO_MIN)) and (self.purity() >= float(P_HOMO))

    def is_heterogeneous(self) -> bool:
        return (self.n >= int(N_HOMO_MIN)) and (len(self.macs) >= 3) and (self.purity() <= float(P_HET))
    
        # ------------------- NEW: dominant-MAC lock helpers -------------------

    def dominant_mac(self) -> Optional[str]:
        if not hasattr(self, "mac_counts") or not self.mac_counts:
            return None
        return max(self.mac_counts.items(), key=lambda kv: kv[1])[0]

    def purity(self) -> float:
        """
        MAC purity using mac_counts (segments per MAC in this cluster).
        """
        if not hasattr(self, "mac_counts") or not self.mac_counts:
            return 0.0
        total = sum(self.mac_counts.values())
        if total <= 0:
            return 0.0
        return float(max(self.mac_counts.values())) / float(total)

    def singleton_fraction(self) -> float:
        """
        Fraction of MACs that appear exactly once in this cluster (churn indicator).
        """
        if not hasattr(self, "mac_counts") or not self.mac_counts:
            return 0.0
        n_macs = len(self.mac_counts)
        if n_macs <= 0:
            return 0.0
        singletons = sum(1 for c in self.mac_counts.values() if int(c) == 1)
        return float(singletons) / float(n_macs)
    
    def dominant_count(self) -> int:
        if not self.mac_counts:
            return 0
        return int(max(self.mac_counts.values()))

    def second_count(self) -> int:
        if not self.mac_counts or len(self.mac_counts) < 2:
            return 0
        vals = sorted([int(v) for v in self.mac_counts.values()], reverse=True)
        return int(vals[1])

    def dom_lock_active(self) -> bool:
        """
        If cluster is clearly anchored by one MAC, treat it as benign-anchored and
        reject NEW MAC entries to keep it clean.
        """
        if self.n < int(DOM_LOCK_MIN_N):
            return False

        if self.purity() < float(DOM_LOCK_PURITY):
            return False

        # If it's already very churny, do NOT lock (likely attacker-ish)
        if (len(self.macs) >= 5) and (self.singleton_fraction() > float(DOM_LOCK_SINGLETON_FRAC_MAX)):
            return False

        return True

    # ---------- covariance & distance ----------

    def cov(self) -> np.ndarray:
        # raw-mode prior for brand new clusters
        if self.n < 2:
            if (not USE_ZSPACE) and (self.prior_diag is not None):
                diag = np.asarray(self.prior_diag, dtype=float)
                diag = np.maximum(diag, (RAW_DIAG_FLOOR_HZ**2) * np.ones_like(diag))
                return np.diag(diag)
            return np.eye(self.d, dtype=float)

        # z-space small-n: avoid near-singular covariance
        if USE_ZSPACE and self.n < int(COV_MIN_N):
            return np.eye(self.d, dtype=float)

        C = self.M2 / max(1, (self.n - 1))
        C = 0.5 * (C + C.T)

        if USE_ZSPACE:
            diag = np.diag(C).copy()
            diag = np.maximum(diag, float(Z_COV_DIAG_FLOOR))
            C[np.diag_indices_from(C)] = diag

            alpha = float(COV_SHRINK_TO_I)
            if alpha > 0:
                C = (1.0 - alpha) * C + alpha * np.eye(self.d, dtype=float)

        return C

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

    def merge_segment(self, t_s: float, t_e: float, mac: str, keys_in_seg: Set[str], tag_type: Optional[str] = None) -> None:
        self.t_min = min(self.t_min, float(t_s))
        self.t_max = max(self.t_max, float(t_e))

        if isinstance(mac, str) and mac != "":
            self.macs.add(mac)
            self.mac_counts[mac] = int(self.mac_counts.get(mac, 0) + 1)

        for k in keys_in_seg:
            self.keys.add(k)
        if tag_type:
            self.keys.add(f"tag:{tag_type}")

        # quarantine update (benign-like, stable)
        thr = ETA_TRACE_MAX_Z if USE_ZSPACE else ETA_TRACE_MAX_RAW
        if (not self.is_quarantined) and (self.n >= int(N_QUAR)) and (self.purity() >= float(P_QUAR)) and (self.trace() <= thr):
            self.is_quarantined = True

    # ---------- scoring / reporting ----------

    def score(self) -> int:
        """
        Base score (your original) + NEW heterogeneity evidence.
        """
        T = self.t_max - self.t_min
        s = 0
        s += int(T >= T_MIN)
        s += int(self.n >= N_MIN)
        s += int(len(self.keys) >= K_MIN)

        thr = ETA_TRACE_MAX_Z if USE_ZSPACE else ETA_TRACE_MAX_RAW
        s += int(self.trace() <= thr)

        # NEW: explicitly reward "CFO-stable but MAC-unstable" (masquerading signature)
        s += int(self.is_heterogeneous())

        return s

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "n_segments": int(self.n),
            "t_min": float(self.t_min) if np.isfinite(self.t_min) else None,
            "t_max": float(self.t_max) if np.isfinite(self.t_max) else None,
            "duration": float(self.t_max - self.t_min) if (np.isfinite(self.t_min) and np.isfinite(self.t_max)) else None,
            "macs": sorted(list(self.macs)),
            "mac_counts": dict(self.mac_counts),
            "purity": float(self.purity()),
            "entropy_norm": float(self.entropy_norm()),
            "is_quarantined": bool(self.is_quarantined),
            "keys": sorted(list(self.keys)),
            "mu": self.mu.tolist(),
            "cov_trace": float(self.trace()),
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
    topk: int = TOPK_CANDIDATES

def _gamma_for_candidate(tau: Tracklet, mac: str, gamma_base: float) -> float:
    """
    Dynamic gating policy:
      - same MAC: looser
      - homogeneous cluster: very strict for new MAC entry
      - heterogeneous cluster: looser for new MAC entry
      - else: base
    """
    gb = float(gamma_base)
    if mac in tau.macs:
        return float(GAMMA_SAME_MAC_MULT) * gb

    if tau.is_quarantined:
        # quarantined clusters should not accept new MACs at all
        return -1.0  # "never pass"

    if tau.is_homogeneous():
        return float(GAMMA_HOMO_NEW_MULT) * gb
    if tau.is_heterogeneous():
        return float(GAMMA_HET_NEW_MULT) * gb
    return float(GAMMA_UNC_NEW_MULT) * gb

def detect_adversary_macs_from_payload(df_raw: pd.DataFrame,
                                      mac_col: str,
                                      payload_col: str,
                                      pattern_hex: str = "4c001219ff") -> Set[str]:
    """
    Return a set of MACs (strings) that appear in rows whose payload contains `pattern_hex`
    (case-insensitive, hex-normalized).

    pattern_hex default corresponds to user rule: "4c001219FF" (we normalize to lowercase).
    """
    adv_pat = "".join(ch for ch in pattern_hex.lower() if ch in "0123456789abcdef")
    if df_raw is None or df_raw.empty:
        return set()
    if mac_col not in df_raw.columns or payload_col not in df_raw.columns:
        return set()

    def _payload_has_pat(x: object) -> bool:
        if not isinstance(x, str):
            return False
        s = x.strip().lower()
        if s.startswith("0x"):
            s = s[2:]
        s = "".join(ch for ch in s if ch in "0123456789abcdef")
        return adv_pat in s

    tmp = df_raw[[mac_col, payload_col]].copy()
    tmp[mac_col] = tmp[mac_col].astype(str)
    tmp = tmp[tmp[mac_col].str.len() > 0]
    if tmp.empty:
        return set()

    mask = tmp[payload_col].apply(_payload_has_pat)
    adv_macs = set(tmp.loc[mask, mac_col].astype(str).tolist())
    return set(m for m in adv_macs if isinstance(m, str) and m != "")

def plot_cluster_purity_barplot_with_adv(df_raw: pd.DataFrame,
                                         seg_df: pd.DataFrame,
                                         mac_col_raw: str,
                                         payload_col_raw: str,
                                         out_pdf: str,
                                         flagged_clusters: Optional[List[Tracklet]] = None,
                                         flagged_ids: Optional[Set[int]] = None,
                                         adv_pattern_hex: str = "4c001219ff",
                                         top_n: Optional[int] = None,
                                         adv_label_bucket: str = "Adversary MACs") -> None:
    """
    Bar plot: purity of each cluster (segment-weighted), annotate bar tops with:
      - B=<unique benign MACs>, A=<unique adversary MACs>
    Flagged clusters are marked by HATCHED bars (strong hatch for PDF visibility).
    """
    if seg_df is None or seg_df.empty:
        return
    if not {"cluster_id", "mac"}.issubset(set(seg_df.columns)):
        return

    # Resolve flagged cluster IDs
    fset: Set[int] = set()
    if flagged_ids is not None:
        fset |= set(int(x) for x in flagged_ids)
    if flagged_clusters is not None:
        fset |= set(int(t.id) for t in flagged_clusters)

    # Detect adversary MACs from RAW packet DF (payload substring rule)
    adv_macs = detect_adversary_macs_from_payload(
        df_raw=df_raw,
        mac_col=mac_col_raw,
        payload_col=payload_col_raw,
        pattern_hex=adv_pattern_hex
    )

    dfp = seg_df[["cluster_id", "mac"]].copy()
    dfp["cluster_id"] = pd.to_numeric(dfp["cluster_id"], errors="coerce")
    dfp = dfp.dropna(subset=["cluster_id"])
    dfp["cluster_id"] = dfp["cluster_id"].astype(int)
    dfp["mac"] = dfp["mac"].astype(str)
    dfp = dfp[dfp["mac"].str.len() > 0].copy()
    if dfp.empty:
        return

    rows = []
    for cid, g in dfp.groupby("cluster_id"):
        macs_seg = g["mac"].values.astype(str)

        vc = pd.Series(macs_seg).value_counts()
        total = int(vc.sum())
        purity = float(vc.max()) / float(total) if total > 0 else 0.0

        uniq = set(macs_seg.tolist())
        n_adv = 0
        n_benign = 0
        for m in uniq:
            if m == adv_label_bucket:
                n_adv += 1
            elif m in adv_macs:
                n_adv += 1
            else:
                n_benign += 1

        rows.append({
            "cluster_id": int(cid),
            "purity": float(purity),
            "n_benign_macs": int(n_benign),
            "n_adv_macs": int(n_adv),
            "n_segments": int(len(macs_seg)),
            "n_unique_macs": int(len(uniq)),
            "is_flagged": int(int(cid) in fset),
        })

    if not rows:
        return

    dfx = pd.DataFrame(rows).sort_values(
        ["purity", "n_adv_macs", "n_unique_macs", "n_segments"],
        ascending=[False, False, False, False]
    )

    if top_n is not None and int(top_n) > 0:
        dfx = dfx.head(int(top_n)).copy()

    # ---- plotting (PDF hatch visibility tweaks) ----
    import matplotlib as mpl
    mpl.rcParams["hatch.linewidth"] = 1.4  # thicker hatch strokes for PDF

    fig_w = max(10.0, min(26.0, 0.55 * len(dfx) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    x = np.arange(len(dfx), dtype=float)

    # Give ALL bars an edge so hatches render reliably on flagged ones
    bars = ax.bar(
        x,
        dfx["purity"].values,
        linewidth=0.6,
        edgecolor="0.25"  # subtle edge for unflagged
    )

    # Hatch flagged bars (dense + black edge)
    for bar, isf in zip(bars, dfx["is_flagged"].values):
        if int(isf) == 1:
            bar.set_hatch("//////")     # denser hatch
            bar.set_edgecolor("black")  # hatch uses edge properties -> make it black
            bar.set_linewidth(1.2)

    ax.set_title("Cluster purity with benign/adversary MAC counts (hatched = flagged)")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Purity (dominant MAC fraction)")
    ax.set_ylim(0.0, 1.08)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(cid)) for cid in dfx["cluster_id"].values], rotation=90, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    # Annotate counts on top: B=..., A=...
    y0, y1 = ax.get_ylim()
    dy = 0.02 * (y1 - y0 + 1e-9)
    for i, (p, nb, na) in enumerate(zip(dfx["purity"].values, dfx["n_benign_macs"].values, dfx["n_adv_macs"].values)):
        ax.text(i, float(p) + dy, f"B={int(nb)}, A={int(na)}",
                ha="center", va="bottom", fontsize=7, rotation=90)

    # Optional: quick legend cue (simple proxy patch)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor="white", edgecolor="black", hatch="//////", label="Flagged cluster")],
        loc="upper right",
        frameon=True
    )

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def aircatch_stream(df: pd.DataFrame,
                    time_col: str,
                    mac_col: str,
                    payload_col: str,
                    cfo_cols: List[str],
                    params: AirCatchParams) -> Tuple[List[Tracklet], List[Tracklet], pd.DataFrame]:
    """
    Option B: candidates are ALL clusters (persistent), but:
      - quarantined clusters are excluded for NEW MACs
      - top-K candidates are evaluated and best passing gate is selected
    """
    tvals = pd.to_numeric(df[time_col], errors="coerce")
    df = df.assign(_t=tvals).replace([np.inf, -np.inf], np.nan).dropna(subset=["_t"])
    df = df.sort_values("_t").drop(columns=["_t"])

    d = len(cfo_cols)
    scaler = OnlineGlobalScaler(d=d, eps=params.eps)

    # NEW: feature weights (applied in clustering space)
    feat_w = build_feature_weights(cfo_cols)

    buffers: Dict[str, List[Tuple[float, np.ndarray, Optional[str]]]] = {}
    clusters: List[Tracklet] = []
    flagged_ids: Set[int] = set()
    seg_records: List[Dict[str, Any]] = []

    next_id = 1
    seg_count = 0

    DEBUG = False
    DEBUG_EVERY_SEG = 1
    DEBUG_TOPK = 5
    DEBUG_SCALER_EVERY = 25

    # counters
    n_rows = 0
    skip_tag = 0
    skip_bad_cfo_pkt = 0
    n_segments_total = 0
    n_spawn = 0
    n_match = 0
    d_best_list: List[float] = []

    mac_to_tag: Dict[str, str] = {}

    gamma_live = float(params.gamma)

    raw_diag_var = np.ones(d, dtype=float)
    raw_diag_scaler = OnlineGlobalScaler(d=d, eps=params.eps)

    def _candidates() -> List[Tracklet]:
        return clusters

    for _, row in df.iterrows():
        n_rows += 1

        t_k = float(pd.to_numeric(row[time_col], errors="coerce"))
        m_k = str(row[mac_col])
        p_k = row.get(payload_col, None) if payload_col else None

        if isinstance(p_k, str) and m_k not in mac_to_tag:
            mac_to_tag[m_k] = extract_tag_type(p_k)

        if REQUIRE_LOSTMODE_PREFIX:
            if not (isinstance(p_k, str) and has_lostmode_prefix(p_k)):
                skip_tag += 1
                continue

        K_k = extract_public_key(p_k, m_k) if isinstance(p_k, str) else None

        f_k = estimate_cfo_features(row, cfo_cols)
        if not np.all(np.isfinite(f_k)):
            skip_bad_cfo_pkt += 1
            continue

        buffers.setdefault(m_k, []).append((t_k, f_k, K_k))
        if len(buffers[m_k]) < params.p:
            continue

        t_s, t_e, f_bar, K_s = finalize_segment(buffers[m_k])
        buffers[m_k].clear()

        if not np.all(np.isfinite(f_bar)):
            continue

        # ---- warmup / feature vector ----
        if USE_ZSPACE:
            if scaler.n < int(WARMUP_SEGS):
                scaler.update(f_bar)
                seg_count += 1
                continue
            x = scaler.transform(f_bar)
        else:
            raw_diag_scaler.update(f_bar)
            raw_diag_var = raw_diag_scaler.var()
            x = f_bar.copy()

        # NEW: apply feature weights in clustering space (dominates distance with CFO_Hz)
        x = x * feat_w


        # ---- candidate distances (ALL clusters) ----
        C = _candidates()

        dist_list: List[Tuple[float, int, Tracklet]] = []
        if C:
            for tau in C:
                # 1) quarantined: never accept new MACs
                if tau.is_quarantined and (m_k not in tau.macs):
                    continue

                # 2) EARLY GUARD (pre-lock): if a cluster already has a repeating dominant MAC,
                #    do NOT allow brand-new MACs to enter and pollute it.
                if DOM_GUARD_BLOCK_NEW_MAC and (m_k not in tau.macs):
                    dom = tau.dominant_mac()
                    if dom is not None:
                        if (tau.dominant_count() >= int(DOM_GUARD_MIN_DOMCOUNT)) and (tau.purity() >= float(DOM_GUARD_PURITY)):
                            # This cluster is "benign-anchored" already -> reject new MAC
                            continue

                # 3) DOM LOCK (stronger, later)
                if tau.dom_lock_active():
                    dom = tau.dominant_mac()
                    if dom is not None and m_k != dom:
                        continue

                if tau.n < MIN_CLUSTER_SIZE_FOR_MATCH:
                    if m_k not in tau.macs:
                        continue 

                dist = tau.mahalanobis(x, eps=params.eps)
                if m_k not in tau.macs:
                    dist *= float(NEW_MAC_DIST_PENALTY)
                dist_list.append((dist, tau.id, tau))

        dist_list.sort(key=lambda z: z[0])

        # ---- pick among top-K that passes gate ----
        spawned = False
        tau = None
        d_best = float("inf")

        topk = dist_list[:max(0, int(params.topk))] if dist_list else []
        chosen = None
        chosen_d = float("inf")

        for (dist, _, cand) in topk:
            g_cand = _gamma_for_candidate(cand, m_k, gamma_live)
            if g_cand < 0:
                continue
            if dist <= g_cand:
                chosen = cand
                chosen_d = dist
                break  # already sorted by distance

        if chosen is None:
            # spawn
            tau = Tracklet(id=next_id, d=d)
            next_id += 1
            if not USE_ZSPACE:
                # NEW: x is scaled by feat_w, so covariance prior must be scaled by feat_w^2
                tau.prior_diag = (raw_diag_var.copy() * (feat_w ** 2))
            clusters.append(tau)
            spawned = True
            n_spawn += 1
            d_best = float(topk[0][0]) if topk else float("inf")
        else:
            tau = chosen
            n_match += 1
            d_best = float(chosen_d)

        # update stats + merge
        tau.update_stats_welford(x)
        tau.merge_segment(
            t_s=t_s,
            t_e=t_e,
            mac=m_k,
            keys_in_seg=K_s,
            tag_type=mac_to_tag.get(m_k)
        )

        # record segment
        rec = {
            "cluster_id": int(tau.id),
            "t_s": float(t_s),
            "t_e": float(t_e),
            "mac": m_k,
            "n_keys": int(len(K_s)),
            "purity_cluster": float(tau.purity()),
            "entropy_cluster": float(tau.entropy_norm()),
            "is_quarantined": int(tau.is_quarantined),
            "decision": "SPAWN" if spawned else "MATCH",
            "d_best": float(d_best) if np.isfinite(d_best) else np.nan,
        }
        for j, colname in enumerate(cfo_cols):
            rec[f"seg_{colname}"] = float(f_bar[j])
        seg_records.append(rec)

        # counters
        seg_count += 1
        n_segments_total += 1
        d_best_list.append(d_best)

        # global scaler update (z-space)
        if USE_ZSPACE and (seg_count % GLOBAL_NORM_UPDATE_EVERY == 0):
            if (GLOBAL_NORM_FREEZE_AFTER is None) or (scaler.n < int(GLOBAL_NORM_FREEZE_AFTER)):
                scaler.update(f_bar)

        # pruning (disabled)
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

        # decision: flagged
        if tau.score() >= params.theta:
            flagged_ids.add(tau.id)

        # debug prints
        if DEBUG and (n_segments_total % DEBUG_EVERY_SEG == 0):
            dur = float(t_e - t_s)
            print("-" * 110)
            print(f"[SEG {n_segments_total:>4}] mac={m_k} seg_dur={dur:.3f}s "
                  f"clusters_now={len(clusters)} gamma={gamma_live:.3f} space={'z' if USE_ZSPACE else 'raw'}")
            print(f"          decision={'SPAWN' if spawned else 'MATCH'} -> cluster_id={tau.id} d_best={d_best:.3f} "
                  f"purity={tau.purity():.2f} ent={tau.entropy_norm():.2f} quarantined={tau.is_quarantined}")
            if topk:
                show = topk[:max(0, int(DEBUG_TOPK))]
                topk_str = ", ".join([f"(id={cid}, d={dist:.2f})" for (dist, cid, _) in show])
                print(f"          top{len(show)} nearest: {topk_str}")

        if DEBUG and USE_ZSPACE and (seg_count % DEBUG_SCALER_EVERY == 0):
            frozen = (GLOBAL_NORM_FREEZE_AFTER is not None and scaler.n >= int(GLOBAL_NORM_FREEZE_AFTER))
            print(f"[scaler] n={scaler.n} mean={np.array2string(scaler.mean, precision=3)} "
                  f"std={np.array2string(scaler.std(), precision=3)} "
                  f"{'(FROZEN)' if frozen else ''}")

    flagged = [tau for tau in clusters if tau.id in flagged_ids]
    seg_df = pd.DataFrame(seg_records)

    return clusters, flagged, seg_df

def aircatch_chunked(df: pd.DataFrame,
                     time_col: str,
                     mac_col: str,
                     payload_col: str,
                     cfo_cols: List[str],
                     params: AirCatchParams,
                     chunk_minutes: float = 15.0,
                     basic_clustering: bool = True,
                     # --- adversary-flagging heuristics (tune if needed) ---
                     adv_min_duration_min: float = 45.0,
                     adv_min_unique_macs: int = 12,
                     adv_min_macs_per_chunk: float = 2.5,
                     # Optional: if True, still apply REQUIRE_LOSTMODE_PREFIX filter in chunking
                     enforce_lostmode_filter: Optional[bool] = None
                     ) -> Tuple[List[Tracklet], List[Tracklet], pd.DataFrame]:
    """
    Chunked AirCatch (persistent clusters), clustering on MAC-chunk means.

    What it does:
      1) Sort rows by time.
      2) Split into fixed time chunks (chunk_minutes).
      3) Within each chunk:
           - (optional) filter payloads using has_lostmode_prefix
           - group by MAC (AdvA) and compute mean CFO features for that MAC in that chunk
           - treat each (MAC, chunk) as a "segment" with feature vector = chunk-mean CFO
           - assign/spawn into PERSISTENT clusters using Mahalanobis distance in the chosen space
             (z-space or raw), with feature weighting (CFO_Hz dominates).
      4) After all chunks, compute "adversary-like" flags based on:
           - long duration
           - many unique MACs
           - high MACs-per-chunk-spanned

    basic_clustering=True:
      - ignores quarantine, homogeneity/heterogeneity, dominant-MAC lock, dynamic gamma multipliers
      - uses a single fixed gate gamma (params.gamma) for all candidates
      - does not penalize new MACs for entering clusters (better for MAC-rotating adversary)

    Returns:
      clusters : all clusters after processing all chunks
      flagged  : clusters flagged as adversary-like (heuristic)
      seg_df   : per (MAC, chunk) records
    """
    if df is None or df.empty:
        return [], [], pd.DataFrame()

    # Decide whether to enforce the lostmode filter inside this function.
    # Default behavior follows your global REQUIRE_LOSTMODE_PREFIX.
    if enforce_lostmode_filter is None:
        enforce_lostmode_filter = bool(REQUIRE_LOSTMODE_PREFIX)

    # ----------------------------
    # 1) sanitize + sort by time
    # ----------------------------
    tvals = pd.to_numeric(df[time_col], errors="coerce")
    df = df.assign(_t=tvals).replace([np.inf, -np.inf], np.nan).dropna(subset=["_t"])
    if df.empty:
        return [], [], pd.DataFrame()
    df = df.sort_values("_t").reset_index(drop=True)

    # Dimension of CFO feature vector
    d = len(cfo_cols)

    # Z-space scaler and raw-space diag estimator
    scaler = OnlineGlobalScaler(d=d, eps=params.eps)
    raw_diag_var = np.ones(d, dtype=float)
    raw_diag_scaler = OnlineGlobalScaler(d=d, eps=params.eps)

    # Feature weights in clustering space (overall CFO dominates distance)
    feat_w = build_feature_weights(cfo_cols)

    # Persistent clustering state
    clusters: List[Tracklet] = []
    next_id = 1

    # For segment records
    seg_records: List[Dict[str, Any]] = []

    # Tag/cache for MAC->tag_type (optional)
    mac_to_tag: Dict[str, str] = {}

    gamma_live = float(params.gamma)

    # ----------------------------
    # 2) define chunk boundaries
    # ----------------------------
    chunk_s = float(chunk_minutes) * 60.0
    t_min = float(df["_t"].min())
    t_max = float(df["_t"].max())
    if (not np.isfinite(t_min)) or (not np.isfinite(t_max)) or chunk_s <= 0:
        return [], [], pd.DataFrame()

    n_chunks = int(math.floor((t_max - t_min) / chunk_s)) + 1

    # helper: choose candidates (all persistent clusters)
    def _candidates() -> List[Tracklet]:
        return clusters

    seg_count = 0  # counts MAC-chunk segments processed (used for warmup & scaler update)

    # ----------------------------
    # 3) process each time chunk
    # ----------------------------
    for ck in range(n_chunks):
        t0 = t_min + ck * chunk_s
        t1 = t0 + chunk_s

        dfc = df[(df["_t"] >= t0) & (df["_t"] < t1)]
        if dfc.empty:
            continue

        # Optional: payload filtering (Find My / lostmode prefix)
        if enforce_lostmode_filter:
            if not payload_col:
                raise RuntimeError("Lostmode filter requested but payload_col is empty/None.")
            dfc = dfc[dfc[payload_col].apply(lambda x: isinstance(x, str) and has_lostmode_prefix(x))]
            if dfc.empty:
                continue

        # Keep only needed cols
        cols_keep = [mac_col] + cfo_cols + ([payload_col] if payload_col else [])
        tmp = dfc[cols_keep].copy()

        # Clean MACs
        tmp[mac_col] = tmp[mac_col].astype(str)
        tmp = tmp[tmp[mac_col].str.len() > 0]
        if tmp.empty:
            continue

        # Numeric CFO
        for c in cfo_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp = tmp.replace([np.inf, -np.inf], np.nan)

        # Group by MAC within chunk -> one segment per MAC in this chunk
        for mac, gmac in tmp.groupby(mac_col, sort=False):
            # chunk-mean features for this MAC
            f_bar = gmac[cfo_cols].mean(skipna=True).values.astype(float)
            if not np.all(np.isfinite(f_bar)):
                continue

            # representative payload for (optional) tag + key extraction
            p_k = None
            if payload_col and payload_col in gmac.columns:
                for vv in gmac[payload_col].values:
                    if isinstance(vv, str) and len(vv) > 0:
                        p_k = vv
                        break

            # tag caching (optional)
            if isinstance(p_k, str) and mac not in mac_to_tag:
                mac_to_tag[mac] = extract_tag_type(p_k)

            # placeholder "keys"
            K_k = extract_public_key(p_k, mac) if isinstance(p_k, str) else None
            K_s: Set[str] = set([K_k]) if isinstance(K_k, str) and K_k != "" else set()

            # ----------------------------
            # 3a) build feature vector x
            # ----------------------------
            if USE_ZSPACE:
                # Warmup global scaler on chunk-means (not per packet)
                if scaler.n < int(WARMUP_SEGS):
                    scaler.update(f_bar)
                    seg_count += 1
                    continue
                x = scaler.transform(f_bar)
            else:
                raw_diag_scaler.update(f_bar)
                raw_diag_var = raw_diag_scaler.var()
                x = f_bar.copy()

            # Apply feature weights in clustering space
            x = x * feat_w

            # ----------------------------
            # 3b) compute distances to candidates
            # ----------------------------
            dist_list: List[Tuple[float, int, Tracklet]] = []
            for tau in _candidates():
                # In "basic" mode: do NOT block new MACs from entering clusters.
                # In non-basic mode: keep your original protections.
                if not basic_clustering:
                    # 1) quarantined: never accept new MACs
                    if tau.is_quarantined and (mac not in tau.macs):
                        continue

                    # 2) EARLY GUARD (pre-lock)
                    if DOM_GUARD_BLOCK_NEW_MAC and (mac not in tau.macs):
                        dom = tau.dominant_mac()
                        if dom is not None:
                            if (tau.dominant_count() >= int(DOM_GUARD_MIN_DOMCOUNT)) and (tau.purity() >= float(DOM_GUARD_PURITY)):
                                continue

                    # 3) DOM LOCK
                    if tau.dom_lock_active():
                        dom = tau.dominant_mac()
                        if dom is not None and mac != dom:
                            continue

                    # optional min-size rule
                    if tau.n < MIN_CLUSTER_SIZE_FOR_MATCH:
                        if mac not in tau.macs:
                            continue

                # distance in current clustering space
                dist = tau.mahalanobis(x, eps=params.eps)

                # In basic mode, do NOT penalize a new MAC (we WANT MAC-rotation to co-cluster).
                # In non-basic mode, keep your original behavior.
                if (not basic_clustering) and (mac not in tau.macs):
                    dist *= float(NEW_MAC_DIST_PENALTY)

                dist_list.append((dist, tau.id, tau))

            dist_list.sort(key=lambda z: z[0])

            # ----------------------------
            # 3c) pick best passing candidate (Top-K)
            # ----------------------------
            topk = dist_list[:max(0, int(params.topk))] if dist_list else []
            chosen = None
            chosen_d = float("inf")

            for (dist, _, cand) in topk:
                if basic_clustering:
                    # single fixed gate
                    g_cand = gamma_live
                else:
                    # dynamic gate policy (your original)
                    g_cand = _gamma_for_candidate(cand, mac, gamma_live)

                if g_cand < 0:
                    continue
                if dist <= g_cand:
                    chosen = cand
                    chosen_d = dist
                    break

            # ----------------------------
            # 3d) spawn or match
            # ----------------------------
            spawned = False
            if chosen is None:
                tau = Tracklet(id=next_id, d=d)
                next_id += 1

                if not USE_ZSPACE:
                    # x scaled by feat_w => diag prior scaled by feat_w^2
                    tau.prior_diag = (raw_diag_var.copy() * (feat_w ** 2))

                clusters.append(tau)
                spawned = True
                d_best = float(topk[0][0]) if topk else float("inf")
            else:
                tau = chosen
                d_best = float(chosen_d)

            # ----------------------------
            # 3e) update persistent cluster state
            # ----------------------------
            tau.update_stats_welford(x)
            tau.merge_segment(
                t_s=float(t0),
                t_e=float(t1),
                mac=str(mac),
                keys_in_seg=K_s,
                tag_type=mac_to_tag.get(mac)
            )

            # In basic mode: explicitly ignore quarantine status (don’t let it influence future)
            if basic_clustering:
                tau.is_quarantined = False

            # Record one row per (MAC, chunk)
            rec = {
                "cluster_id": int(tau.id),
                "t_s": float(t0),
                "t_e": float(t1),
                "chunk_idx": int(ck),
                "mac": str(mac),
                "n_keys": int(len(K_s)),
                "purity_cluster": float(tau.purity()),
                "entropy_cluster": float(tau.entropy_norm()),
                "is_quarantined": int(tau.is_quarantined),
                "decision": "SPAWN" if spawned else "MATCH",
                "d_best": float(d_best) if np.isfinite(d_best) else np.nan,
                "n_rows_in_chunk_for_mac": int(len(gmac)),
            }
            for j, colname in enumerate(cfo_cols):
                rec[f"seg_{colname}"] = float(f_bar[j])
            seg_records.append(rec)

            seg_count += 1

            # Update global scaler with chunk-mean (keeps z-space stable)
            if USE_ZSPACE and (seg_count % GLOBAL_NORM_UPDATE_EVERY == 0):
                if (GLOBAL_NORM_FREEZE_AFTER is None) or (scaler.n < int(GLOBAL_NORM_FREEZE_AFTER)):
                    scaler.update(f_bar)

            # Pruning (still disabled by default)
            if PRUNE_INACTIVE_AFTER is not None:
                t_now = float(t0)
                keep: List[Tracklet] = []
                for tt in clusters:
                    # keep flagged if you ever use flagged_ids during streaming; here we don't rely on it
                    if (t_now - tt.t_max) <= float(PRUNE_INACTIVE_AFTER):
                        keep.append(tt)
                clusters = keep

    # ----------------------------
    # 4) Flag "adversary-like" clusters AFTER all chunks
    # ----------------------------
    # Your usecase: adversary cluster has:
    #  - long duration
    #  - many unique MACs
    #  - many MACs per chunk spanned (benign MAC rotation ~1 per chunk)
    adv_min_duration_s = float(adv_min_duration_min) * 60.0

    flagged: List[Tracklet] = []
    for tau in clusters:
        dur = float(tau.t_max - tau.t_min) if (np.isfinite(tau.t_min) and np.isfinite(tau.t_max)) else 0.0
        macs = len(tau.macs)

        chunks_spanned = max(1.0, dur / (chunk_s + 1e-9))
        macs_per_chunk = float(macs) / float(chunks_spanned)

        is_adv = (
            (dur >= adv_min_duration_s) and
            (macs >= int(adv_min_unique_macs)) and
            (macs_per_chunk >= float(adv_min_macs_per_chunk))
        )
        if is_adv:
            flagged.append(tau)

    seg_df = pd.DataFrame(seg_records)
    return clusters, flagged, seg_df

# --------------------------- plotting helpers ---------------------------

def extract_tag_type(payload_hex: str) -> Optional[str]:
    if not isinstance(payload_hex, str):
        return None

    s = payload_hex.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    s = "".join(ch for ch in s if ch in "0123456789abcdef")
    if len(s) < 2:
        return None

    try:
        b = bytes.fromhex(s)
    except ValueError:
        return None

    def parse_ad_structures(ad: bytes) -> Optional[str]:
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

            # Apple Find My
            if ad_type == 0xFF and len(data) >= 4:
                company_id = data[0] | (data[1] << 8)
                if company_id == 0x004C and data[2] == 0x12 and data[3] == 0x19:
                    return "APPLE_FINDMY"

            # Service Data (16-bit UUID)
            if ad_type == 0x16 and len(data) >= 2:
                svc_uuid = data[0] | (data[1] << 8)
                if svc_uuid == 0xFEAA:
                    return "GOOGLE"
                if svc_uuid == 0xFEED:
                    return "TILE"
                if svc_uuid in (0xFD5A, 0xFD59):
                    return "SAMSUNG_SMARTTAG"

            pos = end
        return None

    if len(b) >= 8:
        t = parse_ad_structures(b[6:])
        if t:
            return t
        return parse_ad_structures(b)
    else:
        return parse_ad_structures(b)

def _resolve_plot_cfo_cols_df(df: pd.DataFrame) -> List[str]:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    chosen = []
    for pref in PLOT_CFO_PREFERRED:
        hit = lower_map.get(pref.lower(), "")
        if hit and hit not in chosen:
            chosen.append(hit)
        if len(chosen) >= 5:
            break
    if chosen:
        return chosen
    for c in df.columns:
        lc = str(c).lower()
        if ("cfo" in lc) and ("hz" in lc):
            chosen.append(str(c))
        if len(chosen) >= 5:
            break
    return chosen

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

# --------------------------- MAC-level plots (PDF only) ---------------------------

def plot_mac_cfo_boxplot(df_raw: pd.DataFrame,
                         time_col: str,
                         mac_col: str,
                         payload_col: Optional[str],
                         out_pdf: str) -> None:
    if df_raw is None or df_raw.empty:
        return

    dfp = df_raw.copy()
    if REQUIRE_LOSTMODE_PREFIX and payload_col:
        dfp = dfp[dfp[payload_col].apply(lambda x: isinstance(x, str) and has_lostmode_prefix(x))]

    dfp[mac_col] = dfp[mac_col].astype(str)
    dfp = dfp[dfp[mac_col].str.len() > 0]

    plot_cfo_cols = [c for c in _resolve_plot_cfo_cols_df(dfp) if c in dfp.columns]
    if not plot_cfo_cols:
        return

    mac_counts = dfp[mac_col].value_counts()
    macs = mac_counts.head(int(MAX_MACS_FOR_PLOTS)).index.tolist()
    dfp = dfp[dfp[mac_col].isin(macs)].copy()

    for c in plot_cfo_cols:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=plot_cfo_cols)
    if dfp.empty:
        return

    macs_sorted = sorted(macs)
    n_feat = len(plot_cfo_cols)

    fig_w = max(10.0, min(26.0, 0.55 * len(macs_sorted) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))

    base_positions = np.arange(len(macs_sorted), dtype=float)
    offsets = np.linspace(-0.25, 0.25, n_feat) if n_feat > 1 else np.array([0.0])

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = ["C0", "C1", "C2", "C3", "C4"]

    handles, labels = [], []

    for fi, col in enumerate(plot_cfo_cols):
        data, positions = [], []
        for i, mac in enumerate(macs_sorted):
            vals = dfp.loc[dfp[mac_col] == mac, col].dropna().values
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
        labels.append(col)

    ax.set_title(f"CFO distributions grouped by MAC (top {len(macs_sorted)} MACs)")
    ax.set_ylabel("CFO (Hz)")
    ax.set_xlabel("MAC (AdvA)")
    ax.set_xticks(base_positions)
    ax.set_xticklabels(macs_sorted, rotation=90, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    if handles:
        ax.legend(handles, labels, title="CFO feature", loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def plot_macs_pca3d(df_raw: pd.DataFrame,
                    mac_col: str,
                    payload_col: Optional[str],
                    out_pdf: str) -> None:
    if df_raw is None or df_raw.empty:
        return

    dfp = df_raw.copy()
    if REQUIRE_LOSTMODE_PREFIX and payload_col:
        dfp = dfp[dfp[payload_col].apply(lambda x: isinstance(x, str) and has_lostmode_prefix(x))]

    dfp[mac_col] = dfp[mac_col].astype(str)
    dfp = dfp[dfp[mac_col].str.len() > 0]

    feat_cols = [c for c in _resolve_plot_cfo_cols_df(dfp) if c in dfp.columns]
    if len(feat_cols) < 2:
        return

    mac_counts = dfp[mac_col].value_counts()
    macs = mac_counts.head(int(MAX_MACS_FOR_PLOTS)).index.tolist()
    dfp = dfp[dfp[mac_col].isin(macs)].copy()

    for c in feat_cols:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols)

    if len(dfp) < 10:
        return

    X = dfp[feat_cols].values.astype(float)
    mu = np.mean(X, axis=0)
    sig = np.maximum(np.std(X, axis=0, ddof=1), GLOBAL_NORM_MIN_STD)
    Xz = (X - mu) / sig

    pca = PCA(n_components=3, random_state=0)
    Xp = pca.fit_transform(Xz)
    macs_y = dfp[mac_col].values.astype(str)

    top_macs = set(mac_counts.head(12).index.tolist())

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("tab20")
    unique_macs = sorted(set(macs_y.tolist()))
    for i, mac in enumerate(unique_macs):
        mask = (macs_y == mac)
        pts = Xp[mask]
        if pts.shape[0] == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=10, alpha=0.7,
                   color=cmap(i % 20),
                   label=mac if mac in top_macs else None)

    evr = pca.explained_variance_ratio_
    ax.set_title(f"MAC-colored 3-D PCA of CFO (EVR: {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if len(top_macs) > 0:
        ax.legend(title="Top MACs", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    # plt.show()
    plt.close(fig)

# --------------------------- cluster-level plots (PDF only) ---------------------------

def plot_mac_cfo_boxplot(df_raw: pd.DataFrame,
                         time_col: str,
                         mac_col: str,
                         payload_col: Optional[str],
                         out_pdf: str) -> None:
    if df_raw is None or df_raw.empty:
        return

    dfp = df_raw.copy()
    if REQUIRE_LOSTMODE_PREFIX and payload_col:
        dfp = dfp[dfp[payload_col].apply(lambda x: isinstance(x, str) and has_lostmode_prefix(x))]

    dfp[mac_col] = dfp[mac_col].astype(str)
    dfp = dfp[dfp[mac_col].str.len() > 0]

    plot_cfo_cols = [c for c in _resolve_plot_cfo_cols_df(dfp) if c in dfp.columns]
    if not plot_cfo_cols:
        return

    mac_counts = dfp[mac_col].value_counts()
    macs = mac_counts.head(int(MAX_MACS_FOR_PLOTS)).index.tolist()
    dfp = dfp[dfp[mac_col].isin(macs)].copy()

    for c in plot_cfo_cols:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=plot_cfo_cols)
    if dfp.empty:
        return

    macs_sorted = sorted(macs)
    n_feat = len(plot_cfo_cols)

    fig_w = max(10.0, min(26.0, 0.55 * len(macs_sorted) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))

    base_positions = np.arange(len(macs_sorted), dtype=float)
    offsets = np.linspace(-0.25, 0.25, n_feat) if n_feat > 1 else np.array([0.0])

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = ["C0", "C1", "C2", "C3", "C4"]

    handles, labels = [], []

    for fi, col in enumerate(plot_cfo_cols):
        data, positions = [], []
        for i, mac in enumerate(macs_sorted):
            vals = dfp.loc[dfp[mac_col] == mac, col].dropna().values
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

        # NEW: annotate each individual box (MAC × feature) with its sample count
        y0, y1 = ax.get_ylim()
        dy = 0.01 * (y1 - y0 + 1e-9)  # small vertical offset
        for j, vals in enumerate(data):
            n = int(len(vals))
            if n <= 0:
                continue

            # Each box has 2 whiskers; take the top whisker y as the anchor
            w1 = bp["whiskers"][2 * j].get_ydata()
            w2 = bp["whiskers"][2 * j + 1].get_ydata()
            y_top = float(max(np.max(w1), np.max(w2)))

            ax.text(
                positions[j],
                y_top + dy,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

        handles.append(bp["boxes"][0])
        labels.append(col)

    ax.set_title(f"CFO distributions grouped by MAC (top {len(macs_sorted)} MACs)")
    ax.set_ylabel("CFO (Hz)")
    ax.set_xlabel("MAC (AdvA)")
    ax.set_xticks(base_positions)
    ax.set_xticklabels(macs_sorted, rotation=90, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    if handles:
        ax.legend(handles, labels, title="CFO feature", loc="upper right", frameon=True)

    # Optional: add a bit of headroom so n-labels don't clip
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.08 * (y1 - y0 + 1e-9))

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def plot_clusters_pca3d(seg_df: pd.DataFrame, cfo_cols: List[str], out_pdf: str) -> None:
    if seg_df is None or seg_df.empty:
        return

    seg_feature_cols = [f"seg_{c}" for c in cfo_cols if f"seg_{c}" in seg_df.columns]
    if len(seg_feature_cols) < 2:
        return

    X = seg_df[seg_feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y = seg_df["cluster_id"].values.astype(int)

    ok = np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    if X.shape[0] < 5:
        return

    mu = np.mean(X, axis=0)
    sig = np.maximum(np.std(X, axis=0, ddof=1), GLOBAL_NORM_MIN_STD)
    Xz = (X - mu) / sig

    pca = PCA(n_components=3, random_state=0)
    Xp = pca.fit_transform(Xz)

    counts = pd.Series(y).value_counts()
    top_clusters = set(counts.head(12).index.tolist())

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("tab20")
    for cid in np.unique(y):
        mask = (y == cid)
        pts = Xp[mask]
        if pts.shape[0] == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=10, alpha=0.7,
                   color=cmap(int(cid) % 20),
                   label=str(cid) if cid in top_clusters else None)

    evr = pca.explained_variance_ratio_
    ax.set_title(f"Clusters in 3-D PCA space (EVR: {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if len(top_clusters) > 0:
        ax.legend(title="Top clusters", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    # plt.show()
    plt.close(fig)

def plot_cluster_mac_counts(tracklets: List[Tracklet], out_pdf: str) -> None:
    if not tracklets:
        return

    ids = [t.id for t in sorted(tracklets, key=lambda x: x.id)]
    mac_counts = [len(t.macs) for t in sorted(tracklets, key=lambda x: x.id)]
    quarantined = [int(t.is_quarantined) for t in sorted(tracklets, key=lambda x: x.id)]

    fig_w = max(10.0, min(24.0, 0.45 * len(ids) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    ax.bar([str(i) for i in ids], mac_counts)
    ax.set_title("Unique MACs per cluster (quarantined clusters are benign-locked)")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("# MACs")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(axis="x", labelrotation=90)

    # annotate quarantined clusters
    for i, q in enumerate(quarantined):
        if q == 1:
            ax.text(i, mac_counts[i] + 0.2, "Q", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# --------------------------- reporting ---------------------------

def write_jsonl(path: str, items: List[Tracklet]) -> None:
    with open(path, "w") as f:
        for tau in items:
            f.write(json.dumps(tau.to_jsonable()) + "\n")

def compute_purity(seg_df: pd.DataFrame) -> float:
    if seg_df is None or seg_df.empty:
        return float("nan")
    total = len(seg_df)
    correct = 0
    for _, g in seg_df.groupby("cluster_id"):
        correct += int(g["mac"].value_counts().max())
    return float(correct) / float(total) if total > 0 else float("nan")

def print_cluster_stats(tracklets: List[Tracklet], max_macs: int = 20, max_keys: int = 20) -> None:
    if not tracklets:
        print("[stats] No clusters/tracklets to report.")
        return

    tracklets_sorted = sorted(tracklets, key=lambda t: (-t.score(), -(t.t_max - t.t_min), -t.n, t.id))

    print("\n" + "=" * 90)
    print(f"[stats] Clusters total: {len(tracklets_sorted)}")
    print("=" * 90)

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
        print(f"  score        : {info['score']}")
        print(f"  quarantined  : {info['is_quarantined']}")
        print(f"  n_segments   : {info['n_segments']}")
        print(f"  duration     : {info['duration']:.2f}s" if info["duration"] is not None else "  duration     : None")
        print(f"  cov_trace    : {info['cov_trace']:.3f}")
        print(f"  purity       : {info['purity']:.3f}  (entropy_norm={info['entropy_norm']:.3f})")
        print(f"  mu(z)        : [{mu_str}]")
        print(f"  |MACs|       : {len(macs)}")
        print(f"  |Keys|       : {len(keys)}")
        print("  MACs         : " + (", ".join(macs_show) if macs_show else "(none)") + macs_more)
        print(f"  Keys         : {', '.join(keys_show) if keys_show else '(none)'}{keys_more}")

# --------------------------- per-scenario run ---------------------------

def run_aircatch_on_csv(csv_path: str, out_dir: str,
                        do_mac_plots: bool = True,
                        do_cluster_plots: bool = True) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    # # CRC filter (keep your logic)
    # if _col_lookup_case_insensitive(df, "crc_ok"):
    #     crc_col = _col_lookup_case_insensitive(df, "crc_ok")
    #     df[crc_col] = pd.to_numeric(df[crc_col], errors="coerce").fillna(0).astype(int)
    #     before = len(df)
    #     df = df[df[crc_col] == 1].copy()
    #     after = len(df)
    #     print(f"[i] CRC filter: kept {after}/{before} rows where {crc_col}=1")

    time_col = resolve_time_column(df)
    mac_col = resolve_mac_column(df)
    payload_col = resolve_payload_column(df)
    cfo_cols = resolve_cfo_feature_columns(df)

    if REQUIRE_LOSTMODE_PREFIX and not payload_col:
        raise RuntimeError("REQUIRE_LOSTMODE_PREFIX=True but no payload column found in CSV.")

    # per-scenario plot paths
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    out_mac_box_pdf   = os.path.join(out_dir, f"{base}_macs_cfo_boxplot.pdf")
    out_mac_pca3d_pdf = os.path.join(out_dir, f"{base}_macs_pca3d.pdf")
    out_box_pdf       = os.path.join(out_dir, f"{base}_clusters_cfo_boxplot.pdf")
    out_pca3d_pdf      = os.path.join(out_dir, f"{base}_clusters_pca3d.pdf")
    out_macbar_pdf     = os.path.join(out_dir, f"{base}_clusters_mac_counts.pdf")
    out_mac_hist_pdf = os.path.join(out_dir, f"{base}_mac_histogram.pdf")

    plot_mac_histogram_with_adversary(df, mac_col=mac_col, payload_col=payload_col, out_pdf=out_mac_hist_pdf, top_n=50)

    if do_mac_plots:
        try:
            plot_mac_cfo_boxplot(df, time_col, mac_col, payload_col if payload_col else None, out_mac_box_pdf)
            plot_macs_pca3d(df, mac_col, payload_col if payload_col else None, out_mac_pca3d_pdf)
        except Exception as e:
            print(f"[!] MAC-level plots failed for {base}: {e}")

    params = AirCatchParams(
        p=SEGMENT_SIZE_P,
        dt=ASSOC_GAP_DT,
        gamma=GATE_GAMMA_Z_INIT if USE_ZSPACE else GATE_GAMMA_RAW_INIT,
        theta=THETA,
        eps=EPS,
        topk=TOPK_CANDIDATES,
    )

    # clusters, flagged, seg_df = aircatch_stream(
    #     df=df,
    #     time_col=time_col,
    #     mac_col=mac_col,
    #     payload_col=payload_col,
    #     cfo_cols=cfo_cols,
    #     params=params
    # )

    clusters, flagged, seg_df = aircatch_chunked(
        df=df,
        time_col=time_col,
        mac_col=mac_col,
        payload_col=payload_col,
        basic_clustering=BASIC_CLUSTERING,
        cfo_cols=cfo_cols,
        params=params,
        chunk_minutes=15.0,   # e.g., 15-minute chunks
    )

    flagged_ids = {t.id for t in flagged}
    out_purity_pdf = os.path.join(out_dir, f"{base}_cluster_purity_barplot.pdf")

    plot_cluster_purity_barplot_with_adv(
        df_raw=df,
        seg_df=seg_df,
        mac_col_raw=mac_col,
        payload_col_raw=payload_col,
        out_pdf=out_purity_pdf,
        flagged_ids=flagged_ids,
        adv_pattern_hex="4c001219ff"
    )

    # metrics
    n_clusters = len(clusters)
    n_flagged = len(flagged)
    n_segments = int(len(seg_df)) if seg_df is not None else 0
    purity = compute_purity(seg_df)

    n_quarantined = int(sum(1 for t in clusters if t.is_quarantined))
    n_hetero = int(sum(1 for t in clusters if t.is_heterogeneous()))

    # save outputs
    write_jsonl(os.path.join(out_dir, f"{base}_clusters.jsonl"), clusters)
    write_jsonl(os.path.join(out_dir, f"{base}_flagged.jsonl"), flagged)
    if seg_df is not None:
        seg_df.to_csv(os.path.join(out_dir, f"{base}_segments.csv"), index=False)

    if do_cluster_plots:
        try:
            plot_cluster_cfo_boxplot(seg_df, cfo_cols, out_box_pdf)
            plot_clusters_pca3d(seg_df, cfo_cols, out_pca3d_pdf)
            plot_cluster_mac_counts(clusters, out_macbar_pdf)
        except Exception as e:
            print(f"[!] Cluster-level plots failed for {base}: {e}")

    # optional console stats (can be noisy in batch)
    # print_cluster_stats(clusters, max_macs=30, max_keys=30)

    return {
        "scenario": base,
        "csv": csv_path,
        "n_clusters": n_clusters,
        "n_flagged": n_flagged,
        "n_segments": n_segments,
        "purity": purity,
        "n_quarantined": n_quarantined,
        "n_heterogeneous": n_hetero,
    }

# --------------------------- batch plots (PDF) ---------------------------

def _try_parse_mal_ben(s: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to parse scenario names like:
      mal_10_ben_20
      malicious_10_benign_20
    Returns (mal, ben) or (None, None) if not parseable.
    """
    m = re.search(r"(?:mal|malicious)_(\d+).*(?:ben|benign)_(\d+)", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def plot_batch_results(df: pd.DataFrame, out_root: str):
    os.makedirs(out_root, exist_ok=True)

    # If parseable, plot vs malicious count grouped by benign
    mals, bens = [], []
    for s in df["scenario"].astype(str).tolist():
        mal, ben = _try_parse_mal_ben(s)
        mals.append(mal)
        bens.append(ben)
    df = df.copy()
    df["malicious"] = mals
    df["benign"] = bens

    has_parse = df["malicious"].notna().all() and df["benign"].notna().all()

    if has_parse:
        # Purity vs malicious
        plt.figure(figsize=(6, 4))
        for b in sorted(df["benign"].unique()):
            g = df[df["benign"] == b].sort_values("malicious")
            plt.plot(g["malicious"], g["purity"], marker="o", label=f"benign={int(b)}")
        plt.xlabel("# malicious devices")
        plt.ylabel("Clustering purity (MAC-based)")
        plt.title("AirCatch clustering accuracy vs attackers")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "purity_vs_malicious.pdf"))
        plt.close()

        # Flagged clusters vs malicious
        plt.figure(figsize=(6, 4))
        for b in sorted(df["benign"].unique()):
            g = df[df["benign"] == b].sort_values("malicious")
            plt.plot(g["malicious"], g["n_flagged"], marker="o", label=f"benign={int(b)}")
        plt.xlabel("# malicious devices")
        plt.ylabel("# flagged clusters")
        plt.title("Detected tracking clusters vs attackers")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "flagged_vs_malicious.pdf"))
        plt.close()

        # Quarantined clusters vs malicious
        plt.figure(figsize=(6, 4))
        for b in sorted(df["benign"].unique()):
            g = df[df["benign"] == b].sort_values("malicious")
            plt.plot(g["malicious"], g["n_quarantined"], marker="o", label=f"benign={int(b)}")
        plt.xlabel("# malicious devices")
        plt.ylabel("# quarantined benign clusters")
        plt.title("Benign quarantine effectiveness vs attackers")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "quarantined_vs_malicious.pdf"))
        plt.close()

    else:
        # Fallback: scenario index on x-axis
        df = df.sort_values("scenario")
        x = np.arange(len(df))

        plt.figure(figsize=(8, 3.8))
        plt.plot(x, df["purity"].values, marker="o")
        plt.xticks(x, df["scenario"].astype(str).tolist(), rotation=90, fontsize=7)
        plt.ylabel("Purity")
        plt.title("AirCatch clustering purity across scenarios")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "purity_across_scenarios.pdf"))
        plt.close()

        plt.figure(figsize=(8, 3.8))
        plt.plot(x, df["n_flagged"].values, marker="o")
        plt.xticks(x, df["scenario"].astype(str).tolist(), rotation=90, fontsize=7)
        plt.ylabel("# flagged clusters")
        plt.title("Flagged clusters across scenarios")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "flagged_across_scenarios.pdf"))
        plt.close()

        plt.figure(figsize=(8, 3.8))
        plt.plot(x, df["n_quarantined"].values, marker="o")
        plt.xticks(x, df["scenario"].astype(str).tolist(), rotation=90, fontsize=7)
        plt.ylabel("# quarantined clusters")
        plt.title("Quarantined benign clusters across scenarios")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "quarantined_across_scenarios.pdf"))
        plt.close()

    print("[✓] Saved batch evaluation plots (PDF)")

def plot_mac_histogram_with_adversary(df_raw: pd.DataFrame,
                                      mac_col: str,
                                      payload_col: Optional[str],
                                      out_pdf: str,
                                      top_n: int = 50) -> None:
    """
    Histogram (bar chart) of MACs seen in the dataset.

    - If payload contains substring "4c001219ff" (case-insensitive, hex-normalized),
      treat that row's MAC as belonging to a single "Adversary MACs" bucket.
    - Everything else is counted per-MAC.
    - Saves as PDF (like other plots).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Input dataframe (packets/rows).
    mac_col : str
        Column name containing MAC/AdvA.
    payload_col : Optional[str]
        Column name containing payload hex string. If None/not found, adversary bucketing is skipped.
    out_pdf : str
        Output path for the PDF.
    top_n : int
        Show top-N bars (after bucketing). Remaining are summed into "Other".
    """
    if df_raw is None or df_raw.empty:
        return

    dfp = df_raw.copy()
    if mac_col not in dfp.columns:
        return

    dfp[mac_col] = dfp[mac_col].astype(str)
    dfp = dfp[dfp[mac_col].str.len() > 0].copy()
    if dfp.empty:
        return

    adv_pat = "4c001219ff"

    def _is_adv_payload(x: object) -> bool:
        if not isinstance(x, str):
            return False
        s = x.strip().lower()
        if s.startswith("0x"):
            s = s[2:]
        # keep only hex chars
        s = "".join(ch for ch in s if ch in "0123456789abcdef")
        return adv_pat in s

    # Build label series: either "Adversary MACs" or the actual MAC
    if payload_col and (payload_col in dfp.columns):
        is_adv = dfp[payload_col].apply(_is_adv_payload)
        labels = dfp[mac_col].where(~is_adv, other="Adversary MACs")
    else:
        labels = dfp[mac_col]

    counts = labels.value_counts()

    # Keep top_n, collapse rest into "Other" (but keep "Adversary MACs" if it exists)
    kept = counts.head(int(top_n)).copy()

    if "Adversary MACs" in counts.index and "Adversary MACs" not in kept.index:
        # Ensure adversary is always shown (swap in by dropping last)
        if len(kept) > 0:
            kept = kept.iloc[:-1]
        kept.loc["Adversary MACs"] = int(counts.loc["Adversary MACs"])
        kept = kept.sort_values(ascending=False)

    other_sum = int(counts.iloc[int(top_n):].sum()) if len(counts) > int(top_n) else 0
    if other_sum > 0:
        kept.loc["Other"] = other_sum

    kept = kept.sort_values(ascending=False)

    # Plot
    fig_w = max(10.0, min(26.0, 0.40 * len(kept) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    x = np.arange(len(kept), dtype=float)
    ax.bar(x, kept.values)

    ax.set_title("Histogram of observed MACs (payload '4c001219FF' grouped as Adversary MACs)")
    ax.set_xlabel("MAC label")
    ax.set_ylabel("# packets (rows)")

    ax.set_xticks(x)
    ax.set_xticklabels(kept.index.tolist(), rotation=90, fontsize=8)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    # Annotate counts on top of bars (optional but useful)
    y0, y1 = ax.get_ylim()
    dy = 0.01 * (y1 - y0 + 1e-9)
    for i, v in enumerate(kept.values):
        ax.text(i, float(v) + dy, str(int(v)), ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# --------------------------- main batch ---------------------------

def run_batch(input_path: str, out_root: str, do_mac_plots: bool, do_cluster_plots: bool):
    os.makedirs(out_root, exist_ok=True)

    if os.path.isdir(input_path):
        csvs = sorted(glob.glob(os.path.join(input_path, "*.csv")))
    else:
        csvs = [input_path]

    if not csvs:
        raise RuntimeError(f"No CSV files found under: {input_path}")

    results = []
    per_scenario_root = os.path.join(out_root, "per_scenario")
    os.makedirs(per_scenario_root, exist_ok=True)

    for csv in csvs:
        base = os.path.splitext(os.path.basename(csv))[0]
        scenario_out = os.path.join(per_scenario_root, base)
        print(f"\n=== Running AirCatch on {csv} ===")
        res = run_aircatch_on_csv(csv, scenario_out, do_mac_plots=do_mac_plots, do_cluster_plots=do_cluster_plots)
        results.append(res)

        # break  # for quick testing; remove to run all

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_root, "summary_metrics.csv"), index=False)
    print(f"[✓] Saved: {os.path.join(out_root, 'summary_metrics.csv')}")

    plot_batch_results(df_res, out_root)

def main() -> None:
    args = parse_args()
    run_batch(
        input_path=args.input,
        out_root=args.out,
        do_mac_plots=(not args.no_mac_plots),
        do_cluster_plots=(not args.no_cluster_plots),
    )

if __name__ == "__main__":
    main()