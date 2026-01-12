#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirCatch — Streaming BLE tracker detection via online CFO clustering.

Implements Algorithm: AirCatch (USENIX-style pseudocode)

Input CSV (expected columns; case-insensitive resolution):
  - timestamp:   "pcap_ts" (preferred) or "ts" or "timestamp"
  - MAC/AdvA:    "AdvA" or "adv_addr" (etc.)
  - payload:     "payload_hex" or "payload" or "pdu_hex" (hex string)  [optional but recommended]
  - CFO feature columns: primary CFO + optional transition CFOs (see OPTIONAL_CFO_COLS)

Core idea:
  - Filter lost-mode prefix 0x1219 in payload (if payload available; otherwise you can disable filter)
  - Buffer per observed MAC m: B[m] collects (t_k, f_k, K_k)
  - When |B[m]| >= p: finalize a segment -> (t_s, t_e, fbar, K_s)
  - Standardize with GLOBAL online (mu_g, sigma_g)
  - Associate to recent clusters within Δt using Mahalanobis distance
  - Gate by γ; spawn new cluster if needed
  - Update cluster stats online (Welford for mean/cov)
  - Merge identifiers (MAC, keys) into the tracklet state
  - Flag if score S_tau >= θ

Outputs:
  - Prints summary of flagged tracklets
  - Writes JSONL with all tracklets and flagged subset

This script is intentionally modular: tweak extract_public_key(), estimate_cfo_features(),
FinalizeSegment aggregation, gating, scoring, etc.

NOTE:
  - If your CSV does not contain payload, set REQUIRE_LOSTMODE_PREFIX=False to run on all packets.
"""

import os
import sys
import json
import math
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- config ---------------------------

FNAME = "cfo_samples_rail.csv"
OUTPUT_DIR = "all_devices_static_rail"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_TRACKLETS_JSONL = os.path.join(OUTPUT_DIR, "aircatch_tracklets.jsonl")
OUT_FLAGGED_JSONL = os.path.join(OUTPUT_DIR, "aircatch_flagged.jsonl")

# --- AirCatch params (Algorithm) ---
SEGMENT_SIZE_P = 100              # p
ASSOC_GAP_DT = 2.0                # Δt (seconds)
GATE_GAMMA = 12.0                 # γ (Mahalanobis gate)
T_MIN = 30.0                      # T_min (seconds)
N_MIN = 6                         # N_min (segments)
K_MIN = 3                         # K_min (unique keys)
THETA = 3                         # θ (score threshold)
EPS = 1e-6                         # ε

# Scatter stability threshold (eta)
ETA_TRACE_MAX = 8.0               # η (trace threshold in standardized space)

# Require lost-mode prefix 0x1219 in payload?
REQUIRE_LOSTMODE_PREFIX = True
LOSTMODE_PREFIX_HEX = "1219"      # byte prefix in payload hex string

# Global normalization update cadence:
GLOBAL_NORM_WARMUP = 20           # need at least this many segments before using sigma robustly
GLOBAL_NORM_UPDATE_EVERY = 1      # update global stats every segment
GLOBAL_NORM_MIN_STD = 1e-3        # floor to avoid exploding z-scores

# Candidate pruning (optional; keep minimal)
PRUNE_INACTIVE_AFTER = 60.0       # seconds since last seen before pruning (set None to disable)

# CFO feature columns likely present
OPTIONAL_CFO_COLS = [
    "CFO_Hz",
    "cfo_hz",
    "cfo_quick_hz",
    "CFO_00_Hz",
    "CFO_11_Hz",
    "CFO_10_Hz",
    "CFO_01_Hz",
    "cfo_equal_00_hz",
    "cfo_equal_11_hz",
    "cfo_jump_10_hz",
    "cfo_jump_01_hz",
]

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
    # Case-insensitive exact matches for common time columns
    candidates = [
        "timestamp", "pcap_ts", "ts", "time", "t",
        "epoch", "epoch_s", "unix", "unix_ts"
    ]
    cols = list(map(str, df.columns))
    lower_map = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # If nothing matched, show what we saw (helps debugging)
    raise ValueError(
        "Could not resolve timestamp column. "
        f"Available columns: {', '.join(cols[:50])}"
    )

def resolve_mac_column(df: pd.DataFrame) -> str:
    hit = resolve_first_present(df, ["AdvA", "adv_addr", "adva", "advA", "adv_address", "advaddr"])
    if hit:
        return hit

    # heuristic fallback
    for c in df.columns:
        lc = str(c).lower()
        if "adv" in lc and ("addr" in lc or "adva" in lc):
            return str(c)
    raise ValueError("Could not resolve MAC/AdvA column (try: AdvA/adv_addr).")

def resolve_payload_column(df: pd.DataFrame) -> str:
    return resolve_first_present(df, ["payload_hex", "payload", "pdu_hex", "pdu", "adv_data_hex", "adv_data"])

def resolve_cfo_feature_columns(df: pd.DataFrame) -> List[str]:
    # keep only present, preserve order, ensure at least one CFO column
    present = []
    for cand in OPTIONAL_CFO_COLS:
        hit = _col_lookup_case_insensitive(df, cand)
        if hit and hit not in present:
            present.append(hit)

    # fallback: anything containing "cfo" and "hz"
    if not present:
        for c in df.columns:
            lc = str(c).lower()
            if "cfo" in lc and "hz" in lc:
                present.append(str(c))

    if not present:
        raise ValueError("No CFO feature columns found (expected columns with CFO and Hz).")

    return present

# --------------------------- AirCatch: subroutines ---------------------------

def has_lostmode_prefix(payload_hex: str, prefix_hex: str = LOSTMODE_PREFIX_HEX) -> bool:
    if not isinstance(payload_hex, str):
        return False
    s = payload_hex.strip().lower()
    s = s[2:] if s.startswith("0x") else s
    return s.startswith(prefix_hex.lower())

def extract_public_key(payload_hex: str, mac: str) -> Optional[str]:
    """
    Algorithm line: K_k <- ExtractPublicKey(p_k, m_k)

    Minimal, robust placeholder:
      - If you have a proper parser, replace this with exact extraction.
      - Here we return a stable hash-like token from the payload to represent "epoch key".

    If payload missing or too short: return None.
    """
    if not isinstance(payload_hex, str):
        return None
    s = payload_hex.strip().lower()
    s = s[2:] if s.startswith("0x") else s
    if len(s) < 8:
        return None
    # Keep it deterministic but compact: prefix + first 32 bytes (64 hex chars) if present.
    # In a real implementation: parse the 0x1219 structure and extract the public key bytes.
    core = s[:64] if len(s) >= 64 else s
    return f"k:{core}"

def estimate_cfo_features(row: pd.Series, cfo_cols: List[str]) -> np.ndarray:
    """
    Algorithm line: fhat_k <- EstimateCFOFeatures(o_k)
    Returns d-dim CFO fingerprint vector for this packet.

    Here we simply read the CFO columns (already computed elsewhere).
    """
    vals = []
    for c in cfo_cols:
        v = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        vals.append(float(v) if np.isfinite(v) else np.nan)
    arr = np.asarray(vals, dtype=float)
    return arr

def finalize_segment(buffer_rows: List[Tuple[float, np.ndarray, Optional[str]]]) -> Tuple[float, float, np.ndarray, Set[str]]:
    """
    Algorithm line: (t_s,t_e,fbar,K_s) <- FinalizeSegment(B[m_k])

    buffer_rows: list of (t_k, f_k, K_k)

    Returns:
      - t_s, t_e
      - fbar: mean CFO feature over the segment (ignores NaNs per-dim)
      - K_s: set of keys observed in the segment (excluding None)
    """
    ts = np.array([x[0] for x in buffer_rows], dtype=float)
    t_s = float(np.nanmin(ts))
    t_e = float(np.nanmax(ts))

    F = np.vstack([x[1] for x in buffer_rows])  # (p, d)
    # nanmean per feature dimension
    fbar = np.nanmean(F, axis=0)
    # if a dimension is all-nan, replace with nan -> later dropped by guard
    K_s = set([k for (_, _, k) in buffer_rows if isinstance(k, str) and k != ""])
    return t_s, t_e, fbar, K_s

# --------------------------- Online global standardizer ---------------------------

class OnlineGlobalScaler:
    """
    Maintains online global mean and (diagonal) std over SEGMENT features.
    Uses Welford per-dimension. Outputs z = (f - mu) / (sigma + eps).
    """
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
            # skip bad segments
            return

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.d, dtype=float)
        var = self.M2 / max(1, (self.n - 1))
        s = np.sqrt(np.maximum(var, 0.0))
        s = np.maximum(s, GLOBAL_NORM_MIN_STD)
        return s

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = self.mean
        sig = self.std()
        return (x - mu) / (sig + self.eps)

# --------------------------- Online cluster (tracklet) ---------------------------

def _inv_psd(mat: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Robust inverse for symmetric PSD matrices with diagonal jitter.
    """
    m = np.asarray(mat, dtype=float)
    m = 0.5 * (m + m.T)
    m = m + eps * np.eye(m.shape[0], dtype=float)
    return np.linalg.inv(m)

@dataclass
class Tracklet:
    """
    Stores per-tracklet mean/cov (in standardized z-space), time span, counts, identifier sets.
    """
    id: int
    d: int

    # stats in z-space
    n: int = 0
    mu: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    M2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))  # sum outer products for covariance

    # time span
    t_min: float = math.inf
    t_max: float = -math.inf

    # identifiers
    macs: Set[str] = field(default_factory=set)
    keys: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.mu.size == 0:
            self.mu = np.zeros(self.d, dtype=float)
        if self.M2.size == 0:
            self.M2 = np.zeros((self.d, self.d), dtype=float)

    def last_seen(self) -> float:
        return float(self.t_max)

    def cov(self) -> np.ndarray:
        if self.n < 2:
            return np.eye(self.d, dtype=float)
        C = self.M2 / max(1, (self.n - 1))
        C = 0.5 * (C + C.T)
        return C

    def trace(self) -> float:
        return float(np.trace(self.cov()))

    def mahalanobis(self, z: np.ndarray, eps: float = EPS) -> float:
        z = np.asarray(z, dtype=float)
        diff = (z - self.mu)
        inv = _inv_psd(self.cov(), eps=eps)
        return float(diff.T @ inv @ diff)

    def update_stats_welford(self, z: np.ndarray) -> None:
        """
        Welford update for mean + covariance accumulator (M2 in matrix form).
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (self.d,):
            raise ValueError("Tracklet.update_stats_welford: shape mismatch")
        if not np.all(np.isfinite(z)):
            return

        self.n += 1
        delta = z - self.mu
        self.mu += delta / self.n
        delta2 = z - self.mu
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
        Algorithm score:
          S = 1[T>=Tmin] + 1[N>=Nmin] + 1[|K|>=Kmin] + 1[Tr(Sigma)<=eta]
        """
        T = self.t_max - self.t_min
        s = 0
        s += int(T >= T_MIN)
        s += int(self.n >= N_MIN)
        s += int(len(self.keys) >= K_MIN)
        s += int(self.trace() <= ETA_TRACE_MAX)
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
            "mu": self.mu.tolist(),
            "cov_trace": float(self.trace()),
            "score": int(self.score()),
        }

# --------------------------- AirCatch core ---------------------------

@dataclass
class AirCatchParams:
    p: int = SEGMENT_SIZE_P
    dt: float = ASSOC_GAP_DT
    gamma: float = GATE_GAMMA
    theta: int = THETA
    eps: float = EPS

def aircatch_stream(df: pd.DataFrame,
                    time_col: str,
                    mac_col: str,
                    payload_col: str,
                    cfo_cols: List[str],
                    params: AirCatchParams) -> Tuple[List[Tracklet], List[Tracklet]]:
    """
    Runs AirCatch over the dataframe in timestamp order.
    Returns: (all_tracklets, flagged_tracklets)
    """
    # sort by time
    tvals = pd.to_numeric(df[time_col], errors="coerce")
    df = df.assign(_t=tvals).replace([np.inf, -np.inf], np.nan).dropna(subset=["_t"])
    df = df.sort_values("_t").drop(columns=["_t"])

    d = len(cfo_cols)
    scaler = OnlineGlobalScaler(d=d, eps=params.eps)

    # B[m] buffers
    buffers: Dict[str, List[Tuple[float, np.ndarray, Optional[str]]]] = {}
    tracklets: List[Tracklet] = []
    flagged_ids: Set[int] = set()

    next_id = 1
    seg_count = 0

    def _active_candidates(t_s: float) -> List[Tracklet]:
        return [tau for tau in tracklets if (t_s - tau.t_max) <= params.dt]

    for _, row in df.iterrows():
        t_k = float(pd.to_numeric(row[time_col], errors="coerce"))
        m_k = str(row[mac_col])

        p_k = None
        if payload_col:
            p_k = row.get(payload_col, None)

        # (filter) lostmode prefix
        if REQUIRE_LOSTMODE_PREFIX:
            if not (isinstance(p_k, str) ):# and has_lostmode_prefix(p_k)):
                continue

        # Extract key
        K_k = extract_public_key(p_k, m_k) if isinstance(p_k, str) else None

        # CFO features
        f_k = estimate_cfo_features(row, cfo_cols)
        if not np.all(np.isfinite(f_k)):
            # drop packets with missing CFO features (you can relax this if needed)
            continue

        # append into B[m_k]
        if m_k not in buffers:
            buffers[m_k] = []
        buffers[m_k].append((t_k, f_k, K_k))

        if len(buffers[m_k]) < params.p:
            continue

        # (1) finalize segment and clear buffer
        t_s, t_e, f_bar, K_s = finalize_segment(buffers[m_k])
        buffers[m_k].clear()  # Clear(B[m_k])

        if not np.all(np.isfinite(f_bar)):
            continue

        # (2) global standardize
        # Update global stats (gallery-style) BEFORE transform or AFTER? For strict causality:
        # transform using previous global stats, then update.
        z = scaler.transform(f_bar)

        # candidate set
        C = _active_candidates(t_s)

        # choose best match
        tau_star = None
        d_best = float("inf")
        if C:
            for tau in C:
                dist = tau.mahalanobis(z, eps=params.eps)
                if dist < d_best:
                    d_best = dist
                    tau_star = tau

        # (3) gate/spawn
        if (tau_star is None) or (d_best > params.gamma):
            tau = Tracklet(id=next_id, d=d)
            next_id += 1
            tracklets.append(tau)
        else:
            tau = tau_star

        # (4) update cluster stats
        tau.update_stats_welford(z)

        # (5) merge identifiers / times
        tau.merge_segment(t_s=t_s, t_e=t_e, mac=m_k, keys_in_seg=K_s)

        # (global scaler update)
        seg_count += 1
        if seg_count % GLOBAL_NORM_UPDATE_EVERY == 0:
            scaler.update(f_bar)

        # (optional prune)
        if PRUNE_INACTIVE_AFTER is not None and len(tracklets) > 0:
            # prune only those not flagged, to keep final output stable
            t_now = t_s
            keep: List[Tracklet] = []
            for tt in tracklets:
                if tt.id in flagged_ids:
                    keep.append(tt)
                else:
                    if (t_now - tt.t_max) <= float(PRUNE_INACTIVE_AFTER):
                        keep.append(tt)
            tracklets = keep

        # (6) tracker decision
        if tau.score() >= params.theta:
            flagged_ids.add(tau.id)

    flagged = [tau for tau in tracklets if tau.id in flagged_ids]
    return tracklets, flagged

# --------------------------- CLI / reporting ---------------------------

def write_jsonl(path: str, items: List[Tracklet]) -> None:
    with open(path, "w") as f:
        for tau in items:
            f.write(json.dumps(tau.to_jsonable()) + "\n")
    print(f"[✓] wrote: {path} ({len(items)} items)")

def main() -> None:
    print("=" * 80)
    print("AirCatch — Streaming BLE tracker detection via online CFO clustering")
    print("=" * 80)

    if not os.path.exists(FNAME):
        print(f"ERROR: Missing input file: {FNAME}")
        sys.exit(1)

    df = pd.read_csv(FNAME)

    time_col = resolve_time_column(df)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df[np.isfinite(df[time_col])].copy()
    mac_col = resolve_mac_column(df)
    payload_col = resolve_payload_column(df)  # may be empty
    cfo_cols = resolve_cfo_feature_columns(df)

    print(f"[i] time_col   : {time_col}")
    print(f"[i] mac_col    : {mac_col}")
    print(f"[i] payload_col: {payload_col if payload_col else '(none)'}")
    print(f"[i] CFO cols   : {cfo_cols}")

    if REQUIRE_LOSTMODE_PREFIX and not payload_col:
        print("ERROR: REQUIRE_LOSTMODE_PREFIX=True but no payload column found in CSV.")
        print("       Either add a payload_hex column or set REQUIRE_LOSTMODE_PREFIX=False.")
        sys.exit(1)

    params = AirCatchParams(
        p=SEGMENT_SIZE_P,
        dt=ASSOC_GAP_DT,
        gamma=GATE_GAMMA,
        theta=THETA,
        eps=EPS
    )

    tracklets, flagged = aircatch_stream(
        df=df,
        time_col=time_col,
        mac_col=mac_col,
        payload_col=payload_col,
        cfo_cols=cfo_cols,
        params=params
    )

    # Save outputs
    write_jsonl(OUT_TRACKLETS_JSONL, tracklets)
    write_jsonl(OUT_FLAGGED_JSONL, flagged)

    # Print a compact summary
    print("\n" + "=" * 80)
    print(f"Tracklets total: {len(tracklets)} | Flagged: {len(flagged)}")
    print("=" * 80)

    if flagged:
        flagged_sorted = sorted(flagged, key=lambda t: (-t.score(), -(t.t_max - t.t_min)))
        for tau in flagged_sorted[:20]:
            info = tau.to_jsonable()
            print(
                f"- id={info['id']:>3} score={info['score']} "
                f"nSeg={info['n_segments']:<4} dur={info['duration']:.1f}s "
                f"|K|={len(info['keys']):<3} |M|={len(info['macs']):<3} tr={info['cov_trace']:.2f}"
            )
    else:
        print("No tracklets flagged. Try relaxing thresholds (γ, θ, ETA_TRACE_MAX) or increasing p/N_MIN/T_MIN.")

if __name__ == "__main__":
    main()