#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLE Device Fingerprinting using CFO Statistics (grouped by MAC/AdvA)

NORMAL MODE (stress mode OFF):
  - Groups packets by MAC/AdvA
  - Temporal split per MAC: first 70% -> train, last 30% -> test
  - Builds MANY samples per MAC by windowing packets inside each split
  - Trains RandomForest classifier to predict MAC
  - Outputs confusion matrix + feature heatmap + report

STRESS MODE (fresh pseudonyms; MAC rotates / labels not reusable):
  - Within each MAC, sort packets by time then segment into consecutive chunks of length p
    (each chunk is treated as a fresh pseudonym segment; never reused)
  - Split by segment time: first 70% segments -> gallery, last 30% -> query
  - NO supervised classifier on pseudonyms
  - Evaluate LINKABILITY using ground-truth MAC only for scoring:
      * Rank-1 linking accuracy (nearest neighbor in gallery has same MAC)
      * Verification ROC-AUC (best genuine vs best impostor similarity per query)
      * Clustering ARI/NMI (KMeans; unsupervised; MAC used only for scoring)
  - Additionally, sweep p (1..p_max etc.) and plot Rank-1 (%) + AUC (%) vs p (PDF)

Input:  /home/mishra/Downloads/cfo_samples_rail.csv
Outputs:
  - all_devices_static_rail/ble_fingerprint_results.txt
  - all_devices_static_rail/ble_fingerprint_confusion_matrix.png (normal mode only)
  - all_devices_static_rail/ble_fingerprint_feature_distribution.png
  - all_devices_static_rail/ble_fingerprint_stress_linkability.pdf (stress mode only)
"""

import os
import sys
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- config ---------------------------

FNAME = "cfo_samples_rail.csv"/

OUTPUT_DIR = "all_devices_static_rail"

# NOTE: resolved at runtime to "AdvA" or "adv_addr" (or case-insensitive match)
MAC_COL = "AdvA"

def run_aircatch_on_csv(csv_path: str, out_dir: str) -> Dict[str, Any]:
    """
    Runs AirCatch on ONE CSV and returns summary metrics.
    """
    df = pd.read_csv(csv_path)

    # CRC filter (unchanged logic)
    if _col_lookup_case_insensitive(df, "crc_ok"):
        crc_col = _col_lookup_case_insensitive(df, "crc_ok")
        df[crc_col] = pd.to_numeric(df[crc_col], errors="coerce").fillna(0).astype(int)
        df = df[df[crc_col] == 1].copy()

    time_col = resolve_time_column(df)
    mac_col = resolve_mac_column(df)
    payload_col = resolve_payload_column(df)
    cfo_cols = resolve_cfo_feature_columns(df)

    params = AirCatchParams(
        p=SEGMENT_SIZE_P,
        dt=ASSOC_GAP_DT,
        gamma=GATE_GAMMA_Z_INIT if USE_ZSPACE else GATE_GAMMA_RAW_INIT,
        theta=THETA,
        eps=EPS,
    )

    clusters, flagged, seg_df = aircatch_stream(
        df=df,
        time_col=time_col,
        mac_col=mac_col,
        payload_col=payload_col,
        cfo_cols=cfo_cols,
        params=params
    )

    # ---- metrics ----
    n_clusters = len(clusters)
    n_flagged = len(flagged)
    n_segments = len(seg_df)

    # MAC-purity accuracy
    total = len(seg_df)
    correct = 0
    for _, g in seg_df.groupby("cluster_id"):
        correct += g["mac"].value_counts().max()
    purity = correct / total if total > 0 else np.nan

    # Save per-scenario outputs
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    write_jsonl(os.path.join(out_dir, f"{base}_clusters.jsonl"), clusters)
    write_jsonl(os.path.join(out_dir, f"{base}_flagged.jsonl"), flagged)
    seg_df.to_csv(os.path.join(out_dir, f"{base}_segments.csv"), index=False)

    return {
        "scenario": base,
        "csv": csv_path,
        "n_clusters": n_clusters,
        "n_flagged": n_flagged,
        "n_segments": n_segments,
        "purity": purity,
    }

TRAIN_FRACTION = 0.7          # temporal split within each MAC
MIN_PKTS_PER_MAC = 200        # minimum usable packets per MAC (after dropping NaNs) to keep class

# NORMAL MODE ONLY: create many samples per MAC using fixed packet windows inside train/test split
WINDOW_PKTS = 100             # packets per sample-window
WINDOW_STEP = 10              # step between windows (==WINDOW_PKTS -> non-overlapping)
MIN_WINDOWS_PER_MAC = 2       # require at least this many train windows AND test windows per MAC

# --------------------------- test CSV generation ---------------------------

GEN_OUT_DIR = "generated_test_csvs"
MAX_MALICIOUS = 5
MAX_BENIGN = 11
MALICIOUS_PREFIX = "MAL_"
BENIGN_PREFIX = "BEN_"

# mean repetition to bias RF toward mean (via feature subsampling)
MEAN_REPEATS = 4

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_RESULTS = os.path.join(OUTPUT_DIR, "ble_fingerprint_results.txt")
OUT_CONFUSION = os.path.join(OUTPUT_DIR, "ble_fingerprint_confusion_matrix.png")
OUT_DISTRIBUTION = os.path.join(OUTPUT_DIR, "ble_fingerprint_feature_distribution.png")
OUT_STRESS_PLOT = os.path.join(OUTPUT_DIR, "ble_fingerprint_stress_linkability.pdf")

# Transition CFO columns likely present in cfo_samples_rail.csv (plus fallbacks)
OPTIONAL_CFO_COLS = [
    "CFO_00_Hz",
    "CFO_11_Hz",
    "CFO_10_Hz",
    "CFO_01_Hz",
    "CFO_from_transitions_Hz",
]

# Keep extra stats SMALL and robust
EXTRA_STATS = (
    "median",
    "iqr",
    "p10",
    "p90",
    "mad",
)

HEATMAP_MAX_ROWS = 200  # keep plots readable

# --------------------------- helpers ---------------------------

def _fresh_mac(idx: int) -> str:
    # Deterministic, locally-administered MACs
    return f"02:00:00:{(idx >> 16) & 0xff:02x}:{(idx >> 8) & 0xff:02x}:{idx & 0xff:02x}"

def generate_test_csvs(df: pd.DataFrame, mac_col: str):
    os.makedirs(GEN_OUT_DIR, exist_ok=True)

    all_macs = df[mac_col].dropna().astype(str).unique().tolist()
    if len(all_macs) < MAX_BENIGN + MAX_MALICIOUS:
        raise ValueError("Not enough devices in input CSV to generate all scenarios")

    print("\nGenerating test CSVs for malicious / benign scenarios...")

    for n_mal in range(1, MAX_MALICIOUS + 1):
        for n_ben in range(1, MAX_BENIGN + 1):

            selected_mal = all_macs[:n_mal]
            selected_ben = all_macs[n_mal:n_mal + n_ben]

            rows = []
            mac_counter = 0

            # ---- benign devices: unchanged ----
            for mac in selected_ben:
                rows.append(df[df[mac_col] == mac])

            # ---- malicious devices: p = 1 MAC churn ----
            for mac in selected_mal:
                df_m = df[df[mac_col] == mac].copy()

                new_macs = []
                for _ in range(len(df_m)):
                    new_macs.append(_fresh_mac(mac_counter))
                    mac_counter += 1

                df_m[mac_col] = new_macs
                rows.append(df_m)

            out_df = pd.concat(rows, axis=0).reset_index(drop=True)

            out_name = f"mal_{n_mal}_ben_{n_ben}.csv"
            out_path = os.path.join(GEN_OUT_DIR, out_name)
            out_df.to_csv(out_path, index=False)

            print(f"[✓] Generated {out_path} "
                  f"(malicious={n_mal}, benign={n_ben}, rows={len(out_df)})")

def _col_lookup_case_insensitive(df: pd.DataFrame, name: str) -> str:
    target = name.lower()
    for c in df.columns:
        if str(c).lower() == target:
            return str(c)
    return ""


def resolve_mac_column(df: pd.DataFrame) -> str:
    for cand in ["AdvA", "adv_addr", "advA", "ADV_A", "adv_address", "advaddr"]:
        hit = _col_lookup_case_insensitive(df, cand)
        if hit:
            return hit

    lowers = [str(c).lower() for c in df.columns]
    for c, lc in zip(df.columns, lowers):
        if ("adv" in lc) and ("addr" in lc or lc.endswith("a") or "adva" in lc):
            return str(c)

    raise ValueError("Could not find MAC/AdvA column (expected 'AdvA' or 'adv_addr').")


def find_primary_cfo_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lower = [str(c).lower() for c in cols]

    for c, lc in zip(cols, lower):
        if lc in ["cfo_hz", "cfo_quick_hz", "est_cfo_hz", "cfo_exact_quick_hz"]:
            return str(c)

    for c, lc in zip(cols, lower):
        if "cfo" in lc and "hz" in lc:
            return str(c)

    raise ValueError("No CFO column found in CSV")


def get_feature_choice() -> Tuple[bool, bool]:
    print("\n" + "=" * 80)
    print("STATISTIC SELECTION")
    print("=" * 80)
    print("\nWhich statistics would you like to use per selected CFO column?")
    print("  1) Mean only  (mean is up-weighted via repetition)")
    print("  2) Standard Deviation only")
    print("  3) Both Mean and Standard Deviation")
    print("\nNOTE: Additional robust distribution stats are ALWAYS included:")
    print("      " + ", ".join(EXTRA_STATS))

    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                print("\n✓ Selected: Mean only (+ extra distribution stats)")
                return True, False
            if choice == "2":
                print("\n✓ Selected: Std Dev only (+ extra distribution stats)")
                return False, True
            if choice == "3":
                print("\n✓ Selected: Mean and Std Dev (+ extra distribution stats)")
                return True, True
            print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            sys.exit(0)


def choose_cfo_columns(df: pd.DataFrame) -> List[str]:
    primary = find_primary_cfo_column(df)
    available = set(map(str, df.columns))

    choices = [(primary, True)]
    for c in OPTIONAL_CFO_COLS:
        hit = _col_lookup_case_insensitive(df, c)
        if hit and hit in available and hit != primary:
            choices.append((hit, False))

    print("\n" + "=" * 80)
    print("CFO TYPE SELECTION")
    print("=" * 80)
    print("\nSelect which CFO types to use (features computed per column).")
    print("Enter a comma-separated list of numbers, e.g., 1,3,4")
    print("Press Enter for default (primary CFO only).\n")

    for i, (col, default_on) in enumerate(choices, 1):
        tag = "DEFAULT" if default_on else ""
        print(f"  {i}) {col} {tag}")

    while True:
        try:
            raw = input("\nYour selection: ").strip()
            if raw == "":
                selected = [primary]
                print(f"\n✓ Selected default: [{primary}]")
                return selected

            idxs = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                idxs.append(int(part))
            idxs = sorted(set(idxs))

            if not idxs:
                print("Please choose at least one number.")
                continue

            if min(idxs) < 1 or max(idxs) > len(choices):
                print(f"Invalid selection. Choose numbers in 1..{len(choices)}")
                continue

            selected = [choices[i - 1][0] for i in idxs]
            print("\n✓ Selected CFO columns:")
            for c in selected:
                print(f"  - {c}")
            return selected
        except ValueError:
            print("Invalid input. Use numbers like: 1,2,3")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            sys.exit(0)


def get_stress_mode_choice() -> Tuple[bool, Optional[int]]:
    print("\n" + "=" * 80)
    print("PSEUDONYM STRESS MODE")
    print("=" * 80)
    print("\nEnable pseudonym rotation stress mode?")
    print("  - If enabled: within each MAC, after every p packets, a new pseudonym segment is assigned.")
    print("  - Pseudonyms are assumed fresh (never reused).")
    print("  - We evaluate LINKABILITY (not supervised classification).")

    while True:
        try:
            raw = input("\nEnable stress mode? (y/n): ").strip().lower()
            if raw in ["n", "no"]:
                print("\n✓ Stress mode: OFF")
                return False, None
            if raw in ["y", "yes"]:
                while True:
                    p_raw = input("Enter p (pseudonym changes every p packets, p>=1): ").strip()
                    try:
                        p = int(p_raw)
                        if p < 1:
                            print("p must be >= 1.")
                            continue
                        print(f"\n✓ Stress mode: ON (p={p})")
                        return True, p
                    except ValueError:
                        print("Please enter an integer p (>=1).")
            print("Please answer y or n.")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            sys.exit(0)


def sort_packets_for_temporal_split(df_mac: pd.DataFrame) -> pd.DataFrame:
    if "pcap_ts" in df_mac.columns:
        ts = pd.to_numeric(df_mac["pcap_ts"], errors="coerce")
        if np.isfinite(ts).any():
            return df_mac.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"])
    return df_mac


def _safe_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def compute_stats_vector(vals: np.ndarray, use_mean: bool, use_std: bool) -> List[float]:
    """
    Features per CFO column, per window/segment.
    Mean is repeated MEAN_REPEATS times to bias RF toward mean.
    """
    feats: List[float] = []
    if vals.size == 0:
        if use_mean:
            feats.extend([float("nan")] * MEAN_REPEATS)
        if use_std:
            feats.append(float("nan"))
        feats.extend([float("nan")] * len(EXTRA_STATS))
        return feats

    if use_mean:
        m = float(np.mean(vals))
        feats.extend([m] * MEAN_REPEATS)

    if use_std:
        feats.append(float(np.std(vals)) if vals.size > 1 else 0.0)

    med = float(np.median(vals))
    p10 = _safe_percentile(vals, 10)
    p90 = _safe_percentile(vals, 90)
    q1 = _safe_percentile(vals, 25)
    q3 = _safe_percentile(vals, 75)
    iqr = float(q3 - q1) if np.isfinite(q3) and np.isfinite(q1) else float("nan")
    mad = _mad(vals)

    feats.extend([med, iqr, p10, p90, mad])
    return feats


def build_feature_names(selected_cols: List[str], use_mean: bool, use_std: bool) -> List[str]:
    names: List[str] = []
    for col in selected_cols:
        if use_mean:
            for r in range(1, MEAN_REPEATS + 1):
                names.append(f"{col}:mean_r{r}")
        if use_std:
            names.append(f"{col}:std")
        names.append(f"{col}:median")
        names.append(f"{col}:iqr")
        names.append(f"{col}:p10")
        names.append(f"{col}:p90")
        names.append(f"{col}:mad")
    return names


def _make_windows(mat: np.ndarray, win: int, step: int) -> List[np.ndarray]:
    out = []
    n = mat.shape[0]
    if n < win:
        return out
    for s in range(0, n - win + 1, step):
        out.append(mat[s:s + win, :])
    return out


def _p_values_for_sweep(p_max: int, max_points: int = 40) -> List[int]:
    if p_max <= 1:
        return [1]

    vals = set()
    dense_upto = min(p_max, 20)
    for p in range(1, dense_upto + 1):
        vals.add(p)

    if p_max > dense_upto:
        candidates = [25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
        for c in candidates:
            if 1 <= c <= p_max:
                vals.add(c)
        vals.add(p_max)

    out = sorted(vals)
    if len(out) > max_points:
        head = [p for p in out if p <= 20]
        tail = [p for p in out if p > 20]
        if tail:
            k = max(1, len(tail) // max(1, (max_points - len(head))))
            tail = tail[::k]
        out = sorted(set(head + tail + [p_max]))
    return out


# --------------------------- data collection ---------------------------

def collect_all_data_by_mac(
    df: pd.DataFrame,
    selected_cfo_cols: List[str],
    use_mean: bool,
    use_std: bool,
    stress_mode: bool = False,
    pseudonym_period: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    NORMAL MODE:
      - Many train/test samples per MAC via windowing inside each MAC's temporal split.

    STRESS MODE (fresh pseudonyms):
      - Segment each MAC's time-ordered packets into consecutive segments of length p.
      - Each segment becomes one sample.
      - Train/Test split is by segment time (first 70% segments -> gallery; last 30% -> query).
      - Labels remain ground-truth MAC (only for scoring/linkability).
    """
    if MAC_COL not in df.columns:
        raise ValueError(f"CSV missing required column '{MAC_COL}'")
    for c in selected_cfo_cols:
        if c not in df.columns:
            raise ValueError(f"Selected CFO column missing from CSV: {c}")

    macs = sorted(df[MAC_COL].dropna().astype(str).unique().tolist())
    if not macs:
        raise ValueError("No MAC/AdvA addresses found in CSV.")

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    kept_macs: List[str] = []

    per_col_dim = (MEAN_REPEATS if use_mean else 0) + (1 if use_std else 0) + len(EXTRA_STATS)
    feat_dim = len(selected_cfo_cols) * per_col_dim

    for mac in macs:
        df_mac = df[df[MAC_COL].astype(str) == mac]
        df_mac = sort_packets_for_temporal_split(df_mac)

        # Align packets across all selected CFO columns:
        df_vals = df_mac[selected_cfo_cols].apply(pd.to_numeric, errors="coerce")
        df_vals = df_vals.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if df_vals.shape[0] < MIN_PKTS_PER_MAC:
            continue

        V = df_vals.values.astype(np.float64)  # shape (N, ncols)

        if not stress_mode:
            split_idx = int(V.shape[0] * TRAIN_FRACTION)
            split_idx = max(WINDOW_PKTS, min(split_idx, V.shape[0] - WINDOW_PKTS))

            V_train = V[:split_idx, :]
            V_test = V[split_idx:, :]

            train_windows = _make_windows(V_train, WINDOW_PKTS, WINDOW_STEP)
            test_windows = _make_windows(V_test, WINDOW_PKTS, WINDOW_STEP)

            if len(train_windows) < MIN_WINDOWS_PER_MAC or len(test_windows) < MIN_WINDOWS_PER_MAC:
                continue

            n_train_added = 0
            n_test_added = 0

            for W in train_windows:
                feats = []
                for j in range(W.shape[1]):
                    feats.extend(compute_stats_vector(W[:, j], use_mean, use_std))
                arr = np.asarray(feats, dtype=float)
                if arr.shape[0] == feat_dim and np.all(np.isfinite(arr)):
                    X_train_list.append(arr)
                    y_train_list.append(mac)
                    n_train_added += 1

            for W in test_windows:
                feats = []
                for j in range(W.shape[1]):
                    feats.extend(compute_stats_vector(W[:, j], use_mean, use_std))
                arr = np.asarray(feats, dtype=float)
                if arr.shape[0] == feat_dim and np.all(np.isfinite(arr)):
                    X_test_list.append(arr)
                    y_test_list.append(mac)
                    n_test_added += 1

            if n_train_added > 0 and n_test_added > 0:
                kept_macs.append(mac)

        else:
            if pseudonym_period is None or pseudonym_period < 1:
                raise ValueError("Stress mode requires pseudonym_period >= 1")

            seg_features: List[np.ndarray] = []
            n_rows = V.shape[0]

            for start in range(0, n_rows, pseudonym_period):
                end = min(start + pseudonym_period, n_rows)
                W = V[start:end, :]  # (p, ncols) except maybe last

                feats = []
                ok = True
                for j in range(W.shape[1]):
                    colvals = W[:, j]
                    colvals = colvals[np.isfinite(colvals)]
                    if colvals.size < 1:
                        ok = False
                        break
                    feats.extend(compute_stats_vector(colvals, use_mean, use_std))

                if not ok:
                    continue

                arr = np.asarray(feats, dtype=float)
                if arr.shape[0] == feat_dim and np.all(np.isfinite(arr)):
                    seg_features.append(arr)

            if len(seg_features) < 2:
                continue

            split_idx = int(len(seg_features) * TRAIN_FRACTION)
            split_idx = max(1, min(split_idx, len(seg_features) - 1))

            train_segs = seg_features[:split_idx]
            test_segs = seg_features[split_idx:]

            if len(train_segs) == 0 or len(test_segs) == 0:
                continue

            for arr in train_segs:
                X_train_list.append(arr)
                y_train_list.append(mac)
            for arr in test_segs:
                X_test_list.append(arr)
                y_test_list.append(mac)

            kept_macs.append(mac)

    if not X_train_list or not X_test_list:
        raise ValueError(
            "No valid MAC groups found after filtering/windowing/segmenting. "
            "Try adjusting MIN_PKTS_PER_MAC, WINDOW_PKTS/WINDOW_STEP, MIN_WINDOWS_PER_MAC, or p."
        )

    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    class_names = sorted(set(kept_macs))
    feature_names = build_feature_names(selected_cfo_cols, use_mean, use_std)

    if verbose:
        print(f"\nCSV: {FNAME}")
        print(f"Grouping key: {MAC_COL}")
        print(f"MACs total: {len(macs)} | MACs kept: {len(class_names)} (MIN_PKTS_PER_MAC={MIN_PKTS_PER_MAC})")
        print(f"Selected CFO columns: {selected_cfo_cols}")
        print(f"Mean repeats: {MEAN_REPEATS}")
        print(f"Extra stats used: {', '.join(EXTRA_STATS)}")
        print(f"Feature dim: {X_train.shape[1]}")
        if not stress_mode:
            print(f"Windowing: WINDOW_PKTS={WINDOW_PKTS}, WINDOW_STEP={WINDOW_STEP}, MIN_WINDOWS_PER_MAC={MIN_WINDOWS_PER_MAC}")
            print(f"Training fraction per MAC: {TRAIN_FRACTION:.0%} / Testing: {1-TRAIN_FRACTION:.0%}")
            print(f"Train samples: {X_train.shape[0]} (windows)")
            print(f"Test samples:  {X_test.shape[0]} (windows)")
        else:
            print(f"STRESS MODE: ON (pseudonym_period p={pseudonym_period})")
            print(f"Training fraction per MAC (by segments): {TRAIN_FRACTION:.0%} / Testing: {1-TRAIN_FRACTION:.0%}")
            print(f"Gallery samples (train segments): {X_train.shape[0]}")
            print(f"Query samples (test segments):    {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, class_names, feature_names


# --------------------------- normal training & evaluation ---------------------------

def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, class_names, feature_names):
    print("\nTraining Random Forest (tuned)...")

    rf = RandomForestClassifier(
        n_estimators=2000,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    importances = dict(zip(feature_names, rf.feature_importances_)) if feature_names else None
    oob = getattr(rf, "oob_score_", None)

    return {
        "mode": "classifier",
        "model": rf,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "oob_score": oob,
    }


# --------------------------- stress evaluation: LINKABILITY ---------------------------

def _pairwise_sqeuclidean_chunked(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    B_norm = np.sum(B * B, axis=1)[None, :]
    out = np.empty((A.shape[0], B.shape[0]), dtype=float)

    for i in range(0, A.shape[0], chunk):
        a = A[i:i+chunk]
        a_norm = np.sum(a * a, axis=1)[:, None]
        out[i:i+chunk] = a_norm + B_norm - 2.0 * (a @ B.T)

    np.maximum(out, 0.0, out=out)
    return out


def evaluate_linkability(
    X_gallery: np.ndarray,
    y_gallery: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """
    Linkability metrics (fresh pseudonyms):
      - Rank-1 linking accuracy (nearest neighbor in gallery)
      - Verification ROC-AUC: per-query best genuine vs best impostor similarity
      - Clustering ARI/NMI on (gallery+query) with KMeans(k=n_classes)
    """
    scaler = StandardScaler()
    G = scaler.fit_transform(X_gallery)
    Q = scaler.transform(X_query)

    D = _pairwise_sqeuclidean_chunked(Q, G, chunk=256)

    nn_idx = np.argmin(D, axis=1)
    nn_labels = y_gallery[nn_idx]
    rank1 = float(np.mean(nn_labels == y_query))

    pos_scores = []
    neg_scores = []
    for i in range(D.shape[0]):
        yi = y_query[i]
        same = (y_gallery == yi)
        if not np.any(same):
            continue
        d_row = D[i]
        d_pos = np.min(d_row[same])
        d_neg = np.min(d_row[~same]) if np.any(~same) else np.nan
        if np.isfinite(d_pos) and np.isfinite(d_neg):
            pos_scores.append(-float(d_pos))
            neg_scores.append(-float(d_neg))

    if len(pos_scores) >= 2 and len(neg_scores) >= 2:
        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
        y_score = np.array(pos_scores + neg_scores, dtype=float)
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = float("nan")

    X_all = np.vstack([X_gallery, X_query])
    y_all = np.concatenate([y_gallery, y_query])
    X_all_s = StandardScaler().fit_transform(X_all)

    try:
        km = KMeans(n_clusters=max(2, int(n_classes)), n_init=10, random_state=RANDOM_SEED)
        c = km.fit_predict(X_all_s)
        ari = float(adjusted_rand_score(y_all, c))
        nmi = float(normalized_mutual_info_score(y_all, c))
    except Exception:
        ari = float("nan")
        nmi = float("nan")

    return {"rank1": rank1, "auc": auc, "ari": ari, "nmi": nmi}


# --------------------------- visualization ---------------------------

def plot_feature_distribution(X_train, X_test, y_train, y_test, feature_names, outfile):
    """
    Side-by-side heatmaps. Train/test may have different #samples (stress mode),
    so we sort and index them independently (avoids out-of-bounds errors).
    """
    order_tr = np.argsort(y_train)
    labels_tr = y_train[order_tr]
    X_train_khz = X_train[order_tr] / 1e3

    order_te = np.argsort(y_test)
    labels_te = y_test[order_te]
    X_test_khz = X_test[order_te] / 1e3

    # cap rows for readability
    X_train_khz = X_train_khz[:HEATMAP_MAX_ROWS, :]
    labels_tr = labels_tr[:HEATMAP_MAX_ROWS]
    X_test_khz = X_test_khz[:HEATMAP_MAX_ROWS, :]
    labels_te = labels_te[:HEATMAP_MAX_ROWS]

    fig_h = max(6, 0.25 * max(len(labels_tr), len(labels_te)))
    fig_w = max(10, 0.35 * len(feature_names))

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    sns.heatmap(
        X_train_khz,
        ax=axes[0],
        cmap="viridis",
        cbar=True,
        yticklabels=labels_tr if len(labels_tr) <= 60 else False,
        xticklabels=feature_names,
    )
    axes[0].set_title(f"Train features (first {TRAIN_FRACTION:.0%}) [kHz] (showing up to {HEATMAP_MAX_ROWS})")
    axes[0].tick_params(axis="x", rotation=60)

    sns.heatmap(
        X_test_khz,
        ax=axes[1],
        cmap="viridis",
        cbar=True,
        yticklabels=labels_te if len(labels_te) <= 60 else False,
        xticklabels=feature_names,
    )
    axes[1].set_title(f"Test features (last {1-TRAIN_FRACTION:.0%}) [kHz] (showing up to {HEATMAP_MAX_ROWS})")
    axes[1].tick_params(axis="x", rotation=60)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"[✓] Saved feature distribution: {outfile}")
    plt.close()


def plot_confusion_matrix(results, class_names, outfile):
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(class_names)), 8))
    cm = results["confusion_matrix"]

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(
        cm_norm,
        annot=False if len(class_names) > 30 else True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Count"}
    )

    oob = results.get("oob_score", None)
    title = f"Random Forest Confusion Matrix\nAccuracy: {results['accuracy']:.1%}"
    if isinstance(oob, float):
        title += f" | OOB: {oob:.1%}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True MAC", fontsize=12)
    ax.set_xlabel("Predicted MAC", fontsize=12)
    ax.tick_params(axis="x", rotation=60)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"[✓] Saved confusion matrix: {outfile}")
    plt.close()


def plot_stress_linkability(p_values: List[int], rank1s: List[float], aucs: List[float], outfile: str):
    plt.figure(figsize=(10, 4))
    r1 = [100.0 * v if np.isfinite(v) else np.nan for v in rank1s]
    au = [100.0 * v if np.isfinite(v) else np.nan for v in aucs]

    plt.plot(p_values, r1, marker="o", label="Rank-1 linking (%)")
    plt.plot(p_values, au, marker="o", label="Verification AUC (%)")

    plt.xlabel("p (pseudonym changes every p packets)")
    plt.ylabel("Score (%)")
    plt.title("Linkability vs Pseudonym Rotation (Fresh Pseudonyms)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    print(f"[✓] Saved stress linkability plot: {outfile}")
    plt.close()


# --------------------------- reporting ---------------------------

def write_results_report(
    mode: str,
    selected_cols: List[str],
    use_mean: bool,
    use_std: bool,
    feature_names: List[str],
    outfile: str,
    classifier_results: Optional[Dict] = None,
    class_names: Optional[List[str]] = None,
    y_test: Optional[np.ndarray] = None,
    stress_metrics: Optional[Dict[str, float]] = None,
    stress_p: Optional[int] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
):
    with open(outfile, "w") as f:
        f.write("=" * 80 + "\n")
        if mode == "classifier":
            f.write("BLE DEVICE FINGERPRINTING - RANDOM FOREST (WINDOWED)\n")
        else:
            f.write("BLE DEVICE FINGERPRINTING - LINKABILITY (FRESH PSEUDONYMS)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Input CSV: {FNAME}\n")
        f.write(f"Grouping key: {MAC_COL}\n")
        f.write(f"MIN_PKTS_PER_MAC: {MIN_PKTS_PER_MAC}\n")
        if mode == "classifier":
            f.write(f"Windowing: WINDOW_PKTS={WINDOW_PKTS}, WINDOW_STEP={WINDOW_STEP}, MIN_WINDOWS_PER_MAC={MIN_WINDOWS_PER_MAC}\n")
        f.write(f"Selected CFO columns: {', '.join(selected_cols)}\n")
        f.write(f"Mean repeats (priority): MEAN_REPEATS={MEAN_REPEATS}\n")
        f.write(f"Extra stats: {', '.join(EXTRA_STATS)}\n")
        f.write(f"Stats: {'mean ' if use_mean else ''}{'std' if use_std else ''}\n".strip() + "\n")
        f.write(f"Training fraction: {TRAIN_FRACTION:.0%}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FEATURE NAMES\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(feature_names):
            f.write(f"{i:3d}: {name}\n")
        f.write("\n")

        if mode == "classifier" and classifier_results is not None and class_names is not None and y_test is not None:
            f.write("-" * 80 + "\n")
            f.write("RANDOM FOREST PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of MAC classes: {len(class_names)}\n")
            f.write(f"Accuracy:  {classifier_results['accuracy']:.2%}\n")
            f.write(f"Precision: {classifier_results['precision']:.2%}\n")
            f.write(f"Recall:    {classifier_results['recall']:.2%}\n")
            f.write(f"F1-Score:  {classifier_results['f1']:.2%}\n")
            if isinstance(classifier_results.get("oob_score", None), float):
                f.write(f"OOB Score:  {classifier_results['oob_score']:.2%}\n")
            f.write("\n")

            f.write("Per-MAC Classification Report:\n")
            f.write(classification_report(
                y_test, classifier_results["y_pred"],
                labels=class_names,
                target_names=class_names,
                zero_division=0
            ))
            f.write("\n")

        if mode == "stress" and stress_metrics is not None:
            f.write("-" * 80 + "\n")
            f.write("LINKABILITY METRICS (fresh pseudonyms)\n")
            f.write("-" * 80 + "\n")
            f.write(f"p (pseudonym period): {stress_p}\n")
            if n_train is not None and n_test is not None:
                f.write(f"Gallery samples (train segments): {n_train}\n")
                f.write(f"Query samples (test segments):    {n_test}\n")
            f.write("\n")
            f.write(f"Rank-1 linking accuracy: {stress_metrics['rank1']:.2%}\n")
            f.write(f"Verification ROC-AUC:    {stress_metrics['auc']:.4f}\n")
            f.write(f"Clustering ARI:          {stress_metrics['ari']:.4f}\n")
            f.write(f"Clustering NMI:          {stress_metrics['nmi']:.4f}\n\n")

    print(f"[✓] Saved report: {outfile}")


# --------------------------- stress sweep ---------------------------

def run_stress_sweep(
    df: pd.DataFrame,
    selected_cols: List[str],
    use_mean: bool,
    use_std: bool,
    p_max: int,
    outfile_pdf: str
):
    p_values = _p_values_for_sweep(p_max)
    rank1s: List[float] = []
    aucs: List[float] = []

    print("\nRunning stress sweep (linkability vs p)...")
    for p in p_values:
        try:
            X_tr, X_te, y_tr, y_te, class_names, feature_names = collect_all_data_by_mac(
                df, selected_cols, use_mean, use_std,
                stress_mode=True, pseudonym_period=p, verbose=False
            )
            if len(class_names) < 2:
                rank1s.append(np.nan)
                aucs.append(np.nan)
                continue
            metrics = evaluate_linkability(
                X_gallery=X_tr, y_gallery=y_tr,
                X_query=X_te, y_query=y_te,
                n_classes=len(class_names)
            )
            rank1s.append(metrics["rank1"])
            aucs.append(metrics["auc"])
        except Exception:
            rank1s.append(np.nan)
            aucs.append(np.nan)

    plot_stress_linkability(p_values, rank1s, aucs, outfile_pdf)


# --------------------------- main ---------------------------

def main():
    global MAC_COL

    print("=" * 80)
    print("BLE DEVICE FINGERPRINTING using CFO Statistics (MAC-GROUPED)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/")

    if not os.path.exists(FNAME):
        print(f"ERROR: Missing input file: {FNAME}")
        sys.exit(1)

    df = pd.read_csv(FNAME)

    MAC_COL = resolve_mac_column(df)
    print(f"\nDetected MAC column: {MAC_COL}")

    generate_test_csvs(df, MAC_COL)

    use_mean, use_std = get_feature_choice()
    selected_cols = choose_cfo_columns(df)
    stress_mode, p = get_stress_mode_choice()

    print("\nBuilding train/test feature sets...")
    X_train, X_test, y_train, y_test, class_names, feature_names = collect_all_data_by_mac(
        df, selected_cols, use_mean, use_std,
        stress_mode=stress_mode, pseudonym_period=p, verbose=True
    )

    if len(class_names) < 2:
        print("ERROR: Need at least 2 MACs (classes) for evaluation!")
        sys.exit(1)

    print("\nGenerating outputs...")
    plot_feature_distribution(X_train, X_test, y_train, y_test, feature_names, OUT_DISTRIBUTION)

    if not stress_mode:
        print("\nTraining classifier...")
        results = train_and_evaluate_classifier(X_train, X_test, y_train, y_test, class_names, feature_names)

        print("\n" + "=" * 80)
        print("RESULTS (Classifier)")
        print("=" * 80)
        print(f"Accuracy:  {results['accuracy']:.2%}")
        print(f"Precision: {results['precision']:.2%}")
        print(f"Recall:    {results['recall']:.2%}")
        print(f"F1-Score:  {results['f1']:.2%}")
        if isinstance(results.get("oob_score", None), float):
            print(f"OOB Score:  {results['oob_score']:.2%}")

        plot_confusion_matrix(results, class_names, OUT_CONFUSION)

        write_results_report(
            mode="classifier",
            selected_cols=selected_cols,
            use_mean=use_mean,
            use_std=use_std,
            feature_names=feature_names,
            outfile=OUT_RESULTS,
            classifier_results=results,
            class_names=class_names,
            y_test=y_test
        )

    else:
        print("\nEvaluating linkability (fresh pseudonyms; MAC used only for scoring)...")
        metrics = evaluate_linkability(
            X_gallery=X_train, y_gallery=y_train,
            X_query=X_test, y_query=y_test,
            n_classes=len(class_names)
        )

        print("\n" + "=" * 80)
        print("RESULTS (Linkability)")
        print("=" * 80)
        print(f"Rank-1 linking accuracy: {metrics['rank1']:.2%}")
        print(f"Verification ROC-AUC:    {metrics['auc']:.4f}")
        print(f"Clustering ARI:          {metrics['ari']:.4f}")
        print(f"Clustering NMI:          {metrics['nmi']:.4f}")

        write_results_report(
            mode="stress",
            selected_cols=selected_cols,
            use_mean=use_mean,
            use_std=use_std,
            feature_names=feature_names,
            outfile=OUT_RESULTS,
            stress_metrics=metrics,
            stress_p=p,
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
        )

        if p is not None:
            run_stress_sweep(df, selected_cols, use_mean, use_std, p_max=p, outfile_pdf=OUT_STRESS_PLOT)

    print("\n" + "=" * 80)
    print("DONE! Check output files for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# BLE Device Fingerprinting using CFO Statistics (grouped by MAC/AdvA)

# Improvements (minimal structural changes, but accuracy-focused):
#   - Builds MULTIPLE samples per MAC by windowing packets within each MAC's 70/30 temporal split
#   - Gives higher importance to MEAN by repeating mean features (MEAN_REPEATS)
#   - Adds a small set of robust distribution features (not too many to avoid overfitting)
#   - Improves RF hyperparameters for this regime

# Input: cfo_samples_rail.csv
# Outputs:
#   - all_devices_static_rail/ble_fingerprint_classification_results.txt
#   - all_devices_static_rail/ble_fingerprint_confusion_matrix.png
#   - all_devices_static_rail/ble_fingerprint_feature_distribution.png
# """

# import os
# import sys
# import warnings
# from typing import List, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_recall_fscore_support,
#     confusion_matrix, classification_report
# )

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# # --------------------------- config ---------------------------

# FNAME = "/home/mishra/Downloads/cfo_samples_rail.csv"
# OUTPUT_DIR = "all_devices_static_rail"

# # NOTE: resolved at runtime to "AdvA" or "adv_addr" (or case-insensitive match)
# MAC_COL = "AdvA"

# TRAIN_FRACTION = 0.7          # temporal split within each MAC
# MIN_PKTS_PER_MAC = 200        # minimum usable packets per MAC (after dropping NaNs) to keep class

# # NEW: create many samples per MAC using fixed packet windows inside train/test split
# WINDOW_PKTS = 100              # packets per sample-window (increase if CFO is noisy)
# WINDOW_STEP = 10              # step between windows (==WINDOW_PKTS -> non-overlapping)
# MIN_WINDOWS_PER_MAC = 2       # require at least this many train windows and test windows combined? (we'll gate per split)

# # NEW: give mean more importance by repetition (RF feature-subsampling makes repetition matter)
# MEAN_REPEATS = 4              # mean repeated this many times per CFO column (>=2 strongly biases RF toward mean)

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)

# OUT_RESULTS = os.path.join(OUTPUT_DIR, "ble_fingerprint_classification_results.txt")
# OUT_CONFUSION = os.path.join(OUTPUT_DIR, "ble_fingerprint_confusion_matrix.png")
# OUT_DISTRIBUTION = os.path.join(OUTPUT_DIR, "ble_fingerprint_feature_distribution.png")

# # Transition CFO columns likely present in cfo_samples_rail.csv (plus fallbacks)
# OPTIONAL_CFO_COLS = [
#     "CFO_00_Hz",
#     "CFO_11_Hz",
#     "CFO_10_Hz",
#     "CFO_01_Hz",
#     "CFO_from_transitions_Hz",
# ]

# # NEW: keep extra stats SMALL and robust (too many stats can hurt with few classes/samples)
# EXTRA_STATS = (
#     "median",
#     "iqr",
#     "p10",
#     "p90",
#     "mad",
# )

# # --------------------------- helpers ---------------------------

# def _col_lookup_case_insensitive(df: pd.DataFrame, name: str) -> str:
#     target = name.lower()
#     for c in df.columns:
#         if str(c).lower() == target:
#             return str(c)
#     return ""


# def resolve_mac_column(df: pd.DataFrame) -> str:
#     for cand in ["AdvA", "adv_addr", "advA", "ADV_A", "adv_address", "advaddr"]:
#         hit = _col_lookup_case_insensitive(df, cand)
#         if hit:
#             return hit

#     lowers = [str(c).lower() for c in df.columns]
#     for c, lc in zip(df.columns, lowers):
#         if ("adv" in lc) and ("addr" in lc or lc.endswith("a") or "adva" in lc):
#             return str(c)

#     raise ValueError("Could not find MAC/AdvA column (expected 'AdvA' or 'adv_addr').")


# def find_primary_cfo_column(df: pd.DataFrame) -> str:
#     cols = list(df.columns)
#     lower = [str(c).lower() for c in cols]

#     for c, lc in zip(cols, lower):
#         if lc in ["cfo_hz", "cfo_quick_hz", "est_cfo_hz", "cfo_exact_quick_hz"]:
#             return str(c)

#     for c, lc in zip(cols, lower):
#         if "cfo" in lc and "hz" in lc:
#             return str(c)

#     raise ValueError("No CFO column found in CSV")


# def get_feature_choice() -> Tuple[bool, bool]:
#     print("\n" + "=" * 80)
#     print("STATISTIC SELECTION")
#     print("=" * 80)
#     print("\nWhich statistics would you like to use per selected CFO column?")
#     print("  1) Mean only  (mean is up-weighted via repetition)")
#     print("  2) Standard Deviation only")
#     print("  3) Both Mean and Standard Deviation")
#     print("\nNOTE: Additional robust distribution stats are ALWAYS included:")
#     print("      " + ", ".join(EXTRA_STATS))

#     while True:
#         try:
#             choice = input("\nEnter your choice (1-3): ").strip()
#             if choice == "1":
#                 print("\n✓ Selected: Mean only (+ extra distribution stats)")
#                 return True, False
#             if choice == "2":
#                 print("\n✓ Selected: Std Dev only (+ extra distribution stats)")
#                 return False, True
#             if choice == "3":
#                 print("\n✓ Selected: Mean and Std Dev (+ extra distribution stats)")
#                 return True, True
#             print("Invalid choice. Please enter 1, 2, or 3.")
#         except KeyboardInterrupt:
#             print("\n\nAborted by user.")
#             sys.exit(0)


# def choose_cfo_columns(df: pd.DataFrame) -> List[str]:
#     primary = find_primary_cfo_column(df)
#     available = set(map(str, df.columns))

#     choices = [(primary, True)]
#     for c in OPTIONAL_CFO_COLS:
#         hit = _col_lookup_case_insensitive(df, c)
#         if hit and hit in available and hit != primary:
#             choices.append((hit, False))

#     print("\n" + "=" * 80)
#     print("CFO TYPE SELECTION")
#     print("=" * 80)
#     print("\nSelect which CFO types to use (features computed per column).")
#     print("Enter a comma-separated list of numbers, e.g., 1,3,4")
#     print("Press Enter for default (primary CFO only).\n")

#     for i, (col, default_on) in enumerate(choices, 1):
#         tag = "DEFAULT" if default_on else ""
#         print(f"  {i}) {col} {tag}")

#     while True:
#         try:
#             raw = input("\nYour selection: ").strip()
#             if raw == "":
#                 selected = [primary]
#                 print(f"\n✓ Selected default: [{primary}]")
#                 return selected

#             idxs = []
#             for part in raw.split(","):
#                 part = part.strip()
#                 if not part:
#                     continue
#                 idxs.append(int(part))
#             idxs = sorted(set(idxs))

#             if not idxs:
#                 print("Please choose at least one number.")
#                 continue

#             if min(idxs) < 1 or max(idxs) > len(choices):
#                 print(f"Invalid selection. Choose numbers in 1..{len(choices)}")
#                 continue

#             selected = [choices[i - 1][0] for i in idxs]
#             print("\n✓ Selected CFO columns:")
#             for c in selected:
#                 print(f"  - {c}")
#             return selected
#         except ValueError:
#             print("Invalid input. Use numbers like: 1,2,3")
#         except KeyboardInterrupt:
#             print("\n\nAborted by user.")
#             sys.exit(0)


# def sort_packets_for_temporal_split(df_mac: pd.DataFrame) -> pd.DataFrame:
#     if "pcap_ts" in df_mac.columns:
#         ts = pd.to_numeric(df_mac["pcap_ts"], errors="coerce")
#         if np.isfinite(ts).any():
#             return df_mac.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"])
#     return df_mac


# def _safe_percentile(x: np.ndarray, q: float) -> float:
#     if x.size == 0:
#         return float("nan")
#     return float(np.percentile(x, q))


# def _mad(x: np.ndarray) -> float:
#     if x.size == 0:
#         return float("nan")
#     med = np.median(x)
#     return float(np.median(np.abs(x - med)))


# def compute_stats_vector(vals: np.ndarray, use_mean: bool, use_std: bool) -> List[float]:
#     """
#     Features per CFO column, per window.
#     Mean is repeated MEAN_REPEATS times to bias RF toward mean (via feature subsampling).
#     """
#     feats: List[float] = []
#     if vals.size == 0:
#         # keep fixed shape
#         if use_mean:
#             feats.extend([float("nan")] * MEAN_REPEATS)
#         if use_std:
#             feats.append(float("nan"))
#         feats.extend([float("nan")] * len(EXTRA_STATS))
#         return feats

#     if use_mean:
#         m = float(np.mean(vals))
#         feats.extend([m] * MEAN_REPEATS)

#     if use_std:
#         feats.append(float(np.std(vals)) if vals.size > 1 else 0.0)

#     med = float(np.median(vals))
#     p10 = _safe_percentile(vals, 10)
#     p90 = _safe_percentile(vals, 90)
#     q1 = _safe_percentile(vals, 25)
#     q3 = _safe_percentile(vals, 75)
#     iqr = float(q3 - q1) if np.isfinite(q3) and np.isfinite(q1) else float("nan")
#     mad = _mad(vals)

#     feats.extend([med, iqr, p10, p90, mad])
#     return feats


# def build_feature_names(selected_cols: List[str], use_mean: bool, use_std: bool) -> List[str]:
#     names: List[str] = []
#     for col in selected_cols:
#         if use_mean:
#             for r in range(1, MEAN_REPEATS + 1):
#                 names.append(f"{col}:mean_r{r}")
#         if use_std:
#             names.append(f"{col}:std")
#         names.append(f"{col}:median")
#         names.append(f"{col}:iqr")
#         names.append(f"{col}:p10")
#         names.append(f"{col}:p90")
#         names.append(f"{col}:mad")
#     return names


# def _make_windows(mat: np.ndarray, win: int, step: int) -> List[np.ndarray]:
#     """
#     mat: shape (N, C) packets-by-columns
#     returns list of window slices, each (win, C)
#     """
#     out = []
#     n = mat.shape[0]
#     if n < win:
#         return out
#     for s in range(0, n - win + 1, step):
#         out.append(mat[s:s + win, :])
#     return out


# def collect_all_data_by_mac(
#     df: pd.DataFrame,
#     selected_cfo_cols: List[str],
#     use_mean: bool,
#     use_std: bool
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
#     """
#     Build many training/testing samples per MAC via windowing inside each MAC's temporal split.
#     """
#     if MAC_COL not in df.columns:
#         raise ValueError(f"CSV missing required column '{MAC_COL}'")

#     for c in selected_cfo_cols:
#         if c not in df.columns:
#             raise ValueError(f"Selected CFO column missing from CSV: {c}")

#     macs = sorted(df[MAC_COL].dropna().astype(str).unique().tolist())
#     if not macs:
#         raise ValueError("No MAC addresses found in CSV.")

#     X_train_list, X_test_list = [], []
#     y_train_list, y_test_list = [], []
#     kept_macs = []

#     per_col_dim = (MEAN_REPEATS if use_mean else 0) + (1 if use_std else 0) + len(EXTRA_STATS)
#     feat_dim = len(selected_cfo_cols) * per_col_dim

#     for mac in macs:
#         df_mac = df[df[MAC_COL].astype(str) == mac]
#         df_mac = sort_packets_for_temporal_split(df_mac)

#         # Keep packet alignment across all selected CFO columns:
#         df_vals = df_mac[selected_cfo_cols].apply(pd.to_numeric, errors="coerce")
#         # drop packets where ANY selected col is invalid
#         df_vals = df_vals.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

#         if df_vals.shape[0] < MIN_PKTS_PER_MAC:
#             continue

#         V = df_vals.values.astype(np.float64)  # shape (N, ncols)

#         split_idx = int(V.shape[0] * TRAIN_FRACTION)
#         split_idx = max(WINDOW_PKTS, min(split_idx, V.shape[0] - WINDOW_PKTS))

#         V_train = V[:split_idx, :]
#         V_test = V[split_idx:, :]

#         train_windows = _make_windows(V_train, WINDOW_PKTS, WINDOW_STEP)
#         test_windows = _make_windows(V_test, WINDOW_PKTS, WINDOW_STEP)

#         if len(train_windows) < MIN_WINDOWS_PER_MAC or len(test_windows) < MIN_WINDOWS_PER_MAC:
#             continue

#         # Build features per window
#         for W in train_windows:
#             feats = []
#             for j in range(W.shape[1]):
#                 feats.extend(compute_stats_vector(W[:, j], use_mean, use_std))
#             arr = np.asarray(feats, dtype=float)
#             if arr.shape[0] == feat_dim and np.all(np.isfinite(arr)):
#                 X_train_list.append(arr)
#                 y_train_list.append(mac)

#         for W in test_windows:
#             feats = []
#             for j in range(W.shape[1]):
#                 feats.extend(compute_stats_vector(W[:, j], use_mean, use_std))
#             arr = np.asarray(feats, dtype=float)
#             if arr.shape[0] == feat_dim and np.all(np.isfinite(arr)):
#                 X_test_list.append(arr)
#                 y_test_list.append(mac)

#         if mac not in kept_macs and (y_train_list.count(mac) > 0) and (y_test_list.count(mac) > 0):
#             kept_macs.append(mac)

#     if not X_train_list or not X_test_list:
#         raise ValueError(
#             "No valid MAC groups found after filtering/windowing. "
#             "Try adjusting MIN_PKTS_PER_MAC, WINDOW_PKTS/WINDOW_STEP, or MIN_WINDOWS_PER_MAC."
#         )

#     X_train = np.vstack(X_train_list)
#     X_test = np.vstack(X_test_list)
#     y_train = np.array(y_train_list)
#     y_test = np.array(y_test_list)

#     class_names = sorted(set(kept_macs))
#     feature_names = build_feature_names(selected_cfo_cols, use_mean, use_std)

#     print(f"\nCSV: {FNAME}")
#     print(f"Grouping key: {MAC_COL}")
#     print(f"MACs total: {len(macs)} | MACs kept: {len(class_names)} (MIN_PKTS_PER_MAC={MIN_PKTS_PER_MAC})")
#     print(f"Selected CFO columns: {selected_cfo_cols}")
#     print(f"Mean repeats: {MEAN_REPEATS} (mean is prioritized)")
#     print(f"Extra stats used: {', '.join(EXTRA_STATS)}")
#     print(f"Windowing: WINDOW_PKTS={WINDOW_PKTS}, WINDOW_STEP={WINDOW_STEP}, MIN_WINDOWS_PER_MAC={MIN_WINDOWS_PER_MAC}")
#     print(f"Feature dim: {X_train.shape[1]}")
#     print(f"Training fraction per MAC: {TRAIN_FRACTION:.0%} / Testing: {1-TRAIN_FRACTION:.0%}")
#     print(f"Train samples: {X_train.shape[0]} (windows)")
#     print(f"Test samples:  {X_test.shape[0]} (windows)")

#     return X_train, X_test, y_train, y_test, class_names, feature_names


# # --------------------------- training & evaluation ---------------------------

# def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, feature_names):
#     # RF does not require scaling; keep raw features (helps interpretability too)
#     print("\nTraining Random Forest (tuned)...")

#     rf = RandomForestClassifier(
#         n_estimators=2000,
#         max_depth=18,                 # limit depth to reduce overfit
#         min_samples_split=4,
#         min_samples_leaf=2,
#         max_features="sqrt",          # good default, reduces correlation between trees
#         bootstrap=True,
#         oob_score=True,
#         class_weight="balanced_subsample",
#         random_state=RANDOM_SEED,
#         n_jobs=-1,
#     )

#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_test, y_pred, average="weighted", zero_division=0
#     )
#     cm = confusion_matrix(y_test, y_pred, labels=class_names)

#     importances = dict(zip(feature_names, rf.feature_importances_)) if feature_names else None

#     # OOB can be helpful as a sanity check
#     oob = getattr(rf, "oob_score_", None)

#     return {
#         "model": rf,
#         "y_pred": y_pred,
#         "accuracy": accuracy,
#         "precision": prec,
#         "recall": rec,
#         "f1": f1,
#         "confusion_matrix": cm,
#         "feature_importances": importances,
#         "oob_score": oob,
#     }


# # --------------------------- visualization ---------------------------

# def plot_feature_distribution(X_train, X_test, y_train, feature_names, outfile):
#     order = np.argsort(y_train)
#     labels = y_train[order]

#     X_train_khz = X_train[order] / 1e3
#     X_test_khz = X_test / 1e3  # not aligned to train order necessarily

#     fig_h = max(6, 0.25 * min(len(labels), 200))
#     fig_w = max(10, 0.35 * len(feature_names))

#     fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

#     sns.heatmap(
#         X_train_khz[:200, :],  # cap to keep plot readable
#         ax=axes[0],
#         cmap="viridis",
#         cbar=True,
#         yticklabels=labels[:200] if len(labels) <= 60 else False,
#         xticklabels=feature_names,
#     )
#     axes[0].set_title(f"Train windows (first {TRAIN_FRACTION:.0%}) [kHz] (showing up to 200)")
#     axes[0].tick_params(axis="x", rotation=60)

#     sns.heatmap(
#         X_test_khz[:200, :],   # cap
#         ax=axes[1],
#         cmap="viridis",
#         cbar=True,
#         yticklabels=False,
#         xticklabels=feature_names,
#     )
#     axes[1].set_title(f"Test windows (last {1-TRAIN_FRACTION:.0%}) [kHz] (showing up to 200)")
#     axes[1].tick_params(axis="x", rotation=60)

#     plt.tight_layout()
#     plt.savefig(outfile, dpi=200, bbox_inches="tight")
#     print(f"[✓] Saved feature distribution: {outfile}")
#     plt.close()


# def plot_confusion_matrix(results, class_names, outfile):
#     fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(class_names)), 8))
#     cm = results["confusion_matrix"]

#     cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
#     cm_norm = np.nan_to_num(cm_norm)

#     sns.heatmap(
#         cm_norm,
#         annot=False if len(class_names) > 30 else True,
#         fmt=".2f",
#         cmap="Blues",
#         xticklabels=class_names,
#         yticklabels=class_names,
#         ax=ax,
#         cbar_kws={"label": "Normalized Count"}
#     )

#     oob = results.get("oob_score", None)
#     title = f"Random Forest Confusion Matrix\nAccuracy: {results['accuracy']:.1%}"
#     if isinstance(oob, float):
#         title += f" | OOB: {oob:.1%}"

#     ax.set_title(title, fontsize=14, fontweight="bold")
#     ax.set_ylabel("True MAC", fontsize=12)
#     ax.set_xlabel("Predicted MAC", fontsize=12)
#     ax.tick_params(axis="x", rotation=60)
#     ax.tick_params(axis="y", rotation=0)

#     plt.tight_layout()
#     plt.savefig(outfile, dpi=200, bbox_inches="tight")
#     print(f"[✓] Saved confusion matrix: {outfile}")
#     plt.close()


# def write_results_report(results, class_names, y_test, X_train, X_test, y_train,
#                         feature_names, selected_cols, use_mean, use_std, outfile):
#     with open(outfile, "w") as f:
#         f.write("=" * 80 + "\n")
#         f.write("BLE DEVICE FINGERPRINTING - RANDOM FOREST (MAC-GROUPED)\n")
#         f.write("=" * 80 + "\n\n")

#         f.write(f"Input CSV: {FNAME}\n")
#         f.write(f"Grouping key: {MAC_COL} (each MAC treated as one device)\n")
#         f.write(f"MIN_PKTS_PER_MAC: {MIN_PKTS_PER_MAC}\n")
#         f.write(f"Windowing: WINDOW_PKTS={WINDOW_PKTS}, WINDOW_STEP={WINDOW_STEP}, MIN_WINDOWS_PER_MAC={MIN_WINDOWS_PER_MAC}\n")
#         f.write(f"Mean repeats (priority): MEAN_REPEATS={MEAN_REPEATS}\n\n")

#         f.write(f"Selected CFO columns: {', '.join(selected_cols)}\n")
#         f.write(f"Stats: {'mean ' if use_mean else ''}{'std' if use_std else ''}\n".strip() + "\n")
#         f.write(f"Extra distribution stats: {', '.join(EXTRA_STATS)}\n")
#         f.write(f"Training fraction: {TRAIN_FRACTION:.0%} per MAC\n\n")

#         f.write(f"Number of MAC classes: {len(class_names)}\n")
#         f.write(f"Train samples (windows): {len(y_train)}\n")
#         f.write(f"Test samples  (windows): {len(y_test)}\n\n")

#         f.write("-" * 80 + "\n")
#         f.write("FEATURE NAMES\n")
#         f.write("-" * 80 + "\n")
#         for i, name in enumerate(feature_names):
#             f.write(f"{i:3d}: {name}\n")
#         f.write("\n")

#         if results.get("feature_importances") is not None:
#             f.write("-" * 80 + "\n")
#             f.write("FEATURE IMPORTANCES\n")
#             f.write("-" * 80 + "\n")
#             for feat, imp in sorted(results["feature_importances"].items(), key=lambda x: -x[1]):
#                 f.write(f"{feat:<35}: {imp:.6f}\n")
#             f.write("\n")

#         f.write("-" * 80 + "\n")
#         f.write("RANDOM FOREST PERFORMANCE\n")
#         f.write("-" * 80 + "\n")
#         f.write(f"Accuracy:  {results['accuracy']:.2%}\n")
#         f.write(f"Precision: {results['precision']:.2%}\n")
#         f.write(f"Recall:    {results['recall']:.2%}\n")
#         f.write(f"F1-Score:  {results['f1']:.2%}\n")
#         if isinstance(results.get("oob_score", None), float):
#             f.write(f"OOB Score:  {results['oob_score']:.2%}\n")
#         f.write("\n")

#         f.write("Per-MAC Classification Report:\n")
#         f.write(classification_report(
#             y_test, results["y_pred"],
#             labels=class_names,
#             target_names=class_names,
#             zero_division=0
#         ))
#         f.write("\n")

#     print(f"[✓] Saved detailed results: {outfile}")


# # --------------------------- main ---------------------------

# def main():
#     global MAC_COL

#     print("=" * 80)
#     print("BLE DEVICE FINGERPRINTING using CFO Statistics (MAC-GROUPED)")
#     print("=" * 80)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     print(f"\nOutput directory: {OUTPUT_DIR}/")

#     if not os.path.exists(FNAME):
#         print(f"ERROR: Missing input file: {FNAME}")
#         sys.exit(1)

#     df = pd.read_csv(FNAME)

#     MAC_COL = resolve_mac_column(df)
#     print(f"\nDetected MAC column: {MAC_COL}")

#     use_mean, use_std = get_feature_choice()
#     selected_cols = choose_cfo_columns(df)

#     print("\nBuilding per-MAC train/test feature sets (windowed)...")
#     X_train, X_test, y_train, y_test, class_names, feature_names = collect_all_data_by_mac(
#         df, selected_cols, use_mean, use_std
#     )

#     if len(class_names) < 2:
#         print("ERROR: Need at least 2 MACs (classes) for classification!")
#         sys.exit(1)

#     print("\nTraining classifier...")
#     results = train_and_evaluate(X_train, X_test, y_train, y_test, class_names, feature_names)

#     print("\n" + "=" * 80)
#     print("RESULTS")
#     print("=" * 80)
#     print(f"Accuracy:  {results['accuracy']:.2%}")
#     print(f"Precision: {results['precision']:.2%}")
#     print(f"Recall:    {results['recall']:.2%}")
#     print(f"F1-Score:  {results['f1']:.2%}")
#     if isinstance(results.get("oob_score", None), float):
#         print(f"OOB Score:  {results['oob_score']:.2%}")

#     print("\nGenerating outputs...")
#     plot_feature_distribution(X_train, X_test, y_train, feature_names, OUT_DISTRIBUTION)
#     plot_confusion_matrix(results, class_names, OUT_CONFUSION)
#     write_results_report(results, class_names, y_test, X_train, X_test, y_train,
#                         feature_names, selected_cols, use_mean, use_std, OUT_RESULTS)

#     print("\n" + "=" * 80)
#     print("DONE! Check output files for detailed results.")
#     print("=" * 80)


# if __name__ == "__main__":
#     main()