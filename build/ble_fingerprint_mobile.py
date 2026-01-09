#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLE Device Fingerprinting using CFO Statistics (grouped by MAC address)

Normal mode (stress mode OFF):
- One train sample + one test sample per MAC (features from CFO stats)
- Train RandomForest classifier to predict MAC
- Outputs confusion matrix + feature heatmap + report

Stress mode (fresh pseudonyms):
- For each MAC, sort packets by time, then split into consecutive segments of length p.
  Each segment corresponds to a fresh pseudonym (never reused).
- Train/Test split is by time on segments (first 70% segments -> gallery, last 30% -> query).
- NO supervised classifier on pseudonyms (labels are not reused).
- Evaluate LINKABILITY using ground-truth MAC only for scoring:
    * Rank-1 linking accuracy: nearest neighbor in gallery has same MAC
    * Verification ROC-AUC: per-query best genuine vs best impostor similarity
    * Clustering ARI/NMI: KMeans clusters vs MAC labels (unsupervised; uses MAC only for scoring)
- Additionally, plot Rank-1 (%) and AUC vs p (sweep) and save as PDF.

Outputs:
  - all_devices_static/ble_fingerprint_results.txt
  - all_devices_static/ble_fingerprint_confusion_matrix.png (normal mode only)
  - all_devices_static/ble_fingerprint_feature_distribution.png
  - all_devices_static/ble_fingerprint_stress_linkability.pdf (stress mode only)
"""

import os
import sys
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

FNAME = "/home/mishra/sdrs/BLESDR/build/test1.csv"
OUTPUT_DIR = "all_devices_static"

MAC_COL = "adv_addr"
TRAIN_FRACTION = 0.7
MIN_PKTS_PER_MAC = 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_RESULTS = os.path.join(OUTPUT_DIR, "ble_fingerprint_results.txt")
OUT_CONFUSION = os.path.join(OUTPUT_DIR, "ble_fingerprint_confusion_matrix.png")
OUT_DISTRIBUTION = os.path.join(OUTPUT_DIR, "ble_fingerprint_feature_distribution.png")
OUT_STRESS_PLOT = os.path.join(OUTPUT_DIR, "ble_fingerprint_stress_linkability.pdf")

OPTIONAL_CFO_COLS = [
    "cfo_equal_00_hz",
    "cfo_equal_11_hz",
    "cfo_jump_10_hz",
    "cfo_jump_01_hz",
]

# --------------------------- helpers ---------------------------

def find_primary_cfo_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    for c, lc in zip(cols, lower):
        if lc in ["cfo_quick_hz", "est_cfo_hz", "cfo_exact_quick_hz"]:
            return c

    for c, lc in zip(cols, lower):
        if "cfo" in lc and "hz" in lc:
            return c

    raise ValueError("No CFO column found in CSV")


def get_feature_choice() -> Tuple[bool, bool]:
    print("\n" + "=" * 80)
    print("STATISTIC SELECTION")
    print("=" * 80)
    print("\nWhich statistics would you like to use per selected CFO column?")
    print("  1) Mean only")
    print("  2) Standard Deviation only")
    print("  3) Both Mean and Standard Deviation")

    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                print("\n✓ Selected: Mean only")
                return True, False
            if choice == "2":
                print("\n✓ Selected: Std Dev only")
                return False, True
            if choice == "3":
                print("\n✓ Selected: Mean and Std Dev")
                return True, True
            print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            sys.exit(0)


def choose_cfo_columns(df: pd.DataFrame) -> List[str]:
    primary = find_primary_cfo_column(df)
    available = set(df.columns)

    choices = [(primary, True)]
    for c in OPTIONAL_CFO_COLS:
        if c in available:
            choices.append((c, False))

    print("\n" + "=" * 80)
    print("CFO COLUMN SELECTION")
    print("=" * 80)
    print("\nSelect which CFO columns to use (features will be computed per column).")
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
    print("  - If enabled: within each MAC, after every p packets, a new pseudonym is assigned.")
    print("  - Pseudonyms are assumed fresh (never reused).")
    print("  - We evaluate linkability (not supervised classification).")

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


def compute_stats_vector(vals: np.ndarray, use_mean: bool, use_std: bool) -> List[float]:
    feats: List[float] = []
    if use_mean:
        feats.append(float(np.mean(vals)) if vals.size else np.nan)
    if use_std:
        feats.append(float(np.std(vals)) if vals.size > 1 else np.nan)
    return feats


def build_feature_names(selected_cols: List[str], use_mean: bool, use_std: bool) -> List[str]:
    names = []
    for col in selected_cols:
        if use_mean:
            names.append(f"{col}:mean")
        if use_std:
            names.append(f"{col}:std")
    return names


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
    Normal mode:
      - One train sample + one test sample per MAC.

    Stress mode (fresh pseudonyms):
      - Segment each MAC's time-ordered packets into consecutive chunks of length p.
      - Each segment becomes one sample (a fresh pseudonym).
      - Train/Test split is by segment time.
      - y labels remain ground-truth MAC (only for scoring/linkability).
    """
    if MAC_COL not in df.columns:
        raise ValueError(f"CSV missing required column '{MAC_COL}'")
    for c in selected_cfo_cols:
        if c not in df.columns:
            raise ValueError(f"Selected CFO column missing from CSV: {c}")

    macs = sorted(df[MAC_COL].dropna().astype(str).unique().tolist())
    if not macs:
        raise ValueError("No MAC addresses found in CSV.")

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    kept_macs = []

    feat_dim = len(selected_cfo_cols) * (int(use_mean) + int(use_std))
    needed_per_segment = 2 if use_std else 1

    for mac in macs:
        df_mac = df[df[MAC_COL].astype(str) == mac]
        df_mac = sort_packets_for_temporal_split(df_mac)

        if not stress_mode:
            train_feat_vec: List[float] = []
            test_feat_vec: List[float] = []
            ok = True

            for col in selected_cfo_cols:
                vals = pd.to_numeric(df_mac[col], errors="coerce").values
                vals = vals[np.isfinite(vals)]
                if vals.size < MIN_PKTS_PER_MAC:
                    ok = False
                    break

                split_idx = int(vals.size * TRAIN_FRACTION)
                split_idx = max(1, min(split_idx, vals.size - 1))
                train_vals = vals[:split_idx]
                test_vals = vals[split_idx:]

                train_feat_vec.extend(compute_stats_vector(train_vals, use_mean, use_std))
                test_feat_vec.extend(compute_stats_vector(test_vals, use_mean, use_std))

            if not ok:
                continue

            train_arr = np.asarray(train_feat_vec, dtype=float)
            test_arr = np.asarray(test_feat_vec, dtype=float)

            if train_arr.shape[0] != feat_dim or test_arr.shape[0] != feat_dim:
                continue
            if not (np.all(np.isfinite(train_arr)) and np.all(np.isfinite(test_arr))):
                continue

            X_train_list.append(train_arr)
            X_test_list.append(test_arr)
            y_train_list.append(mac)
            y_test_list.append(mac)
            kept_macs.append(mac)

        else:
            if pseudonym_period is None or pseudonym_period < 1:
                raise ValueError("Stress mode requires pseudonym_period >= 1")

            # Keep minimal original filtering: must have enough packets overall (per column) for this MAC
            mac_ok = True
            for col in selected_cfo_cols:
                vals_all = pd.to_numeric(df_mac[col], errors="coerce").values
                vals_all = vals_all[np.isfinite(vals_all)]
                if vals_all.size < MIN_PKTS_PER_MAC:
                    mac_ok = False
                    break
            if not mac_ok:
                continue

            n_rows = len(df_mac)
            if n_rows < 2:
                continue

            seg_features: List[np.ndarray] = []
            for start in range(0, n_rows, pseudonym_period):
                end = min(start + pseudonym_period, n_rows)
                df_seg = df_mac.iloc[start:end]

                feat_vec: List[float] = []
                seg_ok = True
                for col in selected_cfo_cols:
                    v = pd.to_numeric(df_seg[col], errors="coerce").values
                    v = v[np.isfinite(v)]
                    if v.size < needed_per_segment:
                        seg_ok = False
                        break
                    feat_vec.extend(compute_stats_vector(v, use_mean, use_std))

                if not seg_ok:
                    continue

                arr = np.asarray(feat_vec, dtype=float)
                if arr.shape[0] != feat_dim:
                    continue
                if not np.all(np.isfinite(arr)):
                    continue
                seg_features.append(arr)

            if len(seg_features) < 2:
                continue

            split_idx = int(len(seg_features) * TRAIN_FRACTION)
            split_idx = max(1, min(split_idx, len(seg_features) - 1))

            train_segs = seg_features[:split_idx]
            test_segs = seg_features[split_idx:]

            for arr in train_segs:
                X_train_list.append(arr)
                y_train_list.append(mac)
            for arr in test_segs:
                X_test_list.append(arr)
                y_test_list.append(mac)

            kept_macs.append(mac)

    if not X_train_list or not X_test_list:
        raise ValueError(
            "No valid groups found after filtering. "
            f"Try lowering MIN_PKTS_PER_MAC (currently {MIN_PKTS_PER_MAC}), "
            "or (in stress mode) increasing p / selecting mean-only."
        )

    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    class_names = sorted(set([str(m) for m in kept_macs]))
    feature_names = build_feature_names(selected_cfo_cols, use_mean, use_std)

    if verbose:
        print(f"\nCSV: {FNAME}")
        print(f"MACs total: {len(macs)} | MACs kept: {len(class_names)} (MIN_PKTS_PER_MAC={MIN_PKTS_PER_MAC})")
        print(f"Selected CFO columns: {selected_cfo_cols}")
        print(f"Statistics used: {'mean ' if use_mean else ''}{'std' if use_std else ''}".strip())
        print(f"Feature dim: {X_train.shape[1]}")
        if not stress_mode:
            print(f"Training fraction per MAC: {TRAIN_FRACTION:.0%} / Testing: {1-TRAIN_FRACTION:.0%}")
            print(f"Train samples: {X_train.shape[0]} (one per MAC)")
            print(f"Test samples:  {X_test.shape[0]} (one per MAC)")
        else:
            print(f"STRESS MODE: ON (pseudonym_period p={pseudonym_period})")
            print(f"Training fraction per MAC (by segments): {TRAIN_FRACTION:.0%} / Testing: {1-TRAIN_FRACTION:.0%}")
            print(f"Train samples: {X_train.shape[0]} (segments, gallery)")
            print(f"Test samples:  {X_test.shape[0]} (segments, query)")

    return X_train, X_test, y_train, y_test, class_names, feature_names


# --------------------------- normal training & evaluation ---------------------------

def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, class_names, feature_names):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    importances = dict(zip(feature_names, rf.feature_importances_)) if feature_names else None

    return {
        "mode": "classifier",
        "model": rf,
        "scaler": scaler,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "feature_importances": importances
    }


# --------------------------- stress evaluation: LINKABILITY ---------------------------

def _pairwise_sqeuclidean_chunked(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> np.ndarray:
    """
    Return D where D[i,j] = ||A[i]-B[j]||^2, computed in chunks over A to limit memory.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    B_norm = np.sum(B * B, axis=1)[None, :]  # (1, |B|)
    out = np.empty((A.shape[0], B.shape[0]), dtype=float)

    for i in range(0, A.shape[0], chunk):
        a = A[i:i+chunk]
        a_norm = np.sum(a * a, axis=1)[:, None]  # (chunk,1)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        out[i:i+chunk] = a_norm + B_norm - 2.0 * (a @ B.T)

    # numerical guard
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
    - Verification ROC-AUC using best-genuine vs best-impostor per query
    - Clustering ARI/NMI on (gallery+query) with KMeans(k=n_classes)
    """
    scaler = StandardScaler()
    G = scaler.fit_transform(X_gallery)
    Q = scaler.transform(X_query)

    # Distance matrix: (|Q|, |G|)
    D = _pairwise_sqeuclidean_chunked(Q, G, chunk=256)  # squared euclidean

    # Rank-1: nearest gallery segment for each query
    nn_idx = np.argmin(D, axis=1)
    nn_labels = y_gallery[nn_idx]
    rank1 = float(np.mean(nn_labels == y_query))

    # Verification AUC: per query, compare best genuine (min dist among same MAC)
    # vs best impostor (min dist among different MAC). Use similarity = -dist.
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

    # Clustering on all segments (unsupervised; MAC used only for scoring)
    X_all = np.vstack([X_gallery, X_query])
    y_all = np.concatenate([y_gallery, y_query])
    X_all_s = StandardScaler().fit_transform(X_all)

    # KMeans labels arbitrary; ARI/NMI are permutation-invariant
    try:
        km = KMeans(n_clusters=max(2, int(n_classes)), n_init=10, random_state=RANDOM_SEED)
        c = km.fit_predict(X_all_s)
        ari = float(adjusted_rand_score(y_all, c))
        nmi = float(normalized_mutual_info_score(y_all, c))
    except Exception:
        ari = float("nan")
        nmi = float("nan")

    return {
        "rank1": rank1,
        "auc": auc,
        "ari": ari,
        "nmi": nmi
    }


# --------------------------- visualization ---------------------------

def plot_feature_distribution(X_train, X_test, y_train, y_test, feature_names, outfile):
    """
    Side-by-side heatmaps. Train/test may have different number of samples (stress mode).
    """
    order_tr = np.argsort(y_train)
    labels_tr = y_train[order_tr]
    X_train_khz = X_train[order_tr] / 1e3

    order_te = np.argsort(y_test)
    labels_te = y_test[order_te]
    X_test_khz = X_test[order_te] / 1e3

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
    axes[0].set_title(f"Train features (first {TRAIN_FRACTION:.0%}) [kHz]")
    axes[0].tick_params(axis="x", rotation=60)

    sns.heatmap(
        X_test_khz,
        ax=axes[1],
        cmap="viridis",
        cbar=True,
        yticklabels=labels_te if len(labels_te) <= 60 else False,
        xticklabels=feature_names,
    )
    axes[1].set_title(f"Test features (last {1-TRAIN_FRACTION:.0%}) [kHz]")
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

    ax.set_title(f"Random Forest Confusion Matrix\nAccuracy: {results['accuracy']:.1%}",
                 fontsize=14, fontweight="bold")
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
    # normal mode fields
    classifier_results: Optional[Dict] = None,
    class_names: Optional[List[str]] = None,
    y_test: Optional[np.ndarray] = None,
    # stress mode fields
    stress_metrics: Optional[Dict[str, float]] = None,
    stress_p: Optional[int] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
):
    with open(outfile, "w") as f:
        f.write("=" * 80 + "\n")
        if mode == "classifier":
            f.write("BLE DEVICE FINGERPRINTING - RANDOM FOREST (MAC-GROUPED)\n")
        else:
            f.write("BLE DEVICE FINGERPRINTING - LINKABILITY (FRESH PSEUDONYMS)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Input CSV: {FNAME}\n")
        f.write(f"Grouping key: {MAC_COL}\n")
        f.write(f"MIN_PKTS_PER_MAC: {MIN_PKTS_PER_MAC}\n")
        f.write(f"Selected CFO columns: {', '.join(selected_cols)}\n")
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
            f.write(f"F1-Score:  {classifier_results['f1']:.2%}\n\n")
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
    print("=" * 80)
    print("BLE DEVICE FINGERPRINTING using CFO Statistics (MAC-GROUPED)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/")

    if not os.path.exists(FNAME):
        print(f"ERROR: Missing input file: {FNAME}")
        sys.exit(1)

    df = pd.read_csv(FNAME)

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
# BLE Device Fingerprinting using CFO Statistics (grouped by MAC address)

# - Loads one consolidated CSV (mobile_office_all2.csv)
# - Groups packets by MAC address column (adv_addr). Each MAC = one "device".
# - For each MAC:
#     * Sort by pcap_ts if present, else keep CSV order
#     * Temporal split: first 70% packets -> training features, remaining 30% -> testing features
#     * Features are computed from selected CFO columns (mean and/or std)

# Selectable CFO columns:
#   - cfo_quick_hz (auto-detected, fallback: any column containing 'cfo' and 'hz')
#   - cfo_equal_00_hz
#   - cfo_equal_11_hz
#   - cfo_jump_10_hz
#   - cfo_jump_01_hz

# Outputs:
#   - mobile_office_all_plots2/ble_fingerprint_classification_results.txt
#   - mobile_office_all_plots2/ble_fingerprint_confusion_matrix.png
#   - mobile_office_all_plots2/ble_fingerprint_feature_distribution.png
# """

# import os
# import sys
# import warnings
# from typing import List, Tuple, Dict, Optional

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_recall_fscore_support,
#     confusion_matrix, classification_report
# )

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# # --------------------------- config ---------------------------

# FNAME = "/home/mishra/sdrs/BLESDR/build/test1.csv"
# OUTPUT_DIR = "all_devices_static"

# MAC_COL = "adv_addr"          # MAC address key
# TRAIN_FRACTION = 0.7          # temporal split within each MAC
# MIN_PKTS_PER_MAC = 10         # minimum valid packets per MAC (per selected column) to keep class

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)

# OUT_RESULTS = os.path.join(OUTPUT_DIR, "ble_fingerprint_classification_results.txt")
# OUT_CONFUSION = os.path.join(OUTPUT_DIR, "ble_fingerprint_confusion_matrix.png")
# OUT_DISTRIBUTION = os.path.join(OUTPUT_DIR, "ble_fingerprint_feature_distribution.png")

# # CFO columns the user may choose
# OPTIONAL_CFO_COLS = [
#     "cfo_equal_00_hz",
#     "cfo_equal_11_hz",
#     "cfo_jump_10_hz",
#     "cfo_jump_01_hz",
# ]

# # --------------------------- helpers ---------------------------

# def find_primary_cfo_column(df: pd.DataFrame) -> str:
#     """Find the primary CFO column in Hz."""
#     cols = list(df.columns)
#     lower = [c.lower() for c in cols]

#     for c, lc in zip(cols, lower):
#         if lc in ["cfo_quick_hz", "est_cfo_hz", "cfo_exact_quick_hz"]:
#             return c

#     for c, lc in zip(cols, lower):
#         if "cfo" in lc and "hz" in lc:
#             return c

#     raise ValueError("No CFO column found in CSV")


# def get_feature_choice() -> Tuple[bool, bool]:
#     """Ask user which statistics to use: mean, std, or both."""
#     print("\n" + "=" * 80)
#     print("STATISTIC SELECTION")
#     print("=" * 80)
#     print("\nWhich statistics would you like to use per selected CFO column?")
#     print("  1) Mean only")
#     print("  2) Standard Deviation only")
#     print("  3) Both Mean and Standard Deviation")

#     while True:
#         try:
#             choice = input("\nEnter your choice (1-3): ").strip()
#             if choice == "1":
#                 print("\n✓ Selected: Mean only")
#                 return True, False
#             if choice == "2":
#                 print("\n✓ Selected: Std Dev only")
#                 return False, True
#             if choice == "3":
#                 print("\n✓ Selected: Mean and Std Dev")
#                 return True, True
#             print("Invalid choice. Please enter 1, 2, or 3.")
#         except KeyboardInterrupt:
#             print("\n\nAborted by user.")
#             sys.exit(0)


# def choose_cfo_columns(df: pd.DataFrame) -> List[str]:
#     """
#     Ask user which CFO columns to include.
#     Always offers the primary CFO column + the transition CFO columns if present.
#     """
#     primary = find_primary_cfo_column(df)
#     available = set(df.columns)

#     choices = [(primary, True)]  # primary on by default
#     for c in OPTIONAL_CFO_COLS:
#         if c in available:
#             choices.append((c, False))

#     print("\n" + "=" * 80)
#     print("CFO COLUMN SELECTION")
#     print("=" * 80)
#     print("\nSelect which CFO columns to use (features will be computed per column).")
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
#     """Sort packets inside one MAC group for temporal splitting."""
#     if "pcap_ts" in df_mac.columns:
#         ts = pd.to_numeric(df_mac["pcap_ts"], errors="coerce")
#         if np.isfinite(ts).any():
#             return df_mac.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"])
#     return df_mac


# def compute_stats_vector(vals: np.ndarray, use_mean: bool, use_std: bool) -> List[float]:
#     feats: List[float] = []
#     if use_mean:
#         feats.append(float(np.mean(vals)) if vals.size else np.nan)
#     if use_std:
#         feats.append(float(np.std(vals)) if vals.size > 1 else np.nan)
#     return feats


# def build_feature_names(selected_cols: List[str], use_mean: bool, use_std: bool) -> List[str]:
#     names = []
#     for col in selected_cols:
#         if use_mean:
#             names.append(f"{col}:mean")
#         if use_std:
#             names.append(f"{col}:std")
#     return names


# def collect_all_data_by_mac(
#     df: pd.DataFrame,
#     selected_cfo_cols: List[str],
#     use_mean: bool,
#     use_std: bool
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
#     """
#     Build one training sample + one testing sample per MAC address.
#     Feature vector = concatenation of requested stats for each selected CFO column.
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

#     feat_dim = len(selected_cfo_cols) * (int(use_mean) + int(use_std))

#     for mac in macs:
#         df_mac = df[df[MAC_COL].astype(str) == mac]
#         df_mac = sort_packets_for_temporal_split(df_mac)

#         # For each selected CFO column, compute train/test features; require enough data.
#         train_feat_vec: List[float] = []
#         test_feat_vec: List[float] = []

#         ok = True
#         for col in selected_cfo_cols:
#             vals = pd.to_numeric(df_mac[col], errors="coerce").values
#             vals = vals[np.isfinite(vals)]

#             if vals.size < MIN_PKTS_PER_MAC:
#                 ok = False
#                 break

#             split_idx = int(vals.size * TRAIN_FRACTION)
#             split_idx = max(1, min(split_idx, vals.size - 1))

#             train_vals = vals[:split_idx]
#             test_vals = vals[split_idx:]

#             train_feat_vec.extend(compute_stats_vector(train_vals, use_mean, use_std))
#             test_feat_vec.extend(compute_stats_vector(test_vals, use_mean, use_std))

#         if not ok:
#             continue

#         train_arr = np.asarray(train_feat_vec, dtype=float)
#         test_arr = np.asarray(test_feat_vec, dtype=float)

#         if train_arr.shape[0] != feat_dim or test_arr.shape[0] != feat_dim:
#             continue
#         if not (np.all(np.isfinite(train_arr)) and np.all(np.isfinite(test_arr))):
#             continue

#         X_train_list.append(train_arr)
#         X_test_list.append(test_arr)
#         y_train_list.append(mac)
#         y_test_list.append(mac)
#         kept_macs.append(mac)

#     if not X_train_list:
#         raise ValueError(
#             "No valid MAC groups found after filtering. "
#             f"Try lowering MIN_PKTS_PER_MAC (currently {MIN_PKTS_PER_MAC})."
#         )

#     X_train = np.vstack(X_train_list)
#     X_test = np.vstack(X_test_list)
#     y_train = np.array(y_train_list)
#     y_test = np.array(y_test_list)

#     class_names = sorted(set(kept_macs))
#     feature_names = build_feature_names(selected_cfo_cols, use_mean, use_std)

#     print(f"\nCSV: {FNAME}")
#     print(f"MACs total: {len(macs)} | MACs kept: {len(class_names)} (MIN_PKTS_PER_MAC={MIN_PKTS_PER_MAC})")
#     print(f"Selected CFO columns: {selected_cfo_cols}")
#     print(f"Statistics used: {'mean ' if use_mean else ''}{'std' if use_std else ''}".strip())
#     print(f"Feature dim: {X_train.shape[1]}")
#     print(f"Training fraction per MAC: {TRAIN_FRACTION:.0%} / Testing: {1-TRAIN_FRACTION:.0%}")
#     print(f"Train samples: {X_train.shape[0]} (one per MAC)")
#     print(f"Test samples:  {X_test.shape[0]} (one per MAC)")

#     return X_train, X_test, y_train, y_test, class_names, feature_names


# # --------------------------- training & evaluation ---------------------------

# def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, feature_names):
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     print("\nTraining Random Forest...")
#     rf = RandomForestClassifier(
#         n_estimators=400,
#         max_depth=12,
#         min_samples_split=2,
#         random_state=RANDOM_SEED,
#         n_jobs=-1
#     )
#     rf.fit(X_train_scaled, y_train)
#     y_pred = rf.predict(X_test_scaled)

#     accuracy = accuracy_score(y_test, y_pred)
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_test, y_pred, average="weighted", zero_division=0
#     )
#     cm = confusion_matrix(y_test, y_pred, labels=class_names)

#     importances = dict(zip(feature_names, rf.feature_importances_)) if feature_names else None

#     return {
#         "model": rf,
#         "scaler": scaler,
#         "y_pred": y_pred,
#         "accuracy": accuracy,
#         "precision": prec,
#         "recall": rec,
#         "f1": f1,
#         "confusion_matrix": cm,
#         "feature_importances": importances
#     }


# # --------------------------- visualization ---------------------------

# def plot_feature_distribution(X_train, X_test, y_train, feature_names, outfile):
#     """
#     For multi-dimensional feature vectors, show train and test side-by-side heatmaps (kHz-scaled).
#     """
#     order = np.argsort(y_train)
#     labels = y_train[order]

#     # Convert Hz->kHz for CFO-related features (all are Hz here)
#     X_train_khz = X_train[order] / 1e3
#     X_test_khz = X_test[order] / 1e3

#     fig_h = max(6, 0.25 * len(labels))
#     fig_w = max(10, 0.35 * len(feature_names))

#     fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

#     sns.heatmap(
#         X_train_khz,
#         ax=axes[0],
#         cmap="viridis",
#         cbar=True,
#         yticklabels=labels if len(labels) <= 60 else False,
#         xticklabels=feature_names,
#     )
#     axes[0].set_title(f"Train features (first {TRAIN_FRACTION:.0%}) [kHz]")
#     axes[0].tick_params(axis="x", rotation=60)

#     sns.heatmap(
#         X_test_khz,
#         ax=axes[1],
#         cmap="viridis",
#         cbar=True,
#         yticklabels=False,
#         xticklabels=feature_names,
#     )
#     axes[1].set_title(f"Test features (last {1-TRAIN_FRACTION:.0%}) [kHz]")
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

#     ax.set_title(f"Random Forest Confusion Matrix\nAccuracy: {results['accuracy']:.1%}",
#                  fontsize=14, fontweight="bold")
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
#         f.write(f"MIN_PKTS_PER_MAC: {MIN_PKTS_PER_MAC}\n\n")

#         f.write(f"Selected CFO columns: {', '.join(selected_cols)}\n")
#         f.write(f"Stats: {'mean ' if use_mean else ''}{'std' if use_std else ''}\n".strip() + "\n")
#         f.write(f"Training fraction: {TRAIN_FRACTION:.0%} per MAC\n\n")

#         f.write(f"Number of MAC classes: {len(class_names)}\n")
#         f.write(f"Train samples: {len(y_train)} (one per MAC)\n")
#         f.write(f"Test samples:  {len(y_test)} (one per MAC)\n\n")

#         f.write("-" * 80 + "\n")
#         f.write("FEATURE NAMES\n")
#         f.write("-" * 80 + "\n")
#         for i, name in enumerate(feature_names):
#             f.write(f"{i:3d}: {name}\n")
#         f.write("\n")

#         f.write("-" * 80 + "\n")
#         f.write("FEATURE VALUES (Hz)\n")
#         f.write("-" * 80 + "\n")

#         order = np.argsort(y_train)
#         f.write("MAC," + ",".join([f"train_{n}" for n in feature_names]) + "," + ",".join([f"test_{n}" for n in feature_names]) + "\n")
#         for i in order:
#             row = [y_train[i]] + [f"{v:.6f}" for v in X_train[i]] + [f"{v:.6f}" for v in X_test[i]]
#             f.write(",".join(row) + "\n")
#         f.write("\n")

#         if results["feature_importances"] is not None:
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
#         f.write(f"F1-Score:  {results['f1']:.2%}\n\n")

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
#     print("=" * 80)
#     print("BLE DEVICE FINGERPRINTING using CFO Statistics (MAC-GROUPED)")
#     print("=" * 80)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     print(f"\nOutput directory: {OUTPUT_DIR}/")

#     if not os.path.exists(FNAME):
#         print(f"ERROR: Missing input file: {FNAME}")
#         sys.exit(1)

#     df = pd.read_csv(FNAME)

#     use_mean, use_std = get_feature_choice()
#     selected_cols = choose_cfo_columns(df)

#     print("\nBuilding per-MAC train/test feature sets...")
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