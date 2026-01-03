#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLE Device Fingerprinting using CFO Statistics (grouped by MAC/AdvA)

- Loads one consolidated CSV (cfo_samples_rail.csv)
- Groups packets by MAC address column (AdvA / adv_addr). Each MAC = one "device".
- For each MAC:
    * Sort by pcap_ts if present, else keep CSV order
    * Temporal split: first 70% packets -> training features, remaining 30% -> testing features
    * Features are computed from selected CFO columns (mean and/or std)

Selectable CFO columns (from cfo_samples_rail.csv):
  - Primary CFO: prefers "CFO_Hz" if present, else auto-detect any column containing 'cfo' and 'hz'
  - Transition CFOs (if present):
      * CFO_00_Hz, CFO_11_Hz, CFO_10_Hz, CFO_01_Hz
      * CFO_from_transitions_Hz

Outputs:
  - all_devices_static/ble_fingerprint_classification_results.txt
  - all_devices_static/ble_fingerprint_confusion_matrix.png
  - all_devices_static/ble_fingerprint_feature_distribution.png
"""

import os
import sys
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- config ---------------------------

FNAME = "cfo_samples_rail.csv"
OUTPUT_DIR = "all_devices_static_rail"

# NOTE: resolved at runtime to "AdvA" or "adv_addr" (or case-insensitive match)
MAC_COL = "AdvA"

TRAIN_FRACTION = 0.7          # temporal split within each MAC
MIN_PKTS_PER_MAC = 10         # minimum valid packets per MAC (per selected column) to keep class

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_RESULTS = os.path.join(OUTPUT_DIR, "ble_fingerprint_classification_results.txt")
OUT_CONFUSION = os.path.join(OUTPUT_DIR, "ble_fingerprint_confusion_matrix.png")
OUT_DISTRIBUTION = os.path.join(OUTPUT_DIR, "ble_fingerprint_feature_distribution.png")

# Transition CFO columns likely present in cfo_samples_rail.csv (plus fallbacks)
OPTIONAL_CFO_COLS = [
    "CFO_00_Hz",
    "CFO_11_Hz",
    "CFO_10_Hz",
    "CFO_01_Hz",
    "CFO_from_transitions_Hz",
]

# --------------------------- helpers ---------------------------

def _col_lookup_case_insensitive(df: pd.DataFrame, name: str) -> str:
    """Return actual column name matching `name` case-insensitively, or ''."""
    target = name.lower()
    for c in df.columns:
        if str(c).lower() == target:
            return str(c)
    return ""


def resolve_mac_column(df: pd.DataFrame) -> str:
    """
    Pick the grouping key column from cfo_samples_rail.csv.
    Prefers: AdvA, then adv_addr, then any column containing 'adv' and 'a'/'addr'.
    """
    for cand in ["AdvA", "adv_addr", "advA", "ADV_A", "adv_address", "advaddr"]:
        hit = _col_lookup_case_insensitive(df, cand)
        if hit:
            return hit

    # loose fallback
    lowers = [str(c).lower() for c in df.columns]
    for c, lc in zip(df.columns, lowers):
        if ("adv" in lc) and ("addr" in lc or lc.endswith("a") or "adva" in lc):
            return str(c)

    raise ValueError("Could not find MAC/AdvA column (expected 'AdvA' or 'adv_addr').")


def find_primary_cfo_column(df: pd.DataFrame) -> str:
    """Find the primary CFO column in Hz. Prefers CFO_Hz if present."""
    cols = list(df.columns)
    lower = [str(c).lower() for c in cols]

    # Prefer the output of your sniffer CSV
    for c, lc in zip(cols, lower):
        if lc in ["cfo_hz", "cfo_quick_hz", "est_cfo_hz", "cfo_exact_quick_hz"]:
            return str(c)

    # Any CFO Hz column
    for c, lc in zip(cols, lower):
        if "cfo" in lc and "hz" in lc:
            return str(c)

    raise ValueError("No CFO column found in CSV")


def get_feature_choice() -> Tuple[bool, bool]:
    """Ask user which statistics to use: mean, std, or both."""
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
    """
    Ask user which CFO columns to include.
    Offers the primary CFO column + transition CFO columns if present.
    """
    primary = find_primary_cfo_column(df)
    available = set(map(str, df.columns))

    # Build choices: (col_name, default_on)
    choices = [(primary, True)]

    # Add transition CFOs if present (case-insensitive)
    for c in OPTIONAL_CFO_COLS:
        hit = _col_lookup_case_insensitive(df, c)
        if hit and hit in available and hit != primary:
            choices.append((hit, False))

    print("\n" + "=" * 80)
    print("CFO TYPE SELECTION")
    print("=" * 80)
    print("\nSelect which CFO types to use (features will be computed per column).")
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


def sort_packets_for_temporal_split(df_mac: pd.DataFrame) -> pd.DataFrame:
    """Sort packets inside one MAC group for temporal splitting."""
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


def collect_all_data_by_mac(
    df: pd.DataFrame,
    selected_cfo_cols: List[str],
    use_mean: bool,
    use_std: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build one training sample + one testing sample per MAC address.
    Feature vector = concatenation of requested stats for each selected CFO column.
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

    for mac in macs:
        df_mac = df[df[MAC_COL].astype(str) == mac]
        df_mac = sort_packets_for_temporal_split(df_mac)

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

    if not X_train_list:
        raise ValueError(
            "No valid MAC groups found after filtering. "
            f"Try lowering MIN_PKTS_PER_MAC (currently {MIN_PKTS_PER_MAC})."
        )

    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    class_names = sorted(set(kept_macs))
    feature_names = build_feature_names(selected_cfo_cols, use_mean, use_std)

    print(f"\nCSV: {FNAME}")
    print(f"Grouping key: {MAC_COL}")
    print(f"MACs total: {len(macs)} | MACs kept: {len(class_names)} (MIN_PKTS_PER_MAC={MIN_PKTS_PER_MAC})")
    print(f"Selected CFO columns: {selected_cfo_cols}")
    print(f"Statistics used: {'mean ' if use_mean else ''}{'std' if use_std else ''}".strip())
    print(f"Feature dim: {X_train.shape[1]}")
    print(f"Training fraction per MAC: {TRAIN_FRACTION:.0%} / Testing: {1-TRAIN_FRACTION:.0%}")
    print(f"Train samples: {X_train.shape[0]} (one per MAC)")
    print(f"Test samples:  {X_test.shape[0]} (one per MAC)")

    return X_train, X_test, y_train, y_test, class_names, feature_names


# --------------------------- training & evaluation ---------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, feature_names):
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


# --------------------------- visualization ---------------------------

def plot_feature_distribution(X_train, X_test, y_train, feature_names, outfile):
    """
    For multi-dimensional feature vectors, show train and test side-by-side heatmaps (kHz-scaled).
    """
    order = np.argsort(y_train)
    labels = y_train[order]

    X_train_khz = X_train[order] / 1e3
    X_test_khz = X_test[order] / 1e3

    fig_h = max(6, 0.25 * len(labels))
    fig_w = max(10, 0.35 * len(feature_names))

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    sns.heatmap(
        X_train_khz,
        ax=axes[0],
        cmap="viridis",
        cbar=True,
        yticklabels=labels if len(labels) <= 60 else False,
        xticklabels=feature_names,
    )
    axes[0].set_title(f"Train features (first {TRAIN_FRACTION:.0%}) [kHz]")
    axes[0].tick_params(axis="x", rotation=60)

    sns.heatmap(
        X_test_khz,
        ax=axes[1],
        cmap="viridis",
        cbar=True,
        yticklabels=False,
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


def write_results_report(results, class_names, y_test, X_train, X_test, y_train,
                        feature_names, selected_cols, use_mean, use_std, outfile):
    with open(outfile, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BLE DEVICE FINGERPRINTING - RANDOM FOREST (MAC-GROUPED)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Input CSV: {FNAME}\n")
        f.write(f"Grouping key: {MAC_COL} (each MAC treated as one device)\n")
        f.write(f"MIN_PKTS_PER_MAC: {MIN_PKTS_PER_MAC}\n\n")

        f.write(f"Selected CFO columns: {', '.join(selected_cols)}\n")
        f.write(f"Stats: {'mean ' if use_mean else ''}{'std' if use_std else ''}\n".strip() + "\n")
        f.write(f"Training fraction: {TRAIN_FRACTION:.0%} per MAC\n\n")

        f.write(f"Number of MAC classes: {len(class_names)}\n")
        f.write(f"Train samples: {len(y_train)} (one per MAC)\n")
        f.write(f"Test samples:  {len(y_test)} (one per MAC)\n\n")

        f.write("-" * 80 + "\n")
        f.write("FEATURE NAMES\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(feature_names):
            f.write(f"{i:3d}: {name}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("FEATURE VALUES (Hz)\n")
        f.write("-" * 80 + "\n")

        order = np.argsort(y_train)
        f.write("MAC," + ",".join([f"train_{n}" for n in feature_names]) + "," + ",".join([f"test_{n}" for n in feature_names]) + "\n")
        for i in order:
            row = [y_train[i]] + [f"{v:.6f}" for v in X_train[i]] + [f"{v:.6f}" for v in X_test[i]]
            f.write(",".join(row) + "\n")
        f.write("\n")

        if results["feature_importances"] is not None:
            f.write("-" * 80 + "\n")
            f.write("FEATURE IMPORTANCES\n")
            f.write("-" * 80 + "\n")
            for feat, imp in sorted(results["feature_importances"].items(), key=lambda x: -x[1]):
                f.write(f"{feat:<35}: {imp:.6f}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("RANDOM FOREST PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.2%}\n")
        f.write(f"Precision: {results['precision']:.2%}\n")
        f.write(f"Recall:    {results['recall']:.2%}\n")
        f.write(f"F1-Score:  {results['f1']:.2%}\n\n")

        f.write("Per-MAC Classification Report:\n")
        f.write(classification_report(
            y_test, results["y_pred"],
            labels=class_names,
            target_names=class_names,
            zero_division=0
        ))
        f.write("\n")

    print(f"[✓] Saved detailed results: {outfile}")


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

    # Resolve MAC column for this CSV (AdvA vs adv_addr, etc.)
    MAC_COL = resolve_mac_column(df)
    print(f"\nDetected MAC column: {MAC_COL}")

    use_mean, use_std = get_feature_choice()
    selected_cols = choose_cfo_columns(df)

    print("\nBuilding per-MAC train/test feature sets...")
    X_train, X_test, y_train, y_test, class_names, feature_names = collect_all_data_by_mac(
        df, selected_cols, use_mean, use_std
    )

    if len(class_names) < 2:
        print("ERROR: Need at least 2 MACs (classes) for classification!")
        sys.exit(1)

    print("\nTraining classifier...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test, class_names, feature_names)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1-Score:  {results['f1']:.2%}")

    print("\nGenerating outputs...")
    plot_feature_distribution(X_train, X_test, y_train, feature_names, OUT_DISTRIBUTION)
    plot_confusion_matrix(results, class_names, OUT_CONFUSION)
    write_results_report(results, class_names, y_test, X_train, X_test, y_train,
                        feature_names, selected_cols, use_mean, use_std, OUT_RESULTS)

    print("\n" + "=" * 80)
    print("DONE! Check output files for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    main()