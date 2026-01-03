#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLE Device Fingerprinting using CFO Statistics (grouped by MAC address)

- Loads one consolidated CSV (cfo_data.csv)
- Groups packets by MAC address column (mac). Each MAC = one "device".
- For each MAC:
    * Sort by row order (no timestamp available)
    * Temporal split: first 70% packets -> training features, remaining 30% -> testing features
    * Features are computed from selected CFO columns (mean and/or std)

Selectable CFO columns:
  - cfoTot (total CFO - mean over all bytes)
  - w0, w2, w4, w8 (Hamming-weight specific CFOs)

Outputs:
  - cfo_fingerprinting_results/ble_fingerprint_classification_results.txt
  - cfo_fingerprinting_results/ble_fingerprint_confusion_matrix.png
  - cfo_fingerprinting_results/ble_fingerprint_feature_distribution.png
"""

import os
import sys
import warnings
from typing import List, Tuple, Dict, Optional

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

FNAME = "/home/mishra/BlueShield/cfo_data_static_all.csv" #"/home/mishra/BlueShield/cfo_data.csv"
OUTPUT_DIR = "cfo_fingerprinting_results_all"

MAC_COL = "mac"               # MAC address key
TRAIN_FRACTION = 0.7          # temporal split within each MAC for train/test
MIN_PKTS_PER_MAC = 50         # minimum valid packets per MAC (per selected column) to keep class

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_RESULTS = os.path.join(OUTPUT_DIR, "ble_fingerprint_classification_results.txt")
OUT_CONFUSION = os.path.join(OUTPUT_DIR, "ble_fingerprint_confusion_matrix.png")
OUT_DISTRIBUTION = os.path.join(OUTPUT_DIR, "ble_fingerprint_feature_distribution.png")

# CFO columns available (Hamming weight based)
AVAILABLE_CFO_COLS = [
    "cfoTot",
    "w0",
    "w2",
    "w4",
    "w8",
]

# --------------------------- helpers ---------------------------

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
    Offers cfoTot (default) and Hamming weight columns (w0, w2, w4, w8).
    """
    available = set(df.columns)
    choices = []
    
    for col in AVAILABLE_CFO_COLS:
        if col in available:
            is_default = (col == "cfoTot")
            choices.append((col, is_default))

    if not choices:
        raise ValueError("No CFO columns found in CSV!")

    print("\n" + "=" * 80)
    print("CFO COLUMN SELECTION")
    print("=" * 80)
    print("\nSelect which CFO columns to use (features will be computed per column).")
    print("Enter a comma-separated list of numbers, e.g., 1,3,4")
    print("Press Enter for default (cfoTot only).\n")

    for i, (col, default_on) in enumerate(choices, 1):
        tag = "DEFAULT" if default_on else ""
        desc = ""
        if col == "cfoTot":
            desc = "(Total CFO - mean over all bytes)"
        elif col.startswith("w"):
            desc = f"(Hamming weight {col})"
        print(f"  {i}) {col} {desc} {tag}")

    while True:
        try:
            raw = input("\nYour selection: ").strip()
            if raw == "":
                selected = ["cfoTot"]
                print(f"\n✓ Selected default: [cfoTot]")
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

        # For each selected CFO column, compute train/test features; require enough data.
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

    print("\nTraining optimized Random Forest...")
    print("Note: With only 1 sample per MAC, we use carefully chosen hyperparameters")
    
    # For small datasets with 1 sample per class, use parameters optimized for 
    # low-variance, high-diversity ensemble
    rf = RandomForestClassifier(
        n_estimators=1000,        # More trees for stability
        max_depth=None,           # No depth limit for flexibility
        min_samples_split=2,      # Minimum for splitting
        min_samples_leaf=1,       # Allow single-sample leaves
        max_features='sqrt',      # Reduce correlation between trees
        criterion='entropy',      # Information gain often works better for fingerprinting
        bootstrap=True,           # Use bootstrap sampling
        oob_score=True,          # Out-of-bag score as internal validation
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    print(f"Out-of-bag score (internal validation): {rf.oob_score_:.2%}")
    
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
        "feature_importances": importances,
        "oob_score": rf.oob_score_
    }


# --------------------------- visualization ---------------------------

def plot_feature_distribution(X_train, X_test, y_train, feature_names, outfile):
    """
    For multi-dimensional feature vectors, show train and test side-by-side heatmaps (kHz-scaled).
    """
    order = np.argsort(y_train)
    labels = y_train[order]

    # Convert Hz->kHz for CFO-related features (all are Hz here)
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
        f.write("Hamming Weight CFO Features\n")
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
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write("Random Forest optimized for small dataset:\n")
        f.write("  - n_estimators: 1000 (high for stability)\n")
        f.write("  - max_depth: None (unlimited)\n")
        f.write("  - max_features: sqrt (reduce tree correlation)\n")
        f.write("  - criterion: entropy (information gain)\n")
        f.write("  - bootstrap: True with out-of-bag scoring\n")
        if "oob_score" in results:
            f.write(f"\nOut-of-bag score: {results['oob_score']:.2%}\n")
        f.write("\n")

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
    print("=" * 80)
    print("BLE DEVICE FINGERPRINTING using Hamming Weight CFO (MAC-GROUPED)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/")

    if not os.path.exists(FNAME):
        print(f"ERROR: Missing input file: {FNAME}")
        sys.exit(1)

    df = pd.read_csv(FNAME)

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