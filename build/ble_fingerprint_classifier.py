#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLE Device Fingerprinting using CFO Statistics

For each device:
- Computes statistics (mean, std dev) of CFO from first 70% of packets (training)
- Computes statistics of CFO from remaining 30% of packets (testing)

User can choose which features to use: mean only, std only, or both.

Uses Random Forest classifier.

Expected CSV files:
  ble_packets_fingerprints_with_headers_{dtype}{idx}.csv
  where dtype ∈ {apple, mi, hello, smart} and idx ∈ {1,2,3,4}

Outputs:
  - ble_fingerprint_classification_results.txt (metrics)
  - ble_fingerprint_confusion_matrix.png (confusion matrix)
  - ble_fingerprint_feature_distribution.png (feature distribution)
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple

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

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --------------------------- config ---------------------------

DEVICE_GROUPS = {
    "mi":     "MiTag",
    "apple":  "AirTag",
    "hello":  "HelloTag",
    "smart":  "SmartTag",
}

SUB_IDS = [1, 2, 3, 4]
FNAME_TPL = "{dtype}{idx}.csv"

# Training fraction (temporal split)
TRAIN_FRACTION = 0.7  # First 70% of packets for computing training features

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Output files
OUT_RESULTS = "ble_fingerprint_classification_results.txt"
OUT_CONFUSION = "ble_fingerprint_confusion_matrix.png"
OUT_DISTRIBUTION = "ble_fingerprint_feature_distribution.png"


# --------------------------- helpers ---------------------------

def find_cfo_column(df: pd.DataFrame) -> str:
    """Find the CFO column in Hz."""
    cols = [c for c in df.columns]
    lower = [c.lower() for c in cols]
    
    # Prefer specific names
    for c, lc in zip(cols, lower):
        if lc in ["cfo_quick_hz", "est_cfo_hz", "cfo_exact_quick_hz"]:
            return c
    
    # Heuristic: contains 'cfo' and 'hz'
    for c, lc in zip(cols, lower):
        if "cfo" in lc and "hz" in lc:
            return c
    
    raise ValueError("No CFO column found in CSV")


def get_feature_choice() -> Tuple[bool, bool]:
    """
    Ask user which features to use.
    
    Returns:
        (use_mean, use_std)
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    print("\nWhich CFO features would you like to use for fingerprinting?")
    print("  1) Mean only")
    print("  2) Standard Deviation only")
    print("  3) Both Mean and Standard Deviation")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                print("\n✓ Selected: Mean CFO only")
                return True, False
            elif choice == '2':
                print("\n✓ Selected: Standard Deviation CFO only")
                return False, True
            elif choice == '3':
                print("\n✓ Selected: Both Mean and Standard Deviation CFO")
                return True, True
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            sys.exit(0)


def load_device_cfo_features(dtype: str, idx: int, use_mean: bool, use_std: bool) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load CFO data for one device and compute requested features for train/test splits.
    
    Args:
        dtype: Device type (apple, mi, etc.)
        idx: Device index (1-4)
        use_mean: Whether to include mean as feature
        use_std: Whether to include std as feature
    
    Returns:
        (train_features, test_features, label)
        where features can be [mean], [std], or [mean, std]
    """
    fname = FNAME_TPL.format(dtype=dtype, idx=idx)
    
    if not os.path.exists(fname):
        warnings.warn(f"Missing file: {fname}")
        n_features = int(use_mean) + int(use_std)
        return np.full(n_features, np.nan), np.full(n_features, np.nan), ""
    
    df = pd.read_csv(fname)
    
    if len(df) == 0:
        n_features = int(use_mean) + int(use_std)
        return np.full(n_features, np.nan), np.full(n_features, np.nan), ""
    
    # Find CFO column
    cfo_col = find_cfo_column(df)
    
    # Extract CFO values
    cfo_vals = pd.to_numeric(df[cfo_col], errors='coerce').values
    
    # Remove NaN and inf
    cfo_vals = cfo_vals[np.isfinite(cfo_vals)]
    
    if len(cfo_vals) < 10:  # Need reasonable number of packets
        warnings.warn(f"Too few valid CFO values in {fname}")
        n_features = int(use_mean) + int(use_std)
        return np.full(n_features, np.nan), np.full(n_features, np.nan), ""
    
    # Temporal split
    split_idx = int(len(cfo_vals) * TRAIN_FRACTION)
    
    train_cfos = cfo_vals[:split_idx]
    test_cfos = cfo_vals[split_idx:]
    
    # Compute requested features
    train_features = []
    test_features = []
    
    if use_mean:
        train_features.append(np.mean(train_cfos) if len(train_cfos) > 0 else np.nan)
        test_features.append(np.mean(test_cfos) if len(test_cfos) > 0 else np.nan)
    
    if use_std:
        train_features.append(np.std(train_cfos) if len(train_cfos) > 1 else np.nan)
        test_features.append(np.std(test_cfos) if len(test_cfos) > 1 else np.nan)
    
    # Create label
    label = f"{DEVICE_GROUPS[dtype]}{idx}"
    
    return np.array(train_features), np.array(test_features), label


def collect_all_data(use_mean: bool, use_std: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Collect CFO features for all devices.
    
    Returns:
        (X_train, X_test, y_train, y_test, labels, feature_names)
    """
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    all_labels = []
    
    for dtype in DEVICE_GROUPS.keys():
        for idx in SUB_IDS:
            train_features, test_features, label = load_device_cfo_features(dtype, idx, use_mean, use_std)
            
            if not np.all(np.isfinite(train_features)) or not np.all(np.isfinite(test_features)):
                continue
            
            X_train_list.append(train_features)
            X_test_list.append(test_features)
            y_train_list.append(label)
            y_test_list.append(label)
            all_labels.append(label)
    
    if len(X_train_list) == 0:
        raise ValueError("No valid data found!")
    
    # Convert to arrays
    X_train = np.array(X_train_list)
    X_test = np.array(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    unique_labels = sorted(list(set(all_labels)))
    
    # Build feature names
    feature_names = []
    if use_mean:
        feature_names.append('Mean CFO')
    if use_std:
        feature_names.append('Std Dev CFO')
    
    print(f"\nLoaded {len(X_train)} devices")
    print(f"Features: {feature_names}")
    print(f"  Computed from first {TRAIN_FRACTION:.0%} of packets (training)")
    print(f"  Computed from remaining {1-TRAIN_FRACTION:.0%} of packets (testing)")
    print(f"Training samples: {len(X_train)} (one per device)")
    print(f"Testing samples: {len(X_test)} (one per device)")
    
    return X_train, X_test, y_train, y_test, unique_labels, feature_names


# --------------------------- training & evaluation ---------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, feature_names):
    """Train Random Forest classifier and return results."""
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    # Feature importances (only if multiple features)
    importances = None
    if len(feature_names) > 1:
        importances = dict(zip(feature_names, rf.feature_importances_))
    
    results = {
        'model': rf,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'feature_importances': importances
    }
    
    return results


# --------------------------- visualization ---------------------------

def plot_feature_distribution(X_train, X_test, y_train, class_names, feature_names, use_mean, use_std, outfile):
    """Plot distribution of feature values."""
    
    # Determine layout based on features
    n_features = len(feature_names)
    if n_features == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    
    # Sort by device labels
    sorted_indices = np.argsort(y_train)
    x_positions = np.arange(len(class_names))
    labels = y_train[sorted_indices]
    
    # Convert to kHz for better readability
    X_train_khz = X_train[sorted_indices] / 1e3
    X_test_khz = X_test[sorted_indices] / 1e3
    
    plot_idx = 0
    
    if use_mean:
        # Training Mean
        axes[plot_idx].bar(x_positions, X_train_khz[:, 0] if n_features > 1 else X_train_khz.flatten(), 
                          alpha=0.7, color='steelblue', edgecolor='black')
        axes[plot_idx].set_xticks(x_positions)
        axes[plot_idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[plot_idx].set_ylabel('Mean CFO (kHz)')
        axes[plot_idx].set_title(f'Training: Mean CFO (First {TRAIN_FRACTION:.0%})')
        axes[plot_idx].grid(axis='y', alpha=0.3)
        plot_idx += 1
        
        # Testing Mean
        axes[plot_idx].bar(x_positions, X_test_khz[:, 0] if n_features > 1 else X_test_khz.flatten(),
                          alpha=0.7, color='coral', edgecolor='black')
        axes[plot_idx].set_xticks(x_positions)
        axes[plot_idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[plot_idx].set_ylabel('Mean CFO (kHz)')
        axes[plot_idx].set_title(f'Testing: Mean CFO (Remaining {1-TRAIN_FRACTION:.0%})')
        axes[plot_idx].grid(axis='y', alpha=0.3)
        plot_idx += 1
    
    if use_std:
        col_idx = 1 if (use_mean and n_features > 1) else 0
        
        # Training Std Dev
        axes[plot_idx].bar(x_positions, X_train_khz[:, col_idx] if n_features > 1 else X_train_khz.flatten(),
                          alpha=0.7, color='seagreen', edgecolor='black')
        axes[plot_idx].set_xticks(x_positions)
        axes[plot_idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[plot_idx].set_ylabel('Std Dev CFO (kHz)')
        axes[plot_idx].set_title(f'Training: Std Dev CFO (First {TRAIN_FRACTION:.0%})')
        axes[plot_idx].grid(axis='y', alpha=0.3)
        plot_idx += 1
        
        # Testing Std Dev
        axes[plot_idx].bar(x_positions, X_test_khz[:, col_idx] if n_features > 1 else X_test_khz.flatten(),
                          alpha=0.7, color='gold', edgecolor='black')
        axes[plot_idx].set_xticks(x_positions)
        axes[plot_idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[plot_idx].set_ylabel('Std Dev CFO (kHz)')
        axes[plot_idx].set_title(f'Testing: Std Dev CFO (Remaining {1-TRAIN_FRACTION:.0%})')
        axes[plot_idx].grid(axis='y', alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"[✓] Saved feature distribution: {outfile}")
    plt.close()


def plot_confusion_matrix(results, class_names, outfile):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm = results['confusion_matrix']
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    ax.set_title(f'Random Forest Confusion Matrix\nAccuracy: {results["accuracy"]:.1%}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Device', fontsize=12)
    ax.set_xlabel('Predicted Device', fontsize=12)
    
    # Rotate labels for readability
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"[✓] Saved confusion matrix: {outfile}")
    plt.close()


def write_results_report(results, class_names, y_test, X_train, X_test, y_train, 
                        feature_names, use_mean, use_std, outfile):
    """Write detailed results to text file."""
    with open(outfile, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BLE DEVICE FINGERPRINTING - RANDOM FOREST CLASSIFICATION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Features: {', '.join(feature_names)} of CFO (Hz)\n")
        f.write(f"Training: Computed from first {TRAIN_FRACTION:.0%} of packets per device\n")
        f.write(f"Testing: Computed from remaining {1-TRAIN_FRACTION:.0%} of packets per device\n")
        f.write(f"Number of devices: {len(class_names)}\n")
        f.write(f"Training samples: {len(y_train)} (one per device)\n")
        f.write(f"Testing samples: {len(y_test)} (one per device)\n\n")
        
        # Show feature values
        f.write("-"*80 + "\n")
        f.write("FEATURE VALUES (Hz)\n")
        f.write("-"*80 + "\n")
        
        # Build header
        header = f"{'Device':<15} "
        if use_mean:
            header += f"{'Train Mean':<15} "
        if use_std:
            header += f"{'Train Std':<15} "
        if use_mean:
            header += f"{'Test Mean':<15} "
        if use_std:
            header += f"{'Test Std':<15}"
        f.write(header + "\n")
        f.write("-"*80 + "\n")
        
        sorted_indices = np.argsort(y_train)
        for i in sorted_indices:
            row = f"{y_train[i]:<15} "
            col_idx = 0
            
            if use_mean:
                train_val = X_train[i, col_idx] if X_train.ndim > 1 else X_train[i]
                row += f"{train_val:>15.2f} "
                col_idx += 1
            
            if use_std:
                train_val = X_train[i, col_idx] if X_train.ndim > 1 else X_train[i]
                row += f"{train_val:>15.2f} "
            
            col_idx = 0
            if use_mean:
                test_val = X_test[i, col_idx] if X_test.ndim > 1 else X_test[i]
                row += f"{test_val:>15.2f} "
                col_idx += 1
            
            if use_std:
                test_val = X_test[i, col_idx] if X_test.ndim > 1 else X_test[i]
                row += f"{test_val:>15.2f}"
            
            f.write(row + "\n")
        f.write("\n")
        
        # Feature importances (if applicable)
        if results['feature_importances'] is not None:
            f.write("-"*80 + "\n")
            f.write("FEATURE IMPORTANCES\n")
            f.write("-"*80 + "\n")
            for feat, imp in results['feature_importances'].items():
                f.write(f"{feat:<20}: {imp:.4f}\n")
            f.write("\n")
        
        # Performance metrics
        f.write("-"*80 + "\n")
        f.write("RANDOM FOREST PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.2%}\n")
        f.write(f"Precision: {results['precision']:.2%}\n")
        f.write(f"Recall:    {results['recall']:.2%}\n")
        f.write(f"F1-Score:  {results['f1']:.2%}\n\n")
        
        # Per-device classification report
        f.write("Per-Device Classification Report:\n")
        f.write(classification_report(
            y_test, results['y_pred'],
            labels=class_names,
            target_names=class_names,
            zero_division=0
        ))
        f.write("\n")
        
        # Confusion matrix (raw counts)
        f.write("-"*80 + "\n")
        f.write("CONFUSION MATRIX (Raw Counts)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'':>15} ")
        for label in class_names:
            f.write(f"{label:<10} ")
        f.write("\n")
        
        cm = results['confusion_matrix']
        for i, true_label in enumerate(class_names):
            f.write(f"{true_label:>15} ")
            for j in range(len(class_names)):
                f.write(f"{cm[i, j]:<10} ")
            f.write("\n")
    
    print(f"[✓] Saved detailed results: {outfile}")


# --------------------------- main ---------------------------

def main():
    print("="*80)
    print("BLE DEVICE FINGERPRINTING using CFO Statistics")
    print("="*80)
    
    # Get feature selection from user
    use_mean, use_std = get_feature_choice()
    
    # Load all data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test, class_names, feature_names = collect_all_data(use_mean, use_std)
    
    if len(class_names) < 2:
        print("ERROR: Need at least 2 devices for classification!")
        sys.exit(1)
    
    # Show statistics
    print(f"\nFeature Statistics (Hz):")
    if use_mean:
        col_idx = 0
        mean_train = X_train[:, col_idx] if X_train.ndim > 1 else X_train
        mean_test = X_test[:, col_idx] if X_test.ndim > 1 else X_test
        print(f"  Training - Mean CFO range: [{mean_train.min():.2f}, {mean_train.max():.2f}]")
        print(f"  Testing  - Mean CFO range: [{mean_test.min():.2f}, {mean_test.max():.2f}]")
    
    if use_std:
        col_idx = 1 if (use_mean and X_train.ndim > 1) else 0
        std_train = X_train[:, col_idx] if X_train.ndim > 1 else X_train
        std_test = X_test[:, col_idx] if X_test.ndim > 1 else X_test
        print(f"  Training - Std CFO range:  [{std_train.min():.2f}, {std_train.max():.2f}]")
        print(f"  Testing  - Std CFO range:  [{std_test.min():.2f}, {std_test.max():.2f}]")
    
    # Train and evaluate
    print("\nTraining classifier...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test, class_names, feature_names)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1-Score:  {results['f1']:.2%}")
    
    if results['feature_importances'] is not None:
        print("\nFeature Importances:")
        for feat, imp in results['feature_importances'].items():
            print(f"  {feat}: {imp:.4f}")
    
    # Generate outputs
    print("\nGenerating visualizations...")
    plot_feature_distribution(X_train, X_test, y_train, class_names, feature_names, 
                             use_mean, use_std, OUT_DISTRIBUTION)
    plot_confusion_matrix(results, class_names, OUT_CONFUSION)
    write_results_report(results, class_names, y_test, X_train, X_test, y_train,
                        feature_names, use_mean, use_std, OUT_RESULTS)
    
    print("\n" + "="*80)
    print("DONE! Check output files for detailed results.")
    print("="*80)


if __name__ == "__main__":
    main()