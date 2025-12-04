#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make consolidated box plots and violin plots for ALL numeric features from BLE fingerprint CSVs.

CSV file naming expected:
  {dtype}{idx}.csv where dtype ∈ {apple, mi, hello, smart} and idx ∈ {1,2,3,4}

Generates plots for each numeric feature:
  - Grouped boxplot (4 device types × 4 sub-devices)
  - Grouped violin plot (PDF)

Outputs:
  ble_packets_{feature_name}_boxplot_all_types.png
  ble_packets_{feature_name}_violin_all_types.png
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- config ---------------------------

DEVICE_GROUPS = {
    "mi":     "MiTags",
    "apple":  "AirTags",
    "hello":  "HelloTags",
    "smart":  "SmartTags",
}

# SUB_IDS = [1, 2, 3, 4]
SUB_IDS = [1]
# FNAME_TPL = "{dtype}{idx}.csv"
# FNAME_TPL = "{dtype}_long.csv"
FNAME_TPL = "mobile_office_all.csv"


# Features to skip (non-numeric or identifiers)
SKIP_FEATURES = {
    'pkt_idx', 'pcap_ts', 'adv_addr', 'access_address', 
    'rf_channel', 'pdu_type', 'sample_start', 'sample_end'
}

# Output directory
OUTPUT_DIR = "mobile_office_all_plots"

np.random.seed(7)


# --------------------------- helpers ---------------------------

def get_numeric_features(df: pd.DataFrame) -> List[str]:
    """Get list of numeric feature columns to plot."""
    numeric_cols = []
    
    for col in df.columns:
        # Skip identifier columns
        if col in SKIP_FEATURES:
            continue
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    return numeric_cols


def find_all_features() -> Set[str]:
    """Scan all CSV files to find all available numeric features."""
    all_features = set()
    
    for dtype in DEVICE_GROUPS.keys():
        for idx in SUB_IDS:
            fname = FNAME_TPL.format(dtype=dtype, idx=idx)
            if os.path.exists(fname):
                df = pd.read_csv(fname, nrows=1)  # Just read header
                features = get_numeric_features(df)
                all_features.update(features)
    
    return all_features


def load_feature_data(filename: str, feature: str, convert_to_khz: bool = False) -> np.ndarray:
    """Load a specific feature from CSV and return values (nan-filtered)."""
    if not os.path.exists(filename):
        return np.array([])
    
    try:
        df = pd.read_csv(filename)
        
        if feature not in df.columns:
            return np.array([])
        
        vals = pd.to_numeric(df[feature], errors="coerce").values
        
        # Convert Hz to kHz if requested
        if convert_to_khz and ('hz' in feature.lower() or 'cfo' in feature.lower()):
            vals = vals / 1e3
        
        vals = vals[np.isfinite(vals)]
        return vals
    except Exception as e:
        warnings.warn(f"Error loading {filename}: {e}")
        return np.array([])


def collect_feature_data(feature: str, convert_to_khz: bool = False) -> Tuple[List[str], Dict[str, List[np.ndarray]]]:
    """Collect data for a specific feature across all devices."""
    xlabels = [DEVICE_GROUPS[d] for d in DEVICE_GROUPS.keys()]
    data: Dict[str, List[np.ndarray]] = {}
    
    for dtype in DEVICE_GROUPS.keys():
        series_per_sub: List[np.ndarray] = []
        for idx in SUB_IDS:
            fname = FNAME_TPL.format(dtype=dtype, idx=idx)
            series_per_sub.append(load_feature_data(fname, feature, convert_to_khz))
        data[dtype] = series_per_sub
    
    return xlabels, data


def make_group_positions(n_groups: int, n_sub: int, group_gap: float = 1.0, 
                         width: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Compute x positions for grouped plotting."""
    centers = np.arange(n_groups) * (n_sub * width + group_gap)
    offsets = (np.arange(n_sub) - (n_sub - 1) / 2.0) * (width * 1.2)
    positions = centers[None, :] + offsets[:, None]
    return centers, positions


# --------------------------- plotting ---------------------------

def plot_grouped_boxplot(xlabels: List[str], data: Dict[str, List[np.ndarray]], 
                         feature: str, outfile: str, ylabel: str):
    """Grouped boxplot for a specific feature."""
    n_groups = len(xlabels)
    n_sub = len(SUB_IDS)
    width = 0.18
    centers, positions = make_group_positions(n_groups, n_sub, group_gap=1.0, width=width)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dtype_keys = list(DEVICE_GROUPS.keys())
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sub_colors = [colors[i % len(colors)] for i in range(n_sub)]
    
    legend_handles = []
    for s_idx in range(n_sub):
        group_series = [data[dtype_keys[g]][s_idx] for g in range(n_groups)]
        
        to_plot = []
        for arr in group_series:
            if arr.size == 0:
                to_plot.append(np.array([np.nan]))
            else:
                to_plot.append(arr)
        
        bp = ax.boxplot(
            to_plot,
            positions=positions[s_idx, :],
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor(sub_colors[s_idx])
            patch.set_alpha(0.45)
        
        if s_idx == 0:
            legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                         markerfacecolor=sub_colors[j], alpha=0.6,
                                         markeredgecolor='k', label=f"Device {SUB_IDS[j]}")
                              for j in range(n_sub)]
    
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(handles=legend_handles, ncols=2, loc='best', frameon=True, fontsize=10)
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"[✓] Saved {outfile}")
    plt.close(fig)


def plot_grouped_violins(xlabels: List[str], data: Dict[str, List[np.ndarray]], 
                         feature: str, outfile: str, ylabel: str):
    """Grouped violin plot for a specific feature."""
    n_groups = len(xlabels)
    n_sub = len(SUB_IDS)
    width = 0.18
    centers, positions = make_group_positions(n_groups, n_sub, group_gap=1.0, width=width)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dtype_keys = list(DEVICE_GROUPS.keys())
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sub_colors = [colors[i % len(colors)] for i in range(n_sub)]
    
    for s_idx in range(n_sub):
        group_series = [data[dtype_keys[g]][s_idx] for g in range(n_groups)]
        
        for g_idx, arr in enumerate(group_series):
            x_pos = positions[s_idx, g_idx]
            
            if arr.size == 0 or np.all(~np.isfinite(arr)):
                continue
            
            # Clip outliers
            if len(arr) > 10:
                low, high = np.percentile(arr, [1, 99])
                arr = arr[(arr >= low) & (arr <= high)]
            
            if len(arr) < 3:  # Need minimum samples for violin
                continue
            
            parts = ax.violinplot([arr], positions=[x_pos], showmeans=True, 
                                 widths=width*1.8)
            
            for pc in parts['bodies']:
                pc.set_facecolor(sub_colors[s_idx])
                pc.set_edgecolor('black')
                pc.set_alpha(0.35)
            
            if 'cbars' in parts: parts['cbars'].set_alpha(0.6)
            if 'cmins' in parts: parts['cmins'].set_alpha(0.6)
            if 'cmaxes' in parts: parts['cmaxes'].set_alpha(0.6)
            if 'cmeans' in parts:
                parts['cmeans'].set_color('black')
                parts['cmeans'].set_linewidth(1.2)
    
    legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                 markerfacecolor=sub_colors[j], alpha=0.6,
                                 markeredgecolor='k', label=f"Device {SUB_IDS[j]}")
                      for j in range(n_sub)]
    
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(handles=legend_handles, ncols=2, loc='best', frameon=True, fontsize=10)
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"[✓] Saved {outfile}")
    plt.close(fig)


def get_ylabel(feature: str) -> str:
    """Generate appropriate y-axis label for a feature."""
    # Convert to kHz for frequency features
    if 'hz' in feature.lower() or 'cfo' in feature.lower():
        return f"{feature} (kHz)"
    elif 'us' in feature.lower() or 'time' in feature.lower():
        return f"{feature} (μs)"
    elif 'db' in feature.lower():
        return f"{feature} (dB)"
    elif 'deg' in feature.lower():
        return f"{feature} (degrees)"
    else:
        return feature


def compute_feature_statistics(feature: str, data: Dict[str, List[np.ndarray]]) -> dict:
    """Compute variance statistics for a feature across all devices."""
    all_stds = []
    all_ranges = []
    all_means = []
    
    for dtype in data.keys():
        for sub_data in data[dtype]:
            if sub_data.size > 1:
                all_stds.append(np.std(sub_data))
                all_ranges.append(np.max(sub_data) - np.min(sub_data))
                all_means.append(np.mean(sub_data))
    
    if len(all_stds) == 0:
        return None
    
    return {
        'mean_std': np.mean(all_stds),
        'median_std': np.median(all_stds),
        'mean_range': np.mean(all_ranges),
        'mean_mean': np.mean(all_means)
    }


# --------------------------- main ---------------------------

def main():
    print("="*80)
    print("BLE FINGERPRINT FEATURE VISUALIZATION - ALL FEATURES")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    
    # Find all available features
    print("\nScanning CSV files for available features...")
    all_features = find_all_features()
    
    if not all_features:
        print("ERROR: No CSV files found or no numeric features detected!")
        sys.exit(1)
    
    # Sort features for consistent ordering
    features_sorted = sorted(all_features)
    
    print(f"\nFound {len(features_sorted)} numeric features:")
    for feat in features_sorted:
        print(f"  - {feat}")
    
    # Compute statistics for all features
    feature_stats = {}
    
    print("\n" + "="*80)
    print("Processing features...")
    print("="*80)
    
    for idx, feature in enumerate(features_sorted, 1):
        print(f"\n[{idx}/{len(features_sorted)}] Processing: {feature}")
        
        # Determine if this is a frequency feature (convert to kHz)
        convert_to_khz = ('hz' in feature.lower() or 'cfo' in feature.lower())
        
        # Collect data for this feature
        xlabels, data = collect_feature_data(feature, convert_to_khz)
        
        # Check if we have any data
        total_samples = sum(arr.size for d in data.values() for arr in d)
        if total_samples == 0:
            print(f"  ⚠ Skipping {feature} - no data")
            continue
        
        print(f"  → {total_samples} total samples")
        
        # Compute statistics
        stats = compute_feature_statistics(feature, data)
        if stats:
            feature_stats[feature] = stats
            print(f"  → Mean std dev: {stats['mean_std']:.2f}")
        
        # Generate y-axis label
        ylabel = get_ylabel(feature)
        
        # Generate plots
        box_out = os.path.join(OUTPUT_DIR, f"ble_packets_{feature}_boxplot_all_types.png")
        violin_out = os.path.join(OUTPUT_DIR, f"ble_packets_{feature}_violin_all_types.png")
        
        plot_grouped_boxplot(xlabels, data, feature, box_out, ylabel)
        plot_grouped_violins(xlabels, data, feature, violin_out, ylabel)
    
    # Generate summary report
    print("\n" + "="*80)
    print("FEATURE STATISTICS SUMMARY")
    print("="*80)
    
    if feature_stats:
        # Sort by stability (lowest std dev)
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['mean_std'])
        
        print(f"\n{'Feature':<40} {'Mean Std':<15} {'Mean Range':<15}")
        print("-"*80)
        
        for rank, (feat, stats) in enumerate(sorted_features, 1):
            marker = "★" if rank <= 3 else " "
            print(f"{rank:2}. {marker} {feat:<35} {stats['mean_std']:<15.2f} {stats['mean_range']:<15.2f}")
        
        print("\n" + "="*80)
        print("TOP 3 MOST STABLE FEATURES (for fingerprinting):")
        print("="*80)
        
        for rank, (feat, stats) in enumerate(sorted_features[:3], 1):
            print(f"{rank}. {feat}")
            print(f"   Mean Std Dev: {stats['mean_std']:.2f}")
            print(f"   Mean Range:   {stats['mean_range']:.2f}")
        
        # Save summary to file
        summary_file = os.path.join(OUTPUT_DIR, "feature_statistics_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BLE FINGERPRINT FEATURE STATISTICS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total features analyzed: {len(feature_stats)}\n\n")
            
            f.write("Stability Ranking (by Mean Std Dev):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<40} {'Mean Std':<15} {'Mean Range':<15}\n")
            f.write("-"*80 + "\n")
            
            for rank, (feat, stats) in enumerate(sorted_features, 1):
                f.write(f"{rank:<6} {feat:<40} {stats['mean_std']:<15.2f} {stats['mean_range']:<15.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATION:\n")
            f.write("="*80 + "\n")
            f.write(f"\nTop 3 most stable features for fingerprinting:\n")
            for rank, (feat, stats) in enumerate(sorted_features[:3], 1):
                f.write(f"  {rank}. {feat} (std={stats['mean_std']:.2f})\n")
        
        print(f"\n[✓] Saved summary: {summary_file}")
    
    print("\n" + "="*80)
    print(f"COMPLETE! Generated plots for {len(feature_stats)} features in {OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()