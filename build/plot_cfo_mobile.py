#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make box plots and violin plots for ALL numeric features from a BLE fingerprint CSV,
grouped by MAC address (adv_addr) as x-axis labels.

Input:
  - One consolidated CSV: mobile_office_all.csv (configurable)

Outputs (per feature):
  ble_packets_{feature_name}_boxplot_by_mac.png
  ble_packets_{feature_name}_violin_by_mac.png

Also writes:
  feature_statistics_summary.txt
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- config ---------------------------

FNAME = "test1.csv"
OUTPUT_DIR = "static_all" #"mobile_office_all_plots2"

# Columns to skip (non-numeric or identifiers)
SKIP_FEATURES = {
    "pkt_idx", "pcap_ts", "adv_addr", "access_address",
    "rf_channel", "pdu_type", "sample_start", "sample_end"
}

# Column that contains MAC addresses
MAC_COL = "adv_addr"

# Keep plots readable: only plot top-K MAC addresses by number of samples
TOP_K_MACS = None  # set None to plot all, but it can get messy

# Ignore MACs with fewer than this many samples for a feature
MIN_SAMPLES_PER_MAC = 5

np.random.seed(7)

# --------------------------- helpers ---------------------------

def get_numeric_features(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        if col in SKIP_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def get_ylabel(feature: str) -> str:
    if "hz" in feature.lower() or "cfo" in feature.lower():
        return f"{feature} (kHz)"
    elif "us" in feature.lower() or "time" in feature.lower():
        return f"{feature} (μs)"
    elif "db" in feature.lower():
        return f"{feature} (dB)"
    elif "deg" in feature.lower():
        return f"{feature} (degrees)"
    else:
        return feature


def load_df(filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV not found: {filename}")
    df = pd.read_csv(filename)
    if MAC_COL not in df.columns:
        raise ValueError(f"Expected column '{MAC_COL}' not found in CSV.")
    return df


def collect_feature_data_by_mac(
    df: pd.DataFrame,
    feature: str,
    convert_to_khz: bool = False,
    top_k: Optional[int] = TOP_K_MACS,
    min_samples_per_mac: int = MIN_SAMPLES_PER_MAC
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Returns:
      xlabels: list of MACs (strings)
      series: list of np arrays (values per MAC)
    """
    if feature not in df.columns:
        return [], []

    tmp = df[[MAC_COL, feature]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp = tmp[np.isfinite(tmp[feature].values)]

    # Convert Hz->kHz if requested
    if convert_to_khz and ("hz" in feature.lower() or "cfo" in feature.lower()):
        tmp[feature] = tmp[feature] / 1e3

    # Group by MAC
    grouped = tmp.groupby(MAC_COL)[feature].apply(lambda s: s.values)

    # Filter MACs with too few samples
    grouped = grouped[grouped.apply(lambda arr: len(arr) >= min_samples_per_mac)]

    if grouped.empty:
        return [], []

    # Pick top-K MACs by sample count for readability
    if top_k is not None:
        counts = grouped.apply(len).sort_values(ascending=False)
        keep_macs = counts.head(top_k).index
        grouped = grouped.loc[keep_macs]

    # Sort MACs by count (desc) then lexicographically for stable order
    counts = grouped.apply(len)
    order = sorted(grouped.index.tolist(), key=lambda m: (-counts[m], str(m)))
    xlabels = order
    series = [grouped[m] for m in order]

    return xlabels, series


def compute_feature_statistics_by_mac(series: List[np.ndarray]) -> Optional[dict]:
    all_stds, all_ranges, all_means = [], [], []
    for arr in series:
        if arr.size > 1:
            all_stds.append(np.std(arr))
            all_ranges.append(np.max(arr) - np.min(arr))
            all_means.append(np.mean(arr))
    if not all_stds:
        return None
    return {
        "mean_std": float(np.mean(all_stds)),
        "median_std": float(np.median(all_stds)),
        "mean_range": float(np.mean(all_ranges)),
        "mean_mean": float(np.mean(all_means)),
    }

# --------------------------- plotting ---------------------------

def plot_boxplot_by_mac(xlabels: List[str], series: List[np.ndarray],
                        feature: str, outfile: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(xlabels)), 6))

    to_plot = [arr if arr.size else np.array([np.nan]) for arr in series]

    bp = ax.boxplot(
        to_plot,
        patch_artist=True,
        showfliers=False
    )

    # style
    for patch in bp["boxes"]:
        patch.set_alpha(0.45)

    ax.set_xticks(np.arange(1, len(xlabels) + 1))
    ax.set_xticklabels(xlabels, rotation=60, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{feature} (grouped by MAC)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"[✓] Saved {outfile}")
    plt.close(fig)


def plot_violin_by_mac(xlabels: List[str], series: List[np.ndarray],
                       feature: str, outfile: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(xlabels)), 6))

    positions = np.arange(1, len(xlabels) + 1)

    # Violin needs enough samples; we pre-filtered via MIN_SAMPLES_PER_MAC,
    # but also clip outliers per MAC to avoid insane tails.
    clipped_series = []
    clipped_pos = []
    for pos, arr in zip(positions, series):
        if arr.size < 3:
            continue
        arr2 = arr
        if len(arr2) > 10:
            low, high = np.percentile(arr2, [1, 99])
            arr2 = arr2[(arr2 >= low) & (arr2 <= high)]
        if arr2.size >= 3:
            clipped_series.append(arr2)
            clipped_pos.append(pos)

    if clipped_series:
        parts = ax.violinplot(clipped_series, positions=clipped_pos, showmeans=True, widths=0.9)
        for pc in parts["bodies"]:
            pc.set_edgecolor("black")
            pc.set_alpha(0.35)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.2)
        if "cbars" in parts: parts["cbars"].set_alpha(0.6)
        if "cmins" in parts: parts["cmins"].set_alpha(0.6)
        if "cmaxes" in parts: parts["cmaxes"].set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, rotation=60, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{feature} (grouped by MAC)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"[✓] Saved {outfile}")
    plt.close(fig)

# --------------------------- main ---------------------------

def main():
    print("=" * 80)
    print("BLE FEATURE VISUALIZATION - GROUPED BY MAC ADDRESS")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}/")

    try:
        df = load_df(FNAME)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    features = sorted(get_numeric_features(df))
    if not features:
        print("ERROR: No numeric features detected!")
        sys.exit(1)

    # Report basic MAC info
    mac_counts = df[MAC_COL].value_counts(dropna=True)
    print(f"\nLoaded {len(df)} rows from {FNAME}")
    print(f"Unique MACs: {len(mac_counts)}")
    print(f"Top MACs by packets:\n{mac_counts.head(10).to_string()}\n")

    feature_stats = {}

    print("=" * 80)
    print("Processing features...")
    print("=" * 80)

    for i, feature in enumerate(features, 1):
        print(f"\n[{i}/{len(features)}] Processing: {feature}")

        convert_to_khz = ("hz" in feature.lower() or "cfo" in feature.lower())

        xlabels, series = collect_feature_data_by_mac(
            df, feature, convert_to_khz=convert_to_khz,
            top_k=TOP_K_MACS, min_samples_per_mac=MIN_SAMPLES_PER_MAC
        )

        total_samples = int(sum(arr.size for arr in series))
        if total_samples == 0 or len(series) == 0:
            print(f"  ⚠ Skipping {feature} - no data after filtering")
            continue

        print(f"  → MACs plotted: {len(series)} (TOP_K_MACS={TOP_K_MACS}, MIN_SAMPLES_PER_MAC={MIN_SAMPLES_PER_MAC})")
        print(f"  → Total samples used: {total_samples}")

        stats = compute_feature_statistics_by_mac(series)
        if stats:
            feature_stats[feature] = stats
            print(f"  → Mean std dev (across MAC groups): {stats['mean_std']:.2f}")

        ylabel = get_ylabel(feature)

        safe_feature = feature.replace("/", "_").replace(" ", "_")
        box_out = os.path.join(OUTPUT_DIR, f"ble_packets_{safe_feature}_boxplot_by_mac.png")
        vio_out = os.path.join(OUTPUT_DIR, f"ble_packets_{safe_feature}_violin_by_mac.png")

        plot_boxplot_by_mac(xlabels, series, feature, box_out, ylabel)
        plot_violin_by_mac(xlabels, series, feature, vio_out, ylabel)

    print("\n" + "=" * 80)
    print("FEATURE STATISTICS SUMMARY")
    print("=" * 80)

    if feature_stats:
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]["mean_std"])

        print(f"\n{'Feature':<40} {'Mean Std':<15} {'Mean Range':<15}")
        print("-" * 80)
        for rank, (feat, stats) in enumerate(sorted_features, 1):
            marker = "★" if rank <= 3 else " "
            print(f"{rank:2}. {marker} {feat:<35} {stats['mean_std']:<15.2f} {stats['mean_range']:<15.2f}")

        summary_file = os.path.join(OUTPUT_DIR, "feature_statistics_summary.txt")
        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BLE FEATURE STATISTICS SUMMARY (GROUPED BY MAC)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Input CSV: {FNAME}\n")
            f.write(f"TOP_K_MACS: {TOP_K_MACS}\n")
            f.write(f"MIN_SAMPLES_PER_MAC: {MIN_SAMPLES_PER_MAC}\n\n")
            f.write(f"Total features analyzed: {len(feature_stats)}\n\n")

            f.write("Stability Ranking (by Mean Std Dev across MAC groups):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<40} {'Mean Std':<15} {'Mean Range':<15}\n")
            f.write("-" * 80 + "\n")

            for rank, (feat, stats) in enumerate(sorted_features, 1):
                f.write(f"{rank:<6} {feat:<40} {stats['mean_std']:<15.2f} {stats['mean_range']:<15.2f}\n")

            f.write("\nTop 3 most stable:\n")
            for rank, (feat, stats) in enumerate(sorted_features[:3], 1):
                f.write(f"  {rank}. {feat} (std={stats['mean_std']:.2f}, range={stats['mean_range']:.2f})\n")

        print(f"\n[✓] Saved summary: {summary_file}")

    print("\n" + "=" * 80)
    print(f"COMPLETE! Generated plots for {len(feature_stats)} features in {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()