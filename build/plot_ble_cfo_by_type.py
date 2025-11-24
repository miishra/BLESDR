#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make two consolidated plots from BLE fingerprint CSVs:
1) Grouped boxplot of est_cfo (kHz) with four device types on X axis and 4 sub-devices each.
2) Grouped "PDF" plot (violin density) with the same layout.

CSV file naming expected:
  ble_packets_fingerprints_with_headers_apple1.csv ... apple4.csv
  ble_packets_fingerprints_with_headers_mi1.csv     ... mi4.csv
  ble_packets_fingerprints_with_headers_hello1.csv  ... hello4.csv
  ble_packets_fingerprints_with_headers_smart1.csv  ... smart4.csv

Column detection:
- Prefers 'est_cfo_Hz' (case-insensitive).
- Otherwise falls back to the first column whose name contains both 'est' and 'cfo'.

Outputs (consolidated):
  ble_packets_est_cfo_boxplot_kHz_all_types.png
  ble_packets_est_cfo_pdf_kHz_all_types.png

Legacy-compatible duplicates (same figure content):
  ble_packets_est_cfo_boxplot_kHz_{apple,mi,hello,smart}{1..4}.png
  ble_packets_est_cfo_pdf_kHz_{apple,mi,hello,smart}{1..4}.png
"""

import os
import sys
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- config ---------------------------

# device groups and pretty labels (x-axis)
DEVICE_GROUPS = {
    "mi":     "MiTags",
    "apple":  "AirTags",
    "hello":  "HelloTags",
    "smart":  "SmartTags",
}

# sub-device identifiers (the “1..4” in filenames)
SUB_IDS = [1, 2, 3, 4]

# filename template
FNAME_TPL = "{dtype}{idx}.csv"

# output files (consolidated)
OUT_BOX = "ble_packets_est_cfo_boxplot_kHz_all_types.png"
OUT_PDF = "ble_packets_est_cfo_pdf_kHz_all_types.png"

# legacy duplicates to keep downstream stable (same plot, different names)
LEGACY_BOX_TPL = "ble_packets_est_cfo_boxplot_kHz_{dtype}{idx}.png"
LEGACY_PDF_TPL = "ble_packets_est_cfo_pdf_kHz_{dtype}{idx}.png"

# random seed for any jitter we might add (currently not used)
np.random.seed(7)


# --------------------------- helpers ---------------------------

def find_cfo_column(df: pd.DataFrame) -> str:
    """Find the CFO column in Hz. Prefer 'est_cfo_Hz' (case-insensitive).
    Otherwise any column containing both 'est' and 'cfo' (case-insensitive).
    Raises if not found.
    """
    cols = [c for c in df.columns]
    lower = [c.lower() for c in cols]

    # exact preferred name
    for c, lc in zip(cols, lower):
        if lc == "cfo_quick_hz" or lc == "est_cfo_hz":
            return c

    # heuristic: contains both 'est' and 'cfo'
    for c, lc in zip(cols, lower):
        if ("est" in lc) and ("cfo" in lc):
            return c

    raise ValueError("No CFO column found (looked for 'est_cfo_Hz' or any column containing both 'est' and 'cfo').")


def load_cfo_khz(filename: str) -> np.ndarray:
    """Load a CSV and return est_cfo in kHz (nan-filtered)."""
    if not os.path.exists(filename):
        warnings.warn(f"Missing file: {filename}")
        return np.array([])

    df = pd.read_csv(filename)
    col = find_cfo_column(df)
    vals_hz = pd.to_numeric(df[col], errors="coerce").values
    vals_khz = vals_hz / 1e3
    vals_khz = vals_khz[np.isfinite(vals_khz)]
    return vals_khz


def collect_data() -> Tuple[List[str], Dict[str, List[np.ndarray]]]:
    """Return ordered device-type labels and mapping dtype -> list of 4 arrays (one per sub-device)."""
    xlabels = [DEVICE_GROUPS[d] for d in DEVICE_GROUPS.keys()]  # pretty names in fixed order of dict
    data: Dict[str, List[np.ndarray]] = {}

    for dtype in DEVICE_GROUPS.keys():
        series_per_sub: List[np.ndarray] = []
        for idx in SUB_IDS:
            fname = FNAME_TPL.format(dtype=dtype, idx=idx)
            series_per_sub.append(load_cfo_khz(fname))
        data[dtype] = series_per_sub
    return xlabels, data


def make_group_positions(n_groups: int, n_sub: int, group_gap: float = 1.0, width: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Compute x positions for grouped plotting.
    Returns: (group_centers, positions_per_sub) where positions_per_sub has shape (n_sub, n_groups).
    """
    centers = np.arange(n_groups) * (n_sub * width + group_gap)
    # symmetric offsets like (-1.5w, -0.5w, +0.5w, +1.5w) for n_sub=4
    offsets = (np.arange(n_sub) - (n_sub - 1) / 2.0) * (width * 1.2)
    # broadcast to per-group positions
    positions = centers[None, :] + offsets[:, None]
    return centers, positions


# --------------------------- plotting ---------------------------

def plot_grouped_boxplot(xlabels: List[str], data: Dict[str, List[np.ndarray]], outfile: str):
    """Grouped boxplot: four device types on X; each with 4 sub-devices side-by-side."""
    n_groups = len(xlabels)
    n_sub = len(SUB_IDS)
    width = 0.18
    centers, positions = make_group_positions(n_groups, n_sub, group_gap=1.0, width=width)

    fig, ax = plt.subplots(figsize=(12, 5))
    dtype_keys = list(DEVICE_GROUPS.keys())

    # colors just to differentiate sub-devices (optional)
    # If you prefer the default matplotlib colors, remove 'patch_artist' and 'facecolor' bits.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sub_colors = [colors[i % len(colors)] for i in range(n_sub)]

    legend_handles = []
    for s_idx in range(n_sub):
        # collect s_idx-th series from each group in order
        group_series = [data[dtype_keys[g]][s_idx] for g in range(n_groups)]
        # filter out empty to avoid boxplot errors – but keep position alignment
        # use masked NaNs for empties
        to_plot = []
        for arr in group_series:
            if arr.size == 0:
                to_plot.append(np.array([np.nan]))  # will produce empty box
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
        # Keep one proxy for legend
        if s_idx == 0:
            legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                         markerfacecolor=sub_colors[j], alpha=0.6,
                                         markeredgecolor='k', label=f"Device {SUB_IDS[j]}")
                              for j in range(n_sub)]

    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Estimated CFO (kHz)")
    # ax.set_title("Grouped Boxplot of est_cfo across device types and sub-devices")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(handles=legend_handles, ncols=4, loc='upper right', frameon=True)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"[✓] Saved {outfile}")
    return fig


def plot_grouped_violins(xlabels: List[str], data: Dict[str, List[np.ndarray]], outfile: str):
    """Grouped violin plot = probability density visualization (your PDF plot)."""
    n_groups = len(xlabels)
    n_sub = len(SUB_IDS)
    width = 0.18
    centers, positions = make_group_positions(n_groups, n_sub, group_gap=1.0, width=width)

    fig, ax = plt.subplots(figsize=(12, 5))
    dtype_keys = list(DEVICE_GROUPS.keys())

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sub_colors = [colors[i % len(colors)] for i in range(n_sub)]

    legend_handles = []
    for s_idx in range(n_sub):
        # gather s_idx-th series from each group
        group_series = [data[dtype_keys[g]][s_idx] for g in range(n_groups)]
        # Draw one violin per group at the precomputed position.
        for g_idx, arr in enumerate(group_series):
            x_pos = positions[s_idx, g_idx]
            if arr.size == 0 or np.all(~np.isfinite(arr)):
                # skip if no data
                continue
            # Clip to remove long tails (e.g., keep central 98% range)
            if len(arr) > 10:
                low, high = np.percentile(arr, [2.5, 97.5])
                arr = arr[(arr >= low) & (arr <= high)]
            parts = ax.violinplot([arr], positions=[x_pos], showmeans=True, widths=width*1.8)
            # colorize
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

    # legend (sub-device colors)
    legend_handles = [plt.Line2D([0], [0], marker='s', linestyle='',
                                 markerfacecolor=sub_colors[j], alpha=0.6,
                                 markeredgecolor='k', label=f"Device {SUB_IDS[j]}")
                      for j in range(n_sub)]

    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Estimated CFO (kHz)")
    # ax.set_title("Grouped PDF (violin) of est_cfo across device types and sub-devices")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(handles=legend_handles, ncols=4, loc='upper right', frameon=True)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"[✓] Saved {outfile}")
    return fig


def write_legacy_copies(fig_box, fig_pdf):
    """Save the same consolidated figures using legacy per-type/per-sub filenames."""
    # Boxplot duplicates
    for dtype in DEVICE_GROUPS.keys():
        for idx in SUB_IDS:
            out_name = LEGACY_BOX_TPL.format(dtype=dtype, idx=idx)
            fig_box.savefig(out_name, dpi=200)
    print("[✓] Wrote legacy boxplot duplicates.")

    # PDF (violin) duplicates
    for dtype in DEVICE_GROUPS.keys():
        for idx in SUB_IDS:
            out_name = LEGACY_PDF_TPL.format(dtype=dtype, idx=idx)
            fig_pdf.savefig(out_name, dpi=200)
    print("[✓] Wrote legacy PDF duplicates.")


# --------------------------- main ---------------------------

def main():
    xlabels, data = collect_data()

    # quick sanity: count available samples
    total = sum(arr.size for d in data.values() for arr in d)
    if total == 0:
        print("WARNING: No input CSVs found or CFO columns missing. "
              "Expected patterns like 'ble_packets_fingerprints_with_headers_apple1.csv'.")
        sys.exit(0)

    fig_box = plot_grouped_boxplot(xlabels, data, OUT_BOX)
    fig_pdf = plot_grouped_violins(xlabels, data, OUT_PDF)

    # # keep the old names alive (same content)
    # write_legacy_copies(fig_box, fig_pdf)

    plt.close(fig_box)
    plt.close(fig_pdf)


if __name__ == "__main__":
    main()