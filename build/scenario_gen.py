#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scenario generator for SDR-capture CSVs (UPDATED: adversary candidates are UNIQUE TAGS, not rows)

What changed (per your latest message)
--------------------------------------
- The "adversary candidate" identity is the ADV payload TAG itself.
  i.e., if you only use ["4c001219ff"] today, that is ONE adversary candidate,
  even if it matches 739 rows.

- You now pass a LIST of adversary tags via CLI:
    --adv-tags "4c001219ff,deadbeef..."

- Baseline removal:
    remove ALL rows whose payload contains ANY tag in --adv-tags

- Selection:
    adversaries are selected by TAG (not by MAC, not by derived adv_id).

- Behavior injection:
    adversary rows are grouped by TAG for tx/rotation changes.
    (So each tag acts like a “device” identity; when you add more tags later,
     you can select multiple adversary “devices”.)

Samsung remains:
- Candidates are persistent PRIVIDs (devices), and selection is by PRIVID.

APPLE/GOOGLE/TILE remain:
- Candidates are persistent MACs (>= 30 min), and selection is by MAC.

"""

from __future__ import annotations

import argparse
import os
import sys
import time
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Scenario parameter grids
# ---------------------------

TX_PERIODS_SEC = [2, 10, 15, 30, 60]  # 2s, 10s, 15s, 30s, 1m
ROT_PERIODS_SEC = [2, 10, 30, 60, 300, 900]     # 2s, 10s, 30s, 1m, 5m, 15m
PERSIST_SEC_DEFAULT = 30 * 60                   # 30 minutes


# ---------------------------
# Column detection / canonicalization
# ---------------------------

TAG_CANON = {"apple": "APPLE", "google": "GOOGLE", "samsung": "SAMSUNG", "tile": "TILE", "unknown": "UNKNOWN"}
COMMON_TAG_COLS = ["tag_type", "tag", "eco", "ecosystem", "brand", "vendor", "type"]
MAC_COL_CANDIDATES = ["AdvA", "adv_addr", "mac", "device_mac", "addr"]

_HEX_DIGITS = set("0123456789abcdefABCDEF")


def _norm_mac(x: str) -> str:
    return str(x).strip().lower()


def _pretty_seconds(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s % 60 == 0:
        m = s // 60
        return f"{m}min" if m < 60 else f"{m//60}h{m%60:02d}m"
    return f"{s}s"


def detect_tag_col(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit if explicit in df.columns else None
    for c in COMMON_TAG_COLS:
        if c in df.columns:
            return c
    return None


def canon_tag_value(v: str) -> str:
    vv = str(v).strip().lower()
    return TAG_CANON.get(vv, str(v).strip().upper())


def parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [p.strip().lower() for p in s.split(",") if p.strip()]


# ---------------------------
# Hex helpers + Samsung PRIVID extraction
# ---------------------------

def _clean_hexpayload(payload_hex: str) -> str:
    if payload_hex is None:
        return ""
    s = str(payload_hex).strip()
    if not s:
        return ""
    s = s.replace(" ", "").replace(":", "").replace("-", "").replace("\n", "").replace("\r", "").replace("\t", "")
    s = "".join(ch for ch in s if ch in _HEX_DIGITS)
    if len(s) < 2:
        return ""
    if len(s) % 2 == 1:
        s = s[:-1]
    return s.lower()


def _hex_to_bytes(hx: str) -> bytes:
    try:
        return bytes.fromhex(hx)
    except Exception:
        return b""


def get_samsung_privid_from_payload(payload_hex: str) -> str:
    """
    Extract Samsung PRIVID from payload if present.
    PRIVID lives inside a Service Data (AD type 0x16) with UUID=0xFD5A.
    PRIVID is 8 bytes at data[4:12] where data[0:2] is UUID.
    Returns PRIVID hex (lowercase) or "".
    """
    hx = _clean_hexpayload(payload_hex)
    if not hx:
        return ""
    b = _hex_to_bytes(hx)
    if not b:
        return ""

    def extract_from_ad(ad_data: bytes) -> str:
        pos = 0
        n = len(ad_data)
        while pos + 1 < n:
            length = ad_data[pos]
            if length == 0:
                break
            if pos + 1 + length > n:
                break
            ad_type = ad_data[pos + 1]
            data = ad_data[pos + 2: pos + 1 + length]
            if ad_type == 0x16 and len(data) >= 12:
                svc_uuid = data[0] | (data[1] << 8)
                if svc_uuid == 0xFD5A:
                    return data[4:12].hex()
            pos += 1 + length
        return ""

    # Layout A: AdvA(6) + AD...
    if len(b) >= 7:
        v = extract_from_ad(b[6:])
        if v:
            return v
    # Layout B: Header(2) + AdvA(6) + AD...
    if len(b) >= 9:
        v = extract_from_ad(b[8:])
        if v:
            return v
    return ""


# ---------------------------
# Core transforms (tx omit + MAC rotation)
# ---------------------------

def downsample_by_period(df_rows: pd.DataFrame, ts_col: str, tx_period_sec: int) -> pd.DataFrame:
    """Keep at most one row every tx_period_sec (deterministic)."""
    if df_rows.empty:
        return df_rows
    d = df_rows.sort_values(ts_col).copy()
    ts = d[ts_col].astype(float).tolist()
    keep_idx = []
    last_kept = None
    for i, t in enumerate(ts):
        if last_kept is None or (t - last_kept) >= tx_period_sec:
            keep_idx.append(d.index[i])
            last_kept = t
    return d.loc[keep_idx]


def rotate_mac_pseudonyms(
    df_rows: pd.DataFrame,
    ts_col: str,
    mac_col: str,
    rot_period_sec: int,
    pseudonym_pool: List[str],
    rng: random.Random,
) -> pd.DataFrame:
    """
    Replace mac_col using time-windowed pseudonyms:
      window_id = floor((t - t0) / rot_period_sec)
    """
    if df_rows.empty or rot_period_sec <= 0 or not pseudonym_pool:
        return df_rows
    d = df_rows.sort_values(ts_col).copy()
    t0 = float(d[ts_col].iloc[0])
    ts = d[ts_col].astype(float).tolist()

    win_to_pseudo: Dict[int, str] = {}
    pool = pseudonym_pool[:]
    rng.shuffle(pool)
    pool_i = 0

    new_macs = []
    for t in ts:
        win = int((float(t) - t0) // rot_period_sec)
        if win not in win_to_pseudo:
            win_to_pseudo[win] = pool[pool_i % len(pool)]
            pool_i += 1
        new_macs.append(win_to_pseudo[win])

    d[mac_col] = new_macs
    return d


def apply_behavior_grouped(
    df_rows: pd.DataFrame,
    group_key: str,
    ts_col: str,
    mac_col: str,
    tx_period_sec: int,
    rot_period_sec: int,
    pseudonym_pool: List[str],
    rng: random.Random,
    also_replace_cols: List[str],
) -> pd.DataFrame:
    """
    Apply tx+rotation per group_key (device identity).
    Within each group, we downsample timestamps and rotate MAC values.
    """
    if df_rows.empty:
        return df_rows

    parts = []
    for gid, g in df_rows.groupby(group_key, sort=True):
        d = g.copy()
        d = downsample_by_period(d, ts_col=ts_col, tx_period_sec=tx_period_sec)
        d = rotate_mac_pseudonyms(
            d, ts_col=ts_col, mac_col=mac_col, rot_period_sec=rot_period_sec,
            pseudonym_pool=pseudonym_pool, rng=rng
        )
        cols_to_fix = [c for c in also_replace_cols if c in d.columns and c != mac_col]
        for c in cols_to_fix:
            d[c] = d[mac_col].astype(str)
        parts.append(d)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values(ts_col).reset_index(drop=True)
    return out


# ---------------------------
# Removal rules (baseline background)
# ---------------------------

def compute_persistent_macs_non_samsung(
    df: pd.DataFrame, ts_col: str, mac_col: str, tag_col: str, tag_type: str, persist_sec: int
) -> Set[str]:
    d = df[df[tag_col] == tag_type].copy()
    if d.empty:
        return set()
    g = d.groupby(mac_col)[ts_col].agg(["min", "max"])
    g["dur"] = g["max"].astype(float) - g["min"].astype(float)
    return set(g.index[g["dur"] >= persist_sec].astype(str).map(_norm_mac).tolist())


def compute_persistent_samsung_privids(
    df: pd.DataFrame, ts_col: str, payload_col: str, tag_col: str, persist_sec: int
) -> Set[str]:
    d = df[df[tag_col] == "SAMSUNG"].copy()
    if d.empty:
        return set()
    d["_privid"] = d[payload_col].astype(str).map(get_samsung_privid_from_payload)
    d = d[d["_privid"] != ""]
    if d.empty:
        return set()
    g = d.groupby("_privid")[ts_col].agg(["min", "max"])
    g["dur"] = g["max"].astype(float) - g["min"].astype(float)
    return set(g.index[g["dur"] >= persist_sec].astype(str).tolist())


def build_adv_mask_from_tags(df: pd.DataFrame, payload_col: str, adv_tags: List[str]) -> pd.Series:
    """
    adv_mask is True for rows whose payload contains ANY tag in adv_tags.
    """
    payload = df[payload_col].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for t in adv_tags:
        t = str(t).strip().lower()
        if not t:
            continue
        mask |= payload.str.contains(t, na=False)
    return mask


def build_baseline_background(
    df_full: pd.DataFrame,
    ts_col: str,
    mac_col: str,
    payload_col: str,
    tag_col: Optional[str],
    adv_tags: List[str],
    persist_sec: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Set[str]], Set[str]]:
    """
    Returns:
      df_base: background after removing:
          (1) adversary rows by payload tags (ANY of adv_tags)
          (2) persistent known tags by your rules
      adv_mask: boolean Series for adversary rows
      persistent_macs_by_type: {"APPLE","GOOGLE","TILE"} -> persistent MACs
      persistent_samsung_privids: persistent PRIVIDs (device IDs)
    """
    adv_mask = build_adv_mask_from_tags(df_full, payload_col, adv_tags)

    persistent_macs_by_type: Dict[str, Set[str]] = {"APPLE": set(), "GOOGLE": set(), "TILE": set()}
    persistent_samsung_privids: Set[str] = set()

    if tag_col:
        for t in ["APPLE", "GOOGLE", "TILE"]:
            persistent_macs_by_type[t] = compute_persistent_macs_non_samsung(
                df_full, ts_col, mac_col, tag_col, t, persist_sec
            )
        persistent_samsung_privids = compute_persistent_samsung_privids(
            df_full, ts_col, payload_col, tag_col, persist_sec
        )

        mask_known = pd.Series(False, index=df_full.index)
        known_macs = set().union(*persistent_macs_by_type.values())
        if known_macs:
            mask_known |= df_full[mac_col].isin(known_macs)

        if persistent_samsung_privids:
            priv = df_full[payload_col].astype(str).map(get_samsung_privid_from_payload)
            mask_known |= ((df_full[tag_col] == "SAMSUNG") & (priv.isin(persistent_samsung_privids)))

        df_base = df_full.loc[~(adv_mask | mask_known)].copy()
    else:
        df_base = df_full.loc[~adv_mask].copy()

    df_base = df_base.sort_values(ts_col).reset_index(drop=True)
    return df_base, adv_mask, persistent_macs_by_type, persistent_samsung_privids


# ---------------------------
# Selection model (UPDATED)
# ---------------------------

@dataclass
class Selection:
    # adversaries selected by TAG (device identifier)
    adv_tags: List[str]
    # tags selected by MAC (APPLE/GOOGLE/TILE)
    apple_macs: List[str]
    google_macs: List[str]
    tile_macs: List[str]
    # Samsung selected by PRIVID (device)
    samsung_privids: List[str]

    def selected_tag_macs(self) -> Set[str]:
        return set(map(_norm_mac, self.apple_macs + self.google_macs + self.tile_macs))


# ---------------------------
# Adversary tag helpers
# ---------------------------

def adv_tag_match_counts(df: pd.DataFrame, payload_col: str, adv_tags: List[str]) -> Dict[str, int]:
    """Return cleaned-hex payload match counts for each tag."""
    payload_clean = df[payload_col].astype(str).map(_clean_hexpayload)
    out: Dict[str, int] = {}
    for t in adv_tags:
        tt = str(t).strip().lower().replace(" ", "")
        if not tt:
            continue
        out[tt] = int(payload_clean.str.contains(re.escape(tt), na=False, regex=True).sum())
    return out


def filter_adv_tags_with_matches(df: pd.DataFrame, payload_col: str, adv_tags: List[str]) -> List[str]:
    """Keep only tags that match at least one row (cleaned-hex payload search)."""
    counts = adv_tag_match_counts(df, payload_col, adv_tags)
    # preserve input order
    return [t for t in [str(x).strip().lower().replace(" ", "") for x in adv_tags if str(x).strip()] if counts.get(t, 0) > 0]


# ---------------------------
# Interactive selection
# ---------------------------

def _prompt_int(prompt: str, default: int = 0) -> int:
    s = input(f"{prompt} [default={default}]: ").strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        print("  Invalid integer; using default.")
        return default


def _prompt_list(prompt: str) -> List[str]:
    s = input(prompt).strip()
    if not s:
        return []
    return [p.strip().lower() for p in s.split(",") if p.strip()]


def interactive_select(
    adv_tag_candidates: List[str],
    apple_candidates: List[str],
    google_candidates: List[str],
    tile_candidates: List[str],
    samsung_priv_candidates: List[str],
    rng: random.Random,
) -> Selection:
    print("\n=== Device selection ===")
    print("You can either:")
    print("  A) Provide explicit lists (comma-separated), OR")
    print("  B) Provide counts and let the script randomly choose from candidates.\n")

    adv_tags_exp = _prompt_list("Enter adversary TAGs (comma-separated) or press Enter to choose by count: ")
    apple_exp = _prompt_list("Enter APPLE MACs (comma-separated) or press Enter to choose by count: ")
    google_exp = _prompt_list("Enter GOOGLE MACs (comma-separated) or press Enter to choose by count: ")
    tile_exp = _prompt_list("Enter TILE MACs (comma-separated) or press Enter to choose by count: ")
    samsung_exp = _prompt_list("Enter SAMSUNG PRIVIDs (comma-separated) or press Enter to choose by count: ")

    if any([adv_tags_exp, apple_exp, google_exp, tile_exp, samsung_exp]):
        return Selection(
            adv_tags=adv_tags_exp,
            apple_macs=[_norm_mac(x) for x in apple_exp],
            google_macs=[_norm_mac(x) for x in google_exp],
            tile_macs=[_norm_mac(x) for x in tile_exp],
            samsung_privids=samsung_exp,
        )

    print("\nNo explicit IDs provided. Selecting by count from candidates.")
    n_adv = _prompt_int(f"How many adversary TAG-devices? (candidates={len(adv_tag_candidates)})", default=0)
    n_apple = _prompt_int(f"How many APPLE tags? (candidates={len(apple_candidates)})", default=0)
    n_google = _prompt_int(f"How many GOOGLE tags? (candidates={len(google_candidates)})", default=0)
    n_tile = _prompt_int(f"How many TILE tags? (candidates={len(tile_candidates)})", default=0)
    n_samsung = _prompt_int(f"How many SAMSUNG devices? (persistent PRIVID candidates={len(samsung_priv_candidates)})", default=0)

    def sample_from(lst: List[str], k: int) -> List[str]:
        if k <= 0:
            return []
        if k >= len(lst):
            return lst[:]
        return rng.sample(lst, k)

    return Selection(
        adv_tags=sample_from(adv_tag_candidates, n_adv),
        apple_macs=[_norm_mac(x) for x in sample_from(apple_candidates, n_apple)],
        google_macs=[_norm_mac(x) for x in sample_from(google_candidates, n_google)],
        tile_macs=[_norm_mac(x) for x in sample_from(tile_candidates, n_tile)],
        samsung_privids=sample_from(samsung_priv_candidates, n_samsung),
    )


def classify_tag_ecosystem_from_payload(payload_hex: str) -> str:
    """
    Parse AD structures to classify ecosystem:
      - APPLE: Manufacturer Specific (0xFF), CompanyID=0x004C, prefix bytes 0x12 0x19
      - GOOGLE: Service Data 16-bit (0x16), UUID=0xFEAA
      - TILE:   Service Data 16-bit (0x16), UUID=0xFEED
      - SAMSUNG:Service Data 16-bit (0x16), UUID=0xFD5A
      - UNKNOWN otherwise

    We try two layouts:
      A) payload = AdvA(6) + AD...
      B) payload = Header(2) + AdvA(6) + AD...
    """
    hx = _clean_hexpayload(payload_hex)
    if len(hx) < 16:
        return "UNKNOWN"

    b = _hex_to_bytes(hx)
    if len(b) < 8:
        return "UNKNOWN"

    def parse_ad_structures(ad_data: bytes) -> str:
        pos = 0
        ad_len = len(ad_data)

        while pos + 1 < ad_len:
            length = ad_data[pos]
            if length == 0:
                break
            if pos + 1 + length > ad_len:
                break

            ad_type = ad_data[pos + 1]
            data = ad_data[pos + 2: pos + 1 + length]  # excludes length byte

            # Manufacturer Specific (0xFF): [CompanyID LE (2)][...]
            if ad_type == 0xFF and len(data) >= 4:
                company_id = data[0] | (data[1] << 8)
                if company_id == 0x004C:
                    # FindMy prefix bytes: 0x12 0x19
                    if data[2] == 0x12 and data[3] == 0x19:
                        return "APPLE"

            # Service Data - 16-bit UUID (0x16): [UUID LE (2)][...]
            if ad_type == 0x16 and len(data) >= 2:
                svc_uuid = data[0] | (data[1] << 8)
                if svc_uuid == 0xFEAA:
                    return "GOOGLE"
                if svc_uuid == 0xFEED:
                    return "TILE"
                if svc_uuid == 0xFD5A:
                    return "SAMSUNG"

            pos += 1 + length

        return "UNKNOWN"

    # Layout A: AdvA(6) + AD...
    if len(b) >= 7:
        eco = parse_ad_structures(b[6:])
        if eco != "UNKNOWN":
            return eco

    # Layout B: Header(2) + AdvA(6) + AD...
    if len(b) >= 9:
        eco = parse_ad_structures(b[8:])
        if eco != "UNKNOWN":
            return eco

    return "UNKNOWN"


# ---------------------------
# Post-generation analysis
# ---------------------------

def _parse_tx_rot_from_scenario_filename(name: str) -> tuple[str, str]:
    """Extract tx/rot tokens from scenario filenames like scenario_tx-10s_rot-5min.csv."""
    base = os.path.basename(name)
    m = re.search(r"tx-([^_]+)_rot-([^\.]+)", base)
    if not m:
        return ("unknown", "unknown")
    return (m.group(1), m.group(2))


def _seconds_from_token(tok: str) -> float:
    t = str(tok).strip().lower()
    if t.endswith("min"):
        return float(t[:-3]) * 60.0
    if t.endswith("s"):
        return float(t[:-1])
    if t.endswith("h"):
        return float(t[:-1]) * 3600.0
    # fallback
    return float(re.sub(r"[^0-9.]+", "", t) or 0.0)


def plot_effective_tx_rot_boxplots(
    outdir: str,
    ts_col: str,
    mac_col: str,
    payload_col: str,
    sel: "Selection",
    max_samples_per_scenario: int = 5000,
) -> Optional[str]:
    """Create box plots of effective TX period and effective ROT period across scenarios.

    - effective TX period: inter-arrival times (seconds) between consecutive injected packets
      (computed per injected-identity group; pooled per scenario).

    - effective ROT period: time between MAC changes (seconds) in the injected stream
      (computed by detecting when mac_col changes; pooled per scenario).

    Results are grouped by configured tx-<tok> and rot-<tok> for comparison.
    """
    outp = os.path.join(outdir, "scenario_effective_tx_rot_boxplots.png")
    scen_files = sorted([p for p in os.listdir(outdir) if p.endswith(".csv") and p.startswith("scenario_")])
    if not scen_files:
        return None

    sel_adv_tags = [t.lower() for t in (sel.adv_tags or [])]
    sel_tag_macs = set(map(_norm_mac, sel.selected_tag_macs())) if hasattr(sel, "selected_tag_macs") else set()
    sel_samsung_priv = set([x.lower() for x in (sel.samsung_privids or [])])

    def adv_mask_for_df(d: pd.DataFrame) -> pd.Series:
        m = pd.Series(False, index=d.index)
        if sel_adv_tags:
            payload = d[payload_col].astype(str).str.lower()
            for t in sel_adv_tags:
                if t:
                    m |= payload.str.contains(t, na=False)
        if sel_tag_macs:
            m |= d[mac_col].astype(str).map(_norm_mac).isin(sel_tag_macs)
        if sel_samsung_priv:
            priv = d[payload_col].astype(str).map(get_samsung_privid_from_payload)
            m |= priv.astype(str).str.lower().isin(sel_samsung_priv)
        return m

    # Per grouping bucket
    tx_groups: Dict[str, List[float]] = {}
    rot_groups: Dict[str, List[float]] = {}

    for fn in scen_files:
        p = os.path.join(outdir, fn)
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if ts_col not in d.columns or payload_col not in d.columns or mac_col not in d.columns:
            continue

        d = d.dropna(subset=[ts_col]).copy()
        d[ts_col] = pd.to_numeric(d[ts_col], errors="coerce")
        d = d[np.isfinite(d[ts_col])].copy()
        if d.empty:
            continue

        da = d.loc[adv_mask_for_df(d)].copy()
        if da.empty:
            continue

        da = da.sort_values(ts_col).reset_index(drop=True)

        tx_tok, rot_tok = _parse_tx_rot_from_scenario_filename(fn)

        # Effective TX: pooled inter-arrivals within each injected identity group
        # Use (payload-tag match OR privid OR mac) as an identity group key.
        # For simplicity, approximate by mac_col after rotation (works for most cases).
        eff_tx: List[float] = []
        for _, g in da.groupby(mac_col, sort=False):
            tt = g[ts_col].astype(float).values
            if len(tt) < 2:
                continue
            dt = np.diff(tt)
            dt = dt[np.isfinite(dt) & (dt >= 0)]
            if len(dt):
                eff_tx.extend(dt.tolist())

        # Effective ROT: time between MAC changes in the injected stream
        eff_rot: List[float] = []
        macs = da[mac_col].astype(str).map(_norm_mac).values
        ts = da[ts_col].astype(float).values
        if len(ts) >= 2:
            last_change_t = float(ts[0])
            last_mac = macs[0]
            for i in range(1, len(ts)):
                if macs[i] != last_mac:
                    eff_rot.append(float(ts[i]) - last_change_t)
                    last_change_t = float(ts[i])
                    last_mac = macs[i]

        # subsample to keep plots light
        if max_samples_per_scenario > 0:
            if len(eff_tx) > max_samples_per_scenario:
                eff_tx = eff_tx[:max_samples_per_scenario]
            if len(eff_rot) > max_samples_per_scenario:
                eff_rot = eff_rot[:max_samples_per_scenario]

        tx_groups.setdefault(str(tx_tok), []).extend(eff_tx)
        rot_groups.setdefault(str(rot_tok), []).extend(eff_rot)

    if not tx_groups and not rot_groups:
        return None

    # Plot
    plt.figure(figsize=(14, 8))

    # TX boxplot
    ax1 = plt.subplot(2, 1, 1)
    tx_keys = sorted(tx_groups.keys(), key=lambda k: _seconds_from_token(k) if k != "unknown" else 1e18)
    tx_data = [tx_groups[k] for k in tx_keys]
    ax1.boxplot(tx_data, labels=tx_keys, showfliers=False)
    ax1.set_title("Effective transmission period (inter-arrival times) by configured tx")
    ax1.set_ylabel("seconds")
    ax1.grid(True, alpha=0.3)

    # ROT boxplot
    ax2 = plt.subplot(2, 1, 2)
    rot_keys = sorted(rot_groups.keys(), key=lambda k: _seconds_from_token(k) if k != "unknown" else 1e18)
    rot_data = [rot_groups[k] for k in rot_keys]
    ax2.boxplot(rot_data, labels=rot_keys, showfliers=False)
    ax2.set_title("Effective rotation period (time between MAC changes) by configured rot")
    ax2.set_ylabel("seconds")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()

    return outp


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True, help="Path to input SDR capture CSV.")
    ap.add_argument("--outdir", default="controlled/SDR_Adv/", help="Output folder (default: auto-named).")

    ap.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")
    ap.add_argument("--mac-col", default="AdvA", help="MAC column name.")
    ap.add_argument("--payload-col", default="payload", help="Payload column name (hex string).")

    # THIS is the key change:
    ap.add_argument(
        "--adv-tags",
        default="4c001219ff, 4c001219fc, 4c001219fd, 4c001219fe",
        help='Comma-separated list of adversary payload tags (device identifiers). '
             'Example: "4c001219ff,abcd1234". Default: "4c001219ff".'
    )

    ap.add_argument("--persist-minutes", type=int, default=30, help="Persistence threshold in minutes (default 30).")
    ap.add_argument("--tag-col", default=None,
                    help="Column holding tag type labels (APPLE/GOOGLE/SAMSUNG/TILE). "
                         "If omitted, script tries common names.")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed for repeatability.")

    # Optional explicit selection
    ap.add_argument("--select-adv-tags", default=None, help="Comma-separated adversary TAGs to re-introduce.")
    ap.add_argument("--select-apple", default=None, help="Comma-separated APPLE MACs.")
    ap.add_argument("--select-google", default=None, help="Comma-separated GOOGLE MACs.")
    ap.add_argument("--select-tile", default=None, help="Comma-separated TILE MACs.")
    ap.add_argument("--select-samsung-privids", default=None, help="Comma-separated SAMSUNG PRIVIDs.")

    ap.add_argument("--no-crc-filter", action="store_true",
                    help="If set, do not filter crc_ok==1 even if present.")
    ap.add_argument("--mirror-mac-cols", default="adv_addr,mac,device_mac",
                    help="Comma-separated list of other columns to overwrite with rotated MACs when present.")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    df_full = pd.read_csv(args.input).copy()

    ts_col = args.timestamp_col
    mac_col = args.mac_col
    payload_col = args.payload_col

    for c in [ts_col, mac_col, payload_col]:
        if c not in df_full.columns:
            print(f"ERROR: required column '{c}' not found.", file=sys.stderr)
            print(f"Columns: {list(df_full.columns)}", file=sys.stderr)
            return 2

    # Normalize + CRC filter
    df_full[payload_col] = df_full[payload_col].astype(str).str.lower()
    df_full[mac_col] = df_full[mac_col].astype(str).map(_norm_mac)

    if (not args.no_crc_filter) and ("crc_ok" in df_full.columns):
        df_full = df_full[df_full["crc_ok"] == 1].copy()

    df_full = df_full.dropna(subset=[ts_col, mac_col]).copy()
    df_full = df_full.sort_values(ts_col).reset_index(drop=True)

    # adv tags list (device identifiers)
    adv_tags = parse_csv_list(args.adv_tags)
    if not adv_tags:
        print("ERROR: --adv-tags list is empty; provide at least one tag.", file=sys.stderr)
        return 2

    tag_col = detect_tag_col(df_full, args.tag_col)
    if tag_col:
        df_full[tag_col] = df_full[tag_col].astype(str).map(canon_tag_value)
        print(f"[i] Using tag column: {tag_col}")
    else:
        # --- NEW: compute tag type from payload if column missing ---
        tag_col = "eco"
        df_full[tag_col] = df_full[payload_col].astype(str).apply(classify_tag_ecosystem_from_payload)
        df_full[tag_col] = df_full[tag_col].astype(str).map(canon_tag_value)
        print(f"[i] Tag column missing. Derived '{tag_col}' from payload.")

        # optional debug
        print("[i] Derived tag distribution:")
        print(df_full[tag_col].value_counts(dropna=False).to_string())

    persist_sec = int(args.persist_minutes) * 60

    # 1) Baseline background
    df_base, adv_mask, persistent_macs_by_type, persistent_samsung_privids = build_baseline_background(
        df_full=df_full,
        ts_col=ts_col,
        mac_col=mac_col,
        payload_col=payload_col,
        tag_col=tag_col,
        adv_tags=adv_tags,
        persist_sec=persist_sec,
    )

    all_macs_full = set(df_full[mac_col].tolist())
    all_macs_base = set(df_base[mac_col].tolist())

    # Candidate reporting (fixed):
    adv_row_count = int(adv_mask.sum())

    # Filter adversary candidates to tags that actually match this capture
    adv_counts_all = adv_tag_match_counts(df_full, payload_col, adv_tags)
    adv_tag_candidates = [t for t in adv_tags if adv_counts_all.get(str(t).strip().lower().replace(" ", ""), 0) > 0]

    if len(adv_tag_candidates) != len(adv_tags):
        dropped = [t for t in adv_tags if t not in adv_tag_candidates]
        print(f"[w] Dropping adversary tags with 0 matches in this capture: {dropped}")

    samsung_priv_candidates = sorted(list(persistent_samsung_privids))

    apple_candidates = sorted(list(persistent_macs_by_type.get("APPLE", set()))) if tag_col else []
    google_candidates = sorted(list(persistent_macs_by_type.get("GOOGLE", set()))) if tag_col else []
    tile_candidates = sorted(list(persistent_macs_by_type.get("TILE", set()))) if tag_col else []

    print("\n=== Candidate summary ===")
    print(f"Total unique MACs (full capture)                          : {len(all_macs_full)}")
    print(f"Adversary rows matched (ANY adversary tag)                 : {adv_row_count}")
    print(f"Adversary candidates (UNIQUE TAG identifiers)              : {len(adv_tag_candidates)}  -> {adv_tag_candidates}")
    if tag_col:
        print(f"APPLE candidates (persistent MACs >= {args.persist_minutes} min): {len(apple_candidates)}")
        print(f"GOOGLE candidates (persistent MACs >= {args.persist_minutes} min): {len(google_candidates)}")
        print(f"TILE candidates (persistent MACs >= {args.persist_minutes} min)  : {len(tile_candidates)}")
        print(f"SAMSUNG candidates (persistent PRIVIDs >= {args.persist_minutes} min): {len(samsung_priv_candidates)}")
    print(f"Background passers-by unique MACs remaining               : {len(all_macs_base)}")
    print(f"Background rows remaining                                 : {len(df_base)}")

    # 2) Selection (explicit or interactive)
    explicit = any([
        args.select_adv_tags, args.select_apple, args.select_google, args.select_tile, args.select_samsung_privids
    ])

    if explicit:
        sel = Selection(
            adv_tags=parse_csv_list(args.select_adv_tags),
            apple_macs=[_norm_mac(x) for x in parse_csv_list(args.select_apple)],
            google_macs=[_norm_mac(x) for x in parse_csv_list(args.select_google)],
            tile_macs=[_norm_mac(x) for x in parse_csv_list(args.select_tile)],
            samsung_privids=parse_csv_list(args.select_samsung_privids),
        )

        # If user explicitly asked for adversary tags, keep only those that match
        if sel.adv_tags:
            sel.adv_tags = filter_adv_tags_with_matches(df_full, payload_col, sel.adv_tags)
    else:
        sel = interactive_select(
            adv_tag_candidates=adv_tag_candidates,
            apple_candidates=apple_candidates,
            google_candidates=google_candidates,
            tile_candidates=tile_candidates,
            samsung_priv_candidates=samsung_priv_candidates,
            rng=rng,
        )

    if not (sel.adv_tags or sel.apple_macs or sel.google_macs or sel.tile_macs or sel.samsung_privids):
        print("[i] No devices selected to re-introduce. Will export background-only dataset.")

        # Build output folder name based on "0 selections"
        base = os.path.splitext(os.path.basename(args.input))[0]
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        counts_part = "adv0_apple0_google0_samsung0_tile0"

        # Join properly with args.outdir (args.outdir is a base folder like "controlled/Car_Adv/")
        outdir = os.path.join(args.outdir, f"scenarios_{base}__{counts_part}__{stamp}")
        os.makedirs(outdir, exist_ok=True)
        print(f"\n[i] Writing scenario CSVs to: {outdir}")

        # Save a single background-only CSV (same columns as input)
        out_path = os.path.join(outdir, "background_only.csv")
        df_base.sort_values(ts_col).to_csv(out_path, index=False)
        print(f"[✓] Wrote background-only CSV: {out_path}")
        return 0

    print("\n=== Selected set to re-introduce ===")
    print(f"Adversary TAGs ({len(sel.adv_tags)}): {sel.adv_tags}")
    print(f"APPLE MACs     ({len(sel.apple_macs)}): {sel.apple_macs}")
    print(f"GOOGLE MACs    ({len(sel.google_macs)}): {sel.google_macs}")
    print(f"TILE MACs      ({len(sel.tile_macs)}): {sel.tile_macs}")
    print(f"SAMSUNG PRIVIDs ({len(sel.samsung_privids)}): {sel.samsung_privids}")

    # 3) Build rows to re-introduce from FULL capture
    parts_to_modify = []

    # Adversaries: select rows by TAG occurrence
    if sel.adv_tags:
        payload = df_full[payload_col].astype(str).str.lower()
        adv_sel_mask = pd.Series(False, index=df_full.index)
        for t in sel.adv_tags:
            adv_sel_mask |= payload.str.contains(t.lower(), na=False)
        adv_sel_rows = df_full.loc[adv_sel_mask].copy()
        if not adv_sel_rows.empty:
            # group by which tag matched (if multiple tags could match one row, assign first match)
            def first_matching_tag(p: str) -> str:
                ph = str(p).lower()
                for t in sel.adv_tags:
                    if t.lower() in ph:
                        return t.lower()
                return "unknown_tag"
            adv_sel_rows["_group"] = adv_sel_rows[payload_col].astype(str).map(first_matching_tag)
            parts_to_modify.append(adv_sel_rows)

    # APPLE/GOOGLE/TILE: select by MAC
    tag_selected_macs = sel.selected_tag_macs()
    if tag_selected_macs:
        tag_sel_rows = df_full[df_full[mac_col].isin(tag_selected_macs)].copy()
        if not tag_sel_rows.empty:
            tag_sel_rows["_group"] = tag_sel_rows[mac_col]  # group by MAC
            parts_to_modify.append(tag_sel_rows)

    # SAMSUNG: select rows by PRIVID
    if sel.samsung_privids and tag_col:
        s = df_full[df_full[tag_col] == "SAMSUNG"].copy()
        if not s.empty:
            s["_privid"] = s[payload_col].astype(str).map(get_samsung_privid_from_payload)
            s_sel_rows = s[s["_privid"].isin([x.lower() for x in sel.samsung_privids])].copy()
            if not s_sel_rows.empty:
                s_sel_rows["_group"] = s_sel_rows["_privid"]  # group by PRIVID (device)
                parts_to_modify.append(s_sel_rows)

    if not parts_to_modify:
        print("[w] None of the selected items matched rows in the capture. Nothing to re-introduce.")
        return 0

    df_reintro = pd.concat(parts_to_modify, axis=0, ignore_index=True).copy()
    df_reintro = df_reintro.sort_values(ts_col).reset_index(drop=True)

    # 4) Pseudonym pool: existing MACs, use background MACs for realism
    pseudonym_pool = sorted(list(all_macs_base)) or sorted(list(all_macs_full))

    # Mirror MAC columns
    also_replace_cols = [c.strip() for c in args.mirror_mac_cols.split(",") if c.strip()]
    for c in MAC_COL_CANDIDATES:
        if c not in also_replace_cols:
            also_replace_cols.append(c)

    # 5) Output folder  (UPDATED: encode the user's *count choices* in the folder name)
    base = os.path.splitext(os.path.basename(args.input))[0]
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # Counts of what the user selected to re-introduce
    n_adv = len(sel.adv_tags)
    n_apple = len(sel.apple_macs)
    n_google = len(sel.google_macs)
    n_tile = len(sel.tile_macs)
    n_samsung = len(sel.samsung_privids)

    # Example folder:
    # scenarios_SDR_Adv__adv1_apple4_google10_samsung2_tile16__20260124_153045
    counts_part = f"adv{n_adv}_apple{n_apple}_google{n_google}_samsung{n_samsung}_tile{n_tile}"

    outdir = os.path.join(args.outdir, f"scenarios_{base}__{counts_part}__{stamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n[i] Writing scenario CSVs to: {outdir}")

    # 6) Generate scenarios
    total = 0
    for tx in TX_PERIODS_SEC:
        for rot in ROT_PERIODS_SEC:
            scen_seed = (args.seed * 1000003) ^ (tx * 1009) ^ (rot * 9176)
            scen_rng = random.Random(scen_seed)

            df_mod = apply_behavior_grouped(
                df_rows=df_reintro,
                group_key="_group",
                ts_col=ts_col,
                mac_col=mac_col,
                tx_period_sec=tx,
                rot_period_sec=rot,
                pseudonym_pool=pseudonym_pool,
                rng=scen_rng,
                also_replace_cols=also_replace_cols,
            )

            df_out = pd.concat([df_base, df_mod], axis=0, ignore_index=True)
            df_out = df_out.sort_values(ts_col).reset_index(drop=True)

            out_name = f"scenario_tx-{_pretty_seconds(tx)}_rot-{_pretty_seconds(rot)}.csv"
            df_out.to_csv(os.path.join(outdir, out_name), index=False)
            total += 1

    print(f"[✓] Done. Wrote {total} scenario CSVs.")

    # 7) Post-check: plot effective TX/ROT as boxplots
    try:
        p = plot_effective_tx_rot_boxplots(
            outdir=outdir,
            ts_col=ts_col,
            mac_col=mac_col,
            payload_col=payload_col,
            sel=sel,
            max_samples_per_scenario=5000,
        )
        if p:
            print(f"[✓] Wrote effective tx/rot boxplots: {p}")
    except Exception as e:
        print(f"[w] Failed to plot effective tx/rot boxplots: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())