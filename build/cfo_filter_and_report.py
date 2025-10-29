#!/usr/bin/env python3
"""
cfo_filter_and_report.py

Quality-gate BLE CFO estimates from a features CSV and report medians/IQRs.
- ADV-only by default (pdu_type ∈ {0,2,4,6})
- Stricter gates on PNR, gate length, bandwidth, and CFO std columns
- Gross CFO magnitude cap
- MAD-based outlier trimming on the *selected* reporting estimator
- Reporting estimator preference: centroid → two-stage → quick

Outputs:
- Console summary (kept count, medians/IQRs)
- Deadband (±200 Hz) sign-agreement table on the kept set

Usage:
  python3 cfo_filter_and_report.py features_fixed_overwrite.csv
  # optional overrides:
  python3 cfo_filter_and_report.py features.csv --pnr-min 15 --gate-min 560 --bw-min-khz 300 --bw-max-khz 1400 \
      --cfo-std-max 20000 --cfo-std-sym-max 20000 --abs-cfo-max 80000 --mad-k 3.5 \
      --pdu-allow 0,2,4,6
"""

import argparse
import csv
import math
from pathlib import Path
import numpy as np


# ------------------------- parsing helpers -------------------------
def f(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def read_rows(path):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        rows = list(r)
        hdr = r.fieldnames or []
    return rows, hdr


def get_col(rows, name, default=np.nan):
    if not rows or name not in rows[0]:
        return np.full(len(rows), default, float)
    return np.array([f(r.get(name, default)) for r in rows], float)


def get_str_col(rows, name, default=""):
    if not rows or name not in rows[0]:
        return np.array([default] * len(rows), dtype=object)
    return np.array([(r.get(name) or default) for r in rows], dtype=object)


def iqr(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return q25, q50, q75


def med_iqr(a, mask):
    m = mask & np.isfinite(a)
    if not np.any(m):
        return (np.nan, np.nan)
    q25, q50, q75 = iqr(a[m])
    return (q50, q75 - q25)


# ------------------------- agreement (sign) -------------------------
def sign_with_deadband(x, deadband=200.0):
    """Return -1, 0, +1 depending on sign after applying ±deadband around 0."""
    x = np.asarray(x, float)
    s = np.zeros_like(x, dtype=int)
    s[x > +deadband] = +1
    s[x < -deadband] = -1
    return s


def percent_agree(x, y, mask, deadband=200.0):
    m = mask & np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return float("nan")
    sx = sign_with_deadband(x[m], deadband)
    sy = sign_with_deadband(y[m], deadband)
    return 100.0 * (np.mean(sx == sy))


# ------------------------- MAD trimming -------------------------
def mad_trim(x, base_mask, k=3.5):
    """
    Robustly trim outliers based on MAD around the median on the selected series x.
    Returns a refined mask = base_mask & |x - med| <= k * 1.4826 * MAD
    """
    m = base_mask & np.isfinite(x)
    if not np.any(m):
        return base_mask
    xm = x[m]
    med = np.median(xm)
    mad = np.median(np.abs(xm - med))
    # avoid divide-by-zero; no trimming if degenerate
    if not np.isfinite(mad) or mad < 1e-12:
        return base_mask
    rad = k * 1.4826 * mad
    keep = np.abs(x - med) <= rad
    return base_mask & keep


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_csv", help="CSV with columns incl. CFO, PNR, gated_len_us, bw_3db_hz, pdu_type, etc.")
    # Gates (defaults from the discussion)
    ap.add_argument("--pnr-min", type=float, default=15.0, help="Minimum psd_pnr_db (dB)")
    ap.add_argument("--gate-min", type=float, default=560.0, help="Minimum gated_len_us (µs)")
    ap.add_argument("--bw-min-khz", type=float, default=300.0, help="Min 3dB BW in kHz")
    ap.add_argument("--bw-max-khz", type=float, default=1400.0, help="Max 3dB BW in kHz")
    ap.add_argument("--cfo-std-max", type=float, default=20000.0, help="Max cfo_std_hz (Hz)")
    ap.add_argument("--cfo-std-sym-max", type=float, default=20000.0, help="Max cfo_std_sym_hz (Hz)")
    ap.add_argument("--abs-cfo-max", type=float, default=80000.0, help="Absolute CFO cap on selected series (Hz)")
    ap.add_argument("--mad-k", type=float, default=3.5, help="MAD k-multiple for robust trimming")
    ap.add_argument("--deadband", type=float, default=200.0, help="Deadband for sign agreement (Hz)")
    ap.add_argument("--pdu-allow", default="0,2,4,6",
                    help="Comma list of PDU types to keep (ADV_* by default). e.g., '0,2,4,6'")
    ap.add_argument("--write-filtered", default="", help="Optional path to write filtered rows (CSV)")
    args = ap.parse_args()

    rows, hdr = read_rows(args.features_csv)
    if not rows:
        print("No rows found.")
        return

    # --- Pull needed columns
    # estimators
    c_quick = get_col(rows, "cfo_quick_hz")
    c_cent  = get_col(rows, "cfo_centroid_hz")
    c_two   = get_col(rows, "cfo_two_stage_hz")
    c_coarse= get_col(rows, "cfo_two_stage_coarse_hz")  # optional; may be NaN

    # quality
    pnr   = get_col(rows, "psd_pnr_db")
    glen  = get_col(rows, "gated_len_us")
    bw    = get_col(rows, "bw_3db_hz")
    cs    = get_col(rows, "cfo_std_hz")
    css   = get_col(rows, "cfo_std_sym_hz")

    # meta
    pdu_s = get_col(rows, "pdu_type")  # numeric if present
    if np.all(~np.isfinite(pdu_s)):    # if missing or non-numeric, try string then cast
        pdu_str = get_str_col(rows, "pdu_type", default="")
        def _pdu_parse(s):
            s = (s or "").strip()
            if s.lower().startswith("0x"):
                try: return int(s, 16)
                except Exception: return float("nan")
            try: return int(s)
            except Exception: return float("nan")
        pdu_s = np.array([_pdu_parse(s) for s in pdu_str], float)

    # ADV-only selection
    allow = set()
    for tok in args.pdu_allow.split(","):
        tok = tok.strip()
        if tok:
            try:
                allow.add(int(tok, 0))  # supports "0x.." or decimal
            except Exception:
                pass
    adv_mask = np.isin(pdu_s, list(allow))

    # --- Build the base quality mask
    bw_min_hz = args.bw_min_khz * 1e3
    bw_max_hz = args.bw_max_khz * 1e3

    base_mask = (
        np.isfinite(pnr) & (pnr >= args.pnr_min) &
        np.isfinite(glen) & (glen >= args.gate_min) &
        np.isfinite(bw) & (bw >= bw_min_hz) & (bw <= bw_max_hz) &
        adv_mask
    )

    # Optional std caps: only apply if explicitly set (not None / not negative)
    def _cap(arr, thr):
        if thr is None or thr < 0:
            return np.ones_like(arr, dtype=bool)
        return np.isfinite(arr) & (arr <= thr)

    # Interpret “very large” thresholds as “off”
    std_cap = None if args.cfo_std_max >= 1e8 else args.cfo_std_max
    std_sym_cap = None if args.cfo_std_sym_max >= 1e8 else args.cfo_std_sym_max

    base_mask &= _cap(cs, std_cap)
    base_mask &= _cap(css, std_sym_cap)

    # --- Choose the reporting estimator: centroid → two-stage → quick
    csel = np.where(np.isfinite(c_cent), c_cent,
           np.where(np.isfinite(c_two),  c_two, c_quick))

    # gross CFO magnitude limit on the selected series
    mask = base_mask & np.isfinite(csel) & (np.abs(csel) <= args.abs_cfo_max)

    # --- MAD trimming on the selected series
    mask = mad_trim(csel, mask, k=args.mad_k)

    kept = int(np.sum(mask))
    n_all = len(rows)
    print("\n=== QUALITY GATING ===")
    print(f"kept {kept}/{n_all} rows  ({100.0*kept/n_all:.1f}%)  "
          f"[PNR≥{args.pnr_min:.1f} dB, gate≥{args.gate_min:.0f} µs, "
          f"{args.bw_min_khz:.0f}–{args.bw_max_khz:.0f} kHz, ADV only, "
          f"std caps + |CFO|≤{args.abs_cfo_max/1e3:.0f} kHz, MAD k={args.mad_k:.1f}]")

    # --- Report per-estimator stats on the kept set
    def pr(name, arr):
        med, rng = med_iqr(arr, mask)
        if np.isfinite(med):
            print(f"{name:<9}: median={med:+.1f} Hz, IQR={rng:.1f} Hz")
        else:
            print(f"{name:<9}: median=nan, IQR=nan")

    print("")
    pr("two_stage", c_two)
    pr("quick",     c_quick)
    pr("centroid",  c_cent)
    pr("consensus", csel)

    # --- Agreement table (deadband sign agreement) on the kept set
    print("\n=== AGREEMENT on gated set (±{:.0f} Hz deadband) ===".format(args.deadband))
    def pa(n1, a, n2, b):
        val = percent_agree(a, b, mask, deadband=args.deadband)
        if np.isfinite(val):
            print(f"{n1:<6} vs {n2:<8}: {val:5.2f}%")
        else:
            print(f"{n1:<6} vs {n2:<8}: n/a")

    pa("quick", c_quick, "centroid", c_cent)
    pa("quick", c_quick, "two",      c_two)
    pa("cent.", c_cent,  "two",      c_two)

    # --- optionally write filtered CSV
    if args.write_filtered:
        outp = Path(args.write_filtered)
        outp.parent.mkdir(parents=True, exist_ok=True)
        # keep only rows under mask
        kept_rows = [r for r, m in zip(rows, mask) if m]
        if kept_rows:
            fieldnames = list(kept_rows[0].keys())
            with open(outp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(kept_rows)
            print(f"\n[OK] wrote filtered rows → {outp}")
        else:
            print(f"\n[INFO] No rows passed; nothing written to {outp}")

if __name__ == "__main__":
    main()