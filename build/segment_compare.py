#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""segment_compare.py

Compare the *nature of segments* between two controlled datasets (e.g., Benign vs Car_Adv).

This script is intentionally lightweight and does NOT import AirCatch.py (to avoid
side-effects from global config edits). It re-implements the segmentation logic used
in your current AirCatch:
  - per time bucket (WINDOW_S)
  - per ecosystem (eco)
  - per device id: AdvA for non-Samsung, PRIVID (fallback AdvA) for Samsung

Outputs:
  - out/segment_compare__file_level.csv
  - out/segment_compare__segment_level.csv
  - out/segment_compare__summary.txt
  - out/segment_compare__plots.png

Usage:
  python segment_compare.py \
    --root controlled \
    --a Benign \
    --b Car_Adv \
    --window-s 1800
"""

import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HEX_DIGITS = "0123456789abcdefABCDEF"
ADV_PAYLOAD_TAG = "4c001219ff"  # used only for packet-level counts (not a filter)

CFO_COLS_RAW = ["CFO_Hz", "CFO_00_Hz", "CFO_11_Hz", "CFO_10_Hz", "CFO_01_Hz"]

# Feature thresholds (tweak as needed)
LONG_SEGMENT_S_DEFAULT = 300.0
HEAVY_SEGMENT_PKTS_DEFAULT = 30


def _collect_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    if path.is_dir():
        return sorted([p for p in path.rglob("*.csv") if p.is_file()])
    return []


def _clean_hexpayload(payload_hex: str) -> str:
    if payload_hex is None:
        return ""
    s = str(payload_hex).strip()
    if not s:
        return ""
    s = s.replace(" ", "").replace(":", "").replace("-", "").replace("\n", "").replace("\r", "").replace("\t", "")
    s = "".join(ch for ch in s if ch in HEX_DIGITS)
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


def classify_tag_ecosystem_from_payload(payload_hex: str) -> str:
    """APPLE/GOOGLE/TILE/SAMSUNG/UNKNOWN; matches your AirCatch logic."""
    hx = _clean_hexpayload(payload_hex)
    if len(hx) < 16:
        return "UNKNOWN"

    b = _hex_to_bytes(hx)
    if len(b) < 8:
        return "UNKNOWN"

    def parse_ad_structures(ad_data: bytes) -> str:
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

            # Apple FindMy
            if ad_type == 0xFF and len(data) >= 4:
                company_id = data[0] | (data[1] << 8)
                if company_id == 0x004C and data[2] == 0x12 and data[3] == 0x19:
                    return "APPLE"

            # Service Data 16-bit
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


def get_samsung_privid_from_payload(payload_hex: str) -> str:
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

    if len(b) >= 7:
        v = extract_from_ad(b[6:])
        if v:
            return v
    if len(b) >= 9:
        v = extract_from_ad(b[8:])
        if v:
            return v
    return ""


def prepare_segments(df: pd.DataFrame, window_s: int) -> pd.DataFrame:
    """Return per-(seg_id, eco, dev_id) aggregated segments."""
    df = df.copy()

    if "payload" not in df.columns or "timestamp" not in df.columns or "AdvA" not in df.columns:
        raise RuntimeError("Missing required columns: timestamp, AdvA, payload")

    # Normalize
    df["payload"] = df["payload"].astype(str).str.lower()
    df["AdvA"] = df["AdvA"].astype(str)

    # Clean CFO columns presence
    missing_cfo = [c for c in CFO_COLS_RAW if c not in df.columns]
    if missing_cfo:
        raise RuntimeError(f"Missing CFO columns: {missing_cfo}")

    df = df.dropna(subset=["timestamp", "AdvA"] + CFO_COLS_RAW)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[np.isfinite(df["timestamp"])].copy()
    df = df.sort_values("timestamp")

    df["eco"] = df["payload"].apply(classify_tag_ecosystem_from_payload)

    # Samsung PRIVID
    df["privid"] = ""
    samsung_mask = (df["eco"] == "SAMSUNG")
    if samsung_mask.any():
        df.loc[samsung_mask, "privid"] = df.loc[samsung_mask, "payload"].apply(get_samsung_privid_from_payload)
        missing_priv = samsung_mask & (df["privid"].astype(str) == "")
        if missing_priv.any():
            df.loc[missing_priv, "privid"] = df.loc[missing_priv, "AdvA"].astype(str)

    df["dev_id"] = df["AdvA"].astype(str)
    df.loc[samsung_mask, "dev_id"] = df.loc[samsung_mask, "privid"].astype(str)

    df["seg_id"] = (df["timestamp"] // float(window_s)).astype(int)

    # Packet-level counts useful as discriminators
    df["is_adv_tag"] = df["payload"].str.contains(ADV_PAYLOAD_TAG, na=False)

    # Aggregate packets into per-device segments
    gcols = ["seg_id", "eco", "dev_id"]

    def _iqr(x: np.ndarray) -> float:
        if x is None or len(x) == 0:
            return float("nan")
        x = np.asarray(x, dtype=float)
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    aggs = {
        "timestamp": ["min", "max", "count"],
        "AdvA": ["nunique"],
        "is_adv_tag": ["sum"],
    }

    for c in CFO_COLS_RAW:
        aggs[c] = ["mean", "std", _iqr]

    out = df.groupby(gcols, sort=True).agg(aggs)
    out.columns = ["_".join([c for c in col if c]) for col in out.columns.to_flat_index()]
    out = out.reset_index()

    out = out.rename(columns={
        "timestamp_min": "t_start",
        "timestamp_max": "t_end",
        "timestamp_count": "n_packets",
        "AdvA_nunique": "n_unique_adva_in_seg",
        "is_adv_tag_sum": "n_adv_tag_packets",
    })

    out["duration_obs"] = (out["t_end"] - out["t_start"]).astype(float)
    out["duration_est"] = float(window_s)

    # Some derived CFO norms (useful to see if one dataset has wider CFO spread)
    out["cfo_std_norm"] = np.sqrt(
        np.square(out["CFO_Hz_std"].fillna(0.0)) +
        np.square(out["CFO_00_Hz_std"].fillna(0.0)) +
        np.square(out["CFO_11_Hz_std"].fillna(0.0)) +
        np.square(out["CFO_10_Hz_std"].fillna(0.0)) +
        np.square(out["CFO_01_Hz_std"].fillna(0.0))
    )

    return out


def file_level_summary(df_raw: pd.DataFrame, seg: pd.DataFrame, label: str, src_file: str) -> dict:
    eco_counts = Counter(df_raw["eco"].astype(str).tolist()) if "eco" in df_raw.columns else Counter()
    total_pkts = int(len(df_raw))

    return {
        "label": label,
        "src_file": src_file,
        "n_packets": total_pkts,
        "n_segments": int(len(seg)),
        "n_windows": int(seg["seg_id"].nunique()) if len(seg) else 0,
        "eco_frac_apple": float(eco_counts.get("APPLE", 0) / max(total_pkts, 1)),
        "eco_frac_google": float(eco_counts.get("GOOGLE", 0) / max(total_pkts, 1)),
        "eco_frac_tile": float(eco_counts.get("TILE", 0) / max(total_pkts, 1)),
        "eco_frac_samsung": float(eco_counts.get("SAMSUNG", 0) / max(total_pkts, 1)),
        "eco_frac_unknown": float(eco_counts.get("UNKNOWN", 0) / max(total_pkts, 1)),
        "adv_tag_pkt_frac": float(df_raw.get("is_adv_tag", pd.Series([0]*total_pkts)).mean()) if total_pkts else 0.0,
        "seg_packets_p50": float(seg["n_packets"].median()) if len(seg) else float("nan"),
        "seg_packets_p90": float(seg["n_packets"].quantile(0.90)) if len(seg) else float("nan"),
        "seg_duration_obs_p50": float(seg["duration_obs"].median()) if len(seg) else float("nan"),
        "cfo_std_norm_p50": float(seg["cfo_std_norm"].median()) if len(seg) else float("nan"),
        "cfo_std_norm_p90": float(seg["cfo_std_norm"].quantile(0.90)) if len(seg) else float("nan"),
    }


def _robust_quantiles(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {"p50": np.nan, "p90": np.nan, "p99": np.nan, "mean": np.nan}
    return {
        "p50": float(x.median()),
        "p90": float(x.quantile(0.9)),
        "p99": float(x.quantile(0.99)),
        "mean": float(x.mean()),
    }


def _safe_frac(mask: pd.Series) -> float:
    if mask is None:
        return 0.0
    mask = mask.astype(bool)
    return float(mask.mean()) if len(mask) else 0.0


def featurize_file_from_segments(seg: pd.DataFrame, *, long_s: float, heavy_pkts: int) -> dict:
    """Compute per-file features from segment-level table (no ADV tag)."""
    if seg is None or len(seg) == 0:
        return {
            "dev_ids_per_window_mean": 0.0,
            "dev_ids_per_window_p90": 0.0,
            "segments_per_window_mean": 0.0,
            "segments_per_window_p90": 0.0,
            "long_segment_frac": 0.0,
            "heavy_segment_frac": 0.0,
            "active_cfo_frac": 0.0,
            "seg_n_packets_p50": np.nan,
            "seg_n_packets_p90": np.nan,
            "seg_duration_obs_p50": np.nan,
            "seg_duration_obs_p90": np.nan,
            "cfo_std_norm_p50": np.nan,
            "cfo_std_norm_p90": np.nan,
            "cfo_std_norm_p99": np.nan,
        }

    # Per-window counts
    win = seg.groupby("seg_id", as_index=False).agg(
        dev_ids_in_window=("dev_id", "nunique"),
        segments_in_window=("dev_id", "size"),
    )

    dev_q = _robust_quantiles(win["dev_ids_in_window"])
    seg_q = _robust_quantiles(win["segments_in_window"])

    # Segment-level thresholds
    long_frac = _safe_frac(pd.to_numeric(seg["duration_obs"], errors="coerce") > float(long_s))
    heavy_frac = _safe_frac(pd.to_numeric(seg["n_packets"], errors="coerce") > int(heavy_pkts))

    # CFO activity: any non-zero CFO spread (std norm)
    cfo_std_norm = pd.to_numeric(seg["cfo_std_norm"], errors="coerce").fillna(0.0)
    active_cfo_frac = float((cfo_std_norm > 0).mean()) if len(cfo_std_norm) else 0.0

    np_q = _robust_quantiles(seg["n_packets"])
    dur_q = _robust_quantiles(seg["duration_obs"])
    cfo_q = _robust_quantiles(cfo_std_norm)

    out = {
        "dev_ids_per_window_mean": dev_q["mean"],
        "dev_ids_per_window_p90": dev_q["p90"],
        "segments_per_window_mean": seg_q["mean"],
        "segments_per_window_p90": seg_q["p90"],
        "long_segment_frac": float(long_frac),
        "heavy_segment_frac": float(heavy_frac),
        "active_cfo_frac": float(active_cfo_frac),
        "seg_n_packets_p50": np_q["p50"],
        "seg_n_packets_p90": np_q["p90"],
        "seg_duration_obs_p50": dur_q["p50"],
        "seg_duration_obs_p90": dur_q["p90"],
        "cfo_std_norm_p50": cfo_q["p50"],
        "cfo_std_norm_p90": cfo_q["p90"],
        "cfo_std_norm_p99": cfo_q["p99"],
    }

    # Per-dimension CFO spread features (median/std of stds)
    for c in CFO_COLS_RAW:
        col_std = f"{c}_std"
        if col_std in seg.columns:
            q = _robust_quantiles(seg[col_std])
            out[f"{c}_std_p50"] = q["p50"]
            out[f"{c}_std_p90"] = q["p90"]

    return out


def _separability_rank(files_feat: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Rank features by simple effect size (abs mean diff / pooled std)."""
    df = files_feat.copy()
    feats = [c for c in df.columns if c not in {"label", "src_file"}]

    rows = []
    da = df[df["label"] == label_a]
    db = df[df["label"] == label_b]

    for c in feats:
        xa = pd.to_numeric(da[c], errors="coerce").dropna()
        xb = pd.to_numeric(db[c], errors="coerce").dropna()
        if len(xa) < 2 or len(xb) < 2:
            continue
        ma = float(xa.mean())
        mb = float(xb.mean())
        sa = float(xa.std(ddof=1))
        sb = float(xb.std(ddof=1))
        sp = float(np.sqrt((sa * sa + sb * sb) / 2.0))
        if sp <= 1e-12:
            continue
        d = abs(ma - mb) / sp
        rows.append({
            "feature": c,
            f"mean_{label_a}": ma,
            f"mean_{label_b}": mb,
            "effect": d,
        })

    out = pd.DataFrame(rows).sort_values("effect", ascending=False)
    return out


def run_one_folder(root: Path, subfolder: str, window_s: int, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / subfolder
    csvs = _collect_csv_files(base)
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under: {base}")

    file_rows = []
    seg_rows = []

    for p in csvs:
        df = pd.read_csv(p)
        df["_src_file"] = str(p)

        # add packet-level eco for summaries (no ADV tag usage in features)
        df["payload"] = df["payload"].astype(str).str.lower()
        df["eco"] = df["payload"].apply(classify_tag_ecosystem_from_payload)

        seg = prepare_segments(df, window_s=window_s)
        seg["label"] = label
        seg["src_file"] = str(p)

        file_rows.append(file_level_summary(df, seg, label=label, src_file=str(p)))
        seg_rows.append(seg)

    return pd.DataFrame(file_rows), pd.concat(seg_rows, ignore_index=True) if seg_rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="controlled")
    ap.add_argument("--a", required=True, help="First subfolder under root (e.g., Benign)")
    ap.add_argument("--b", required=True, help="Second subfolder under root (e.g., Car_Adv)")
    ap.add_argument("--window-s", type=int, default=1800)
    ap.add_argument("--out", default="out")
    ap.add_argument("--long-s", type=float, default=LONG_SEGMENT_S_DEFAULT, help="Long segment threshold (seconds)")
    ap.add_argument("--heavy-pkts", type=int, default=HEAVY_SEGMENT_PKTS_DEFAULT, help="Heavy segment threshold (packets)")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    files_a, seg_a = run_one_folder(root, args.a, args.window_s, label=args.a)
    files_b, seg_b = run_one_folder(root, args.b, args.window_s, label=args.b)

    files = pd.concat([files_a, files_b], ignore_index=True)
    segs = pd.concat([seg_a, seg_b], ignore_index=True)

    out_files = outdir / "segment_compare__file_level.csv"
    out_segs = outdir / "segment_compare__segment_level.csv"
    out_txt = outdir / "segment_compare__summary.txt"
    out_png = outdir / "segment_compare__plots.png"
    out_feat = outdir / "segment_compare__file_features.csv"
    out_rank = outdir / "segment_compare__feature_rank.csv"

    files.to_csv(out_files, index=False)
    segs.to_csv(out_segs, index=False)

    # Per-file engineered features (no ADV tag)
    feat_rows = []
    for (lab, src), g in segs.groupby(["label", "src_file"], sort=False):
        r = {"label": lab, "src_file": src}
        r.update(featurize_file_from_segments(g, long_s=float(args.long_s), heavy_pkts=int(args.heavy_pkts)))
        feat_rows.append(r)

    files_feat = pd.DataFrame(feat_rows)
    files_feat.to_csv(out_feat, index=False)

    rank = _separability_rank(files_feat, args.a, args.b)
    rank.to_csv(out_rank, index=False)

    # Append top-ranked features to summary text
    with open(out_txt, "a", encoding="utf-8") as f:
        f.write("\n=== NO-ADV FEATURE RANK (effect size) ===\n")
        if rank is None or len(rank) == 0:
            f.write("(no features ranked)\n")
        else:
            f.write(rank.head(25).to_string(index=False))
            f.write("\n")

    # Text summary (quick discriminators)
    def _summ(name: str, s: pd.Series) -> str:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return f"{name}: n=0\n"
        return (
            f"{name}: n={len(s)} p50={s.median():.3g} p90={s.quantile(0.9):.3g} p99={s.quantile(0.99):.3g} mean={s.mean():.3g}\n"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== FILE LEVEL ===\n")
        for lab in [args.a, args.b]:
            d = files[files["label"] == lab]
            f.write(f"\n[{lab}] files={len(d)}\n")
            f.write(_summ("n_packets", d["n_packets"]))
            f.write(_summ("n_segments", d["n_segments"]))
            f.write(_summ("adv_tag_pkt_frac", d["adv_tag_pkt_frac"]))
            f.write(_summ("cfo_std_norm_p50", d["cfo_std_norm_p50"]))
            f.write(_summ("cfo_std_norm_p90", d["cfo_std_norm_p90"]))

        f.write("\n=== SEGMENT LEVEL ===\n")
        for lab in [args.a, args.b]:
            d = segs[segs["label"] == lab]
            f.write(f"\n[{lab}] segments={len(d)}\n")
            f.write(_summ("n_packets", d["n_packets"]))
            f.write(_summ("duration_obs", d["duration_obs"]))
            f.write(_summ("cfo_std_norm", d["cfo_std_norm"]))
            f.write(_summ("n_adv_tag_packets", d["n_adv_tag_packets"]))

    # Plots
    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 2, 1)
    for lab, color in [(args.a, "steelblue"), (args.b, "darkorange")]:
        d = segs[segs["label"] == lab]
        x = pd.to_numeric(d["n_packets"], errors="coerce").dropna().values
        ax1.hist(x, bins=40, alpha=0.5, label=lab, color=color)
    ax1.set_title("Segment n_packets")
    ax1.set_xlabel("n_packets")
    ax1.set_ylabel("count")
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    for lab, color in [(args.a, "steelblue"), (args.b, "darkorange")]:
        d = segs[segs["label"] == lab]
        x = pd.to_numeric(d["cfo_std_norm"], errors="coerce").dropna().values
        ax2.hist(x, bins=40, alpha=0.5, label=lab, color=color)
    ax2.set_title("Segment CFO spread (std norm)")
    ax2.set_xlabel("cfo_std_norm")
    ax2.set_ylabel("count")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    files_plot = files.copy()
    files_plot["label"] = files_plot["label"].astype(str)
    ax3.boxplot(
        [files_plot[files_plot["label"] == args.a]["n_segments"].dropna().values,
         files_plot[files_plot["label"] == args.b]["n_segments"].dropna().values],
        labels=[args.a, args.b],
        showfliers=False,
    )
    ax3.set_title("Segments per file")
    ax3.set_ylabel("n_segments")

    ax4 = plt.subplot(2, 2, 4)
    # ecosystem mix
    eco_cols = ["eco_frac_apple", "eco_frac_google", "eco_frac_tile", "eco_frac_samsung", "eco_frac_unknown"]
    means = files.groupby("label")[eco_cols].mean(numeric_only=True)
    x = np.arange(len(eco_cols))
    w = 0.35
    ax4.bar(x - w/2, means.loc[args.a].values, width=w, label=args.a, color="steelblue")
    ax4.bar(x + w/2, means.loc[args.b].values, width=w, label=args.b, color="darkorange")
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.replace("eco_frac_", "") for c in eco_cols], rotation=30, ha="right")
    ax4.set_title("Mean ecosystem fraction")
    ax4.set_ylim(0, 1)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print(f"Wrote: {out_files}")
    print(f"Wrote: {out_segs}")
    print(f"Wrote: {out_feat}")
    print(f"Wrote: {out_rank}")
    print(f"Wrote: {out_txt}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
