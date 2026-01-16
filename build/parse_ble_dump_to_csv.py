#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_ble_dump_to_csv.py

Parse BLE debug text dumps ("---- BLE PACKET ----" blocks) into a CSV compatible with
cfo_samples_rail.csv.

Expected per-block fields in the .txt (when present):
  Timestamp: HH:MM:SS.mmm
  AA corr: <int> AA pos: <int> PRE corr: <int>
  Channel: <int> Phase: <int> Polarity: <int>
  Slip: <int>
  CFO Total: <int> Hz | 00: <int> | 11: <int> | 10: <int> | 01: <int>
  PDU: <NAME> (type=<int>)
  Len: <int> ...
  AdvA: XX:XX:XX:XX:XX:XX
  Tag: ... is_tag_ecosystem=true/false ...
  CRC ok: TRUE/FALSE ...
  payload (hex): <hex>

Columns that are not present in the dump are filled with defaults:
  - CFO_from_transitions_Hz: empty
  - nprod_*: 0
  - window_start/window_end: -1
  - cfo_window: "unknown"
"""

import argparse
import csv
import math
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


CSV_COLUMNS = [
    "timestamp",
    "AdvA",
    "payload",
    "CFO_Hz",
    "CFO_00_Hz",
    "CFO_11_Hz",
    "CFO_10_Hz",
    "CFO_01_Hz",
    "CFO_from_transitions_Hz",
    "nprod_00",
    "nprod_11",
    "nprod_10",
    "nprod_01",
    "nprod_total",
    "crc_ok",
    "is_tag_ecosystem",
    "pdu_type",
    "pdu_type_name",
    "length",
    "channel",
    "phase",
    "polarity",
    "slip",
    "window_start",
    "window_end",
    "aa_pos",
    "aa_corr",
    "pre_corr",
    "cfo_window",
]


@dataclass
class Row:
    timestamp: float
    AdvA: str
    payload: str
    CFO_Hz: Optional[float]
    CFO_00_Hz: Optional[float]
    CFO_11_Hz: Optional[float]
    CFO_10_Hz: Optional[float]
    CFO_01_Hz: Optional[float]
    CFO_from_transitions_Hz: float
    nprod_00: int
    nprod_11: int
    nprod_10: int
    nprod_01: int
    nprod_total: int
    crc_ok: int
    is_tag_ecosystem: int
    pdu_type: int
    pdu_type_name: str
    length: int
    channel: int
    phase: int
    polarity: int
    slip: int
    window_start: int
    window_end: int
    aa_pos: int
    aa_corr: int
    pre_corr: int
    cfo_window: str


RE_TS = re.compile(r"^Timestamp:\s*([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,6}))?\s*$")
RE_AA = re.compile(r"^AA\s+corr:\s*(-?\d+)\s+AA\s+pos:\s*(-?\d+)\s+PRE\s+corr:\s*(-?\d+)\s*$")
RE_CH = re.compile(r"^Channel:\s*(\d+)\s+Phase:\s*(-?\d+)\s+Polarity:\s*(-?\d+)\s*$")
RE_SLIP = re.compile(r"^Slip:\s*(-?\d+)\s*$")
RE_CFO = re.compile(
    r"^CFO\s+Total:\s*(-?\d+(?:\.\d+)?)\s*Hz\s*\|\s*00:\s*(-?\d+(?:\.\d+)?)\s*\|\s*11:\s*(-?\d+(?:\.\d+)?)\s*\|\s*10:\s*(-?\d+(?:\.\d+)?)\s*\|\s*01:\s*(-?\d+(?:\.\d+)?)\s*$"
)
RE_PDU = re.compile(r"^PDU:\s*(.+?)\s*\(type=(\d+)\)\s*$")
RE_LEN = re.compile(r"^Len:\s*(\d+)\s+.*$")
RE_ADVA = re.compile(r"^AdvA:\s*([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})\s*$")
RE_TAG = re.compile(r"^Tag:.*?\bis_tag_ecosystem=(true|false)\b.*$", re.IGNORECASE)
RE_CRC = re.compile(r"^CRC\s+ok:\s*(TRUE|FALSE)\b.*$", re.IGNORECASE)
RE_PAYHEX = re.compile(r"^payload\s+\(hex\):\s*([0-9A-Fa-f]+)\s*$")


def ts_to_seconds(hh: int, mm: int, ss: int, frac: str) -> float:
    base = hh * 3600 + mm * 60 + ss
    if frac is None:
        return float(base)
    # Normalize fraction to microseconds precision (pad/right-trim)
    frac_norm = (frac + "000000")[:6]
    return base + int(frac_norm) / 1_000_000.0


def parse_dump(path: str) -> List[Row]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    rows: List[Row] = []

    # State for relative timestamps (handle day wrap)
    t0_abs: Optional[float] = None
    last_abs: Optional[float] = None
    day_offset = 0.0  # seconds

    cur: Dict[str, object] = {}

    def finalize_current():
        nonlocal cur
        if not cur:
            return

        # Mandatory-ish fields
        adva = str(cur.get("AdvA", ""))
        payhex = str(cur.get("payload", ""))

        # If we don't even have an AdvA/payload, skip
        if not adva or not payhex:
            cur = {}
            return

        # timestamp (relative)
        ts_rel = float(cur.get("timestamp_rel", 0.0))

        # Fill defaults for missing fields
        row = Row(
            timestamp=ts_rel,
            AdvA=adva,
            payload=payhex,
            CFO_Hz=cur.get("CFO_Hz", None),
            CFO_00_Hz=cur.get("CFO_00_Hz", None),
            CFO_11_Hz=cur.get("CFO_11_Hz", None),
            CFO_10_Hz=cur.get("CFO_10_Hz", None),
            CFO_01_Hz=cur.get("CFO_01_Hz", None),
            CFO_from_transitions_Hz=float("nan"),  # not present in dump
            nprod_00=0,
            nprod_11=0,
            nprod_10=0,
            nprod_01=0,
            nprod_total=0,
            crc_ok=int(cur.get("crc_ok", 0)),
            is_tag_ecosystem=int(cur.get("is_tag_ecosystem", 0)),
            pdu_type=int(cur.get("pdu_type", -1)),
            pdu_type_name=str(cur.get("pdu_type_name", "")),
            length=int(cur.get("length", -1)),
            channel=int(cur.get("channel", -1)),
            phase=int(cur.get("phase", -1)),
            polarity=int(cur.get("polarity", 0)),
            slip=int(cur.get("slip", 0)),
            window_start=-1,
            window_end=-1,
            aa_pos=int(cur.get("aa_pos", -1)),
            aa_corr=int(cur.get("aa_corr", -1)),
            pre_corr=int(cur.get("pre_corr", -1)),
            cfo_window="unknown",
        )
        rows.append(row)
        cur = {}

    for ln in lines:
        ln_stripped = ln.strip()

        if ln_stripped.startswith("---- BLE PACKET ----"):
            # Start of a new block
            finalize_current()
            continue

        m = RE_TS.match(ln_stripped)
        if m:
            hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            frac = m.group(4)
            abs_s = ts_to_seconds(hh, mm, ss, frac)
            # handle day wrap
            if last_abs is not None and abs_s + day_offset < last_abs:
                day_offset += 24.0 * 3600.0
            abs_s += day_offset
            last_abs = abs_s
            if t0_abs is None:
                t0_abs = abs_s
            cur["timestamp_rel"] = abs_s - t0_abs
            continue

        m = RE_AA.match(ln_stripped)
        if m:
            cur["aa_corr"] = int(m.group(1))
            cur["aa_pos"] = int(m.group(2))
            cur["pre_corr"] = int(m.group(3))
            continue

        m = RE_CH.match(ln_stripped)
        if m:
            cur["channel"] = int(m.group(1))
            cur["phase"] = int(m.group(2))
            cur["polarity"] = int(m.group(3))
            continue

        m = RE_SLIP.match(ln_stripped)
        if m:
            cur["slip"] = int(m.group(1))
            continue

        m = RE_CFO.match(ln_stripped)
        if m:
            cur["CFO_Hz"] = float(m.group(1))
            cur["CFO_00_Hz"] = float(m.group(2))
            cur["CFO_11_Hz"] = float(m.group(3))
            cur["CFO_10_Hz"] = float(m.group(4))
            cur["CFO_01_Hz"] = float(m.group(5))
            continue

        m = RE_PDU.match(ln_stripped)
        if m:
            cur["pdu_type_name"] = m.group(1).strip()
            cur["pdu_type"] = int(m.group(2))
            continue

        m = RE_LEN.match(ln_stripped)
        if m:
            cur["length"] = int(m.group(1))
            continue

        m = RE_ADVA.match(ln_stripped)
        if m:
            cur["AdvA"] = m.group(1).upper()
            continue

        m = RE_TAG.match(ln_stripped)
        if m:
            cur["is_tag_ecosystem"] = 1 if m.group(1).lower() == "true" else 0
            continue

        m = RE_CRC.match(ln_stripped)
        if m:
            cur["crc_ok"] = 1 if m.group(1).upper() == "TRUE" else 0
            continue

        m = RE_PAYHEX.match(ln_stripped)
        if m:
            cur["payload"] = m.group(1).lower()
            continue

    # finalize last
    finalize_current()
    return rows


def write_csv(rows: List[Row], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            # Print NaN as empty
            if isinstance(d.get("CFO_from_transitions_Hz"), float) and math.isnan(d["CFO_from_transitions_Hz"]):
                d["CFO_from_transitions_Hz"] = ""
            w.writerow({k: d.get(k, "") for k in CSV_COLUMNS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input .txt dump file")
    ap.add_argument("--out", dest="out", required=True, help="Output .csv path")
    args = ap.parse_args()

    rows = parse_dump(args.inp)
    if not rows:
        raise SystemExit("No packets parsed. Check that the file contains '---- BLE PACKET ----' blocks.")

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()