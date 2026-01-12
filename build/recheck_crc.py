#!/usr/bin/env python3
"""
recheck_crc.py — CRC recheck + bit repair attempts for BLE legacy adv packets.

Reads BOTH:
  A) "---- BLE PACKET ----" block logs (like airtag_packets.txt)
  B) JSONL records (one JSON object per line)

Preset C (ALL methods):
  - GLOBAL_SLIP_SHIFTCRC (±1..±3, all bit patterns), CRC field = last 24 bits
  - INS_SHIFTCRC         (insert 1 bit anywhere in FULL bitstream, drop last bit), CRC field = last 24 bits
  - DEL_SHIFTCRC         (delete 1 bit anywhere in FULL bitstream, append 1 bit), CRC field = last 24 bits
  - INS_TRUST_KEEPCRC    (insert 1 bit into header+payload, then drop 1 bit at end of header+payload),
                          CRC bits unchanged, compare to trusted/original CRC
  - DEL_TRUST_KEEPCRC    (delete 1 bit from header+payload, then append 1 bit at end of header+payload),
                          CRC bits unchanged, compare to trusted/original CRC

Outputs:
  --out-ok    JSONL of packets that end CRC_OK (orig ok + fixed)
  --out-fixed JSONL of packets that were originally CRC_BAD but got fixed
"""

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Tuple, Optional, List

import numpy as np


# ----------------------------
# BLESDR SwapBits + reverse_crc (exactly like your sniffer)
# ----------------------------
_SWAP_LUT = np.zeros(256, dtype=np.uint8)
for _v in range(256):
    x = _v
    y = 0
    for _ in range(8):
        y = (y << 1) | (x & 1)
        x >>= 1
    _SWAP_LUT[_v] = y

def swap_bits8(v: int) -> int:
    return int(_SWAP_LUT[v & 0xFF])

def blesdr_reverse_crc(data_msb: bytes, init_adv: bool = True) -> int:
    dst0 = 0x55 if init_adv else 0x00
    dst1 = 0x55 if init_adv else 0x00
    dst2 = 0x55 if init_adv else 0x00

    for byte in data_msb:
        d = swap_bits8(byte)
        for _ in range(8):
            t = (dst0 >> 7) & 1

            dst0 = ((dst0 << 1) & 0xFF) | ((dst1 >> 7) & 1)
            dst1 = ((dst1 << 1) & 0xFF) | ((dst2 >> 7) & 1)
            dst2 = ((dst2 << 1) & 0xFF)

            if t != (d & 1):
                dst2 ^= 0x5B
                dst1 ^= 0x06
            d >>= 1

    return ((dst0 << 16) | (dst1 << 8) | dst2) & 0xFFFFFF


# ----------------------------
# Bits/bytes helpers (MSB-in-time domain)
# ----------------------------
def bits_from_msb_bytes(pkt_msb: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(pkt_msb, dtype=np.uint8), bitorder="big").astype(np.uint8, copy=False)

def msb_bytes_from_bits(bits: np.ndarray) -> bytes:
    bb = np.packbits(bits.astype(np.uint8, copy=False), bitorder="big")
    return bb.tobytes()

def crc_field_from_last24_bits(bits: np.ndarray) -> int:
    last24 = bits[-24:]
    b = msb_bytes_from_bits(last24)
    if len(b) != 3:
        return -1
    return int.from_bytes(b, "big") & 0xFFFFFF

def trusted_crc_from_record(rec: Dict[str, Any], pkt_msb: bytes) -> int:
    hx = rec.get("crc_rx_msb_hex", None)
    if isinstance(hx, str) and len(hx.strip()) >= 6:
        try:
            return int(hx.strip()[:6], 16) & 0xFFFFFF
        except Exception:
            pass
    # fallback: last 3 bytes of pkt_msb
    if len(pkt_msb) >= 3:
        return int.from_bytes(pkt_msb[-3:], "big") & 0xFFFFFF
    return -1


# ----------------------------
# Input parsers
# ----------------------------
def parse_jsonl_records(text: str) -> Tuple[List[Dict[str, Any]], Counter]:
    recs = []
    reasons = Counter()
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if not (ln.startswith("{") and ln.endswith("}")):
            reasons["non_json_line"] += 1
            continue
        try:
            recs.append(json.loads(ln))
        except Exception:
            reasons["json_parse_error"] += 1
    return recs, reasons

_BLOCK_SPLIT = "---- BLE PACKET ----"

_re_pdu = re.compile(r"PDU:\s*(?P<name>.+?)\s*\(type=(?P<type>\d+)\)")
_re_len = re.compile(r"Len:\s*(?P<len>\d+)\s+TxAdd:\s*(?P<tx>\d+)\s+RxAdd:\s*(?P<rx>\d+)")
_re_crc = re.compile(r"CRC rx:\s*0x(?P<crc>[0-9a-fA-F]{6,8})")
_re_payload_hex = re.compile(r"payload\s*\(hex\):\s*(?P<hx>[0-9a-fA-F]+)")

def parse_block_records(text: str) -> Tuple[List[Dict[str, Any]], Counter]:
    recs = []
    reasons = Counter()
    if _BLOCK_SPLIT not in text:
        reasons["no_block_marker"] += 1
        return recs, reasons

    blocks = text.split(_BLOCK_SPLIT)
    for b in blocks:
        b = b.strip()
        if not b:
            continue

        m_pdu = _re_pdu.search(b)
        m_len = _re_len.search(b)
        m_crc = _re_crc.search(b)
        m_pay = _re_payload_hex.search(b)

        if not m_pdu:
            reasons["missing_pdu_line"] += 1
            continue
        if not m_len:
            reasons["missing_len_line"] += 1
            continue
        if not m_crc:
            reasons["missing_crc_rx"] += 1
            continue
        if not m_pay:
            reasons["missing_payload_hex"] += 1
            continue

        try:
            pdu_type = int(m_pdu.group("type"))
            length = int(m_len.group("len"))
            txadd = int(m_len.group("tx"))
            rxadd = int(m_len.group("rx"))
            crc_rx = int(m_crc.group("crc")[:6], 16) & 0xFFFFFF
            payload_std = bytes.fromhex(m_pay.group("hx"))

            # Rebuild STANDARD header bytes
            h0_std = (pdu_type & 0x0F) | ((txadd & 1) << 6) | ((rxadd & 1) << 7)
            h1_std = (length & 0x3F)

            # Convert to MSB-time (BLESDR ExtractByte domain): swap_bits8()
            hdr_msb = bytes([swap_bits8(h0_std), swap_bits8(h1_std)])
            payload_msb = bytes(swap_bits8(x) for x in payload_std)
            crc_msb = crc_rx.to_bytes(3, "big")

            pkt_msb = hdr_msb + payload_msb + crc_msb
            recs.append({
                "source_format": "block",
                "pdu_type": pdu_type,
                "length": length,
                "txadd": txadd,
                "rxadd": rxadd,
                "payload_std_hex": payload_std.hex(),
                "crc_rx_msb_hex": f"{crc_rx:06x}",
                "pkt_msb_dewhitened_hex": pkt_msb.hex(),  # reconstructed dewhitened msb-domain bytes
            })
        except Exception:
            reasons["block_reconstruct_error"] += 1

    return recs, reasons

def parse_record_to_bits_and_pkt_msb(rec: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[bytes], Optional[str]]:
    # Preferred: pkt_msb_dewhitened_hex
    hx = rec.get("pkt_msb_dewhitened_hex", None)
    if isinstance(hx, str) and len(hx.strip()) >= 10:
        try:
            pkt_msb = bytes.fromhex(hx.strip())
            if len(pkt_msb) < 5:
                return None, None, "pkt_msb_dewhitened_hex too short"
            return bits_from_msb_bytes(pkt_msb), pkt_msb, None
        except Exception as e:
            return None, None, f"bad pkt_msb_dewhitened_hex: {e}"

    # Alternative: pkt_bits_msb
    s = rec.get("pkt_bits_msb", None)
    if isinstance(s, str) and s and all(c in "01" for c in s.strip()):
        s = s.strip()
        if (len(s) % 8) != 0:
            return None, None, "pkt_bits_msb length not multiple of 8"
        bits = np.fromiter((1 if c == "1" else 0 for c in s), dtype=np.uint8)
        pkt_msb = msb_bytes_from_bits(bits)
        if len(pkt_msb) < 5:
            return None, None, "pkt_bits_msb too short after packbits"
        return bits, pkt_msb, None

    return None, None, "missing pkt_msb_dewhitened_hex or pkt_bits_msb"


# ----------------------------
# Checks
# ----------------------------
def _check_shiftcrc(bits: np.ndarray) -> bool:
    if bits.size < 40 or (bits.size % 8) != 0:
        return False
    hp_bits = bits[:-24]
    if (hp_bits.size % 8) != 0:
        return False
    hp_msb = msb_bytes_from_bits(hp_bits)
    crc_calc = blesdr_reverse_crc(hp_msb, init_adv=True)
    crc_field = crc_field_from_last24_bits(bits)
    return (crc_field >= 0) and (crc_calc == crc_field)

def _check_trust(bits: np.ndarray, trusted_crc: int) -> bool:
    if bits.size < 40 or (bits.size % 8) != 0:
        return False
    hp_bits = bits[:-24]
    if (hp_bits.size % 8) != 0:
        return False
    hp_msb = msb_bytes_from_bits(hp_bits)
    crc_calc = blesdr_reverse_crc(hp_msb, init_adv=True)
    return (trusted_crc >= 0) and (crc_calc == trusted_crc)


# ----------------------------
# Repair methods (Preset C)
# ----------------------------
def try_global_slip_shiftcrc(bits: np.ndarray, kmax: int = 3) -> Optional[Tuple[str, np.ndarray]]:
    n = bits.size
    if n < 40:
        return None
    for k in range(1, kmax + 1):
        patterns = []
        for v in range(1 << k):
            pat = np.array([(v >> (k - 1 - i)) & 1 for i in range(k)], dtype=np.uint8)
            patterns.append(pat)

        for pat in patterns:  # +k
            b2 = np.concatenate([pat, bits[: n - k]])
            if _check_shiftcrc(b2):
                return (f"GLOBAL_SLIP_SHIFTCRC+{k}", b2)

        for pat in patterns:  # -k
            b2 = np.concatenate([bits[k:], pat])
            if _check_shiftcrc(b2):
                return (f"GLOBAL_SLIP_SHIFTCRC-{k}", b2)
    return None

def try_ins_shiftcrc(bits: np.ndarray) -> Optional[Tuple[str, np.ndarray]]:
    n = bits.size
    if n < 40:
        return None
    for pos in range(0, n):
        prefix = bits[:pos]
        suffix = bits[pos:]
        for b in (0, 1):
            b2 = np.concatenate([prefix, np.array([b], dtype=np.uint8), suffix])
            b2 = b2[:n]  # drop last
            if _check_shiftcrc(b2):
                return ("INS_SHIFTCRC", b2)
    return None

def try_del_shiftcrc(bits: np.ndarray) -> Optional[Tuple[str, np.ndarray]]:
    n = bits.size
    if n < 40:
        return None
    for pos in range(0, n):
        prefix = bits[:pos]
        suffix = bits[pos+1:]
        for tail in (0, 1):
            b2 = np.concatenate([prefix, suffix, np.array([tail], dtype=np.uint8)])
            if b2.size == n and _check_shiftcrc(b2):
                return ("DEL_SHIFTCRC", b2)
    return None

def try_ins_trust_keepcrc(bits: np.ndarray, trusted_crc: int) -> Optional[Tuple[str, np.ndarray]]:
    n = bits.size
    if n < 40:
        return None
    hp = bits[:-24]
    crc_bits = bits[-24:]
    for pos in range(0, hp.size + 1):
        prefix = hp[:pos]
        suffix = hp[pos:]
        for b in (0, 1):
            hp2 = np.concatenate([prefix, np.array([b], dtype=np.uint8), suffix])
            hp2 = hp2[:hp.size]  # drop at end of HP; CRC untouched
            b2 = np.concatenate([hp2, crc_bits])
            if _check_trust(b2, trusted_crc):
                return ("INS_TRUST_KEEPCRC", b2)
    return None

def try_del_trust_keepcrc(bits: np.ndarray, trusted_crc: int) -> Optional[Tuple[str, np.ndarray]]:
    n = bits.size
    if n < 40:
        return None
    hp = bits[:-24]
    crc_bits = bits[-24:]
    for pos in range(0, hp.size):
        prefix = hp[:pos]
        suffix = hp[pos+1:]
        for tail in (0, 1):
            hp2 = np.concatenate([prefix, suffix, np.array([tail], dtype=np.uint8)])
            if hp2.size != hp.size:
                continue
            b2 = np.concatenate([hp2, crc_bits])
            if _check_trust(b2, trusted_crc):
                return ("DEL_TRUST_KEEPCRC", b2)
    return None


# ----------------------------
# Worker
# ----------------------------
def process_one(rec: Dict[str, Any]) -> Tuple[bool, bool, Optional[str], Optional[str], str, Optional[str]]:
    """
    Returns:
      (usable, orig_ok, ok_jsonl_line, fixed_jsonl_line, method, unusable_reason)
    """
    bits, pkt_msb, err = parse_record_to_bits_and_pkt_msb(rec)
    if err is not None or bits is None or pkt_msb is None:
        return (False, False, None, None, "UNUSABLE", err or "unknown")

    trusted_crc = trusted_crc_from_record(rec, pkt_msb)

    orig_ok = _check_trust(bits, trusted_crc)
    if orig_ok:
        rec2 = dict(rec)
        rec2["_recheck_orig_crc_ok"] = True
        rec2["_recheck_method"] = "ORIG_OK"
        if "pkt_msb_dewhitened_hex" not in rec2:
            rec2["pkt_msb_dewhitened_hex"] = pkt_msb.hex()
        ok_line = json.dumps(rec2, separators=(",", ":")) + "\n"
        return (True, True, ok_line, None, "ORIG_OK", None)

    # CRC_BAD => try Preset C in order
    got = try_global_slip_shiftcrc(bits, kmax=3)
    if got is None:
        got = try_ins_shiftcrc(bits)
    if got is None:
        got = try_del_shiftcrc(bits)
    if got is None:
        got = try_ins_trust_keepcrc(bits, trusted_crc)
    if got is None:
        got = try_del_trust_keepcrc(bits, trusted_crc)

    if got is not None:
        method, bfix = got
        rec2 = dict(rec)
        rec2["_recheck_orig_crc_ok"] = False
        rec2["_recheck_method"] = method
        rec2["_recheck_final_crc_ok"] = True
        rec2["pkt_bits_msb_repaired"] = "".join("1" if x else "0" for x in bfix.tolist())
        rec2["pkt_msb_dewhitened_hex_repaired"] = msb_bytes_from_bits(bfix).hex()
        ok_line = json.dumps(rec2, separators=(",", ":")) + "\n"
        return (True, False, ok_line, ok_line, method, None)

    return (True, False, None, None, "STILL_BAD", None)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input file: block log or JSONL")
    ap.add_argument("--out-ok", default="crc_ok.jsonl", help="Output JSONL of CRC_OK packets (orig ok + fixed)")
    ap.add_argument("--out-fixed", default="crc_fixed.jsonl", help="Output JSONL of fixed packets (orig bad -> ok)")
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores, 1 = no MP")
    ap.add_argument("--chunksize", type=int, default=64, help="ProcessPool chunksize")
    args = ap.parse_args()

    workers = args.workers
    if workers <= 0:
        workers = os.cpu_count() or 1
    if workers < 1:
        workers = 1

    text = open(args.inp, "r", errors="replace").read()
    print(f"[INFO] Parsed blocks: {len(text.splitlines())}")

    # Auto-detect format
    if _BLOCK_SPLIT in text:
        recs, parse_reasons = parse_block_records(text)
        fmt = "block"
    else:
        recs, parse_reasons = parse_jsonl_records(text)
        fmt = "jsonl"

    print(f"[INFO] Detected input format: {fmt}")
    print(f"[INFO] Usable records parsed from file: {len(recs)}")
    if parse_reasons:
        top = parse_reasons.most_common(8)
        print("[INFO] Parse notes (top):", top)

    usable = 0
    orig_ok = 0
    orig_bad = 0
    fixed_total = 0
    fixed_by = Counter()
    final_ok = 0
    final_bad = 0
    unusable_reasons = Counter()

    print(f"[INFO] Methods enabled: Preset C (ALL)")
    print(f"[INFO] workers={workers} chunksize={args.chunksize}")

    out_ok_f = open(args.out_ok, "w")
    out_fx_f = open(args.out_fixed, "w")

    try:
        if workers == 1:
            it = (process_one(r) for r in recs)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                it = ex.map(process_one, recs, chunksize=max(1, int(args.chunksize)))

        for (u, o_ok, ok_line, fx_line, method, u_reason) in it:
            if not u:
                unusable_reasons[u_reason or "unknown"] += 1
                continue

            usable += 1
            if o_ok:
                orig_ok += 1
            else:
                orig_bad += 1

            if ok_line is not None:
                out_ok_f.write(ok_line)
                final_ok += 1
            else:
                final_bad += 1

            if fx_line is not None:
                out_fx_f.write(fx_line)
                fixed_total += 1
                fixed_by[method] += 1

    finally:
        out_ok_f.close()
        out_fx_f.close()

    def pct(x, d):
        return 0.0 if d <= 0 else 100.0 * x / float(d)

    print("[STATS] CRC recheck + Preset C repairs")
    print(f"  Total usable packets                       : {usable}")
    print(f"  Original CRC OK (recomputed)               : {orig_ok} ({pct(orig_ok, usable):.2f}%)")
    print(f"  Original CRC BAD                           : {orig_bad} ({pct(orig_bad, usable):.2f}%)")

    if fixed_by:
        for m, c in fixed_by.most_common():
            print(f"  Fixed by {m:<30} : {c} ({pct(c, usable):.2f}%)")
    else:
        print("  Fixed by (any enabled method)              : 0 (0.00%)")

    print(f"  Fixed total (any enabled method)           : {fixed_total} ({pct(fixed_total, usable):.2f}%)")
    print(f"  Final CRC OK (orig_ok + fixed)             : {final_ok} ({pct(final_ok, usable):.2f}%)")
    print(f"  Still CRC BAD after enabled fixes          : {final_bad} ({pct(final_bad, usable):.2f}%)")
    print(f"[INFO] Wrote: {args.out_ok} (CRC_OK packets), {args.out_fixed} (fixed-only)")

    if unusable_reasons:
        print("[INFO] Unusable-record reasons (top):", unusable_reasons.most_common(10))

if __name__ == "__main__":
    main()