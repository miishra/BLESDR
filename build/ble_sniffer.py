#!/usr/bin/env python3
"""
ble_sniffer.py — BLE 1M legacy advertising sniffer for raw IQ streams (SPI int16 IQ or CF32).

FAST + STATS + CFO (preamble->AA->header+payload; CRC excluded from CFO window):
  - Resamples ONCE for the whole capture
  - Computes discriminator ONCE
  - Fast AA correlation using np.correlate
  - Reports window-level percentages (preamble / AA / both / parsed / CRC)
  - Uses BLESDR-compatible dewhitening + CRC in the SAME byte space as BLESDR:
        * build "MSB-first" bytes from on-air bits (matches BLESDR ExtractByte domain)
        * btle_reverse_whiten(): lfsr = SwapBits(chan) | 2, poly 0x11, MSB stepping
        * btle_reverse_crc(): BLESDR reverse CRC (init 0x555555 for adv)
  - CFO estimate (Hz) for candidates that have:
        preamble match + AA match + parsable header+payload
    CFO window:
      preamble (8) + AA (32) + header (16) + payload (8*len)
    (CRC bits excluded from CFO estimation)
  - Groups CFO by AdvA ONLY (regardless of CRC) and saves a boxplot to PDF.
  - Optional AirTag/FindMy + “tag ecosystem” detection and filtering.

DEBUGS ADDED (minimal changes, opt-in):
  - --slip-sweep : expands bit-slip search around AA boundary to ±slip-max bits
  - --crc-diag   : prints deterministic diagnostics when AA+preamble matches but CRC fails
                 and prints histograms (slip/phase/polarity/channel) for CRC-OK outcomes.
  - --auto-fs-scan : scans fs-in around the provided value to see if CRC is killed by fs/SPS mismatch

CHANGE (minimal):
  - Instead of energy-based burst detection, we scan continuously for AA matches and build
    small decode windows around those candidates.
    (The --thr parameter is now ignored.)
"""

import os
import argparse
from fractions import Fraction
from collections import defaultdict, Counter

import numpy as np
from scipy.signal import resample_poly
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

FS_IN_DEFAULT = 1344106.900141  # 1_365_333.33
FS_OUT_TARGET = 4_000_000
SYMBOL_RATE = 1_000_000

# SPI framing (EFR32-style)
SPI_CHUNK_BYTES = 2048
SPI_SKIP_BYTES = 0

# Advertising Access Address
AA_ADV = 0x8E89BED6  # fixed for adv channels

# BLE legacy adv PDU type names
PDU_TYPE_NAMES = {
    0x0: "ADV_IND",
    0x1: "ADV_DIRECT_IND",
    0x2: "ADV_NONCONN_IND",
    0x3: "SCAN_REQ",
    0x4: "SCAN_RSP",
    0x5: "CONNECT_IND",
    0x6: "ADV_SCAN_IND",
}

# Burst detector defaults (kept for CLI compatibility; not used in continuous scan mode)
DEFAULT_THR_K = 8.0
DEFAULT_SMOOTH_US = 50.0
DEFAULT_MIN_LEN_US = 80.0
DEFAULT_GAP_US = 30.0
DEFAULT_PRE_US = 140.0
DEFAULT_POST_US = 220.0


# ============================================================
# BIT ORDER HELPERS (on-air)
# ============================================================

def bytes_to_bits_lsbfirst(b: bytes) -> np.ndarray:
    """On-air bit order: LSB-first per octet, octets in given order."""
    out = np.empty(len(b) * 8, dtype=np.uint8)
    k = 0
    for byte in b:
        for i in range(8):
            out[k] = (byte >> i) & 1
            k += 1
    return out

def bits_to_u8_lsbfirst(bits8: np.ndarray) -> int:
    v = 0
    for i in range(8):
        v |= (int(bits8[i]) & 1) << i
    return v

def bits_to_bytes_lsbfirst(bits: np.ndarray) -> bytes:
    n = (bits.size // 8) * 8
    bits = bits[:n]
    out = bytearray()
    for i in range(0, n, 8):
        out.append(bits_to_u8_lsbfirst(bits[i:i+8]))
    return bytes(out)

def fmt_mac(b: bytes) -> str:
    # BLE addresses are typically displayed MSB-first (reverse of on-air/payload byte order)
    b = b[::-1]
    return ":".join(f"{x:02X}" for x in b)


# ============================================================
# BUILD AA BIT SEQUENCE IN *ON-AIR* ORDER
# ============================================================

def aa_bits_onair(aa_u32: int) -> np.ndarray:
    aa_bytes_le = aa_u32.to_bytes(4, byteorder="little", signed=False)
    return bytes_to_bits_lsbfirst(aa_bytes_le)

AA_BITS = aa_bits_onair(AA_ADV)
AA_PM = (AA_BITS.astype(np.int8) * 2 - 1)  # 0->-1, 1->+1

def correlate_access_address_fast(bits: np.ndarray):
    """Fast AA correlation using dot-product over +/-1 sequences."""
    if bits.size < 32:
        return None, -1
    b_pm = bits.astype(np.int8) * 2 - 1
    dots = np.correlate(b_pm, AA_PM, mode="valid")  # [-32, +32]
    i = int(np.argmax(dots))
    best_dot = int(dots[i])
    best_matches = (best_dot + 32) // 2
    return i, int(best_matches)


# ============================================================
# PREAMBLE (8 bits) CORRELATION (aligned to AA)
# ============================================================

AA_FIRST_BIT = int(AA_BITS[0])
PRE_BITS = (np.array([0, 1] * 4, dtype=np.uint8)
            if AA_FIRST_BIT == 0 else
            np.array([1, 0] * 4, dtype=np.uint8))

def preamble_corr_at(bits: np.ndarray, aa_pos: int) -> int:
    if aa_pos is None or aa_pos < 8:
        return -1
    return int(np.sum(bits[aa_pos-8:aa_pos] == PRE_BITS))


# ============================================================
# LOADERS
# ============================================================

def load_cf32(filename: str) -> np.ndarray:
    print(f"[INFO] Loading CF32 IQ from {filename}")
    iq = np.fromfile(filename, dtype=np.complex64)
    if iq.size == 0:
        raise ValueError("CF32 file empty/unreadable.")
    iq = iq - np.mean(iq)
    return iq

def load_spi_int16_iq(
    filename: str,
    spi_chunk_bytes: int = SPI_CHUNK_BYTES,
    skip_bytes: int = SPI_SKIP_BYTES,
) -> np.ndarray:
    print(f"[INFO] Loading RAW SPI int16 IQ from {filename}")
    raw = np.fromfile(filename, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError("SPI file empty/unreadable.")

    n_chunks = raw.size // spi_chunk_bytes
    if n_chunks == 0:
        raise ValueError("SPI file smaller than one chunk; check chunk parameters.")
    raw = raw[: n_chunks * spi_chunk_bytes]

    payload = []
    for i in range(n_chunks):
        s = i * spi_chunk_bytes + skip_bytes
        e = (i + 1) * spi_chunk_bytes
        payload.append(raw[s:e])
    payload = np.concatenate(payload) if payload else np.array([], dtype=np.uint8)

    if payload.size < 4:
        raise ValueError("Not enough payload after de-framing.")

    if payload.size % 2 != 0:
        payload = payload[:-1]

    words = payload.view("<i2")  # little-endian int16
    I = words[0::2].astype(np.float32) / 32768.0
    Q = words[1::2].astype(np.float32) / 32768.0
    iq = (I + 1j * Q).astype(np.complex64)
    iq = iq - np.mean(iq)
    return iq

def load_iq_auto(filename: str) -> np.ndarray:
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".cf32", ".cfile"]:
        return load_cf32(filename)
    return load_spi_int16_iq(filename)


# ============================================================
# RESAMPLER
# ============================================================

def pick_resample_ratio(fs_in: float, fs_out: float, max_den: int = 4096):
    frac = Fraction(fs_out / fs_in).limit_denominator(max_den)
    return frac.numerator, frac.denominator

def upsample(iq: np.ndarray, L: int, M: int) -> np.ndarray:
    return resample_poly(iq, L, M).astype(np.complex64)


# ============================================================
# DISCRIMINATOR
# ============================================================

def gfsk_discriminator(iq: np.ndarray) -> np.ndarray:
    prod = iq[1:] * np.conj(iq[:-1])
    return np.angle(prod).astype(np.float32)


# ============================================================
# CONTINUOUS AA SCAN -> WINDOWS
# ============================================================

def find_decode_windows_continuous(
    freq_all: np.ndarray,
    sps: int,
    aa_corr_min: int,
    preamble_min: int,
    slip_max: int,
    max_windows: int = 0,
    debug: bool = False,
):
    """
    Continuously scan symbol streams (for every phase and polarity) for AA matches,
    then build small sample-index windows around each AA candidate.

    Returns:
      windows: list of (s_up, e_up) in freq_all sample indices (upsampled domain)
    """
    if sps <= 0:
        return []

    # Conservative window sizes (symbols) around AA start
    MAX_PAYLOAD = 37
    MAX_TOTAL_BITS = 8 + 32 + (2 + MAX_PAYLOAD + 3) * 8  # 376 bits @1Mbps
    pre_symbols = max(48, 8 + 32 + 8 + abs(int(slip_max)))  # enough to re-find preamble+AA after re-phasing
    post_symbols = MAX_TOTAL_BITS + 32 + abs(int(slip_max))  # header+payload+crc + margin

    # Gather candidates as (aa_sample, score_tuple)
    cand = []

    # Scan per phase/polarity over symbol domain (length ~ len(freq_all)/sps)
    for phase in range(sps):
        sym = freq_all[phase::sps]
        if sym.size < 64:
            continue

        for polarity in (+1, -1):
            x = sym if polarity > 0 else -sym
            bits = (x > 0).astype(np.uint8)

            # AA correlation for entire stream
            b_pm = bits.astype(np.int8) * 2 - 1
            dots = np.correlate(b_pm, AA_PM, mode="valid")
            # convert to "matches"
            matches = ((dots + 32) // 2).astype(np.int16)

            idx = np.flatnonzero(matches >= aa_corr_min)
            if idx.size == 0:
                continue

            # preamble gate
            for aa_pos in idx.tolist():
                pre_corr = preamble_corr_at(bits, aa_pos)
                if pre_corr < preamble_min:
                    continue

                aa_sample = int(phase + aa_pos * sps)  # freq_all sample index at AA start (approx)
                sc = (int(matches[aa_pos]), int(pre_corr), int(phase), int(polarity))
                cand.append((aa_sample, sc))

    if not cand:
        return []

    # Sort by sample location
    cand.sort(key=lambda t: t[0])

    # Non-maximum suppression to deduplicate near-duplicates across phase/polarity
    # Use a separation ~ half a max packet to avoid multiple picks inside same packet
    min_sep_samples = int((MAX_TOTAL_BITS * sps) // 2)
    kept = []

    for aa_sample, sc in cand:
        if not kept:
            kept.append([aa_sample, sc])
            continue

        prev_sample, prev_sc = kept[-1]
        if aa_sample - prev_sample <= min_sep_samples:
            # keep the better score within the cluster
            if sc > prev_sc:
                kept[-1] = [aa_sample, sc]
        else:
            kept.append([aa_sample, sc])

    if max_windows and max_windows > 0:
        kept = kept[:max_windows]

    # Build decode windows in sample domain
    windows = []
    for aa_sample, _sc in kept:
        s_up = max(0, aa_sample - pre_symbols * sps)
        e_up = min(freq_all.size, aa_sample + post_symbols * sps)
        if e_up - s_up >= (64 * sps):
            windows.append((int(s_up), int(e_up)))

    if debug:
        print(f"[DEBUG] Continuous scan candidates={len(cand)} kept={len(windows)} "
              f"(pre_symbols={pre_symbols}, post_symbols={post_symbols}, min_sep_samples={min_sep_samples})")

    return windows


# ============================================================
# BLESDR SwapBits, Whitening, CRC (BYTE DOMAIN)
# ============================================================

# Precompute SwapBits(0..255)
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

def blesdr_reverse_whiten(chan: int, data: bytearray) -> None:
    """
    Exact BLESDR logic:

        lfsr = SwapBits(chan) | 2;
        for each byte:
            for mask = 0x80..0x01:
                if (lfsr & 0x80) { lfsr ^= 0x11; byte ^= mask; }
                lfsr <<= 1;

    This assumes 'data' bytes are in BLESDR's "ExtractByte" domain (MSB-first per byte).
    """
    lfsr = (swap_bits8(chan) | 0x02) & 0xFF
    for i in range(len(data)):
        b = data[i]
        mask = 0x80
        while mask:
            if (lfsr & 0x80) != 0:
                lfsr ^= 0x11
                b ^= mask
            lfsr = ((lfsr << 1) & 0xFF)
            mask >>= 1
        data[i] = b

def blesdr_reverse_crc(data: bytes, init_adv: bool = True) -> int:
    """
    Exact BLESDR btle_reverse_crc() behavior.

    init for advertising uses dst = [0x55,0x55,0x55], then shifts left.
    Returns 24-bit integer assembled big-endian from dst bytes.
    """
    dst0 = 0x55 if init_adv else 0x00
    dst1 = 0x55 if init_adv else 0x00
    dst2 = 0x55 if init_adv else 0x00

    for byte in data:
        d = swap_bits8(byte)
        for _ in range(8):
            t = (dst0 >> 7) & 1

            # shift left dst0..2
            dst0 = ((dst0 << 1) & 0xFF) | ((dst1 >> 7) & 1)
            dst1 = ((dst1 << 1) & 0xFF) | ((dst2 >> 7) & 1)
            dst2 = ((dst2 << 1) & 0xFF)

            if t != (d & 1):
                dst2 ^= 0x5B
                dst1 ^= 0x06
            d >>= 1

    return ((dst0 << 16) | (dst1 << 8) | dst2) & 0xFFFFFF


# ============================================================
# Convert on-air bits -> BLESDR "ExtractByte" domain bytes
# ============================================================

def bits_to_bytes_msbfirst_time(bits: np.ndarray) -> bytes:
    """
    Build bytes so that the *first* bit in time becomes bit7 (MSB) of the byte,
    next becomes bit6, ..., last becomes bit0.

    This matches BLESDR's ExtractByte(): byte |= Q(l+c) << (7-c)
    under the assumption Q(l),Q(l+1),... enumerate bits in time order.
    """
    n = (bits.size // 8) * 8
    bits = bits[:n]
    out = bytearray(n // 8)
    bi = 0
    for i in range(len(out)):
        b = 0
        for k in range(8):
            b |= (int(bits[bi + k]) & 1) << (7 - k)
        out[i] = b
        bi += 8
    return bytes(out)


# ============================================================
# Parse (after dewhiten) in STANDARD BLE byte order (LSB-first)
# ============================================================

def parse_legacy_adv_from_blesdr_bytes(packet_data_msb: bytes):
    """
    packet_data_msb: header(2) + payload(length) + crc(3) all in BLESDR byte domain,
                     already dewhitened.

    We interpret header/payload by SwapBits() each byte to convert to normal on-air byte value.
    """
    if len(packet_data_msb) < 2 + 3:
        raise ValueError("Too short")

    packet_std = bytes(swap_bits8(b) for b in packet_data_msb)

    h0 = packet_std[0]
    h1 = packet_std[1]
    pdu_type = h0 & 0x0F
    txadd = (h0 >> 6) & 1
    rxadd = (h0 >> 7) & 1
    length = h1 & 0x3F

    if length > 60:
        raise ValueError(f"Legacy length invalid: {length}")

    need = 2 + length + 3
    if len(packet_std) < need:
        raise ValueError("Not enough bytes for payload+CRC")

    payload = packet_std[2:2+length]
    return h0, h1, pdu_type, txadd, rxadd, length, payload


def extract_addresses(pdu_type: int, payload: bytes):
    out = {}
    if pdu_type in [0x0, 0x2, 0x6, 0x4]:
        if len(payload) >= 6:
            out["AdvA"] = payload[0:6]
    elif pdu_type == 0x1:
        if len(payload) >= 12:
            out["AdvA"] = payload[0:6]
            out["TargetA"] = payload[6:12]
    elif pdu_type == 0x3:
        if len(payload) >= 12:
            out["ScanA"] = payload[0:6]
            out["AdvA"] = payload[6:12]
    elif pdu_type == 0x5:
        if len(payload) >= 12:
            out["InitA"] = payload[0:6]
            out["AdvA"] = payload[6:12]
    return out


# ============================================================
# Tag detection (AirTag/FindMy + other tag ecosystems)
# ============================================================

def _is_tag_service_uuid(uuid16: int) -> bool:
    return uuid16 in (0xFEAA, 0xFEED, 0xFD5A)

def is_findmy_or_tag_ecosystem(pdu_std_bytes: bytes):
    """
    pdu_std_bytes: header(2) + payload + crc(3) in STANDARD BLE byte values.
    Mirrors your C helper logic.
    Returns: (is_airtag_findmy, is_tag_ecosystem, reason_str)
    """
    if len(pdu_std_bytes) < 8:
        return False, False, ""

    payload_len = len(pdu_std_bytes) - 5
    if payload_len < 6:
        return False, False, ""

    payload = pdu_std_bytes[2:2+payload_len]
    ad_data = payload[6:]
    ad_len = len(ad_data)

    pos = 0
    while pos + 1 < ad_len:
        length = ad_data[pos]
        if length == 0:
            break
        if pos + 1 + length > ad_len:
            break

        ad_type = ad_data[pos + 1]

        if ad_type == 0xFF and length >= 3:
            if pos + 4 <= ad_len:
                company_id = ad_data[pos + 2] | (ad_data[pos + 3] << 8)
                if company_id == 0x004C and length >= 4:
                    if pos + 5 < ad_len:
                        findmy_prefix = ad_data[pos + 4]
                        if (findmy_prefix == 0x12) and (pos + 6 < ad_len) and (ad_data[pos + 5] == 0x19):
                            return True, True, "Apple 0x004C + 0x12 0x19"

        if ad_type == 0x16 and length >= 3:
            if pos + 4 <= ad_len:
                svc_uuid = ad_data[pos + 2] | (ad_data[pos + 3] << 8)
                if _is_tag_service_uuid(svc_uuid):
                    return False, True, f"ServiceData UUID 0x{svc_uuid:04X}"

        pos += 1 + length

    return False, False, ""


# ============================================================
# Debug helper: CRC comparison variants (to spot endian/domain mismatch)
# ============================================================

def crc_diag_variants(crc_rx: int, crc_calc: int):
    rx_b = crc_rx.to_bytes(3, "big")
    cc_b = crc_calc.to_bytes(3, "big")
    return {
        "rx": rx_b.hex(),
        "calc": cc_b.hex(),
        "rx==calc": (crc_rx == crc_calc),
        "rx==calc_byteswapped": (rx_b == cc_b[::-1]),
        "rx==calc_xor": (crc_rx ^ crc_calc),
    }


# ============================================================
# CFO ESTIMATION (Hz) over selectable window
# ============================================================

def estimate_cfo_hz(
    freq_burst: np.ndarray,
    fs_out: float,
    phase: int,
    aa_pos: int,
    sps: int,
    payload_len_bytes: int,
    window: str = "pre_aa_hdr_payload",
    *,
    polarity: int = +1,                 # needed to classify bits (0/1) consistently
    iq_burst: np.ndarray = None,        # complex IQ aligned to freq_burst (len = len(freq_burst)+1 recommended)
    return_transitions: bool = True,    # if False, returns only overall CFO float
):
    """
    window options (all start at PREAMBLE start = aa_pos-8):
      - "pre_aa"              : preamble(8) + AA(32)
      - "pre_aa_hdr"          : preamble(8) + AA(32) + header(16)
      - "pre_aa_hdr_payload"  : preamble(8) + AA(32) + header(16) + payload(8*len)

    Returns:
      - if return_transitions=False: overall_cfo_hz (float)  [trimmed-median discriminator CFO]
      - else: (overall_cfo_hz, trans_dict)
        where trans_dict contains:
            cfo_equal_00, cfo_equal_11, cfo_jump_10, cfo_jump_01, cfo_overall_from_transitions
        (all in Hz; NaN if not computable)
    """
    if aa_pos is None or aa_pos < 8 or sps <= 0:
        return None if not return_transitions else (None, None)

    # total bits in requested CFO window (starting at preamble)
    if window == "pre_aa":
        total_bits = 8 + 32
    elif window == "pre_aa_hdr":
        total_bits = 8 + 32 + 16
    else:
        total_bits = 8 + 32 + 16 + payload_len_bytes * 8

    start_sym = aa_pos - 8  # preamble start in symbol index
    start = phase + start_sym * sps
    end = start + total_bits * sps

    if start < 0 or end > freq_burst.size or end <= start:
        return None if not return_transitions else (None, None)

    # -------------------------
    # (A) Your existing CFO (trimmed median of discriminator -> Hz)
    # -------------------------
    seg = freq_burst[start:end].astype(np.float64)
    hz = seg * (fs_out / (2.0 * np.pi))

    if hz.size < 32:
        overall_cfo = float(np.median(hz))
    else:
        lo = int(0.10 * hz.size)
        hi = int(0.90 * hz.size)
        hz_sorted = np.sort(hz)
        hz_trim = hz_sorted[lo:hi] if hi > lo else hz_sorted
        overall_cfo = float(np.median(hz_trim))

    if not return_transitions:
        return overall_cfo

    # -------------------------
    # (B) Transition CFOs via phasor sums (matches your C Accum/add_prod/accum_to_cfo)
    #     IMPORTANT: transitions computed on *packet bits only* (AA+HDR+PAYLOAD...), NOT preamble.
    # -------------------------
    trans = {
        "cfo_equal_00": float("nan"),
        "cfo_equal_11": float("nan"),
        "cfo_jump_10": float("nan"),
        "cfo_jump_01": float("nan"),
        "cfo_overall_from_transitions": float("nan"),
        "nprod_00": 0,
        "nprod_11": 0,
        "nprod_10": 0,
        "nprod_01": 0,
        "nprod_total": 0,
    }

    if iq_burst is None:
        return overall_cfo, trans

    # Build symbol bits (same slicer as decoder)
    sym = freq_burst[phase::sps]
    x = sym if polarity > 0 else -sym
    bits_all = (x > 0).astype(np.uint8)

    # Transition CFOs should be over *packet bits*, excluding preamble.
    trans_bits_total = total_bits - 8
    if trans_bits_total < 2:
        return overall_cfo, trans

    aa_start_sym = aa_pos                 # AA start (symbol index in bits_all)
    aa_start_samp = phase + aa_start_sym * sps

    if aa_start_sym < 0 or (aa_start_sym + trans_bits_total) > bits_all.size:
        return overall_cfo, trans

    bits = bits_all[aa_start_sym:aa_start_sym + trans_bits_total]
    if bits.size < 2:
        return overall_cfo, trans

    # IQ samples aligned to bit-0 start (AA bit0), length = bits*sps
    win_samples = int(bits.size) * int(sps)
    if aa_start_samp < 0 or (aa_start_samp + win_samples) > iq_burst.size:
        return overall_cfo, trans

    iq_win = iq_burst[aa_start_samp: aa_start_samp + win_samples].astype(np.complex64, copy=False)
    if iq_win.size < 2:
        return overall_cfo, trans

    # C-style accumulators
    Re00 = Im00 = 0.0
    Re11 = Im11 = 0.0
    Re10 = Im10 = 0.0
    Re01 = Im01 = 0.0
    Ret  = Imt  = 0.0
    n00 = n11 = n10 = n01 = nt = 0

    def add_prod(re_im_n, x0, x1):
        z = x1 * np.conj(x0)
        re_im_n[0] += float(np.real(z))
        re_im_n[1] += float(np.imag(z))
        re_im_n[2] += 1

    A00 = [Re00, Im00, n00]
    A11 = [Re11, Im11, n11]
    A10 = [Re10, Im10, n10]
    A01 = [Re01, Im01, n01]
    At  = [Ret,  Imt,  nt ]

    # Include products inside the first symbol (bit 0)
    first_bit = int(bits[0])
    first_acc = A00 if first_bit == 0 else A11
    for n in range(1, min(sps, iq_win.size)):
        add_prod(first_acc, iq_win[n - 1], iq_win[n])
        add_prod(At,        iq_win[n - 1], iq_win[n])

    # Handle transitions i-1 -> i; assign all products of symbol i to that transition bin
    for i in range(1, bits.size):
        a = i * sps
        b = (i + 1) * sps
        if b > iq_win.size:
            break

        prevb = int(bits[i - 1])
        curb  = int(bits[i])

        if prevb == 0 and curb == 0:
            tgt = A00
        elif prevb == 1 and curb == 1:
            tgt = A11
        elif prevb == 1 and curb == 0:
            tgt = A10
        else:
            tgt = A01

        for n in range(a, b):
            add_prod(tgt, iq_win[n - 1], iq_win[n])
            add_prod(At,  iq_win[n - 1], iq_win[n])

    def accum_to_cfo(acc):
        if acc[2] <= 0:
            return float("nan")
        ang = float(np.arctan2(acc[1], acc[0]))
        return ang * (fs_out / (2.0 * np.pi))

    trans["cfo_equal_00"] = accum_to_cfo(A00)
    trans["cfo_equal_11"] = accum_to_cfo(A11)
    trans["cfo_jump_10"]  = accum_to_cfo(A10)
    trans["cfo_jump_01"]  = accum_to_cfo(A01)
    trans["cfo_overall_from_transitions"] = accum_to_cfo(At)

    trans["nprod_00"] = int(A00[2])
    trans["nprod_11"] = int(A11[2])
    trans["nprod_10"] = int(A10[2])
    trans["nprod_01"] = int(A01[2])
    trans["nprod_total"] = int(At[2])

    return overall_cfo, trans


# ============================================================
# CFO BOXPLOT (group by AdvA only)
# ============================================================

def save_cfo_boxplot_pdf_by_adva(
    cfo_by_adva: dict,
    out_pdf: str,
    top_n: int,
    tag_advas: set,
    min_count: int = 2,
):
    items = [(k, v) for k, v in cfo_by_adva.items()
             if (k != "NO_AdvA")
             and (k in tag_advas)
             and (v is not None) and (len(v) >= min_count)]

    if not items:
        print(f"[CFO] No CFO samples available to plot (after excluding NO_AdvA, tag-only, and min_count>={min_count}).")
        return False

    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    if top_n and top_n > 0:
        items = items[:top_n]

    labels = [k for k, _ in items]
    print(f"[CFO] Plotting CFO violin+box for top {len(labels)} AdvAs (tag ecosystem only, min_count>={min_count}).")
    data = [v for _, v in items]

    plt.figure(figsize=(max(10, 0.55 * len(labels)), 4.5))
    ax = plt.gca()

    vp = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        widths=0.25,
    )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("CFO (Hz)")
    ax.set_title("BLE CFO grouped by AdvA (tag ecosystem only)")
    plt.xticks(rotation=30, ha="right")

    for i, (_, vals) in enumerate(items, start=1):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            continue
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        ax.text(i, 0.5 * (q1 + q3), f"n={arr.size}",
                ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()

    print(f"[CFO] Saved violin+box plot to: {out_pdf}")
    print(f"[CFO] Groups plotted: {len(labels)} (excluded: NO_AdvA; tag ecosystem only; min_count>={min_count})")
    return True

def save_transition_cfo_violin_boxplots_by_adva(
    trans_cfo_by_adva: dict,
    out_pdf_prefix: str,
    top_n: int,
    tag_advas: set,
    min_count: int = 2,
    keys: tuple = (
        "cfo_equal_00_hz",
        "cfo_equal_11_hz",
        "cfo_jump_10_hz",
        "cfo_jump_01_hz",
        # optionally also:
        # "cfo_overall_from_transitions_hz",
    ),
):
    out_dir = os.path.dirname(out_pdf_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    saved_any = False

    for key in keys:
        items = []
        for adva, d in trans_cfo_by_adva.items():
            if adva == "NO_AdvA":
                continue
            if adva not in tag_advas:
                continue
            if not isinstance(d, dict):
                continue
            vals = d.get(key, None)
            if vals is None or len(vals) < min_count:
                continue
            items.append((adva, vals))

        if not items:
            print(f"[CFO] No samples to plot for {key} (tag-only, min_count>={min_count}).")
            continue

        items.sort(key=lambda kv: len(kv[1]), reverse=True)
        if top_n and top_n > 0:
            items = items[:top_n]

        labels = [k for k, _ in items]
        data = [v for _, v in items]

        out_pdf = f"{out_pdf_prefix}_{key}.pdf"
        print(f"[CFO] Plotting transition CFO '{key}' for {len(labels)} AdvAs -> {out_pdf}")

        plt.figure(figsize=(max(10, 0.55 * len(labels)), 4.5))
        ax = plt.gca()

        ax.violinplot(
            data,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            widths=0.25,
        )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylabel("CFO (Hz)")
        ax.set_title(f"Transition CFO grouped by AdvA (tag ecosystem only): {key}")
        plt.xticks(rotation=30, ha="right")

        for i, (_, vals) in enumerate(items, start=1):
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                continue
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            ax.text(
                i, 0.5 * (q1 + q3),
                f"n={arr.size}",
                ha="center", va="center", fontsize=8
            )

        plt.tight_layout()
        plt.savefig(out_pdf, format="pdf")
        plt.close()

        saved_any = True

    return saved_any


# ============================================================
# PER-WINDOW DECODE (BLESDR byte-domain dewhiten+CRC)
# ============================================================

def decode_one_burst_from_freq(
    freq_burst: np.ndarray,
    channel: int,
    sps: int,
    aa_corr_min: int,
    preamble_min: int,
    debug: bool = False,
    slip_sweep: bool = False,
    slip_max: int = 8,
    crc_diag: bool = False,
):
    aa_hit_any = False
    pre_hit_any = False
    both_hit_any = False
    parsed_any = False
    crc_ok_any = False

    best_any = None
    best_any_score = None  # (aa_corr, pre_corr, length, crc_ok)

    best_crc = None
    best_crc_score = None

    if slip_sweep:
        slip_list = list(range(-abs(int(slip_max)), abs(int(slip_max)) + 1))
    else:
        slip_list = [0, -1, 1, -2, 2, -3, 3]

    for phase in range(sps):
        sym = freq_burst[phase::sps]
        for polarity in (+1, -1):
            x = sym if polarity > 0 else -sym
            bits = (x > 0).astype(np.uint8)

            aa_pos, corr = correlate_access_address_fast(bits)
            if aa_pos is None:
                continue

            pre_corr = preamble_corr_at(bits, aa_pos)

            if corr >= aa_corr_min:
                aa_hit_any = True
            if pre_corr >= preamble_min:
                pre_hit_any = True
            if (corr >= aa_corr_min) and (pre_corr >= preamble_min):
                both_hit_any = True
            else:
                continue

            for slip in slip_list:
                aa0 = aa_pos + slip
                if aa0 < 8:
                    continue
                pre_corr0 = preamble_corr_at(bits, aa0)
                if pre_corr0 < preamble_min:
                    continue

                start = aa0 + 32
                if start + (2+3)*8 > bits.size:
                    continue

                hdr_bits = bits[start:start + 16]
                hdr_msb = bytearray(bits_to_bytes_msbfirst_time(hdr_bits))
                hdr_msb_before = bytes(hdr_msb)

                blesdr_reverse_whiten(channel, hdr_msb)
                hdr_msb_after = bytes(hdr_msb)

                h0 = swap_bits8(hdr_msb[0])
                h1 = swap_bits8(hdr_msb[1])
                length = h1 & 0x3F
                pdu_type = h0 & 0x0F
                txadd = (h0 >> 6) & 1
                rxadd = (h0 >> 7) & 1

                if length > 37:
                    continue

                total_bytes = 2 + length + 3
                total_bits = total_bytes * 8
                if start + total_bits > bits.size:
                    continue

                pkt_bits = bits[start:start + total_bits]
                pkt_msb = bytearray(bits_to_bytes_msbfirst_time(pkt_bits))
                pkt_msb_before = bytes(pkt_msb)

                blesdr_reverse_whiten(channel, pkt_msb)
                pkt_msb_after = bytes(pkt_msb)

                crc_calc = blesdr_reverse_crc(bytes(pkt_msb[:2+length]), init_adv=True)
                crc_rx = ((pkt_msb[2+length] << 16) |
                          (pkt_msb[2+length+1] << 8) |
                          (pkt_msb[2+length+2])) & 0xFFFFFF
                crc_ok = (crc_rx == crc_calc)

                if crc_ok:
                    crc_ok_any = True

                try:
                    _, _, pdu_type2, txadd2, rxadd2, length2, payload = parse_legacy_adv_from_blesdr_bytes(bytes(pkt_msb))
                    parsed_any = True
                except Exception:
                    continue

                addrs = extract_addresses(pdu_type2, payload)

                pkt_std = bytes(swap_bits8(b) for b in pkt_msb)
                is_airtag, is_tag, reason = is_findmy_or_tag_ecosystem(pkt_std)

                pkt = {
                    "aa_pos": int(aa0),
                    "aa_corr": int(corr),
                    "pre_corr": int(pre_corr0),
                    "channel": int(channel),
                    "phase": int(phase),
                    "polarity": int(polarity),
                    "slip": int(slip),

                    "pdu_type": int(pdu_type2),
                    "pdu_type_name": PDU_TYPE_NAMES.get(pdu_type2, f"UNKNOWN({pdu_type2})"),
                    "txadd": int(txadd2),
                    "rxadd": int(rxadd2),
                    "length": int(length2),

                    "payload_hex": payload.hex(),

                    "crc_ok": bool(crc_ok),
                    "crc_rx": int(crc_rx),
                    "crc_calc": int(crc_calc),

                    "is_airtag": bool(is_airtag),
                    "is_tag_ecosystem": bool(is_tag),
                    "tag_reason": reason,
                }
                for k, v in addrs.items():
                    pkt[k] = fmt_mac(v)

                sc = (corr, pre_corr0, length2, int(crc_ok))
                if best_any is None or sc > best_any_score:
                    best_any, best_any_score = pkt, sc

                if crc_ok:
                    if best_crc is None or sc > best_crc_score:
                        best_crc, best_crc_score = pkt, sc

                if crc_diag and (not crc_ok):
                    diag = crc_diag_variants(crc_rx, crc_calc)
                    if debug:
                        print("[CRC_DIAG] AA+PRE matched but CRC failed")
                        print(f"  phase={phase} polarity={polarity} ch={channel} slip={slip}")
                        print(f"  aa_pos={aa0} aa_corr={corr} pre_corr={pre_corr0}")
                        print(f"  hdr_msb_before={hdr_msb_before.hex()} hdr_msb_after={hdr_msb_after.hex()}  hdr_std={h0:02x}{h1:02x} len={length} type={pdu_type}")
                        print(f"  pkt_msb_before[0:8]={pkt_msb_before[:8].hex()} pkt_msb_after[0:8]={pkt_msb_after[:8].hex()}")
                        print(f"  crc_rx={diag['rx']} crc_calc={diag['calc']} rx==calc={diag['rx==calc']} rx==swap={diag['rx==calc_byteswapped']} xor=0x{diag['rx==calc_xor']:06x}")

                if crc_ok:
                    break

    summary = {
        "aa_hit": bool(aa_hit_any),
        "pre_hit": bool(pre_hit_any),
        "both_hit": bool(both_hit_any),
        "parsed": bool(parsed_any),
        "crc_ok": bool(crc_ok_any),
    }
    return best_any, best_crc, summary


# ============================================================
# TOP-LEVEL SNIFFER
# ============================================================

def ble_sniffer(
    iq: np.ndarray,
    fs_in: float,
    channel: int,
    try_adv_channels: bool,
    max_packets: int,
    aa_corr_min: int,
    preamble_min: int,
    thr_k: float,
    do_crc_filter: bool,
    debug: bool,
    max_bursts: int,
    plot_cfo: bool,
    cfo_pdf: str,
    cfo_top: int,
    filter_tags: str,
    slip_sweep: bool,
    slip_max: int,
    crc_diag: bool,
    crc_diag_max: int,
    cfo_window: str,
    cfo_csv: str
):
    L, M = pick_resample_ratio(fs_in, FS_OUT_TARGET, max_den=4096)
    fs_out = fs_in * (L / M)
    sps = int(round(fs_out / SYMBOL_RATE))
    scale = float(L) / float(M)

    print(f"[INFO] Resample ratio L/M = {L}/{M} => fs_out ≈ {fs_out:.2f} (target {FS_OUT_TARGET})")
    print(f"[INFO] SPS = {sps} (fs_out/symbol_rate), CRC_filter={'on' if do_crc_filter else 'off'}")
    print(f"[INFO] Dewhiten: BLESDR byte/MSB stepping (SwapBits(chan)|2, poly 0x11)")
    print(f"[INFO] CRC: BLESDR reverse_crc (init 0x555555 for adv)")
    print(f"[INFO] Continuous scan enabled: energy burst detector disabled (note: --thr ignored)")
    print(f"[INFO] CFO window: {cfo_window}")
    if plot_cfo:
        print(f"[INFO] CFO grouping: AdvA only (top {cfo_top if cfo_top > 0 else 'ALL'})")
    print(f"[INFO] Tag filter: {filter_tags}")
    if slip_sweep:
        print(f"[INFO] Slip sweep enabled: ±{slip_max} bits around AA")
    if crc_diag:
        print(f"[INFO] CRC diagnostics enabled (max detailed prints={crc_diag_max}, gated by --debug for per-candidate dumps)")

    iq_up = upsample(iq, L, M)
    freq_all = gfsk_discriminator(iq_up)

    ch_list = [37, 38, 39] if try_adv_channels else [channel]

    # Build decode windows by continuous AA scanning (upsampled-domain indices)
    windows = find_decode_windows_continuous(
        freq_all=freq_all,
        sps=sps,
        aa_corr_min=aa_corr_min,
        preamble_min=preamble_min,
        slip_max=slip_max,
        max_windows=max_bursts if (max_bursts and max_bursts > 0) else 0,
        debug=debug,
    )
    print(f"[INFO] Decode windows found (analyzed): {len(windows)}")

    stats = {
        "bursts": 0,   # now "windows"
        "aa_hit": 0,
        "pre_hit": 0,
        "both_hit": 0,
        "parsed": 0,
        "crc_ok": 0,
        "airtag": 0,
        "tag_any": 0,
        "kept": 0,
    }

    crcok_slip_hist = Counter()
    crcok_phase_hist = Counter()
    crcok_pol_hist = Counter()
    crcok_chan_hist = Counter()

    crc_fail_printed = 0

    packets = []
    tag_advas = set()
    cfo_by_adva = defaultdict(list)
    trans_cfo_by_adva = defaultdict(lambda: defaultdict(list))
    cfo_rows = []

    def _keep(pkt):
        if filter_tags == "none":
            return True
        if filter_tags == "drop-airtag":
            return not pkt.get("is_airtag", False)
        if filter_tags == "only-airtag":
            return pkt.get("is_airtag", False)
        if filter_tags == "drop-tags":
            return not pkt.get("is_tag_ecosystem", False)
        if filter_tags == "only-tags":
            return pkt.get("is_tag_ecosystem", False)
        return True

    for (s_up, e_up) in windows:
        stats["bursts"] += 1

        # For reporting (original domain): approx inverse-scale
        s_orig = int(round(s_up / scale))
        e_orig = int(round(e_up / scale))

        e_f = max(s_up, e_up - 1)
        if s_up >= freq_all.size or e_f <= s_up:
            continue

        freq_burst_base = freq_all[s_up:e_f]
        iq_burst_base = iq_up[s_up:e_f+1]  # +1 so len(iq)==len(freq)+1

        burst_flags = {"aa_hit": False, "pre_hit": False, "both_hit": False, "parsed": False, "crc_ok": False}

        best_any = None
        best_any_score = None

        best_crc = None
        best_crc_score = None

        for ch in ch_list:
            any_pkt, crc_pkt, summ = decode_one_burst_from_freq(
                freq_burst=freq_burst_base,
                channel=ch,
                sps=sps,
                aa_corr_min=aa_corr_min,
                preamble_min=preamble_min,
                debug=debug,
                slip_sweep=slip_sweep,
                slip_max=slip_max,
                crc_diag=crc_diag,
            )

            for k in burst_flags:
                burst_flags[k] |= summ[k]

            if any_pkt is not None:
                sc = (any_pkt["aa_corr"], any_pkt["pre_corr"], any_pkt["length"], int(any_pkt["crc_ok"]))
                if best_any is None or sc > best_any_score:
                    best_any, best_any_score = any_pkt, sc

            if crc_pkt is not None:
                sc = (crc_pkt["aa_corr"], crc_pkt["pre_corr"], crc_pkt["length"], 1)
                if best_crc is None or sc > best_crc_score:
                    best_crc, best_crc_score = crc_pkt, sc

        if burst_flags["aa_hit"]:
            stats["aa_hit"] += 1
        if burst_flags["pre_hit"]:
            stats["pre_hit"] += 1
        if burst_flags["both_hit"]:
            stats["both_hit"] += 1
        if burst_flags["parsed"]:
            stats["parsed"] += 1
        if burst_flags["crc_ok"]:
            stats["crc_ok"] += 1

        if best_any is not None:
            if best_any.get("is_airtag", False):
                stats["airtag"] += 1
            if best_any.get("is_tag_ecosystem", False):
                stats["tag_any"] += 1

        if best_crc is not None and best_crc.get("crc_ok", False):
            crcok_slip_hist[best_crc.get("slip", 0)] += 1
            crcok_phase_hist[best_crc.get("phase", -1)] += 1
            crcok_pol_hist[best_crc.get("polarity", 0)] += 1
            crcok_chan_hist[best_crc.get("channel", -1)] += 1

        if crc_diag and best_any is not None and (not best_any.get("crc_ok", False)) and (crc_fail_printed < crc_diag_max):
            if best_any.get("aa_corr", 0) >= aa_corr_min and best_any.get("pre_corr", 0) >= preamble_min:
                diag = crc_diag_variants(best_any["crc_rx"], best_any["crc_calc"])
                print("[CRC_FAIL] AA+PRE matched but best decode still CRC_BAD")
                print(f"  window_samp_in=({s_orig},{e_orig}) window_samp_up=({s_up},{e_up}) ch={best_any['channel']} phase={best_any['phase']} pol={best_any['polarity']} slip={best_any.get('slip',0)}")
                print(f"  pdu={best_any['pdu_type_name']} len={best_any['length']} AdvA={best_any.get('AdvA','NO_AdvA')}")
                print(f"  crc_rx={diag['rx']} crc_calc={diag['calc']} rx==swap={diag['rx==calc_byteswapped']} xor=0x{diag['rx==calc_xor']:06x}")
                crc_fail_printed += 1

        pkt_for_print = best_crc if do_crc_filter else best_any
        if pkt_for_print is None:
            continue

        if not _keep(pkt_for_print):
            continue

        if pkt_for_print.get("is_tag_ecosystem", False):
            adva = pkt_for_print.get("AdvA", "NO_AdvA")
            if adva != "NO_AdvA":
                tag_advas.add(adva)

        stats["kept"] += 1

        if max_packets > 0 and len(packets) < max_packets:
            pkt_for_print["start"] = int(s_orig)
            pkt_for_print["end"] = int(e_orig)
            packets.append(pkt_for_print)

        if plot_cfo:
            overall_cfo_hz, trans = estimate_cfo_hz(
                freq_burst=freq_burst_base,
                fs_out=fs_out,
                phase=pkt_for_print["phase"],
                aa_pos=pkt_for_print["aa_pos"],
                sps=sps,
                payload_len_bytes=pkt_for_print["length"],
                window=cfo_window,
                polarity=pkt_for_print["polarity"],
                iq_burst=iq_burst_base,
                return_transitions=True,
            )

            if overall_cfo_hz is not None and np.isfinite(overall_cfo_hz):
                adva = pkt_for_print.get("AdvA", "NO_AdvA")
                cfo_by_adva[adva].append(float(overall_cfo_hz))

                if bool(pkt_for_print.get("is_tag_ecosystem", False)):
                    c00 = float(trans.get("cfo_equal_00", float("nan")))
                    c11 = float(trans.get("cfo_equal_11", float("nan")))
                    c10 = float(trans.get("cfo_jump_10",  float("nan")))
                    c01 = float(trans.get("cfo_jump_01",  float("nan")))
                    cft = float(trans.get("cfo_overall_from_transitions", float("nan")))

                    if np.isfinite(c00):
                        trans_cfo_by_adva[adva]["cfo_equal_00_hz"].append(c00)
                    if np.isfinite(c11):
                        trans_cfo_by_adva[adva]["cfo_equal_11_hz"].append(c11)
                    if np.isfinite(c10):
                        trans_cfo_by_adva[adva]["cfo_jump_10_hz"].append(c10)
                    if np.isfinite(c01):
                        trans_cfo_by_adva[adva]["cfo_jump_01_hz"].append(c01)

                    cfo_rows.append({
                        "AdvA": adva,
                        "CFO_Hz": float(overall_cfo_hz),
                        "CFO_00_Hz": c00,
                        "CFO_11_Hz": c11,
                        "CFO_10_Hz": c10,
                        "CFO_01_Hz": c01,
                        "CFO_from_transitions_Hz": cft,
                        "nprod_00": int(trans.get("nprod_00", 0)),
                        "nprod_11": int(trans.get("nprod_11", 0)),
                        "nprod_10": int(trans.get("nprod_10", 0)),
                        "nprod_01": int(trans.get("nprod_01", 0)),
                        "nprod_total": int(trans.get("nprod_total", 0)),
                        "crc_ok": int(bool(pkt_for_print.get("crc_ok", False))),
                        "is_tag_ecosystem": 1,
                        "pdu_type": int(pkt_for_print.get("pdu_type", -1)),
                        "pdu_type_name": pkt_for_print.get("pdu_type_name", ""),
                        "length": int(pkt_for_print.get("length", -1)),
                        "channel": int(pkt_for_print.get("channel", -1)),
                        "phase": int(pkt_for_print.get("phase", -1)),
                        "polarity": int(pkt_for_print.get("polarity", 0)),
                        "slip": int(pkt_for_print.get("slip", 0)),
                        "window_start": int(s_orig),
                        "window_end": int(e_orig),
                        "aa_pos": int(pkt_for_print.get("aa_pos", -1)),
                        "aa_corr": int(pkt_for_print.get("aa_corr", -1)),
                        "pre_corr": int(pkt_for_print.get("pre_corr", -1)),
                        "cfo_window": cfo_window,
                    })

    def pct(x, denom):
        return 0.0 if denom <= 0 else 100.0 * x / float(denom)

    print("[STATS] Window-level detection quality (continuous AA scan)")
    print(f"  Windows total                                 : {stats['bursts']}")
    print(f"  AA match (>= {aa_corr_min}/32)                            : {stats['aa_hit']} ({pct(stats['aa_hit'], stats['bursts']):.2f}%)")
    print(f"  Preamble match (>= {preamble_min}/8)                      : {stats['pre_hit']} ({pct(stats['pre_hit'], stats['bursts']):.2f}%)")
    print(f"  BOTH (AA & preamble)                           : {stats['both_hit']} ({pct(stats['both_hit'], stats['bursts']):.2f}%)")
    print(f"  Parsed header/payload (BOTH-based)             : {stats['parsed']} ({pct(stats['parsed'], stats['bursts']):.2f}%)")
    print(f"  CRC OK (any BOTH-based candidate)              : {stats['crc_ok']} ({pct(stats['crc_ok'], stats['bursts']):.2f}%)")
    print(f"  AirTag/FindMy (best_any per window)            : {stats['airtag']} ({pct(stats['airtag'], stats['bursts']):.2f}%)")
    print(f"  Tag-ecosystem (best_any; incl. AirTag)         : {stats['tag_any']} ({pct(stats['tag_any'], stats['bursts']):.2f}%)")
    print(f"  Kept after tag filter (printing/CFO inputs)    : {stats['kept']} ({pct(stats['kept'], stats['bursts']):.2f}%)")

    if crc_diag:
        print("[DIAG] CRC-OK parameter histograms (if any CRC OK occurred)")
        if sum(crcok_slip_hist.values()) == 0:
            print("  No CRC_OK packets found => very likely symbol timing / SPS / slicer issue (not CRC math).")
        else:
            print("  Top slips:", crcok_slip_hist.most_common(10))
            print("  Top phases:", crcok_phase_hist.most_common(10))
            print("  Polarity:", crcok_pol_hist.most_common(10))
            print("  Channel seeds:", crcok_chan_hist.most_common(10))

    print(f"[INFO] Packets stored for printing: {len(packets)} (max={max_packets})")

    if plot_cfo:
        saved = save_cfo_boxplot_pdf_by_adva(cfo_by_adva, cfo_pdf, top_n=cfo_top, tag_advas=tag_advas, min_count=100)

        save_transition_cfo_violin_boxplots_by_adva(
            trans_cfo_by_adva,
            out_pdf_prefix="transition_cfo_by_adva",
            top_n=20,
            tag_advas=tag_advas,
            min_count=100,
        )
        if saved:
            total_cfo = sum(len(v) for v in cfo_by_adva.values())
            uniq = sum(1 for v in cfo_by_adva.values() if len(v) > 0)
            print(f"[CFO] Total CFO samples plotted: {total_cfo}  (unique AdvA groups: {uniq})")

        if cfo_rows:
            fieldnames = list(cfo_rows[0].keys())
            with open(cfo_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(cfo_rows)
            print(f"[CFO] Saved per-packet CFO samples to CSV: {cfo_csv}  (rows={len(cfo_rows)})")
        else:
            print("[CFO] No per-packet CFO samples to write to CSV.")

    return packets, stats


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filename", help="IQ file: .cf32/.cfile or raw SPI .bin")
    ap.add_argument("--fs-in", type=float, default=FS_IN_DEFAULT, help="Input IQ sampling rate (Hz)")
    ap.add_argument("--channel", type=int, default=37, help="BLE channel (37/38/39)")
    ap.add_argument("--try-adv-channels", action="store_true", help="Try channels 37/38/39 per window")

    ap.add_argument("--max", type=int, default=20000, help="Max packets to store/print (stats still over all windows)")
    ap.add_argument("--stats-only", action="store_true", help="Only print STATS (and optional CFO plot), do not print packets")
    ap.add_argument("--max-bursts", type=int, default=0, help="Analyze only first N windows (0 = all)")

    ap.add_argument("--aa-min", type=int, default=28, help="Min AA bit-match correlation (0..32)")
    ap.add_argument("--pre-min", type=int, default=7, help="Min preamble bit-match (0..8)")
    ap.add_argument("--thr", type=float, default=DEFAULT_THR_K, help="(Ignored) Burst threshold multiplier (MAD-based)")

    ap.add_argument("--no-crc", action="store_true",
                    help="Disable CRC FILTERING for printed packets (CRC is still computed for stats)")

    ap.add_argument(
        "--cfo-window",
        type=str,
        default="pre_aa_hdr_payload",
        choices=["pre_aa", "pre_aa_hdr", "pre_aa_hdr_payload"],
        help="CFO estimation window: preamble+AA, preamble+AA+header, or preamble+AA+header+payload"
    )

    ap.add_argument("--debug", action="store_true", help="Verbose debug prints")

    ap.add_argument("--plot-cfo", action="store_true", help="Compute CFO per decoded candidate and save a boxplot PDF")
    ap.add_argument("--cfo-pdf", type=str, default="cfo_boxplot_by_adva.pdf", help="Output PDF path for CFO boxplot")
    ap.add_argument("--cfo-top", type=int, default=100, help="Plot only top-N AdvA groups (by sample count). 0 = all")

    ap.add_argument(
        "--filter-tags",
        type=str,
        default="none",
        choices=["none", "drop-airtag", "only-airtag", "drop-tags", "only-tags"],
        help="Filter packets by AirTag/FindMy or tag-ecosystem detection"
    )

    ap.add_argument("--cfo-csv", type=str, default="cfo_samples_rail.csv",
                    help="Output CSV path for per-packet CFO samples (written when --plot-cfo is set)")

    ap.add_argument("--slip-sweep", action="store_true", help="Sweep bit-slip around AA boundary to deterministically detect off-by-k alignment")
    ap.add_argument("--slip-max", type=int, default=8, help="Max slip magnitude (bits) used with --slip-sweep (default ±8)")
    ap.add_argument("--crc-diag", action="store_true", help="Enable CRC diagnostics and histograms (deterministic)")
    ap.add_argument("--crc-diag-max", type=int, default=20, help="Max per-window CRC_FAIL summaries to print (default 20)")

    ap.add_argument("--auto-fs-scan", action="store_true", help="Scan fs-in around provided value and pick the one maximizing CRC_OK")
    ap.add_argument("--fs-scan-span", type=float, default=0.05, help="Fractional span for fs scan (±span). default 0.05")
    ap.add_argument("--fs-scan-steps", type=int, default=21, help="Number of fs points (odd recommended). default 21")

    args = ap.parse_args()

    iq = load_iq_auto(args.filename)
    max_packets = 0 if args.stats_only else args.max

    def run_one(fs_in_val: float):
        packets, stats = ble_sniffer(
            iq,
            fs_in=fs_in_val,
            channel=args.channel,
            try_adv_channels=args.try_adv_channels,
            max_packets=max_packets,
            aa_corr_min=args.aa_min,
            preamble_min=args.pre_min,
            thr_k=args.thr,
            do_crc_filter=(not args.no_crc),
            debug=args.debug,
            max_bursts=args.max_bursts,
            plot_cfo=args.plot_cfo,
            cfo_pdf=args.cfo_pdf,
            cfo_top=args.cfo_top,
            filter_tags=args.filter_tags,
            slip_sweep=args.slip_sweep,
            slip_max=args.slip_max,
            crc_diag=args.crc_diag,
            crc_diag_max=args.crc_diag_max,
            cfo_window=args.cfo_window,
            cfo_csv=args.cfo_csv
        )
        return packets, stats

    if args.auto_fs_scan:
        steps = int(args.fs_scan_steps)
        if steps < 3:
            steps = 3
        if steps % 2 == 0:
            steps += 1
        span = float(args.fs_scan_span)
        center = float(args.fs_in)
        grid = np.linspace(center * (1.0 - span), center * (1.0 + span), steps)

        best_fs = None
        best_crc = -1
        best = None

        print("[AUTO_FS] Scanning fs-in to maximize CRC_OK ...")
        for fs_try in grid:
            print("=" * 72)
            print(f"[AUTO_FS] fs_in={fs_try:.6f}")
            _, st = run_one(fs_try)
            crc_ok = st.get("crc_ok", 0)
            wins = st.get("bursts", 1)
            print(f"[AUTO_FS] CRC_OK={crc_ok}/{wins} ({(100.0*crc_ok/max(1,wins)):.2f}%)")
            if crc_ok > best_crc:
                best_crc = crc_ok
                best_fs = fs_try
                best = st

        print("=" * 72)
        if best_fs is not None:
            print(f"[AUTO_FS] Best fs_in={best_fs:.6f}  CRC_OK={best_crc}/{best.get('bursts',1)}")
            run_one(best_fs)
        return

    run_one(args.fs_in)

if __name__ == "__main__":
    main()

    # for p in packets:
    #     print("---- BLE PACKET ----")
    #     print("AA corr:", p["aa_corr"], "AA pos:", p["aa_pos"], "PRE corr:", p.get("pre_corr", -1))
    #     print("Channel:", p["channel"], "Phase:", p["phase"], "Polarity:", p["polarity"])
    #     print("Slip:", p.get("slip", 0))
    #     print("PDU:", p["pdu_type_name"], f"(type={p['pdu_type']})")
    #     print("Len:", p["length"], "TxAdd:", p["txadd"], "RxAdd:", p["rxadd"])
    #     if "AdvA" in p:
    #         print("AdvA:", p["AdvA"])
    #     if "ScanA" in p:
    #         print("ScanA:", p["ScanA"])
    #     if "InitA" in p:
    #         print("InitA:", p["InitA"])
    #     if "TargetA" in p:
    #         print("TargetA:", p["TargetA"])
    #     print(f"Tag: is_airtag={p.get('is_airtag', False)}  is_tag_ecosystem={p.get('is_tag_ecosystem', False)}  reason={p.get('tag_reason','')}")
    #     print(f"CRC ok: {p['crc_ok']}  CRC rx: 0x{p['crc_rx']:06x}  CRC calc: 0x{p['crc_calc']:06x}")
    #     print()

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# ble_sniffer.py — BLE 1M legacy advertising sniffer for raw IQ streams (SPI int16 IQ or CF32).

# FAST + STATS + CFO (preamble->AA->header+payload; CRC excluded from CFO window):
#   - Resamples ONCE for the whole capture
#   - Computes discriminator ONCE
#   - Fast AA correlation using np.correlate
#   - Reports candidate-level percentages (AA / preamble / both / parsed / CRC)
#   - Uses BLESDR-compatible dewhitening + CRC in the SAME byte space as BLESDR:
#         * build "MSB-first" bytes from on-air bits (matches BLESDR ExtractByte domain)
#         * btle_reverse_whiten(): lfsr = SwapBits(chan) | 2, poly 0x11, MSB stepping
#         * btle_reverse_crc(): BLESDR reverse CRC (init 0x555555 for adv)
#   - CFO estimate (Hz) for candidates that have:
#         preamble match + AA match + parsable header+payload
#     CFO window:
#       preamble (8) + AA (32) + header (16) + payload (8*len)
#     (CRC bits excluded from CFO estimation)
#   - Groups CFO by AdvA ONLY (regardless of CRC) and saves a boxplot to PDF.
#   - Optional AirTag/FindMy + “tag ecosystem” detection and filtering.

# CONTINUOUS SCAN MODE (minimal changes vs burst mode):
#   - Instead of detecting bursts by energy, it scans the *entire* discriminator stream continuously:
#       * slice bits for each (phase, polarity)
#       * find AA hits (>= aa-min)
#       * require preamble (>= pre-min)
#       * decode packet around those candidates
#   - Uses a simple "refractory" skip to avoid re-decoding the same packet multiple times.

# DEBUGS ADDED (minimal changes, opt-in):
#   - --slip-sweep : expands bit-slip search around AA boundary to ±slip-max bits
#   - --crc-diag   : prints deterministic diagnostics when AA+preamble matches but CRC fails
#                  and prints histograms (slip/phase/polarity/channel) for CRC-OK outcomes.
#   - --auto-fs-scan : scans fs-in around the provided value to see if CRC is killed by fs/SPS mismatch
# """

# import os
# import argparse
# from fractions import Fraction
# from collections import defaultdict, Counter

# import numpy as np
# from scipy.signal import resample_poly
# import csv

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt


# # ============================================================
# # DEFAULT PARAMETERS
# # ============================================================

# FS_IN_DEFAULT = 1344039.694796
# FS_OUT_TARGET = 4_000_000
# SYMBOL_RATE = 1_000_000

# # SPI framing (EFR32-style)
# SPI_CHUNK_BYTES = 2048
# SPI_SKIP_BYTES = 0

# # Advertising Access Address
# AA_ADV = 0x8E89BED6  # fixed for adv channels

# # BLE legacy adv PDU type names
# PDU_TYPE_NAMES = {
#     0x0: "ADV_IND",
#     0x1: "ADV_DIRECT_IND",
#     0x2: "ADV_NONCONN_IND",
#     0x3: "SCAN_REQ",
#     0x4: "SCAN_RSP",
#     0x5: "CONNECT_IND",
#     0x6: "ADV_SCAN_IND",
# }

# # (Kept for compatibility; not used in continuous scan mode)
# DEFAULT_THR_K = 8.0
# DEFAULT_SMOOTH_US = 50.0
# DEFAULT_MIN_LEN_US = 80.0
# DEFAULT_GAP_US = 30.0
# DEFAULT_PRE_US = 140.0
# DEFAULT_POST_US = 220.0


# # ============================================================
# # CFO helper (omega) (kept; not used in this version’s decode path)
# # ============================================================

# def estimate_omega_from_preamble_aa(
#     freq_burst: np.ndarray,
#     fs_out: float,
#     phase: int,
#     aa_pos: int,
#     sps: int,
# ) -> float:
#     """
#     Estimate CFO as omega (rad/sample) from discriminator samples over preamble+AA only.
#     Returns omega. If not computable, returns 0.0 (no correction).
#     """
#     if aa_pos is None or aa_pos < 8 or sps <= 0:
#         return 0.0

#     total_bits = 8 + 32  # preamble + AA
#     start = phase + (aa_pos - 8) * sps
#     end = start + total_bits * sps
#     if start < 0 or end > freq_burst.size or end <= start:
#         return 0.0

#     seg = freq_burst[start:end].astype(np.float64)
#     if seg.size < 8:
#         return 0.0

#     if seg.size < 32:
#         omega = float(np.median(seg))
#     else:
#         lo = int(0.10 * seg.size)
#         hi = int(0.90 * seg.size)
#         srt = np.sort(seg)
#         trim = srt[lo:hi] if hi > lo else srt
#         omega = float(np.median(trim))

#     if not np.isfinite(omega):
#         return 0.0

#     omega_max = 2.0 * np.pi * (200e3 / float(fs_out))
#     if omega > omega_max:
#         omega = omega_max
#     elif omega < -omega_max:
#         omega = -omega_max

#     return omega


# # ============================================================
# # Bit slicers
# # ============================================================

# def slice_bits_integrate(
#     freq: np.ndarray,
#     phase: int,
#     sps: int,
#     polarity: int,
#     *,
#     use_mid_frac: float = 0.5,
# ) -> np.ndarray:
#     """
#     Build 1 bit per symbol from discriminator stream using integrate-and-dump.
#     Uses only the middle fraction of each symbol to avoid edge smearing.
#     Returns: bits_all (len = floor((len(freq)-phase)/sps))
#     """
#     if sps <= 0:
#         return np.zeros(0, dtype=np.uint8)

#     mid = max(0.1, min(float(use_mid_frac), 1.0))
#     edge = 0.5 * (1.0 - mid)
#     w0 = int(round(edge * sps))
#     w1 = int(round((1.0 - edge) * sps))
#     w0 = max(0, min(w0, sps - 1))
#     w1 = max(w0 + 1, min(w1, sps))

#     n_syms = (freq.size - phase) // sps
#     if n_syms <= 0:
#         return np.zeros(0, dtype=np.uint8)

#     bits = np.empty(int(n_syms), dtype=np.uint8)
#     pol = 1.0 if polarity > 0 else -1.0

#     for i in range(int(n_syms)):
#         a = phase + i * sps + w0
#         b = phase + i * sps + w1
#         if b > freq.size:
#             bits = bits[:i]
#             break
#         soft = float(np.sum(freq[a:b], dtype=np.float64)) * pol
#         bits[i] = 1 if soft > 0.0 else 0

#     return bits


# def slice_bits_tracked(
#     freq_corr: np.ndarray,
#     phase0: int,
#     sps: int,
#     sym_start: int,
#     n_bits: int,
#     *,
#     polarity: int = +1,
#     block_syms: int = 16,
#     search: int = 2,
#     use_mid_frac: float = 0.5,
# ):
#     """
#     Crude timing recovery:
#       - Keep symbol grid fixed (sym_start + i).
#       - Every block_syms symbols, try phase in [phase-cur-search .. phase-cur+search]
#         and pick the one that maximizes sum(|soft|) over that block.
#       - soft for a symbol = sum(discriminator samples in a mid-window of the symbol).
#       - bit = 1 if (polarity*soft) > 0 else 0.

#     Returns:
#       bits (np.uint8), final_phase (int)
#     """
#     if sps <= 0 or n_bits <= 0:
#         return np.zeros(0, dtype=np.uint8), int(phase0)

#     phase_cur = int(phase0)
#     phase_cur = max(0, min(phase_cur, sps - 1))

#     mid = max(0.1, min(float(use_mid_frac), 1.0))
#     edge = 0.5 * (1.0 - mid)
#     w0 = int(round(edge * sps))
#     w1 = int(round((1.0 - edge) * sps))
#     w0 = max(0, min(w0, sps - 1))
#     w1 = max(w0 + 1, min(w1, sps))

#     bits_out = np.empty(int(n_bits), dtype=np.uint8)

#     i = 0
#     while i < n_bits:
#         blk = min(int(block_syms), n_bits - i)

#         p_lo = max(0, phase_cur - int(search))
#         p_hi = min(sps - 1, phase_cur + int(search))

#         best_p = phase_cur
#         best_score = -1.0

#         for p in range(p_lo, p_hi + 1):
#             score = 0.0
#             ok = True
#             for k in range(blk):
#                 sym_idx = int(sym_start + i + k)
#                 a = p + sym_idx * sps + w0
#                 b = p + sym_idx * sps + w1
#                 if a < 0 or b > freq_corr.size:
#                     ok = False
#                     break
#                 soft = float(np.sum(freq_corr[a:b], dtype=np.float64))
#                 score += abs(soft)
#             if ok and score > best_score:
#                 best_score = score
#                 best_p = p

#         phase_cur = best_p

#         for k in range(blk):
#             sym_idx = int(sym_start + i + k)
#             a = phase_cur + sym_idx * sps + w0
#             b = phase_cur + sym_idx * sps + w1
#             if a < 0 or b > freq_corr.size:
#                 return bits_out[:i+k].astype(np.uint8, copy=False), int(phase_cur)

#             soft = float(np.sum(freq_corr[a:b], dtype=np.float64))
#             soft *= (1.0 if polarity > 0 else -1.0)
#             bits_out[i + k] = 1 if soft > 0.0 else 0

#         i += blk

#     return bits_out, int(phase_cur)


# # ============================================================
# # BIT ORDER HELPERS (on-air)
# # ============================================================

# def bytes_to_bits_lsbfirst(b: bytes) -> np.ndarray:
#     """On-air bit order: LSB-first per octet, octets in given order."""
#     out = np.empty(len(b) * 8, dtype=np.uint8)
#     k = 0
#     for byte in b:
#         for i in range(8):
#             out[k] = (byte >> i) & 1
#             k += 1
#     return out

# def bits_to_u8_lsbfirst(bits8: np.ndarray) -> int:
#     v = 0
#     for i in range(8):
#         v |= (int(bits8[i]) & 1) << i
#     return v

# def bits_to_bytes_lsbfirst(bits: np.ndarray) -> bytes:
#     n = (bits.size // 8) * 8
#     bits = bits[:n]
#     out = bytearray()
#     for i in range(0, n, 8):
#         out.append(bits_to_u8_lsbfirst(bits[i:i+8]))
#     return bytes(out)

# def fmt_mac(b: bytes) -> str:
#     b = b[::-1]
#     return ":".join(f"{x:02X}" for x in b)


# # ============================================================
# # BUILD AA BIT SEQUENCE IN *ON-AIR* ORDER
# # ============================================================

# def aa_bits_onair(aa_u32: int) -> np.ndarray:
#     aa_bytes_le = aa_u32.to_bytes(4, byteorder="little", signed=False)
#     return bytes_to_bits_lsbfirst(aa_bytes_le)

# AA_BITS = aa_bits_onair(AA_ADV)
# AA_PM = (AA_BITS.astype(np.int8) * 2 - 1)  # 0->-1, 1->+1


# def correlate_access_address_fast(bits: np.ndarray):
#     """Fast AA correlation using dot-product over +/-1 sequences (best only)."""
#     if bits.size < 32:
#         return None, -1
#     b_pm = bits.astype(np.int8) * 2 - 1
#     dots = np.correlate(b_pm, AA_PM, mode="valid")  # [-32, +32]
#     i = int(np.argmax(dots))
#     best_dot = int(dots[i])
#     best_matches = (best_dot + 32) // 2
#     return i, int(best_matches)


# def correlate_access_address_positions(bits: np.ndarray, aa_corr_min: int):
#     """
#     Return ALL AA candidate positions where match>=aa_corr_min.
#     Returns: (pos_array, match_array)
#     """
#     if bits.size < 32:
#         return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int16)

#     b_pm = bits.astype(np.int8) * 2 - 1
#     dots = np.correlate(b_pm, AA_PM, mode="valid")
#     matches = (dots.astype(np.int16) + 32) // 2  # 0..32

#     pos = np.nonzero(matches >= int(aa_corr_min))[0]
#     if pos.size == 0:
#         return pos.astype(np.int64, copy=False), np.zeros(0, dtype=np.int16)

#     return pos.astype(np.int64, copy=False), matches[pos].astype(np.int16, copy=False)


# # ============================================================
# # PREAMBLE (8 bits) CORRELATION (aligned to AA)
# # ============================================================

# AA_FIRST_BIT = int(AA_BITS[0])
# PRE_BITS = (np.array([0, 1] * 4, dtype=np.uint8)
#             if AA_FIRST_BIT == 0 else
#             np.array([1, 0] * 4, dtype=np.uint8))

# def preamble_corr_at(bits: np.ndarray, aa_pos: int) -> int:
#     if aa_pos is None or aa_pos < 8:
#         return -1
#     return int(np.sum(bits[aa_pos-8:aa_pos] == PRE_BITS))


# # ============================================================
# # LOADERS
# # ============================================================

# def load_cf32(filename: str) -> np.ndarray:
#     print(f"[INFO] Loading CF32 IQ from {filename}")
#     iq = np.fromfile(filename, dtype=np.complex64)
#     if iq.size == 0:
#         raise ValueError("CF32 file empty/unreadable.")
#     iq = iq - np.mean(iq)
#     return iq

# def load_spi_int16_iq(
#     filename: str,
#     spi_chunk_bytes: int = SPI_CHUNK_BYTES,
#     skip_bytes: int = SPI_SKIP_BYTES,
# ) -> np.ndarray:
#     print(f"[INFO] Loading RAW SPI int16 IQ from {filename}")
#     raw = np.fromfile(filename, dtype=np.uint8)
#     if raw.size == 0:
#         raise ValueError("SPI file empty/unreadable.")

#     n_chunks = raw.size // spi_chunk_bytes
#     if n_chunks == 0:
#         raise ValueError("SPI file smaller than one chunk; check chunk parameters.")
#     raw = raw[: n_chunks * spi_chunk_bytes]

#     payload = []
#     for i in range(n_chunks):
#         s = i * spi_chunk_bytes + skip_bytes
#         e = (i + 1) * spi_chunk_bytes
#         payload.append(raw[s:e])
#     payload = np.concatenate(payload) if payload else np.array([], dtype=np.uint8)

#     if payload.size < 4:
#         raise ValueError("Not enough payload after de-framing.")

#     if payload.size % 2 != 0:
#         payload = payload[:-1]

#     words = payload.view("<i2")
#     I = words[0::2].astype(np.float32) / 32768.0
#     Q = words[1::2].astype(np.float32) / 32768.0
#     iq = (I + 1j * Q).astype(np.complex64)
#     iq = iq - np.mean(iq)
#     return iq

# def load_iq_auto(filename: str) -> np.ndarray:
#     ext = os.path.splitext(filename)[1].lower()
#     if ext in [".cf32", ".cfile"]:
#         return load_cf32(filename)
#     return load_spi_int16_iq(filename)


# # ============================================================
# # RESAMPLER
# # ============================================================

# def pick_resample_ratio(fs_in: float, fs_out: float, max_den: int = 4096):
#     frac = Fraction(fs_out / fs_in).limit_denominator(max_den)
#     return frac.numerator, frac.denominator

# def upsample(iq: np.ndarray, L: int, M: int) -> np.ndarray:
#     return resample_poly(iq, L, M).astype(np.complex64)


# # ============================================================
# # DISCRIMINATOR
# # ============================================================

# def gfsk_discriminator(iq: np.ndarray) -> np.ndarray:
#     prod = iq[1:] * np.conj(iq[:-1])
#     return np.angle(prod).astype(np.float32)


# # ============================================================
# # BLESDR SwapBits, Whitening, CRC (BYTE DOMAIN)
# # ============================================================

# _SWAP_LUT = np.zeros(256, dtype=np.uint8)
# for _v in range(256):
#     x = _v
#     y = 0
#     for _ in range(8):
#         y = (y << 1) | (x & 1)
#         x >>= 1
#     _SWAP_LUT[_v] = y

# def swap_bits8(v: int) -> int:
#     return int(_SWAP_LUT[v & 0xFF])

# def blesdr_reverse_whiten(chan: int, data: bytearray) -> None:
#     lfsr = (swap_bits8(chan) | 0x02) & 0xFF
#     for i in range(len(data)):
#         b = data[i]
#         mask = 0x80
#         while mask:
#             if (lfsr & 0x80) != 0:
#                 lfsr ^= 0x11
#                 b ^= mask
#             lfsr = ((lfsr << 1) & 0xFF)
#             mask >>= 1
#         data[i] = b

# def blesdr_reverse_crc(data: bytes, init_adv: bool = True) -> int:
#     dst0 = 0x55 if init_adv else 0x00
#     dst1 = 0x55 if init_adv else 0x00
#     dst2 = 0x55 if init_adv else 0x00

#     for byte in data:
#         d = swap_bits8(byte)
#         for _ in range(8):
#             t = (dst0 >> 7) & 1
#             dst0 = ((dst0 << 1) & 0xFF) | ((dst1 >> 7) & 1)
#             dst1 = ((dst1 << 1) & 0xFF) | ((dst2 >> 7) & 1)
#             dst2 = ((dst2 << 1) & 0xFF)
#             if t != (d & 1):
#                 dst2 ^= 0x5B
#                 dst1 ^= 0x06
#             d >>= 1

#     return ((dst0 << 16) | (dst1 << 8) | dst2) & 0xFFFFFF


# # ============================================================
# # Convert on-air bits -> BLESDR "ExtractByte" domain bytes
# # ============================================================

# def bits_to_bytes_msbfirst_time(bits: np.ndarray) -> bytes:
#     n = (bits.size // 8) * 8
#     bits = bits[:n]
#     out = bytearray(n // 8)
#     bi = 0
#     for i in range(len(out)):
#         b = 0
#         for k in range(8):
#             b |= (int(bits[bi + k]) & 1) << (7 - k)
#         out[i] = b
#         bi += 8
#     return bytes(out)


# # ============================================================
# # Parse (after dewhiten) in STANDARD BLE byte order (LSB-first)
# # ============================================================

# def parse_legacy_adv_from_blesdr_bytes(packet_data_msb: bytes):
#     if len(packet_data_msb) < 2 + 3:
#         raise ValueError("Too short")

#     packet_std = bytes(swap_bits8(b) for b in packet_data_msb)

#     h0 = packet_std[0]
#     h1 = packet_std[1]
#     pdu_type = h0 & 0x0F
#     txadd = (h0 >> 6) & 1
#     rxadd = (h0 >> 7) & 1
#     length = h1 & 0x3F

#     if length > 60:
#         raise ValueError(f"Legacy length invalid: {length}")

#     need = 2 + length + 3
#     if len(packet_std) < need:
#         raise ValueError("Not enough bytes for payload+CRC")

#     payload = packet_std[2:2+length]
#     return h0, h1, pdu_type, txadd, rxadd, length, payload


# def extract_addresses(pdu_type: int, payload: bytes):
#     out = {}
#     if pdu_type in [0x0, 0x2, 0x6, 0x4]:
#         if len(payload) >= 6:
#             out["AdvA"] = payload[0:6]
#     elif pdu_type == 0x1:
#         if len(payload) >= 12:
#             out["AdvA"] = payload[0:6]
#             out["TargetA"] = payload[6:12]
#     elif pdu_type == 0x3:
#         if len(payload) >= 12:
#             out["ScanA"] = payload[0:6]
#             out["AdvA"] = payload[6:12]
#     elif pdu_type == 0x5:
#         if len(payload) >= 12:
#             out["InitA"] = payload[0:6]
#             out["AdvA"] = payload[6:12]
#     return out


# # ============================================================
# # Tag detection
# # ============================================================

# def _is_tag_service_uuid(uuid16: int) -> bool:
#     return uuid16 in (0xFEAA, 0xFEED, 0xFD5A)

# def is_findmy_or_tag_ecosystem(pdu_std_bytes: bytes):
#     if len(pdu_std_bytes) < 8:
#         return False, False, ""

#     payload_len = len(pdu_std_bytes) - 5
#     if payload_len < 6:
#         return False, False, ""

#     payload = pdu_std_bytes[2:2+payload_len]
#     ad_data = payload[6:]
#     ad_len = len(ad_data)

#     pos = 0
#     while pos + 1 < ad_len:
#         length = ad_data[pos]
#         if length == 0:
#             break
#         if pos + 1 + length > ad_len:
#             break

#         ad_type = ad_data[pos + 1]

#         if ad_type == 0xFF and length >= 3:
#             if pos + 4 <= ad_len:
#                 company_id = ad_data[pos + 2] | (ad_data[pos + 3] << 8)
#                 if company_id == 0x004C and length >= 4:
#                     if pos + 5 < ad_len:
#                         findmy_prefix = ad_data[pos + 4]
#                         if (findmy_prefix == 0x12) and (pos + 6 < ad_len) and (ad_data[pos + 5] == 0x19):
#                             return True, True, "Apple 0x004C + 0x12 0x19"

#         pos += 1 + length

#     return False, False, ""


# # ============================================================
# # Debug helper: CRC comparison variants
# # ============================================================

# def crc_diag_variants(crc_rx: int, crc_calc: int):
#     rx_b = crc_rx.to_bytes(3, "big")
#     cc_b = crc_calc.to_bytes(3, "big")
#     return {
#         "rx": rx_b.hex(),
#         "calc": cc_b.hex(),
#         "rx==calc": (crc_rx == crc_calc),
#         "rx==calc_byteswapped": (rx_b == cc_b[::-1]),
#         "rx==calc_xor": (crc_rx ^ crc_calc),
#     }


# # ============================================================
# # CFO ESTIMATION (Hz) over selectable window
# # ============================================================

# def estimate_cfo_hz(
#     freq_burst: np.ndarray,
#     fs_out: float,
#     phase: int,
#     aa_pos: int,
#     sps: int,
#     payload_len_bytes: int,
#     window: str = "pre_aa_hdr_payload",
#     *,
#     polarity: int = +1,
#     iq_burst: np.ndarray = None,
#     return_transitions: bool = True,
# ):
#     if aa_pos is None or aa_pos < 8 or sps <= 0:
#         return None if not return_transitions else (None, None)

#     if window == "pre_aa":
#         total_bits = 8 + 32
#     elif window == "pre_aa_hdr":
#         total_bits = 8 + 32 + 16
#     else:
#         total_bits = 8 + 32 + 16 + payload_len_bytes * 8

#     start_sym = aa_pos - 8
#     start = phase + start_sym * sps
#     end = start + total_bits * sps

#     if start < 0 or end > freq_burst.size or end <= start:
#         return None if not return_transitions else (None, None)

#     seg = freq_burst[start:end].astype(np.float64)
#     hz = seg * (fs_out / (2.0 * np.pi))

#     if hz.size < 32:
#         overall_cfo = float(np.median(hz))
#     else:
#         lo = int(0.10 * hz.size)
#         hi = int(0.90 * hz.size)
#         hz_sorted = np.sort(hz)
#         hz_trim = hz_sorted[lo:hi] if hi > lo else hz_sorted
#         overall_cfo = float(np.median(hz_trim))

#     if not return_transitions:
#         return overall_cfo

#     trans = {
#         "cfo_equal_00": float("nan"),
#         "cfo_equal_11": float("nan"),
#         "cfo_jump_10": float("nan"),
#         "cfo_jump_01": float("nan"),
#         "cfo_overall_from_transitions": float("nan"),
#         "nprod_00": 0,
#         "nprod_11": 0,
#         "nprod_10": 0,
#         "nprod_01": 0,
#         "nprod_total": 0,
#     }

#     if iq_burst is None:
#         return overall_cfo, trans

#     sym = freq_burst[phase::sps]
#     x = sym if polarity > 0 else -sym
#     bits_all = (x > 0).astype(np.uint8)

#     trans_bits_total = total_bits - 8
#     if trans_bits_total < 2:
#         return overall_cfo, trans

#     aa_start_sym = aa_pos
#     aa_start_samp = phase + aa_start_sym * sps

#     if aa_start_sym < 0 or (aa_start_sym + trans_bits_total) > bits_all.size:
#         return overall_cfo, trans

#     bits = bits_all[aa_start_sym:aa_start_sym + trans_bits_total]
#     if bits.size < 2:
#         return overall_cfo, trans

#     win_samples = int(bits.size) * int(sps)
#     if aa_start_samp < 0 or (aa_start_samp + win_samples) > iq_burst.size:
#         return overall_cfo, trans

#     iq_win = iq_burst[aa_start_samp: aa_start_samp + win_samples].astype(np.complex64, copy=False)
#     if iq_win.size < 2:
#         return overall_cfo, trans

#     def add_prod(acc, x0, x1):
#         z = x1 * np.conj(x0)
#         acc[0] += float(np.real(z))
#         acc[1] += float(np.imag(z))
#         acc[2] += 1

#     A00 = [0.0, 0.0, 0]
#     A11 = [0.0, 0.0, 0]
#     A10 = [0.0, 0.0, 0]
#     A01 = [0.0, 0.0, 0]
#     At  = [0.0, 0.0, 0]

#     first_bit = int(bits[0])
#     first_acc = A00 if first_bit == 0 else A11
#     for n in range(1, min(sps, iq_win.size)):
#         add_prod(first_acc, iq_win[n - 1], iq_win[n])
#         add_prod(At,        iq_win[n - 1], iq_win[n])

#     for i in range(1, bits.size):
#         a = i * sps
#         b = (i + 1) * sps
#         if b > iq_win.size:
#             break

#         prevb = int(bits[i - 1])
#         curb  = int(bits[i])

#         if prevb == 0 and curb == 0:
#             tgt = A00
#         elif prevb == 1 and curb == 1:
#             tgt = A11
#         elif prevb == 1 and curb == 0:
#             tgt = A10
#         else:
#             tgt = A01

#         for n in range(a, b):
#             add_prod(tgt, iq_win[n - 1], iq_win[n])
#             add_prod(At,  iq_win[n - 1], iq_win[n])

#     def accum_to_cfo(acc):
#         if acc[2] <= 0:
#             return float("nan")
#         ang = float(np.arctan2(acc[1], acc[0]))
#         return ang * (fs_out / (2.0 * np.pi))

#     trans["cfo_equal_00"] = accum_to_cfo(A00)
#     trans["cfo_equal_11"] = accum_to_cfo(A11)
#     trans["cfo_jump_10"]  = accum_to_cfo(A10)
#     trans["cfo_jump_01"]  = accum_to_cfo(A01)
#     trans["cfo_overall_from_transitions"] = accum_to_cfo(At)

#     trans["nprod_00"] = int(A00[2])
#     trans["nprod_11"] = int(A11[2])
#     trans["nprod_10"] = int(A10[2])
#     trans["nprod_01"] = int(A01[2])
#     trans["nprod_total"] = int(At[2])

#     return overall_cfo, trans


# # ============================================================
# # CFO BOXPLOT (group by AdvA only)
# # ============================================================

# def save_cfo_boxplot_pdf_by_adva(
#     cfo_by_adva: dict,
#     out_pdf: str,
#     top_n: int,
#     tag_advas: set,
#     min_count: int = 2,
# ):
#     items = [(k, v) for k, v in cfo_by_adva.items()
#              if (k != "NO_AdvA")
#              and (k in tag_advas)
#              and (v is not None) and (len(v) >= min_count)]

#     if not items:
#         print(f"[CFO] No CFO samples available to plot (after excluding NO_AdvA, tag-only, and min_count>={min_count}).")
#         return False

#     items.sort(key=lambda kv: len(kv[1]), reverse=True)
#     if top_n and top_n > 0:
#         items = items[:top_n]

#     labels = [k for k, _ in items]
#     print(f"[CFO] Plotting CFO violin+box for top {len(labels)} AdvAs (tag ecosystem only, min_count>={min_count}).")
#     data = [v for _, v in items]

#     plt.figure(figsize=(max(10, 0.55 * len(labels)), 4.5))
#     ax = plt.gca()

#     ax.violinplot(
#         data,
#         showmeans=False,
#         showmedians=False,
#         showextrema=False,
#     )

#     ax.boxplot(
#         data,
#         labels=labels,
#         showfliers=False,
#         widths=0.25,
#     )

#     ax.axhline(0.0, linewidth=1.0)
#     ax.set_ylabel("CFO (Hz)")
#     ax.set_title("BLE CFO grouped by AdvA (tag ecosystem only)")
#     plt.xticks(rotation=30, ha="right")

#     for i, (_, vals) in enumerate(items, start=1):
#         arr = np.asarray(vals, dtype=np.float64)
#         if arr.size == 0:
#             continue
#         q1 = np.percentile(arr, 25)
#         q3 = np.percentile(arr, 75)
#         ax.text(i, 0.5 * (q1 + q3), f"n={arr.size}",
#                 ha="center", va="center", fontsize=8)

#     plt.tight_layout()
#     plt.savefig(out_pdf, format="pdf")
#     plt.close()

#     print(f"[CFO] Saved violin+box plot to: {out_pdf}")
#     print(f"[CFO] Groups plotted: {len(labels)} (excluded: NO_AdvA; tag ecosystem only; min_count>={min_count})")
#     return True


# def save_transition_cfo_violin_boxplots_by_adva(
#     trans_cfo_by_adva: dict,
#     out_pdf_prefix: str,
#     top_n: int,
#     tag_advas: set,
#     min_count: int = 2,
#     keys: tuple = (
#         "cfo_equal_00_hz",
#         "cfo_equal_11_hz",
#         "cfo_jump_10_hz",
#         "cfo_jump_01_hz",
#     ),
# ):
#     out_dir = os.path.dirname(out_pdf_prefix)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)

#     saved_any = False

#     for key in keys:
#         items = []
#         for adva, d in trans_cfo_by_adva.items():
#             if adva == "NO_AdvA":
#                 continue
#             if adva not in tag_advas:
#                 continue
#             if not isinstance(d, dict):
#                 continue
#             vals = d.get(key, None)
#             if vals is None or len(vals) < min_count:
#                 continue
#             items.append((adva, vals))

#         if not items:
#             print(f"[CFO] No samples to plot for {key} (tag-only, min_count>={min_count}).")
#             continue

#         items.sort(key=lambda kv: len(kv[1]), reverse=True)
#         if top_n and top_n > 0:
#             items = items[:top_n]

#         labels = [k for k, _ in items]
#         data = [v for _, v in items]

#         out_pdf = f"{out_pdf_prefix}_{key}.pdf"
#         print(f"[CFO] Plotting transition CFO '{key}' for {len(labels)} AdvAs -> {out_pdf}")

#         plt.figure(figsize=(max(10, 0.55 * len(labels)), 4.5))
#         ax = plt.gca()

#         ax.violinplot(
#             data,
#             showmeans=False,
#             showmedians=False,
#             showextrema=False,
#         )

#         ax.boxplot(
#             data,
#             labels=labels,
#             showfliers=False,
#             widths=0.25,
#         )

#         ax.axhline(0.0, linewidth=1.0)
#         ax.set_ylabel("CFO (Hz)")
#         ax.set_title(f"Transition CFO grouped by AdvA (tag ecosystem only): {key}")
#         plt.xticks(rotation=30, ha="right")

#         for i, (_, vals) in enumerate(items, start=1):
#             arr = np.asarray(vals, dtype=np.float64)
#             if arr.size == 0:
#                 continue
#             q1 = np.percentile(arr, 25)
#             q3 = np.percentile(arr, 75)
#             ax.text(
#                 i, 0.5 * (q1 + q3),
#                 f"n={arr.size}",
#                 ha="center", va="center", fontsize=8
#             )

#         plt.tight_layout()
#         plt.savefig(out_pdf, format="pdf")
#         plt.close()

#         saved_any = True

#     return saved_any


# # ============================================================
# # Candidate decode (continuous scan): known (phase,polarity,aa_pos)
# # ============================================================

# def decode_one_candidate_from_stream(
#     freq_all: np.ndarray,
#     channel: int,
#     sps: int,
#     aa_corr_min: int,
#     preamble_min: int,
#     *,
#     phase: int,
#     polarity: int,
#     aa_pos: int,
#     slip_sweep: bool = False,
#     slip_max: int = 8,
#     debug: bool = False,
#     crc_diag: bool = False,
# ):
#     """
#     Decode around a specific AA candidate at symbol index aa_pos (AA start) on a given (phase, polarity).
#     Uses the same tracked slicing logic as your burst decoder but without scanning phases.
#     """
#     aa_hit_any = False
#     pre_hit_any = False
#     both_hit_any = False
#     parsed_any = False
#     crc_ok_any = False

#     best_any = None
#     best_any_score = None
#     best_crc = None
#     best_crc_score = None

#     if slip_sweep:
#         slip_list = list(range(-abs(int(slip_max)), abs(int(slip_max)) + 1))
#     else:
#         slip_list = [0, -1, 1, -2, 2, -3, 3]

#     # This candidate already exists because AA>=aa_min in coarse scan,
#     # but slip can move it; we re-validate after tracked slicing.
#     for slip in slip_list:
#         aa0 = int(aa_pos) + int(slip)
#         if aa0 < 8:
#             continue

#         aa_hit_any = True

#         sym0 = aa0 - 8  # preamble start
#         total_bits_max = 40 + (2 + 37 + 3) * 8  # pre+AA + max(header+payload+crc) bits

#         bits_local, _phase_end = slice_bits_tracked(
#             freq_corr=freq_all,
#             phase0=int(phase),
#             sps=int(sps),
#             sym_start=int(sym0),
#             n_bits=int(total_bits_max),
#             polarity=int(polarity),
#             block_syms=16,
#             search=2,
#             use_mid_frac=0.5,
#         )

#         if bits_local.size < 56:
#             continue

#         pre_corr0 = int(np.sum(bits_local[0:8] == PRE_BITS))
#         if pre_corr0 >= preamble_min:
#             pre_hit_any = True
#         else:
#             continue

#         corr0 = int(np.sum(bits_local[8:40] == AA_BITS))
#         if corr0 >= aa_corr_min:
#             both_hit_any = True
#         else:
#             continue

#         # Header bits -> BLESDR domain -> dewhiten header
#         hdr_bits = bits_local[40:56]
#         hdr_msb = bytearray(bits_to_bytes_msbfirst_time(hdr_bits))
#         hdr_msb_before = bytes(hdr_msb)

#         blesdr_reverse_whiten(channel, hdr_msb)
#         hdr_msb_after = bytes(hdr_msb)

#         h0 = swap_bits8(hdr_msb[0])
#         h1 = swap_bits8(hdr_msb[1])
#         length = h1 & 0x3F
#         pdu_type = h0 & 0x0F

#         if length > 37:
#             continue

#         total_bytes = 2 + length + 3
#         total_bits = total_bytes * 8

#         if bits_local.size < 40 + total_bits:
#             continue

#         pkt_bits = bits_local[40:40 + total_bits]
#         pkt_msb = bytearray(bits_to_bytes_msbfirst_time(pkt_bits))
#         pkt_msb_before = bytes(pkt_msb)

#         blesdr_reverse_whiten(channel, pkt_msb)
#         pkt_msb_after = bytes(pkt_msb)

#         crc_calc = blesdr_reverse_crc(bytes(pkt_msb[:2 + length]), init_adv=True)
#         crc_rx = ((pkt_msb[2 + length] << 16) |
#                   (pkt_msb[2 + length + 1] << 8) |
#                   (pkt_msb[2 + length + 2])) & 0xFFFFFF
#         crc_ok = (crc_rx == crc_calc)

#         if crc_ok:
#             crc_ok_any = True

#         try:
#             _, _, pdu_type2, txadd2, rxadd2, length2, payload = parse_legacy_adv_from_blesdr_bytes(bytes(pkt_msb))
#             parsed_any = True
#         except Exception:
#             continue

#         addrs = extract_addresses(pdu_type2, payload)

#         pkt_std = bytes(swap_bits8(b) for b in pkt_msb)
#         is_airtag, is_tag, reason = is_findmy_or_tag_ecosystem(pkt_std)

#         pkt = {
#             "aa_pos": int(aa0),
#             "aa_corr": int(corr0),
#             "pre_corr": int(pre_corr0),
#             "channel": int(channel),
#             "phase": int(phase),
#             "polarity": int(polarity),
#             "slip": int(slip),

#             "pdu_type": int(pdu_type2),
#             "pdu_type_name": PDU_TYPE_NAMES.get(pdu_type2, f"UNKNOWN({pdu_type2})"),
#             "txadd": int(txadd2),
#             "rxadd": int(rxadd2),
#             "length": int(length2),

#             "payload_hex": payload.hex(),

#             "crc_ok": bool(crc_ok),
#             "crc_rx": int(crc_rx),
#             "crc_calc": int(crc_calc),

#             "is_airtag": bool(is_airtag),
#             "is_tag_ecosystem": bool(is_tag),
#             "tag_reason": reason,
#         }
#         for k, v in addrs.items():
#             pkt[k] = fmt_mac(v)

#         sc = (int(corr0), int(pre_corr0), int(length2), int(crc_ok))
#         if best_any is None or sc > best_any_score:
#             best_any, best_any_score = pkt, sc

#         if crc_ok:
#             if best_crc is None or sc > best_crc_score:
#                 best_crc, best_crc_score = pkt, sc

#         if crc_diag and (not crc_ok) and debug:
#             diag = crc_diag_variants(crc_rx, crc_calc)
#             print("[CRC_DIAG] AA+PRE matched but CRC failed")
#             print(f"  phase={phase} polarity={polarity} ch={channel} slip={slip}")
#             print(f"  aa_pos={aa0} aa_corr={corr0} pre_corr={pre_corr0}")
#             print(f"  hdr_msb_before={hdr_msb_before.hex()} hdr_msb_after={hdr_msb_after.hex()}  hdr_std={h0:02x}{h1:02x} len={length} type={pdu_type}")
#             print(f"  pkt_msb_before[0:8]={pkt_msb_before[:8].hex()} pkt_msb_after[0:8]={pkt_msb_after[:8].hex()}")
#             print(f"  crc_rx={diag['rx']} crc_calc={diag['calc']} rx==calc={diag['rx==calc']} rx==swap={diag['rx==calc_byteswapped']} xor=0x{diag['rx==calc_xor']:06x}")

#         if crc_ok:
#             break

#     summary = {
#         "aa_hit": bool(aa_hit_any),
#         "pre_hit": bool(pre_hit_any),
#         "both_hit": bool(both_hit_any),
#         "parsed": bool(parsed_any),
#         "crc_ok": bool(crc_ok_any),
#     }
#     return best_any, best_crc, summary


# # ============================================================
# # TOP-LEVEL SNIFFER (CONTINUOUS SCAN)
# # ============================================================

# def ble_sniffer(
#     iq: np.ndarray,
#     fs_in: float,
#     channel: int,
#     try_adv_channels: bool,
#     max_packets: int,
#     aa_corr_min: int,
#     preamble_min: int,
#     thr_k: float,            # kept for CLI compatibility; not used
#     do_crc_filter: bool,
#     debug: bool,
#     max_bursts: int,         # reused as "max candidates to process" in continuous scan
#     plot_cfo: bool,
#     cfo_pdf: str,
#     cfo_top: int,
#     filter_tags: str,
#     slip_sweep: bool,
#     slip_max: int,
#     crc_diag: bool,
#     crc_diag_max: int,
#     cfo_window: str,
#     cfo_csv: str
# ):
#     L, M = pick_resample_ratio(fs_in, FS_OUT_TARGET, max_den=4096)
#     fs_out = fs_in * (L / M)
#     sps = int(round(fs_out / SYMBOL_RATE))
#     scale = float(L) / float(M)

#     print(f"[INFO] Resample ratio L/M = {L}/{M} => fs_out ≈ {fs_out:.2f} (target {FS_OUT_TARGET})")
#     print(f"[INFO] SPS = {sps} (fs_out/symbol_rate), CRC_filter={'on' if do_crc_filter else 'off'}")
#     print(f"[INFO] Dewhiten: BLESDR byte/MSB stepping (SwapBits(chan)|2, poly 0x11)")
#     print(f"[INFO] CRC: BLESDR reverse_crc (init 0x555555 for adv)")
#     print(f"[INFO] Continuous scan: AA correlation over full stream (no energy burst detector)")
#     print(f"[INFO] CFO window: {cfo_window}")
#     if plot_cfo:
#         print(f"[INFO] CFO grouping: AdvA only (top {cfo_top if cfo_top > 0 else 'ALL'})")
#     print(f"[INFO] Tag filter: {filter_tags}")
#     if slip_sweep:
#         print(f"[INFO] Slip sweep enabled: ±{slip_max} bits around AA")
#     if crc_diag:
#         print(f"[INFO] CRC diagnostics enabled (max detailed prints={crc_diag_max}, gated by --debug for per-candidate dumps)")

#     iq_up = upsample(iq, L, M)
#     freq_all = gfsk_discriminator(iq_up)

#     ch_list = [37, 38, 39] if try_adv_channels else [channel]

#     stats = {
#         "candidates_total": 0,
#         "aa_hit": 0,
#         "pre_hit": 0,
#         "both_hit": 0,
#         "parsed": 0,
#         "crc_ok": 0,
#         "airtag": 0,
#         "tag_any": 0,
#         "kept": 0,
#         "decoded": 0,
#     }

#     crcok_slip_hist = Counter()
#     crcok_phase_hist = Counter()
#     crcok_pol_hist = Counter()
#     crcok_chan_hist = Counter()

#     crc_fail_printed = 0

#     packets = []
#     tag_advas = set()
#     cfo_by_adva = defaultdict(list)
#     trans_cfo_by_adva = defaultdict(lambda: defaultdict(list))
#     cfo_rows = []

#     def _keep(pkt):
#         if filter_tags == "none":
#             return True
#         if filter_tags == "drop-airtag":
#             return not pkt.get("is_airtag", False)
#         if filter_tags == "only-airtag":
#             return pkt.get("is_airtag", False)
#         if filter_tags == "drop-tags":
#             return not pkt.get("is_tag_ecosystem", False)
#         if filter_tags == "only-tags":
#             return pkt.get("is_tag_ecosystem", False)
#         return True

#     # ------------------------------------------------------------
#     # Build AA+preamble candidates across all (phase, polarity)
#     # ------------------------------------------------------------
#     candidates = []
#     aa_hits = 0
#     pre_hits = 0

#     for phase in range(sps):
#         for polarity in (+1, -1):
#             bits = slice_bits_integrate(freq_all, phase, sps, polarity, use_mid_frac=0.5)
#             if bits.size < 40:
#                 continue

#             pos, corrv = correlate_access_address_positions(bits, aa_corr_min)
#             if pos.size == 0:
#                 continue

#             aa_hits += int(pos.size)

#             for aa_pos, aa_corr in zip(pos.tolist(), corrv.tolist()):
#                 pre_corr = preamble_corr_at(bits, int(aa_pos))
#                 if pre_corr >= preamble_min:
#                     pre_hits += 1
#                     samp = int(phase + int(aa_pos) * sps)
#                     candidates.append((samp, int(phase), int(polarity), int(aa_pos), int(aa_corr), int(pre_corr)))

#     candidates.sort(key=lambda t: t[0])

#     stats["aa_hit"] = int(aa_hits)
#     stats["pre_hit"] = int(pre_hits)
#     stats["both_hit"] = int(pre_hits)  # in continuous mode, "candidates" are AA+PRE

#     if max_bursts > 0:
#         candidates = candidates[:max_bursts]

#     stats["candidates_total"] = int(len(candidates))
#     print(f"[INFO] Candidates (AA>= {aa_corr_min} and PRE>= {preamble_min}) to process: {len(candidates)}")

#     # Refractory skip in output-sample units (avoid re-decoding same packet repeatedly)
#     next_allowed_sample = 0

#     # ------------------------------------------------------------
#     # Process candidates in time order
#     # ------------------------------------------------------------
#     for (cand_samp, cand_phase, cand_pol, cand_aa_pos, cand_aa_corr, cand_pre_corr) in candidates:
#         if cand_samp < next_allowed_sample:
#             continue

#         best_any = None
#         best_any_score = None
#         best_crc = None
#         best_crc_score = None

#         cand_flags = {"aa_hit": True, "pre_hit": True, "both_hit": True, "parsed": False, "crc_ok": False}

#         for ch in ch_list:
#             any_pkt, crc_pkt, summ = decode_one_candidate_from_stream(
#                 freq_all=freq_all,
#                 channel=ch,
#                 sps=sps,
#                 aa_corr_min=aa_corr_min,
#                 preamble_min=preamble_min,
#                 phase=cand_phase,
#                 polarity=cand_pol,
#                 aa_pos=cand_aa_pos,
#                 slip_sweep=slip_sweep,
#                 slip_max=slip_max,
#                 debug=debug,
#                 crc_diag=crc_diag,
#             )

#             for k in cand_flags:
#                 cand_flags[k] |= summ.get(k, False)

#             if any_pkt is not None:
#                 sc = (any_pkt["aa_corr"], any_pkt["pre_corr"], any_pkt["length"], int(any_pkt["crc_ok"]))
#                 if best_any is None or sc > best_any_score:
#                     best_any, best_any_score = any_pkt, sc

#             if crc_pkt is not None:
#                 sc = (crc_pkt["aa_corr"], crc_pkt["pre_corr"], crc_pkt["length"], 1)
#                 if best_crc is None or sc > best_crc_score:
#                     best_crc, best_crc_score = crc_pkt, sc

#         if cand_flags["parsed"]:
#             stats["parsed"] += 1
#         if cand_flags["crc_ok"]:
#             stats["crc_ok"] += 1

#         if best_any is not None:
#             stats["decoded"] += 1
#             if best_any.get("is_airtag", False):
#                 stats["airtag"] += 1
#             if best_any.get("is_tag_ecosystem", False):
#                 stats["tag_any"] += 1

#         if best_crc is not None and best_crc.get("crc_ok", False):
#             crcok_slip_hist[best_crc.get("slip", 0)] += 1
#             crcok_phase_hist[best_crc.get("phase", -1)] += 1
#             crcok_pol_hist[best_crc.get("polarity", 0)] += 1
#             crcok_chan_hist[best_crc.get("channel", -1)] += 1

#         if crc_diag and best_any is not None and (not best_any.get("crc_ok", False)) and (crc_fail_printed < crc_diag_max):
#             if best_any.get("aa_corr", 0) >= aa_corr_min and best_any.get("pre_corr", 0) >= preamble_min:
#                 diag = crc_diag_variants(best_any["crc_rx"], best_any["crc_calc"])
#                 print("[CRC_FAIL] AA+PRE matched but best decode still CRC_BAD")
#                 print(f"  cand_samp={cand_samp} ch={best_any['channel']} phase={best_any['phase']} pol={best_any['polarity']} slip={best_any.get('slip',0)}")
#                 print(f"  pdu={best_any['pdu_type_name']} len={best_any['length']} AdvA={best_any.get('AdvA','NO_AdvA')}")
#                 print(f"  crc_rx={diag['rx']} crc_calc={diag['calc']} rx==swap={diag['rx==calc_byteswapped']} xor=0x{diag['rx==calc_xor']:06x}")
#                 crc_fail_printed += 1

#         pkt_for_print = best_crc if do_crc_filter else best_any
#         if pkt_for_print is None:
#             continue

#         if not _keep(pkt_for_print):
#             continue

#         # Tag AdvA collection (for plotting filter)
#         if pkt_for_print.get("is_tag_ecosystem", False):
#             adva = pkt_for_print.get("AdvA", "NO_AdvA")
#             if adva != "NO_AdvA":
#                 tag_advas.add(adva)

#         stats["kept"] += 1

#         # Compute start/end in *input* sample indices (best-effort)
#         aa0 = int(pkt_for_print["aa_pos"])
#         ph = int(pkt_for_print["phase"])
#         length = int(pkt_for_print["length"])
#         total_bytes = 2 + length + 3
#         total_bits = 40 + total_bytes * 8  # pre+AA + hdr/payload/crc bytes

#         start_out = ph + (aa0 - 8) * sps
#         end_out = start_out + total_bits * sps

#         start_in = int(round(start_out / scale))
#         end_in = int(round(end_out / scale))

#         start_in = max(0, min(start_in, int(iq.size)))
#         end_in = max(0, min(end_in, int(iq.size)))

#         pkt_for_print["start"] = int(start_in)
#         pkt_for_print["end"] = int(end_in)

#         # Refractory: skip until after this decoded packet ends (output domain)
#         next_allowed_sample = max(next_allowed_sample, int(end_out))

#         if max_packets > 0 and len(packets) < max_packets:
#             packets.append(pkt_for_print)

#         if plot_cfo:
#             overall_cfo_hz, trans = estimate_cfo_hz(
#                 freq_burst=freq_all,
#                 fs_out=fs_out,
#                 phase=pkt_for_print["phase"],
#                 aa_pos=pkt_for_print["aa_pos"],
#                 sps=sps,
#                 payload_len_bytes=pkt_for_print["length"],
#                 window=cfo_window,
#                 polarity=pkt_for_print["polarity"],
#                 iq_burst=iq_up,
#                 return_transitions=True,
#             )

#             if overall_cfo_hz is not None and np.isfinite(overall_cfo_hz):
#                 adva = pkt_for_print.get("AdvA", "NO_AdvA")
#                 cfo_by_adva[adva].append(float(overall_cfo_hz))

#                 # Per-packet CFO log row (CSV) — ONLY tag ecosystem packets
#                 if pkt_for_print.get("is_tag_ecosystem", False) is True:
#                     c00 = float(trans.get("cfo_equal_00", float("nan")))
#                     c11 = float(trans.get("cfo_equal_11", float("nan")))
#                     c10 = float(trans.get("cfo_jump_10",  float("nan")))
#                     c01 = float(trans.get("cfo_jump_01",  float("nan")))
#                     cft = float(trans.get("cfo_overall_from_transitions", float("nan")))

#                     if np.isfinite(c00):
#                         trans_cfo_by_adva[adva]["cfo_equal_00_hz"].append(c00)
#                     if np.isfinite(c11):
#                         trans_cfo_by_adva[adva]["cfo_equal_11_hz"].append(c11)
#                     if np.isfinite(c10):
#                         trans_cfo_by_adva[adva]["cfo_jump_10_hz"].append(c10)
#                     if np.isfinite(c01):
#                         trans_cfo_by_adva[adva]["cfo_jump_01_hz"].append(c01)

#                     cfo_rows.append({
#                         "AdvA": adva,
#                         "CFO_Hz": float(overall_cfo_hz),
#                         "CFO_00_Hz": c00,
#                         "CFO_11_Hz": c11,
#                         "CFO_10_Hz": c10,
#                         "CFO_01_Hz": c01,
#                         "CFO_from_transitions_Hz": cft,
#                         "nprod_00": int(trans.get("nprod_00", 0)),
#                         "nprod_11": int(trans.get("nprod_11", 0)),
#                         "nprod_10": int(trans.get("nprod_10", 0)),
#                         "nprod_01": int(trans.get("nprod_01", 0)),
#                         "nprod_total": int(trans.get("nprod_total", 0)),
#                         "crc_ok": int(bool(pkt_for_print.get("crc_ok", False))),
#                         "is_tag_ecosystem": 1,
#                         "pdu_type": int(pkt_for_print.get("pdu_type", -1)),
#                         "pdu_type_name": pkt_for_print.get("pdu_type_name", ""),
#                         "length": int(pkt_for_print.get("length", -1)),
#                         "channel": int(pkt_for_print.get("channel", -1)),
#                         "phase": int(pkt_for_print.get("phase", -1)),
#                         "polarity": int(pkt_for_print.get("polarity", 0)),
#                         "slip": int(pkt_for_print.get("slip", 0)),
#                         "burst_start": int(start_in),
#                         "burst_end": int(end_in),
#                         "aa_pos": int(pkt_for_print.get("aa_pos", -1)),
#                         "aa_corr": int(pkt_for_print.get("aa_corr", -1)),
#                         "pre_corr": int(pkt_for_print.get("pre_corr", -1)),
#                         "cfo_window": cfo_window,
#                     })

#     def pct(x, denom):
#         return 0.0 if denom <= 0 else 100.0 * x / float(denom)

#     print("[STATS] Continuous scan quality (candidate-level)")
#     print(f"  AA hits across all (phase,pol) (>= {aa_corr_min}/32)      : {stats['aa_hit']}")
#     print(f"  Candidates with PRE (>= {preamble_min}/8)                 : {stats['candidates_total']}")
#     print(f"  Parsed among candidates                                    : {stats['parsed']} ({pct(stats['parsed'], stats['candidates_total']):.2f}%)")
#     print(f"  CRC OK among candidates                                    : {stats['crc_ok']} ({pct(stats['crc_ok'], stats['candidates_total']):.2f}%)")
#     print(f"  Decoded packets (best_any exists)                          : {stats['decoded']} ({pct(stats['decoded'], stats['candidates_total']):.2f}%)")
#     print(f"  AirTag/FindMy (best_any per candidate)                     : {stats['airtag']} ({pct(stats['airtag'], stats['candidates_total']):.2f}%)")
#     print(f"  Tag-ecosystem (best_any; incl. AirTag)                     : {stats['tag_any']} ({pct(stats['tag_any'], stats['candidates_total']):.2f}%)")
#     print(f"  Kept after tag filter (printing/CFO inputs)                : {stats['kept']} ({pct(stats['kept'], stats['candidates_total']):.2f}%)")

#     if crc_diag:
#         print("[DIAG] CRC-OK parameter histograms (if any CRC OK occurred)")
#         if sum(crcok_slip_hist.values()) == 0:
#             print("  No CRC_OK packets found => very likely symbol timing / SPS / slicer issue (not CRC math).")
#         else:
#             print("  Top slips:", crcok_slip_hist.most_common(10))
#             print("  Top phases:", crcok_phase_hist.most_common(10))
#             print("  Polarity:", crcok_pol_hist.most_common(10))
#             print("  Channel seeds:", crcok_chan_hist.most_common(10))

#     print(f"[INFO] Packets stored for printing: {len(packets)} (max={max_packets})")

#     if plot_cfo:
#         saved = save_cfo_boxplot_pdf_by_adva(cfo_by_adva, cfo_pdf, top_n=cfo_top, tag_advas=tag_advas, min_count=100)

#         save_transition_cfo_violin_boxplots_by_adva(
#             trans_cfo_by_adva,
#             out_pdf_prefix="transition_cfo_by_adva",
#             top_n=20,
#             tag_advas=tag_advas,
#             min_count=100,
#         )

#         if saved:
#             total_cfo = sum(len(v) for v in cfo_by_adva.values())
#             uniq = sum(1 for v in cfo_by_adva.values() if len(v) > 0)
#             print(f"[CFO] Total CFO samples plotted: {total_cfo}  (unique AdvA groups: {uniq})")

#         if cfo_rows:
#             fieldnames = list(cfo_rows[0].keys())
#             with open(cfo_csv, "w", newline="") as f:
#                 w = csv.DictWriter(f, fieldnames=fieldnames)
#                 w.writeheader()
#                 w.writerows(cfo_rows)
#             print(f"[CFO] Saved per-packet CFO samples to CSV: {cfo_csv}  (rows={len(cfo_rows)})")
#         else:
#             print("[CFO] No per-packet CFO samples to write to CSV.")

#     return packets, stats


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("filename", help="IQ file: .cf32/.cfile or raw SPI .bin")
#     ap.add_argument("--fs-in", type=float, default=FS_IN_DEFAULT, help="Input IQ sampling rate (Hz)")
#     ap.add_argument("--channel", type=int, default=37, help="BLE channel (37/38/39)")
#     ap.add_argument("--try-adv-channels", action="store_true", help="Try channels 37/38/39 per burst/candidate")

#     ap.add_argument("--max", type=int, default=20000, help="Max packets to store/print (stats still over all candidates)")
#     ap.add_argument("--stats-only", action="store_true", help="Only print STATS (and optional CFO plot), do not print packets")
#     ap.add_argument("--max-bursts", type=int, default=0, help="In continuous mode: process only first N candidates (0 = all)")

#     ap.add_argument("--aa-min", type=int, default=28, help="Min AA bit-match correlation (0..32)")
#     ap.add_argument("--pre-min", type=int, default=7, help="Min preamble bit-match (0..8)")
#     ap.add_argument("--thr", type=float, default=DEFAULT_THR_K, help="(unused in continuous mode) Burst threshold multiplier")

#     ap.add_argument("--no-crc", action="store_true",
#                     help="Disable CRC FILTERING for printed packets (CRC is still computed for stats)")

#     ap.add_argument(
#         "--cfo-window",
#         type=str,
#         default="pre_aa_hdr_payload",
#         choices=["pre_aa", "pre_aa_hdr", "pre_aa_hdr_payload"],
#         help="CFO estimation window: preamble+AA, preamble+AA+header, or preamble+AA+header+payload"
#     )

#     ap.add_argument("--debug", action="store_true", help="Verbose debug prints")

#     ap.add_argument("--plot-cfo", action="store_true", help="Compute CFO per decoded candidate and save a boxplot PDF")
#     ap.add_argument("--cfo-pdf", type=str, default="cfo_boxplot_by_adva.pdf", help="Output PDF path for CFO boxplot")
#     ap.add_argument("--cfo-top", type=int, default=100, help="Plot only top-N AdvA groups (by sample count). 0 = all")

#     ap.add_argument(
#         "--filter-tags",
#         type=str,
#         default="none",
#         choices=["none", "drop-airtag", "only-airtag", "drop-tags", "only-tags"],
#         help="Filter packets by AirTag/FindMy or tag-ecosystem detection"
#     )

#     ap.add_argument("--cfo-csv", type=str, default="cfo_samples_rail.csv",
#                     help="Output CSV path for per-packet CFO samples (written when --plot-cfo is set)")

#     ap.add_argument("--slip-sweep", action="store_true", help="Sweep bit-slip around AA boundary to deterministically detect off-by-k alignment")
#     ap.add_argument("--slip-max", type=int, default=8, help="Max slip magnitude (bits) used with --slip-sweep (default ±8)")
#     ap.add_argument("--crc-diag", action="store_true", help="Enable CRC diagnostics and histograms (deterministic)")
#     ap.add_argument("--crc-diag-max", type=int, default=20, help="Max CRC_FAIL summaries to print (default 20)")

#     ap.add_argument("--auto-fs-scan", action="store_true", help="Scan fs-in around provided value and pick the one maximizing CRC_OK")
#     ap.add_argument("--fs-scan-span", type=float, default=0.05, help="Fractional span for fs scan (±span). default 0.05")
#     ap.add_argument("--fs-scan-steps", type=int, default=21, help="Number of fs points (odd recommended). default 21")

#     args = ap.parse_args()

#     iq = load_iq_auto(args.filename)
#     max_packets = 0 if args.stats_only else args.max

#     def run_one(fs_in_val: float):
#         packets, stats = ble_sniffer(
#             iq,
#             fs_in=fs_in_val,
#             channel=args.channel,
#             try_adv_channels=args.try_adv_channels,
#             max_packets=max_packets,
#             aa_corr_min=args.aa_min,
#             preamble_min=args.pre_min,
#             thr_k=args.thr,
#             do_crc_filter=(not args.no_crc),
#             debug=args.debug,
#             max_bursts=args.max_bursts,
#             plot_cfo=args.plot_cfo,
#             cfo_pdf=args.cfo_pdf,
#             cfo_top=args.cfo_top,
#             filter_tags=args.filter_tags,
#             slip_sweep=args.slip_sweep,
#             slip_max=args.slip_max,
#             crc_diag=args.crc_diag,
#             crc_diag_max=args.crc_diag_max,
#             cfo_window=args.cfo_window,
#             cfo_csv=args.cfo_csv
#         )
#         return packets, stats

#     if args.auto_fs_scan:
#         steps = int(args.fs_scan_steps)
#         if steps < 3:
#             steps = 3
#         if steps % 2 == 0:
#             steps += 1
#         span = float(args.fs_scan_span)
#         center = float(args.fs_in)
#         grid = np.linspace(center * (1.0 - span), center * (1.0 + span), steps)

#         best_fs = None
#         best_crc = -1
#         best = None

#         print("[AUTO_FS] Scanning fs-in to maximize CRC_OK ...")
#         for fs_try in grid:
#             print("=" * 72)
#             print(f"[AUTO_FS] fs_in={fs_try:.6f}")
#             _, st = run_one(fs_try)
#             crc_ok = st.get("crc_ok", 0)
#             cand = max(1, st.get("candidates_total", 1))
#             print(f"[AUTO_FS] CRC_OK={crc_ok}/{cand} ({(100.0*crc_ok/max(1,cand)):.2f}%)")
#             if crc_ok > best_crc:
#                 best_crc = crc_ok
#                 best_fs = fs_try
#                 best = st

#         print("=" * 72)
#         if best_fs is not None:
#             print(f"[AUTO_FS] Best fs_in={best_fs:.6f}  CRC_OK={best_crc}/{best.get('candidates_total',1)}")
#             run_one(best_fs)
#         return

#     run_one(args.fs_in)

#     if args.stats_only:
#         return

# if __name__ == "__main__":
#     main()