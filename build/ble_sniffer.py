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

PERFORMANCE (minimal logic-preserving changes):
  - Optional multiprocessing across decode windows to use all CPU cores.
  - Uses shared memory for freq_all and iq_up to avoid duplicating huge arrays per worker on macOS (spawn).
"""

import os
import argparse
from fractions import Fraction
from collections import defaultdict, Counter
import csv
import math

import numpy as np
from scipy.signal import resample_poly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Multiprocessing (minimal additions)
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

FS_IN_DEFAULT = 1344106.900141  # 1_365_333.33, 1344106.900141
FS_OUT_TARGET = 4_000_000
SYMBOL_RATE = 1_000_000 

# FindMy-relevant: score CRC_OK only for long packets (per-window best CRC candidate)
FS_SCAN_LONG_MIN = 24   # bytes (adjust to 20 if you want)

# SPI framing (EFR32-style)
SPI_CHUNK_BYTES = 2048
SPI_SKIP_BYTES = 4  # skip SPI header per chunk

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

CRC_DIST_MAX = 0      # accept near-valid packets
TOPK_HYP = 5          # number of phase/slip hypotheses kept

# ============================================================
# BIT ORDER HELPERS (on-air)
# ============================================================

def soft_slice(bits_f):
    """
    Returns hard bits + confidence per bit.
    bits_f: discriminator samples at symbol rate
    """
    sigma = np.std(bits_f) + 1e-6
    soft = bits_f / sigma
    bits = (soft > 0).astype(np.uint8)
    conf = np.abs(soft)
    return bits, conf

def apply_cfo_correction(iq: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    """
    Rotate IQ to remove carrier frequency offset.
    """
    if cfo_hz is None or not np.isfinite(cfo_hz):
        return iq
    n = np.arange(iq.size, dtype=np.float64)
    return iq * np.exp(-1j * 2.0 * np.pi * cfo_hz * n / fs)

def refine_sps(freq_all, sps, test_offsets=(-0.02, -0.01, 0, 0.01, 0.02)):
    best = sps
    best_score = -1
    for eps in test_offsets:
        sps_try = sps * (1.0 + eps)
        hits = 0
        for k in range(0, len(freq_all) - int(10*sps_try), int(100*sps_try)):
            idx = np.arange(k, k + int(200*sps_try), sps_try).astype(int)
            bits = (freq_all[idx] > 0)
            _, corr = correlate_access_address_fast(bits)
            hits += (corr >= 28)
        if hits > best_score:
            best_score = hits
            best = sps_try
    return best

def crc_distance(crc_rx, crc_calc):
    return bin((crc_rx ^ crc_calc) & 0xFFFFFF).count("1")

def gardner_timing(freq: np.ndarray, sps: int, mu: float = 0.0, gain: float = 0.01):
    """
    Gardner timing recovery for real-valued discriminator samples.

    freq : discriminator samples
    sps  : samples per symbol (integer)
    mu   : initial fractional timing phase [0, sps)
    gain : loop gain (0.005–0.02 typical)

    Returns:
        1-sample-per-symbol recovered sequence
    """
    out = []
    i = 0
    N = len(freq)

    # We must be able to safely read:
    #   i + mu
    #   i + mu + sps/2
    #   i + mu + sps
    # so enforce a strict upper bound
    while True:
        i0 = int(i + mu)
        i1 = int(i + mu + 0.5 * sps)
        i2 = int(i + mu + sps)

        if i2 >= N or i1 >= N or i0 >= N:
            break

        s0 = freq[i0]
        s1 = freq[i1]
        s2 = freq[i2]

        # Gardner timing error detector
        err = s1 * (s0 - s2)

        mu += gain * err

        # Keep mu bounded to [0, sps)
        while mu >= sps:
            mu -= sps
        while mu < 0:
            mu += sps

        out.append(s1)
        i += sps

    return np.asarray(out, dtype=np.float32)

def timing_quality(sym, aa_pos):
    """
    Score timing based on preamble + AA symbol separation.
    Higher = cleaner eye opening.
    """
    if aa_pos < 8 or aa_pos + 32 >= sym.size:
        return -1e9

    # use preamble + AA region only
    region = sym[aa_pos-8 : aa_pos+32]
    # eye opening proxy: mean absolute value
    return float(np.mean(np.abs(region)))

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
    """
    Load RAW SPI int16 IQ data.

    IMPORTANT:
    - If skip_bytes > 0, those bytes are assumed to be a per-chunk SPI header.
    - The header is REMOVED without deleting time:
        it is replaced by zero-valued IQ samples.
    - This preserves sample count, phase continuity, and symbol timing.
    """
    print(f"[INFO] Loading RAW SPI int16 IQ from {filename}")

    raw = np.fromfile(filename, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError("SPI file empty/unreadable.")

    n_chunks = raw.size // spi_chunk_bytes
    if n_chunks == 0:
        raise ValueError("SPI file smaller than one chunk; check chunk parameters.")

    raw = raw[: n_chunks * spi_chunk_bytes]

    payload_chunks = []

    for i in range(n_chunks):
        chunk = raw[i * spi_chunk_bytes : (i + 1) * spi_chunk_bytes]

        if skip_bytes > 0:
            # Replace header bytes with zero-valued IQ samples
            # 4 bytes = 1 complex sample (I16,Q16)
            if skip_bytes % 4 != 0:
                raise ValueError("skip_bytes must be a multiple of 4 to preserve IQ alignment")

            n_dummy = skip_bytes
            dummy = np.zeros(n_dummy, dtype=np.uint8)

            payload_chunks.append(dummy)
            payload_chunks.append(chunk[skip_bytes:])
        else:
            payload_chunks.append(chunk)

    payload = np.concatenate(payload_chunks) if payload_chunks else np.array([], dtype=np.uint8)

    if payload.size < 4:
        raise ValueError("Not enough payload after de-framing.")

    # Ensure int16 alignment
    if payload.size % 2 != 0:
        payload = payload[:-1]

    words = payload.view("<i2")  # little-endian int16

    # I/Q extraction
    I = words[0::2].astype(np.float32) / 32768.0
    Q = words[1::2].astype(np.float32) / 32768.0

    iq = (I + 1j * Q).astype(np.complex64)

    # Remove DC offset
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
            matches = ((dots + 32) // 2).astype(np.int16)

            idx = np.flatnonzero(matches >= aa_corr_min)
            if idx.size == 0:
                continue

            # preamble gate (kept as-is to preserve logic)
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
    min_sep_samples = int((MAX_TOTAL_BITS * sps) // 2)
    kept = []

    for aa_sample, sc in cand:
        if not kept:
            kept.append([aa_sample, sc])
            continue

        prev_sample, prev_sc = kept[-1]
        if aa_sample - prev_sample <= min_sep_samples:
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
    This matches BLESDR's ExtractByte().
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
    polarity: int = +1,
    iq_burst: np.ndarray = None,
    return_transitions: bool = True,
):
    """
    Returns overall CFO (trimmed-median discriminator CFO) and optional transition CFOs.
    """
    if aa_pos is None or aa_pos < 8 or sps <= 0:
        return None if not return_transitions else (None, None)

    if window == "pre_aa":
        total_bits = 8 + 32
    elif window == "pre_aa_hdr":
        total_bits = 8 + 32 + 16
    else:
        total_bits = 8 + 32 + 16 + payload_len_bytes * 8

    start_sym = aa_pos - 8
    start = phase + start_sym * sps
    end = start + total_bits * sps

    if start < 0 or end > freq_burst.size or end <= start:
        return None if not return_transitions else (None, None)

    seg = freq_burst[start:end].astype(np.float64, copy=False)
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

    sym = freq_burst[phase::sps]
    x = sym if polarity > 0 else -sym
    bits_all = (x > 0).astype(np.uint8)

    trans_bits_total = total_bits - 8
    if trans_bits_total < 2:
        return overall_cfo, trans

    aa_start_sym = aa_pos
    aa_start_samp = phase + aa_start_sym * sps

    if aa_start_sym < 0 or (aa_start_sym + trans_bits_total) > bits_all.size:
        return overall_cfo, trans

    bits = bits_all[aa_start_sym:aa_start_sym + trans_bits_total]
    if bits.size < 2:
        return overall_cfo, trans

    win_samples = int(bits.size) * int(sps)
    if aa_start_samp < 0 or (aa_start_samp + win_samples) > iq_burst.size:
        return overall_cfo, trans

    iq_win = iq_burst[aa_start_samp: aa_start_samp + win_samples].astype(np.complex64, copy=False)
    if iq_win.size < 2:
        return overall_cfo, trans

    # C-style accumulators
    A00 = [0.0, 0.0, 0]
    A11 = [0.0, 0.0, 0]
    A10 = [0.0, 0.0, 0]
    A01 = [0.0, 0.0, 0]
    At  = [0.0, 0.0, 0]

    def add_prod(acc, x0, x1):
        z = x1 * np.conj(x0)
        acc[0] += float(np.real(z))
        acc[1] += float(np.imag(z))
        acc[2] += 1

    first_bit = int(bits[0])
    first_acc = A00 if first_bit == 0 else A11
    for n in range(1, min(sps, iq_win.size)):
        add_prod(first_acc, iq_win[n - 1], iq_win[n])
        add_prod(At,        iq_win[n - 1], iq_win[n])

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
    *,
    iq_burst: np.ndarray = None,
    fs_out: float = FS_OUT_TARGET,
):
    """
    Decode ONE candidate window from discriminator samples.

    Keeps your working base (interp-based fractional timing scan + eps step repair),
    and adds a *local refinement* for longer payloads (Apple FindMy etc.) to recover CRC.

    IMPORTANT:
      - Purely on freq_burst
      - BLESDR dewhiten+CRC unchanged
    """

    # ---------------------------
    # helpers
    # ---------------------------
    def _sym_stream_interp(x: np.ndarray, phase0: float, step: float):
        """
        1-sample-per-symbol stream by sampling freq_burst at float indices.
        Linear interpolation is IMPORTANT here (works better than nearest for discriminator).
        """
        if step <= 0:
            return None
        N = x.size
        if N < 64:
            return None
        max_k = int((N - 2 - phase0) / step)   # need < N-1 for interp
        if max_k <= 0:
            return None
        idx = phase0 + step * np.arange(max_k + 1, dtype=np.float64)
        if idx.size < 64:
            return None
        xp = np.arange(N, dtype=np.float64)
        return np.interp(idx, xp, x).astype(np.float32, copy=False)

    def _try_cfo_second_stage(iq_src: np.ndarray, ph: int, frac0: float, pol: int):
        """Estimate CFO from the candidate (preamble+AA), correct IQ, re-discriminate and decode."""
        if iq_src is None or iq_src.size < 32:
            return

        # Build a symbol stream close to the current best timing (eps=0 pass)
        phase0 = float(ph) + float(frac0) * float(sps)
        sym0 = _sym_stream_interp(freq_burst, phase0=phase0, step=float(sps))
        if sym0 is None or sym0.size < 64:
            return

        x0 = sym0 if pol > 0 else -sym0
        bits0 = (x0 > 0).astype(np.uint8)
        aa_pos0, aa_corr0 = correlate_access_address_fast(bits0)
        if aa_pos0 is None:
            return
        pre_corr0 = preamble_corr_at(bits0, aa_pos0)
        if aa_corr0 < aa_corr_min or pre_corr0 < preamble_min:
            return

        # Estimate CFO using the discriminator (robust enough for residual CFO)
        # Use preamble+AA only (no need to know length/CRC).
        try:
            cfo_hz, _trans = estimate_cfo_hz(
                freq_burst,
                fs_out=float(fs_out),
                phase=int(ph),
                aa_pos=int(aa_pos0),
                sps=int(sps),
                payload_len_bytes=0,
                window="pre_aa",
                polarity=int(pol),
                iq_burst=None,
                return_transitions=False,
            )
        except Exception:
            return

        if cfo_hz is None:
            return
        if not np.isfinite(cfo_hz):
            return

        # Sanity clamp: reject absurd estimates
        if abs(float(cfo_hz)) > 250_000:
            return

        iq_corr = apply_cfo_correction(iq_src, float(cfo_hz), float(fs_out))
        freq_corr = gfsk_discriminator(iq_corr)

        # Decode again using the same passes, but on corrected discriminator.
        # Keep it cheap: only run the fast pass around current (phase, frac0).
        sym = _sym_stream_interp(freq_corr, phase0=phase0, step=float(sps))
        if sym is None or sym.size < 64:
            return

        x = sym if pol > 0 else -sym
        bits = (x > 0).astype(np.uint8)
        aa_pos, aa_corr = correlate_access_address_fast(bits)
        if aa_pos is None:
            return
        pre_corr = preamble_corr_at(bits, aa_pos)
        if aa_corr < aa_corr_min or pre_corr < preamble_min:
            return

        _try_decode(bits, aa_pos, aa_corr, pre_corr, phase=int(ph), frac=float(frac0), eps=0.0, step=float(sps), polarity=int(pol))

    def _try_local_step_refine(freq_src: np.ndarray, ph: int, frac0: float, pol: int, step0: float):
        """Try a tiny local sweep of symbol step (effective SPS) to fix long-payload drift."""
        if freq_src is None or freq_src.size < 256:
            return

        # ppm-scale search. This is NOT a wide search: +/- 200 ppm.
        # Over a 300+ bit payload, 100 ppm can cause noticeable sampling drift.
        ppm_grid = (-200, -120, -80, -40, -20, 0, 20, 40, 80, 120, 200)
        for ppm in ppm_grid:
            step = float(step0) * (1.0 + (float(ppm) * 1e-6))
            phase0 = float(ph) + float(frac0) * float(sps)

            sym = _sym_stream_interp(freq_src, phase0=phase0, step=step)
            if sym is None or sym.size < 64:
                continue

            x = sym if pol > 0 else -sym
            bits = (x > 0).astype(np.uint8)

            aa_pos, aa_corr = correlate_access_address_fast(bits)
            if aa_pos is None:
                continue
            pre_corr = preamble_corr_at(bits, aa_pos)
            if aa_corr < aa_corr_min or pre_corr < preamble_min:
                continue

            _try_decode(bits, aa_pos, aa_corr, pre_corr, phase=int(ph), frac=float(frac0), eps=0.0, step=float(step), polarity=int(pol))

            # stop early if we recovered a CRC_OK
            if crc_ok_any:
                return

    def _try_decode(bits: np.ndarray, aa_pos: int, aa_corr: int, pre_corr: int,
                    phase: int, frac: float, eps: float, step: float, polarity: int):
        """
        Given a bitstream and AA position, try slips and return candidate packets.
        """
        nonlocal aa_hit_any, pre_hit_any, both_hit_any, parsed_any, crc_ok_any

        # gates already satisfied before calling, but keep flags consistent
        aa_hit_any = True
        pre_hit_any = True
        both_hit_any = True

        for slip in slip_list:
            aa0 = aa_pos + slip
            if aa0 < 8:
                continue

            # NOTE: keep your original behavior (re-check preamble after slip)
            # This is safer when AA slips near the boundary.
            pre_corr0 = preamble_corr_at(bits, aa0)
            if pre_corr0 < preamble_min:
                continue

            start = aa0 + 32
            if start + 16 > bits.size:
                continue

            # ---------- HEADER ----------
            hdr_bits = bits[start:start + 16]
            hdr_msb = bytearray(bits_to_bytes_msbfirst_time(hdr_bits))
            blesdr_reverse_whiten(channel, hdr_msb)

            h0 = swap_bits8(hdr_msb[0])
            h1 = swap_bits8(hdr_msb[1])
            length = h1 & 0x3F
            if length > 37:
                continue

            total_bits = (2 + length + 3) * 8
            if start + total_bits > bits.size:
                continue

            # ---------- FULL PACKET ----------
            pkt_bits = bits[start:start + total_bits]
            pkt_msb = bytearray(bits_to_bytes_msbfirst_time(pkt_bits))
            blesdr_reverse_whiten(channel, pkt_msb)

            crc_calc = blesdr_reverse_crc(pkt_msb[:2 + length], True)
            crc_rx = (
                (pkt_msb[2 + length] << 16)
                | (pkt_msb[3 + length] << 8)
                | (pkt_msb[4 + length])
            ) & 0xFFFFFF
            crc_ok = (crc_rx == crc_calc)
            if crc_ok:
                crc_ok_any = True

            try:
                _, _, pdu_t, txadd, rxadd, l2, payload = parse_legacy_adv_from_blesdr_bytes(pkt_msb)
                parsed_any = True
            except Exception:
                continue

            addrs = extract_addresses(pdu_t, payload)
            pkt_std = bytes(swap_bits8(b) for b in pkt_msb)
            is_airtag, is_tag, tag_reason = is_findmy_or_tag_ecosystem(pkt_std)

            pkt = {
                "aa_pos": int(aa0),
                "aa_corr": int(aa_corr),
                "pre_corr": int(pre_corr0),

                "phase": int(phase),
                "frac": float(frac),
                "eps": float(eps),
                "step": float(step),

                "polarity": int(polarity),
                "slip": int(slip),
                "channel": int(channel),

                "pdu_type": int(pdu_t),
                "pdu_type_name": PDU_TYPE_NAMES.get(pdu_t, "?"),
                "length": int(l2),
                "payload_hex": payload.hex(),

                "crc_ok": bool(crc_ok),
                "crc_rx": int(crc_rx),
                "crc_calc": int(crc_calc),

                "is_airtag": bool(is_airtag),
                "is_tag_ecosystem": bool(is_tag),
                "tag_reason": tag_reason,
            }
            for k, v in addrs.items():
                pkt[k] = fmt_mac(v)

            hypotheses.append(pkt)

    # ---------------------------
    # stats flags
    # ---------------------------
    aa_hit_any = False
    pre_hit_any = False
    both_hit_any = False
    parsed_any = False
    crc_ok_any = False

    # --- instrumentation: stage attribution (per-window) ---
    inst = {
        "crc_ok_base": 0,        # CRC_OK achieved in main pass loop
        "crc_ok_local": 0,       # CRC_OK achieved via local (eps/frac neighborhood) refine
        "crc_ok_step": 0,        # CRC_OK achieved via local step/SPS ppm refine
        "crc_ok_cfo": 0,         # CRC_OK achieved via CFO second stage
        "ran_local": 0,
        "ran_step": 0,
        "ran_cfo": 0,
        "is_long": 0,
        "is_tag": 0,
    }

    hypotheses = []

    slip_list = (
        list(range(-abs(int(slip_max)), abs(int(slip_max)) + 1))
        if slip_sweep else [0, -1, 1]
    )

    frac_grid = (-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5)
    eps_grid = (-0.0002, -0.0001, -0.00005, 0.0, 0.00005, 0.0001, 0.0002)

    passes = [
        {"name": "fast", "eps_list": (0.0,), "frac_list": (0.0,)},
        {"name": "repair", "eps_list": eps_grid, "frac_list": frac_grid},
    ]

    for pass_idx, P in enumerate(passes):
        for polarity in (+1, -1):
            for phase in range(sps):
                for eps in P["eps_list"]:
                    step = float(sps) * (1.0 + float(eps))
                    for frac in P["frac_list"]:
                        phase0 = float(phase) + float(frac) * float(sps)

                        sym = _sym_stream_interp(freq_burst, phase0=phase0, step=step)
                        if sym is None or sym.size < 64:
                            continue

                        x = sym if polarity > 0 else -sym
                        bits = (x > 0).astype(np.uint8)

                        aa_pos, aa_corr = correlate_access_address_fast(bits)
                        if aa_pos is None:
                            continue

                        if aa_corr >= aa_corr_min:
                            aa_hit_any = True
                        pre_corr = preamble_corr_at(bits, aa_pos)
                        if pre_corr >= preamble_min:
                            pre_hit_any = True

                        if aa_corr < aa_corr_min or pre_corr < preamble_min:
                            continue

                        both_hit_any = True

                        _try_decode(bits, aa_pos, aa_corr, pre_corr, phase, frac, eps, step, polarity)

                        # early stop if fast pass got CRC_OK
                        if pass_idx == 0 and crc_ok_any:
                            break
                    if pass_idx == 0 and crc_ok_any:
                        break
                if pass_idx == 0 and crc_ok_any:
                    break

        if pass_idx == 0 and crc_ok_any:
            break

    # If CRC was achieved during base passes, mark it
    if crc_ok_any:
        inst["crc_ok_base"] = 1

    summary = {
        "aa_hit": bool(aa_hit_any),
        "pre_hit": bool(pre_hit_any),
        "both_hit": bool(both_hit_any),
        "parsed": bool(parsed_any),
        "crc_ok": bool(crc_ok_any),
        "inst": inst,
    }

    if not hypotheses:
        return None, None, summary

    # rank
    hypotheses.sort(
        key=lambda p: (
            1 if p.get("crc_ok", False) else 0,
            p.get("aa_corr", 0),
            p.get("pre_corr", 0),
            p.get("length", 0),
        ),
        reverse=True,
    )
    best_any = hypotheses[0]
    best_crc = next((p for p in hypotheses if p.get("crc_ok", False)), None)

    # ---------------------------
    # Local refinement for long payloads / FindMy-like
    # ---------------------------
    if best_crc is None:
        L = int(best_any.get("length", 0))
        likely_long = (L >= 20) or bool(best_any.get("is_tag_ecosystem", False))

        inst["is_long"] = 1 if (L >= int(FS_SCAN_LONG_MIN)) else 0
        inst["is_tag"] = 1 if bool(best_any.get("is_tag_ecosystem", False)) else 0

        if likely_long:
            ph = int(best_any.get("phase", 0))
            pol = int(best_any.get("polarity", +1))
            frac0 = float(best_any.get("frac", 0.0))
            eps0 = float(best_any.get("eps", 0.0))
            step0 = float(best_any.get("step", float(sps)))

            # tight neighborhood (do NOT blow up runtime)
            frac_ref = (frac0 - 0.12, frac0 - 0.06, frac0, frac0 + 0.06, frac0 + 0.12)
            eps_ref  = (eps0 - 0.006, eps0 - 0.003, eps0, eps0 + 0.003, eps0 + 0.006)
            frac_ref = tuple(max(-0.5, min(0.5, f)) for f in frac_ref)

            crc_before = bool(crc_ok_any)
            inst["ran_local"] = 1

            before = len(hypotheses)
            for eps in eps_ref:
                step = float(sps) * (1.0 + float(eps))
                for frac in frac_ref:
                    phase0 = float(ph) + float(frac) * float(sps)
                    sym = _sym_stream_interp(freq_burst, phase0=phase0, step=step)
                    if sym is None or sym.size < 64:
                        continue

                    x = sym if pol > 0 else -sym
                    bits = (x > 0).astype(np.uint8)

                    aa_pos, aa_corr = correlate_access_address_fast(bits)
                    if aa_pos is None:
                        continue
                    pre_corr = preamble_corr_at(bits, aa_pos)
                    if aa_corr < aa_corr_min or pre_corr < preamble_min:
                        continue

                    _try_decode(bits, aa_pos, aa_corr, pre_corr, ph, frac, eps, step, pol)

            # refresh bests after local refine
            if len(hypotheses) > before:
                hypotheses.sort(
                    key=lambda p: (
                        1 if p.get("crc_ok", False) else 0,
                        p.get("aa_corr", 0),
                        p.get("pre_corr", 0),
                        p.get("length", 0),
                    ),
                    reverse=True,
                )
                best_any = hypotheses[0]
                best_crc = next((p for p in hypotheses if p.get("crc_ok", False)), None)

            if best_crc is not None and (not crc_before):
                inst["crc_ok_local"] = 1

            summary["crc_ok"] = (best_crc is not None)

            # If still no CRC_OK, try a very local step (SPS) refinement.
            if best_crc is None and not crc_ok_any:
                inst["ran_step"] = 1
                crc_before = bool(crc_ok_any)
                _try_local_step_refine(freq_burst, ph=ph, frac0=frac0, pol=pol, step0=step0)

                # IMPORTANT: local step refine appends to hypotheses; re-rank
                if hypotheses:
                    hypotheses.sort(
                        key=lambda p: (
                            1 if p.get("crc_ok", False) else 0,
                            p.get("aa_corr", 0),
                            p.get("pre_corr", 0),
                            p.get("length", 0),
                        ),
                        reverse=True,
                    )
                    best_any = hypotheses[0]
                    best_crc = next((p for p in hypotheses if p.get("crc_ok", False)), None)

                if best_crc is not None and (not crc_before):
                    inst["crc_ok_step"] = 1
                summary["crc_ok"] = (best_crc is not None)

            # If still no CRC_OK, try CFO correction as a second stage
            if best_crc is None:
                inst["ran_cfo"] = 1
                crc_before = bool(crc_ok_any)
                _try_cfo_second_stage(iq_src=iq_burst, ph=ph, frac0=frac0, pol=pol)

                if hypotheses:
                    hypotheses.sort(
                        key=lambda p: (
                            1 if p.get("crc_ok", False) else 0,
                            p.get("aa_corr", 0),
                            p.get("pre_corr", 0),
                            p.get("length", 0),
                        ),
                        reverse=True,
                    )
                    best_any = hypotheses[0]
                    best_crc = next((p for p in hypotheses if p.get("crc_ok", False)), None)

                if best_crc is not None and (not crc_before):
                    inst["crc_ok_cfo"] = 1
                summary["crc_ok"] = (best_crc is not None)

    return best_any, best_crc, summary

# ============================================================
# MULTIPROCESS SHARED-MEM WORKER (minimal additions)
# ============================================================

_MP_FREQ = None
_MP_IQ = None
_MP_SHM_FREQ = None
_MP_SHM_IQ = None

def _mp_init_shared(freq_shm_name, freq_shape, freq_dtype_str,
                    iq_shm_name, iq_shape, iq_dtype_str):
    """Initializer: attach to shared memory once per worker process."""
    global _MP_FREQ, _MP_IQ, _MP_SHM_FREQ, _MP_SHM_IQ
    _MP_SHM_FREQ = shared_memory.SharedMemory(name=freq_shm_name)
    _MP_SHM_IQ = shared_memory.SharedMemory(name=iq_shm_name)
    _MP_FREQ = np.ndarray(tuple(freq_shape), dtype=np.dtype(freq_dtype_str), buffer=_MP_SHM_FREQ.buf)
    _MP_IQ = np.ndarray(tuple(iq_shape), dtype=np.dtype(iq_dtype_str), buffer=_MP_SHM_IQ.buf)

def _mp_close_shared():
    """Best-effort close in workers (not strictly required)."""
    global _MP_SHM_FREQ, _MP_SHM_IQ
    try:
        if _MP_SHM_FREQ is not None:
            _MP_SHM_FREQ.close()
    except Exception:
        pass
    try:
        if _MP_SHM_IQ is not None:
            _MP_SHM_IQ.close()
    except Exception:
        pass

def _keep_filter(pkt, filter_tags: str):
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

def _process_one_window_task(task):
    """
    Worker: process one window using global shared-memory arrays.
    Returns a compact dict for aggregation.
    """
    # Unpack task (keep it flat for speed/serialization)
    (idx, s_up, e_up,
    scale, sps, fs_out, fs_in,   # <-- ADD fs_in
    ch_list,
    aa_corr_min, preamble_min,
    do_crc_filter, debug,
    slip_sweep, slip_max,
    crc_diag, crc_diag_max_unused,
    cfo_window, plot_cfo, filter_tags,
    need_pkt, need_cfo_row) = task

    # Local result skeleton
    out = {
        "idx": idx,
        "bursts": 1,
        "aa_hit": 0,
        "pre_hit": 0,
        "both_hit": 0,
        "parsed": 0,
        "crc_ok": 0,
        "crc_ok_long": 0,   # NEW
        "airtag": 0,
        "tag_any": 0,
        "kept": 0,
        "pkt": None,
        "tag_adva": None,
        "cfo": None,
        "adva_for_cfo": None,
        "is_tag_for_cfo": False,
        "trans": None,
        "cfo_row": None,
        "crc_fail_summary": None,
        "crcok_hist": None,  # optional: (slip, phase, pol, chan) if CRC_OK
        "inst": None,
    }

    # For reporting (original domain): approx inverse-scale
    s_orig = int(round(s_up / scale))
    e_orig = int(round(e_up / scale))

    e_f = max(s_up, e_up - 1)
    if s_up >= _MP_FREQ.size or e_f <= s_up:
        return out

    freq_burst_base = _MP_FREQ[s_up:e_f]
    iq_burst_base = _MP_IQ[s_up:e_f+1]  # +1 so len(iq)==len(freq)+1

    # ---------------------------------------------------------
    # CFO pre-correction (coarse) before decoding
    # ---------------------------------------------------------
    freq_burst_use = freq_burst_base

        # ---------------------------------------------------------
    # Coarse CFO pre-correction + median-centering (works)
    # ---------------------------------------------------------
    # - For GFSK discriminator, constant CFO appears as a DC bias.
    # - We estimate coarse CFO from median(freq_burst) in radians/sample,
    #   rotate IQ, recompute discriminator, then median-center again.
    try:
        # (ii) Coarse CFO that actually works (no AA needed)
        dc_rad = float(np.median(freq_burst_base.astype(np.float64, copy=False)))
        cfo_hz = dc_rad * (fs_out / (2.0 * np.pi))

        iq_corr = apply_cfo_correction(iq_burst_base, cfo_hz, fs_out)
        freq_burst_use = gfsk_discriminator(iq_corr)

        # (i) Median-center per window (very cheap, very effective)
        freq_burst_use = freq_burst_use - np.median(freq_burst_use)

    except Exception:
        # Fallback: just median-center raw discriminator
        freq_burst_use = freq_burst_base - np.median(freq_burst_base)

    except Exception:
        freq_burst_use = freq_burst_base

    burst_flags = {"aa_hit": False, "pre_hit": False, "both_hit": False, "parsed": False, "crc_ok": False}

    best_any = None
    best_any_score = None

    best_crc = None
    best_crc_score = None

    for ch in ch_list:
        any_pkt, crc_pkt, summ = decode_one_burst_from_freq(
            freq_burst=freq_burst_use,
            channel=ch,
            sps=sps,
            aa_corr_min=aa_corr_min,
            preamble_min=preamble_min,
            debug=debug,
            slip_sweep=slip_sweep,
            slip_max=slip_max,
            crc_diag=crc_diag,
            iq_burst=iq_burst_base,
            fs_out=float(fs_out),
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

    if best_crc is not None and best_crc.get("crc_ok", False):
        if int(best_crc.get("length", 0)) >= int(FS_SCAN_LONG_MIN):
            out["crc_ok_long"] = 1

    # Update window-level stats
    if burst_flags["aa_hit"]:
        out["aa_hit"] = 1
    if burst_flags["pre_hit"]:
        out["pre_hit"] = 1
    if burst_flags["both_hit"]:
        out["both_hit"] = 1
    if burst_flags["parsed"]:
        out["parsed"] = 1
    if burst_flags["crc_ok"]:
        out["crc_ok"] = 1

    if best_any is not None:
        if best_any.get("is_airtag", False):
            out["airtag"] = 1
        if best_any.get("is_tag_ecosystem", False):
            out["tag_any"] = 1

    if best_crc is not None and best_crc.get("crc_ok", False):
        out["crcok_hist"] = (
            int(best_crc.get("slip", 0)),
            int(best_crc.get("phase", -1)),
            int(best_crc.get("polarity", 0)),
            int(best_crc.get("channel", -1)),
        )

    # CRC_FAIL summary (printed in main with global limit)
    if crc_diag and best_any is not None and (not best_any.get("crc_ok", False)):
        if best_any.get("aa_corr", 0) >= aa_corr_min and best_any.get("pre_corr", 0) >= preamble_min:
            diag = crc_diag_variants(best_any["crc_rx"], best_any["crc_calc"])
            out["crc_fail_summary"] = (
                "[CRC_FAIL] AA+PRE matched but best decode still CRC_BAD\n"
                f"  window_samp_in=({s_orig},{e_orig}) window_samp_up=({s_up},{e_up}) "
                f"ch={best_any['channel']} phase={best_any['phase']} pol={best_any['polarity']} slip={best_any.get('slip',0)}\n"
                f"  pdu={best_any['pdu_type_name']} len={best_any['length']} AdvA={best_any.get('AdvA','NO_AdvA')}\n"
                f"  crc_rx={diag['rx']} crc_calc={diag['calc']} rx==swap={diag['rx==calc_byteswapped']} xor=0x{diag['rx==calc_xor']:06x}"
            )

    pkt_for_print = best_crc if do_crc_filter else best_any
    if pkt_for_print is None:
        return out

    if not _keep_filter(pkt_for_print, filter_tags):
        return out

    out["kept"] = 1

    # Tag AdvA set for plotting filter later
    if pkt_for_print.get("is_tag_ecosystem", False):
        adva = pkt_for_print.get("AdvA", "NO_AdvA")
        if adva != "NO_AdvA":
            out["tag_adva"] = adva

    # Packet capture for printing (only if requested)
    if need_pkt:
        pkt_for_print = dict(pkt_for_print)  # detach from any shared references
        pkt_for_print["start"] = int(s_orig)
        pkt_for_print["end"] = int(e_orig)
        out["pkt"] = pkt_for_print

    # CFO (only if requested)
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
            out["cfo"] = float(overall_cfo_hz)
            out["adva_for_cfo"] = adva
            out["is_tag_for_cfo"] = bool(pkt_for_print.get("is_tag_ecosystem", False))
            out["trans"] = trans

            if need_cfo_row and out["is_tag_for_cfo"]:
                c00 = float(trans.get("cfo_equal_00", float("nan")))
                c11 = float(trans.get("cfo_equal_11", float("nan")))
                c10 = float(trans.get("cfo_jump_10",  float("nan")))
                c01 = float(trans.get("cfo_jump_01",  float("nan")))
                cft = float(trans.get("cfo_overall_from_transitions", float("nan")))

                aa_pos_i = int(pkt_for_print.get("aa_pos", -1))
                phase_i  = int(pkt_for_print.get("phase", 0))

                if aa_pos_i >= 0 and fs_in > 0:
                    aa_start_up_global = int(s_up + phase_i + aa_pos_i * sps)
                    timestamp_s = aa_start_up_global / fs_out
                else:
                    timestamp_s = float("nan")

                out["cfo_row"] = {
                    "timestamp": timestamp_s,   # <-- ADD THIS
                    "AdvA": adva,
                    "payload": pkt_for_print.get("payload_hex", ""),
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
                }

    # attach per-window instrumentation (choose best available inst)
    if out.get("inst") is None:
        out["inst"] = summ.get("inst", None)

    return out


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
    cfo_csv: str,
    workers: int,
    mp_chunksize: int,
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

    # Resample once + discriminator once
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
        "crc_ok_long": 0,   # NEW: CRC_OK where length >= FS_SCAN_LONG_MIN
        "airtag": 0,
        "tag_any": 0,
        "kept": 0,
        "inst_crc_ok_base": 0,
        "inst_crc_ok_local": 0,
        "inst_crc_ok_step": 0,
        "inst_crc_ok_cfo": 0,
        "inst_ran_local": 0,
        "inst_ran_step": 0,
        "inst_ran_cfo": 0,
        "inst_long_windows": 0,
        "inst_tag_windows": 0,
        "inst_crc_ok_step_long": 0,
        "inst_crc_ok_step_tag": 0,
        "inst_crc_ok_cfo_long": 0,
        "inst_crc_ok_cfo_tag": 0,
    }

    crcok_slip_hist = Counter()
    crcok_phase_hist = Counter()
    crcok_pol_hist = Counter()
    crcok_chan_hist = Counter()

    packets = []
    tag_advas = set()
    cfo_by_adva = defaultdict(list)
    trans_cfo_by_adva = defaultdict(lambda: defaultdict(list))
    cfo_rows = []

    # -------------------------
    # Multiprocessing over windows (minimal change, preserves logic)
    # -------------------------
    use_mp = (workers is not None and int(workers) > 1 and len(windows) > 0)

    # If nothing to do, return quickly
    if not windows:
        return packets, stats

    # Determine what we actually need to return from workers
    need_pkt = (max_packets > 0)
    need_cfo_row = bool(plot_cfo)

    if use_mp:
        # Shared memory to avoid duplicating huge arrays per worker (macOS uses spawn)
        freq_shm = shared_memory.SharedMemory(create=True, size=freq_all.nbytes)
        iq_shm = shared_memory.SharedMemory(create=True, size=iq_up.nbytes)

        try:
            freq_sh = np.ndarray(freq_all.shape, dtype=freq_all.dtype, buffer=freq_shm.buf)
            iq_sh = np.ndarray(iq_up.shape, dtype=iq_up.dtype, buffer=iq_shm.buf)

            # Copy once
            freq_sh[:] = freq_all
            iq_sh[:] = iq_up

            # Build tasks
            tasks = []
            for idx, (s_up, e_up) in enumerate(windows):
                tasks.append((
                    idx, int(s_up), int(e_up),
                    scale, sps, float(fs_out), float(fs_in),   # <-- ADD fs_in
                    ch_list,
                    int(aa_corr_min), int(preamble_min),
                    bool(do_crc_filter), bool(debug),
                    bool(slip_sweep), int(slip_max),
                    bool(crc_diag), int(crc_diag_max),
                    str(cfo_window), bool(plot_cfo), str(filter_tags),
                    bool(need_pkt), bool(need_cfo_row),
                ))


            # Default chunksize: keep tasks per worker moderate to reduce overhead
            chunksize = int(mp_chunksize) if mp_chunksize and mp_chunksize > 0 else 8

            print(f"[INFO] Multiprocessing enabled: workers={workers}, windows={len(windows)}, chunksize={chunksize}")
            ctx = mp.get_context("spawn")

            crc_fail_summaries = []

            with ProcessPoolExecutor(
                max_workers=int(workers),
                mp_context=ctx,
                initializer=_mp_init_shared,
                initargs=(
                    freq_shm.name, freq_all.shape, freq_all.dtype.str,
                    iq_shm.name, iq_up.shape, iq_up.dtype.str,
                ),
            ) as ex:
                # Iterate results as they complete (but we collect then sort by idx for deterministic packet order)
                results = list(ex.map(_process_one_window_task, tasks, chunksize=chunksize))

            # Aggregate in idx order (keeps packet list and diag prints deterministic)
            results.sort(key=lambda d: d.get("idx", 0))

            for r in results:
                stats["bursts"] += int(r.get("bursts", 0))
                stats["aa_hit"] += int(r.get("aa_hit", 0))
                stats["pre_hit"] += int(r.get("pre_hit", 0))
                stats["both_hit"] += int(r.get("both_hit", 0))
                stats["parsed"] += int(r.get("parsed", 0))
                stats["crc_ok"] += int(r.get("crc_ok", 0))
                stats["crc_ok_long"] += int(r.get("crc_ok_long", 0))
                stats["airtag"] += int(r.get("airtag", 0))
                stats["tag_any"] += int(r.get("tag_any", 0))
                stats["kept"] += int(r.get("kept", 0))

                h = r.get("crcok_hist", None)
                if h is not None:
                    slip_v, ph_v, pol_v, ch_v = h
                    crcok_slip_hist[slip_v] += 1
                    crcok_phase_hist[ph_v] += 1
                    crcok_pol_hist[pol_v] += 1
                    crcok_chan_hist[ch_v] += 1

                if crc_diag:
                    s = r.get("crc_fail_summary", None)
                    if s:
                        crc_fail_summaries.append(s)

                ta = r.get("tag_adva", None)
                if ta:
                    tag_advas.add(ta)

                if plot_cfo:
                    cfo = r.get("cfo", None)
                    adva = r.get("adva_for_cfo", None)
                    if cfo is not None and adva is not None:
                        cfo_by_adva[adva].append(float(cfo))

                        if r.get("is_tag_for_cfo", False):
                            trans = r.get("trans", None)
                            if isinstance(trans, dict):
                                c00 = float(trans.get("cfo_equal_00", float("nan")))
                                c11 = float(trans.get("cfo_equal_11", float("nan")))
                                c10 = float(trans.get("cfo_jump_10",  float("nan")))
                                c01 = float(trans.get("cfo_jump_01",  float("nan")))
                                if np.isfinite(c00):
                                    trans_cfo_by_adva[adva]["cfo_equal_00_hz"].append(c00)
                                if np.isfinite(c11):
                                    trans_cfo_by_adva[adva]["cfo_equal_11_hz"].append(c11)
                                if np.isfinite(c10):
                                    trans_cfo_by_adva[adva]["cfo_jump_10_hz"].append(c10)
                                if np.isfinite(c01):
                                    trans_cfo_by_adva[adva]["cfo_jump_01_hz"].append(c01)

                            row = r.get("cfo_row", None)
                            if isinstance(row, dict):
                                cfo_rows.append(row)

                if need_pkt:
                    p = r.get("pkt", None)
                    if p is not None:
                        packets.append(p)

                inst = r.get("inst", None)
                if isinstance(inst, dict):
                    stats["inst_crc_ok_base"] += int(inst.get("crc_ok_base", 0))
                    stats["inst_crc_ok_local"] += int(inst.get("crc_ok_local", 0))
                    stats["inst_crc_ok_step"] += int(inst.get("crc_ok_step", 0))
                    stats["inst_crc_ok_cfo"] += int(inst.get("crc_ok_cfo", 0))
                    stats["inst_ran_local"] += int(inst.get("ran_local", 0))
                    stats["inst_ran_step"] += int(inst.get("ran_step", 0))
                    stats["inst_ran_cfo"] += int(inst.get("ran_cfo", 0))
                    stats["inst_long_windows"] += int(inst.get("is_long", 0))
                    stats["inst_tag_windows"] += int(inst.get("is_tag", 0))

                    if int(inst.get("is_long", 0)):
                        stats["inst_crc_ok_step_long"] += int(inst.get("crc_ok_step", 0))
                        stats["inst_crc_ok_cfo_long"] += int(inst.get("crc_ok_cfo", 0))
                    if int(inst.get("is_tag", 0)):
                        stats["inst_crc_ok_step_tag"] += int(inst.get("crc_ok_step", 0))
                        stats["inst_crc_ok_cfo_tag"] += int(inst.get("crc_ok_cfo", 0))

            # Apply max_packets (same meaning; now deterministic by window order)
            if max_packets > 0 and len(packets) > max_packets:
                packets = packets[:max_packets]

            # Print CRC_FAIL summaries up to crc_diag_max
            if crc_diag and crc_fail_summaries:
                for s in crc_fail_summaries[:max(0, int(crc_diag_max))]:
                    print(s)

        finally:
            # Cleanup shared memory in parent
            try:
                freq_shm.close()
            except Exception:
                pass
            try:
                iq_shm.close()
            except Exception:
                pass
            try:
                freq_shm.unlink()
            except Exception:
                pass
            try:
                iq_shm.unlink()
            except Exception:
                pass

    else:
        # -------------------------
        # Original single-process loop (kept intact)
        # -------------------------
        crc_fail_printed = 0

        def _keep(pkt):
            return _keep_filter(pkt, filter_tags)

        for (s_up, e_up) in windows:
            stats["bursts"] += 1

            s_orig = int(round(s_up / scale))
            e_orig = int(round(e_up / scale))

            e_f = max(s_up, e_up - 1)
            if s_up >= freq_all.size or e_f <= s_up:
                continue

            freq_burst_base = freq_all[s_up:e_f]
            iq_burst_base = iq_up[s_up:e_f+1]  # +1 so len(iq)==len(freq)+1

            # ---------------------------------------------------------
            # Coarse CFO pre-correction + median-centering (works)
            # ---------------------------------------------------------
            try:
                dc_rad = float(np.median(freq_burst_base.astype(np.float64, copy=False)))
                cfo_hz = dc_rad * (fs_out / (2.0 * np.pi))

                iq_corr = apply_cfo_correction(iq_burst_base, cfo_hz, fs_out)
                freq_burst_use = gfsk_discriminator(iq_corr)

                # (i) Median-center per window
                freq_burst_use = freq_burst_use - np.median(freq_burst_use)

            except Exception:
                freq_burst_use = freq_burst_base - np.median(freq_burst_base)

            burst_flags = {"aa_hit": False, "pre_hit": False, "both_hit": False, "parsed": False, "crc_ok": False}

            best_any = None
            best_any_score = None

            best_crc = None
            best_crc_score = None

            win_inst = None

            for ch in ch_list:
                any_pkt, crc_pkt, summ = decode_one_burst_from_freq(
                    freq_burst=freq_burst_use,
                    channel=ch,
                    sps=sps,
                    aa_corr_min=aa_corr_min,
                    preamble_min=preamble_min,
                    debug=debug,
                    slip_sweep=slip_sweep,
                    slip_max=slip_max,
                    crc_diag=crc_diag,
                    iq_burst=iq_burst_base,
                    fs_out=float(fs_out),
                )

                if win_inst is None and isinstance(summ, dict):
                    win_inst = summ.get("inst", None)

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
            if best_crc is not None and best_crc.get("crc_ok", False):
                if int(best_crc.get("length", 0)) >= int(FS_SCAN_LONG_MIN):
                    stats["crc_ok_long"] += 1

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
                    # NOTE: don't call pct() here; it's defined later in the function.
                    print(f"  CRC OK (len>={FS_SCAN_LONG_MIN}B; per-window best_crc)      : {stats['crc_ok_long']}/{stats['bursts']}")
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

                        # --- add this right before cfo_rows.append(...) ---
                        aa_pos_i = int(pkt_for_print.get("aa_pos", -1))
                        phase_i  = int(pkt_for_print.get("phase", 0))

                        if aa_pos_i >= 0:
                            aa_start_up = int(s_up + phase_i + aa_pos_i * sps)          # upsampled-domain sample index
                            aa_start_orig = int(round(aa_start_up / scale))             # original-domain sample index
                            timestamp_s = float(aa_start_orig) / float(fs_in)           # seconds from start of capture
                        else:
                            timestamp_s = float("nan")

                        cfo_rows.append({
                            "timestamp": timestamp_s,
                            "AdvA": adva,
                            "payload": pkt_for_print.get("payload_hex", ""),
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
                            "phase": phase_i,
                            "polarity": int(pkt_for_print.get("polarity", 0)),
                            "slip": int(pkt_for_print.get("slip", 0)),
                            "window_start": int(s_orig),
                            "window_end": int(e_orig),
                            "aa_pos": aa_pos_i,
                            "aa_corr": int(pkt_for_print.get("aa_corr", -1)),
                            "pre_corr": int(pkt_for_print.get("pre_corr", -1)),
                            "cfo_window": cfo_window,
                        })

            if isinstance(win_inst, dict):
                stats["inst_crc_ok_base"] += int(win_inst.get("crc_ok_base", 0))
                stats["inst_crc_ok_local"] += int(win_inst.get("crc_ok_local", 0))
                stats["inst_crc_ok_step"] += int(win_inst.get("crc_ok_step", 0))
                stats["inst_crc_ok_cfo"] += int(win_inst.get("crc_ok_cfo", 0))
                stats["inst_ran_local"] += int(win_inst.get("ran_local", 0))
                stats["inst_ran_step"] += int(win_inst.get("ran_step", 0))
                stats["inst_ran_cfo"] += int(win_inst.get("ran_cfo", 0))
                stats["inst_long_windows"] += int(win_inst.get("is_long", 0))
                stats["inst_tag_windows"] += int(win_inst.get("is_tag", 0))

                if int(win_inst.get("is_long", 0)):
                    stats["inst_crc_ok_step_long"] += int(win_inst.get("crc_ok_step", 0))
                    stats["inst_crc_ok_cfo_long"] += int(win_inst.get("crc_ok_cfo", 0))
                if int(win_inst.get("is_tag", 0)):
                    stats["inst_crc_ok_step_tag"] += int(win_inst.get("crc_ok_step", 0))
                    stats["inst_crc_ok_cfo_tag"] += int(win_inst.get("crc_ok_cfo", 0))

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

    print("[STATS] CRC recovery attribution (per window; mutually non-exclusive flags)")
    print(f"  CRC_OK achieved during base passes            : {stats['inst_crc_ok_base']}")
    print(f"  CRC_OK recovered by local eps/frac refine     : {stats['inst_crc_ok_local']}")
    print(f"  CRC_OK recovered by local step/SPS refine     : {stats['inst_crc_ok_step']}")
    print(f"  CRC_OK recovered by CFO second stage          : {stats['inst_crc_ok_cfo']}")
    print(f"  Windows where local refine ran                : {stats['inst_ran_local']}")
    print(f"  Windows where step/SPS refine ran             : {stats['inst_ran_step']}")
    print(f"  Windows where CFO second stage ran            : {stats['inst_ran_cfo']}")
    print(f"  Windows classified long (len>={FS_SCAN_LONG_MIN}) : {stats['inst_long_windows']}")
    print(f"  Windows classified tag-ecosystem              : {stats['inst_tag_windows']}")
    print(f"  Step refine recovered CRC on long windows     : {stats['inst_crc_ok_step_long']}")
    print(f"  Step refine recovered CRC on tag windows      : {stats['inst_crc_ok_step_tag']}")
    print(f"  CFO stage recovered CRC on long windows       : {stats['inst_crc_ok_cfo_long']}")
    print(f"  CFO stage recovered CRC on tag windows        : {stats['inst_crc_ok_cfo_tag']}")

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

    ap.add_argument("--max", type=int, default=20000000, help="Max packets to store/print (stats still over all windows)")
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

    ap.add_argument("--out-airtag", type=str, default="airtag_packets.txt",
                    help="Write ONLY Tag packets to this text file (includes CRC recheck fields).")

    # Multiprocessing controls (minimal additions)
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker processes for window decoding (0 = use all cores; 1 = disable MP).")
    ap.add_argument("--mp-chunksize", type=int, default=8,
                    help="Chunksize for multiprocessing map over windows (default 8).")

    args = ap.parse_args()

    iq = load_iq_auto(args.filename)
    max_packets = 0 if args.stats_only else args.max

    # Resolve workers
    if args.workers is None or int(args.workers) < 0:
        workers = 0
    else:
        workers = int(args.workers)
    if workers == 0:
        workers = os.cpu_count() or 1
    if workers < 1:
        workers = 1

    def run_one(fs_in_val: float):
        return ble_sniffer(
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
            cfo_csv=args.cfo_csv,
            workers=workers,
            mp_chunksize=args.mp_chunksize,
        )

    # ------------------------------------------------------------
    # Auto fs-in scan path
    # ------------------------------------------------------------
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
        best_score = -1
        best_stats = None

        print(f"[AUTO_FS] Scanning fs-in to maximize CRC_OK for length >= {FS_SCAN_LONG_MIN} ...")

        for fs_try in grid:
            print("=" * 72)
            print(f"[AUTO_FS] fs_in={fs_try:.6f}")

            _, st = run_one(fs_try)

            wins = int(st.get("bursts", 1))
            crc_all = int(st.get("crc_ok", 0))
            crc_long = int(st.get("crc_ok_long", 0))

            score = (1_000_000 * crc_long) + crc_all

            print(f"[AUTO_FS] CRC_OK(all)  ={crc_all}/{wins} ({(100.0*crc_all/max(1,wins)):.2f}%)")
            print(f"[AUTO_FS] CRC_OK(long) ={crc_long}/{wins} ({(100.0*crc_long/max(1,wins)):.2f}%)")

            if score > best_score:
                best_score = score
                best_fs = fs_try
                best_stats = st

        print("=" * 72)

        if best_fs is None:
            print("[AUTO_FS] No candidate produced stats; falling back to provided --fs-in.")
            packets, stats = run_one(center)
            return packets, stats, args

        best_wins = int(best_stats.get("bursts", 1))
        best_crc_all = int(best_stats.get("crc_ok", 0))
        best_crc_long = int(best_stats.get("crc_ok_long", 0))

        print(
            f"[AUTO_FS] Best fs_in={best_fs:.6f}  "
            f"CRC_OK(long)={best_crc_long}/{best_wins}  "
            f"CRC_OK(all)={best_crc_all}/{best_wins}"
        )

        packets, stats = run_one(best_fs)
        return packets, stats, args

    # ------------------------------------------------------------
    # Normal (non-auto-scan) path
    # ------------------------------------------------------------
    packets, stats = run_one(float(args.fs_in))
    return packets, stats, args


if __name__ == "__main__":
    packets, stats, args = main()