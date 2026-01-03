#!/usr/bin/env python3
"""
ble_sniffer.py — BLE 1M legacy advertising sniffer for raw IQ streams (SPI int16 IQ or CF32).

FAST + STATS + CFO (preamble->AA->header+payload; CRC excluded from CFO window):
  - Resamples ONCE for the whole capture
  - Computes discriminator ONCE
  - Fast AA correlation using np.correlate
  - Reports burst-level percentages (preamble / AA / both / parsed / CRC)
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
"""

import os
import argparse
from fractions import Fraction
from collections import defaultdict, Counter

import numpy as np
from scipy.signal import resample_poly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

FS_IN_DEFAULT = 1344106.900141 #1_365_333.33
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

# Burst detector defaults
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
# BURST DETECTOR
# ============================================================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), kernel, mode="same")

def detect_bursts_power(
    iq: np.ndarray,
    fs_in: float,
    thr_k: float = DEFAULT_THR_K,
    smooth_us: float = DEFAULT_SMOOTH_US,
    min_len_us: float = DEFAULT_MIN_LEN_US,
    gap_us: float = DEFAULT_GAP_US,
    pre_us: float = DEFAULT_PRE_US,
    post_us: float = DEFAULT_POST_US,
    debug: bool = False,
):
    N = iq.size
    if N < 64:
        return []

    p = (iq.real * iq.real + iq.imag * iq.imag).astype(np.float32)
    smooth_w = max(1, int(round((smooth_us * 1e-6) * fs_in)))
    env = moving_average(p, smooth_w)

    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-12
    sigma_robust = 1.4826 * mad
    thr = med + thr_k * sigma_robust
    mask = env > thr

    min_len = max(1, int(round((min_len_us * 1e-6) * fs_in)))
    gap = max(1, int(round((gap_us * 1e-6) * fs_in)))
    pre = max(0, int(round((pre_us * 1e-6) * fs_in)))
    post = max(0, int(round((post_us * 1e-6) * fs_in)))

    bursts = []
    i = 0
    while i < N:
        if mask[i]:
            s = i
            while i < N and mask[i]:
                i += 1
            e = i

            while True:
                j = e
                k = min(N, e + gap)
                if j < N and np.any(mask[j:k]):
                    nxt = j + int(np.argmax(mask[j:k]))
                    e = nxt
                    while e < N and mask[e]:
                        e += 1
                else:
                    break

            if (e - s) >= min_len:
                ss = max(0, s - pre)
                ee = min(N, e + post)
                bursts.append((ss, ee))
        i += 1

    bursts.sort()
    merged = []
    for (s, e) in bursts:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    merged = [(int(s), int(e)) for s, e in merged]

    if debug:
        print(f"[DEBUG] env: med={med:.3e} mad={mad:.3e} thr={thr:.3e} smooth_w={smooth_w} bursts={len(merged)}")

    return merged


# ============================================================
# DISCRIMINATOR
# ============================================================

def gfsk_discriminator(iq: np.ndarray) -> np.ndarray:
    prod = iq[1:] * np.conj(iq[:-1])
    return np.angle(prod).astype(np.float32)


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
# CFO ESTIMATION (Hz) over PREAMBLE->AA->HEADER+PAYLOAD (NO CRC)
# ============================================================

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
    window: str = "pre_aa_hdr_payload",  # options below
):
    """
    window options:
      - "pre_aa"              : preamble(8) + AA(32)
      - "pre_aa_hdr"          : preamble(8) + AA(32) + header(16)
      - "pre_aa_hdr_payload"  : preamble(8) + AA(32) + header(16) + payload(8*len)
    """
    if aa_pos is None or aa_pos < 8 or sps <= 0:
        return None

    if window == "pre_aa":
        total_bits = 8 + 32
    elif window == "pre_aa_hdr":
        total_bits = 8 + 32 + 16
    else:
        total_bits = 8 + 32 + 16 + payload_len_bytes * 8

    start = phase + (aa_pos - 8) * sps
    end = start + total_bits * sps

    if start < 0 or end > freq_burst.size or end <= start:
        return None

    seg = freq_burst[start:end].astype(np.float64)
    hz = seg * (fs_out / (2.0 * np.pi))

    if hz.size < 32:
        return float(np.median(hz))

    lo = int(0.10 * hz.size)
    hi = int(0.90 * hz.size)
    hz_sorted = np.sort(hz)
    hz_trim = hz_sorted[lo:hi] if hi > lo else hz_sorted
    return float(np.median(hz_trim))


# ============================================================
# CFO BOXPLOT (group by AdvA only)
# ============================================================

def save_cfo_boxplot_pdf_by_adva(
    cfo_by_adva: dict,
    out_pdf: str,
    top_n: int,
    tag_advas: set,
    min_count: int = 2,   # <-- NEW
):
    items = [(k, v) for k, v in cfo_by_adva.items()
             if (k != "NO_AdvA")
             and (k in tag_advas)
             and (v is not None) and (len(v) >= min_count)]   # <-- CHANGED (>= min_count)

    if not items:
        print(f"[CFO] No CFO samples available to plot (after excluding NO_AdvA, tag-only, and min_count>={min_count}).")
        return False

    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    if top_n and top_n > 0:
        items = items[:top_n]

    labels = [k for k, _ in items]
    print(f"[CFO] Plotting CFO boxplot for top {len(labels)} AdvAs (tag ecosystem only, min_count>={min_count}).")
    data = [v for _, v in items]

    plt.figure(figsize=(max(10, 0.55 * len(labels)), 4.5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.axhline(0.0, linewidth=1.0)
    plt.ylabel("CFO (Hz)")
    plt.title("BLE CFO grouped by AdvA (tag ecosystem only)")
    plt.xticks(rotation=30, ha="right")

    # annotate count inside each box
    ax = plt.gca()
    for i, (_, vals) in enumerate(items, start=1):
        arr = np.asarray(vals, dtype=np.float64)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        ax.text(i, 0.5 * (q1 + q3), f"n={arr.size}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()

    print(f"[CFO] Saved boxplot to: {out_pdf}")
    print(f"[CFO] Groups plotted: {len(labels)} (excluded: NO_AdvA; tag ecosystem only; min_count>={min_count})")
    return True


# ============================================================
# PER-BURST DECODE (BLESDR byte-domain dewhiten+CRC)
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

    # Default small slips; optional full sweep ±slip_max
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

            # Try bit slips around AA boundary
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

                # Header bits -> BLESDR byte domain -> dewhiten header
                hdr_bits = bits[start:start + 16]
                hdr_msb = bytearray(bits_to_bytes_msbfirst_time(hdr_bits))
                hdr_msb_before = bytes(hdr_msb)

                blesdr_reverse_whiten(channel, hdr_msb)
                hdr_msb_after = bytes(hdr_msb)

                # Convert header to standard bytes
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

                # CRC in BLESDR domain
                crc_calc = blesdr_reverse_crc(bytes(pkt_msb[:2+length]), init_adv=True)
                crc_rx = ((pkt_msb[2+length] << 16) |
                          (pkt_msb[2+length+1] << 8) |
                          (pkt_msb[2+length+2])) & 0xFFFFFF
                crc_ok = (crc_rx == crc_calc)

                if crc_ok:
                    crc_ok_any = True

                # Parse standard header/payload
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

                # Optional deterministic CRC diagnostics (only when AA+PRE matched but CRC fails)
                if crc_diag and (not crc_ok):
                    diag = crc_diag_variants(crc_rx, crc_calc)
                    # Print once per (phase,polarity,slip) failure can be too noisy;
                    # keep it gated by --debug as well.
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
):
    L, M = pick_resample_ratio(fs_in, FS_OUT_TARGET, max_den=4096)
    fs_out = fs_in * (L / M)
    sps = int(round(fs_out / SYMBOL_RATE))
    scale = float(L) / float(M)

    print(f"[INFO] Resample ratio L/M = {L}/{M} => fs_out ≈ {fs_out:.2f} (target {FS_OUT_TARGET})")
    print(f"[INFO] SPS = {sps} (fs_out/symbol_rate), CRC_filter={'on' if do_crc_filter else 'off'}")
    print(f"[INFO] Dewhiten: BLESDR byte/MSB stepping (SwapBits(chan)|2, poly 0x11)")
    print(f"[INFO] CRC: BLESDR reverse_crc (init 0x555555 for adv)")
    print(f"[INFO] CFO window: preamble+AA+header+payload (CRC excluded)")
    if plot_cfo:
        print(f"[INFO] CFO grouping: AdvA only (top {cfo_top if cfo_top > 0 else 'ALL'})")
    print(f"[INFO] Tag filter: {filter_tags}")
    if slip_sweep:
        print(f"[INFO] Slip sweep enabled: ±{slip_max} bits around AA")
    if crc_diag:
        print(f"[INFO] CRC diagnostics enabled (max detailed prints={crc_diag_max}, gated by --debug for per-candidate dumps)")
    print(f"[INFO] CFO window: {cfo_window}")

    bursts = detect_bursts_power(
        iq,
        fs_in=fs_in,
        thr_k=thr_k,
        smooth_us=DEFAULT_SMOOTH_US,
        min_len_us=DEFAULT_MIN_LEN_US,
        gap_us=DEFAULT_GAP_US,
        pre_us=DEFAULT_PRE_US,
        post_us=DEFAULT_POST_US,
        debug=debug,
    )
    if max_bursts > 0:
        bursts = bursts[:max_bursts]
    print(f"[INFO] Bursts detected (analyzed): {len(bursts)}")

    iq_up = upsample(iq, L, M)
    freq_all = gfsk_discriminator(iq_up)

    ch_list = [37, 38, 39] if try_adv_channels else [channel]

    stats = {
        "bursts": 0,
        "aa_hit": 0,
        "pre_hit": 0,
        "both_hit": 0,
        "parsed": 0,
        "crc_ok": 0,
        "airtag": 0,
        "tag_any": 0,
        "kept": 0,
    }

    # Debug histograms (deterministic)
    crcok_slip_hist = Counter()
    crcok_phase_hist = Counter()
    crcok_pol_hist = Counter()
    crcok_chan_hist = Counter()

    # Limited, high-signal “why CRC failed?” prints (per-burst best_any only)
    crc_fail_printed = 0

    packets = []
    tag_advas = set()
    cfo_by_adva = defaultdict(list)

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

    for (s, e) in bursts:
        stats["bursts"] += 1

        s_up = int(round(s * scale))
        e_up = int(round(e * scale))
        e_f = max(s_up, e_up - 1)

        if s_up >= freq_all.size or e_f <= s_up:
            continue

        freq_burst_base = freq_all[s_up:e_f]

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

        # Histograms for CRC-OK outcomes (deterministic indicators)
        if best_crc is not None and best_crc.get("crc_ok", False):
            crcok_slip_hist[best_crc.get("slip", 0)] += 1
            crcok_phase_hist[best_crc.get("phase", -1)] += 1
            crcok_pol_hist[best_crc.get("polarity", 0)] += 1
            crcok_chan_hist[best_crc.get("channel", -1)] += 1

        # High-signal CRC failure summary (per burst, best_any only)
        if crc_diag and best_any is not None and (not best_any.get("crc_ok", False)) and (crc_fail_printed < crc_diag_max):
            if best_any.get("aa_corr", 0) >= aa_corr_min and best_any.get("pre_corr", 0) >= preamble_min:
                diag = crc_diag_variants(best_any["crc_rx"], best_any["crc_calc"])
                print("[CRC_FAIL] AA+PRE matched but best decode still CRC_BAD")
                print(f"  burst_samp=({s},{e}) ch={best_any['channel']} phase={best_any['phase']} pol={best_any['polarity']} slip={best_any.get('slip',0)}")
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
            pkt_for_print["start"] = int(s)
            pkt_for_print["end"] = int(e)
            packets.append(pkt_for_print)

        if plot_cfo:
            cfo_hz = estimate_cfo_hz(
                freq_burst=freq_burst_base,
                fs_out=fs_out,
                phase=pkt_for_print["phase"],
                aa_pos=pkt_for_print["aa_pos"],
                sps=sps,
                payload_len_bytes=pkt_for_print["length"],
                window=cfo_window,
            )
            if cfo_hz is not None and np.isfinite(cfo_hz):
                adva = pkt_for_print.get("AdvA", "NO_AdvA")
                cfo_by_adva[adva].append(float(cfo_hz))

    def pct(x, denom):
        return 0.0 if denom <= 0 else 100.0 * x / float(denom)

    print("[STATS] Burst-level detection quality")
    print(f"  Bursts total                                  : {stats['bursts']}")
    print(f"  AA match (>= {aa_corr_min}/32)                            : {stats['aa_hit']} ({pct(stats['aa_hit'], stats['bursts']):.2f}%)")
    print(f"  Preamble match (>= {preamble_min}/8)                      : {stats['pre_hit']} ({pct(stats['pre_hit'], stats['bursts']):.2f}%)")
    print(f"  BOTH (AA & preamble)                           : {stats['both_hit']} ({pct(stats['both_hit'], stats['bursts']):.2f}%)")
    print(f"  Parsed header/payload (BOTH-based)             : {stats['parsed']} ({pct(stats['parsed'], stats['bursts']):.2f}%)")
    print(f"  CRC OK (any BOTH-based candidate)              : {stats['crc_ok']} ({pct(stats['crc_ok'], stats['bursts']):.2f}%)")
    print(f"  AirTag/FindMy (best_any per burst)             : {stats['airtag']} ({pct(stats['airtag'], stats['bursts']):.2f}%)")
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
        saved = save_cfo_boxplot_pdf_by_adva(cfo_by_adva, cfo_pdf, top_n=cfo_top, tag_advas=tag_advas, min_count=5)
        if saved:
            total_cfo = sum(len(v) for v in cfo_by_adva.values())
            uniq = sum(1 for v in cfo_by_adva.values() if len(v) > 0)
            print(f"[CFO] Total CFO samples plotted: {total_cfo}  (unique AdvA groups: {uniq})")

    return packets, stats


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filename", help="IQ file: .cf32/.cfile or raw SPI .bin")
    ap.add_argument("--fs-in", type=float, default=FS_IN_DEFAULT, help="Input IQ sampling rate (Hz)")
    ap.add_argument("--channel", type=int, default=37, help="BLE channel (37/38/39)")
    ap.add_argument("--try-adv-channels", action="store_true", help="Try channels 37/38/39 per burst")

    ap.add_argument("--max", type=int, default=20000, help="Max packets to store/print (stats still over all bursts)")
    ap.add_argument("--stats-only", action="store_true", help="Only print STATS (and optional CFO plot), do not print packets")
    ap.add_argument("--max-bursts", type=int, default=0, help="Analyze only first N bursts (0 = all)")

    ap.add_argument("--aa-min", type=int, default=32, help="Min AA bit-match correlation (0..32)")
    ap.add_argument("--pre-min", type=int, default=8, help="Min preamble bit-match (0..8)")
    ap.add_argument("--thr", type=float, default=DEFAULT_THR_K, help="Burst threshold multiplier (MAD-based)")

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

    # CFO / plotting
    ap.add_argument("--plot-cfo", action="store_true", help="Compute CFO per decoded candidate and save a boxplot PDF")
    ap.add_argument("--cfo-pdf", type=str, default="cfo_boxplot_by_adva.pdf", help="Output PDF path for CFO boxplot")
    ap.add_argument("--cfo-top", type=int, default=100, help="Plot only top-N AdvA groups (by sample count). 0 = all")

    # Tag filter
    ap.add_argument(
        "--filter-tags",
        type=str,
        default="none",
        choices=["none", "drop-airtag", "only-airtag", "drop-tags", "only-tags"],
        help="Filter packets by AirTag/FindMy or tag-ecosystem detection"
    )

    # Debugs requested
    ap.add_argument("--slip-sweep", action="store_true", help="Sweep bit-slip around AA boundary to deterministically detect off-by-k alignment")
    ap.add_argument("--slip-max", type=int, default=8, help="Max slip magnitude (bits) used with --slip-sweep (default ±8)")
    ap.add_argument("--crc-diag", action="store_true", help="Enable CRC diagnostics and histograms (deterministic)")
    ap.add_argument("--crc-diag-max", type=int, default=20, help="Max per-burst CRC_FAIL summaries to print (default 20)")

    # Optional: fs scan (deterministic check for wrong fs/SPS)
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

        best = None
        best_fs = None
        best_crc = -1

        print("[AUTO_FS] Scanning fs-in to maximize CRC_OK ...")
        for fs_try in grid:
            print("=" * 72)
            print(f"[AUTO_FS] fs_in={fs_try:.6f}")
            _, st = run_one(fs_try)
            crc_ok = st.get("crc_ok", 0)
            bursts = st.get("bursts", 1)
            print(f"[AUTO_FS] CRC_OK={crc_ok}/{bursts} ({(100.0*crc_ok/max(1,bursts)):.2f}%)")
            if crc_ok > best_crc:
                best_crc = crc_ok
                best_fs = fs_try
                best = st

        print("=" * 72)
        print(f"[AUTO_FS] Best fs_in={best_fs:.6f}  CRC_OK={best_crc}/{best.get('bursts',1)}")
        # re-run best fs with packet printing if not stats-only
        if best_fs is not None:
            run_one(best_fs)
        return

    packets, _ = run_one(args.fs_in)

    if args.stats_only:
        return

    for p in packets:
        print("---- BLE PACKET ----")
        print("AA corr:", p["aa_corr"], "AA pos:", p["aa_pos"], "PRE corr:", p.get("pre_corr", -1))
        print("Channel:", p["channel"], "Phase:", p["phase"], "Polarity:", p["polarity"])
        print("Slip:", p.get("slip", 0))
        print("PDU:", p["pdu_type_name"], f"(type={p['pdu_type']})")
        print("Len:", p["length"], "TxAdd:", p["txadd"], "RxAdd:", p["rxadd"])
        if "AdvA" in p:
            print("AdvA:", p["AdvA"])
        if "ScanA" in p:
            print("ScanA:", p["ScanA"])
        if "InitA" in p:
            print("InitA:", p["InitA"])
        if "TargetA" in p:
            print("TargetA:", p["TargetA"])
        print(f"Tag: is_airtag={p.get('is_airtag', False)}  is_tag_ecosystem={p.get('is_tag_ecosystem', False)}  reason={p.get('tag_reason','')}")
        print(f"CRC ok: {p['crc_ok']}  CRC rx: 0x{p['crc_rx']:06x}  CRC calc: 0x{p['crc_calc']:06x}")
        print()

if __name__ == "__main__":
    main()