#!/usr/bin/env python3
"""
features_from_iqdir.py

Compute BLE PHY features for pkt_*.fc32 bursts and *enhance* the output CSV
by attaching per-packet metadata:
  - adv_addr          (advertiser MAC, aa:bb:cc:dd:ee:ff; lower-case)
  - pdu_type          (integer, e.g., 0x00 ADV_IND, 0x03 SCAN_REQ, etc.)
  - access_address    (8-hex string, big-endian, e.g., 8E89BED6 for adv channels)
  - pcap_ts           (float seconds from PCAP ts_sec + ts_usec; when --pcap used)

Metadata sources (priority):
  1) --index-csv file with columns: file,pdu_type,adv_addr,access_address[,pcap_ts]
  2) --pcap file (DLT 256 or 251). If filenames have numeric indices (pkt_000123.fc32),
     rows are aligned by that index; otherwise by order.

You can also filter which bursts to include with:
  --pdu-allow   (comma list, e.g. "0x00,0x02,0x04,0x06")
  --adv-addr    (only include a specific advertiser MAC)

Example:
  python3 features_from_iqdir.py \
    --dir iq_dump_37_2 \
    --fs 2e6 \
    --out-csv features_ch37_2.csv \
    --plots-dir plots_37_2 \
    --make-cdfs --make-hists \
    --pcap out_ch37.pcap
"""

import argparse, csv, os, re, sys, struct
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------- Basic helpers ----------------
def rm_dc(x): return x - np.mean(x)

def norm_pow(x):
    p = np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
    return x / p

def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")

def _flt(v):
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")

def _symbol_rate(fs_hz):
    """BLE 1M → ~2 sps at 2 Msps after decim; keep at least 2."""
    return max(2, int(round(fs_hz/1e6)))

def _disc(x):  # frequency discriminator (rad/sample)
    return np.angle(x[1:] * np.conj(x[:-1]))

def _preamble_ref_symbols(n_symbols):
    # alternating reference (+1, -1, +1, ...)
    r = np.fromiter(((+1) if (k & 1) else -1 for k in range(n_symbols)), float)
    return r

def _downsample_to_symbols(arr, sps):
    L = (len(arr)//sps)*sps
    if L < sps:
        return np.array([])
    a = arr[:L].reshape(-1, sps).mean(axis=1)
    return a

def lock_cfo_sign_with_preamble(x, fs, n_ref_sym=40):
    sps = _symbol_rate(fs)
    d = _disc(x)
    d_sym = _downsample_to_symbols(d, sps)
    if d_sym.size < n_ref_sym:
        return +1
    d_win = d_sym[:n_ref_sym]
    r = _preamble_ref_symbols(n_ref_sym)
    c = np.dot(np.sign(d_win), r)
    return +1 if c >= 0 else -1

# ---------------- IQ / envelope features ----------------
def iq_imbalance(x):
    I, Q = x.real, x.imag
    mII, mQQ, mIQ = np.mean(I*I), np.mean(Q*Q), np.mean(I*Q)
    alpha = np.sqrt(max(mII,1e-16) / max(mQQ,1e-16))
    phi = 0.5*np.arctan2(2*mIQ, (mII - mQQ + 1e-16))
    return float(alpha), float(np.degrees(phi))

def rise_time_us(x, fs, tail=200):
    env = np.abs(x)
    steady = np.mean(env[-tail:]) if len(env) > tail else np.mean(env)
    if steady <= 0:
        return 0.0
    n10 = int(np.argmax(env >= 0.1*steady))
    n90 = int(np.argmax(env >= 0.9*steady))
    return (n90-n10)*1e6/fs if n90 > n10 else 0.0

def gate_burst(x, fs, pre_us=60, post_us=60, smooth_us=8, thr_db_above=5):
    env = np.abs(x)
    k = max(1, int(round(smooth_us*1e-6*fs)))
    if k > 1:
        win = np.ones(k)/k
        env = np.convolve(env, win, mode="same")
    nf = np.median(env) + 1e-12
    thr = nf * (10**(thr_db_above/20))
    thr = min(thr, 0.5*np.max(env))  # clamp
    on = np.where(env >= thr)[0]
    if on.size == 0:
        return x
    i0, i1 = on[0], on[-1]
    pad_pre  = int(round(pre_us*1e-6*fs))
    pad_post = int(round(post_us*1e-6*fs))
    a = max(0, i0 - pad_pre); b = min(len(x), i1 + pad_post + 1)
    return x[a:b]

# ---------------- PSD / spectral stats ----------------
def welch_psd_complex(x, fs, nfft=4096):
    L = min(len(x), nfft)
    if L < 32:
        f = np.array([0.0]); S = np.array([0.0])
        return f, S
    w = np.hanning(L)
    X = np.fft.fft(w * x[:L], n=L)
    S = (np.abs(X)**2) / (np.sum(w**2) + 1e-18)
    S = np.fft.fftshift(S)
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1/fs))
    return f, S

def spectral_stats(x, fs):
    f, S = welch_psd_complex(x, fs)
    S = np.real(S) + 1e-18
    centroid = float(np.sum(f*S) / np.sum(S))
    pnr = float(10*np.log10(np.max(S) / (np.median(S) + 1e-18)))
    bw_mask = S > (np.max(S) * 10**(-3/10))
    bw = float((f[bw_mask][-1] - f[bw_mask][0]) if np.any(bw_mask) else 0.0)
    return centroid, pnr, bw

# ---------------- CFO estimators ----------------
def estimate_cfo(x, fs):
    if len(x) < 2:
        return 0.0
    ph = np.angle(x[1:] * np.conj(x[:-1]))
    return float((fs/(2*np.pi)) * np.median(ph))

def cfo_std_symbol_avg(x, fs):
    sps = _symbol_rate(fs)
    if len(x) < sps + 2:
        return 0.0
    ph = np.angle(x[1:] * np.conj(x[:-1]))
    k = np.ones(sps) / sps
    ph_avg = np.convolve(ph, k, mode='valid')[::sps]
    return float((fs/(2*np.pi)) * np.std(ph_avg))

def _find_burst_start(x, fs, smooth_us=8, thr_db_above=6):
    env = np.abs(x)
    k = max(1, int(round(smooth_us*1e-6*fs)))
    if k > 1:
        env = np.convolve(env, np.ones(k)/k, mode="same")
    nf = np.median(env) + 1e-12
    thr = nf * (10**(thr_db_above/20))
    on = np.where(env >= thr)[0]
    return int(on[0]) if on.size else 0

def ble_adv_preamble_aa_bits():
    pre = [1,0,1,0,1,0,1,0]  # 0xAA
    aa = 0x8E89BED6
    aa_bits = []
    for byte in aa.to_bytes(4, 'little'):
        for b in range(8):
            aa_bits.append((byte >> b) & 1)
    seq = np.array([1 if b else -1 for b in (pre + aa_bits)], dtype=float)
    return seq

def correlate_preamble_aa_symbol_aligned(x, fs):
    sps = _symbol_rate(fs)
    min_syms = 48
    if len(x) < sps * min_syms:
        return 0, sps, 0.0
    d = np.angle(x[1:] * np.conj(x[:-1]))
    Ls = (len(d) // sps)
    if Ls < min_syms:
        return 0, sps, 0.0
    d_sym = d[:Ls*sps].reshape(Ls, sps).mean(axis=1)
    ref = ble_adv_preamble_aa_bits().astype(float)
    m = len(ref)
    if Ls < m:
        return 0, sps, 0.0
    ds = np.sign(d_sym)
    best_i, best_val = 0, -1e9
    ref_norm = np.linalg.norm(ref) + 1e-12
    for i in range(Ls - m + 1):
        seg = ds[i:i+m]
        val = float(np.dot(seg, ref) / ((np.linalg.norm(seg) + 1e-12) * ref_norm))
        if val > best_val:
            best_val, best_i = val, i
    return best_i, sps, float(best_val)

def estimate_cfo_centroid(x, fs, nfft=16384, f_lim_hz=200e3):
    L = min(len(x), nfft)
    if L < 32:
        return 0.0
    w = np.hanning(L)
    X = np.fft.fft(w * x[:L], n=L)
    S = np.abs(X)**2
    S = np.fft.fftshift(S)
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1/fs))
    mask = np.abs(f) <= float(f_lim_hz)
    if not np.any(mask):
        return 0.0
    f_win = f[mask]
    S_win = S[mask] + 1e-18
    return float(np.sum(f_win * S_win) / np.sum(S_win))

def _cfo_ls(seg, fs):
    if len(seg) < 4:
        return 0.0
    ph = np.unwrap(np.angle(seg))
    t  = np.arange(len(seg)) / fs
    slope = np.polyfit(t, ph, 1)[0]
    return float(slope / (2*np.pi))

def _nudge_window_for_max_slope_energy(x, a, b, fs, sps, search_half_syms=1):
    best_a, best_b = a, b
    best_val = -1.0
    half = int(search_half_syms * sps)
    for delta in range(-half, half+1):
        aa = max(0, a + delta)
        bb = min(len(x), b + delta)
        if bb - aa < sps * 20:
            continue
        seg = x[aa:bb]
        dp = seg[1:] * np.conj(seg[:-1])
        val = np.abs(np.sum(dp)) / (len(dp) + 1e-12)
        if val > best_val:
            best_val = val
            best_a, best_b = aa, bb
    return best_a, best_b

def estimate_cfo_preamble_aa(x, fs, bits=80, settle_us=6):
    sps = _symbol_rate(fs)
    n_settle = int(round(settle_us*1e-6*fs))
    n_pkt = int(bits * sps)
    i0 = _find_burst_start(x, fs)
    a = min(max(i0 + n_settle, 0), len(x))
    b = min(a + n_pkt, len(x))
    seg = x[a:b] if (b - a) >= 8 else x
    if len(seg) < 2:
        return 0.0, (a, b)
    ph = np.angle(seg[1:] * np.conj(seg[:-1]))
    cfo_hz = (fs/(2*np.pi)) * np.median(ph)
    return float(cfo_hz), (a, b)

def estimate_cfo_two_stage(x, fs,
                           bits=160, settle_us=8,
                           fwin1=200e3, fwin2=120e3,
                           min_symbols=120, min_corr=0.45,
                           do_nudge=True):
    f0 = estimate_cfo_centroid(x, fs, f_lim_hz=fwin1)
    n = np.arange(len(x))
    x0 = x * np.exp(-1j * 2*np.pi * f0 * n / fs)
    f1 = estimate_cfo_centroid(x0, fs, f_lim_hz=fwin2)
    x1 = x0 * np.exp(-1j * 2*np.pi * f1 * n / fs)
    f_coarse = f0 + f1
    i_burst = _find_burst_start(x1, fs)
    sym0, sps, score = correlate_preamble_aa_symbol_aligned(x1[i_burst:], fs)
    if score < min_corr:
        return None, (0, 0), dict(reason="low_corr", score=score, coarse=f_coarse)
    sps = int(sps)
    num_sym = max(bits, min_symbols)
    n_settle = int(round(settle_us*1e-6*fs))
    a = i_burst + sym0 * sps + n_settle
    b = min(a + num_sym * sps, len(x1))
    if b - a < (min_symbols * sps):
        return None, (a, b), dict(reason="short_window", score=score, coarse=f_coarse)
    if do_nudge:
        a, b = _nudge_window_for_max_slope_energy(x1, a, b, fs, sps, search_half_syms=1)
    seg = x1[a:b]
    cfo_fine = _cfo_ls(seg, fs)
    return float(f_coarse + cfo_fine), (a, b), dict(score=score, coarse=f_coarse)

# ---------------- Index/PCAP helpers ----------------
def _read_index_csv(path):
    """
    Read an index CSV with columns: file,pdu_type,adv_addr,access_address[,pcap_ts]
    - file: IQ filename (e.g., pkt_000123.fc32)
    - pdu_type: like 0x00 or 3
    - adv_addr: aa:bb:cc:dd:ee:ff (case/sep flexible)
    - access_address: 8-hex like 8E89BED6 (case flexible)
    - pcap_ts: float seconds (optional)
    """
    idx = {}
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fn = row.get("file") or row.get("filename")
            if not fn:
                continue
            fn = Path(fn).name
            # pdu_type
            pt = row.get("pdu_type")
            try:
                pt_int = int(str(pt), 0) if pt is not None and str(pt).strip() != "" else None
            except Exception:
                pt_int = None
            # adv addr
            aa = row.get("adv_addr") or row.get("advertising_address")
            adv = str(aa).strip().lower() if aa else None
            # access address
            acc = row.get("access_address") or row.get("aa") or row.get("accessaddr")
            if isinstance(acc, str):
                acc = acc.strip().replace("0x", "").replace(":", "")
                acc = acc.upper() if re.fullmatch(r"[0-9A-Fa-f]{8}", acc or "") else None
            else:
                acc = None
            # optional timestamp
            ts = row.get("pcap_ts")
            try:
                ts_f = float(ts) if ts is not None and str(ts).strip() != "" else None
            except Exception:
                ts_f = None
            idx[fn] = {"pdu_type": pt_int, "adv_addr": adv, "access_address": acc, "pcap_ts": ts_f}
    return idx

def _extract_access_address(buf, incl, linktype):
    """
    Return (AA_hexstr, LL_off) or (None, LL_off) if cannot parse.
    For DLT 256: [10B hdr][4B AA][LL...]; for DLT 251: [4B AA][LL...]
    """
    if linktype == 256:
        if incl >= 10 + 4:
            aa_bytes = buf[10:14]
            aa = "".join(f"{b:02X}" for b in aa_bytes[::-1])  # AA is little-endian; display big-endian
            return aa, 10 + 4
        return None, 10 + 4
    elif linktype == 251:
        if incl >= 4:
            aa_bytes = buf[0:4]
            aa = "".join(f"{b:02X}" for b in aa_bytes[::-1])
            return aa, 4
        return None, 4
    else:
        return None, 0

def _parse_pcap_records(pcap_path):
    """
    Parse PCAP and return a list of dicts with keys:
      {'pdu_type': int or None,
       'adv_addr': str or None,
       'access_address': str or None,
       'pcap_ts': float or None}
    """
    recs = []
    with open(pcap_path, "rb") as f:
        gh = f.read(24)
        if len(gh) != 24:
            raise RuntimeError("PCAP: short global header")
        magic_le = struct.unpack("<I", gh[:4])[0]
        magic_be = struct.unpack(">I", gh[:4])[0]
        if magic_le == 0xA1B2C3D4:
            endian = "<"
            _,_,_,_,_,_, linktype = struct.unpack("<IHHIIII", gh)
        elif magic_be == 0xA1B2C3D4:
            endian = ">"
            _,_,_,_,_,_, linktype = struct.unpack(">IHHIIII", gh)
        else:
            raise RuntimeError("PCAP: unsupported magic")
        rec_hdr = endian + "IIII"
        while True:
            rh = f.read(16)
            if len(rh) != 16: break
            ts_sec, ts_usec, incl, _ = struct.unpack(rec_hdr, rh)
            buf = f.read(incl)
            if len(buf) != incl: break
            ts = ts_sec + ts_usec * 1e-6

            aa_hex, LL_OFF = _extract_access_address(buf, incl, linktype)
            pdu = None
            adv = None
            if incl >= LL_OFF + 2 + 3:
                h0 = buf[LL_OFF+0] if (LL_OFF+0) < incl else 0
                h1 = buf[LL_OFF+1] if (LL_OFF+1) < incl else 0
                pdu = h0 & 0x0F
                ln  = h1 & 0x3F
                if incl >= LL_OFF + 2 + ln + 3:
                    pl = buf[LL_OFF+2 : LL_OFF+2+ln]
                    if pdu in (0,2,6,4):          # ADV_IND, ADV_NONCONN_IND, ADV_SCAN_IND, SCAN_RSP
                        if len(pl) >= 6:
                            adv = ":".join(f"{b:02X}" for b in pl[0:6][::-1]).lower()
                    elif pdu in (3,5):           # SCAN_REQ, CONNECT_REQ
                        if len(pl) >= 12:
                            adv = ":".join(f"{b:02X}" for b in pl[6:12][::-1]).lower()

            recs.append({"pdu_type": pdu,
                         "adv_addr": adv,
                         "access_address": aa_hex,
                         "pcap_ts": ts})
    return recs

def _filename_index(fn):
    m = re.search(r"(\d+)", fn)
    return int(m.group(1)) if m else None

def build_mapping(files, index_csv=None, pcap_path=None):
    """
    Build {filename -> {'pdu_type','adv_addr','access_address','pcap_ts'}}.
    Priority: index CSV > PCAP > {}.
    If PCAP used, first try filename numeric index; else align by order.
    """
    if index_csv:
        idx = _read_index_csv(index_csv)
        return {Path(f).name: idx.get(Path(f).name, {}) for f in files}

    if pcap_path:
        recs = _parse_pcap_records(pcap_path)
        file_nums = [(_filename_index(Path(f).name), Path(f).name) for f in files]
        if all(n is not None for n, _ in file_nums):
            mapping = {}
            for (n, name) in file_nums:
                if 0 <= n < len(recs):
                    mapping[name] = recs[n]
            return mapping
        else:
            n = min(len(files), len(recs))
            return {Path(files[i]).name: recs[i] for i in range(n)}

    return {Path(f).name: {} for f in files}

# ---------------- Packet feature packer ----------------
def packet_features(x, fs):
    x = norm_pow(rm_dc(x))
    fcent, pnr, bw = spectral_stats(x, fs)
    gated_len_us = len(x) * 1e6 / fs
    alpha, phi_deg = iq_imbalance(x)
    rt_us = rise_time_us(x, fs)
    PNR_MIN = 10.0
    MIN_US  = 120.0
    keep = (pnr >= PNR_MIN) and (gated_len_us >= MIN_US)

    if keep:
        cfo_full = estimate_cfo(x, fs)
        cfo_centroid = estimate_cfo_centroid(x, fs, f_lim_hz=200e3)
        cfo_two, (iA, iB), info = estimate_cfo_two_stage(
            x, fs, bits=160, settle_us=8, fwin1=200e3, fwin2=120e3,
            min_symbols=120, min_corr=0.45, do_nudge=True
        )
        if cfo_two is None:
            cfo_two = np.nan
            cfo_preaa_hz, _ = estimate_cfo_preamble_aa(x, fs, bits=80, settle_us=6)
        else:
            cfo_preaa_hz = np.nan
    else:
        cfo_full = np.nan
        cfo_centroid = np.nan
        cfo_two = np.nan
        cfo_preaa_hz = np.nan
        info = dict(score=np.nan, coarse=np.nan, reason="quality_gated")

    if keep and len(x) >= 2:
        cfo_inst = (fs/(2*np.pi)) * np.angle(x[1:] * np.conj(x[:-1]))
        cfo_std_all = float(np.std(cfo_inst))
        cfo_std_sym = cfo_std_symbol_avg(x, fs)
    else:
        cfo_std_all = np.nan
        cfo_std_sym = np.nan

    return {
        "cfo_hz": _flt(cfo_full),
        "cfo_two_stage_hz": _flt(cfo_two),
        "cfo_preaa_hz": _flt(cfo_preaa_hz),
        "cfo_centroid_hz": _flt(cfo_centroid),
        "cfo_std_hz": _flt(cfo_std_all),
        "cfo_std_hz_sym": _flt(cfo_std_sym),
        "iq_gain_alpha": _flt(alpha),
        "iq_phase_deg": _flt(phi_deg),
        "rise_time_us": _flt(rt_us),
        "psd_centroid_hz": _flt(fcent),
        "psd_pnr_db": _flt(pnr),
        "bw_3db_hz": _flt(bw),
        "mag_mean": _flt(np.mean(np.abs(x))),
        "i_std": _flt(np.std(np.real(x))),
        "q_std": _flt(np.std(np.imag(x))),
        "gated_len_us": _flt(gated_len_us),
        "cfo_two_stage_score": _flt(info.get("score", np.nan)),
        "cfo_two_stage_coarse_hz": _flt(info.get("coarse", np.nan)),
    }

# ---------------- Plot helpers ----------------
def ecdf(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(a)
    n = xs.size
    # F = np.arange(1, n+1), dtype=float
    F = np.arange(1, n+1, dtype=float) / n
    return xs, F

def save_cdf(feature_name, values, outdir):
    xs, F = ecdf(values)
    if xs.size == 0:
        return None
    fig = plt.figure()
    plt.plot(xs, F, drawstyle="steps-post")
    plt.xlabel(feature_name)
    plt.ylabel("F(x)")
    plt.title(f"CDF: {feature_name}  (N={len(values)})")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    path = os.path.join(outdir, f"cdf_{sanitize(feature_name)}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path

def save_hist(feature_name, values, outdir, bins=60):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    fig = plt.figure()
    plt.hist(a, bins=bins)
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(f"Histogram: {feature_name}  (N={len(values)})")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    path = os.path.join(outdir, f"hist_{sanitize(feature_name)}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with pkt_*.fc32 files (float32 interleaved I,Q)")
    ap.add_argument("--fs", type=float, required=True, help="Complex sample rate AFTER decim (e.g., 2e6)")
    ap.add_argument("--out-csv", default="features.csv")
    ap.add_argument("--plots-dir", default="plots", help="Directory to save plots (created if missing)")
    ap.add_argument("--make-cdfs", action="store_true", default=True, help="Make one CDF per feature (default on)")
    ap.add_argument("--no-cdfs", action="store_true", help="Disable CDF plots")
    ap.add_argument("--make-hists", action="store_true", help="Also save histograms per feature")
    ap.add_argument("--max-files", type=int, default=0, help="Process at most this many packets (0 = all)")
    # NEW: metadata sources / filtering
    ap.add_argument("--index-csv", help="CSV mapping: file,pdu_type,adv_addr,access_address[,pcap_ts]")
    ap.add_argument("--pcap", help="PCAP (DLT 256/251) to auto-extract pdu_type/adv_addr/access_address/pcap_ts")
    ap.add_argument("--adv-addr", help="Only process bursts from this advertiser MAC (aa:bb:cc:dd:ee:ff)")
    ap.add_argument("--pdu-allow", default="0x00,0x02,0x04,0x06",
                    help="Comma list of PDU types to include (default advertiser-originated)")
    args = ap.parse_args()
    make_cdfs = args.make_cdfs and not args.no_cdfs

    files = sorted(Path(args.dir).glob("pkt_*.fc32"))
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        print("No pkt_*.fc32 files found.", file=sys.stderr)
        return

    # Build per-file metadata: pdu_type, adv_addr, access_address, pcap_ts
    meta_map = build_mapping(files, index_csv=args.index_csv, pcap_path=args.pcap)

    # Filtering by PDU / AdvA
    allow_set = set(int(s.strip(), 0) for s in args.pdu_allow.split(",") if s.strip())
    adv_filter = args.adv_addr.lower() if args.adv_addr else None

    def _keep(fn):
        m = meta_map.get(fn.name, {})
        pt = m.get("pdu_type")
        aa = m.get("adv_addr")
        if pt is None or pt not in allow_set:
            return False
        if adv_filter and aa != adv_filter:
            return False
        return True

    # If we have metadata, apply filtering; if not, keep all (but still write empty fields)
    if any(meta_map.values()):
        files = [fn for fn in files if _keep(fn)]
        if not files:
            print("No files after PDU/MAC filtering (check --index-csv/--pcap and filters).")
            return

    rows = []
    for i, fn in enumerate(files):
        iq = np.fromfile(fn, dtype=np.float32)
        if iq.size < 4:
            continue
        x = iq.view(np.complex64)
        x = gate_burst(x, args.fs)
        feats = packet_features(x, args.fs)
        feats["file"] = fn.name

        # Attach metadata (may be missing)
        meta = meta_map.get(fn.name, {}) if meta_map else {}
        adv = meta.get("adv_addr")
        pt  = meta.get("pdu_type")
        acc = meta.get("access_address")
        ts  = meta.get("pcap_ts")

        feats["adv_addr"] = adv if isinstance(adv, str) else ""
        feats["pdu_type"] = pt if (pt is not None) else ""
        feats["access_address"] = acc if isinstance(acc, str) else ""
        feats["pcap_ts"] = f"{ts:.6f}" if isinstance(ts, (int, float)) else ""
        rows.append(feats)

    if not rows:
        print("No feature rows produced")
        return

    # Write CSV (include new metadata columns)
    keys_meta = ["file", "adv_addr", "pdu_type", "access_address", "pcap_ts"]
    other_keys = sorted([k for k in rows[0].keys() if k not in keys_meta])
    keys = keys_meta + other_keys

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows → {args.out_csv}")

    # Aggregate per-feature arrays (for optional plots)
    feat_arrays = {k: [] for k in other_keys}
    for r in rows:
        for k in feat_arrays.keys():
            feat_arrays[k].append(r.get(k, float("nan")))

    # Make plot directory
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir

    # Save plots
    saved = []
    for feat_name, vals in feat_arrays.items():
        if make_cdfs:
            p = save_cdf(feat_name, vals, plots_dir)
            if p: saved.append(p)
        if args.make_hists:
            p = save_hist(feat_name, vals, plots_dir)
            if p: saved.append(p)
    if saved:
        print(f"Saved {len(saved)} plot(s) → {plots_dir}")

    # Tiny text summary (robust stats ignore NaN automatically)
    def robust_stats(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0: return None
        q25, q50, q75 = np.percentile(a, [25, 50, 75])
        iqr = q75 - q25
        return dict(median=float(q50), iqr=float(iqr), min=float(np.min(a)), max=float(np.max(a)))

    print("\nFeature summaries (quality-filtered via NaNs on low-PNR/short bursts):")
    for k, vals in feat_arrays.items():
        s = robust_stats(vals)
        if s:
            print(f"  {k:>22}: median={s['median']:.4g}  IQR={s['iqr']:.4g}  min={s['min']:.4g}  max={s['max']:.4g}")

if __name__ == "__main__":
    main()