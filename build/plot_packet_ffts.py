#!/usr/bin/env python3
"""
plot_packet_ffts.py

Make per-packet FFT plots from pkt_*.fc32 (float32 interleaved I,Q) and mark CFO estimates.

For each packet:
  - Gate the burst
  - Compute PSD using Hann window (dB, normalized to 0 dB peak)
  - Estimate CFO via several methods:
      * cfo_centroid_hz   (spectral centroid in a limited window)
      * cfo_hz            (instantaneous phase median)
      * cfo_preaa_hz      (preamble+AA window)
      * cfo_two_stage_hz  (coarse centroid -> align preamble -> LS on phase)
  - Plot the FFT with vertical markers at 0 Hz and each CFO estimate
  - Optionally include adv_addr/pdu_type/access_address from --index-csv or --pcap
  - Save PNGs to --fft-dir (default: fft_plots)

Example:
  python3 plot_packet_ffts.py \
    --dir iq_dump_37_2 --fs 2e6 --pcap out_ch37.pcap \
    --fft-dir fft_plots --nfft 8192 --span-hz 300e3 --max-files 2000
"""

import argparse, csv, os, re, struct, sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------- small helpers ----------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True); return p
def sanitize(s): return str(s).replace("/", "_").replace(" ", "_").replace(":", "")
def rm_dc(x): return x - np.mean(x)
def norm_pow(x): return x / (np.sqrt(np.mean(np.abs(x)**2) + 1e-12) or 1.0)
def _flt(v):
    try: return float(v)
    except Exception: return float("nan")

def welch_psd_complex(x, fs, nfft=4096):
    L = min(len(x), nfft)
    if L < 32:
        return np.array([0.0]), np.array([0.0])
    w = np.hanning(L)
    X = np.fft.fft(w * x[:L], n=L)
    S = np.abs(X)**2 / (np.sum(w**2) + 1e-18)
    S = np.fft.fftshift(S)
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1/fs))
    return f, S

def gate_burst(x, fs, pre_us=60, post_us=60, smooth_us=8, thr_db_above=5):
    env = np.abs(x)
    k = max(1, int(round(smooth_us*1e-6*fs)))
    if k > 1:
        env = np.convolve(env, np.ones(k)/k, mode="same")
    nf = np.median(env) + 1e-12
    thr = min(nf * (10**(thr_db_above/20)), 0.5*np.max(env))
    on = np.where(env >= thr)[0]
    if on.size == 0:  # fallback
        return x
    i0, i1 = on[0], on[-1]
    a = max(0, i0 - int(round(pre_us*1e-6*fs)))
    b = min(len(x), i1 + int(round(post_us*1e-6*fs)) + 1)
    return x[a:b]

def _symbol_rate(fs_hz):  # BLE 1M → ~2 sps at 2 Msps
    return max(2, int(round(fs_hz/1e6)))

def estimate_cfo_centroid(x, fs, nfft=16384, f_lim_hz=200e3):
    L = min(len(x), nfft)
    if L < 32: return 0.0
    w = np.hanning(L)
    X = np.fft.fft(w * x[:L], n=L)
    S = np.abs(X)**2
    S = np.fft.fftshift(S)
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1/fs))
    m = np.abs(f) <= float(f_lim_hz)
    if not np.any(m): return 0.0
    f_win, S_win = f[m], S[m] + 1e-18
    return float(np.sum(f_win * S_win) / np.sum(S_win))

def estimate_cfo(x, fs):
    if len(x) < 2: return 0.0
    ph = np.angle(x[1:] * np.conj(x[:-1]))
    return float((fs/(2*np.pi)) * np.median(ph))

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
    return np.array([1 if b else -1 for b in (pre + aa_bits)], float)

def correlate_preamble_aa_symbol_aligned(x, fs):
    sps = _symbol_rate(fs)
    min_syms = 48
    if len(x) < sps * min_syms: return 0, sps, 0.0
    d = np.angle(x[1:] * np.conj(x[:-1]))
    Ls = (len(d)//sps)
    if Ls < min_syms: return 0, sps, 0.0
    d_sym = d[:Ls*sps].reshape(Ls, sps).mean(axis=1)
    ref = ble_adv_preamble_aa_bits()
    m = len(ref)
    if Ls < m: return 0, sps, 0.0
    ds = np.sign(d_sym)
    best_i, best_val = 0, -1e9
    refn = np.linalg.norm(ref) + 1e-12
    for i in range(Ls - m + 1):
        seg = ds[i:i+m]
        val = float(np.dot(seg, ref) / ((np.linalg.norm(seg) + 1e-12) * refn))
        if val > best_val:
            best_val, best_i = val, i
    return best_i, sps, float(best_val)

def _cfo_ls(seg, fs):
    if len(seg) < 4: return 0.0
    ph = np.unwrap(np.angle(seg))
    t  = np.arange(len(seg)) / fs
    slope = np.polyfit(t, ph, 1)[0]
    return float(slope / (2*np.pi))

def estimate_cfo_preamble_aa(x, fs, bits=80, settle_us=6):
    sps = _symbol_rate(fs)
    i0 = _find_burst_start(x, fs)
    a = i0 + int(round(settle_us*1e-6*fs))
    b = min(a + int(bits*sps), len(x))
    seg = x[a:b] if (b-a) >= 8 else x
    if len(seg) < 2: return 0.0, (a,b)
    ph = np.angle(seg[1:] * np.conj(seg[:-1]))
    return float((fs/(2*np.pi))*np.median(ph)), (a,b)

def estimate_cfo_two_stage(x, fs,
                           bits=160, settle_us=8,
                           fwin1=200e3, fwin2=120e3,
                           min_symbols=120, min_corr=0.45,
                           do_nudge=True):
    # coarse centroid then refine once
    f0 = estimate_cfo_centroid(x, fs, f_lim_hz=fwin1)
    n = np.arange(len(x))
    x0 = x * np.exp(-1j * 2*np.pi * f0 * n / fs)
    f1 = estimate_cfo_centroid(x0, fs, f_lim_hz=fwin2)
    x1 = x0 * np.exp(-1j * 2*np.pi * f1 * n / fs)
    f_coarse = f0 + f1

    i_burst = _find_burst_start(x1, fs)
    sym0, sps, score = correlate_preamble_aa_symbol_aligned(x1[i_burst:], fs)
    if score < min_corr:
        return None, (0,0), dict(reason="low_corr", score=score, coarse=f_coarse)

    sps = int(sps)
    num_sym = max(bits, min_symbols)
    a = i_burst + sym0*sps + int(round(settle_us*1e-6*fs))
    b = min(a + num_sym*sps, len(x1))
    if b - a < (min_symbols*sps):
        return None, (a,b), dict(reason="short_window", score=score, coarse=f_coarse)

    if do_nudge:
        # simple “energy-of-phase-slope” nudge
        best_a, best_b = a, b
        best_val = -1.0
        for dlt in range(-sps, sps+1):
            aa = max(0, a+dlt); bb = min(len(x1), b+dlt)
            if bb-aa < 20*sps: continue
            seg = x1[aa:bb]
            dp = seg[1:] * np.conj(seg[:-1])
            val = np.abs(np.sum(dp)) / (len(dp)+1e-12)
            if val > best_val: best_val, best_a, best_b = val, aa, bb
        a, b = best_a, best_b

    cfo_fine = _cfo_ls(x1[a:b], fs)
    return float(f_coarse + cfo_fine), (a,b), dict(score=score, coarse=f_coarse)

# ---------- metadata from index/pcap (optional) ----------
def _read_index_csv(path):
    idx = {}
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            fn = Path(row.get("file") or row.get("filename") or "").name
            if not fn: continue
            # pdu_type
            pt = row.get("pdu_type"); 
            try: pt = int(str(pt), 0) if str(pt).strip() else None
            except Exception: pt = None
            # adv_addr
            aa = row.get("adv_addr") or row.get("advertising_address")
            adv = str(aa).strip().lower() if aa else None
            # access address
            acc = row.get("access_address") or row.get("aa") or row.get("accessaddr")
            if isinstance(acc, str):
                acc = acc.strip().replace("0x","").replace(":","")
                acc = acc.upper() if re.fullmatch(r"[0-9A-Fa-f]{8}", acc or "") else None
            else:
                acc = None
            idx[fn] = {"pdu_type": pt, "adv_addr": adv, "access_address": acc}
    return idx

def _extract_access_address(buf, incl, linktype):
    if linktype == 256:  # [10B hdr][4B AA][LL...]
        if incl >= 14:
            aa = "".join(f"{b:02X}" for b in buf[10:14][::-1])
            return aa, 14
        return None, 14
    elif linktype == 251:  # [4B AA][LL...]
        if incl >= 4:
            aa = "".join(f"{b:02X}" for b in buf[0:4][::-1])
            return aa, 4
        return None, 4
    else:
        return None, 0

def _parse_pcap_records(pcap_path):
    recs = []
    with open(pcap_path, "rb") as f:
        gh = f.read(24)
        if len(gh) != 24: raise RuntimeError("PCAP: short global header")
        magic_le = struct.unpack("<I", gh[:4])[0]
        magic_be = struct.unpack(">I", gh[:4])[0]
        if magic_le == 0xA1B2C3D4:
            endian = "<"; _,_,_,_,_,_, linktype = struct.unpack("<IHHIIII", gh)
        elif magic_be == 0xA1B2C3D4:
            endian = ">"; _,_,_,_,_,_, linktype = struct.unpack(">IHHIIII", gh)
        else:
            raise RuntimeError("PCAP: unsupported magic")
        rec_hdr = endian + "IIII"
        while True:
            rh = f.read(16)
            if len(rh) != 16: break
            ts_sec, ts_usec, incl, _ = struct.unpack(rec_hdr, rh)
            buf = f.read(incl)
            if len(buf) != incl: break
            aa_hex, LL_OFF = _extract_access_address(buf, incl, linktype)
            pdu = None; adv = None
            if incl >= LL_OFF + 2 + 3:
                h0 = buf[LL_OFF+0] if (LL_OFF+0) < incl else 0
                h1 = buf[LL_OFF+1] if (LL_OFF+1) < incl else 0
                pdu = h0 & 0x0F; ln = h1 & 0x3F
                if incl >= LL_OFF + 2 + ln + 3:
                    pl = buf[LL_OFF+2 : LL_OFF+2+ln]
                    if pdu in (0,2,6,4):
                        if len(pl) >= 6: adv = ":".join(f"{b:02X}" for b in pl[0:6][::-1]).lower()
                    elif pdu in (3,5):
                        if len(pl) >= 12: adv = ":".join(f"{b:02X}" for b in pl[6:12][::-1]).lower()
            recs.append({"pdu_type": pdu, "adv_addr": adv, "access_address": aa_hex})
    return recs

def build_mapping(files, index_csv=None, pcap_path=None):
    if index_csv:
        idx = _read_index_csv(index_csv)
        return {Path(f).name: idx.get(Path(f).name, {}) for f in files}
    if pcap_path:
        recs = _parse_pcap_records(pcap_path)
        # align by numeric index if present, otherwise by order
        nums = []
        for f in files:
            m = re.search(r"(\d+)", Path(f).name)
            nums.append(int(m.group(1)) if m else None)
        if all(n is not None for n in nums):
            mp = {}
            for n, f in zip(nums, files):
                if 0 <= n < len(recs): mp[Path(f).name] = recs[n]
            return mp
        n = min(len(files), len(recs))
        return {Path(files[i]).name: recs[i] for i in range(n)}
    return {Path(f).name: {} for f in files}

# ---------- plotting ----------
def plot_fft_with_markers(x, fs, nfft, span_hz, out_png, title="", marks=None):
    """
    x: complex burst
    marks: list of (freq_hz, label) to draw vertical lines
    """
    f, S = welch_psd_complex(x, fs, nfft=nfft)
    SdB = 10*np.log10(np.maximum(S, 1e-18))
    SdB -= np.max(SdB)  # normalize to 0 dB peak

    xmin = -span_hz if span_hz else f[0]
    xmax =  span_hz if span_hz else f[-1]
    m = (f >= xmin) & (f <= xmax)

    fig = plt.figure(figsize=(7.2, 4.2))
    plt.plot(f[m], SdB[m], linewidth=1.1, label="PSD (Hann, normalized)")
    plt.axvline(0.0, linestyle="--", linewidth=1.0, label="DC (0 Hz)")
    if marks:
        for hz, lbl in marks:
            if hz is None or not np.isfinite(hz): continue
            plt.axvline(float(hz), linewidth=1.0)
            plt.text(float(hz), np.max(SdB[m]) - 3.0, lbl, rotation=90,
                     va="top", ha="center", fontsize=8)

    plt.xlim(xmin, xmax)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB, peak=0)")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with pkt_*.fc32 (float32 interleaved I,Q)")
    ap.add_argument("--fs", type=float, required=True, help="Complex sample rate after decim (e.g., 2e6)")
    ap.add_argument("--fft-dir", default="fft_plots", help="Where to save per-packet FFT PNGs")
    ap.add_argument("--nfft", type=int, default=8192, help="FFT size (uses min(len, nfft))")
    ap.add_argument("--span-hz", type=float, default=300e3, help="±span to show around DC (Hz); 0=auto")
    ap.add_argument("--max-files", type=int, default=0, help="Process at most N packets (0 = all)")
    # optional metadata for labels
    ap.add_argument("--index-csv", help="CSV with file,pdu_type,adv_addr,access_address")
    ap.add_argument("--pcap", help="PCAP (DLT 256/251) to auto-extract adv_addr/pdu_type/access_address")
    # quality gate used for CFO calculations (plots still saved even if gated out)
    ap.add_argument("--pnr-min", type=float, default=10.0)
    ap.add_argument("--min-us", type=float, default=120.0)
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("pkt_*.fc32"))
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        print("No pkt_*.fc32 found.", file=sys.stderr); sys.exit(1)

    meta = build_mapping(files, index_csv=args.index_csv, pcap_path=args.pcap)
    outdir = ensure_dir(Path(args.fft_dir))

    for i, fn in enumerate(files, 1):
        iq = np.fromfile(fn, dtype=np.float32)
        if iq.size < 4: continue
        x = iq.view(np.complex64)

        # gate + normalize for cleaner FFT and CFO
        xg = gate_burst(x, args.fs)
        xg = norm_pow(rm_dc(xg))

        # spectrum stats to gate CFO (still plot regardless)
        fcent, S = welch_psd_complex(xg, args.fs, nfft=min(len(xg), args.nfft))
        SdB = 10*np.log10(np.maximum(S,1e-18))
        pnr_db = float(np.max(SdB) - np.median(SdB))
        gated_len_us = len(xg) * 1e6 / args.fs
        ok = (pnr_db >= args.pnr_min) and (gated_len_us >= args.min_us)

        # CFO estimates (NaN if quality-gated)
        if ok:
            cfo_cent = estimate_cfo_centroid(xg, args.fs, nfft=args.nfft, f_lim_hz=min(args.span_hz or 200e3, 250e3))
            cfo_disc = estimate_cfo(xg, args.fs)
            cfo_pre, _ = estimate_cfo_preamble_aa(xg, args.fs, bits=80, settle_us=6)
            cfo_twostage, _, info = estimate_cfo_two_stage(
                xg, args.fs, bits=160, settle_us=8, fwin1=200e3, fwin2=120e3,
                min_symbols=120, min_corr=0.45, do_nudge=True
            )
            if cfo_twostage is None: cfo_twostage = np.nan
        else:
            cfo_cent = cfo_disc = cfo_pre = cfo_twostage = np.nan

        m = meta.get(fn.name, {})
        mac = m.get("adv_addr") or "UNK"
        pdu = m.get("pdu_type")
        aa  = m.get("access_address")

        # markers with short labels (kHz)
        marks = [
            (0.0, "0 Hz"),
            (cfo_cent,      f"cent {cfo_cent/1e3:.1f} kHz"),
            (cfo_disc,      f"disc {cfo_disc/1e3:.1f} kHz"),
            (cfo_pre,       f"preAA {cfo_pre/1e3:.1f} kHz"),
            (cfo_twostage,  f"2stage {cfo_twostage/1e3:.1f} kHz"),
        ]

        title = f"{fn.name} | MAC {mac} | PDU {pdu if pdu is not None else 'NA'} | AA {aa or 'NA'}"
        out_png = outdir / f"fft_{sanitize(mac)}__pdu_{pdu if pdu is not None else 'NA'}__{fn.stem}.png"
        plot_fft_with_markers(xg, args.fs, args.nfft, args.span_hz, str(out_png), title=title, marks=marks)

        if (i % 200) == 0:
            print(f"[{i}/{len(files)}] saved {out_png.name}")

    print(f"Done. Saved FFT plots → {outdir}")

if __name__ == "__main__":
    main()