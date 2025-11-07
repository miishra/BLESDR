/*
 *  Copyright 2012 by Jiang Wei <jiangwei@jiangwei.org>
 *  Copyright (c) 2014 Omri Iluz (omri@il.uz / http://cyberexplorer.me)
 *
 * This file is part of some open source application.
 *
 * Some open source application is free software: you can redistribute
 * it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * Some open source application is distributed in the hope that it will
 * be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#include "BLESDR.hpp"
#include <iostream>
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>
#include <inttypes.h>

#define RB(l) rb_buf[(rb_head+(l))%RB_SIZE]
#define Q(l) Quantize(l)
#define RB_SIZE 1000

// Prefer including the real header instead of this
// void btle_reverse_whiten(uint8_t chan, uint8_t* data, uint8_t len);

namespace {
using cf = std::complex<float>;

// ---- helpers reused from iq2pcap's feature code ----
static inline std::vector<float> discr(const std::vector<cf>& x){
    std::vector<float> d;
    d.reserve(x.size()>1? x.size()-1 : 0);
    for (size_t i=1;i<x.size();++i){
        cf z = x[i]*std::conj(x[i-1]);
        d.push_back(std::atan2(z.imag(), z.real()));
    }
    return d;
}

static inline float median(std::vector<float> v){
    if (v.empty()) return 0.f;
    size_t mid = v.size()/2;
    std::nth_element(v.begin(), v.begin()+mid, v.end());
    return v[mid];
}

// ---- your enhanced CFO estimator ----
static inline float cfo_quick(const std::vector<cf>& x, double fs){
    if (x.size()<8) return 0.f;
    auto d = discr(x);
    double m = 0; for (float v: d) m += v; if (!d.empty()) m/=d.size();
    double cfo_mean = (fs/(2.0*M_PI))*m;

    float med = median(d);
    if (std::fabs(med) > 2.5f){ // near wrap -> LS fallback
        std::vector<double> t(x.size());
        for (size_t i=0;i<t.size();++i) t[i]= (double)i/fs;
        // unwrap
        std::vector<double> ph(x.size());
        ph[0] = std::arg(x[0]);
        for (size_t i=1;i<x.size();++i){
            double a = std::arg(x[i]);
            double b = std::arg(x[i-1]);
            double dp = a - b;
            if (dp >  M_PI) a -= 2*M_PI;
            if (dp < -M_PI) a += 2*M_PI;
            ph[i] = ph[i-1] + (a - b);
        }
        // simple LS fit
        double Sx=0,Sy=0,Sxx=0,Sxy=0; size_t N=ph.size();
        for (size_t i=0;i<N;++i){ Sx+=t[i]; Sy+=ph[i]; Sxx+=t[i]*t[i]; Sxy+=t[i]*ph[i]; }
        double slope = (N*Sxy - Sx*Sy)/std::max(1e-18, (N*Sxx - Sx*Sx));
        return (float)(slope/(2*M_PI));
    }
    return (float)((fs/(2*M_PI)) * med);
}

// ---- keep the LS reference variant ----
static float cfo_ls(const std::vector<cf>& x, double fs){
    if (x.size() < 4) return 0.f;
    std::vector<double> ph(x.size());
    ph[0] = std::arg(x[0]);
    for (size_t i=1;i<x.size();++i){
        double a = std::arg(x[i]), b = std::arg(x[i-1]), dp = a-b;
        if (dp >  M_PI) a -= 2*M_PI;
        if (dp < -M_PI) a += 2*M_PI;
        ph[i] = ph[i-1] + (a-b);
    }
    std::vector<double> t(x.size());
    for (size_t i=0;i<t.size();++i) t[i]=(double)i/fs;
    double Sx=0,Sy=0,Sxx=0,Sxy=0; size_t N=ph.size();
    for (size_t i=0;i<N;++i){ Sx+=t[i]; Sy+=ph[i]; Sxx+=t[i]*t[i]; Sxy+=t[i]*ph[i]; }
    double slope = (N*Sxy - Sx*Sy)/std::max(1e-18, (N*Sxx - Sx*Sx));
    return (float)(slope/(2.0*M_PI));
}
} // anonymous

void BLESDR::RB_init(void) {
	rb_buf = (int16_t *)malloc(RB_SIZE * 2);
    abs_cursor = 0;
}

void BLESDR::RB_inc(void) {
	rb_head++;
	rb_head = (rb_head) % RB_SIZE;
}

inline bool BLESDR::Quantize(int16_t l) {
	return RB(l*g_srate) > g_threshold;
}

uint8_t BLESDR::SwapBits(uint8_t a) {
	return (uint8_t)(((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16);
}


void BLESDR::ExtractBytes(int l, uint8_t* buffer, int count) {
	int t;
	for (t = 0; t < count; t++) {
		buffer[t] = ExtractByte(l + t * 8);
	}
}

uint8_t BLESDR::ExtractByte(int l) {
	uint8_t byte = 0;
	int c;
	for (c = 0; c < 8; c++) byte |= Q(l + c) << (7 - c);
	return byte;
}

bool BLESDR::DetectPreamble(void) {
	int transitions = 0;
	int c;

	/* preamble sequence is based on the 9th symbol (either 0x55555555 or 0xAAAAAAAA) */
	if (Q(9)) {
		for (c = 0; c < 8; c++) {
			transitions += Q(c) > Q(c + 1);
		}
	}
	else {
		for (c = 0; c < 8; c++) {
			transitions += Q(c) < Q(c + 1);
		}
	}
	return transitions == 4 && abs(g_threshold) < 15500;
}

int32_t BLESDR::ExtractThreshold(void) {
	int32_t threshold = 0;
	int c;
	for (c = 0; c < 8 * g_srate; c++) {
		threshold += (int32_t)RB(c);
	}
	return (int32_t)threshold / (8 * g_srate);
}


void BLESDR::Receiver(size_t channel, float* samples, size_t samples_len) {

	chan = uint8_t(channel);
	//fmdemod
	double phase, dphase;
	for (int i = 0; i < samples_len; i++)
	{
		phase = atan2(samples[i * 2 + 1], samples[i * 2]);
		dphase = phase - last_phase;

		if (dphase < -M_PI) dphase += 2 * M_PI;
		if (dphase > M_PI) dphase -= 2 * M_PI;

		feedOne(uint16_t(dphase / M_PI*UINT16_MAX));

		last_phase = phase;
	}

}

bool BLESDR::feedOne(const uint16_t sample) {

	RB_inc();
	RB(0) = (int)sample;

	if (--skipSamples < 20)
	{
		if (DecodePacket(++samples, srate))
		{
			skipSamples = 20;
			return true;
		}
	}
	return false;
}

bool BLESDR::DecodePacket(int32_t sample, int srate) {
	bool packet_detected = false;
	g_srate = srate;
	g_threshold = ExtractThreshold();

	if (DetectPreamble()) {

		packet_detected |= DecodeBTLEPacket(sample, srate);

	}
	return packet_detected;
}


// bool BLESDR::DecodeBTLEPacket(int32_t sample, int srate) {
// 	int c;
// 	//	struct timeval tv;
// 	uint8_t packet_data[500];
// 	int packet_length;
// 	uint32_t packet_crc;
// 	uint32_t calced_crc;
// 	uint64_t packet_addr_l;
// 	uint32_t packet_addr;
// 	uint8_t crc[3];
// 	uint8_t packet_header_arr[2];

// 	g_srate = srate;

// 	/* extract address */
// 	packet_addr_l = 0;
// 	for (c = 0; c < 4; c++) packet_addr_l |= ((uint64_t)SwapBits(ExtractByte((c + 1) * 8))) << (8 * c);


// 	/* extract pdu header */
// 	ExtractBytes(5 * 8, packet_header_arr, 2);

// 	/* whiten header only so we can extract pdu length */
// 	btle_reverse_whiten(chan,packet_header_arr, 2);

// 	if (packet_addr_l == LE_ADV_AA) {  // Advertisement packet

// 		packet_length = SwapBits(packet_header_arr[1]) & 0x3F;

// 		if (packet_length < 2) {
// 			return false;
// 		}

// 	}
// 	else {

// 		packet_length = 0;			// TODO: data packets unsupported

// 	}

// 	/* extract and whiten pdu+crc */
// 	ExtractBytes(5 * 8, packet_data, packet_length + 2 + 3);
// 	btle_reverse_whiten(chan,packet_data, packet_length + 2 + 3);

// 	if (packet_addr_l == LE_ADV_AA) {  // Advertisement packet
// 		packet_addr = LE_ADV_AA;
// 		crc[0] = crc[1] = crc[2] = 0x55;

// 	}
// 	else {
// 		crc[0] = crc[1] = crc[2] = 0;		// TODO: data packets unsupported
// 	}

// 	/* calculate packet crc */

// 	calced_crc = btle_reverse_crc(packet_data, packet_length + 2, crc);

// 	packet_crc = 0;
// 	for (c = 0; c < 3; c++) packet_crc = (packet_crc << 8) | packet_data[packet_length + 2 + c];

// 	/* BTLE packet found, dump information */
// 	if (packet_crc == calced_crc) {

// 		int i = 0;
// 		lell_packet packet;

// 		packet.access_address = packet_addr;// Advertisement packet
// 		packet.channel_idx = chan;
// 		packet.adv_type = packet_data[0] & 0xf;
// 		packet.adv_tx_add = packet_data[0] & 0x40 ? 1 : 0;
// 		packet.adv_rx_add = packet_data[0] & 0x80 ? 1 : 0;
// 		packet.flags.as_bits.access_address_ok = (packet.access_address == 0x8e89bed6);//TODO
// 		packet.access_address_offenses = 0;//TODO

// 		packet.symbols[0] = packet_addr;
// 		packet.symbols[1] = packet_addr >> 8;
// 		packet.symbols[2] = packet_addr >> 16;
// 		packet.symbols[3] = packet_addr >> 24;

// 		packet.length = packet_length;

// 		for (i = 0; i < packet_length + 2 + 3; i++) {
// 			packet.symbols[i + 4] = (SwapBits(packet_data[i]));
// 		}

// 		callback(packet);
// 		return true;
// 	}
// 	else return false;
// }


void BLESDR::btle_reverse_whiten(uint8_t chan,uint8_t* data, uint8_t len) {

	uint8_t  i;
	uint8_t lfsr = SwapBits(chan) | 2;
	while (len--) {
		for (i = 0x80; i; i >>= 1) {

			if (lfsr & 0x80) {

				lfsr ^= 0x11;
				(*data) ^= i;
			}
			lfsr <<= 1;
		}
		data++;
	}
}

// ==============================
// BLESDRDecoder.cpp
// ==============================
bool BLESDR::DecodeBTLEPacket(int32_t /*sample*/, int sps /*samples per symbol*/)
{
    // ---- Normalize SPS to COMPLEX-samples per symbol ----
    int sps_complex = sps;
    if (sps_complex <= 0) sps_complex = g_srate > 0 ? g_srate : 2;       // default BLE1M: 2 cx/sym
    if ((sps_complex % 2 == 0) && (sps_complex >= 4)) sps_complex /= 2;  // floats/sym -> complex/sym
    g_srate = sps_complex;

    // --- locals ---
    uint8_t  packet_data[500];
    uint8_t  packet_header_arr[2];
    uint8_t  crc_init[3] = {0};
    uint32_t calced_crc  = 0;
    uint32_t packet_crc  = 0;
    uint64_t packet_addr_l = 0;
    int      packet_length = 0;

    // ========== 1) Access Address (AA) ==========
    packet_addr_l = 0;
    for (int c = 0; c < 4; ++c) {
        packet_addr_l |= (static_cast<uint64_t>(SwapBits(ExtractByte((c + 1) * 8))) << (8 * c));
    }

    // ========== 2) Header (still whitened) ==========
    ExtractBytes(5 * 8, packet_header_arr, 2);
    btle_reverse_whiten(chan, packet_header_arr, 2);

    if (packet_addr_l == 0x8E89BED6 /*LE_ADV_AA*/) {
        packet_length = SwapBits(packet_header_arr[1]) & 0x3F; // ADV-only length
        if (packet_length < 2) return false;
    } else {
        packet_length = 0; // DATA not implemented here
    }

    // ========== 3) PDU + CRC (still whitened) ==========
    const int bytes_pdu_crc = packet_length + 2 + 3;
    ExtractBytes(5 * 8, packet_data, bytes_pdu_crc);
    btle_reverse_whiten(chan, packet_data, bytes_pdu_crc);

    if (packet_addr_l == 0x8E89BED6) {
        crc_init[0] = crc_init[1] = crc_init[2] = 0x55;
    } else {
        crc_init[0] = crc_init[1] = crc_init[2] = 0x00;
    }

    // ========== 4) CRC ==========
    calced_crc = btle_reverse_crc(packet_data, packet_length + 2, crc_init);
    packet_crc = 0;
    for (int c = 0; c < 3; ++c) {
        packet_crc = (packet_crc << 8) | packet_data[packet_length + 2 + c];
    }

    // ========== 5) Span in COMPLEX samples (center→edge) ==========
    // Symbols: preamble(8) + AA(32) + (len + 2 + 3) * 8
    const int symbols_total_with_preamble = 8 + 32 + (packet_length + 2 + 3) * 8;
    const uint64_t span_cx  = static_cast<uint64_t>(symbols_total_with_preamble) * static_cast<uint64_t>(g_srate);
    const uint64_t half_cx  = span_cx / 2;
    const uint64_t raw_center = abs_cursor;

    uint64_t sample_end   = raw_center + half_cx;                                  // edge-aligned
    uint64_t sample_start = (sample_end >= span_cx) ? (sample_end - span_cx) : 0;  // edge-aligned

    // ---- OVERRIDE with the decoder’s chosen detect window (SNAP or EXACT) ----
    uint64_t dw_s = 0, dw_e = 0;
    if (take_detect_window(dw_s, dw_e)) {
        if (dw_e > dw_s) {
            const uint64_t dw_span = dw_e - dw_s;
            if (dw_span >= (span_cx * 9) / 10 && dw_span <= (span_cx * 11) / 10) {
                sample_start = dw_s;
                sample_end   = dw_e;
            } else {
                // lengths disagree slightly: trust start, align to expected span
                sample_start = dw_s;
                sample_end   = dw_s + span_cx;
            }
            // fprintf(stderr, "[exact-override] using detect window [%llu,%llu) span=%llu\n",
            //     (unsigned long long)sample_start, (unsigned long long)sample_end,
            //     (unsigned long long)(sample_end - sample_start));
        }
    }

//     // Advance center cursor by one full span (center→center), preserving original semantics
//     abs_cursor = raw_center + span_cx;

    // ========== 6) Emit on CRC OK ==========
    if (packet_crc == calced_crc) {
        lell_packet pkt{};
        const uint32_t aa32 = static_cast<uint32_t>(packet_addr_l);

        pkt.access_address = (aa32 == 0x8E89BED6) ? 0x8E89BED6 : aa32;
        pkt.channel_idx    = chan;
        pkt.length         = packet_length;

        pkt.adv_type   = packet_data[0] & 0x0F;
        pkt.adv_tx_add = (packet_data[0] & 0x40) ? 1 : 0;
        pkt.adv_rx_add = (packet_data[0] & 0x80) ? 1 : 0;

        pkt.flags.as_bits.access_address_ok = (pkt.access_address == 0x8E89BED6);
        pkt.access_address_offenses = 0;

        // First 4 bytes in symbols[] are AA bytes (LSB first after SwapBits)
        pkt.symbols[0] = static_cast<uint8_t>(pkt.access_address >>  0);
        pkt.symbols[1] = static_cast<uint8_t>(pkt.access_address >>  8);
        pkt.symbols[2] = static_cast<uint8_t>(pkt.access_address >> 16);
        pkt.symbols[3] = static_cast<uint8_t>(pkt.access_address >> 24);
        for (int i = 0; i < packet_length + 2 + 3; ++i) {
            pkt.symbols[i + 4] = SwapBits(packet_data[i]);
        }

        // Export COMPLEX-sample stamps (the window actually used by demod)
        pkt.sample_start = sample_start;
        pkt.sample_end   = sample_end;

        // In this code base, srate_hz field carries COMPLEX SPS (not Hz)
        pkt.srate_hz = g_srate;

        // ---------- CFO on the exact used window (robust LS over mid-60%) ----------
        // ---------- CFO on the exact used window (conjugate-product mean over mid-60%) ----------
        // ---------- CFO on the exact used window (whole packet, robust) ----------
        // ---------- CFO (whole packet, robust sym-lag) ----------
        // ---------- CFO (whole packet, robust sym-lag) ----------
        if (iq_provider_) {
            std::vector<std::complex<float>> x;
        
            // Guard invalid ranges
            if (sample_end > sample_start &&
                iq_provider_(sample_start, sample_end, x) &&
                x.size() >= 64)
            {
                const int    cx_per_sym = std::max(1, g_srate);       // BLE1M: 2
                const double fs         = 1.0e6 * double(cx_per_sym); // complex sample rate (Hz)
        
                auto window_rms = [](const std::vector<std::complex<float>>& v)->double {
                    if (v.empty()) return 0.0;
                    long double a = 0.0L;
                    for (auto& z : v) a += (long double)std::norm(z);
                    return std::sqrt((double)(a / (long double)v.size()));
                };
        
                auto cfo_sym_lag_weighted = [&](const std::vector<std::complex<float>>& s,
                                                int K, double fs_hz)->double
                {
                    if ((int)s.size() <= K) return 0.0;
                    const size_t N   = s.size();
                    const size_t pad = std::min((size_t)16, (size_t)std::max<size_t>(1, (size_t)(0.02 * N)));
                    size_t a = std::max((size_t)K + pad, (size_t)K + 1);
                    size_t b = (N > pad ? N - pad : N);
                    if (b <= a + 8) { a = K + 1; b = N; }
                    if (b <= a + 8) return 0.0;
        
                    long double Sx = 0.0L, Sy = 0.0L;
                    size_t used = 0;
                    for (size_t n = a; n < b; ++n) {
                        const std::complex<float> d = s[n] * std::conj(s[n - K]);
                        const float w = std::min(std::norm(s[n]), std::norm(s[n - K]));
                        if (w <= 0.0f) continue;
                        const float re = d.real(), im = d.imag();
                        const float mag = std::hypot(re, im);
                        if (mag <= 0.0f) continue;
                        Sx += (long double)w * (long double)(re / mag);
                        Sy += (long double)w * (long double)(im / mag);
                        ++used;
                    }
                    if (used < (size_t)(8 * cx_per_sym)) return 0.0;
                    const long double ang = std::atan2((double)Sy, (double)Sx);
                    const double dphi_per_K = (double)ang;
                    return (dphi_per_K / (2.0 * M_PI)) * (fs_hz / (double)K);
                };
        
                // --- try the chosen window ---
                double r = window_rms(x);
        
                // --- autosnap fallback if RMS is too low (detect window missing/consumed) ---
                if (r < 0.25) {
                    // search ±10% of span, step = 1 symbol
                    const uint64_t span = sample_end - sample_start;
                    const uint64_t max_shift = span / 10;
                    const uint64_t step = (uint64_t)std::max<uint64_t>(1, (uint64_t)cx_per_sym);
        
                    double best_r = r;
                    uint64_t best_s = sample_start;
                    std::vector<std::complex<float>> best_x = x;
        
                    // center around the current start; try forward then backward
                    for (uint64_t sh = step; sh <= max_shift; sh += step) {
                        // forward
                        if (sample_start + sh + span <= UINT64_MAX) {
                            std::vector<std::complex<float>> xf;
                            if (iq_provider_(sample_start + sh, sample_start + sh + span, xf) && xf.size() == x.size()) {
                                double rf = window_rms(xf);
                                if (rf > best_r) { best_r = rf; best_s = sample_start + sh; best_x.swap(xf); }
                            }
                        }
                        // backward (guard underflow)
                        if (sample_start >= sh) {
                            std::vector<std::complex<float>> xb;
                            if (iq_provider_(sample_start - sh, sample_start - sh + span, xb) && xb.size() == x.size()) {
                                double rb = window_rms(xb);
                                if (rb > best_r) { best_r = rb; best_s = sample_start - sh; best_x.swap(xb); }
                            }
                        }
                    }
        
                    if (best_r > r) {
                        x.swap(best_x);
                        r = best_r;
                        // optional: update exported stamps to reflect autosnap
                        pkt.sample_start = best_s;
                        pkt.sample_end   = best_s + span;
                    }
                }
        
                if (r >= 0.25) {
                    const int K1 = cx_per_sym;
                    const int K2 = 2 * cx_per_sym;
        
                    const double cfo1 = cfo_sym_lag_weighted(x, K1, fs);
                    const double cfo2 = (x.size() > (size_t)(K2 + 32))
                                      ? cfo_sym_lag_weighted(x, K2, fs)
                                      : cfo1;
        
                    // block-median guard (6 blocks) on K1
                    auto cfo_block_median = [&](int K)->double {
                        const size_t N = x.size();
                        const int blocks = 6;
                        std::array<double, 6> est{};
                        int filled = 0;
                        for (int b = 0; b < blocks; ++b) {
                            size_t s0 = (size_t)((N * b) / blocks);
                            size_t s1 = (size_t)((N * (b + 1)) / blocks);
                            if (s1 <= s0 + (size_t)(K + 8)) continue;
                            std::vector<std::complex<float>> sub(x.begin() + s0, x.begin() + s1);
                            const double e = cfo_sym_lag_weighted(sub, K, fs);
                            if (e != 0.0) est[filled++] = e;
                        }
                        if (filled == 0) return 0.0;
                        std::nth_element(est.begin(), est.begin() + filled/2, est.begin() + filled);
                        return est[filled/2];
                    };
        
                    const double cfo_med = cfo_block_median(K1);
                    double cfo = cfo1;
                    if (cfo_med != 0.0) {
                        const double diff = std::fabs(cfo1 - cfo_med);
                        const double tol  = std::max(500.0, 0.15 * std::fabs(cfo_med)); // 500 Hz or 15%
                        if (diff > tol) cfo = (std::fabs(cfo2 - cfo_med) < diff) ? cfo2 : cfo_med;
                    }
        
                    pkt.cfo_exact_quick_hz = (float)cfo;
                    pkt.cfo_exact_ls_hz    = (float)cfo;
                    std::fprintf(stderr, "[cfo] fs=%.0fHz N=%zu rms=%.3f -> CFO=%.2f Hz (autosnap=%s)\n",
                                 fs, x.size(), r, cfo, (r < 0.30 ? "yes" : "no"));
                } else {
                    pkt.cfo_exact_quick_hz = 0.0f;
                    pkt.cfo_exact_ls_hz    = 0.0f;
                    // fprintf(stderr, "[cfo-skip] low-RMS window after autosnap\n");
                }
            }
        }

        // deliver
        callback(pkt);
        
        // Advance center cursor by one full span (center→center), preserving original semantics
        abs_cursor = raw_center + span_cx;
        return true;
    }
    
    // Advance center cursor by one full span (center→center), preserving original semantics
    abs_cursor = raw_center + span_cx;

    return false;
}

uint32_t BLESDR::btle_reverse_crc(const uint8_t* data, uint8_t len, uint8_t* dst) {

	uint8_t v, t, d;
	uint32_t crc = 0;
	while (len--) {

		d = SwapBits(*data++);
		for (v = 0; v < 8; v++, d >>= 1) {

			t = dst[0] >> 7;

			dst[0] <<= 1;
			if (dst[1] & 0x80) dst[0] |= 1;
			dst[1] <<= 1;
			if (dst[2] & 0x80) dst[1] |= 1;
			dst[2] <<= 1;


			if (t != (d & 1)) {

				dst[2] ^= 0x5B;
				dst[1] ^= 0x06;
			}
		}
	}
	for (v = 0; v < 3; v++) crc = (crc << 8) | dst[v];
	return crc;
}