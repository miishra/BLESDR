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

bool BLESDR::DecodeBTLEPacket(int32_t sample, int srate) {
    // ---- DEBUG counters ----
    static unsigned long long g_total_samples_processed = 0ULL;  // assumes per-sample calls
    static unsigned long long g_packet_seq = 0ULL;

    static uint64_t g_abs_samples = 0;  // persists across calls

    g_total_samples_processed++;             // count this call as one processed sample
    unsigned long long entry_sample_idx = g_total_samples_processed;

    int c;
    //  struct timeval tv;
    uint8_t packet_data[500];
    int packet_length;
    uint32_t packet_crc;
    uint32_t calced_crc;
    uint64_t packet_addr_l;
    uint32_t packet_addr;
    uint8_t crc[3];
    uint8_t packet_header_arr[2];

    g_srate = srate;

    // DEBUG: function entry
    // fprintf(stderr,
    //         "[DEBUG][Decode] >>> Enter | seq=%llu | total_samples=%llu | srate=%d | chan=%d\n",
    //         g_packet_seq + 1ULL, g_total_samples_processed, srate, chan);

    /* extract address */
    packet_addr_l = 0;
    for (c = 0; c < 4; c++)
        packet_addr_l |= ((uint64_t)SwapBits(ExtractByte((c + 1) * 8))) << (8 * c);

    // fprintf(stderr,
    //         "[DEBUG][Addr ] AccessAddress(raw, swapped bytes) = 0x%08" PRIx64 "\n",
    //         (uint64_t)packet_addr_l);

    /* extract pdu header */
    ExtractBytes(5 * 8, packet_header_arr, 2);

    // Save raw header before whitening (for debug)
    uint8_t header_raw[2] = { packet_header_arr[0], packet_header_arr[1] };

    /* whiten header only so we can extract pdu length */
    btle_reverse_whiten(chan, packet_header_arr, 2);

    // fprintf(stderr,
    //         "[DEBUG][Head ] Header raw:    [%02x %02x]\n"
    //         "[DEBUG][Head ] Header white⁻¹:[%02x %02x]\n",
    //         header_raw[0], header_raw[1], packet_header_arr[0], packet_header_arr[1]);

    if (packet_addr_l == LE_ADV_AA) {  // Advertisement packet
        packet_length = SwapBits(packet_header_arr[1]) & 0x3F;

        // fprintf(stderr,
        //         "[DEBUG][Len  ] ADV header length(field)= %d (0x%02x & 0x3F), type=0x%01x\n",
        //         packet_length, SwapBits(packet_header_arr[1]),
        //         (packet_header_arr[0] & 0x0F));

        if (packet_length < 2) {
            // fprintf(stderr,
            //         "[DEBUG][Exit ] Packet too short (len=%d). Returning false. "
            //         "samples_total=%llu\n",
            //         packet_length, g_total_samples_processed);
            return false;
        }
    } else {
        packet_length = 0;  // TODO: data packets unsupported
        // fprintf(stderr,
        //         "[DEBUG][Info ] Non-ADV Access Address detected (0x%08" PRIx64
        //         "). Data packets unsupported; setting length=%d\n",
        //         (uint64_t)packet_addr_l, packet_length);
    }

    /* extract and whiten pdu+crc */
    const int bytes_pdu_crc = packet_length + 2 + 3; // PDU header(2) + CRC(3)
    ExtractBytes(5 * 8, packet_data, bytes_pdu_crc);

    // Keep a copy pre-whitening for debug
    uint8_t packet_data_raw_preview[16] = {0};
    const int preview_n = (bytes_pdu_crc < 16) ? bytes_pdu_crc : 16;
    for (int i = 0; i < preview_n; i++) packet_data_raw_preview[i] = packet_data[i];

    btle_reverse_whiten(chan, packet_data, bytes_pdu_crc);

    // fprintf(stderr, "[DEBUG][Whiten] PDU+CRC bytes = %d\n", bytes_pdu_crc);
    // fprintf(stderr, "[DEBUG][Data ] Raw(pre  ) first %dB:", preview_n);
    for (int i = 0; i < preview_n; i++) fprintf(stderr, " %02x", packet_data_raw_preview[i]);
    // fprintf(stderr, "\n[DEBUG][Data ] White⁻¹(post) first %dB:", preview_n);
    for (int i = 0; i < preview_n; i++) fprintf(stderr, " %02x", packet_data[i]);
    // fprintf(stderr, "\n");

    if (packet_addr_l == LE_ADV_AA) {  // Advertisement packet
        packet_addr = LE_ADV_AA;
        crc[0] = crc[1] = crc[2] = 0x55;
        // fprintf(stderr, "[DEBUG][CRC  ] Using ADV init CRC: 55 55 55\n");
    } else {
        crc[0] = crc[1] = crc[2] = 0;  // TODO: data packets unsupported
        // fprintf(stderr, "[DEBUG][CRC  ] Using DATA init CRC: 00 00 00 (unsupported path)\n");
    }

    /* calculate packet crc */
    calced_crc = btle_reverse_crc(packet_data, packet_length + 2, crc);

    packet_crc = 0;
    for (c = 0; c < 3; c++)
        packet_crc = (packet_crc << 8) | packet_data[packet_length + 2 + c];

    // fprintf(stderr,
    //         "[DEBUG][CRC  ] Calculated=0x%06x | Received=0x%06x | pdu_len=%d | chan=%d\n",
    //         calced_crc, packet_crc, packet_length, chan);

    // total symbols: preamble(8) + [AA(32) + (len+2+3)*8]
    const int symbols_total_with_preamble =
        8 + 32 + (packet_length + 2 + 3) * 8;

    // sps is “samples per symbol” (~2 at 2 MS/s for 1M BLE)
    const double sps = (double)g_srate;  // you already fixed this
    const uint64_t sample_span = (uint64_t) llround(symbols_total_with_preamble * sps);

    // compute BEFORE CRC check
    const uint64_t sample_start = abs_cursor;
    const uint64_t sample_end   = sample_start + sample_span;

    // ALWAYS advance the cursor (valid or not)
    abs_cursor = sample_end;

    /* BTLE packet found, dump information */
    if (packet_crc == calced_crc) {
        int i = 0;
        lell_packet packet;

        packet.access_address = packet_addr;  // Advertisement packet
        packet.channel_idx = chan;
        packet.adv_type = packet_data[0] & 0xf;
        packet.adv_tx_add = packet_data[0] & 0x40 ? 1 : 0;
        packet.adv_rx_add = packet_data[0] & 0x80 ? 1 : 0;
        packet.flags.as_bits.access_address_ok = (packet.access_address == 0x8e89bed6);  // TODO
        packet.access_address_offenses = 0;  // TODO

        packet.symbols[0] = packet_addr;
        packet.symbols[1] = packet_addr >> 8;
        packet.symbols[2] = packet_addr >> 16;
        packet.symbols[3] = packet_addr >> 24;

        packet.length = packet_length;

		// fprintf(stderr,
		// "[DEBUG][CFO ] symbols(no_preamble)=%d symbols(with_preamble)=%d | "
		// "sps=%.3f | samples(no_preamble)=%d samples(with_preamble)=%d\n",
		// symbols_total_no_preamble, symbols_total_with_preamble,
		// sps, samples_no_preamble, samples_with_preamble);

        for (i = 0; i < packet_length + 2 + 3; i++) {
            packet.symbols[i + 4] = (SwapBits(packet_data[i]));
        }

        // Per-packet summary before callback
        const int bytes_total_packet =
            4 /*AA*/ + /*hdr*/ + (packet_length + 2 + 3) /*pdu+crc*/;

        // --- Compute absolute IQ sample indices for this packet ---
        // We treat g_total_samples_processed as a 1-based counter of processed samples.
        // Use "entry_sample_idx" (captured on function entry) as the sample index
        // when decode is triggered (end of the packet window). Build a half-open range.
        // uint64_t sample_end   = entry_sample_idx; // end is exclusive
        // uint64_t sample_span  = (samples_with_preamble >= 0) ? (uint64_t)samples_with_preamble : 0ULL;
        // uint64_t sample_begin = (sample_end > sample_span) ? (sample_end - sample_span) : 0ULL;

        packet.sample_start = sample_start;
        packet.sample_end   = sample_end;        // [start, end)
        packet.srate_hz     = g_srate;

        // if (!(sps > 0.1 && sps < 10.0)) {
        //     fprintf(stderr,
        //             "[ERR] sps=%.6f looks wrong (expected ~2.0). "
        //             "Did you pass Fs(Hz) or SPS?\n"
        //             "    g_srate=%d (as received)\n",
        //             sps, g_srate);
        // }

        // fprintf(stderr,
        //         "[DEBUG][OK   ] CRC match. seq=%llu | len=%d | adv_type=0x%x | "
        //         "tx_add=%d rx_add=%d | AA=0x%08x | bytes_total=%d | "
        //         "entry_sample=%llu current_total_samples=%llu | "
        //         "sample_range=[%llu,%llu) span=%llu\n",
        //         g_packet_seq + 1ULL, packet_length, packet.adv_type,
        //         packet.adv_tx_add, packet.adv_rx_add, packet_addr,
        //         bytes_total_packet, entry_sample_idx, g_total_samples_processed,
        //         (unsigned long long)packet.sample_start,
        //         (unsigned long long)packet.sample_end,
        //         (unsigned long long)(packet.sample_end - packet.sample_start));

        // Optional: small hexdump of PDU header+payload (excluding CRC) for context
        const int dump_bytes = (packet_length + 2 < 24) ? (packet_length + 2) : 24;
        // fprintf(stderr, "[DEBUG][PDU  ] first %dB:", dump_bytes);
        for (int k = 0; k < dump_bytes; ++k) fprintf(stderr, " %02x", packet_data[k]);
        // fprintf(stderr, "\n");

        callback(packet);

        g_packet_seq++;
        // fprintf(stderr, "[DEBUG][Decode] <<< Exit(SUCCESS) seq=%llu | total_samples=%llu\n",
        //         g_packet_seq, g_total_samples_processed);
        return true;
    } else {
        // fprintf(stderr,
        //         "[DEBUG][FAIL ] CRC mismatch. seq=%llu | calc=0x%06x recv=0x%06x | "
        //         "len=%d | entry_sample=%llu current_total_samples=%llu\n",
        //         g_packet_seq + 1ULL, calced_crc, packet_crc, packet_length,
        //         entry_sample_idx, g_total_samples_processed);

        // fprintf(stderr, "[DEBUG][Decode] <<< Exit(FAIL) seq=%llu | total_samples=%llu\n",
        //         g_packet_seq + 1ULL, g_total_samples_processed);
        return false;
    }
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