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

// ==============================
// BLESDRDecoder.cpp
// ==============================
bool BLESDR::DecodeBTLEPacket(int32_t /*sample*/, int srate) {
    // Assumes BLESDR has members:
    //   uint64_t abs_cursor;   // advanced externally with set_abs_cursor()
    //   int      chan;         // current RF channel
    //   int      g_srate;      // samples-per-symbol (~2 at Fs=2e6 for BLE1M)
    //   std::function<void(lell_packet)> callback;

    // --- Locals ---
    uint8_t  packet_data[500];
    uint8_t  packet_header_arr[2];
    uint8_t  crc[3] = {0};
    uint32_t calced_crc = 0;
    uint32_t packet_crc = 0;
    uint64_t packet_addr_l = 0;
    uint32_t packet_addr   = 0;
    int      packet_length = 0;

    g_srate = srate;

    // --- Extract Access Address (AA) ---
    packet_addr_l = 0;
    for (int c = 0; c < 4; ++c) {
        packet_addr_l |= ((uint64_t)SwapBits(ExtractByte((c + 1) * 8))) << (8 * c);
    }

    // --- Pull the 2B PDU header (still whitened) ---
    ExtractBytes(5 * 8, packet_header_arr, 2);

    // De-whiten header to read length/type
    btle_reverse_whiten(chan, packet_header_arr, 2);

    // Determine payload length for ADV (DATA not supported here)
    if (packet_addr_l == LE_ADV_AA) {
        packet_length = SwapBits(packet_header_arr[1]) & 0x3F;
        if (packet_length < 2) return false;  // need at least header+something
    } else {
        packet_length = 0; // DATA path not implemented
    }

    // --- Extract PDU+CRC (still whitened) ---
    const int bytes_pdu_crc = packet_length + 2 + 3; // header(2) + CRC(3)
    ExtractBytes(5 * 8, packet_data, bytes_pdu_crc);

    // De-whiten PDU+CRC and set CRC init
    btle_reverse_whiten(chan, packet_data, bytes_pdu_crc);
    if (packet_addr_l == LE_ADV_AA) {
        crc[0] = crc[1] = crc[2] = 0x55;  // ADV init CRC
    } else {
        crc[0] = crc[1] = crc[2] = 0x00;  // placeholder (DATA not implemented)
    }

    // --- CRC check on (header+payload), excluding CRC bytes ---
    calced_crc = btle_reverse_crc(packet_data, packet_length + 2, crc);
    packet_crc = 0;
    for (int c = 0; c < 3; ++c) {
        packet_crc = (packet_crc << 8) | packet_data[packet_length + 2 + c];
    }

    // --- Compute the packet span (COMPLEX samples) ---
    // symbols: preamble(8) + AA(32) + (len+2+3)*8
    const int symbols_total_with_preamble = 8 + 32 + (packet_length + 2 + 3) * 8;
    const double sps = (double)g_srate;                  // SPS (≈ 2 at 2 MS/s)
    const uint64_t sample_span = (uint64_t) llround(symbols_total_with_preamble * sps);

    // Absolute indices based on the decoder’s cursor
    const uint64_t sample_start = abs_cursor;
    const uint64_t sample_end   = sample_start + sample_span;

    // Move cursor forward for what we just consumed
    abs_cursor = sample_end;

    // --- If CRC matches, build the lell_packet and callback ---
    if (packet_crc == calced_crc) {
        lell_packet pkt{};
        pkt.access_address = (packet_addr_l == LE_ADV_AA) ? LE_ADV_AA : (uint32_t)packet_addr_l;
        pkt.channel_idx    = chan;
        pkt.length         = packet_length;

        // De-whitened PDU first byte is type/addrs
        pkt.adv_type   = packet_data[0] & 0x0F;
        pkt.adv_tx_add = (packet_data[0] & 0x40) ? 1 : 0;
        pkt.adv_rx_add = (packet_data[0] & 0x80) ? 1 : 0;

        pkt.flags.as_bits.access_address_ok = (pkt.access_address == 0x8E89BED6);
        pkt.access_address_offenses = 0;

        // 4 bytes of AA (little-endian here to match your downstream)
        pkt.symbols[0] = (uint8_t)(pkt.access_address >>  0);
        pkt.symbols[1] = (uint8_t)(pkt.access_address >>  8);
        pkt.symbols[2] = (uint8_t)(pkt.access_address >> 16);
        pkt.symbols[3] = (uint8_t)(pkt.access_address >> 24);

        // header+payload+crc (whiten-reversed, then bit-swapped for output)
        for (int i = 0; i < packet_length + 2 + 3; ++i) {
            pkt.symbols[i + 4] = SwapBits(packet_data[i]);
        }

        // Absolute sample stamps (COMPLEX-sample units)
        pkt.sample_start = sample_start;
        pkt.sample_end   = sample_end;
        pkt.srate_hz     = g_srate;

        callback(pkt);
        return true;
    }

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