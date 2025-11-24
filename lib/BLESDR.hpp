/*
 *  BLESDR.hpp - BLE SDR Decoder Header (Fixed with proper I/Q alignment)
 *  Copyright 2017 by Jiang Wei <jiangwei@jiangwei.org>
 */

#pragma once
#include <stdint.h>
#include <vector>
#include <functional>
#include <cstddef>
#include <complex>

#define MAX_NUM_PHY_SAMPLE 1520
#define MAX_NUM_CHAR_CMD (256)
#define MAX_NUM_PHY_BYTE (47)
#define SAMPLE_PER_SYMBOL 2
#define LEN_GAUSS_FILTER (4)

#define MAX_LE_SYMBOLS 64
#define LE_ADV_AA 0x8E89BED6

#define ADV_IND         0
#define ADV_DIRECT_IND  1
#define ADV_NONCONN_IND 2
#define SCAN_REQ        3
#define SCAN_RSP        4
#define CONNECT_REQ     5
#define ADV_SCAN_IND    6

#define RB_SIZE 1000

struct lell_packet {
    // Raw unwhitened bytes of packet, including access address
    uint8_t symbols[MAX_LE_SYMBOLS];

    uint32_t access_address;

    // Channel index
    uint8_t channel_idx;
    uint8_t channel_k;

    // Number of symbols
    int length;

    uint32_t clk100ns;

    // Advertising packet header info
    uint8_t adv_type;
    int adv_tx_add;
    int adv_rx_add;

    unsigned access_address_offenses;
    uint32_t refcount;

    // I/Q sample index window for this packet in the original stream
    // Half-open range: [sample_start, sample_end)
    // These are ABSOLUTE complex-sample indices aligned with preamble start
    uint64_t sample_start = 0;
    uint64_t sample_end   = 0;
    
    // CFO estimates computed on exact packet window
    double cfo_exact_quick_hz = 0.0;
    double cfo_exact_ls_hz    = 0.0;

    // Sample rate used when decoding (Hz)
    int srate_hz = 0;

    // Absolute cursor position when packet was detected
    uint64_t head_at_detect = 0;

    float cfo_hz{};
    float phi0_rad{};

    /* Flags */
    union {
        struct {
            uint32_t access_address_ok : 1;
        } as_bits;
        uint32_t as_word;
    } flags;
};

struct PacketSpaceMapping {
		// Symbol space: Where detection/extraction happens
		int rb_head_snapshot = 0;      // rb_head when preamble detected
		uint64_t abs_cursor_snapshot = 0;  // abs_cursor when preamble detected
		
		// I/Q space: Absolute complex sample indices for chunk extraction
		uint64_t iq_start_abs = 0;
		uint64_t iq_end_abs = 0;
		
		bool valid = false;
		
		// ========================================================================
		// Core mapping function: RB(offset) → absolute I/Q index
		// ========================================================================
		uint64_t rb_offset_to_iq_absolute(int rb_offset) const {
			if (!valid) return 0;
			
			// RB(k) = rb_buf[(rb_head + k) % RB_SIZE]
			const int rb_slot = (rb_head_snapshot + rb_offset) % RB_SIZE;
			
			// For circular buffer after wrap (abs_cursor >= RB_SIZE):
			//   rb_buf[slot] contains sample from absolute position:
			//     - If slot > rb_head: old data, absolute index = slot
			//     - If slot <= rb_head: new data, absolute index = (abs_cursor - rb_head + slot)
			
			if (abs_cursor_snapshot >= RB_SIZE) {
				if (rb_slot > rb_head_snapshot) {
					return rb_slot;  // Old data
				} else {
					return abs_cursor_snapshot - rb_head_snapshot + rb_slot;  // New data
				}
			} else {
				// No wrap yet, simple 1:1 mapping
				return rb_slot;
			}
		}
		
		// ========================================================================
		// Calculate I/Q range for a packet extracted from RB(start_off)..RB(end_off)
		// ========================================================================
		void calculate_iq_range(int rb_start_offset, int rb_end_offset) {
			iq_start_abs = rb_offset_to_iq_absolute(rb_start_offset);
			iq_end_abs = rb_offset_to_iq_absolute(rb_end_offset) + 1;  // +1 for exclusive end
		}
	};

class BLESDR {
public:
    BLESDR();
    ~BLESDR();
    
    // Callback type for providing I/Q windows based on absolute indices
    using IQWindowProvider =
        std::function<bool(uint64_t start_cx, uint64_t end_cx,
                           std::vector<std::complex<float>>& out)>;

    // Set the I/Q provider callback
    void set_iq_provider(IQWindowProvider p) { iq_provider_ = std::move(p); }

    double get_channel_freq(int channel_number);

    // Absolute I/Q sample cursor (complex samples, post-decimation)
    uint64_t abs_cursor = 0;

    void set_abs_cursor(uint64_t v) { abs_cursor = v; }
    uint64_t get_abs_cursor() const { return abs_cursor; }

	uint64_t abs_cursor_at_preamble_detect = 0;

    double get_sample_rate() {
        return 2e6; // TODO: make configurable
    }

    // Callback invoked when a packet is successfully decoded
    std::function<void(lell_packet)> callback;

    // Configure decoder parameters
    void Configure(int sps, uint8_t chan, int skip) {
        this->srate       = sps;
        this->chan        = chan;
        this->skipSamples = skip;
        this->RB_init();
    }

    // Generate sample data for transmission
    std::vector<float> sample_for_ADV_IND(size_t chan, uint8_t data_type, 
                                          uint8_t* buff, size_t bufflen);
    std::vector<float> sample_for_RAW(uint8_t* buff, size_t bufflen);
    std::vector<float> sample_for_iBeacon(size_t chan, uint8_t* uuid, 
                                          uint16_t Major, uint16_t Minor);
    std::vector<float> sample_for_Packet(size_t chan, lell_packet pocket);

    // Receive and decode samples
    void Receiver(size_t channel, float* samples, size_t samples_len);
    
    // Detection window tracking
    struct DetectWindow {
        bool     valid = false;
        uint64_t start = 0;  // Absolute complex index
        uint64_t end   = 0;  // Absolute complex index (exclusive)
    };
    
    DetectWindow last_detect_window_;

	const PacketSpaceMapping& get_packet_mapping() const { return packet_map_; }
    
    inline void set_detect_window(uint64_t s, uint64_t e) {
        last_detect_window_.valid = (e > s);
        last_detect_window_.start = s;
        last_detect_window_.end   = e;
    }
    
    inline bool take_detect_window(uint64_t& s, uint64_t& e) {
        if (!last_detect_window_.valid) return false;
        s = last_detect_window_.start;
        e = last_detect_window_.end;
        last_detect_window_.valid = false; // Consume once
        return true;
    }

private:
    std::vector<float> iqsamples;
    
    // I/Q window provider callback
    IQWindowProvider iq_provider_;

    size_t byte_to_bits(uint8_t* byte, size_t len, char* bits);

    float* generate_gaussian_taps(unsigned samples_per_sym, unsigned L, double bt);

    void btle_calc_crc(void* src, uint8_t len, uint8_t* dst);

    void btle_whiten(uint8_t chan, uint8_t* buf, uint8_t len);

#define chunk(x,y) ((btle_pdu_chunk*)(x.payload+y))

    struct btle_pdu_chunk {
        uint8_t size;
        uint8_t type;
        uint8_t data[];
    };

    struct btle_adv_pdu {
        // Packet header
        uint8_t pdu_type;
        uint8_t pl_size;

        // MAC address
        uint8_t mac[6];

        // Payload (including 3 bytes for CRC)
        uint8_t payload[42];
    };

	// ============================================================================
	// DUAL-SPACE MAPPING: Clean Implementation
	// ============================================================================
	// Maintains explicit mapping between Symbol Space (RB ring buffer) and 
	// I/Q Space (absolute complex sample indices) to eliminate timing ambiguity

	// Add to BLESDR.hpp:

	PacketSpaceMapping packet_map_;  // Current packet mapping

    int gen_sample_from_phy_bit(char *bit, float *sample, int num_bit);
    float tmp_phy_bit_over_sampling[MAX_NUM_PHY_SAMPLE + 2 * LEN_GAUSS_FILTER*SAMPLE_PER_SYMBOL];
    float tmp_phy_bit_over_sampling1[MAX_NUM_PHY_SAMPLE];
    float * gauss_coef;

    uint8_t chan;
    int32_t g_threshold;
    int g_srate;
    int32_t samples;
    int skipSamples;
    int srate;
    double last_phase;

    int rb_head = -1;
    int16_t *rb_buf;
    
    void RB_init(void);
    void RB_inc(void);

    uint8_t SwapBits(uint8_t a);
    bool Quantize(int16_t l);
    int32_t ExtractThreshold(void);
    bool DetectPreamble(void);
    uint8_t inline ExtractByte(int l);
    void ExtractBytes(int l, uint8_t* buffer, int count);
    bool feedOne(const uint16_t sample);
    bool DecodePacket(int32_t sample, int srate);
    bool DecodeBTLEPacket(int32_t sample, int srate);

    void btle_reverse_whiten(uint8_t chan, uint8_t* data, uint8_t len);
    uint32_t btle_reverse_crc(const uint8_t* data, uint8_t len, uint8_t* dst);
};