// make_cfo_iq.cpp
// Build two files: ble_20pkts_cfo0.f32 and ble_20pkts_cfo+50k.f32
// Uses your BLESDR class to synthesize one BLE advertising packet, repeats 20 times,
// inserts gaps, and applies a complex rotation for CFO.
//
// Compile (example):
//   g++ -O3 -std=c++17 -o make_cfo_iq make_cfo_iq.cpp BLESDR.cpp  -lm
//
// Run (defaults to 20 packets, ch37, fs=1e6*SAMPLE_PER_SYMBOL if available else 4e6):
//   ./make_cfo_iq
//
// Optional args:
//   --num 20           (number of packets)
//   --chan 37          (BLE channel: 37, 38, 39, or data channels 0..36)
//   --gap-ms 2.0       (zero-IQ gap between packets)
//   --fs 4000000       (sample rate in Hz; if omitted, tries 1e6*SAMPLE_PER_SYMBOL)
//   --cfo 50000        (+CFO in Hz for the second file)
//   --out0 ble_20pkts_cfo0.f32
//   --out1 ble_20pkts_cfo+50k.f32
//
// Notes:
// - Files are float32, interleaved I/Q, compatible with GNU Radio, Inspectrum, etc.
// - CFO is modeled as baseband rotation: x[n]*e^{j 2π f_cfo n / fs}.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Your encoder header:
#include "BLESDR.hpp"  // Must provide sample_for_ADV_IND(...)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- Helpers ----------
static inline void write_float32_iq(const std::string& path, const std::vector<float>& iq) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open '" << path << "' for writing.\n";
        std::exit(1);
    }
    f.write(reinterpret_cast<const char*>(iq.data()), static_cast<std::streamsize>(iq.size() * sizeof(float)));
    if (!f) {
        std::cerr << "ERROR: write failed for '" << path << "'.\n";
        std::exit(1);
    }
}

static inline void append_silence(std::vector<float>& v, size_t n_samp) {
    v.insert(v.end(), 2 * n_samp, 0.0f); // I,Q zeros
}

// Rotate interleaved I/Q by CFO (Hz) at fs (Hz). Uses oscillator recurrence (fast, stable).
static inline void apply_cfo(std::vector<float>& iq, double f_cfo_hz, double fs_hz, float phase0 = 0.0f) {
    if (f_cfo_hz == 0.0) return;
    const double dtheta = 2.0 * M_PI * f_cfo_hz / fs_hz;
    float c = std::cos(phase0), s = std::sin(phase0);
    const float cd = std::cos((float)dtheta), sd = std::sin((float)dtheta);

    for (size_t n = 0; n + 1 < iq.size(); n += 2) {
        float i = iq[n], q = iq[n + 1];
        float ip = i * c - q * s;
        float qp = i * s + q * c;
        iq[n] = ip; iq[n + 1] = qp;

        float c_next = c * cd - s * sd;
        float s_next = c * sd + s * cd;
        c = c_next; s = s_next;
    }
}

static inline double detect_default_fs_hz() {
#ifdef SAMPLE_PER_SYMBOL
    // BLE 1M PHY is 1 Msym/s
    return 1e6 * (double)SAMPLE_PER_SYMBOL;
#else
    // Fallback if macro not visible here; override with --fs if needed.
    return 4e6;
#endif
}

// Build the two buffers and write them to disk.
static void generate_two_files_with_cfo(
    BLESDR& enc,
    size_t chan,
    double fs_hz,
    size_t num_packets,
    double gap_ms,
    double pos_cfo_hz,
    const std::string& out0,
    const std::string& out1)
{
    const size_t gap_samp = static_cast<size_t>(std::llround(fs_hz * (gap_ms * 1e-3)));

    std::vector<float> buf_cfo0;
    std::vector<float> buf_cfo_pos;

    // Example payload (customize as needed)
    uint8_t payload[] = { 'H','e','l','l','o','B','L','E' };
    const uint8_t data_type = 0xFF; // Manufacturer-specific data (ok for demo)

    for (size_t k = 0; k < num_packets; ++k) {
        // One BLE ADV_IND packet baseband I/Q
        auto pkt = enc.sample_for_ADV_IND(chan, data_type, payload, sizeof(payload));

        // 0 Hz CFO stream
        buf_cfo0.insert(buf_cfo0.end(), pkt.begin(), pkt.end());
        append_silence(buf_cfo0, gap_samp);

        // +CFO stream
        auto pkt_cfo = pkt;
        apply_cfo(pkt_cfo, pos_cfo_hz, fs_hz);
        buf_cfo_pos.insert(buf_cfo_pos.end(), pkt_cfo.begin(), pkt_cfo.end());
        append_silence(buf_cfo_pos, gap_samp);
    }

    write_float32_iq(out0, buf_cfo0);
    write_float32_iq(out1, buf_cfo_pos);
    std::cout << "Wrote: " << out0 << " (" << buf_cfo0.size()/2 << " complex samples)\n";
    std::cout << "Wrote: " << out1 << " (" << buf_cfo_pos.size()/2 << " complex samples)\n";
}

// ---------- CLI parsing ----------
static inline bool arg_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int main(int argc, char** argv) {
    size_t num_packets = 20;
    size_t chan = 37;               // 37/38/39 or data channel 0..36
    double gap_ms = 2.0;
    double fs_hz = detect_default_fs_hz();
    double cfo_hz = 50e3;
    std::string out0 = "ble_20pkts_cfo0.f32";
    std::string out1 = "ble_20pkts_cfo+50k.f32";

    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--num") && i + 1 < argc)        num_packets = std::stoul(argv[++i]);
        else if (arg_eq(argv[i], "--chan") && i + 1 < argc)  chan = std::stoul(argv[++i]);
        else if (arg_eq(argv[i], "--gap-ms") && i + 1 < argc) gap_ms = std::stod(argv[++i]);
        else if (arg_eq(argv[i], "--fs") && i + 1 < argc)    fs_hz = std::stod(argv[++i]);
        else if (arg_eq(argv[i], "--cfo") && i + 1 < argc)   cfo_hz = std::stod(argv[++i]);
        else if (arg_eq(argv[i], "--out0") && i + 1 < argc)  out0 = argv[++i];
        else if (arg_eq(argv[i], "--out1") && i + 1 < argc)  out1 = argv[++i];
        else if (arg_eq(argv[i], "--help")) {
            std::cout <<
            "Usage: " << argv[0] << " [--num N] [--chan CH] [--gap-ms MS] [--fs HZ] [--cfo HZ]\n"
            "                  [--out0 FILE0] [--out1 FILE1]\n";
            return 0;
        }
    }

    std::cout << "Settings:\n"
              << "  packets : " << num_packets << "\n"
              << "  channel : " << chan << "\n"
              << "  fs (Hz) : " << fs_hz << "\n"
              << "  gap (ms): " << gap_ms << "\n"
              << "  CFO (Hz): +" << cfo_hz << "\n"
              << "  out0    : " << out0 << "\n"
              << "  out1    : " << out1 << "\n";

    try {
        BLESDR enc; // uses your constructor; SAMPLE_PER_SYMBOL & filter are in your code
        generate_two_files_with_cfo(enc, chan, fs_hz, num_packets, gap_ms, cfo_hz, out0, out1);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}