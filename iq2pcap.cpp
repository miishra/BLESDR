// iq2pcap.cpp - Read complex-float I/Q, decode BLE with BLESDR, dump PCAP
// FIXED: write LINKTYPE = DLT 256 (BLUETOOTH_LE_LL_WITH_PHDR) and prepend the
//        required per-packet pseudo-header so Wireshark recognizes ADV_* PDUs.
//        Also include the Access Address in the packet bytes (AA + PDU + CRC).
//
// Build: your existing CMake target that links with lib/BLESDR*.cpp
// Usage:
//   ./iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out_ch37.pcap [--decim 2] [--dump-iq-dir iq_dir] [--prepad-us 200]
//
// Notes:
//   * DLT 256 pseudo-header layout (packed, little-endian):
//       uint8  rf_channel;                // 0..39 (adv: 37/38/39)
//       int8   signal_power;              // dBm; valid if flags bit 0x0002 set
//       int8   noise_power;               // dBm; valid if flags bit 0x0004 set
//       uint8  access_address_offenses;   // valid if flags bit 0x0020 set
//       uint32 ref_access_address;        // valid if flags bit 0x0010 set
//       uint16 flags;                     // see bits below
//       uint8  le_packet[];               // AA + PDU + CRC (no preamble)
//     Flag bits we use here:
//       0x0001 => le_packet is de-whitened (we set this)
//       0x0010 => ref_access_address is valid (we set this)
//     (We DO NOT claim CRC checked/passed; Wireshark will show CRC Checked: False.)
//   * Packet bytes written after the pseudo-header are: 4-byte AA + (2+len+3).
//     Your BLESDR callback already exposes symbols[] = AA(4) + header+payload+CRC.
//   * If your decoder outputs LSB-first bits inside each byte, bit-reverse each byte
//     before writing (helper provided below). If AA appears as 0x8e89bed6 in Wireshark,
//     your bit ordering is fine.

// ------------------ Includes ------------------
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "lib/BLESDR.hpp"   // adjust include path if needed

// ------------------ Simple PCAP writer ------------------
namespace pcap {
static constexpr uint32_t MAGIC   = 0xA1B2C3D4;
static constexpr uint16_t VMAJOR  = 2;
static constexpr uint16_t VMINOR  = 4;
static constexpr uint32_t SNAPLEN = 0xFFFF;
static constexpr uint32_t LINKTYPE_BLE_LL_WITH_PHDR = 256; // DLT 256

#pragma pack(push, 1)
struct le_phdr {
    uint8_t  rf_channel;            // 0..39 (adv: 37/38/39)
    int8_t   signal_power;          // dBm; valid iff flags & 0x0002
    int8_t   noise_power;           // dBm; valid iff flags & 0x0004
    uint8_t  access_address_offenses; // valid iff flags & 0x0020
    uint32_t ref_access_address;    // valid iff flags & 0x0010 (LE)
    uint16_t flags;                 // bitfield, see below
};
// Flag bits (subset used here)
static constexpr uint16_t LE_FLAG_DEWHITENED      = 0x0001;
static constexpr uint16_t LE_FLAG_SIGNAL_VALID    = 0x0002;
static constexpr uint16_t LE_FLAG_NOISE_VALID     = 0x0004;
static constexpr uint16_t LE_FLAG_REF_AA_VALID    = 0x0010;
static constexpr uint16_t LE_FLAG_AA_OFFENSES_OK  = 0x0020;
static constexpr uint16_t LE_FLAG_CRC_CHECKED     = 0x0400;
static constexpr uint16_t LE_FLAG_CRC_VALID       = 0x0800;
#pragma pack(pop)

struct Writer {
    std::FILE* f = nullptr;
    explicit Writer(const std::string& path) {
        f = std::fopen(path.c_str(), "wb");
        if (!f) { throw std::runtime_error("fopen failed: " + path); }
        // global header (native endian, classic pcap)
        uint32_t magic = MAGIC;
        uint16_t vmaj = VMAJOR, vmin = VMINOR;
        uint32_t thiszone = 0, sigfigs = 0, snaplen = SNAPLEN, network = LINKTYPE_BLE_LL_WITH_PHDR;
        std::fwrite(&magic,   4,1,f);
        std::fwrite(&vmaj,    2,1,f);
        std::fwrite(&vmin,    2,1,f);
        std::fwrite(&thiszone,4,1,f);
        std::fwrite(&sigfigs, 4,1,f);
        std::fwrite(&snaplen, 4,1,f);
        std::fwrite(&network, 4,1,f);
    }
    void write_pkt(const uint8_t* data, size_t len, double ts_sec_f = -1.0) {
        using clock = std::chrono::system_clock;
        double now = ts_sec_f >= 0 ? ts_sec_f
                                   : std::chrono::duration<double>(clock::now().time_since_epoch()).count();
        uint32_t ts_sec  = static_cast<uint32_t>(now);
        uint32_t ts_usec = static_cast<uint32_t>((now - ts_sec)*1e6 + 0.5);
        uint32_t incl = static_cast<uint32_t>(len);
        uint32_t orig = static_cast<uint32_t>(len);
        std::fwrite(&ts_sec,  4,1,f);
        std::fwrite(&ts_usec, 4,1,f);
        std::fwrite(&incl,    4,1,f);
        std::fwrite(&orig,    4,1,f);
        if (len) std::fwrite(data, 1, len, f);
    }
    ~Writer(){ if(f) std::fclose(f); }
};
} // namespace pcap

// ------------------ Helpers ------------------
struct Args {
    std::string file;
    std::string out = "out.pcap";
    int channel = 37;       // 37/38/39
    double fs = 4e6;        // input sample rate (complex baseband)
    int decim = 2;          // complex decimation (4->2 typical). 1 = no decimation
    size_t chunk = 1'000'000; // complex samples per read

    // Optional per-packet IQ dumping
    std::string dump_iq_dir = "";  // empty disables
    int prepad_us = 200;           // prepend this many microseconds of IQ before packet
};

static void die(const std::string& s) { std::cerr << "error: " << s << "\n"; std::exit(1); }

static Args parse(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;i++){
        std::string k = argv[i];
        auto need = [&](const char* name)->const char*{
            if (i+1>=argc) die(std::string("missing value after ")+name);
            return argv[++i];
        };
        if (k=="--file")         a.file = need("--file");
        else if (k=="--out")     a.out  = need("--out");
        else if (k=="--fs")      a.fs   = std::stod(need("--fs"));
        else if (k=="--channel") a.channel = std::stoi(need("--channel"));
        else if (k=="--decim")   a.decim= std::stoi(need("--decim"));
        else if (k=="--chunk")   a.chunk= static_cast<size_t>(std::stoll(need("--chunk")));
        else if (k=="--dump-iq-dir") a.dump_iq_dir = need("--dump-iq-dir");
        else if (k=="--prepad-us")   a.prepad_us   = std::stoi(need("--prepad-us"));
        else if (k=="-h" || k=="--help"){
            std::cout <<
"Usage: iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out.pcap [--decim 2] [--chunk 1000000]\n"
"              [--dump-iq-dir iq_dir] [--prepad-us 200]\n"
"  file:    complex float32 interleaved I/Q capture at baseband (CH37/38/39 center)\n"
"  fs:      input sample rate in Hz (e.g., 4e6)\n"
"  channel: 37/38/39 (adv channels)\n"
"  decim:   integer complex decimation (typ 2 for 4->2 MS/s)\n"
"  chunk:   complex samples per read per iteration\n"
"  dump-iq-dir: if set, write pkt_XXXX.fc32 (float32 I,Q) per decoded packet\n"
"  prepad-us: microseconds of I/Q preceding packet to include in the dump\n";
            std::exit(0);
        }
    }
    if (a.file.empty()) die("please specify --file");
    if (a.channel<37 || a.channel>39) die("channel must be 37, 38 or 39");
    if (a.decim<1) a.decim = 1;
    return a;
}

// Decimate interleaved complex float32 stream by N (keep every Nth complex sample)
static size_t decimate_cplx(const float* iq, size_t n_cplx, int N, std::vector<float>& outIQ) {
    if (N < 1) N = 1;
    outIQ.clear();
    outIQ.reserve(2 * (n_cplx / (size_t)N + 16));
    for (size_t k = 0; k < n_cplx; k += (size_t)N) {
        outIQ.push_back(iq[2*k]);     // I
        outIQ.push_back(iq[2*k + 1]); // Q
    }
    return outIQ.size() / 2; // # of complex samples
}

// Optional: bit-reverse if your decoder returns LSB-first bits in each byte.
// If AA shows correctly as 0x8E89BED6 in Wireshark, you probably don't need this.
static inline uint8_t bitrev8(uint8_t x){
    x = (uint8_t)((x>>4) | (x<<4));
    x = (uint8_t)(((x&0xCC)>>2) | ((x&0x33)<<2));
    x = (uint8_t)(((x&0xAA)>>1) | ((x&0x55)<<1));
    return x;
}
static void bitrev_buf(uint8_t* p, size_t n) {
    for (size_t i=0;i<n;i++) p[i] = bitrev8(p[i]);
}

// --------- Packet glue: capture PDUs from BLESDR ----------
struct PduStore {
    std::vector< std::vector<uint8_t> > frames; // full frames written to pcap (phdr + AA+PDU+CRC)
    void clear() { frames.clear(); }
    void add(const std::vector<uint8_t>& v){ frames.push_back(v); }
};

// --------- Ring buffer for I/Q (for per-packet dump) ----------
struct Ring {
    std::vector<float> buf; // interleaved I,Q
    size_t w = 0;           // write index in floats
    size_t cap = 0;         // capacity in complex samples
    void init(size_t complex_len) {
        cap = std::max<size_t>(complex_len, 4096);
        buf.assign(2*cap, 0.0f);
        w = 0;
    }
    inline void push(float I, float Q) {
        buf[w] = I;
        size_t w2 = (w+1)%(2*cap);
        buf[w2] = Q;
        w = (w+2)%(2*cap);
    }
    // copy last 'take_floats' floats (I,Q interleaved) into out
    void copy_tail(size_t take_floats, std::vector<float>& out) const {
        size_t maxf = 2*cap;
        if (take_floats > maxf) take_floats = maxf;
        out.resize(take_floats);
        size_t start = ( (w + maxf) - take_floats ) % maxf;
        for (size_t i=0;i<take_floats;i++) out[i] = buf[(start + i)%maxf];
    }
};

// Context passed to callback for IQ dumping
struct DumpCtx {
    bool enabled=false;
    std::string dir;
    int sps=2;            // samples/symbol at fs_eff (2 for 2 MS/s)
    double fs_eff=2e6;    // complex sample rate after decim
    int prepad_us=200;    // µs of context before packet
    Ring* ring=nullptr;
    size_t pkt_idx=0;
};

// ========= Attach handler using your class member `callback` =========
// BLESDR exposes bytes via lell_packet.symbols[]:
// symbols[0..3] = AA (LE), symbols[4..] = header+payload+CRC. length = payload length.
static void attach_packet_handler(BLESDR& b, PduStore& store, DumpCtx* dctx, pcap::Writer& w, int rf_channel) {
    static size_t seen = 0;
    b.callback = [&](lell_packet pkt){
        // Build the DLT 256 pseudo-header
        pcap::le_phdr ph{};
        ph.rf_channel = static_cast<uint8_t>(rf_channel); // 37/38/39
        ph.signal_power = 127;    // unknown (valid bit not set)
        ph.noise_power  = 127;    // unknown (valid bit not set)
        ph.access_address_offenses = 0; // unknown (valid bit not set)
        ph.ref_access_address = 0x8E89BED6u; // advertising AA (little-endian in file)
        ph.flags = pcap::LE_FLAG_DEWHITENED | pcap::LE_FLAG_REF_AA_VALID;
        // NOTE: Do not set CRC checked/valid bits unless you compute/verify CRC.

        // Packet data to follow: AA (4) + (header+payload+CRC)
        const uint8_t* bytes_aa = pkt.symbols;       // 4 bytes AA
        const uint8_t* bytes_pdu = pkt.symbols + 4;  // header+payload+CRC
        size_t pdu_len = static_cast<size_t>(pkt.length) + 5; // 2 hdr + payload_len + 3 CRC
        size_t frame_len = sizeof(ph) + 4 + pdu_len;

        std::vector<uint8_t> frame;
        frame.resize(frame_len);

        // Copy pseudo-header
        std::memcpy(frame.data(), &ph, sizeof(ph));
        // Copy AA + PDU+CRC (already de-whitened by BLESDR)
        std::memcpy(frame.data()+sizeof(ph), bytes_aa, 4);
        std::memcpy(frame.data()+sizeof(ph)+4, bytes_pdu, pdu_len);

        // If your decoder outputs LSB-first bits per byte, uncomment this:
        // bitrev_buf(frame.data()+sizeof(ph), 4 + pdu_len);

        store.add(frame);
        w.write_pkt(frame.data(), frame.size());

        if (seen < 5) {
            std::fprintf(stderr, "[BLESDR] wrote frame len=%zu ch=%u AA=%02X%02X%02X%02X first16=",
                         frame.size(), pkt.channel_idx,
                         bytes_aa[3], bytes_aa[2], bytes_aa[1], bytes_aa[0]);
            const uint8_t* dbg = frame.data()+sizeof(ph);
            for (size_t i=0;i<std::min<size_t>(4+pdu_len,16);++i) std::fprintf(stderr, "%02X", dbg[i]);
            std::fprintf(stderr, "\n");
        }
        ++seen;

        // 2) Optional: dump interleaved (I,Q) window around this packet
        if (dctx && dctx->enabled && dctx->ring) {
            // Conservative bit count: preamble(8)+AA(32)+hdr(16)+8*len + CRC(24)
            size_t bits = 8 + 32 + 16 + 8*static_cast<size_t>(pkt.length) + 24;
            size_t samps_needed = bits * static_cast<size_t>(dctx->sps);
            size_t prepad_samps = static_cast<size_t>((dctx->fs_eff * dctx->prepad_us)/1e6);
            // Convert to float count (I,Q → 2 floats per complex)
            size_t take_floats = 2 * (prepad_samps + samps_needed);
            if (take_floats < 2*64) take_floats = 2*64;

            std::vector<float> outIQ;
            dctx->ring->copy_tail(take_floats, outIQ);

            char fname[256];
            std::snprintf(fname, sizeof(fname), "%s/pkt_%06zu.fc32", dctx->dir.c_str(), dctx->pkt_idx++);
            if (std::FILE* fp = std::fopen(fname, "wb")) {
                std::fwrite(outIQ.data(), sizeof(float), outIQ.size(), fp);
                std::fclose(fp);
            }
        }
    };
}

int main(int argc, char** argv){
    auto args = parse(argc, argv);

    // Open capture (complex float32 interleaved)
    std::FILE* f = std::fopen(args.file.c_str(), "rb");
    if(!f) die(std::string("cannot open file: ") + args.file + " : " + std::strerror(errno));

    // Effective complex sample rate after decim & derived SPS (for dumping window sizing)
    const double fs_eff = args.fs / args.decim;
    const int sps = std::max(2, static_cast<int>(std::lround(fs_eff / 1e6))); // BLE-1M ⇒ ~2 at 2 MS/s

    if (!args.dump_iq_dir.empty()) {
        std::string cmd = "mkdir -p '" + args.dump_iq_dir + "'";
        std::system(cmd.c_str());
    }

    pcap::Writer w(args.out);
    std::vector<float> bufIQ(2*args.chunk);
    std::vector<float> workIQ;           // decimated complex
    PduStore store;
    Ring ring;                           // ring for IQ dump context
    ring.init(static_cast<size_t>(fs_eff * 0.050)); // 50 ms ring

    BLESDR blesdr;

    DumpCtx dctx;
    dctx.enabled   = !args.dump_iq_dir.empty();
    dctx.dir       = args.dump_iq_dir;
    dctx.sps       = sps;
    dctx.fs_eff    = fs_eff;
    dctx.prepad_us = args.prepad_us;
    dctx.ring      = &ring;

    // Attach the packet handler so every decoded packet goes straight to PCAP (with pseudo-header)
    attach_packet_handler(blesdr, store, dctx.enabled ? &dctx : nullptr, w, args.channel);

    // Feed chunks to the decoder
    size_t total_complex = 0, total_complex_fed = 0, total_frames = 0;

    for(;;){
        // bufIQ has 2*chunk floats; read 'args.chunk' complex samples per fread
        size_t nread = std::fread(bufIQ.data(), sizeof(float)*2, args.chunk, f);
        if (nread == 0) break;
        
        // Complex decimation (keeps I,Q interleaved)
        size_t n_cplx_out = decimate_cplx(bufIQ.data(), nread, args.decim, workIQ);

        // DC-remove + RMS normalize per component
        {
            double meanI=0, meanQ=0;
            for (size_t i=0;i<n_cplx_out;i++){ meanI += workIQ[2*i]; meanQ += workIQ[2*i+1]; }
            if (n_cplx_out) { meanI/=n_cplx_out; meanQ/=n_cplx_out; }
            double e=0;
            for (size_t i=0;i<n_cplx_out;i++){
                workIQ[2*i]   = float(workIQ[2*i]   - meanI);
                workIQ[2*i+1] = float(workIQ[2*i+1] - meanQ);
                e += (double)workIQ[2*i]*workIQ[2*i] + (double)workIQ[2*i+1]*workIQ[2*i+1];
            }
            e = std::sqrt(e / std::max<double>(2.0*n_cplx_out,1.0));
            if (e > 1e-12) for (size_t i=0;i<n_cplx_out;i++){ workIQ[2*i]/=e; workIQ[2*i+1]/=e; }
        }
        
        // Push into ring, then feed BLESDR (expects complex interleaved), samples_len = #complex samples.
        for (size_t i=0;i<n_cplx_out;i++) ring.push(workIQ[2*i], workIQ[2*i+1]);
        blesdr.Receiver((size_t)args.channel, workIQ.data(), n_cplx_out);

        // Count frames
        total_frames += store.frames.size();
        store.clear();

        total_complex     += nread;
        total_complex_fed += n_cplx_out;
    }
    std::fclose(f);

    std::cerr << "Done. Complex read: " << total_complex
              << ", complex fed: " << total_complex_fed
              << ", frames written: " << total_frames
              << (args.dump_iq_dir.empty() ? "" : (", IQ dump dir: " + args.dump_iq_dir))
              << "\n";
    return 0;
}