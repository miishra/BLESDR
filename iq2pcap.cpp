// iq2pcap.cpp - Read complex-float I/Q, decode BLE with BLESDR, dump PCAP (DLT 251).
// Build: your existing CMake target that links with lib/BLESDR*.cpp
// Usage:
//   ./iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out_ch37.pcap [--decim 2] [--dump-iq-dir iq_dir] [--prepad-us 200]
// Notes:
//   * If you captured at 4 MS/s, use --decim 2 to feed ~2 MS/s complex into the decoder (what this BLESDR expects).
//   * This harness assumes BLESDR fires a callback per decoded packet. We slice PCAP bytes from lell_packet.symbols.

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

// --------- Simple PCAP (DLT 251 = BLUETOOTH_LE_LL) ----------
namespace pcap {
static constexpr uint32_t MAGIC   = 0xA1B2C3D4;
static constexpr uint16_t VMAJOR  = 2;
static constexpr uint16_t VMINOR  = 4;
static constexpr uint32_t SNAPLEN = 0xFFFF;
static constexpr uint32_t LINKTYPE_BLE_LL = 251;

struct Writer {
    std::FILE* f = nullptr;
    explicit Writer(const std::string& path) {
        f = std::fopen(path.c_str(), "wb");
        if (!f) { throw std::runtime_error("fopen failed: " + path); }
        // global header
        uint32_t magic = MAGIC;
        uint16_t vmaj = VMAJOR, vmin = VMINOR;
        uint32_t thiszone = 0, sigfigs = 0, snaplen = SNAPLEN, network = LINKTYPE_BLE_LL;
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

// ---------- Helpers ----------
struct Args {
    std::string file;
    std::string out = "out.pcap";
    int channel = 37;       // 37/38/39
    double fs = 4e6;        // input sample rate (complex baseband)
    int decim = 2;          // complex decimation (4->2 typical). 1 = no decimation
    size_t chunk = 1'000'000; // complex samples per read

    // New: per-packet IQ dumping
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

// --------- Packet glue: capture PDUs from BLESDR ----------
struct PduStore {
    std::vector< std::vector<uint8_t> > pdus;
    void clear() { pdus.clear(); }
    void add(const uint8_t* p, size_t n){ pdus.emplace_back(p, p+n); }
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
// This fork exposes bytes via lell_packet.symbols[]:
// symbols[0..3] = AA (LE), symbols[4..] = header+payload+CRC. length = payload length.
static void attach_packet_handler(BLESDR& b, PduStore& store, DumpCtx* dctx, pcap::Writer& w) {
    static size_t seen = 0;
    b.callback = [&](lell_packet pkt){
        // 1) PCAP: header(2) + payload(len) + CRC(3) = length + 5 bytes at symbols+4 (AA in [0..3])
        const uint8_t* ptr = pkt.symbols + 4;
        size_t len = (size_t)pkt.length + 5;
        if (len > 0 && len <= MAX_LE_SYMBOLS - 4) {
            store.add(ptr, len);
            w.write_pkt(ptr, len);
            if (seen < 5) {
                std::fprintf(stderr, "[BLESDR] pkt len=%zu (chan=%u) first16=", len, pkt.channel_idx);
                for (size_t i=0;i<std::min(len,(size_t)16);++i) std::fprintf(stderr, "%02X", ptr[i]);
                std::fprintf(stderr, "\n");
            }
            ++seen;
        }

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

    // Attach the packet handler so every decoded packet goes straight to PCAP (and optionally IQ dump)
    attach_packet_handler(blesdr, store, dctx.enabled ? &dctx : nullptr, w);

    // Feed chunks to the decoder
    size_t total_complex = 0, total_complex_fed = 0, total_pdus = 0;

    for(;;){
        // bufIQ has 2*chunk floats; read 'args.chunk' complex samples per fread
        size_t nread = std::fread(bufIQ.data(), sizeof(float)*2, args.chunk, f);
        if (nread == 0) break;
        
        // Complex decimation (keeps I,Q interleaved)
        size_t n_cplx_out = decimate_cplx(bufIQ.data(), nread, args.decim, workIQ);

        // DC-remove + RMS normalize per component (helps some forks)
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

        // (PCAP writing happens inside the callback already, but keep a counter)
        total_pdus += store.pdus.size();
        store.clear();

        total_complex     += nread;
        total_complex_fed += n_cplx_out;
    }
    std::fclose(f);

    std::cerr << "Done. Complex read: " << total_complex
              << ", complex fed: " << total_complex_fed
              << ", PDUs (callback count): " << total_pdus
              << (args.dump_iq_dir.empty() ? "" : (", IQ dump dir: " + args.dump_iq_dir))
              << "\n";
    return 0;
}
