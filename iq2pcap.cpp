// iq2pcap.cpp - Read complex-float I/Q, decode BLE with BLESDR, dump PCAP + inline features CSV
// Adds per-packet feature extraction (CFO + IQ/PSD stats) directly from the ring buffer IQ.
// NEW: gating options (--gate none|energy|struct|mid) to narrow CFO window,
//      posthoc detrend+sign-fix CSV (features_signfixed.csv),
//      and a joint CFO+IQ imperfection estimator with Nesterov GD.
//
// Build: link with your existing BLESDR lib (lib/BLESDR*.cpp)
// Usage:
//   ./iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out_ch37.pcap [--decim 2]
//             [--dump-iq-dir iq_dir] [--prepad-us 200] [--features-out features.csv]
//             [--gate energy --gate-k 4.0 --gate-pad-us 8]
//             [--gate struct]    // preamble+AA+header slice
//             [--gate mid --gate-mid-a-us 12 --gate-mid-b-us 80]
//
// Notes on features:
//   - Features are computed from a window of I/Q pulled from the trailing ring when a packet is decoded.
//   - CFO is estimated with: quick (median discr / LS), centroid (FFT), two-stage (coarse+fine+LS).
//   - NEW: cfo_joint_hz via iterative fit on a synthesized GFSK packet model with CFO+IQ impairments.
//   - Robust to discriminator polarity, preamble (0xAA vs 0x55), and small SPS/phase uncertainty.

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
#include <numeric>
#include <complex>
#include <limits>   // for quiet_NaN
#include <random>

#include "lib/BLESDR.hpp"   // adjust include path if needed
// --------- Packet glue: capture PDUs, write PCAP & features ----------
#include "ble_packet_50k_iq.hpp"
#include "ble_test_iq_load.hpp"

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
    // returns the timestamp used (seconds, float)
    double write_pkt(const uint8_t* data, size_t len, double ts_sec_f = -1.0) {
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
        return now;
    }
    ~Writer(){ if(f) std::fclose(f); }
};
} // namespace pcap

// ------------------ Helpers ------------------
static void die(const std::string& s) { std::cerr << "error: " << s << "\n"; std::exit(1); }

// ======== Add near the top ========
struct FeatureRow {
    size_t pkt_idx; double pcap_ts; int rf_channel; int pdu_type;
    std::string adv_addr, access_address;
    double cfo_quick_hz, cfo_centroid_hz, cfo_two_stage_hz;
    double cfo_std_hz, cfo_std_sym_hz;
    double iq_gain_alpha, iq_phase_deg_deg;
    double rise_time_us, psd_centroid_hz, psd_pnr_db, bw_3db_hz, gated_len_us;
    double cfo_two_stage_coarse_hz;

    // NEW: joint estimator outputs
    double cfo_joint_hz;     // CFO from joint fit
    double iq_off_i;         // DC I
    double iq_off_q;         // DC Q
    double iq_eps;           // amplitude imbalance epsilon
    double iq_phi_deg;       // phase imbalance (deg)
    double amp_a;            // overall amplitude
    int    fit_iters;        // iterations used
    double fit_cost;         // final L2/N

    
    // NEW: exact-window CFOs (computed on [sample_start, sample_end))
    double cfo_exact_quick_hz = 0.0; // conjugate-product average
    double cfo_exact_ls_hz    = 0.0; // LS (phase slope) estimate
};

struct FeatureRows {
    std::vector<FeatureRow> rows;
    void push(const FeatureRow& r){ rows.push_back(r); }
    void write_csv(const std::string& path, bool with_signfixed=false,
                   const std::vector<double>* signfixed=nullptr) const {
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) throw std::runtime_error("cannot open csv for write: "+path);
        std::fprintf(f,
        "pkt_idx,pcap_ts,rf_channel,pdu_type,adv_addr,access_address,"
        "cfo_quick_hz,cfo_centroid_hz,cfo_two_stage_hz,cfo_std_hz,cfo_std_sym_hz,"
       "iq_gain_alpha,iq_phase_deg_deg,rise_time_us,psd_centroid_hz,psd_pnr_db,"
       "bw_3db_hz,gated_len_us,cfo_two_stage_coarse_hz,"
        "cfo_joint_hz,iq_off_i,iq_off_q,iq_eps,iq_phi_deg,amp_a,fit_iters,fit_cost,"
      "cfo_exact_quick_hz,cfo_exact_ls_hz");
        if (with_signfixed) std::fprintf(f,",cfo_centroid_hz_signfixed");
        std::fprintf(f,"\n");

        for (size_t i=0;i<rows.size();++i){
            const auto& r = rows[i];
            std::fprintf(f,
            "%zu,%.6f,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6g,%.6f,%.6f",
            r.pkt_idx, r.pcap_ts, r.rf_channel, r.pdu_type,
            r.adv_addr.c_str(), r.access_address.c_str(),
            r.cfo_quick_hz, r.cfo_centroid_hz, r.cfo_two_stage_hz,
            r.cfo_std_hz, r.cfo_std_sym_hz,
            r.iq_gain_alpha, r.iq_phase_deg_deg, r.rise_time_us,
            r.psd_centroid_hz, r.psd_pnr_db, r.bw_3db_hz, r.gated_len_us,
            r.cfo_two_stage_coarse_hz,
            r.cfo_joint_hz, r.iq_off_i, r.iq_off_q, r.iq_eps, r.iq_phi_deg, r.amp_a, r.fit_iters, r.fit_cost,
            r.cfo_exact_quick_hz, r.cfo_exact_ls_hz);
            if (with_signfixed) std::fprintf(f,",%.6f", (*signfixed)[i]);
            std::fprintf(f,"\n");
        }
        std::fclose(f);
    }
};

// ======== Robust utilities ========
static inline double median(std::vector<double> v){
    if (v.empty()) return 0.0;
    size_t n=v.size()/2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    double m=v[n];
    if (v.size()%2==0){
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
        m = 0.5*(m+v[n-1]);
    }
    return m;
}
static inline double mad(std::vector<double> v){
    if (v.empty()) return 0.0;
    double m = median(v);
    for (auto& x: v) x = std::fabs(x - m);
    return median(v) * 1.4826; // consistent for Gaussian
}

// Theil–Sen slope for y ~ a + b t
static double theil_sen_slope(const std::vector<double>& t, const std::vector<double>& y){
    std::vector<double> slopes;
    const size_t N=t.size();
    if (N<5) return 0.0;
    slopes.reserve(N*(N-1)/2);
    for (size_t i=0;i<N;i++){
        for (size_t j=i+1;j<N;j++){
            double dt=t[j]-t[i];
            if (std::fabs(dt) > 1e-9) slopes.push_back( (y[j]-y[i]) / dt );
        }
    }
    return median(slopes);
}

// 1-D k-means, k=2, few iterations
struct K2 { double c1, c2; size_t n1, n2; };
static K2 kmeans2_1d(const std::vector<double>& x){
    if (x.empty()) return {0,0,0,0};
    double m = median(x);
    std::vector<double> lo, hi; lo.reserve(x.size()); hi.reserve(x.size());
    for (double v: x) (v<m ? lo:hi).push_back(v);
    double c1 = lo.empty()? m : median(lo);
    double c2 = hi.empty()? m : median(hi);
    for (int it=0; it<20; ++it){
        double s1=0,s2=0; size_t n1=0,n2=0;
        for (double v: x){
            if (std::fabs(v-c1) <= std::fabs(v-c2)) { s1+=v; n1++; }
            else { s2+=v; n2++; }
        }
        double nc1 = (n1? s1/n1 : c1);
        double nc2 = (n2? s2/n2 : c2);
        if (std::fabs(nc1-c1)+std::fabs(nc2-c2) < 1e-6) { c1=nc1; c2=nc2; break; }
        c1=nc1; c2=nc2;
    }
    size_t n1=0,n2=0;
    for (double v: x){
        if (std::fabs(v-c1) <= std::fabs(v-c2)) n1++; else n2++;
    }
    return {c1,c2,n1,n2};
}

// Run detrend + polarity autofix; returns sign-fixed CFOs aligned with input
static std::vector<double> signfix_cfo_centroid(const std::vector<double>& ts,
                                                const std::vector<double>& cfo_centroid_hz,
                                                double* out_slope=nullptr,
                                                bool* did_flip=nullptr){
    const size_t N = cfo_centroid_hz.size();
    std::vector<double> t0(N), y(N);
    if (N==0) return y;
    double tmin = ts.empty()? 0.0 : ts.front();
    for (size_t i=0;i<N;i++){ t0[i] = ts[i]-tmin; y[i]=cfo_centroid_hz[i]; }
    double b = theil_sen_slope(t0,y);
    if (out_slope) *out_slope = b;
    std::vector<double> yd(N);
    for (size_t i=0;i<N;i++) yd[i] = y[i] - b*t0[i];
    auto km = kmeans2_1d(yd);
    bool opp_sign = (km.c1*km.c2) < 0.0;
    double r = std::min(std::fabs(km.c1), std::fabs(km.c2)) / std::max(1e-12,std::max(std::fabs(km.c1), std::fabs(km.c2)));
    double bal = (double)std::min(km.n1,km.n2) / std::max<size_t>(1, std::max(km.n1,km.n2));
    bool flip = opp_sign && (r >= 0.5) && (bal >= 0.30);
    std::vector<double> yfixed(N);
    if (flip){
        double m = median(yd);
        int keep_sign = (m>=0)? +1 : -1;
        for (size_t i=0;i<N;i++){
            int si = (yd[i]>=0)? +1 : -1;
            double z = yd[i];
            if (si != keep_sign) z = -z;
            yfixed[i] = z; // detrended + sign-fixed
        }
    } else {
        yfixed = yd;
    }
    if (did_flip) *did_flip = flip;
    return yfixed;
}

// ------------------ Args ------------------
enum class GateMode { NONE, ENERGY, STRUCT, MID };

struct Args {
    std::string file;
    std::string out = "out.pcap";
    int channel = 37;       // 37/38/39
    double fs = 4e6;        // input sample rate (complex baseband)
    int decim = 2;          // complex decimation (4->2 typical). 1 = no decimation
    size_t chunk = 1'000'000; // complex samples per read
    std::string dump_iq_dir = "";  // empty disables
    int prepad_us = 200;           // prepend this many microseconds of IQ before packet
    std::string features_out = "features.csv";

    // NEW: gating controls
    GateMode gate = GateMode::NONE;
    double gate_k = 4.0;         // energy threshold μ + k·σ
    int gate_pad_us = 8;         // ± pad around energy window
    int gate_mid_a_us = 12;      // start offset for MID
    int gate_mid_b_us = 80;      // end offset for MID
};

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
        else if (k=="--features-out") a.features_out = need("--features-out");
        else if (k=="--gate") {
            std::string v = need("--gate");
            if (v=="none") a.gate = GateMode::NONE;
            else if (v=="energy") a.gate = GateMode::ENERGY;
            else if (v=="struct") a.gate = GateMode::STRUCT;
            else if (v=="mid")    a.gate = GateMode::MID;
            else die("unknown --gate value (use none|energy|struct|mid)");
        } else if (k=="--gate-k")        a.gate_k = std::stod(need("--gate-k"));
        else if (k=="--gate-pad-us")     a.gate_pad_us = std::stoi(need("--gate-pad-us"));
        else if (k=="--gate-mid-a-us")   a.gate_mid_a_us = std::stoi(need("--gate-mid-a-us"));
        else if (k=="--gate-mid-b-us")   a.gate_mid_b_us = std::stoi(need("--gate-mid-b-us"));
        else if (k=="-h" || k=="--help"){
            std::cout <<
"Usage: iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out.pcap [--decim 2] [--chunk 1000000]\n"
"              [--dump-iq-dir iq_dir] [--prepad-us 200] [--features-out features.csv]\n"
"              [--gate none|energy|struct|mid] [--gate-k 4.0] [--gate-pad-us 8]\n"
"              [--gate-mid-a-us 12] [--gate-mid-b-us 80]\n";
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
static inline uint8_t bitrev8(uint8_t x){
    x = (uint8_t)((x>>4) | (x<<4));
    x = (uint8_t)(((x&0xCC)>>2) | ((x&0x33)<<2));
    x = (uint8_t)(((x&0xAA)>>1) | ((x&0x55)<<1));
    return x;
}
static void bitrev_buf(uint8_t* p, size_t n) {
    for (size_t i=0;i<n;i++) p[i] = bitrev8(p[i]);
}

// --------- Ring buffer for I/Q (for per-packet features) ----------
struct Ring {
    std::vector<float> buf;   // interleaved I,Q
    size_t cap = 0;           // capacity in complex samples
    uint64_t head_abs = 0;    // absolute complex index of NEXT write (one past last sample)
    uint64_t oldest_abs = 0;  // absolute complex index of OLDEST sample still retained

    void init(size_t complex_len) {
        cap = std::max<size_t>(complex_len, 4096);
        buf.assign(2*cap, 0.0f);
        head_abs = 0;
        oldest_abs = 0;
    }

    inline void push(float I, float Q) {
        // absolute complex index for this sample
        uint64_t p = head_abs;
        size_t slot = static_cast<size_t>(p % cap);
        buf[2*slot]   = I;
        buf[2*slot+1] = Q;
        head_abs++;

        // drop oldest if over capacity
        if (head_abs - oldest_abs > cap) {
            oldest_abs = head_abs - cap;
        }
    }

    // Return how many complex samples are currently retained
        inline uint64_t size_complex() const {
            return head_abs - oldest_abs; // absolute complex indices
        }
        
        // Copy window [a,b) in ABSOLUTE COMPLEX indices -> complex<float> vector
        bool copy_absolute_window(uint64_t a, uint64_t b,
                                  std::vector<std::complex<float>>& out) const
        {
            if (b <= a) return false;
            if (a < oldest_abs || b > head_abs) return false; // not in buffer
            const size_t N = static_cast<size_t>(b - a);
            out.resize(N);
            for (size_t k = 0; k < N; ++k) {
                uint64_t idx = a + k;                 // absolute complex index
                size_t slot  = static_cast<size_t>(idx % cap);
                float I = buf[2*slot];
                float Q = buf[2*slot + 1];
                out[k] = {I, Q};
            }
            return true;
        }
        
        // === Compatibility shim: copy the last n_floats from the ring into 'out' (interleaved I,Q) ===
        // Callers in your code always request an EVEN number of floats (2 * complex_count).
        bool copy_tail(size_t n_floats, std::vector<float>& out) const
        {
            // available floats in the ring
            const uint64_t avail_cx = size_complex();
            const uint64_t avail_f  = 2 * avail_cx;
            if (avail_f == 0) { out.clear(); return false; }
        
            if (n_floats > avail_f) n_floats = static_cast<size_t>(avail_f);
            // enforce even (I,Q pairs)
            if (n_floats & 1) n_floats -= 1;
            if (n_floats == 0) { out.clear(); return false; }
        
            const uint64_t start_abs_cx = head_abs - (n_floats / 2); // absolute complex start
            out.resize(n_floats);
        
            // copy in time order
            for (size_t i = 0; i < n_floats/2; ++i) {
                uint64_t abs_cx = start_abs_cx + i;
                size_t slot = static_cast<size_t>(abs_cx % cap);
                out[2*i]     = buf[2*slot];
                out[2*i + 1] = buf[2*slot + 1];
            }
            return true;
        }
};

// ------------------ Feature computations ------------------
namespace feat {

using cf = std::complex<float>;
static inline cf mkc(float I, float Q){ return cf(I,Q); }

static inline void rm_dc_norm(std::vector<cf>& x){
    cf mean(0.f,0.f);
    for (auto &v: x) mean += v;
    if (!x.empty()) mean /= (float)x.size();
    double e=0.0;
    for (auto &v: x){ v -= mean; e += (double)std::norm(v); }
    e = std::sqrt(e / std::max<double>(1.0, (double)x.size()));
    if (e > 1e-12) for (auto &v: x){ v = (float)(1.0/e) * v; }
}

static inline std::vector<float> discr(const std::vector<cf>& x){
    std::vector<float> d;
    if (x.size()<2) return d;
    d.resize(x.size()-1);
    for (size_t i=1;i<x.size();++i){
        cf z = x[i]*std::conj(x[i-1]);
        d[i-1] = std::atan2(z.imag(), z.real());
    }
    return d;
}

static inline float median(std::vector<float> v){
    if (v.empty()) return 0.f;
    size_t n=v.size()/2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    float m = v[n];
    if (v.size()%2==0){
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
        m = 0.5f*(m+v[n-1]);
    }
    return m;
}

static inline float cfo_quick(const std::vector<cf>& x, double fs){
    if (x.size()<8) return 0.f;
    auto d = discr(x);
    double m = 0; for (float v: d) m += v; if (!d.empty()) m/=d.size();
    double cfo_mean = (fs/(2.0*M_PI))*m;
    // std::fprintf(stderr, "[CFO-DBG] mean discr CFO over x = %.1f Hz (N=%zu)\n", cfo_mean, d.size());
    // std::fprintf(stderr, "[DEBUG][CFO quick] using %zu samples (%zu discr steps)\n",
    //              x.size(), d.size());
    // std::fprintf(stderr, "[CFO-DBG] quick used=%zu discr=%zu\n",
    //          x.size(), x.size()? x.size()-1 : 0);
    float med = median(d);
    if (std::fabs(med) > 2.5f){ // near wrap -> LS
        // LS on phase
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
        // simple linear fit (slope)
        double Sx=0,Sy=0,Sxx=0,Sxy=0; size_t N=ph.size();
        for (size_t i=0;i<N;++i){ Sx+=t[i]; Sy+=ph[i]; Sxx+=t[i]*t[i]; Sxy+=t[i]*ph[i]; }
        double slope = (N*Sxy - Sx*Sy)/std::max(1e-18, (N*Sxx - Sx*Sx));
        return (float)(slope/(2*M_PI));
    }
    return (float)((fs/(2*M_PI)) * med);
}

static inline float stddev(const std::vector<float>& a){
    if (a.size()<2) return 0.f;
    double m=0; for (auto v:a) m+=v; m/=a.size();
    double v=0; for (auto x:a){ double d=x-m; v+=d*d; }
    v/= (a.size()-1);
    return (float)std::sqrt(v);
}

static inline float cfo_std_all(const std::vector<cf>& x, double fs){
    if (x.size()<2) return NAN;
    auto d = discr(x);
    std::vector<float> cfo(d.size());
    for (size_t i=0;i<d.size();++i) cfo[i] = (float)((fs/(2*M_PI))*d[i]);
    return stddev(cfo);
}

static inline int sps_int(double fs){ return std::max(2, (int)std::lround(fs/1e6)); }

static inline float cfo_std_symbol_avg(const std::vector<cf>& x, double fs){
    int sps = sps_int(fs);
    if ((int)x.size() < sps+2) return NAN;
    auto d = discr(x);
    // boxcar average per symbol then downsample
    std::vector<float> ph_avg;
    for (size_t i=0;i+ (size_t)sps <= d.size(); i+= (size_t)sps){
        double m=0; for (int k=0;k<sps;++k) m += d[i+k];
        ph_avg.push_back((float)(m/sps));
    }
    for (auto &v: ph_avg) v = (float)((fs/(2*M_PI))*v);
    return stddev(ph_avg);
}

// Simple PSD with Hann; returns f (Hz), S (power) — naive DFT is OK at 4k points
static inline void psd_hann(const std::vector<cf>& x, double fs, std::vector<double>& f, std::vector<double>& S){
    size_t L = std::min<size_t>(x.size(), 4096);
    f.clear(); S.clear();
    if (L < 32){ f.push_back(0.0); S.push_back(0.0); return; }
    std::vector<cf> X(L);
    std::vector<double> w(L);
    for (size_t i=0;i<L;++i) w[i]=0.5*(1.0-std::cos(2*M_PI*i/(L-1)));
    // naive DFT on windowed x
    std::vector<cf> xw(L);
    for (size_t i=0;i<L;++i) xw[i] = (float)w[i]*x[i];
    X.assign(L,cf(0,0));
    for (size_t k=0;k<L;++k){
        cf acc(0,0);
        for (size_t n=0;n<L;++n){
            double ang = -2*M_PI*(double)k*(double)n/(double)L;
            acc += xw[n]*cf(std::cos(ang), std::sin(ang));
        }
        X[k]=acc;
    }
    S.resize(L);
    double w2=0; for (auto v:w) w2 += v*v;
    for (size_t i=0;i<L;++i) S[i] = std::norm(X[i]) / std::max(1e-18, w2);
    // fftshift
    std::vector<double> S2(L), f2(L);
    for (size_t i=0;i<L;++i){
        size_t j = (i + L/2) % L;
        S2[i] = S[j];
        double freq = ((double)i - (double)L/2)/ (double)L * fs;
        f2[i] = freq;
    }
    S.swap(S2); f.swap(f2);
}

static inline void spectral_stats(const std::vector<cf>& x, double fs,
                                  double& centroid, double& pnr_db, double& bw_3db){
    std::vector<double> f,S;
    psd_hann(x, fs, f, S);
    double sumS=0, sumfS=0, med=0;
    if (S.empty()){ centroid=0; pnr_db=0; bw_3db=0; return; }
    for (auto v:S){ sumS += v; }
    for (size_t i=0;i<S.size();++i) sumfS += f[i]*S[i];
    centroid = (sumS>0)? (sumfS/sumS):0.0;
    // PNR: peak versus median
    std::vector<double> Sc = S;
    std::nth_element(Sc.begin(), Sc.begin()+Sc.size()/2, Sc.end());
    med = Sc[Sc.size()/2];
    double peak = *std::max_element(S.begin(), S.end());
    pnr_db = 10.0*std::log10( std::max(peak,1e-18) / std::max(med,1e-18) );
    // 3 dB bandwidth around peak
    double thr = peak * std::pow(10.0, -3.0/10.0);
    size_t i0=0, i1=S.size()-1;
    for (size_t i=0;i<S.size();++i){ if (S[i]>=thr){ i0=i; break; } }
    for (size_t i=S.size(); i-->0; ){ if (S[i]>=thr){ i1=i; break; } }
    bw_3db = (i1>i0)? (f[i1]-f[i0]) : 0.0;
}

static inline double rise_time_us(const std::vector<cf>& x, double fs, int tail=200){
    if (x.empty()) return 0.0;
    std::vector<double> env(x.size());
    for (size_t i=0;i<x.size();++i) env[i] = std::abs(x[i]);
    double steady=0.0;
    if ((int)x.size()>tail){ for (int i=(int)x.size()-tail;i<(int)x.size();++i) steady += env[i]; steady/=tail; }
    else { for (auto v:env) steady += v; steady/= std::max<size_t>(1, env.size()); }
    if (steady <= 0) return 0.0;
    size_t n10=0,n90=0;
    for (size_t i=0;i<env.size();++i){ if (env[i] >= 0.1*steady) { n10=i; break; } }
    for (size_t i=0;i<env.size();++i){ if (env[i] >= 0.9*steady) { n90=i; break; } }
    return (n90>n10)? ( (double)(n90-n10)*1e6/fs ) : 0.0;
}

static inline void iq_imbalance(const std::vector<cf>& x, double& alpha, double& phi_deg){
    if (x.empty()){ alpha=1.0; phi_deg=0.0; return; }
    double mII=0, mQQ=0, mIQ=0;
    for (auto v:x){ mII += (double)v.real()*v.real(); mQQ += (double)v.imag()*v.imag(); mIQ += (double)v.real()*v.imag(); }
    mII/=x.size(); mQQ/=x.size(); mIQ/=x.size();
    alpha = std::sqrt( std::max(mII,1e-16) / std::max(mQQ,1e-16) );
    phi_deg = 0.5 * std::atan2(2*mIQ, (mII - mQQ + 1e-16)) * 180.0/M_PI;
}

// FFT centroid over settled early window
static inline float cfo_centroid(const std::vector<cf>& x, double fs, double f_lim=120e3, double settle_us=8){
    if (x.size()<32) return 0.f;
    size_t n_settle = (size_t)std::llround(settle_us*1e-6*fs);
    size_t a = std::min(n_settle, x.size());
    std::vector<cf> xw(x.begin()+a, x.end());
    if (xw.size()<32) xw = x; // fallback
    // PSD
    std::vector<double> f,S;
    psd_hann(xw, fs, f, S);
    double sum=0, sumf=0;
    for (size_t i=0;i<f.size();++i){
        if (std::fabs(f[i])<=f_lim){ sum += S[i]; sumf += f[i]*S[i]; }
    }
    return (sum>0)? (float)(sumf/sum) : 0.f;
}

static inline float cfo_ls_window(const std::vector<cf>& x, double fs){
    if (x.size()<4) return 0.f;
    std::vector<double> ph(x.size());
    ph[0] = std::arg(x[0]);
    for (size_t i=1;i<x.size();++i){
        double a = std::arg(x[i]);
        double b = std::arg(x[i-1]);
        double dp = a-b;
        if (dp >  M_PI) a -= 2*M_PI;
        if (dp < -M_PI) a += 2*M_PI;
        ph[i] = ph[i-1] + (a-b);
    }
    std::vector<double> t(x.size());
    for (size_t i=0;i<t.size();++i) t[i]=(double)i/fs;
    double Sx=0,Sy=0,Sxx=0,Sxy=0; size_t N=ph.size();
    for (size_t i=0;i<N;++i){ Sx+=t[i]; Sy+=ph[i]; Sxx+=t[i]*t[i]; Sxy+=t[i]*ph[i]; }
    double slope = (N*Sxy - Sx*Sy)/std::max(1e-18, (N*Sxx - Sx*Sx));
    return (float)(slope/(2*M_PI));
}

// 2-stage CFO estimate: centroid coarse + fine on derotated + LS on stable window
static inline float cfo_two_stage(const std::vector<cf>& x, double fs, float& coarse){
    float f0 = cfo_centroid(x, fs, 200e3, 8.0);
    std::vector<cf> x0(x.size());
    for (size_t n=0;n<x.size();++n){
        float ang = (float)(-2*M_PI*f0*(double)n/fs);
        x0[n] = x[n]*cf(std::cos(ang), std::sin(ang));
    }
    float f1 = cfo_centroid(x0, fs, 120e3, 8.0);
    std::vector<cf> x1(x.size());
    for (size_t n=0;n<x.size();++n){
        float ang = (float)(-2*M_PI*f1*(double)n/fs);
        x1[n] = x0[n]*cf(std::cos(ang), std::sin(ang));
    }
    coarse = f0+f1;
    // simple LS on middle/stable part (skip first few microseconds)
    size_t n_settle = (size_t)std::llround(8e-6*fs);
    size_t a = std::min(n_settle, x1.size());
    size_t b = x1.size();
    // std::fprintf(stderr, "[DEBUG][CFO 2stage] coarse=%.1f Hz, total=%zu, LS_window=[%zu..%zu) len=%zu\n",
                //  (double)coarse, x.size(), a, b, (b>a? b-a:0));
    if (b>a+std::max<size_t>(40,(size_t)(120*fs/1e6))){
        std::vector<cf> seg(x1.begin()+a, x1.begin()+b);
        return (float)(coarse + cfo_ls_window(seg, fs));
    } else {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

} // namespace feat

// ------------------ JOINT CFO+IQ ESTIMATOR ------------------
namespace joint {

// Model parameters
struct Params {
    double fo_hz;     // CFO
    double phi0;      // carrier phase (rad)
    double A;         // amplitude
    double eps;       // amplitude imbalance (epsilon)
    double phi;       // IQ phase imbalance (rad)
    double I0;        // I offset
    double Q0;        // Q offset
};

// Gaussian pulse for GFSK (BT=0.5 default), discrete taps
static inline std::vector<double> gaussian_taps(double BT, int sps, int span_sym=6){
    // normalized Gaussian: h(t) ~ exp(- (pi * BT * t)^2 / ln(2))
    const int L = span_sym * sps;
    std::vector<double> h(2*L+1);
    const double a = std::sqrt(2.0 * std::log(2.0)) / (M_PI * BT); // time scaling
    for (int n=-L;n<=L;++n){
        double t = (double)n / (double)sps;
        double g = std::exp( - (t*t) / (2*a*a) );
        h[n+L] = g;
    }
    // normalize to unit area (sum to sps)
    double sum=0; for (double v:h) sum+=v;
    if (sum>0) for (double& v:h) v = v * (double)sps / sum;
    return h;
}

// Recover a +/-1 symbol stream from discriminator (sign of phase steps)
static inline std::vector<double> recover_symbols(const std::vector<feat::cf>& x, int sps){
    auto d = feat::discr(x); // length N-1
    if (d.empty()) return std::vector<double>(x.size(), 0.0);
    // Average over each symbol (boxcar) to reduce noise
    const size_t Ns = d.size() / std::max(1, sps);
    std::vector<double> sym(Ns);
    for (size_t i=0;i<Ns;i++){
        double m=0;
        for (int k=0;k<sps;k++){
            size_t idx = i*(size_t)sps + (size_t)k;
            if (idx<d.size()) m += d[idx];
        }
        m /= (double)sps;
        sym[i] = (m>=0 ? +1.0 : -1.0);
    }
    // Upsample back to samples with ZOH per sample
    std::vector<double> m(x.size(), 0.0);
    for (size_t i=0;i<m.size();++i){
        size_t si = std::min((size_t)(i / (size_t)sps), Ns?Ns-1:0);
        m[i] = (Ns? sym[si] : 0.0);
    }
    return m;
}

// Build a CLEAN GFSK baseband from recovered +/-1 symbols using Gaussian filter (BT=0.5, h=0.5)
static inline void synth_gfsk(const std::vector<double>& m_pm1, int sps, double h,
                              double BT, std::vector<feat::cf>& y0)
{
    const size_t N = m_pm1.size();
    if (N==0){ y0.clear(); return; }
    // Convolve with Gaussian taps (frequency shaping)
    auto g = gaussian_taps(BT, sps, 6);
    const int L = (int)g.size();
    std::vector<double> fdev(N, 0.0);
    for (size_t n=0;n<N;n++){
        double acc=0;
        // fast FIR with guards
        int n0 = (int)n - (L/2);
        for (int k=0;k<L;k++){
            int idx = n0 + k;
            if ((unsigned)idx < (unsigned)N){
                acc += g[k] * m_pm1[(size_t)idx];
            }
        }
        fdev[n] = acc;
    }
    // Integrate to phase: phase[n] = 2π * h * ∫ fdev / sps
    std::vector<double> phase(N, 0.0);
    double acc=0;
    const double scale = 2.0 * M_PI * h / (double)sps;
    for (size_t n=0;n<N;n++){
        acc += fdev[n];
        phase[n] = scale * acc;
    }
    y0.resize(N);
    for (size_t n=0;n<N;n++){
        y0[n] = feat::cf((float)std::cos(phase[n]), (float)std::sin(phase[n]));
    }
}

// Apply CFO + IQ imbalance + DC offset + amplitude to a clean baseband
static inline void apply_impairments(std::vector<feat::cf>& y, double fs,
                                     const Params& p)
{
    const size_t N = y.size();
    for (size_t n=0;n<N;n++){
        // CFO rotation
        double ang = p.phi0 + 2.0*M_PI*p.fo_hz * ((double)n/fs);
        double ca = std::cos(ang), sa = std::sin(ang);
        float Ii = y[n].real(), Qq = y[n].imag();
        // IQ imbalance (amplitude & phase), small-signal model:
        // (1 - eps/2) * cos(θ - φ/2) + j (1 + eps/2) * sin(θ + φ/2)
        double ci = std::cos(-p.phi*0.5), si = std::sin(-p.phi*0.5);
        double cq = std::cos(+p.phi*0.5), sq = std::sin(+p.phi*0.5);
        double I_bal = (1.0 - 0.5*p.eps) * (Ii*ci - Qq*si);
        double Q_bal = (1.0 + 0.5*p.eps) * (Ii*sq + Qq*cq);
        // carrier rotate
        double Irot = I_bal*ca - Q_bal*sa;
        double Qrot = I_bal*sa + Q_bal*ca;
        // amplitude + DC
        double I = p.A*Irot + p.I0;
        double Q = p.A*Qrot + p.Q0;
        y[n] = feat::cf((float)I, (float)Q);
    }
}

// L2 cost & (optionally) residual normalization
static inline double cost_L2(const std::vector<feat::cf>& y_hat,
                             const std::vector<feat::cf>& y)
{
    const size_t N = std::min(y_hat.size(), y.size());
    if (N==0) return 0.0;
    double c=0.0;
    for (size_t n=0;n<N;n++){
        double di = (double)y_hat[n].real() - (double)y[n].real();
        double dq = (double)y_hat[n].imag() - (double)y[n].imag();
        c += di*di + dq*dq;
    }
    return c / (double)N;
}

// Numerical gradient (finite differences)
static inline Params grad_numeric(const std::vector<double>& m_pm1, int sps, double h, double BT,
                                  double fs, const std::vector<feat::cf>& y,
                                  const Params& p, const Params& step)
{
    auto eval = [&](const Params& pp)->double{
        std::vector<feat::cf> y0;
        synth_gfsk(m_pm1, sps, h, BT, y0);
        apply_impairments(y0, fs, pp);
        return cost_L2(y0, y);
    };
    const double c0 = eval(p);
    Params g{};
    auto inc = [&](Params q, double& gout, double Params::* field, double d)->void{
        Params qp = q; qp.*field += d;
        double c1 = eval(qp);
        gout = (c1 - c0) / d;
    };
    double dummy;
    inc(p, g.fo_hz,   &Params::fo_hz, step.fo_hz);
    inc(p, g.phi0,    &Params::phi0,  step.phi0);
    inc(p, g.A,       &Params::A,     step.A);
    inc(p, g.eps,     &Params::eps,   step.eps);
    inc(p, g.phi,     &Params::phi,   step.phi);
    inc(p, g.I0,      &Params::I0,    step.I0);
    inc(p, g.Q0,      &Params::Q0,    step.Q0);
    (void)dummy;
    return g;
}

// Simple Nesterov AGD on parameter vector with backoff if cost increases
static inline void nesterov_fit(const std::vector<double>& m_pm1, int sps,
                                double fs, const std::vector<feat::cf>& y,
                                Params& p, int& iters, double& final_cost,
                                double BT=0.5, double h=0.5,
                                int max_iters=35, double lr=0.2, double mu=0.85)
{
    // Steps for finite diff
    Params step{ 5.0, 1e-2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3 };
    // Momentum buffer
    Params v{0,0,0,0,0,0,0};
    auto add = [](Params a, const Params& b, double s)->Params{
        a.fo_hz += s*b.fo_hz; a.phi0 += s*b.phi0; a.A += s*b.A;
        a.eps += s*b.eps; a.phi += s*b.phi; a.I0 += s*b.I0; a.Q0 += s*b.Q0;
        return a;
    };

    auto eval_cost = [&](const Params& pp)->double{
        std::vector<feat::cf> y0;
        synth_gfsk(m_pm1, sps, h, BT, y0);
        apply_impairments(y0, fs, pp);
        return cost_L2(y0, y);
    };

    double best_cost = eval_cost(p);
    Params best = p;
    for (int it=0; it<max_iters; ++it){
        // lookahead
        Params yk = add(p, v, mu);
        // gradient at lookahead
        Params g = grad_numeric(m_pm1, sps, h, BT, fs, y, yk, step);
        // update momentum
        v = add(v, g, -lr);
        // apply
        Params p_new = add(yk, v, 1.0);
        // Small regularizations to keep params sane
        if (p_new.A <= 0) p_new.A = 1e-3;
        if (std::fabs(p_new.eps) > 0.5) p_new.eps = (p_new.eps > 0 ? 0.5 : -0.5);
        // evaluate
        double c_new = eval_cost(p_new);
        if (c_new <= best_cost){
            p = p_new;
            best_cost = c_new;
            best = p_new;
            // slight LR growth
            lr *= 1.05;
        } else {
            // backoff
            v = add(v, g, +lr); // undo last
            lr *= 0.5;
        }
        if (lr < 1e-4) break;
        iters = it+1;
    }
    p = best;
    final_cost = best_cost;
}

// Initialization: CFO via quick estimate; offsets via mean; imbalance via covariance
static inline joint::Params init_from_signal(const std::vector<feat::cf>& x, double fs){
    joint::Params p{};
    // CFO init
    double f0 = (double)feat::cfo_quick(x, fs);
    p.fo_hz = f0;
    p.phi0  = 0.0;
    // de-rotate to estimate offsets/imbalance roughly
    std::vector<feat::cf> z(x.size());
    for (size_t n=0;n<x.size();++n){
        double ang = -2.0*M_PI*f0*((double)n/fs);
        float ca = (float)std::cos(ang), sa=(float)std::sin(ang);
        float I = x[n].real()*ca - x[n].imag()*sa;
        float Q = x[n].real()*sa + x[n].imag()*ca;
        z[n] = feat::cf(I,Q);
    }
    // DC offsets
    double mi=0,mq=0;
    for (auto& v:z){ mi+=v.real(); mq+=v.imag(); }
    mi/=std::max<size_t>(1,z.size()); mq/=std::max<size_t>(1,z.size());
    p.I0 = mi; p.Q0 = mq;
    // Amplitude guess
    double rms=0; for (auto& v:z){ double I=v.real()-mi, Q=v.imag()-mq; rms+= I*I + Q*Q; }
    rms = std::sqrt(rms/std::max<size_t>(1,z.size()));
    p.A = (rms>1e-12? 1.0 : 0.5);
    // Simple imbalance init from covariance (small-signal)
    double sII=0,sQQ=0,sIQ=0;
    for (auto& v:z){ double I=v.real()-mi, Q=v.imag()-mq; sII+=I*I; sQQ+=Q*Q; sIQ+=I*Q; }
    sII/=std::max<size_t>(1,z.size()); sQQ/=std::max<size_t>(1,z.size()); sIQ/=std::max<size_t>(1,z.size());
    double alpha = std::sqrt( std::max(sII,1e-16) / std::max(sQQ,1e-16) );
    p.eps = (alpha-1.0); // approx
    p.phi = 0.5 * std::atan2(2*sIQ, (sII - sQQ + 1e-16)); // radians
    return p;
}

} // namespace joint

// ------------------ Features CSV writer ------------------
struct FeatureCSV {
    std::FILE* f=nullptr;
    explicit FeatureCSV(const std::string& path){
        f = std::fopen(path.c_str(), "w");
        if (!f) throw std::runtime_error("cannot open features csv: " + path);
        std::fprintf(f,
            "pkt_idx,pcap_ts,rf_channel,pdu_type,adv_addr,access_address,"
            "cfo_quick_hz,cfo_centroid_hz,cfo_two_stage_hz,cfo_std_hz,cfo_std_sym_hz,"
            "iq_gain_alpha,iq_phase_deg_deg,rise_time_us,psd_centroid_hz,psd_pnr_db,bw_3db_hz,gated_len_us,"
            "cfo_two_stage_coarse_hz,"
            "cfo_joint_hz,iq_off_i,iq_off_q,iq_eps,iq_phi_deg,amp_a,fit_iters,fit_cost\n");
        std::fflush(f);
    }
    void row(const FeatureRow& r){
        std::fprintf(f,
            "%zu,%.6f,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6g\n",
            r.pkt_idx, r.pcap_ts, r.rf_channel, r.pdu_type,
            r.adv_addr.c_str(), r.access_address.c_str(),
            r.cfo_quick_hz, r.cfo_centroid_hz, r.cfo_two_stage_hz, r.cfo_std_hz, r.cfo_std_sym_hz,
            r.iq_gain_alpha, r.iq_phase_deg_deg, r.rise_time_us, r.psd_centroid_hz, r.psd_pnr_db,
            r.bw_3db_hz, r.gated_len_us,
            r.cfo_two_stage_coarse_hz,
            r.cfo_joint_hz, r.iq_off_i, r.iq_off_q, r.iq_eps, r.iq_phi_deg, r.amp_a, r.fit_iters, r.fit_cost);
        std::fflush(f);
    }
    ~FeatureCSV(){ if (f) std::fclose(f); }
};

// --------- Packet glue: capture PDUs, write PCAP & features ----------
namespace blehelpers {
static inline std::string to_adv_addr(const uint8_t* hdr_payload, int pdu_type){
    char s[32];
    if (pdu_type==0 || pdu_type==2 || pdu_type==6 || pdu_type==4){
        std::snprintf(s, sizeof(s), "%02x:%02x:%02x:%02x:%02x:%02x",
            hdr_payload[5], hdr_payload[4], hdr_payload[3], hdr_payload[2], hdr_payload[1], hdr_payload[0]);
        return std::string(s);
    } else if (pdu_type==3 || pdu_type==5){
        std::snprintf(s, sizeof(s), "%02x:%02x:%02x:%02x:%02x:%02x",
            hdr_payload[11], hdr_payload[10], hdr_payload[9], hdr_payload[8], hdr_payload[7], hdr_payload[6]);
        return std::string(s);
    }
    return "";
}
static inline std::string aa_hex_be(const uint8_t* aa_le){
    char s[9];
    std::snprintf(s,sizeof(s), "%02X%02X%02X%02X", aa_le[3], aa_le[2], aa_le[1], aa_le[0]);
    return std::string(s);
}
} // namespace blehelpers

// ---------- Gating helpers (operate on x that includes prepad) ----------
static bool find_energy_window(const std::vector<feat::cf>& x, double fs,
                               size_t prepad_samps, double K, size_t pad_samps,
                               size_t& a, size_t& b)
{
    if (x.empty()) return false;

    // Use only the prepad to estimate noise floor
    size_t n0 = std::min(prepad_samps, x.size());
    if (n0 == 0) n0 = std::min<size_t>(200, x.size());

    double mu=0, s2=0, amax=0;
    for (size_t i=0;i<x.size();++i) {
        double a0 = std::abs(x[i]);
        if (i < n0) { mu += a0; s2 += a0*a0; }
        if (a0 > amax) amax = a0;
    }
    mu/=std::max<size_t>(1,n0); s2/=std::max<size_t>(1,n0);
    double sd = std::sqrt(std::max(0.0, s2 - mu*mu));

    // Robust threshold: if sd≈0 (all zeros), use a fraction of the global peak
    double T = mu + K*sd;
    if (sd < 1e-9 || T <= 0.0) T = 0.15 * amax;  // 15% of peak as a sane floor

    a = 0; b = x.size();
    bool got_a=false, got_b=false;
    for (size_t i=0;i<x.size();++i){ if (std::abs(x[i])>=T){ a=i; got_a=true; break; } }
    for (size_t i=x.size(); i-->0; ){ if (std::abs(x[i])>=T){ b=i+1; got_b=true; break; } }
    if (!(got_a && got_b) || b<=a) return false;

    if (pad_samps){
        a = (a>pad_samps)? a-pad_samps : 0;
        b = std::min(b+pad_samps, x.size());
    }
    return (b>a);
}

static void apply_gate_energy(std::vector<feat::cf>& x, double fs, size_t prepad_samps, double K, int pad_us){
    size_t pad = (size_t)std::llround(std::max(0, pad_us) * 1e-6 * fs);
    size_t a=0,b=0;
    if (find_energy_window(x, fs, prepad_samps, K, pad, a, b) && b>a && (b-a)>=32){
        x = std::vector<feat::cf>(x.begin()+a, x.begin()+b);
    }
}

static void apply_gate_struct(std::vector<feat::cf>& x, double fs, size_t prepad_samps){
    size_t a0=0,b0=0;
    if (!find_energy_window(x, fs, prepad_samps, 4.0, 0, a0, b0)) return;
    size_t off = (size_t)std::llround(8e-6*fs);
    size_t span = (size_t)std::llround(56e-6*fs);
    size_t a = std::min(a0 + off, x.size());
    size_t b = std::min(a + span, x.size());
    if (b>a && (b-a)>=32) x = std::vector<feat::cf>(x.begin()+a, x.begin()+b);
}

static void apply_gate_mid(std::vector<feat::cf>& x, double fs, size_t prepad_samps, int a_us, int b_us){
    size_t a0=0,b0=0;
    if (!find_energy_window(x, fs, prepad_samps, 4.0, 0, a0, b0)) return;
    if (b_us < a_us) std::swap(b_us, a_us);
    size_t a = std::min(a0 + (size_t)std::llround(std::max(0,a_us)*1e-6*fs), x.size());
    size_t b = std::min(a0 + (size_t)std::llround(std::max(0,b_us)*1e-6*fs), x.size());
    if (b>a && (b-a)>=32) x = std::vector<feat::cf>(x.begin()+a, x.begin()+b);
}

struct DumpCtx {
    bool enabled=false;
    bool test_mode = false;   // set true to inject the header IQ
    uint64_t ring_abs_head = 0; // complex-sample counter, incremented by producer
    std::string dir;
    int sps=2;
    double fs_eff=2e6;
    int prepad_us=200;
    Ring* ring=nullptr;
    size_t pkt_idx=0;

    FeatureCSV* featcsv=nullptr; // live writer
    FeatureRows* feats_all=nullptr; // collector for posthoc sign-fix
    // gating args snapshot
    GateMode gate = GateMode::NONE;
    double gate_k = 4.0;
    int gate_pad_us = 8;
    int gate_mid_a_us = 12;
    int gate_mid_b_us = 80;
};

// // --------- Packet glue: capture PDUs, write PCAP & features ----------
// static void attach_packet_handler(BLESDR& b, pcap::Writer& w, int rf_channel, DumpCtx& dctx) {
//     using feat::cf;
//     b.callback = [&](lell_packet pkt){
//         // --- Build the DLT 256 pseudo-header
//         pcap::le_phdr ph{};
//         ph.rf_channel = static_cast<uint8_t>(rf_channel);
//         ph.signal_power = 127;
//         ph.noise_power  = 127;
//         ph.access_address_offenses = 0;
//         ph.ref_access_address = 0x8E89BED6u;
//         ph.flags = pcap::LE_FLAG_DEWHITENED | pcap::LE_FLAG_REF_AA_VALID;

//         // --- Packet bytes to follow: AA(4) + header+payload+CRC
//         const uint8_t* bytes_aa  = pkt.symbols;
//         const uint8_t* bytes_pdu = pkt.symbols + 4;
//         size_t pdu_len = static_cast<size_t>(pkt.length) + 5;
//         size_t frame_len = sizeof(ph) + 4 + pdu_len;

//         std::vector<uint8_t> frame(frame_len);
//         std::memcpy(frame.data(), &ph, sizeof(ph));
//         std::memcpy(frame.data()+sizeof(ph), bytes_aa, 4);
//         std::memcpy(frame.data()+sizeof(ph)+4, bytes_pdu, pdu_len);

//         // --- PCAP write (capture ts for CSV)
//         double ts = w.write_pkt(frame.data(), frame.size());

//         // --- Extract minimal metadata for CSV
//         int pdu_type = (pdu_len>=2) ? (bytes_pdu[0] & 0x0F) : -1;
//         int payload_len = (pdu_len>=2) ? (bytes_pdu[1] & 0x3F) : 0;
//         std::string adv_addr = "";
//         if ((pdu_type==0 || pdu_type==2 || pdu_type==6 || pdu_type==4) && payload_len>=6){
//             adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
//         } else if ((pdu_type==3 || pdu_type==5) && payload_len>=12){
//             adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
//         }
//         char sAA[9];
//         // std::snprintf(sAA,sizeof(sAA), "%02X%02X%02X%02X", bytes_aa[3], bytes_aa[2], bytes_aa[1], bytes_aa[0]);
//         std::string aa_be = std::string(sAA);

//         // --- Feature window: pull IQ from ring around this packet
//         size_t bits = 8 + 32 + 16 + 8 * (size_t)pkt.length + 24;
//         size_t sps  = (size_t)std::max(2, dctx.sps);
//         size_t exact_samps = (pkt.sample_end > pkt.sample_start)
//                   ? (size_t)(pkt.sample_end - pkt.sample_start) : 0;
//         // size_t samps_needed = std::max((size_t)64, bits * sps);
//         size_t samps_needed = std::max({ (size_t)64, bits * sps, exact_samps });
//         size_t prepad_samps = (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6);
//         // size_t take_cplx    = prepad_samps + samps_needed;
//         // Cap by ring capacity to avoid over-asking
//         size_t take_cplx    = std::min(prepad_samps + samps_needed, dctx.ring->cap);

//         std::vector<float> rawIQ;
//         dctx.ring->copy_tail(2*take_cplx, rawIQ); // floats (I,Q)
//         // Convert to complex
//         std::vector<cf> x;
//         x.reserve(rawIQ.size()/2);
//         for (size_t i=0;i+1<rawIQ.size(); i+=2) x.emplace_back(rawIQ[i], rawIQ[i+1]);

//         // --- Normalize (remove DC/RMS)
//         // feat::rm_dc_norm(x);

//         // --- Apply gating (optional)
//         switch (dctx.gate){
//             case GateMode::ENERGY:
//                 apply_gate_energy(x, dctx.fs_eff, prepad_samps, dctx.gate_k, dctx.gate_pad_us);
//                 break;
//             case GateMode::STRUCT:
//                 apply_gate_struct(x, dctx.fs_eff, prepad_samps);
//                 break;
//             case GateMode::MID:
//                 apply_gate_mid(x, dctx.fs_eff, prepad_samps, dctx.gate_mid_a_us, dctx.gate_mid_b_us);
//                 break;
//             case GateMode::NONE:
//             default: break;
//         }

//         double cfo_exact_quick = 0;
//         double cfo_exact_ls    = 0;

//         // --- after you have 'std::vector<feat::cf> x;' and after applying gating ---
//         // A: basic sizes we believe we used
//         const double fs = dctx.fs_eff;
//         const int sps_i = feat::sps_int(fs);
//         // size_t xN = x.size();


//         // --- CFO on the exact packet indices only (last exact_samps in x)
//         if (exact_samps > 0) {
//             size_t use = std::min(exact_samps, x.size());
//             if (use >= 8) { // need a few samples to be meaningful
//                 std::vector<feat::cf> x_exact(x.end() - use, x.end());
//                 cfo_exact_quick = feat::cfo_quick(x_exact, dctx.fs_eff);
//                 cfo_exact_ls    = feat::cfo_quick(x_exact, dctx.fs_eff);
//             }
//         }

//         // --- Sanity: check exact_samps vs expected BLE packet length ---
//         // std::fprintf(stderr,
//         //     "[CFO-DBG] pkt=%zu ch=%d | pkt.len=%d bytes | bits=%zu | sps=%d | exact_samps=%zu | start=%llu end=%llu | xN=%zu\n",
//         //     dctx.pkt_idx,
//         //     rf_channel,
//         //     pkt.length,
//         //     (size_t)(8 + 32 + 16 + 8 * (size_t)pkt.length + 24),
//         //     dctx.sps,
//         //     (size_t)exact_samps,
//         //     (unsigned long long)pkt.sample_start,
//         //     (unsigned long long)pkt.sample_end,
//         //     x.size());

//         // --- After gating, before computing CFOs
//         int sps_dbg = feat::sps_int(dctx.fs_eff);

//         // --- Compute classical features
//         double fcent=0, pnr_db=0, bw3=0;
//         feat::spectral_stats(x, dctx.fs_eff, fcent, pnr_db, bw3);
//         double gated_len_us = (double)x.size()*1e6/dctx.fs_eff;
//         double alpha=1.0, phi_deg=0.0;
//         feat::iq_imbalance(x, alpha, phi_deg);
//         double rt_us = feat::rise_time_us(x, dctx.fs_eff);

//         double cfo_q = feat::cfo_quick(x, dctx.fs_eff);
//         double cfo_c = feat::cfo_centroid(x, dctx.fs_eff, 120e3, 8.0);
//         float coarse=std::numeric_limits<float>::quiet_NaN();
//         double cfo_two = feat::cfo_two_stage(x, dctx.fs_eff, coarse);
//         double cfo_std_all = feat::cfo_std_all(x, dctx.fs_eff);
//         double cfo_std_sym = feat::cfo_std_symbol_avg(x, dctx.fs_eff);

//         // --- JOINT CFO + IQ estimation over full packet window
//         // 1) recover +/-1 symbol stream from discriminator
//         // sps_i = feat::sps_int(dctx.fs_eff);
//         auto m_pm1 = joint::recover_symbols(x, sps_i);
//         // 2) init params from signal
//         joint::Params p0 = joint::init_from_signal(x, dctx.fs_eff);
//         // 3) run Nesterov
//         joint::Params p = p0;
//         int iters=0; double J=0.0;
//         joint::nesterov_fit(m_pm1, sps_i, dctx.fs_eff, x, p, iters, J, /*BT*/0.5, /*h*/0.5, /*maxI*/35, /*lr*/0.2, /*mu*/0.85);

//         FeatureRow row{
//             dctx.pkt_idx, ts, rf_channel, pdu_type, adv_addr, aa_be,
//             cfo_q, cfo_c, cfo_two, cfo_std_all, cfo_std_sym,
//             alpha, phi_deg, rt_us, fcent, pnr_db, bw3, gated_len_us, (double)coarse,
//             p.fo_hz, p.I0, p.Q0, p.eps, p.phi*180.0/M_PI, p.A, iters, J,
//             cfo_exact_quick, cfo_exact_ls
//         };

//         // live CSV
//         if (dctx.featcsv) dctx.featcsv->row(row);
//         // collect for posthoc sign-fix
//         if (dctx.feats_all) dctx.feats_all->push(row);

//         dctx.pkt_idx++;
//     };
// }

// static void attach_packet_handler(BLESDR& b, pcap::Writer& w, int rf_channel, DumpCtx& dctx) {
//     using feat::cf;
//     b.callback = [&](lell_packet pkt){
//         // --- Build the DLT 256 pseudo-header
//         pcap::le_phdr ph{};
//         ph.rf_channel = static_cast<uint8_t>(rf_channel);
//         ph.signal_power = 127;
//         ph.noise_power  = 127;
//         ph.access_address_offenses = 0;
//         ph.ref_access_address = 0x8E89BED6u;
//         ph.flags = pcap::LE_FLAG_DEWHITENED | pcap::LE_FLAG_REF_AA_VALID;

//         // --- Packet bytes to follow: AA(4) + header+payload+CRC
//         const uint8_t* bytes_aa  = pkt.symbols;
//         const uint8_t* bytes_pdu = pkt.symbols + 4;
//         size_t pdu_len = static_cast<size_t>(pkt.length) + 5;
//         size_t frame_len = sizeof(ph) + 4 + pdu_len;

//         std::vector<uint8_t> frame(frame_len);
//         std::memcpy(frame.data(), &ph, sizeof(ph));
//         std::memcpy(frame.data()+sizeof(ph), bytes_aa, 4);
//         std::memcpy(frame.data()+sizeof(ph)+4, bytes_pdu, pdu_len);

//         // --- PCAP write (capture ts for CSV)
//         double ts = w.write_pkt(frame.data(), frame.size());

//         // --- Minimal metadata for CSV
//         int pdu_type = (pdu_len>=2) ? (bytes_pdu[0] & 0x0F) : -1;
//         int payload_len = (pdu_len>=2) ? (bytes_pdu[1] & 0x3F) : 0;
//         std::string adv_addr = "";
//         if ((pdu_type==0 || pdu_type==2 || pdu_type==6 || pdu_type==4) && payload_len>=6){
//             adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
//         } else if ((pdu_type==3 || pdu_type==5) && payload_len>=12){
//             adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
//         }
//         char sAA[9]; sAA[0] = '\0';
//         std::string aa_be = std::string(sAA);

//         // --- Feature window: either inject test IQ or pull from ring
//         size_t bits = 8 + 32 + 16 + 8 * (size_t)pkt.length + 24;
//         size_t sps  = (size_t)std::max(2, dctx.sps);

//         // default: exact samps from sample_start/end (live path)
//         size_t exact_samps = (pkt.sample_end > pkt.sample_start)
//                   ? (size_t)(pkt.sample_end - pkt.sample_start) : 0;

//         std::vector<cf> x;       // complex IQ window
//         size_t xN_report = 0;    // for debug prints

//         if (dctx.test_mode) {
//             // ===== Test mode: use pre-packaged IQ with ~ +50..60 kHz CFO =====
//             const size_t N = ble_test_iq::N_COMPLEX;      // 664
//             const size_t N_exact = ble_test_iq::N_EXACT;  // 464 (tail)
//             // build complex vector from interleaved pairs
//             x = testload::make_vec_from_pairs(ble_test_iq::IQ_PAIRS, N);
//             // force expected fs for all feature functions
//             dctx.fs_eff = ble_test_iq::FS_HZ;             // 2e6
//             dctx.sps    = (int)ble_test_iq::SPS_INT;      // 2
//             // make “exact” span hit the tail of this vector
//             exact_samps = N_exact;
//             xN_report   = x.size();
//             if (x.size() > ble_test_iq::N_EXACT) {
//                 // Keep just the last exact_samps samples (packet tail)
//                 x.erase(x.begin(), x.end() - ble_test_iq::N_EXACT);
//             }
//             // No prepad in test mode
//             dctx.prepad_us = 0;

//             // Remove DC & normalize power so spectral/centroid aren’t biased
//             // feat::rm_dc_norm(x);
//         } else {
//             // ===== Live mode: pull from ring =====
//             size_t samps_needed = std::max({ (size_t)64, bits * sps, exact_samps });
//             size_t prepad_samps = (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6);
//             size_t take_cplx    = std::min(prepad_samps + samps_needed, dctx.ring->cap);

//             std::vector<float> rawIQ;
//             dctx.ring->copy_tail(2*take_cplx, rawIQ); // floats (I,Q)

//             x.reserve(rawIQ.size()/2);
//             for (size_t i=0;i+1<rawIQ.size(); i+=2) x.emplace_back(rawIQ[i], rawIQ[i+1]);
//             xN_report = x.size();
//         }

//         // --- Normalize if you like
//         // feat::rm_dc_norm(x);

//         // --- Apply gating (optional)
//         switch (dctx.gate){
//             case GateMode::ENERGY:
//                 apply_gate_energy(x, dctx.fs_eff,
//                                   (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6),
//                                   dctx.gate_k, dctx.gate_pad_us);
//                 break;
//             case GateMode::STRUCT:
//                 apply_gate_struct(x, dctx.fs_eff,
//                                   (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6));
//                 break;
//             case GateMode::MID:
//                 apply_gate_mid(x, dctx.fs_eff,
//                                (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6),
//                                dctx.gate_mid_a_us, dctx.gate_mid_b_us);
//                 break;
//             case GateMode::NONE:
//             default: break;
//         }

//         double cfo_exact_quick = 0.0;
//         double cfo_exact_ls    = 0.0;

//         // --- CFO on the exact packet indices only (last exact_samps in x)
//         if (exact_samps > 0) {
//             size_t use = std::min(exact_samps, x.size());
//             if (use >= 8) {
//                 std::vector<feat::cf> x_exact(x.end() - use, x.end());
//                 // If you have separate exact estimators, call them here:
//                 // cfo_exact_quick = feat::cfo_exact_quick(x_exact, dctx.fs_eff);
//                 // cfo_exact_ls    = feat::cfo_exact_ls(x_exact, dctx.fs_eff);
//                 // Otherwise use the quick estimator as a smoke test:
//                 cfo_exact_quick = feat::cfo_quick(x_exact, dctx.fs_eff);
//                 cfo_exact_ls    = feat::cfo_quick(x_exact, dctx.fs_eff);
//             }
//         }

//         // --- Classical features over the (gated) window
//         double fcent=0, pnr_db=0, bw3=0;
//         feat::spectral_stats(x, dctx.fs_eff, fcent, pnr_db, bw3);
//         double gated_len_us = (double)x.size()*1e6/dctx.fs_eff;
//         double alpha=1.0, phi_deg=0.0;
//         feat::iq_imbalance(x, alpha, phi_deg);
//         double rt_us = feat::rise_time_us(x, dctx.fs_eff);

//         double cfo_q = feat::cfo_quick(x, dctx.fs_eff);
//         double cfo_c = feat::cfo_centroid(x, dctx.fs_eff, 120e3, 8.0f);
//         float coarse=std::numeric_limits<float>::quiet_NaN();
//         double cfo_two = feat::cfo_two_stage(x, dctx.fs_eff, coarse);
//         double cfo_std_all = feat::cfo_std_all(x, dctx.fs_eff);
//         double cfo_std_sym = feat::cfo_std_symbol_avg(x, dctx.fs_eff);

//         // --- JOINT CFO + IQ estimation
//         const int sps_i = feat::sps_int(dctx.fs_eff);
//         auto m_pm1 = joint::recover_symbols(x, sps_i);
//         joint::Params p0 = joint::init_from_signal(x, dctx.fs_eff);
//         joint::Params p = p0;
//         int iters=0; double J=0.0;
//         joint::nesterov_fit(m_pm1, sps_i, dctx.fs_eff, x, p, iters, J,
//                             /*BT*/0.5, /*h*/0.5, /*maxI*/35, /*lr*/0.2, /*mu*/0.85);

//         // double fo_hz;
//         // if constexpr (/* p.fo is radians per *sample* */ true) {
//         //     fo_hz = p.fo * (dctx.fs_eff / (2.0*M_PI));
//         // } else { // radians per *symbol* (common if the model is on the symbol grid)
//         //     const int sps_i = feat::sps_int(dctx.fs_eff);
//         //     fo_hz = p.fo * (dctx.fs_eff / (2.0*M_PI) / sps_i);
//         // }

//         // --- CSV row
//         FeatureRow row{
//             dctx.pkt_idx, ts, rf_channel, pdu_type, adv_addr, aa_be,
//             cfo_q, cfo_c, cfo_two, cfo_std_all, cfo_std_sym,
//             alpha, phi_deg, rt_us, fcent, pnr_db, bw3, gated_len_us, (double)coarse,
//             p.fo_hz, p.I0, p.Q0, p.eps, p.phi*180.0/M_PI, p.A, iters, J,
//             cfo_exact_quick, cfo_exact_ls
//         };

//         // live CSV
//         if (dctx.featcsv) dctx.featcsv->row(row);
//         if (dctx.feats_all) dctx.feats_all->push(row);

//         // Optional: debug print to confirm nonzero CFOs
//         // std::fprintf(stderr,
//         //   "[CFO-DBG] pkt=%zu ch=%d | len=%d | sps=%d | exact=%zu | xN=%zu | "
//         //   "cfo_q=%.1fHz cfo_c=%.1fHz cfo_two=%.1fHz cfo_exact_q=%.1fHz joint=%.1fHz\n",
//         //   dctx.pkt_idx, rf_channel, pkt.length, dctx.sps, exact_samps, xN_report,
//         //   cfo_q, cfo_c, cfo_two, cfo_exact_quick, p.fo_hz);

//         dctx.pkt_idx++;
//     };
// }

// ==============================
// iq2pcap.cpp  (attach handler)
// ==============================
// add if not already present at top of file
static void attach_packet_handler(BLESDR& b, pcap::Writer& w, int rf_channel, DumpCtx& dctx) {
    using feat::cf;

    // ---------- helpers (local to this TU) ----------
    auto vec_rms = [](const std::vector<cf>& x) -> double {
        if (x.empty()) return 0.0;
        long double acc = 0;
        for (const auto& z : x) acc += (long double)std::norm(z);
        return std::sqrt((double)(acc / x.size()));
    };

    // Fetch exactly [end_cx - want_cx, end_cx) in *complex* units via tail copy in floats
    auto slice_exact = [&](uint64_t end_cx, size_t want_cx, std::vector<cf>& out,
                           uint64_t& lo_cx, uint64_t& hi_cx, double& rms) {
        out.clear(); lo_cx = hi_cx = 0; rms = 0.0;
        if (!dctx.ring || dctx.ring->cap == 0 || want_cx == 0) return;

        const uint64_t head_cx = dctx.ring_abs_head;
        const uint64_t cap_cx  = (uint64_t)(dctx.ring->cap - 1);
        const uint64_t oldest_cx = (cap_cx && head_cx > cap_cx) ? (head_cx - cap_cx) : 0ull;

        uint64_t start_cx = (end_cx > want_cx) ? (end_cx - want_cx) : 0ull;
        if (start_cx < oldest_cx) start_cx = oldest_cx;
        if (end_cx   > head_cx  ) end_cx   = head_cx;

        const uint64_t dist_to_head_cx = (head_cx > end_cx) ? (head_cx - end_cx) : 0ull;
        uint64_t fetch_cx = (end_cx > start_cx) ? (end_cx - start_cx) : 0ull;
        fetch_cx += dist_to_head_cx;
        if (cap_cx && fetch_cx > cap_cx) fetch_cx = cap_cx;

        std::vector<float> iq;
        dctx.ring->copy_tail(2 * fetch_cx, iq);            // floats
        if (dist_to_head_cx) {
            const size_t drop = (size_t)(2 * dist_to_head_cx);
            if (iq.size() > drop) iq.erase(iq.end() - drop, iq.end());
        }
        const size_t need_f = (size_t)(2 * (end_cx - start_cx));
        if (iq.size() > need_f) iq.erase(iq.begin(), iq.end() - need_f);

        out.reserve(iq.size() / 2);
        for (size_t i = 0; i + 1 < iq.size(); i += 2) out.emplace_back(iq[i], iq[i + 1]);

        lo_cx = start_cx;
        hi_cx = end_cx;
        rms   = vec_rms(out);

        std::fprintf(stderr,
            "[DBG] exact window head=%llu oldest=%llu base=%llu -> range=[%llu,%llu) xN=%zu rms=%.5g\n",
            (unsigned long long)head_cx,
            (unsigned long long)oldest_cx,
            (unsigned long long)((want_cx>0)?(end_cx - want_cx):end_cx),
            (unsigned long long)lo_cx, (unsigned long long)hi_cx,
            out.size(), rms);
    };

    // Search ±R around end_cx for the max-energy window of size want_cx (all *complex* units)
    auto slice_snapped = [&](uint64_t end_cx, size_t want_cx, size_t R,
                             std::vector<cf>& out, uint64_t& lo_cx, uint64_t& hi_cx, double& rms) {
        out.clear(); lo_cx = hi_cx = 0; rms = 0.0;
        if (!dctx.ring || dctx.ring->cap == 0 || want_cx == 0) return;

        const uint64_t head_cx   = dctx.ring_abs_head;
        const uint64_t cap_cx    = (uint64_t)(dctx.ring->cap - 1);
        const uint64_t oldest_cx = (cap_cx && head_cx > cap_cx) ? (head_cx - cap_cx) : 0ull;

        // Desired search region in complex units
        uint64_t srch_lo = (end_cx > R) ? (end_cx - R) : 0ull;
        uint64_t srch_hi = end_cx + R;

        // Ensure we can place a full window anywhere in [srch_lo, srch_hi]
        if (srch_lo < oldest_cx) srch_lo = oldest_cx;
        if (srch_hi > head_cx)   srch_hi = head_cx;
        if (srch_hi <= srch_lo)  return;

        // Build a buffer that surely spans [srch_lo - want_cx, srch_hi] (in floats)
        const uint64_t want_f   = 2ull * (uint64_t)want_cx;
        const uint64_t srch_lo_f = 2ull * srch_lo;
        const uint64_t srch_hi_f = 2ull * srch_hi;
        const uint64_t head_f    = 2ull * head_cx;
        const uint64_t cap_f     = 2ull * cap_cx;
        const uint64_t oldest_f  = 2ull * oldest_cx;

        uint64_t buf_end_f = srch_hi_f;
        uint64_t buf_beg_f = (srch_lo_f >= want_f) ? (srch_lo_f - want_f) : oldest_f;
        if (buf_beg_f > buf_end_f) buf_beg_f = srch_lo_f; // defensive
        uint64_t need_f = (buf_end_f > buf_beg_f) ? (buf_end_f - buf_beg_f) : 0ull;

        const uint64_t dist_to_head_f = (head_f > buf_end_f) ? (head_f - buf_end_f) : 0ull;
        uint64_t fetch_f = need_f + dist_to_head_f;
        if (cap_f && fetch_f > cap_f) fetch_f = cap_f;

        std::vector<float> iq;
        dctx.ring->copy_tail(fetch_f, iq); // floats
        if (dist_to_head_f && iq.size() > dist_to_head_f)
            iq.erase(iq.end() - dist_to_head_f, iq.end());
        if (iq.size() > need_f) iq.erase(iq.begin(), iq.end() - need_f);

        // Sliding RMS over *complex* windows of size 'want_cx' whose end lies in [srch_lo, srch_hi]
        // Map absolute complex index k ↔ position in iq: pos_f = 2*(k - (buf_beg_f/2))
        const uint64_t buf_beg_cx = buf_beg_f / 2ull;
        const size_t   buf_cx     = iq.size() / 2;

        if (buf_cx < want_cx) return;

        // Precompute mag^2 array for sliding sum
        std::vector<double> m2(buf_cx);
        for (size_t i = 0, j = 0; i < buf_cx; ++i, j += 2) {
            const double re = iq[j], im = iq[j + 1];
            m2[i] = re * re + im * im;
        }

        // Candidate window ends: k_end ∈ [srch_lo, srch_hi]
        // Convert to buffer indices
        const int64_t k0 = (int64_t)(srch_lo - buf_beg_cx);
        const int64_t k1 = (int64_t)(srch_hi - buf_beg_cx);

        // Sliding sum
        double cur = 0.0;
        size_t best_lo = 0, best_hi = 0;
        double bestE = -1.0;

        auto push = [&](size_t idx){ cur += m2[idx]; };
        auto pop  = [&](size_t idx){ cur -= m2[idx]; };

        // Initialize at first possible window that ends at max(k0, want_cx)
        int64_t end_idx = std::max<int64_t>(k0, (int64_t)want_cx);
        if ((size_t)end_idx > buf_cx) end_idx = (int64_t)buf_cx;

        size_t win_lo = (size_t)(end_idx - (int64_t)want_cx);
        size_t win_hi = (size_t)end_idx;

        for (size_t i = win_lo; i < win_hi; ++i) push(i);
        bestE = cur; best_lo = win_lo; best_hi = win_hi;

        for (int64_t e = end_idx + 1; e <= k1 && (size_t)e <= buf_cx; ++e) {
            // slide by one complex
            pop((size_t)(e - 1 - (int64_t)want_cx));
            push((size_t)(e - 1));
            // prefer later on tie
            if (cur >= bestE) { bestE = cur; best_lo = (size_t)(e - (int64_t)want_cx); best_hi = (size_t)e; }
        }

        // Extract best window
        out.reserve(want_cx);
        for (size_t i = 0, j = 2 * best_lo; i < want_cx; ++i, j += 2)
            out.emplace_back(iq[j], iq[j + 1]);

        lo_cx = buf_beg_cx + best_lo;
        hi_cx = buf_beg_cx + best_hi;
        rms   = vec_rms(out);

        std::fprintf(stderr,
            "[DBG] snapped   head=%llu oldest=%llu range=[%llu,%llu) xN=%zu rms=%.5g (from search [%llu,%llu) complex)\n",
            (unsigned long long)dctx.ring_abs_head,
            (unsigned long long)((cap_cx && dctx.ring_abs_head > cap_cx)?(dctx.ring_abs_head - cap_cx):0ull),
            (unsigned long long)lo_cx, (unsigned long long)hi_cx,
            out.size(), rms,
            (unsigned long long)srch_lo, (unsigned long long)srch_hi);
    };

    // --------------------------------------------------

    b.callback = [&](lell_packet pkt){
        // --- DLT 256 pseudo-header
        pcap::le_phdr ph{};
        ph.rf_channel = static_cast<uint8_t>(rf_channel);
        ph.signal_power = 127;
        ph.noise_power  = 127;
        ph.access_address_offenses = 0;
        ph.ref_access_address = 0x8E89BED6u;
        ph.flags = pcap::LE_FLAG_DEWHITENED | pcap::LE_FLAG_REF_AA_VALID;

        // --- Packet bytes (AA + PDU)
        const uint8_t* bytes_aa  = pkt.symbols;
        const uint8_t* bytes_pdu = pkt.symbols + 4;
        const size_t   pdu_len   = static_cast<size_t>(pkt.length) + 5; // hdr(2)+payload+CRC(3)
        const size_t   frame_len = sizeof(ph) + 4 + pdu_len;

        std::vector<uint8_t> frame(frame_len);
        std::memcpy(frame.data(), &ph, sizeof(ph));
        std::memcpy(frame.data()+sizeof(ph),      bytes_aa,  4);
        std::memcpy(frame.data()+sizeof(ph)+4,    bytes_pdu, pdu_len);

        const double ts = w.write_pkt(frame.data(), frame.size());

        // --- Minimal CSV meta
        const int pdu_type    = (pdu_len >= 2) ? (bytes_pdu[0] & 0x0F) : -1;
        const int payload_len = (pdu_len >= 2) ? (bytes_pdu[1] & 0x3F) : 0;

        std::string adv_addr;
        if ((pdu_type==0 || pdu_type==2 || pdu_type==6 || pdu_type==4) && payload_len>=6){
            adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
        } else if ((pdu_type==3 || pdu_type==5) && payload_len>=12){
            adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
        }
        char sAA[9]; sAA[0] = '\0';
        std::string aa_be = std::string(sAA);

        // ---------- feature window sizing ----------
        const size_t bits = 8 + 32 + 16 + 8 * (size_t)pkt.length + 24; // preamble+AA+hdr+payload+CRC
        const size_t sps  = (size_t)std::max(2, dctx.sps);
        const size_t W_eval_cx = bits * sps;                            // ~464 at Fs=2e6, sps=2

        // guard vs cap
        const size_t cap_cx = (dctx.ring && dctx.ring->cap) ? (size_t)(dctx.ring->cap - 1) : W_eval_cx;
        const size_t W_eval = std::max((size_t)64, std::min(W_eval_cx, cap_cx));

        // ---------- exact vs. snapped pick ----------
        // exact from decoder’s stamps (complex)
        std::vector<cf> x_exact; uint64_t ex_lo=0, ex_hi=0; double rms_exact=0.0;
        slice_exact(pkt.sample_end, W_eval, x_exact, ex_lo, ex_hi, rms_exact);

        // wide snap search: allow several packet lengths
        const size_t R = std::min<size_t>(5 * W_eval, std::max(W_eval, cap_cx / 4));
        std::vector<cf> x_snap; uint64_t sn_lo=0, sn_hi=0; double rms_snap=0.0;
        slice_snapped(pkt.sample_end, W_eval, R, x_snap, sn_lo, sn_hi, rms_snap);

        std::fprintf(stderr,
            "[diag] pkt_idx=%zu ch=%d pdu=%d len=%d | bits=%zu sps=%zu expect=%zu | "
            "EXACT=[%llu,%llu) rms=%.4g | SNAPPED=[%llu,%llu) rms=%.4g | head=%llu cap=%zu\n",
            (size_t)dctx.pkt_idx, rf_channel, pdu_type, (int)pkt.length,
            bits, sps, W_eval,
            (unsigned long long)ex_lo, (unsigned long long)ex_hi, rms_exact,
            (unsigned long long)sn_lo, (unsigned long long)sn_hi, rms_snap,
            (unsigned long long)dctx.ring_abs_head, cap_cx + 1);

        // Choose window by RMS (with flat guard and slight bias to snap)
        const double FLAT = 0.12;
        const bool exact_ok = (rms_exact > FLAT) && (x_exact.size() == W_eval);
        const bool snap_ok  = (rms_snap  > FLAT) && (x_snap.size()  == W_eval);

        std::vector<cf> x; x.reserve(W_eval);
        if (snap_ok && (!exact_ok || rms_snap >= 1.02 * rms_exact)) {
            x.swap(x_snap);
            std::fprintf(stderr,
                "[diag] pick=SNAP exact=[%llu,%llu) rms=%.4g  snap=[%llu,%llu) rms=%.4g  (R=%zu)\n",
                (unsigned long long)ex_lo, (unsigned long long)ex_hi, rms_exact,
                (unsigned long long)sn_lo, (unsigned long long)sn_hi, rms_snap, R);
//             b.set_detect_window(sn_lo, sn_hi);
        } else {
            x.swap(x_exact);
            std::fprintf(stderr,
                "[diag] pick=EXACT exact=[%llu,%llu) rms=%.4g  snap=[%llu,%llu) rms=%.4g  (R=%zu)\n",
                (unsigned long long)ex_lo, (unsigned long long)ex_hi, rms_exact,
                (unsigned long long)sn_lo, (unsigned long long)sn_hi, rms_snap, R);
//             b.set_detect_window(ex_lo, ex_hi);
        }

        // Effective Fs and prepad (you already manage fs_eff externally)
        dctx.prepad_us = 0;
        const size_t prepad_samps_now = (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6);

        // --- Gating
        switch (dctx.gate){
            case GateMode::ENERGY:
                apply_gate_energy(x, dctx.fs_eff, prepad_samps_now, dctx.gate_k, dctx.gate_pad_us);
                break;
            case GateMode::STRUCT:
                apply_gate_struct(x, dctx.fs_eff, prepad_samps_now);
                break;
            case GateMode::MID:
                apply_gate_mid(x, dctx.fs_eff, prepad_samps_now, dctx.gate_mid_a_us, dctx.gate_mid_b_us);
                break;
            case GateMode::NONE:
            default: break;
        }

        // --- CFO on exact packet tail (quick + LS)
        double cfo_exact_quick = 0.0, cfo_exact_ls = 0.0;
        if (!x.empty()) {
            const size_t use = std::min(x.size(), W_eval);
            std::vector<cf> x_exact_tail(x.end() - use, x.end());
            cfo_exact_quick = feat::cfo_quick(x_exact_tail, dctx.fs_eff);
            cfo_exact_ls    = feat::cfo_quick(x_exact_tail, dctx.fs_eff); // keep same estimator name
        }

        // --- Classical features (over gated x)
        double fcent=0, pnr_db=0, bw3=0;
        feat::spectral_stats(x, dctx.fs_eff, fcent, pnr_db, bw3);
        const double gated_len_us = (double)x.size()*1e6/dctx.fs_eff;
        double alpha=1.0, phi_deg=0.0;
        feat::iq_imbalance(x, alpha, phi_deg);
        const double rt_us = feat::rise_time_us(x, dctx.fs_eff);

        const double cfo_q = feat::cfo_quick(x, dctx.fs_eff);
        const double cfo_c = feat::cfo_centroid(x, dctx.fs_eff, 120e3, 8.0f);
        float coarse = std::numeric_limits<float>::quiet_NaN();
        const double cfo_two = feat::cfo_two_stage(x, dctx.fs_eff, coarse);
        const double cfo_std_all = feat::cfo_std_all(x, dctx.fs_eff);
        const double cfo_std_sym = feat::cfo_std_symbol_avg(x, dctx.fs_eff);

        // --- JOINT CFO + IQ (gated)
        const int sps_i = feat::sps_int(dctx.fs_eff);
        auto m_pm1 = joint::recover_symbols(x, sps_i);
        joint::Params p0 = joint::init_from_signal(x, dctx.fs_eff);
        joint::Params p = p0;
        int iters=0; double J=0.0;
        joint::nesterov_fit(m_pm1, sps_i, dctx.fs_eff, x, p, iters, J,
                            /*BT*/0.5, /*h*/0.5, /*maxI*/35, /*lr*/0.2, /*mu*/0.85);

        // --- CSV row
        FeatureRow row{
            dctx.pkt_idx, ts, rf_channel, pdu_type, adv_addr, aa_be,
            cfo_q, cfo_c, cfo_two, cfo_std_all, cfo_std_sym,
            alpha, phi_deg, rt_us, fcent, pnr_db, bw3, gated_len_us, (double)coarse,
            p.fo_hz, p.I0, p.Q0, p.eps, p.phi*180.0/M_PI, p.A, iters, J,
            pkt.cfo_exact_quick_hz, pkt.cfo_exact_ls_hz
        };

        if (dctx.featcsv)   dctx.featcsv->row(row);
        if (dctx.feats_all) dctx.feats_all->push(row);

        dctx.pkt_idx++;
    };
}

int main(int argc, char** argv){
    auto args = parse(argc, argv);

    // Open capture (complex float32 interleaved)
    std::FILE* f = std::fopen(args.file.c_str(), "rb");
    if(!f) die(std::string("cannot open file: ") + args.file + " : " + std::strerror(errno));

    // Effective complex sample rate after decim & derived SPS
    const double fs_eff = args.fs / args.decim;
    const int sps = std::max(2, (int)std::lround(fs_eff / 1e6)); // BLE-1M ⇒ ~2 at 2 MS/s

    if (!args.dump_iq_dir.empty()) {
        std::string cmd = "mkdir -p '" + args.dump_iq_dir + "'";
        std::system(cmd.c_str());
    }

    pcap::Writer w(args.out);
    FeatureCSV featcsv(args.features_out);
    FeatureRows feats; // collector for sign-fix CSV

    std::vector<float> bufIQ(2*args.chunk);
    std::vector<float> workIQ;           // decimated interleaved complex floats
    Ring ring;                           // ring for feature windows
    ring.init((size_t)(fs_eff * 0.250)); // 50 ms ring

    BLESDR blesdr;

    DumpCtx dctx;
    dctx.enabled   = !args.dump_iq_dir.empty();
    dctx.dir       = args.dump_iq_dir;
    dctx.sps       = sps;
    dctx.fs_eff    = fs_eff;
    dctx.prepad_us = args.prepad_us;
    dctx.ring      = &ring;
    dctx.featcsv   = &featcsv;
    dctx.feats_all = &feats;
    dctx.gate      = args.gate;
    dctx.gate_k    = args.gate_k;
    dctx.gate_pad_us = args.gate_pad_us;
    dctx.gate_mid_a_us = args.gate_mid_a_us;
    dctx.gate_mid_b_us = args.gate_mid_b_us;

    // // init BLE decoder state to match our stream
    // blesdr.RB_init();              // allocate ringbuffer once
    // blesdr.srate = sps;            // samples per symbol (≈2 at 2 MS/s)
    // blesdr.chan  = (uint8_t)args.channel;  // whitening seed for header/payload
    // blesdr.skipSamples = 0;        // (optional) ensure we don't delay early packets

    blesdr.Configure(sps, (uint8_t)args.channel, /*skip=*/0);
    
    // 👉 ADD THIS RIGHT HERE 👇
    blesdr.set_iq_provider([&](uint64_t a, uint64_t b, std::vector<std::complex<float>>& out){
        // Provide IQ samples between absolute complex indices [a, b)
        return ring.copy_absolute_window(a, b, out);
    });

    // Attach the packet handler so every decoded packet writes PCAP + features CSV
    attach_packet_handler(blesdr, w, args.channel, dctx);

    // Feed chunks to the decoder
    size_t total_complex = 0, total_complex_fed = 0;

    for(;;){
        size_t nread = std::fread(bufIQ.data(), sizeof(float)*2, args.chunk, f);
        if (nread == 0) break;

        // Complex decimation (keeps I,Q interleaved)
        size_t n_cplx_out = decimate_cplx(bufIQ.data(), nread, args.decim, workIQ);

        // DC-remove + RMS normalize per component (pre-conditioning for BLESDR and ring)
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

        // // Push into ring, then feed BLESDR (expects interleaved floats), samples_len = #complex samples.
        // for (size_t i=0;i<n_cplx_out;i++) ring.push(workIQ[2*i], workIQ[2*i+1]);
        // blesdr.Receiver((size_t)args.channel, workIQ.data(), n_cplx_out);

        // // Push into ring, then update absolute head (complex samples), then feed BLESDR
        // for (size_t i=0;i<n_cplx_out;i++) ring.push(workIQ[2*i], workIQ[2*i+1]);

        // // IMPORTANT: keep this in complex-sample units *after* decimation
        // dctx.ring_abs_head += n_cplx_out;

        // // keep decoder’s cursor in lockstep with the ring
        // blesdr.set_abs_cursor(dctx.ring_abs_head);

        // --- BEFORE feeding this chunk ---
        // Snapshot the head *before* we append this chunk to the ring.
        // All absolute indices in this chunk start at 'head_prev'.
        const uint64_t head_prev = dctx.ring_abs_head;
        blesdr.set_abs_cursor(head_prev);

        // 1) Push into ring
        for (size_t i = 0; i < n_cplx_out; ++i)
            ring.push(workIQ[2*i], workIQ[2*i+1]);

        // 2) Advance absolute head in COMPLEX samples *after* the push
        dctx.ring_abs_head += n_cplx_out;

        // 3) Decode using this chunk (the decoder now knows absolute indices)
        blesdr.Receiver((size_t)args.channel, workIQ.data(), n_cplx_out);

        // blesdr.Receiver((size_t)args.channel, workIQ.data(), n_cplx_out);

        static bool once = false;
        if (!once) {
            fprintf(stderr, "[DBG] set_abs_cursor(%llu)\n",
                    (unsigned long long)dctx.ring_abs_head);
            once = true;
        }

        total_complex     += nread;
        total_complex_fed += n_cplx_out;
    }
    std::fclose(f);

    // -------- Posthoc: detrend + sign-fix column, write *_signfixed.csv --------
    feats.write_csv(args.features_out /*raw*/);

    std::vector<double> ts, cfoC; ts.reserve(feats.rows.size()); cfoC.reserve(feats.rows.size());
    for (const auto& r : feats.rows){ ts.push_back(r.pcap_ts); cfoC.push_back(r.cfo_centroid_hz); }
    double slope=0; bool flipped=false;
    auto cfo_fixed = signfix_cfo_centroid(ts, cfoC, &slope, &flipped);

    std::string out2 = args.features_out;
    if (out2.size()>=4 && out2.substr(out2.size()-4)==".csv") out2.insert(out2.size()-4, "_signfixed");
    else out2 += "_signfixed.csv";
    feats.write_csv(out2, true, &cfo_fixed);

    std::cerr << "[posthoc] detrend slope = " << slope << " Hz/s, flip=" << (flipped?"true":"false")
              << ", wrote: " << out2 << "\n";

    std::cerr << "Done. Complex read: " << total_complex
              << ", complex fed: " << total_complex_fed
              << ", packets (approx): " << dctx.pkt_idx
              << ", features CSV: " << args.features_out
              << ", PCAP: " << args.out
              << "\n";
    return 0;
}