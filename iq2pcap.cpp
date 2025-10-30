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
        "cfo_joint_hz,iq_off_i,iq_off_q,iq_eps,iq_phi_deg,amp_a,fit_iters,fit_cost");
        if (with_signfixed) std::fprintf(f,",cfo_centroid_hz_signfixed");
        std::fprintf(f,"\n");

        for (size_t i=0;i<rows.size();++i){
            const auto& r = rows[i];
            std::fprintf(f,
            "%zu,%.6f,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6g",
            r.pkt_idx, r.pcap_ts, r.rf_channel, r.pdu_type,
            r.adv_addr.c_str(), r.access_address.c_str(),
            r.cfo_quick_hz, r.cfo_centroid_hz, r.cfo_two_stage_hz,
            r.cfo_std_hz, r.cfo_std_sym_hz,
            r.iq_gain_alpha, r.iq_phase_deg_deg, r.rise_time_us,
            r.psd_centroid_hz, r.psd_pnr_db, r.bw_3db_hz, r.gated_len_us,
            r.cfo_two_stage_coarse_hz,
            r.cfo_joint_hz, r.iq_off_i, r.iq_off_q, r.iq_eps, r.iq_phi_deg, r.amp_a, r.fit_iters, r.fit_cost);
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
    size_t n0 = std::min(prepad_samps, x.size());
    if (n0 == 0) n0 = std::min<size_t>(200, x.size());
    double mu=0, s2=0;
    for (size_t i=0;i<n0;i++){ double a0 = std::abs(x[i]); mu+=a0; s2+=a0*a0; }
    mu/=std::max<size_t>(1,n0); s2/=std::max<size_t>(1,n0);
    double sd = std::sqrt(std::max(0.0, s2 - mu*mu));
    double T = mu + K*sd;
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

// --------- Packet glue: capture PDUs, write PCAP & features ----------
static void attach_packet_handler(BLESDR& b, pcap::Writer& w, int rf_channel, DumpCtx& dctx) {
    using feat::cf;
    b.callback = [&](lell_packet pkt){
        // --- Build the DLT 256 pseudo-header
        pcap::le_phdr ph{};
        ph.rf_channel = static_cast<uint8_t>(rf_channel);
        ph.signal_power = 127;
        ph.noise_power  = 127;
        ph.access_address_offenses = 0;
        ph.ref_access_address = 0x8E89BED6u;
        ph.flags = pcap::LE_FLAG_DEWHITENED | pcap::LE_FLAG_REF_AA_VALID;

        // --- Packet bytes to follow: AA(4) + header+payload+CRC
        const uint8_t* bytes_aa  = pkt.symbols;
        const uint8_t* bytes_pdu = pkt.symbols + 4;
        size_t pdu_len = static_cast<size_t>(pkt.length) + 5;
        size_t frame_len = sizeof(ph) + 4 + pdu_len;

        std::vector<uint8_t> frame(frame_len);
        std::memcpy(frame.data(), &ph, sizeof(ph));
        std::memcpy(frame.data()+sizeof(ph), bytes_aa, 4);
        std::memcpy(frame.data()+sizeof(ph)+4, bytes_pdu, pdu_len);

        // --- PCAP write (capture ts for CSV)
        double ts = w.write_pkt(frame.data(), frame.size());

        // --- Extract minimal metadata for CSV
        int pdu_type = (pdu_len>=2) ? (bytes_pdu[0] & 0x0F) : -1;
        int payload_len = (pdu_len>=2) ? (bytes_pdu[1] & 0x3F) : 0;
        std::string adv_addr = "";
        if ((pdu_type==0 || pdu_type==2 || pdu_type==6 || pdu_type==4) && payload_len>=6){
            adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
        } else if ((pdu_type==3 || pdu_type==5) && payload_len>=12){
            adv_addr = blehelpers::to_adv_addr(bytes_pdu+2, pdu_type);
        }
        char sAA[9];
        std::snprintf(sAA,sizeof(sAA), "%02X%02X%02X%02X", bytes_aa[3], bytes_aa[2], bytes_aa[1], bytes_aa[0]);
        std::string aa_be = std::string(sAA);

        // --- Feature window: pull IQ from ring around this packet
        size_t bits = 8 + 32 + 16 + 8 * (size_t)pkt.length + 24;
        size_t sps  = (size_t)std::max(2, dctx.sps);
        size_t samps_needed = std::max((size_t)64, bits * sps);
        size_t prepad_samps = (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6);
        size_t take_cplx    = prepad_samps + samps_needed;

        std::vector<float> rawIQ;
        dctx.ring->copy_tail(2*take_cplx, rawIQ); // floats (I,Q)
        // Convert to complex
        std::vector<cf> x;
        x.reserve(rawIQ.size()/2);
        for (size_t i=0;i+1<rawIQ.size(); i+=2) x.emplace_back(rawIQ[i], rawIQ[i+1]);

        // --- Normalize (remove DC/RMS)
        // feat::rm_dc_norm(x);

        // --- Apply gating (optional)
        switch (dctx.gate){
            case GateMode::ENERGY:
                apply_gate_energy(x, dctx.fs_eff, prepad_samps, dctx.gate_k, dctx.gate_pad_us);
                break;
            case GateMode::STRUCT:
                apply_gate_struct(x, dctx.fs_eff, prepad_samps);
                break;
            case GateMode::MID:
                apply_gate_mid(x, dctx.fs_eff, prepad_samps, dctx.gate_mid_a_us, dctx.gate_mid_b_us);
                break;
            case GateMode::NONE:
            default: break;
        }

        // Guard
        if (x.size() < 64){
            // write a sparse row and continue
            double fcent=0, pnr_db=0, bw3=0, alpha=1.0, phi_deg=0.0, rt_us=0.0;
            feat::spectral_stats(x, dctx.fs_eff, fcent, pnr_db, bw3);
            feat::iq_imbalance(x, alpha, phi_deg);
            double cfo_q = feat::cfo_quick(x, dctx.fs_eff);
            double cfo_c = feat::cfo_centroid(x, dctx.fs_eff, 120e3, 8.0);
            float coarse=std::numeric_limits<float>::quiet_NaN();
            double cfo_two = feat::cfo_two_stage(x, dctx.fs_eff, coarse);
            FeatureRow row{
              dctx.pkt_idx, ts, rf_channel, pdu_type, adv_addr, aa_be,
              cfo_q, cfo_c, cfo_two,
              feat::cfo_std_all(x, dctx.fs_eff), feat::cfo_std_symbol_avg(x, dctx.fs_eff),
              alpha, phi_deg, rt_us, fcent, pnr_db, bw3, (double)x.size()*1e6/dctx.fs_eff, (double)coarse,
              std::numeric_limits<double>::quiet_NaN(), 0,0,0,0,0, 0, 0.0
            };
            if (dctx.featcsv) dctx.featcsv->row(row);
            if (dctx.feats_all) dctx.feats_all->push(row);
            dctx.pkt_idx++;
            return;
        }

        // --- Compute classical features
        double fcent=0, pnr_db=0, bw3=0;
        feat::spectral_stats(x, dctx.fs_eff, fcent, pnr_db, bw3);
        double gated_len_us = (double)x.size()*1e6/dctx.fs_eff;
        double alpha=1.0, phi_deg=0.0;
        feat::iq_imbalance(x, alpha, phi_deg);
        double rt_us = feat::rise_time_us(x, dctx.fs_eff);

        double cfo_q = feat::cfo_quick(x, dctx.fs_eff);
        double cfo_c = feat::cfo_centroid(x, dctx.fs_eff, 120e3, 8.0);
        float coarse=std::numeric_limits<float>::quiet_NaN();
        double cfo_two = feat::cfo_two_stage(x, dctx.fs_eff, coarse);
        double cfo_std_all = feat::cfo_std_all(x, dctx.fs_eff);
        double cfo_std_sym = feat::cfo_std_symbol_avg(x, dctx.fs_eff);

        // --- JOINT CFO + IQ estimation over full packet window
        // 1) recover +/-1 symbol stream from discriminator
        int sps_i = feat::sps_int(dctx.fs_eff);
        auto m_pm1 = joint::recover_symbols(x, sps_i);
        // 2) init params from signal
        joint::Params p0 = joint::init_from_signal(x, dctx.fs_eff);
        // 3) run Nesterov
        joint::Params p = p0;
        int iters=0; double J=0.0;
        joint::nesterov_fit(m_pm1, sps_i, dctx.fs_eff, x, p, iters, J, /*BT*/0.5, /*h*/0.5, /*maxI*/35, /*lr*/0.2, /*mu*/0.85);

        FeatureRow row{
            dctx.pkt_idx, ts, rf_channel, pdu_type, adv_addr, aa_be,
            cfo_q, cfo_c, cfo_two, cfo_std_all, cfo_std_sym,
            alpha, phi_deg, rt_us, fcent, pnr_db, bw3, gated_len_us, (double)coarse,
            p.fo_hz, p.I0, p.Q0, p.eps, p.phi*180.0/M_PI, p.A, iters, J
        };

        // live CSV
        if (dctx.featcsv) dctx.featcsv->row(row);
        // collect for posthoc sign-fix
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
    ring.init((size_t)(fs_eff * 0.050)); // 50 ms ring

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

    // init BLE decoder state to match our stream
    blesdr.RB_init();              // allocate ringbuffer once
    blesdr.srate = sps;            // samples per symbol (≈2 at 2 MS/s)
    blesdr.chan  = (uint8_t)args.channel;  // whitening seed for header/payload
    blesdr.skipSamples = 0;        // (optional) ensure we don't delay early packets

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

        // Push into ring, then feed BLESDR (expects interleaved floats), samples_len = #complex samples.
        for (size_t i=0;i<n_cplx_out;i++) ring.push(workIQ[2*i], workIQ[2*i+1]);
        blesdr.Receiver((size_t)args.channel, workIQ.data(), n_cplx_out);

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

// // iq2pcap.cpp - Read complex-float I/Q, decode BLE with BLESDR, dump PCAP + inline features CSV
// // Adds per-packet feature extraction (CFO + IQ/PSD stats) directly from the ring buffer IQ.
// // NEW: gating options (--gate none|energy|struct|mid) to narrow CFO window,
// //      plus posthoc detrend+sign-fix CSV (features_signfixed.csv).
// //
// // Build: link with your existing BLESDR lib (lib/BLESDR*.cpp)
// // Usage:
// //   ./iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out_ch37.pcap [--decim 2]
// //             [--dump-iq-dir iq_dir] [--prepad-us 200] [--features-out features.csv]
// //             [--gate energy --gate-k 4.0 --gate-pad-us 8]
// //             [--gate struct]    // preamble+AA+header slice
// //             [--gate mid --gate-mid-a-us 12 --gate-mid-b-us 80]
// //
// // Notes on features:
// //   - Features are computed from a window of I/Q pulled from the trailing ring when a packet is decoded.
// //   - Base window size ~ prepad_us (before) + bits_estimated*sps (after). You can adjust prepad_us.
// //   - CFO is estimated with: quick (median discr / LS), centroid (FFT), two-stage (coarse+fine+LS).
// //   - Robust to discriminator polarity, preamble (0xAA vs 0x55), and small SPS/phase uncertainty.

// #include <cstdio>
// #include <cstdint>
// #include <cstdlib>
// #include <cstring>
// #include <cerrno>
// #include <string>
// #include <vector>
// #include <functional>
// #include <memory>
// #include <iostream>
// #include <fstream>
// #include <chrono>
// #include <cmath>
// #include <algorithm>
// #include <numeric>
// #include <complex>
// #include <limits>   // for quiet_NaN

// #include "lib/BLESDR.hpp"   // adjust include path if needed

// // ------------------ Simple PCAP writer ------------------
// namespace pcap {
// static constexpr uint32_t MAGIC   = 0xA1B2C3D4;
// static constexpr uint16_t VMAJOR  = 2;
// static constexpr uint16_t VMINOR  = 4;
// static constexpr uint32_t SNAPLEN = 0xFFFF;
// static constexpr uint32_t LINKTYPE_BLE_LL_WITH_PHDR = 256; // DLT 256

// #pragma pack(push, 1)
// struct le_phdr {
//     uint8_t  rf_channel;            // 0..39 (adv: 37/38/39)
//     int8_t   signal_power;          // dBm; valid iff flags & 0x0002
//     int8_t   noise_power;           // dBm; valid iff flags & 0x0004
//     uint8_t  access_address_offenses; // valid iff flags & 0x0020
//     uint32_t ref_access_address;    // valid iff flags & 0x0010 (LE)
//     uint16_t flags;                 // bitfield, see below
// };
// // Flag bits (subset used here)
// static constexpr uint16_t LE_FLAG_DEWHITENED      = 0x0001;
// static constexpr uint16_t LE_FLAG_SIGNAL_VALID    = 0x0002;
// static constexpr uint16_t LE_FLAG_NOISE_VALID     = 0x0004;
// static constexpr uint16_t LE_FLAG_REF_AA_VALID    = 0x0010;
// static constexpr uint16_t LE_FLAG_AA_OFFENSES_OK  = 0x0020;
// static constexpr uint16_t LE_FLAG_CRC_CHECKED     = 0x0400;
// static constexpr uint16_t LE_FLAG_CRC_VALID       = 0x0800;
// #pragma pack(pop)

// struct Writer {
//     std::FILE* f = nullptr;
//     explicit Writer(const std::string& path) {
//         f = std::fopen(path.c_str(), "wb");
//         if (!f) { throw std::runtime_error("fopen failed: " + path); }
//         // global header (native endian, classic pcap)
//         uint32_t magic = MAGIC;
//         uint16_t vmaj = VMAJOR, vmin = VMINOR;
//         uint32_t thiszone = 0, sigfigs = 0, snaplen = SNAPLEN, network = LINKTYPE_BLE_LL_WITH_PHDR;
//         std::fwrite(&magic,   4,1,f);
//         std::fwrite(&vmaj,    2,1,f);
//         std::fwrite(&vmin,    2,1,f);
//         std::fwrite(&thiszone,4,1,f);
//         std::fwrite(&sigfigs, 4,1,f);
//         std::fwrite(&snaplen, 4,1,f);
//         std::fwrite(&network, 4,1,f);
//     }
//     // returns the timestamp used (seconds, float)
//     double write_pkt(const uint8_t* data, size_t len, double ts_sec_f = -1.0) {
//         using clock = std::chrono::system_clock;
//         double now = ts_sec_f >= 0 ? ts_sec_f
//                                    : std::chrono::duration<double>(clock::now().time_since_epoch()).count();
//         uint32_t ts_sec  = static_cast<uint32_t>(now);
//         uint32_t ts_usec = static_cast<uint32_t>((now - ts_sec)*1e6 + 0.5);
//         uint32_t incl = static_cast<uint32_t>(len);
//         uint32_t orig = static_cast<uint32_t>(len);
//         std::fwrite(&ts_sec,  4,1,f);
//         std::fwrite(&ts_usec, 4,1,f);
//         std::fwrite(&incl,    4,1,f);
//         std::fwrite(&orig,    4,1,f);
//         if (len) std::fwrite(data, 1, len, f);
//         return now;
//     }
//     ~Writer(){ if(f) std::fclose(f); }
// };
// } // namespace pcap

// // ------------------ Helpers ------------------
// static void die(const std::string& s) { std::cerr << "error: " << s << "\n"; std::exit(1); }

// // ======== Add near the top ========
// struct FeatureRow {
//     size_t pkt_idx; double pcap_ts; int rf_channel; int pdu_type;
//     std::string adv_addr, access_address;
//     double cfo_quick_hz, cfo_centroid_hz, cfo_two_stage_hz;
//     double cfo_std_hz, cfo_std_sym_hz;
//     double iq_gain_alpha, iq_phase_deg_deg;
//     double rise_time_us, psd_centroid_hz, psd_pnr_db, bw_3db_hz, gated_len_us;
//     double cfo_two_stage_coarse_hz;
// };

// struct FeatureRows {
//     std::vector<FeatureRow> rows;
//     void push(const FeatureRow& r){ rows.push_back(r); }
//     void write_csv(const std::string& path, bool with_signfixed=false,
//                    const std::vector<double>* signfixed=nullptr) const {
//         std::FILE* f = std::fopen(path.c_str(), "w");
//         if (!f) throw std::runtime_error("cannot open csv for write: "+path);
//         std::fprintf(f,
//         "pkt_idx,pcap_ts,rf_channel,pdu_type,adv_addr,access_address,"
//         "cfo_quick_hz,cfo_centroid_hz,cfo_two_stage_hz,cfo_std_hz,cfo_std_sym_hz,"
//         "iq_gain_alpha,iq_phase_deg_deg,rise_time_us,psd_centroid_hz,psd_pnr_db,"
//         "bw_3db_hz,gated_len_us,cfo_two_stage_coarse_hz");
//         if (with_signfixed) std::fprintf(f,",cfo_centroid_hz_signfixed");
//         std::fprintf(f,"\n");

//         for (size_t i=0;i<rows.size();++i){
//             const auto& r = rows[i];
//             std::fprintf(f,
//             "%zu,%.6f,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
//             r.pkt_idx, r.pcap_ts, r.rf_channel, r.pdu_type,
//             r.adv_addr.c_str(), r.access_address.c_str(),
//             r.cfo_quick_hz, r.cfo_centroid_hz, r.cfo_two_stage_hz,
//             r.cfo_std_hz, r.cfo_std_sym_hz,
//             r.iq_gain_alpha, r.iq_phase_deg_deg, r.rise_time_us,
//             r.psd_centroid_hz, r.psd_pnr_db, r.bw_3db_hz, r.gated_len_us,
//             r.cfo_two_stage_coarse_hz);
//             if (with_signfixed) std::fprintf(f,",%.6f", (*signfixed)[i]);
//             std::fprintf(f,"\n");
//         }
//         std::fclose(f);
//     }
// };

// // ======== Robust utilities ========
// static inline double median(std::vector<double> v){
//     if (v.empty()) return 0.0;
//     size_t n=v.size()/2;
//     std::nth_element(v.begin(), v.begin()+n, v.end());
//     double m=v[n];
//     if (v.size()%2==0){
//         std::nth_element(v.begin(), v.begin()+n-1, v.end());
//         m = 0.5*(m+v[n-1]);
//     }
//     return m;
// }
// static inline double mad(std::vector<double> v){
//     if (v.empty()) return 0.0;
//     double m = median(v);
//     for (auto& x: v) x = std::fabs(x - m);
//     return median(v) * 1.4826; // consistent for Gaussian
// }

// // Theil–Sen slope for y ~ a + b t
// static double theil_sen_slope(const std::vector<double>& t, const std::vector<double>& y){
//     std::vector<double> slopes;
//     const size_t N=t.size();
//     if (N<5) return 0.0;
//     slopes.reserve(N*(N-1)/2);
//     for (size_t i=0;i<N;i++){
//         for (size_t j=i+1;j<N;j++){
//             double dt=t[j]-t[i];
//             if (std::fabs(dt) > 1e-9) slopes.push_back( (y[j]-y[i]) / dt );
//         }
//     }
//     return median(slopes);
// }

// // 1-D k-means, k=2, few iterations
// struct K2 { double c1, c2; size_t n1, n2; };
// static K2 kmeans2_1d(const std::vector<double>& x){
//     if (x.empty()) return {0,0,0,0};
//     double m = median(x);
//     std::vector<double> lo, hi; lo.reserve(x.size()); hi.reserve(x.size());
//     for (double v: x) (v<m ? lo:hi).push_back(v);
//     double c1 = lo.empty()? m : median(lo);
//     double c2 = hi.empty()? m : median(hi);
//     for (int it=0; it<20; ++it){
//         double s1=0,s2=0; size_t n1=0,n2=0;
//         for (double v: x){
//             if (std::fabs(v-c1) <= std::fabs(v-c2)) { s1+=v; n1++; }
//             else { s2+=v; n2++; }
//         }
//         double nc1 = (n1? s1/n1 : c1);
//         double nc2 = (n2? s2/n2 : c2);
//         if (std::fabs(nc1-c1)+std::fabs(nc2-c2) < 1e-6) { c1=nc1; c2=nc2; break; }
//         c1=nc1; c2=nc2;
//     }
//     size_t n1=0,n2=0;
//     for (double v: x){
//         if (std::fabs(v-c1) <= std::fabs(v-c2)) n1++; else n2++;
//     }
//     return {c1,c2,n1,n2};
// }

// // Run detrend + polarity autofix; returns sign-fixed CFOs aligned with input
// static std::vector<double> signfix_cfo_centroid(const std::vector<double>& ts,
//                                                 const std::vector<double>& cfo_centroid_hz,
//                                                 double* out_slope=nullptr,
//                                                 bool* did_flip=nullptr){
//     const size_t N = cfo_centroid_hz.size();
//     std::vector<double> t0(N), y(N);
//     if (N==0) return y;
//     double tmin = ts.empty()? 0.0 : ts.front();
//     for (size_t i=0;i<N;i++){ t0[i] = ts[i]-tmin; y[i]=cfo_centroid_hz[i]; }
//     double b = theil_sen_slope(t0,y);
//     if (out_slope) *out_slope = b;
//     std::vector<double> yd(N);
//     for (size_t i=0;i<N;i++) yd[i] = y[i] - b*t0[i];
//     auto km = kmeans2_1d(yd);
//     bool opp_sign = (km.c1*km.c2) < 0.0;
//     double r = std::min(std::fabs(km.c1), std::fabs(km.c2)) / std::max(1e-12,std::max(std::fabs(km.c1), std::fabs(km.c2)));
//     double bal = (double)std::min(km.n1,km.n2) / std::max<size_t>(1, std::max(km.n1,km.n2));
//     bool flip = opp_sign && (r >= 0.5) && (bal >= 0.30);
//     std::vector<double> yfixed(N);
//     if (flip){
//         double m = median(yd);
//         int keep_sign = (m>=0)? +1 : -1;
//         for (size_t i=0;i<N;i++){
//             int si = (yd[i]>=0)? +1 : -1;
//             double z = yd[i];
//             if (si != keep_sign) z = -z;
//             yfixed[i] = z; // detrended + sign-fixed
//         }
//     } else {
//         yfixed = yd;
//     }
//     if (did_flip) *did_flip = flip;
//     return yfixed;
// }

// // ------------------ Args ------------------
// enum class GateMode { NONE, ENERGY, STRUCT, MID };

// struct Args {
//     std::string file;
//     std::string out = "out.pcap";
//     int channel = 37;       // 37/38/39
//     double fs = 4e6;        // input sample rate (complex baseband)
//     int decim = 2;          // complex decimation (4->2 typical). 1 = no decimation
//     size_t chunk = 1'000'000; // complex samples per read
//     std::string dump_iq_dir = "";  // empty disables
//     int prepad_us = 200;           // prepend this many microseconds of IQ before packet
//     std::string features_out = "features.csv";

//     // NEW: gating controls
//     GateMode gate = GateMode::NONE;
//     double gate_k = 4.0;         // energy threshold μ + k·σ
//     int gate_pad_us = 8;         // ± pad around energy window
//     int gate_mid_a_us = 12;      // start offset for MID
//     int gate_mid_b_us = 80;      // end offset for MID
// };

// static Args parse(int argc, char** argv){
//     Args a;
//     for (int i=1;i<argc;i++){
//         std::string k = argv[i];
//         auto need = [&](const char* name)->const char*{
//             if (i+1>=argc) die(std::string("missing value after ")+name);
//             return argv[++i];
//         };
//         if (k=="--file")         a.file = need("--file");
//         else if (k=="--out")     a.out  = need("--out");
//         else if (k=="--fs")      a.fs   = std::stod(need("--fs"));
//         else if (k=="--channel") a.channel = std::stoi(need("--channel"));
//         else if (k=="--decim")   a.decim= std::stoi(need("--decim"));
//         else if (k=="--chunk")   a.chunk= static_cast<size_t>(std::stoll(need("--chunk")));
//         else if (k=="--dump-iq-dir") a.dump_iq_dir = need("--dump-iq-dir");
//         else if (k=="--prepad-us")   a.prepad_us   = std::stoi(need("--prepad-us"));
//         else if (k=="--features-out") a.features_out = need("--features-out");
//         else if (k=="--gate") {
//             std::string v = need("--gate");
//             if (v=="none") a.gate = GateMode::NONE;
//             else if (v=="energy") a.gate = GateMode::ENERGY;
//             else if (v=="struct") a.gate = GateMode::STRUCT;
//             else if (v=="mid")    a.gate = GateMode::MID;
//             else die("unknown --gate value (use none|energy|struct|mid)");
//         } else if (k=="--gate-k")        a.gate_k = std::stod(need("--gate-k"));
//         else if (k=="--gate-pad-us")     a.gate_pad_us = std::stoi(need("--gate-pad-us"));
//         else if (k=="--gate-mid-a-us")   a.gate_mid_a_us = std::stoi(need("--gate-mid-a-us"));
//         else if (k=="--gate-mid-b-us")   a.gate_mid_b_us = std::stoi(need("--gate-mid-b-us"));
//         else if (k=="-h" || k=="--help"){
//             std::cout <<
// "Usage: iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out.pcap [--decim 2] [--chunk 1000000]\n"
// "              [--dump-iq-dir iq_dir] [--prepad-us 200] [--features-out features.csv]\n"
// "              [--gate none|energy|struct|mid] [--gate-k 4.0] [--gate-pad-us 8]\n"
// "              [--gate-mid-a-us 12] [--gate-mid-b-us 80]\n";
//             std::exit(0);
//         }
//     }
//     if (a.file.empty()) die("please specify --file");
//     if (a.channel<37 || a.channel>39) die("channel must be 37, 38 or 39");
//     if (a.decim<1) a.decim = 1;
//     return a;
// }

// // Decimate interleaved complex float32 stream by N (keep every Nth complex sample)
// static size_t decimate_cplx(const float* iq, size_t n_cplx, int N, std::vector<float>& outIQ) {
//     if (N < 1) N = 1;
//     outIQ.clear();
//     outIQ.reserve(2 * (n_cplx / (size_t)N + 16));
//     for (size_t k = 0; k < n_cplx; k += (size_t)N) {
//         outIQ.push_back(iq[2*k]);     // I
//         outIQ.push_back(iq[2*k + 1]); // Q
//     }
//     return outIQ.size() / 2; // # of complex samples
// }

// // Optional: bit-reverse if your decoder returns LSB-first bits in each byte.
// static inline uint8_t bitrev8(uint8_t x){
//     x = (uint8_t)((x>>4) | (x<<4));
//     x = (uint8_t)(((x&0xCC)>>2) | ((x&0x33)<<2));
//     x = (uint8_t)(((x&0xAA)>>1) | ((x&0x55)<<1));
//     return x;
// }
// static void bitrev_buf(uint8_t* p, size_t n) {
//     for (size_t i=0;i<n;i++) p[i] = bitrev8(p[i]);
// }

// // --------- Ring buffer for I/Q (for per-packet features) ----------
// struct Ring {
//     std::vector<float> buf; // interleaved I,Q
//     size_t w = 0;           // write index in floats
//     size_t cap = 0;         // capacity in complex samples
//     void init(size_t complex_len) {
//         cap = std::max<size_t>(complex_len, 4096);
//         buf.assign(2*cap, 0.0f);
//         w = 0;
//     }
//     inline void push(float I, float Q) {
//         buf[w] = I;
//         size_t w2 = (w+1)%(2*cap);
//         buf[w2] = Q;
//         w = (w+2)%(2*cap);
//     }
//     // copy last 'take_floats' floats (I,Q interleaved) into out
//     void copy_tail(size_t take_floats, std::vector<float>& out) const {
//         size_t maxf = 2*cap;
//         if (take_floats > maxf) take_floats = maxf;
//         out.resize(take_floats);
//         size_t start = ( (w + maxf) - take_floats ) % maxf;
//         for (size_t i=0;i<take_floats;i++) out[i] = buf[(start + i)%maxf];
//     }
// };

// // ------------------ Feature computations ------------------
// namespace feat {

// using cf = std::complex<float>;
// static inline cf mkc(float I, float Q){ return cf(I,Q); }

// static inline void rm_dc_norm(std::vector<cf>& x){
//     cf mean(0.f,0.f);
//     for (auto &v: x) mean += v;
//     if (!x.empty()) mean /= (float)x.size();
//     double e=0.0;
//     for (auto &v: x){ v -= mean; e += (double)std::norm(v); }
//     e = std::sqrt(e / std::max<double>(1.0, (double)x.size()));
//     if (e > 1e-12) for (auto &v: x){ v = (float)(1.0/e) * v; }
// }

// static inline std::vector<float> discr(const std::vector<cf>& x){
//     std::vector<float> d;
//     if (x.size()<2) return d;
//     d.resize(x.size()-1);
//     for (size_t i=1;i<x.size();++i){
//         cf z = x[i]*std::conj(x[i-1]);
//         d[i-1] = std::atan2(z.imag(), z.real());
//     }
//     return d;
// }

// static inline float median(std::vector<float> v){
//     if (v.empty()) return 0.f;
//     size_t n=v.size()/2;
//     std::nth_element(v.begin(), v.begin()+n, v.end());
//     float m = v[n];
//     if (v.size()%2==0){
//         std::nth_element(v.begin(), v.begin()+n-1, v.end());
//         m = 0.5f*(m+v[n-1]);
//     }
//     return m;
// }

// static inline float cfo_quick(const std::vector<cf>& x, double fs){
//     if (x.size()<8) return 0.f;
//     auto d = discr(x);
//     float med = median(d);
//     if (std::fabs(med) > 2.5f){ // near wrap -> LS
//         // LS on phase
//         std::vector<double> t(x.size());
//         for (size_t i=0;i<t.size();++i) t[i]= (double)i/fs;
//         // unwrap
//         std::vector<double> ph(x.size());
//         ph[0] = std::arg(x[0]);
//         for (size_t i=1;i<x.size();++i){
//             double a = std::arg(x[i]);
//             double b = std::arg(x[i-1]);
//             double dp = a - b;
//             if (dp >  M_PI) a -= 2*M_PI;
//             if (dp < -M_PI) a += 2*M_PI;
//             ph[i] = ph[i-1] + (a - b);
//         }
//         // simple linear fit (slope)
//         double Sx=0,Sy=0,Sxx=0,Sxy=0; size_t N=ph.size();
//         for (size_t i=0;i<N;++i){ Sx+=t[i]; Sy+=ph[i]; Sxx+=t[i]*t[i]; Sxy+=t[i]*ph[i]; }
//         double slope = (N*Sxy - Sx*Sy)/std::max(1e-18, (N*Sxx - Sx*Sx));
//         return (float)(slope/(2*M_PI));
//     }
//     return (float)((fs/(2*M_PI)) * med);
// }

// static inline float stddev(const std::vector<float>& a){
//     if (a.size()<2) return 0.f;
//     double m=0; for (auto v:a) m+=v; m/=a.size();
//     double v=0; for (auto x:a){ double d=x-m; v+=d*d; }
//     v/= (a.size()-1);
//     return (float)std::sqrt(v);
// }

// static inline float cfo_std_all(const std::vector<cf>& x, double fs){
//     if (x.size()<2) return NAN;
//     auto d = discr(x);
//     std::vector<float> cfo(d.size());
//     for (size_t i=0;i<d.size();++i) cfo[i] = (float)((fs/(2*M_PI))*d[i]);
//     return stddev(cfo);
// }

// static inline int sps_int(double fs){ return std::max(2, (int)std::lround(fs/1e6)); }

// static inline float cfo_std_symbol_avg(const std::vector<cf>& x, double fs){
//     int sps = sps_int(fs);
//     if ((int)x.size() < sps+2) return NAN;
//     auto d = discr(x);
//     std::vector<float> ph_avg;
//     for (size_t i=0;i+ (size_t)sps <= d.size(); i+= (size_t)sps){
//         double m=0; for (int k=0;k<sps;++k) m += d[i+k];
//         ph_avg.push_back((float)(m/sps));
//     }
//     for (auto &v: ph_avg) v = (float)((fs/(2*M_PI))*v);
//     return stddev(ph_avg);
// }

// // Simple PSD with Hann; returns f (Hz), S (power) — naive DFT is OK at 4k points
// static inline void psd_hann(const std::vector<cf>& x, double fs, std::vector<double>& f, std::vector<double>& S){
//     size_t L = std::min<size_t>(x.size(), 4096);
//     f.clear(); S.clear();
//     if (L < 32){ f.push_back(0.0); S.push_back(0.0); return; }
//     std::vector<cf> X(L);
//     std::vector<double> w(L);
//     for (size_t i=0;i<L;++i) w[i]=0.5*(1.0-std::cos(2*M_PI*i/(L-1)));
//     // naive DFT on windowed x
//     std::vector<cf> xw(L);
//     for (size_t i=0;i<L;++i) xw[i] = (float)w[i]*x[i];
//     X.assign(L,cf(0,0));
//     for (size_t k=0;k<L;++k){
//         cf acc(0,0);
//         for (size_t n=0;n<L;++n){
//             double ang = -2*M_PI*(double)k*(double)n/(double)L;
//             acc += xw[n]*cf(std::cos(ang), std::sin(ang));
//         }
//         X[k]=acc;
//     }
//     S.resize(L);
//     double w2=0; for (auto v:w) w2 += v*v;
//     for (size_t i=0;i<L;++i) S[i] = std::norm(X[i]) / std::max(1e-18, w2);
//     // fftshift
//     std::vector<double> S2(L), f2(L);
//     for (size_t i=0;i<L;++i){
//         size_t j = (i + L/2) % L;
//         S2[i] = S[j];
//         double freq = ((double)i - (double)L/2)/ (double)L * fs;
//         f2[i] = freq;
//     }
//     S.swap(S2); f.swap(f2);
// }

// static inline void spectral_stats(const std::vector<cf>& x, double fs,
//                                   double& centroid, double& pnr_db, double& bw_3db){
//     std::vector<double> f,S;
//     psd_hann(x, fs, f, S);
//     double sumS=0, sumfS=0, med=0;
//     if (S.empty()){ centroid=0; pnr_db=0; bw_3db=0; return; }
//     for (auto v:S){ sumS += v; }
//     for (size_t i=0;i<S.size();++i) sumfS += f[i]*S[i];
//     centroid = (sumS>0)? (sumfS/sumS):0.0;
//     // PNR: peak versus median
//     std::vector<double> Sc = S;
//     std::nth_element(Sc.begin(), Sc.begin()+Sc.size()/2, Sc.end());
//     med = Sc[Sc.size()/2];
//     double peak = *std::max_element(S.begin(), S.end());
//     pnr_db = 10.0*std::log10( std::max(peak,1e-18) / std::max(med,1e-18) );
//     // 3 dB bandwidth around peak
//     double thr = peak * std::pow(10.0, -3.0/10.0);
//     size_t i0=0, i1=S.size()-1;
//     for (size_t i=0;i<S.size();++i){ if (S[i]>=thr){ i0=i; break; } }
//     for (size_t i=S.size(); i-->0; ){ if (S[i]>=thr){ i1=i; break; } }
//     bw_3db = (i1>i0)? (f[i1]-f[i0]) : 0.0;
// }

// static inline double rise_time_us(const std::vector<cf>& x, double fs, int tail=200){
//     if (x.empty()) return 0.0;
//     std::vector<double> env(x.size());
//     for (size_t i=0;i<x.size();++i) env[i] = std::abs(x[i]);
//     double steady=0.0;
//     if ((int)x.size()>tail){ for (int i=(int)x.size()-tail;i<(int)x.size();++i) steady += env[i]; steady/=tail; }
//     else { for (auto v:env) steady += v; steady/= std::max<size_t>(1, env.size()); }
//     if (steady <= 0) return 0.0;
//     size_t n10=0,n90=0;
//     for (size_t i=0;i<env.size();++i){ if (env[i] >= 0.1*steady) { n10=i; break; } }
//     for (size_t i=0;i<env.size();++i){ if (env[i] >= 0.9*steady) { n90=i; break; } }
//     return (n90>n10)? ( (double)(n90-n10)*1e6/fs ) : 0.0;
// }

// static inline void iq_imbalance(const std::vector<cf>& x, double& alpha, double& phi_deg){
//     if (x.empty()){ alpha=1.0; phi_deg=0.0; return; }
//     double mII=0, mQQ=0, mIQ=0;
//     for (auto v:x){ mII += (double)v.real()*v.real(); mQQ += (double)v.imag()*v.imag(); mIQ += (double)v.real()*v.imag(); }
//     mII/=x.size(); mQQ/=x.size(); mIQ/=x.size();
//     alpha = std::sqrt( std::max(mII,1e-16) / std::max(mQQ,1e-16) );
//     phi_deg = 0.5 * std::atan2(2*mIQ, (mII - mQQ + 1e-16)) * 180.0/M_PI;
// }

// // FFT centroid over settled early window
// static inline float cfo_centroid(const std::vector<cf>& x, double fs, double f_lim=120e3, double settle_us=8){
//     if (x.size()<32) return 0.f;
//     size_t n_settle = (size_t)std::llround(settle_us*1e-6*fs);
//     size_t a = std::min(n_settle, x.size());
//     std::vector<cf> xw(x.begin()+a, x.end());
//     if (xw.size()<32) xw = x; // fallback
//     // PSD
//     std::vector<double> f,S;
//     psd_hann(xw, fs, f, S);
//     double sum=0, sumf=0;
//     for (size_t i=0;i<f.size();++i){
//         if (std::fabs(f[i])<=f_lim){ sum += S[i]; sumf += f[i]*S[i]; }
//     }
//     return (sum>0)? (float)(sumf/sum) : 0.f;
// }

// static inline float cfo_ls_window(const std::vector<cf>& x, double fs){
//     if (x.size()<4) return 0.f;
//     std::vector<double> ph(x.size());
//     ph[0] = std::arg(x[0]);
//     for (size_t i=1;i<x.size();++i){
//         double a = std::arg(x[i]);
//         double b = std::arg(x[i-1]);
//         double dp = a-b;
//         if (dp >  M_PI) a -= 2*M_PI;
//         if (dp < -M_PI) a += 2*M_PI;
//         ph[i] = ph[i-1] + (a-b);
//     }
//     std::vector<double> t(x.size());
//     for (size_t i=0;i<t.size();++i) t[i]=(double)i/fs;
//     double Sx=0,Sy=0,Sxx=0,Sxy=0; size_t N=ph.size();
//     for (size_t i=0;i<N;++i){ Sx+=t[i]; Sy+=ph[i]; Sxx+=t[i]*t[i]; Sxy+=t[i]*ph[i]; }
//     double slope = (N*Sxy - Sx*Sy)/std::max(1e-18, (N*Sxx - Sx*Sx));
//     return (float)(slope/(2*M_PI));
// }

// // 2-stage CFO estimate: centroid coarse + fine on derotated + LS on stable window
// static inline float cfo_two_stage(const std::vector<cf>& x, double fs, float& coarse){
//     float f0 = cfo_centroid(x, fs, 200e3, 8.0);
//     std::vector<cf> x0(x.size());
//     for (size_t n=0;n<x.size();++n){
//         float ang = (float)(-2*M_PI*f0*(double)n/fs);
//         x0[n] = x[n]*cf(std::cos(ang), std::sin(ang));
//     }
//     float f1 = cfo_centroid(x0, fs, 120e3, 8.0);
//     std::vector<cf> x1(x.size());
//     for (size_t n=0;n<x.size();++n){
//         float ang = (float)(-2*M_PI*f1*(double)n/fs);
//         x1[n] = x0[n]*cf(std::cos(ang), std::sin(ang));
//     }
//     coarse = f0+f1;
//     // simple LS on middle/stable part (skip first few microseconds)
//     size_t n_settle = (size_t)std::llround(8e-6*fs);
//     size_t a = std::min(n_settle, x1.size());
//     size_t b = x1.size();
//     if (b>a+std::max<size_t>(40,(size_t)(120*fs/1e6))){
//         std::vector<cf> seg(x1.begin()+a, x1.begin()+b);
//         return (float)(coarse + cfo_ls_window(seg, fs));
//     } else {
//         return std::numeric_limits<float>::quiet_NaN();
//     }
// }

// } // namespace feat

// // ------------------ Features CSV writer ------------------
// struct FeatureCSV {
//     std::FILE* f=nullptr;
//     explicit FeatureCSV(const std::string& path){
//         f = std::fopen(path.c_str(), "w");
//         if (!f) throw std::runtime_error("cannot open features csv: " + path);
//         std::fprintf(f,
//             "pkt_idx,pcap_ts,rf_channel,pdu_type,adv_addr,access_address,"
//             "cfo_quick_hz,cfo_centroid_hz,cfo_two_stage_hz,cfo_std_hz,cfo_std_sym_hz,"
//             "iq_gain_alpha,iq_phase_deg_deg,rise_time_us,psd_centroid_hz,psd_pnr_db,bw_3db_hz,gated_len_us,"
//             "cfo_two_stage_coarse_hz\n");
//         std::fflush(f);
//     }
//     void row(const FeatureRow& r){
//         std::fprintf(f,
//             "%zu,%.6f,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
//             r.pkt_idx, r.pcap_ts, r.rf_channel, r.pdu_type,
//             r.adv_addr.c_str(), r.access_address.c_str(),
//             r.cfo_quick_hz, r.cfo_centroid_hz, r.cfo_two_stage_hz, r.cfo_std_hz, r.cfo_std_sym_hz,
//             r.iq_gain_alpha, r.iq_phase_deg_deg, r.rise_time_us, r.psd_centroid_hz, r.psd_pnr_db,
//             r.bw_3db_hz, r.gated_len_us, r.cfo_two_stage_coarse_hz);
//         std::fflush(f);
//     }
//     ~FeatureCSV(){ if (f) std::fclose(f); }
// };

// // --------- Packet glue: capture PDUs, write PCAP & features ----------
// namespace blehelpers {
// static inline std::string to_adv_addr(const uint8_t* hdr_payload, int pdu_type){
//     char s[32];
//     if (pdu_type==0 || pdu_type==2 || pdu_type==6 || pdu_type==4){
//         std::snprintf(s, sizeof(s), "%02x:%02x:%02x:%02x:%02x:%02x",
//             hdr_payload[5], hdr_payload[4], hdr_payload[3], hdr_payload[2], hdr_payload[1], hdr_payload[0]);
//         return std::string(s);
//     } else if (pdu_type==3 || pdu_type==5){
//         std::snprintf(s, sizeof(s), "%02x:%02x:%02x:%02x:%02x:%02x",
//             hdr_payload[11], hdr_payload[10], hdr_payload[9], hdr_payload[8], hdr_payload[7], hdr_payload[6]);
//         return std::string(s);
//     }
//     return "";
// }
// static inline std::string aa_hex_be(const uint8_t* aa_le){
//     char s[9];
//     std::snprintf(s,sizeof(s), "%02X%02X%02X%02X", aa_le[3], aa_le[2], aa_le[1], aa_le[0]);
//     return std::string(s);
// }
// } // namespace blehelpers

// // ---------- Gating helpers (operate on x that includes prepad) ----------
// static bool find_energy_window(const std::vector<feat::cf>& x, double fs,
//                                size_t prepad_samps, double K, size_t pad_samps,
//                                size_t& a, size_t& b)
// {
//     if (x.empty()) return false;
//     size_t n0 = std::min(prepad_samps, x.size());
//     if (n0 == 0) n0 = std::min<size_t>(200, x.size());
//     double mu=0, s2=0;
//     for (size_t i=0;i<n0;i++){ double a0 = std::abs(x[i]); mu+=a0; s2+=a0*a0; }
//     mu/=std::max<size_t>(1,n0); s2/=std::max<size_t>(1,n0);
//     double sd = std::sqrt(std::max(0.0, s2 - mu*mu));
//     double T = mu + K*sd;
//     a = 0; b = x.size();
//     bool got_a=false, got_b=false;
//     for (size_t i=0;i<x.size();++i){ if (std::abs(x[i])>=T){ a=i; got_a=true; break; } }
//     for (size_t i=x.size(); i-->0; ){ if (std::abs(x[i])>=T){ b=i+1; got_b=true; break; } }
//     if (!(got_a && got_b) || b<=a) return false;
//     if (pad_samps){
//         a = (a>pad_samps)? a-pad_samps : 0;
//         b = std::min(b+pad_samps, x.size());
//     }
//     return (b>a);
// }

// static void apply_gate_energy(std::vector<feat::cf>& x, double fs, size_t prepad_samps, double K, int pad_us){
//     size_t pad = (size_t)std::llround(std::max(0, pad_us) * 1e-6 * fs);
//     size_t a=0,b=0;
//     if (find_energy_window(x, fs, prepad_samps, K, pad, a, b) && b>a && (b-a)>=32){
//         x = std::vector<feat::cf>(x.begin()+a, x.begin()+b);
//     }
// }

// static void apply_gate_struct(std::vector<feat::cf>& x, double fs, size_t prepad_samps){
//     // detect rise with energy, then take [rise+8us, rise+56us] (preamble+AA+hdr)
//     size_t a0=0,b0=0;
//     if (!find_energy_window(x, fs, prepad_samps, 4.0, 0, a0, b0)) return;
//     size_t off = (size_t)std::llround(8e-6*fs);
//     size_t span = (size_t)std::llround(56e-6*fs);
//     size_t a = std::min(a0 + off, x.size());
//     size_t b = std::min(a + span, x.size());
//     if (b>a && (b-a)>=32) x = std::vector<feat::cf>(x.begin()+a, x.begin()+b);
// }

// static void apply_gate_mid(std::vector<feat::cf>& x, double fs, size_t prepad_samps, int a_us, int b_us){
//     size_t a0=0,b0=0;
//     if (!find_energy_window(x, fs, prepad_samps, 4.0, 0, a0, b0)) return;
//     if (b_us < a_us) std::swap(b_us, a_us);
//     size_t a = std::min(a0 + (size_t)std::llround(std::max(0,a_us)*1e-6*fs), x.size());
//     size_t b = std::min(a0 + (size_t)std::llround(std::max(0,b_us)*1e-6*fs), x.size());
//     if (b>a && (b-a)>=32) x = std::vector<feat::cf>(x.begin()+a, x.begin()+b);
// }

// struct DumpCtx {
//     bool enabled=false;
//     std::string dir;
//     int sps=2;
//     double fs_eff=2e6;
//     int prepad_us=200;
//     Ring* ring=nullptr;
//     size_t pkt_idx=0;

//     FeatureCSV* featcsv=nullptr; // live writer
//     FeatureRows* feats_all=nullptr; // collector for posthoc sign-fix
//     // gating args snapshot
//     GateMode gate = GateMode::NONE;
//     double gate_k = 4.0;
//     int gate_pad_us = 8;
//     int gate_mid_a_us = 12;
//     int gate_mid_b_us = 80;
// };

// // --------- Packet glue: capture PDUs, write PCAP & features ----------
// static void attach_packet_handler(BLESDR& b, pcap::Writer& w, int rf_channel, DumpCtx& dctx) {
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
//         std::snprintf(sAA,sizeof(sAA), "%02X%02X%02X%02X", bytes_aa[3], bytes_aa[2], bytes_aa[1], bytes_aa[0]);
//         std::string aa_be = std::string(sAA);

//         // --- Feature window: pull IQ from ring around this packet
//         size_t bits = 8 + 32 + 16 + 8 * (size_t)pkt.length + 24;
//         size_t sps  = (size_t)std::max(2, dctx.sps);
//         size_t samps_needed = std::max((size_t)64, bits * sps);
//         size_t prepad_samps = (size_t)std::llround(dctx.fs_eff * dctx.prepad_us / 1e6);
//         size_t take_cplx    = prepad_samps + samps_needed;

//         std::vector<float> rawIQ;
//         dctx.ring->copy_tail(2*take_cplx, rawIQ); // floats (I,Q)
//         // Convert to complex
//         std::vector<feat::cf> x;
//         x.reserve(rawIQ.size()/2);
//         for (size_t i=0;i+1<rawIQ.size(); i+=2) x.emplace_back(rawIQ[i], rawIQ[i+1]);

//         // --- Normalize
//         feat::rm_dc_norm(x);

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

//         // --- Compute features on (possibly gated) x
//         double fcent=0, pnr_db=0, bw3=0;
//         feat::spectral_stats(x, dctx.fs_eff, fcent, pnr_db, bw3);
//         double gated_len_us = (double)x.size()*1e6/dctx.fs_eff;
//         double alpha=1.0, phi_deg=0.0;
//         feat::iq_imbalance(x, alpha, phi_deg);
//         double rt_us = feat::rise_time_us(x, dctx.fs_eff);

//         // CFOs
//         double cfo_q = feat::cfo_quick(x, dctx.fs_eff);
//         double cfo_c = feat::cfo_centroid(x, dctx.fs_eff, 120e3, 8.0);
//         float coarse=std::numeric_limits<float>::quiet_NaN();
//         double cfo_two = feat::cfo_two_stage(x, dctx.fs_eff, coarse);
//         double cfo_std_all = feat::cfo_std_all(x, dctx.fs_eff);
//         double cfo_std_sym = feat::cfo_std_symbol_avg(x, dctx.fs_eff);

//         FeatureRow row{
//             dctx.pkt_idx, ts, rf_channel, pdu_type, adv_addr, aa_be,
//             cfo_q, cfo_c, cfo_two, cfo_std_all, cfo_std_sym,
//             alpha, phi_deg, rt_us, fcent, pnr_db, bw3, gated_len_us, (double)coarse
//         };

//         // live CSV (for backward compat)
//         if (dctx.featcsv) dctx.featcsv->row(row);
//         // collect for posthoc sign-fix
//         if (dctx.feats_all) dctx.feats_all->push(row);

//         dctx.pkt_idx++;
//     };
// }

// int main(int argc, char** argv){
//     auto args = parse(argc, argv);

//     // Open capture (complex float32 interleaved)
//     std::FILE* f = std::fopen(args.file.c_str(), "rb");
//     if(!f) die(std::string("cannot open file: ") + args.file + " : " + std::strerror(errno));

//     // Effective complex sample rate after decim & derived SPS
//     const double fs_eff = args.fs / args.decim;
//     const int sps = std::max(2, (int)std::lround(fs_eff / 1e6)); // BLE-1M ⇒ ~2 at 2 MS/s

//     if (!args.dump_iq_dir.empty()) {
//         std::string cmd = "mkdir -p '" + args.dump_iq_dir + "'";
//         std::system(cmd.c_str());
//     }

//     pcap::Writer w(args.out);
//     FeatureCSV featcsv(args.features_out);
//     FeatureRows feats; // collector for sign-fix CSV

//     std::vector<float> bufIQ(2*args.chunk);
//     std::vector<float> workIQ;           // decimated interleaved complex floats
//     Ring ring;                           // ring for feature windows
//     ring.init((size_t)(fs_eff * 0.050)); // 50 ms ring

//     BLESDR blesdr;

//     DumpCtx dctx;
//     dctx.enabled   = !args.dump_iq_dir.empty();
//     dctx.dir       = args.dump_iq_dir;
//     dctx.sps       = sps;
//     dctx.fs_eff    = fs_eff;
//     dctx.prepad_us = args.prepad_us;
//     dctx.ring      = &ring;
//     dctx.featcsv   = &featcsv;
//     dctx.feats_all = &feats;
//     dctx.gate      = args.gate;
//     dctx.gate_k    = args.gate_k;
//     dctx.gate_pad_us = args.gate_pad_us;
//     dctx.gate_mid_a_us = args.gate_mid_a_us;
//     dctx.gate_mid_b_us = args.gate_mid_b_us;

//     // Attach the packet handler so every decoded packet writes PCAP + features CSV
//     attach_packet_handler(blesdr, w, args.channel, dctx);

//     // Feed chunks to the decoder
//     size_t total_complex = 0, total_complex_fed = 0;

//     for(;;){
//         size_t nread = std::fread(bufIQ.data(), sizeof(float)*2, args.chunk, f);
//         if (nread == 0) break;

//         // Complex decimation (keeps I,Q interleaved)
//         size_t n_cplx_out = decimate_cplx(bufIQ.data(), nread, args.decim, workIQ);

//         // DC-remove + RMS normalize per component (pre-conditioning for BLESDR and ring)
//         {
//             double meanI=0, meanQ=0;
//             for (size_t i=0;i<n_cplx_out;i++){ meanI += workIQ[2*i]; meanQ += workIQ[2*i+1]; }
//             if (n_cplx_out) { meanI/=n_cplx_out; meanQ/=n_cplx_out; }
//             double e=0;
//             for (size_t i=0;i<n_cplx_out;i++){
//                 workIQ[2*i]   = float(workIQ[2*i]   - meanI);
//                 workIQ[2*i+1] = float(workIQ[2*i+1] - meanQ);
//                 e += (double)workIQ[2*i]*workIQ[2*i] + (double)workIQ[2*i+1]*workIQ[2*i+1];
//             }
//             e = std::sqrt(e / std::max<double>(2.0*n_cplx_out,1.0));
//             if (e > 1e-12) for (size_t i=0;i<n_cplx_out;i++){ workIQ[2*i]/=e; workIQ[2*i+1]/=e; }
//         }

//         // Push into ring, then feed BLESDR (expects interleaved floats), samples_len = #complex samples.
//         for (size_t i=0;i<n_cplx_out;i++) ring.push(workIQ[2*i], workIQ[2*i+1]);
//         blesdr.Receiver((size_t)args.channel, workIQ.data(), n_cplx_out);

//         total_complex     += nread;
//         total_complex_fed += n_cplx_out;
//     }
//     std::fclose(f);

//     // -------- Posthoc: detrend + sign-fix column, write *_signfixed.csv --------
//     feats.write_csv(args.features_out /*raw*/);

//     std::vector<double> ts, cfoC; ts.reserve(feats.rows.size()); cfoC.reserve(feats.rows.size());
//     for (const auto& r : feats.rows){ ts.push_back(r.pcap_ts); cfoC.push_back(r.cfo_centroid_hz); }
//     double slope=0; bool flipped=false;
//     auto cfo_fixed = signfix_cfo_centroid(ts, cfoC, &slope, &flipped);

//     std::string out2 = args.features_out;
//     if (out2.size()>=4 && out2.substr(out2.size()-4)==".csv") out2.insert(out2.size()-4, "_signfixed");
//     else out2 += "_signfixed.csv";
//     feats.write_csv(out2, true, &cfo_fixed);

//     std::cerr << "[posthoc] detrend slope = " << slope << " Hz/s, flip=" << (flipped?"true":"false")
//               << ", wrote: " << out2 << "\n";

//     std::cerr << "Done. Complex read: " << total_complex
//               << ", complex fed: " << total_complex_fed
//               << ", packets (approx): " << dctx.pkt_idx
//               << ", features CSV: " << args.features_out
//               << ", PCAP: " << args.out
//               << "\n";
//     return 0;
// }

// // iq2pcap.cpp - Read complex-float I/Q, decode BLE with BLESDR, dump PCAP
// // FIXED: write LINKTYPE = DLT 256 (BLUETOOTH_LE_LL_WITH_PHDR) and prepend the
// //        required per-packet pseudo-header so Wireshark recognizes ADV_* PDUs.
// //        Also include the Access Address in the packet bytes (AA + PDU + CRC).
// //
// // Build: your existing CMake target that links with lib/BLESDR*.cpp
// // Usage:
// //   ./iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out_ch37.pcap [--decim 2] [--dump-iq-dir iq_dir] [--prepad-us 200]
// //
// // Notes:
// //   * DLT 256 pseudo-header layout (packed, little-endian):
// //       uint8  rf_channel;                // 0..39 (adv: 37/38/39)
// //       int8   signal_power;              // dBm; valid if flags bit 0x0002 set
// //       int8   noise_power;               // dBm; valid if flags bit 0x0004 set
// //       uint8  access_address_offenses;   // valid if flags bit 0x0020 set
// //       uint32 ref_access_address;        // valid if flags bit 0x0010 set
// //       uint16 flags;                     // see bits below
// //       uint8  le_packet[];               // AA + PDU + CRC (no preamble)
// //     Flag bits we use here:
// //       0x0001 => le_packet is de-whitened (we set this)
// //       0x0010 => ref_access_address is valid (we set this)
// //     (We DO NOT claim CRC checked/passed; Wireshark will show CRC Checked: False.)
// //   * Packet bytes written after the pseudo-header are: 4-byte AA + (2+len+3).
// //     Your BLESDR callback already exposes symbols[] = AA(4) + header+payload+CRC.
// //   * If your decoder outputs LSB-first bits inside each byte, bit-reverse each byte
// //     before writing (helper provided below). If AA appears as 0x8e89bed6 in Wireshark,
// //     your bit ordering is fine.

// // ------------------ Includes ------------------
// #include <cstdio>
// #include <cstdint>
// #include <cstdlib>
// #include <cstring>
// #include <cerrno>
// #include <string>
// #include <vector>
// #include <functional>
// #include <memory>
// #include <iostream>
// #include <fstream>
// #include <chrono>
// #include <cmath>
// #include <algorithm>

// #include "lib/BLESDR.hpp"   // adjust include path if needed

// // ------------------ Simple PCAP writer ------------------
// namespace pcap {
// static constexpr uint32_t MAGIC   = 0xA1B2C3D4;
// static constexpr uint16_t VMAJOR  = 2;
// static constexpr uint16_t VMINOR  = 4;
// static constexpr uint32_t SNAPLEN = 0xFFFF;
// static constexpr uint32_t LINKTYPE_BLE_LL_WITH_PHDR = 256; // DLT 256

// #pragma pack(push, 1)
// struct le_phdr {
//     uint8_t  rf_channel;            // 0..39 (adv: 37/38/39)
//     int8_t   signal_power;          // dBm; valid iff flags & 0x0002
//     int8_t   noise_power;           // dBm; valid iff flags & 0x0004
//     uint8_t  access_address_offenses; // valid iff flags & 0x0020
//     uint32_t ref_access_address;    // valid iff flags & 0x0010 (LE)
//     uint16_t flags;                 // bitfield, see below
// };
// // Flag bits (subset used here)
// static constexpr uint16_t LE_FLAG_DEWHITENED      = 0x0001;
// static constexpr uint16_t LE_FLAG_SIGNAL_VALID    = 0x0002;
// static constexpr uint16_t LE_FLAG_NOISE_VALID     = 0x0004;
// static constexpr uint16_t LE_FLAG_REF_AA_VALID    = 0x0010;
// static constexpr uint16_t LE_FLAG_AA_OFFENSES_OK  = 0x0020;
// static constexpr uint16_t LE_FLAG_CRC_CHECKED     = 0x0400;
// static constexpr uint16_t LE_FLAG_CRC_VALID       = 0x0800;
// #pragma pack(pop)

// struct Writer {
//     std::FILE* f = nullptr;
//     explicit Writer(const std::string& path) {
//         f = std::fopen(path.c_str(), "wb");
//         if (!f) { throw std::runtime_error("fopen failed: " + path); }
//         // global header (native endian, classic pcap)
//         uint32_t magic = MAGIC;
//         uint16_t vmaj = VMAJOR, vmin = VMINOR;
//         uint32_t thiszone = 0, sigfigs = 0, snaplen = SNAPLEN, network = LINKTYPE_BLE_LL_WITH_PHDR;
//         std::fwrite(&magic,   4,1,f);
//         std::fwrite(&vmaj,    2,1,f);
//         std::fwrite(&vmin,    2,1,f);
//         std::fwrite(&thiszone,4,1,f);
//         std::fwrite(&sigfigs, 4,1,f);
//         std::fwrite(&snaplen, 4,1,f);
//         std::fwrite(&network, 4,1,f);
//     }
//     void write_pkt(const uint8_t* data, size_t len, double ts_sec_f = -1.0) {
//         using clock = std::chrono::system_clock;
//         double now = ts_sec_f >= 0 ? ts_sec_f
//                                    : std::chrono::duration<double>(clock::now().time_since_epoch()).count();
//         uint32_t ts_sec  = static_cast<uint32_t>(now);
//         uint32_t ts_usec = static_cast<uint32_t>((now - ts_sec)*1e6 + 0.5);
//         uint32_t incl = static_cast<uint32_t>(len);
//         uint32_t orig = static_cast<uint32_t>(len);
//         std::fwrite(&ts_sec,  4,1,f);
//         std::fwrite(&ts_usec, 4,1,f);
//         std::fwrite(&incl,    4,1,f);
//         std::fwrite(&orig,    4,1,f);
//         if (len) std::fwrite(data, 1, len, f);
//     }
//     ~Writer(){ if(f) std::fclose(f); }
// };
// } // namespace pcap

// // ------------------ Helpers ------------------
// struct Args {
//     std::string file;
//     std::string out = "out.pcap";
//     int channel = 37;       // 37/38/39
//     double fs = 4e6;        // input sample rate (complex baseband)
//     int decim = 2;          // complex decimation (4->2 typical). 1 = no decimation
//     size_t chunk = 1'000'000; // complex samples per read

//     // Optional per-packet IQ dumping
//     std::string dump_iq_dir = "";  // empty disables
//     int prepad_us = 200;           // prepend this many microseconds of IQ before packet
// };

// static void die(const std::string& s) { std::cerr << "error: " << s << "\n"; std::exit(1); }

// static Args parse(int argc, char** argv){
//     Args a;
//     for (int i=1;i<argc;i++){
//         std::string k = argv[i];
//         auto need = [&](const char* name)->const char*{
//             if (i+1>=argc) die(std::string("missing value after ")+name);
//             return argv[++i];
//         };
//         if (k=="--file")         a.file = need("--file");
//         else if (k=="--out")     a.out  = need("--out");
//         else if (k=="--fs")      a.fs   = std::stod(need("--fs"));
//         else if (k=="--channel") a.channel = std::stoi(need("--channel"));
//         else if (k=="--decim")   a.decim= std::stoi(need("--decim"));
//         else if (k=="--chunk")   a.chunk= static_cast<size_t>(std::stoll(need("--chunk")));
//         else if (k=="--dump-iq-dir") a.dump_iq_dir = need("--dump-iq-dir");
//         else if (k=="--prepad-us")   a.prepad_us   = std::stoi(need("--prepad-us"));
//         else if (k=="-h" || k=="--help"){
//             std::cout <<
// "Usage: iq2pcap --file ble_ch37.dat --fs 4e6 --channel 37 --out out.pcap [--decim 2] [--chunk 1000000]\n"
// "              [--dump-iq-dir iq_dir] [--prepad-us 200]\n"
// "  file:    complex float32 interleaved I/Q capture at baseband (CH37/38/39 center)\n"
// "  fs:      input sample rate in Hz (e.g., 4e6)\n"
// "  channel: 37/38/39 (adv channels)\n"
// "  decim:   integer complex decimation (typ 2 for 4->2 MS/s)\n"
// "  chunk:   complex samples per read per iteration\n"
// "  dump-iq-dir: if set, write pkt_XXXX.fc32 (float32 I,Q) per decoded packet\n"
// "  prepad-us: microseconds of I/Q preceding packet to include in the dump\n";
//             std::exit(0);
//         }
//     }
//     if (a.file.empty()) die("please specify --file");
//     if (a.channel<37 || a.channel>39) die("channel must be 37, 38 or 39");
//     if (a.decim<1) a.decim = 1;
//     return a;
// }

// // Decimate interleaved complex float32 stream by N (keep every Nth complex sample)
// static size_t decimate_cplx(const float* iq, size_t n_cplx, int N, std::vector<float>& outIQ) {
//     if (N < 1) N = 1;
//     outIQ.clear();
//     outIQ.reserve(2 * (n_cplx / (size_t)N + 16));
//     for (size_t k = 0; k < n_cplx; k += (size_t)N) {
//         outIQ.push_back(iq[2*k]);     // I
//         outIQ.push_back(iq[2*k + 1]); // Q
//     }
//     return outIQ.size() / 2; // # of complex samples
// }

// // Optional: bit-reverse if your decoder returns LSB-first bits in each byte.
// // If AA shows correctly as 0x8E89BED6 in Wireshark, you probably don't need this.
// static inline uint8_t bitrev8(uint8_t x){
//     x = (uint8_t)((x>>4) | (x<<4));
//     x = (uint8_t)(((x&0xCC)>>2) | ((x&0x33)<<2));
//     x = (uint8_t)(((x&0xAA)>>1) | ((x&0x55)<<1));
//     return x;
// }
// static void bitrev_buf(uint8_t* p, size_t n) {
//     for (size_t i=0;i<n;i++) p[i] = bitrev8(p[i]);
// }

// // --------- Packet glue: capture PDUs from BLESDR ----------
// struct PduStore {
//     std::vector< std::vector<uint8_t> > frames; // full frames written to pcap (phdr + AA+PDU+CRC)
//     void clear() { frames.clear(); }
//     void add(const std::vector<uint8_t>& v){ frames.push_back(v); }
// };

// // --------- Ring buffer for I/Q (for per-packet dump) ----------
// struct Ring {
//     std::vector<float> buf; // interleaved I,Q
//     size_t w = 0;           // write index in floats
//     size_t cap = 0;         // capacity in complex samples
//     void init(size_t complex_len) {
//         cap = std::max<size_t>(complex_len, 4096);
//         buf.assign(2*cap, 0.0f);
//         w = 0;
//     }
//     inline void push(float I, float Q) {
//         buf[w] = I;
//         size_t w2 = (w+1)%(2*cap);
//         buf[w2] = Q;
//         w = (w+2)%(2*cap);
//     }
//     // copy last 'take_floats' floats (I,Q interleaved) into out
//     void copy_tail(size_t take_floats, std::vector<float>& out) const {
//         size_t maxf = 2*cap;
//         if (take_floats > maxf) take_floats = maxf;
//         out.resize(take_floats);
//         size_t start = ( (w + maxf) - take_floats ) % maxf;
//         for (size_t i=0;i<take_floats;i++) out[i] = buf[(start + i)%maxf];
//     }
// };

// // Context passed to callback for IQ dumping
// struct DumpCtx {
//     bool enabled=false;
//     std::string dir;
//     int sps=2;            // samples/symbol at fs_eff (2 for 2 MS/s)
//     double fs_eff=2e6;    // complex sample rate after decim
//     int prepad_us=200;    // µs of context before packet
//     Ring* ring=nullptr;
//     size_t pkt_idx=0;
// };

// // ========= Attach handler using your class member `callback` =========
// // BLESDR exposes bytes via lell_packet.symbols[]:
// // symbols[0..3] = AA (LE), symbols[4..] = header+payload+CRC. length = payload length.
// static void attach_packet_handler(BLESDR& b, PduStore& store, DumpCtx* dctx, pcap::Writer& w, int rf_channel) {
//     static size_t seen = 0;
//     b.callback = [&](lell_packet pkt){
//         // Build the DLT 256 pseudo-header
//         pcap::le_phdr ph{};
//         ph.rf_channel = static_cast<uint8_t>(rf_channel); // 37/38/39
//         ph.signal_power = 127;    // unknown (valid bit not set)
//         ph.noise_power  = 127;    // unknown (valid bit not set)
//         ph.access_address_offenses = 0; // unknown (valid bit not set)
//         ph.ref_access_address = 0x8E89BED6u; // advertising AA (little-endian in file)
//         ph.flags = pcap::LE_FLAG_DEWHITENED | pcap::LE_FLAG_REF_AA_VALID;
//         // NOTE: Do not set CRC checked/valid bits unless you compute/verify CRC.

//         // Packet data to follow: AA (4) + (header+payload+CRC)
//         const uint8_t* bytes_aa = pkt.symbols;       // 4 bytes AA
//         const uint8_t* bytes_pdu = pkt.symbols + 4;  // header+payload+CRC
//         size_t pdu_len = static_cast<size_t>(pkt.length) + 5; // 2 hdr + payload_len + 3 CRC
//         size_t frame_len = sizeof(ph) + 4 + pdu_len;

//         std::vector<uint8_t> frame;
//         frame.resize(frame_len);

//         // Copy pseudo-header
//         std::memcpy(frame.data(), &ph, sizeof(ph));
//         // Copy AA + PDU+CRC (already de-whitened by BLESDR)
//         std::memcpy(frame.data()+sizeof(ph), bytes_aa, 4);
//         std::memcpy(frame.data()+sizeof(ph)+4, bytes_pdu, pdu_len);

//         // If your decoder outputs LSB-first bits per byte, uncomment this:
//         // bitrev_buf(frame.data()+sizeof(ph), 4 + pdu_len);

//         store.add(frame);
//         w.write_pkt(frame.data(), frame.size());

//         if (seen < 5) {
//             std::fprintf(stderr, "[BLESDR] wrote frame len=%zu ch=%u AA=%02X%02X%02X%02X first16=",
//                          frame.size(), pkt.channel_idx,
//                          bytes_aa[3], bytes_aa[2], bytes_aa[1], bytes_aa[0]);
//             const uint8_t* dbg = frame.data()+sizeof(ph);
//             for (size_t i=0;i<std::min<size_t>(4+pdu_len,16);++i) std::fprintf(stderr, "%02X", dbg[i]);
//             std::fprintf(stderr, "\n");
//         }
//         ++seen;

//         // 2) Optional: dump interleaved (I,Q) window around this packet
//         if (dctx && dctx->enabled && dctx->ring) {
//             // Conservative bit count: preamble(8)+AA(32)+hdr(16)+8*len + CRC(24)
//             size_t bits = 8 + 32 + 16 + 8*static_cast<size_t>(pkt.length) + 24;
//             size_t samps_needed = bits * static_cast<size_t>(dctx->sps);
//             size_t prepad_samps = static_cast<size_t>((dctx->fs_eff * dctx->prepad_us)/1e6);
//             // Convert to float count (I,Q → 2 floats per complex)
//             size_t take_floats = 2 * (prepad_samps + samps_needed);
//             if (take_floats < 2*64) take_floats = 2*64;

//             std::vector<float> outIQ;
//             dctx->ring->copy_tail(take_floats, outIQ);

//             char fname[256];
//             std::snprintf(fname, sizeof(fname), "%s/pkt_%06zu.fc32", dctx->dir.c_str(), dctx->pkt_idx++);
//             if (std::FILE* fp = std::fopen(fname, "wb")) {
//                 std::fwrite(outIQ.data(), sizeof(float), outIQ.size(), fp);
//                 std::fclose(fp);
//             }
//         }
//     };
// }

// int main(int argc, char** argv){
//     auto args = parse(argc, argv);

//     // Open capture (complex float32 interleaved)
//     std::FILE* f = std::fopen(args.file.c_str(), "rb");
//     if(!f) die(std::string("cannot open file: ") + args.file + " : " + std::strerror(errno));

//     // Effective complex sample rate after decim & derived SPS (for dumping window sizing)
//     const double fs_eff = args.fs / args.decim;
//     const int sps = std::max(2, static_cast<int>(std::lround(fs_eff / 1e6))); // BLE-1M ⇒ ~2 at 2 MS/s

//     if (!args.dump_iq_dir.empty()) {
//         std::string cmd = "mkdir -p '" + args.dump_iq_dir + "'";
//         std::system(cmd.c_str());
//     }

//     pcap::Writer w(args.out);
//     std::vector<float> bufIQ(2*args.chunk);
//     std::vector<float> workIQ;           // decimated complex
//     PduStore store;
//     Ring ring;                           // ring for IQ dump context
//     ring.init(static_cast<size_t>(fs_eff * 0.050)); // 50 ms ring

//     BLESDR blesdr;

//     DumpCtx dctx;
//     dctx.enabled   = !args.dump_iq_dir.empty();
//     dctx.dir       = args.dump_iq_dir;
//     dctx.sps       = sps;
//     dctx.fs_eff    = fs_eff;
//     dctx.prepad_us = args.prepad_us;
//     dctx.ring      = &ring;

//     // Attach the packet handler so every decoded packet goes straight to PCAP (with pseudo-header)
//     attach_packet_handler(blesdr, store, dctx.enabled ? &dctx : nullptr, w, args.channel);

//     // Feed chunks to the decoder
//     size_t total_complex = 0, total_complex_fed = 0, total_frames = 0;

//     for(;;){
//         // bufIQ has 2*chunk floats; read 'args.chunk' complex samples per fread
//         size_t nread = std::fread(bufIQ.data(), sizeof(float)*2, args.chunk, f);
//         if (nread == 0) break;
        
//         // Complex decimation (keeps I,Q interleaved)
//         size_t n_cplx_out = decimate_cplx(bufIQ.data(), nread, args.decim, workIQ);

//         // DC-remove + RMS normalize per component
//         {
//             double meanI=0, meanQ=0;
//             for (size_t i=0;i<n_cplx_out;i++){ meanI += workIQ[2*i]; meanQ += workIQ[2*i+1]; }
//             if (n_cplx_out) { meanI/=n_cplx_out; meanQ/=n_cplx_out; }
//             double e=0;
//             for (size_t i=0;i<n_cplx_out;i++){
//                 workIQ[2*i]   = float(workIQ[2*i]   - meanI);
//                 workIQ[2*i+1] = float(workIQ[2*i+1] - meanQ);
//                 e += (double)workIQ[2*i]*workIQ[2*i] + (double)workIQ[2*i+1]*workIQ[2*i+1];
//             }
//             e = std::sqrt(e / std::max<double>(2.0*n_cplx_out,1.0));
//             if (e > 1e-12) for (size_t i=0;i<n_cplx_out;i++){ workIQ[2*i]/=e; workIQ[2*i+1]/=e; }
//         }
        
//         // Push into ring, then feed BLESDR (expects complex interleaved), samples_len = #complex samples.
//         for (size_t i=0;i<n_cplx_out;i++) ring.push(workIQ[2*i], workIQ[2*i+1]);
//         blesdr.Receiver((size_t)args.channel, workIQ.data(), n_cplx_out);

//         // Count frames
//         total_frames += store.frames.size();
//         store.clear();

//         total_complex     += nread;
//         total_complex_fed += n_cplx_out;
//     }
//     std::fclose(f);

//     std::cerr << "Done. Complex read: " << total_complex
//               << ", complex fed: " << total_complex_fed
//               << ", frames written: " << total_frames
//               << (args.dump_iq_dir.empty() ? "" : (", IQ dump dir: " + args.dump_iq_dir))
//               << "\n";
//     return 0;
// }