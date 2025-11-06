#pragma once
#include <vector>
#include <complex>
#include "ble_packet_50k_iq.hpp"

namespace testload {
    using cf = std::complex<float>;

    inline std::vector<cf> make_vec_from_pairs(const float* iq_pairs, size_t n_complex) {
        std::vector<cf> x;
        x.reserve(n_complex);
        for (size_t i = 0; i < n_complex; ++i) {
            float I = iq_pairs[2*i + 0];
            float Q = iq_pairs[2*i + 1];
            x.emplace_back(I, Q);
        }
        return x;
    }
}