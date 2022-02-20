// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Defines RZ gate
 */
#pragma once
#include "AVX2Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX2 {
template <class ParamT>
void applyRZFloatInternal(std::complex<float> *arr, const size_t num_qubits,
                          const size_t rev_wire, [[maybe_unused]] bool inverse,
                          ParamT angle) {
    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

    const __m256 real_cos_factor = _mm256_set1_ps(std::cos(angle / 2));
    const __m256 imag_sin_factor =
        _mm256_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin);

    const __m256 imag_sin_parity =
        _mm256_mul_ps(imag_sin_factor, Internal::paritySInternal(rev_wire));
    
    auto* p = reinterpret_cast<float*>(arr);
    for (size_t n = 0; n < (1U << num_qubits);
         n += step_for_complex_precision<float>) {
        __m256 coeffs = _mm256_load_ps(p + 2*n);
        __m256 prod_cos = _mm256_mul_ps(real_cos_factor, coeffs);

        __m256 prod_sin = _mm256_mul_ps(coeffs, imag_sin_parity);

        __m256 prod =
            _mm256_add_ps(prod_cos, _mm256_permute_ps(prod_sin, 0B10110001));
        _mm256_store_ps(p + 2*n, prod);
    }
}
template <class ParamT>
void applyRZFloatExternal(std::complex<float> *arr, const size_t num_qubits,
                          const size_t rev_wire, [[maybe_unused]] bool inverse,
                          ParamT angle) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    const __m256 real_cos_factor = _mm256_set1_ps(std::cos(angle / 2));
    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

    const auto plus_isin_prod = Util::ProdPureImag<float>(isin);
    const auto minus_isin_prod = Util::ProdPureImag<float>(-isin);

    auto* p = reinterpret_cast<float*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256 v0 = _mm256_load_ps(p + 2*i0);
        const __m256 v1 = _mm256_load_ps(p + 2*i1);

        const auto v0_cos = _mm256_mul_ps(v0, real_cos_factor);
        const auto v0_isin = plus_isin_prod.product(v0);

        const auto v1_cos = _mm256_mul_ps(v1, real_cos_factor);
        const auto v1_isin = minus_isin_prod.product(v1);

        _mm256_store_ps(p + 2*i0, _mm256_add_ps(v0_cos, v0_isin));
        _mm256_store_ps(p + 2*i1, _mm256_add_ps(v1_cos, v1_isin));
    }
}

template <class ParamT>
void applyRZDoubleInternal(std::complex<double> *arr,
                           const size_t num_qubits,
                           [[maybe_unused]] bool inverse,
                           ParamT angle) {
    // rev_wire must be 0
    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m256d real_cos_factor = _mm256_set1_pd(std::cos(angle / 2));
    const __m256d imag_sin_factor =
        _mm256_set_pd(-isin, isin, -isin, isin);

    const __m256d imag_sin_parity =
        _mm256_mul_pd(imag_sin_factor, Internal::parityDInternal());

    auto* p = reinterpret_cast<float*>(arr);
    for (size_t n = 0; n < (1U << num_qubits);
         n += step_for_complex_precision<double>) {
        __m256d coeffs = _mm256_load_pd(p + 2*n);
        __m256d prod_cos = _mm256_mul_pd(real_cos_factor, coeffs);

        __m256d prod_sin = _mm256_mul_pd(coeffs, imag_sin_parity);

        __m256d prod =
            _mm256_add_pd(prod_cos, _mm256_permutex_pd(prod_sin, 0B10110001));
        _mm256_store_pd(p + 2*n, prod);
    }
}
template <class ParamT>
void applyRZDoubleExternal(std::complex<double> *arr, const size_t num_qubits,
                           const size_t rev_wire, [[maybe_unused]] bool inverse,
                           ParamT angle) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    const __m256d real_cos_factor = _mm256_set1_pd(std::cos(angle / 2));
    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

    const auto plus_isin_prod = Util::ProdPureImag<double>(isin);
    const auto minus_isin_prod = Util::ProdPureImag<double>(-isin);

    auto* p = reinterpret_cast<float*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256d v0 = _mm256_load_pd(p + 2*i0);
        const __m256d v1 = _mm256_load_pd(p + 2*i1);

        const auto v0_cos = _mm256_mul_pd(v0, real_cos_factor);
        const auto v0_isin = plus_isin_prod.product(v0);

        const auto v1_cos = _mm256_mul_pd(v1, real_cos_factor);
        const auto v1_isin = minus_isin_prod.product(v1);

        _mm256_store_pd(p + 2*i0, _mm256_add_pd(v0_cos, v0_isin));
        _mm256_store_pd(p + 2*i1, _mm256_add_pd(v1_cos, v1_isin));
    }
}
} // namespace Pennylane::Gates::AVX2
