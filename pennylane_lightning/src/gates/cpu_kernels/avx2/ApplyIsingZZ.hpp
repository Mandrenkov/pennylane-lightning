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
 * Defines [] gate
 */
#pragma once
#include "AVX2Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX2 {
/// @cond DEV
template <class ParamT>
void applyIsingZZFloatInternalInternal(std::complex<float> *arr,
                                       size_t num_qubits, bool inverse,
                                       ParamT angle) {
    // rev_wires must be 0 and 1
    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    __m256 parity;

    // clang-format off
    parity = _mm256_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                            -1.0F, -1.0F, 1.0F, 1.0F);
    // clang-format on
    const __m256 real_cos_factor = _mm256_set1_ps(std::cos(angle / 2));
    const __m256 imag_sin_factor =
        _mm256_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin);
    const __m256 imag_sin_parity = _mm256_mul_ps(imag_sin_factor, parity);
    
    auto* p = reinterpret_cast<float*>(arr);
    for (size_t n = 0; n < exp2(num_qubits);
         n += step_for_complex_precision<float>) {
        __m256 coeffs = _mm256_load_ps(p + 2*n);
        __m256 prod_cos = _mm256_mul_ps(real_cos_factor, coeffs);

        __m256 prod_sin = _mm256_mul_ps(coeffs, imag_sin_parity);

        __m256 prod = _mm256_add_ps(prod_cos,
                                    _mm256_permute_ps(prod_sin, 0B10110001));
        _mm256_store_ps(p + 2*n, prod);
    }
}
template <class ParamT>
void applyIsingZZFloatInternalExternal(std::complex<float> *arr,
                                       size_t num_qubits, size_t rev_wire0,
                                       size_t rev_wire1, bool inverse,
                                       ParamT angle) {
    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t max_rev_wire_shift = (static_cast<size_t>(1U) << rev_wire_max);
    const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
    const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m256 real_cos_factor = _mm256_set1_ps(std::cos(angle / 2));
    const __m256 imag_sin_factor =
        _mm256_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin);

    const __m256 imag_sin_parity0 =
        _mm256_mul_ps(imag_sin_factor, Internal::paritySInternal(rev_wire_min));
    const __m256 imag_sin_parity1 =
        _mm256_mul_ps(imag_sin_parity0, _mm256_set1_ps(-1.0L));

    auto* p = reinterpret_cast<float*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m256 v0 = _mm256_load_ps(p + 2*i0);
        const __m256 v1 = _mm256_load_ps(p + 2*i1);

        __m256 prod_cos0 = _mm256_mul_ps(real_cos_factor, v0);
        __m256 prod_sin0 = _mm256_mul_ps(v0, imag_sin_parity0);

        __m256 prod0 =
            _mm256_add_ps(prod_cos0, _mm256_permute_ps(prod_sin0, 0B10110001));

        __m256 prod_cos1 = _mm256_mul_ps(real_cos_factor, v1);
        __m256 prod_sin1 = _mm256_mul_ps(v1, imag_sin_parity1);

        __m256 prod1 =
            _mm256_add_ps(prod_cos1, _mm256_permute_ps(prod_sin1, 0B10110001));

        _mm256_store_ps(p + 2*i0, prod0);
        _mm256_store_ps(p + 2*i1, prod1);
    }
}

template <class ParamT>
void applyIsingZZFloatExternalExternal(std::complex<float> *arr,
                                       const size_t num_qubits,
                                       const size_t rev_wire0,
                                       const size_t rev_wire1,
                                       bool inverse,
                                       ParamT angle) {
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t parity_low = fillTrailingOnes(rev_wire_min);
    const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
    const size_t parity_middle =
        fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m256 real_cos_factor = _mm256_set1_ps(std::cos(angle / 2));

    const auto isin_prod = Util::ProdPureImag<float>(isin);
    const auto minus_isin_prod = Util::ProdPureImag<float>(-isin);

    auto* p = reinterpret_cast<float*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<float>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const __m256 v00 = _mm256_load_ps(p + 2*i00); // 00
        const __m256 v01 = _mm256_load_ps(p + 2*i01); // 01
        const __m256 v10 = _mm256_load_ps(p + 2*i10); // 10
        const __m256 v11 = _mm256_load_ps(p + 2*i11); // 11

        const __m256 prod_cos00 = _mm256_mul_ps(real_cos_factor, v00);
        const __m256 prod_isin00 = isin_prod.product(v00);

        const __m256 prod_cos01 = _mm256_mul_ps(real_cos_factor, v01);
        const __m256 prod_isin01 = minus_isin_prod.product(v01);

        const __m256 prod_cos10 = _mm256_mul_ps(real_cos_factor, v10);
        const __m256 prod_isin10 = minus_isin_prod.product(v10);

        const __m256 prod_cos11 = _mm256_mul_ps(real_cos_factor, v11);
        const __m256 prod_isin11 = isin_prod.product(v11);

        _mm256_store_ps(p + 2*i00, _mm256_add_ps(prod_cos00, prod_isin00));
        _mm256_store_ps(p + 2*i01, _mm256_add_ps(prod_cos01, prod_isin01));
        _mm256_store_ps(p + 2*i10, _mm256_add_ps(prod_cos10, prod_isin10));
        _mm256_store_ps(p + 2*i11, _mm256_add_ps(prod_cos11, prod_isin11));

    }
}

template <class ParamT>
static void
applyIsingZZDoubleInternalExternal(std::complex<double> *arr,
                                   size_t num_qubits,
                                   size_t rev_wire_max,
                                   bool inverse,
                                   ParamT angle) {
    // rev_wire_min must be 0
    const size_t max_rev_wire_shift = (static_cast<size_t>(1U) << rev_wire_max);
    const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
    const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m256d real_cos_factor = _mm256_set1_pd(std::cos(angle / 2));
    const __m256d imag_sin_factor =
        _mm256_set_pd(-isin, isin, -isin, isin);

    const __m256d imag_sin_parity0 =
        _mm256_mul_pd(imag_sin_factor, Internal::parityDInternal());
    const __m256d imag_sin_parity1 =
        _mm256_mul_pd(imag_sin_parity0, _mm256_set1_pd(-1.0L));

    auto* p = reinterpret_cast<double*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m256d v0 = _mm256_load_pd(p + 2*i0);
        const __m256d v1 = _mm256_load_pd(p + 2*i1);

        __m256d prod_cos0 = _mm256_mul_pd(real_cos_factor, v0);
        __m256d prod_sin0 = _mm256_mul_pd(v0, imag_sin_parity0);

        __m256d prod0 =
            _mm256_add_pd(prod_cos0, _mm256_permute_pd(prod_sin0, 0B01010101));

        __m256d prod_cos1 = _mm256_mul_pd(real_cos_factor, v1);
        __m256d prod_sin1 = _mm256_mul_pd(v1, imag_sin_parity1);

        __m256d prod1 =
            _mm256_add_pd(prod_cos1, _mm256_permute_pd(prod_sin1, 0B01010101));

        _mm256_store_pd(p + 2*i0, prod0);
        _mm256_store_pd(p + 2*i1, prod1);
    }
}

template <class ParamT>
static void applyIsingZZDoubleExternalExternal(std::complex<double> *arr,
                                               const size_t num_qubits,
                                               const size_t rev_wire0,
                                               const size_t rev_wire1,
                                               bool inverse, ParamT angle) {
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t parity_low = fillTrailingOnes(rev_wire_min);
    const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
    const size_t parity_middle =
        fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m256d real_cos_factor = _mm256_set1_pd(std::cos(angle / 2));

    const auto isin_prod = Util::ProdPureImag<double>(isin);
    const auto minus_isin_prod = Util::ProdPureImag<double>(-isin);

    auto* p = reinterpret_cast<double*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<double>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const __m256d v00 = _mm256_load_pd(p + 2*i00); // 00
        const __m256d v01 = _mm256_load_pd(p + 2*i01); // 01
        const __m256d v10 = _mm256_load_pd(p + 2*i10); // 10
        const __m256d v11 = _mm256_load_pd(p + 2*i11); // 11

        const __m256d prod_cos00 = _mm256_mul_pd(real_cos_factor, v00);
        const __m256d prod_isin00 = isin_prod.product(v00);

        const __m256d prod_cos01 = _mm256_mul_pd(real_cos_factor, v01);
        const __m256d prod_isin01 = minus_isin_prod.product(v01);

        const __m256d prod_cos10 = _mm256_mul_pd(real_cos_factor, v10);
        const __m256d prod_isin10 = minus_isin_prod.product(v10);

        const __m256d prod_cos11 = _mm256_mul_pd(real_cos_factor, v11);
        const __m256d prod_isin11 = isin_prod.product(v11);

        _mm256_store_pd(p + 2*i00, _mm256_add_pd(prod_cos00, prod_isin00));
        _mm256_store_pd(p + 2*i01, _mm256_add_pd(prod_cos01, prod_isin01));
        _mm256_store_pd(p + 2*i10, _mm256_add_pd(prod_cos10, prod_isin10));
        _mm256_store_pd(p + 2*i11, _mm256_add_pd(prod_cos11, prod_isin11));
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX2
