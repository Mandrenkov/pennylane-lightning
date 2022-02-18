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
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
/// @cond DEV
template <class ParamT>
void applyIsingZZFloatInternalInternal(std::complex<float> *arr,
                                       size_t num_qubits, size_t rev_wire0,
                                       size_t rev_wire1, bool inverse,
                                       ParamT angle) {

    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    __m512 parity;

    // clang-format off
    switch(rev_wire0 ^ rev_wire1) {
    /* Possible values are (max_rev_wire, min_rev_wire) =
     *     {(1, 0) = 1, (2, 0) = 2, (2, 1) = 3}
     * */
    case 1: // (1,0)
        parity = _mm512_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                                -1.0F, -1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F,
                                -1.0F, -1.0F, 1.0F, 1.0F);
        break;
    case 2: // (2,0)
        parity = _mm512_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F,
                                -1.0F, -1.0F, 1.0F, 1.0F,
                                -1.0F, -1.0F, 1.0F, 1.0F);
        break;
    case 3: // (2,1)
        parity = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, -1.0F, -1.0F, -1.0F,
                                -1.0F, -1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F);
        break;
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
    const __m512 imag_sin_factor =
        _mm512_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin, -isin,
                      isin, -isin, isin, -isin, isin, -isin, isin);
    const __m512 imag_sin_parity = _mm512_mul_ps(imag_sin_factor, parity);

    for (size_t n = 0; n < exp2(num_qubits);
         n += step_for_complex_precision<float>) {
        __m512 coeffs = _mm512_load_ps(arr + n);
        __m512 prod_cos = _mm512_mul_ps(real_cos_factor, coeffs);

        __m512 prod_sin = _mm512_mul_ps(coeffs, imag_sin_parity);

        __m512 prod =
            _mm512_add_ps(prod_cos, _mm512_permute_ps(prod_sin, 0B10110001));
        _mm512_store_ps(arr + n, prod);
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
    const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
    const __m512 imag_sin_factor =
        _mm512_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin, -isin,
                      isin, -isin, isin, -isin, isin, -isin, isin);

    const __m512 imag_sin_parity0 =
        _mm512_mul_ps(imag_sin_factor, Internal::paritySInternal(rev_wire_min));
    const __m512 imag_sin_parity1 =
        _mm512_mul_ps(imag_sin_parity0, _mm512_set1_ps(-1.0L));

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m512 v0 = _mm512_load_ps(arr + i0);
        const __m512 v1 = _mm512_load_ps(arr + i1);

        __m512 prod_cos0 = _mm512_mul_ps(real_cos_factor, v0);
        __m512 prod_sin0 = _mm512_mul_ps(v0, imag_sin_parity0);

        __m512 prod0 =
            _mm512_add_ps(prod_cos0, _mm512_permute_ps(prod_sin0, 0B10110001));

        __m512 prod_cos1 = _mm512_mul_ps(real_cos_factor, v1);
        __m512 prod_sin1 = _mm512_mul_ps(v1, imag_sin_parity1);

        __m512 prod1 =
            _mm512_add_ps(prod_cos1, _mm512_permute_ps(prod_sin1, 0B10110001));

        _mm512_store_ps(arr + i0, prod0);
        _mm512_store_ps(arr + i1, prod1);
    }
}

template <class ParamT>
void applyIsingZZFloatExternalExternal(std::complex<float> *arr,
                                       const size_t num_qubits,
                                       const size_t rev_wire0,
                                       const size_t rev_wire1, bool inverse,
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
    const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));

    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<float>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        __m512 v = _mm512_load_ps(arr + i00); // 00
        __m512 prod_cos = _mm512_mul_ps(real_cos_factor, v);
        __m512 prod_isin = Util::productImagS(v, _mm512_set1_ps(isin));
        _mm512_store_ps(arr + i00, _mm512_add_ps(prod_cos, prod_isin));

        v = _mm512_load_ps(arr + i01); // 01
        prod_cos = _mm512_mul_ps(real_cos_factor, v);
        prod_isin = Util::productImagS(v, _mm512_set1_ps(-isin));
        _mm512_store_ps(arr + i01, _mm512_add_ps(prod_cos, prod_isin));

        v = _mm512_load_ps(arr + i10); // 10
        prod_cos = _mm512_mul_ps(real_cos_factor, v);
        prod_isin = Util::productImagS(v, _mm512_set1_ps(-isin));
        _mm512_store_ps(arr + i10, _mm512_add_ps(prod_cos, prod_isin));

        v = _mm512_load_ps(arr + i11); // 11
        prod_cos = _mm512_mul_ps(real_cos_factor, v);
        prod_isin = Util::productImagS(v, _mm512_set1_ps(isin));
        _mm512_store_ps(arr + i11, _mm512_add_ps(prod_cos, prod_isin));
    }
}

template <class ParamT>
void applyIsingZZDoubleInternalInternal(std::complex<double> *arr,
                                        size_t num_qubits,
                                        [[maybe_unused]] size_t rev_wire0,
                                        [[maybe_unused]] size_t rev_wire1,
                                        bool inverse, ParamT angle) {

    // Only rev_wires = (0, 1) is allowed

    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m512d parity =
        _mm512_setr_pd(1.0L, 1.0L, -1.0L, -1.0L, -1.0L, -1.0L, 1.0L, 1.0L);
    const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
    const __m512d imag_sin_factor =
        _mm512_set_pd(-isin, isin, -isin, isin, -isin, isin, -isin, isin);
    const __m512d imag_sin_parity = _mm512_mul_pd(imag_sin_factor, parity);

    for (size_t n = 0; n < exp2(num_qubits);
         n += step_for_complex_precision<double>) {
        __m512d coeffs = _mm512_load_pd(arr + n);
        __m512d prod_cos = _mm512_mul_pd(real_cos_factor, coeffs);

        __m512d prod_sin = _mm512_mul_pd(coeffs, imag_sin_parity);

        __m512d prod =
            _mm512_add_pd(prod_cos, _mm512_permutex_pd(prod_sin, 0B10110001));
        _mm512_store_pd(arr + n, prod);
    }
}
template <class ParamT>
static void
applyIsingZZDoubleInternalExternal(std::complex<double> *arr, size_t num_qubits,
                                   size_t rev_wire0, size_t rev_wire1,
                                   bool inverse, ParamT angle) {
    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t max_rev_wire_shift = (static_cast<size_t>(1U) << rev_wire_max);
    const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
    const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
    const __m512d imag_sin_factor =
        _mm512_set_pd(-isin, isin, -isin, isin, -isin, isin, -isin, isin);

    const __m512d imag_sin_parity0 =
        _mm512_mul_pd(imag_sin_factor, Internal::parityDInternal(rev_wire_min));
    const __m512d imag_sin_parity1 =
        _mm512_mul_pd(imag_sin_parity0, _mm512_set1_pd(-1.0L));

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m512d v0 = _mm512_load_pd(arr + i0);
        const __m512d v1 = _mm512_load_pd(arr + i1);

        __m512d prod_cos0 = _mm512_mul_pd(real_cos_factor, v0);
        __m512d prod_sin0 = _mm512_mul_pd(v0, imag_sin_parity0);

        __m512d prod0 =
            _mm512_add_pd(prod_cos0, _mm512_permutex_pd(prod_sin0, 0B10110001));

        __m512d prod_cos1 = _mm512_mul_pd(real_cos_factor, v1);
        __m512d prod_sin1 = _mm512_mul_pd(v1, imag_sin_parity1);

        __m512d prod1 =
            _mm512_add_pd(prod_cos1, _mm512_permutex_pd(prod_sin1, 0B10110001));

        _mm512_store_pd(arr + i0, prod0);
        _mm512_store_pd(arr + i1, prod1);
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
    const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));

    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<double>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const __m512d v00 = _mm512_load_pd(arr + i00); // 00
        const __m512d v01 = _mm512_load_pd(arr + i01); // 01
        const __m512d v10 = _mm512_load_pd(arr + i10); // 10
        const __m512d v11 = _mm512_load_pd(arr + i11); // 11

        const __m512d prod_cos00 = _mm512_mul_pd(real_cos_factor, v00);
        const __m512d prod_isin00 =
            Util::productImagD(v00, _mm512_set1_pd(isin));

        const __m512d prod_cos01 = _mm512_mul_pd(real_cos_factor, v01);
        const __m512d prod_isin01 =
            Util::productImagD(v01, _mm512_set1_pd(-isin));

        const __m512d prod_cos10 = _mm512_mul_pd(real_cos_factor, v10);
        const __m512d prod_isin10 =
            Util::productImagD(v10, _mm512_set1_pd(-isin));

        const __m512d prod_cos11 = _mm512_mul_pd(real_cos_factor, v11);
        const __m512d prod_isin11 =
            Util::productImagD(v11, _mm512_set1_pd(isin));

        _mm512_store_pd(arr + i00, _mm512_add_pd(prod_cos00, prod_isin00));
        _mm512_store_pd(arr + i01, _mm512_add_pd(prod_cos01, prod_isin01));
        _mm512_store_pd(arr + i10, _mm512_add_pd(prod_cos10, prod_isin10));
        _mm512_store_pd(arr + i11, _mm512_add_pd(prod_cos11, prod_isin11));
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX512
