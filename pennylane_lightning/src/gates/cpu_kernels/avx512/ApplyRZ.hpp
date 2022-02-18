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
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
template <class ParamT>
void applyRZFloatInternal(std::complex<float> *arr, const size_t num_qubits,
                          const size_t rev_wire, [[maybe_unused]] bool inverse,
                          ParamT angle) {
    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

    const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
    const __m512 imag_sin_factor =
        _mm512_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin, -isin,
                      isin, -isin, isin, -isin, isin, -isin, isin);

    const __m512 imag_sin_parity =
        _mm512_mul_ps(imag_sin_factor, Internal::paritySInternal(rev_wire));

    for (size_t n = 0; n < (1U << num_qubits);
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
void applyRZFloatExternal(std::complex<float> *arr, const size_t num_qubits,
                          const size_t rev_wire, [[maybe_unused]] bool inverse,
                          ParamT angle) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
    const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m512 v0 = _mm512_load_ps(arr + i0);
        const __m512 v1 = _mm512_load_ps(arr + i1);

        const auto v0_cos = _mm512_mul_ps(v0, real_cos_factor);
        const auto v0_isin = Util::productImagS(v0, _mm512_set1_ps(isin));

        const auto v1_cos = _mm512_mul_ps(v1, real_cos_factor);
        const auto v1_isin = Util::productImagS(v1, _mm512_set1_ps(-isin));

        _mm512_store_ps(arr + i0, _mm512_add_ps(v0_cos, v0_isin));
        _mm512_store_ps(arr + i1, _mm512_add_ps(v1_cos, v1_isin));
    }
}

template <class ParamT>
void applyRZDoubleInternal(std::complex<double> *arr, const size_t num_qubits,
                           const size_t rev_wire, [[maybe_unused]] bool inverse,
                           ParamT angle) {
    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
    const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
    const __m512d imag_sin_factor =
        _mm512_set_pd(-isin, isin, -isin, isin, -isin, isin, -isin, isin);

    const __m512d imag_sin_parity =
        _mm512_mul_pd(imag_sin_factor, Internal::parityDInternal(rev_wire));

    for (size_t n = 0; n < (1U << num_qubits);
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
void applyRZDoubleExternal(std::complex<double> *arr, const size_t num_qubits,
                           const size_t rev_wire, [[maybe_unused]] bool inverse,
                           ParamT angle) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
    const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m512d v0 = _mm512_load_pd(arr + i0);
        const __m512d v1 = _mm512_load_pd(arr + i1);

        const auto v0_cos = _mm512_mul_pd(v0, real_cos_factor);
        const auto v0_isin = Util::productImagD(v0, _mm512_set1_pd(isin));

        const auto v1_cos = _mm512_mul_pd(v1, real_cos_factor);
        const auto v1_isin = Util::productImagD(v1, _mm512_set1_pd(-isin));

        _mm512_store_pd(arr + i0, _mm512_add_pd(v0_cos, v0_isin));
        _mm512_store_pd(arr + i1, _mm512_add_pd(v1_cos, v1_isin));
    }
}
} // namespace Pennylane::Gates::AVX512
