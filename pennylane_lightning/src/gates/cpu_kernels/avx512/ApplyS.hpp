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
 * Defines S gate
 */
#pragma once
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
/// @cond DEV
template<int rev_wire>
void applySFloatInternal(std::complex<float> *arr,
                                const size_t num_qubits,
                                bool inverse) {
    __m512 factor;
    // clang-format off
    if constexpr (rev_wire == 0) {
        //                       real imag   real  imag
        factor = _mm512_setr_ps(1.0F, 1.0F, -1.0F, 1.0F,
                                1.0F, 1.0F, -1.0F, 1.0F,
                                1.0F, 1.0F, -1.0F, 1.0F,
                                1.0F, 1.0F, -1.0F, 1.0F);
    } else if (rev_wire == 1) {
        factor = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, 1.0F, -1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, 1.0F, -1.0F, 1.0F);
    } else { // rev_wire == 2
        factor = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, 1.0F, -1.0F, 1.0F,
                                -1.0F, 1.0F, -1.0F, 1.0F);
    }
    // clang-format on
    if (inverse) {
        factor = _mm512_mul_ps(factor, _mm512_set1_ps(-1.0F));
    }
    for (size_t k = 0; k < (1U << num_qubits);
            k += step_for_complex_precision<float>) {
        __m512 v = _mm512_load_ps(arr + k);
        if constexpr (rev_wire == 0) {
            v = _mm512_permute_ps(v, 0B10'11'01'00);
            v = _mm512_mul_ps(v, factor);
        } else if (rev_wire == 1) {
            v = _mm512_permutexvar_ps(
                    _mm512_set_epi32(14,15,12,13,
                                     11,10,9,8,
                                     6,7,4,5,
                                     3,2,1,0),
                v);
            v = _mm512_mul_ps(v, factor);
        } else { // rev_wire == 2
            v = _mm512_permutexvar_ps(
                    _mm512_set_epi32(
                        14,15,12,13,
                        10,11,8,9,
                        7,6,5,4,
                        3, 2, 1, 0),
                v);
            v = _mm512_mul_ps(v, factor);
        }
        _mm512_store_ps(arr + k, v);
    }
}

void applySFloatExternal(std::complex<float> *arr,
                                     const size_t num_qubits,
                                     const size_t rev_wire,
                                     bool inverse) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    auto imag_factor = _mm512_load_ps(&Util::ImagFactor<float>::value);
    if (inverse) {
        imag_factor = _mm512_mul_ps(imag_factor, _mm512_set1_ps(-1.0F));
    }
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        __m512 v1 = _mm512_load_ps(arr + i1);
        auto prod_shuffled = _mm512_permute_ps(v1, 0B10110001);
        v1 = _mm512_mul_ps(prod_shuffled, imag_factor);
        _mm512_store_ps(arr + i1, v1);
    }
}

template<size_t rev_wire>
void applySDoubleInternal(std::complex<double> *arr,
                                 const size_t num_qubits,
                                 bool inverse) {
    __m512d factor;
    // clang-format off
    if constexpr (rev_wire == 0) {
        //                       real imag   real  imag
        factor = _mm512_setr_pd(1.0L, 1.0L, -1.0L, 1.0L,
                                1.0L, 1.0L, -1.0L, 1.0L);
    } else if (rev_wire == 1) {
        factor = _mm512_setr_pd(1.0L, 1.0L, 1.0L, 1.0L,
                                -1.0L, 1.0L, -1.0L, 1.0L);
    }
    if (inverse) {
        factor = _mm512_mul_pd(factor, _mm512_set1_pd(-1.0L));
    }
    for (size_t k = 0; k < (1U << num_qubits);
         k += step_for_complex_precision<double>) {
        __m512d v = _mm512_load_pd(arr + k);
        if constexpr (rev_wire == 0) {
            v = _mm512_permutexvar_pd(
                    _mm512_set_epi64(6, 7, 5, 4,
                                     2, 3, 1, 0),
                v);
            v = _mm512_mul_pd(v, factor);
        } else if (rev_wire == 1) {
            v = _mm512_permutexvar_pd(
                    _mm512_set_epi64(6, 7, 4, 5,
                                     3, 2, 1, 0),
                v);
            v = _mm512_mul_pd(v, factor);
        } 
        _mm512_store_pd(arr + k, v);
    }
}

void applySDoubleExternal(std::complex<double> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire,
                                      bool inverse) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    auto imag_factor = _mm512_load_pd(&Util::ImagFactor<double>::value);
    if (inverse) {
        imag_factor = _mm512_mul_pd(imag_factor, _mm512_set1_pd(-1.0L));
    }

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        __m512d v1 = _mm512_load_pd(arr + i1);
        __m512d prod_shuffled = _mm512_permutex_pd(v1, 0B10110001);

        v1 = _mm512_mul_pd(prod_shuffled, imag_factor);
        _mm512_store_pd(arr + i1, v1);
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX512
