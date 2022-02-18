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
 * Defines PauliY gate
 */
#pragma once
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
/// @cond DEV

template <size_t rev_wire> void applyPauliYFloatInternalOp(__m512 &v) {
    if constexpr (rev_wire == 0) {
        const auto factor =
            _mm512_setr_ps(1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F,
                           1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F);
        v = _mm512_permute_ps(v, 0B00011011);
        v = _mm512_mul_ps(v, factor);
    } else if (rev_wire == 1) {
        const auto factor =
            _mm512_setr_ps(1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F,
                           1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
        const auto shuffle_idx = _mm512_set_epi32(10, 11, 8, 9, 14, 15, 12, 13,
                                                  2, 3, 0, 1, 6, 7, 4, 5);
        v = _mm512_permutexvar_ps(shuffle_idx, v);
        v = _mm512_mul_ps(v, factor);
    } else if (rev_wire == 2) {
        const auto factor =
            _mm512_setr_ps(1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F,
                           -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
        const auto shuffle_idx = _mm512_set_epi32(6, 7, 4, 5, 2, 3, 0, 1, 14,
                                                  15, 12, 13, 10, 11, 8, 9);
        v = _mm512_permutexvar_ps(shuffle_idx, v);
        v = _mm512_mul_ps(v, factor);
    }
}

template <size_t rev_wire>
void applyPauliYFloatInternal(std::complex<float> *arr,
                              const size_t num_qubits) {
    for (size_t k = 0; k < (1U << num_qubits);
         k += step_for_complex_precision<float>) {
        __m512 v = _mm512_load_ps(arr + k);
        applyPauliYFloatInternalOp<rev_wire>(v);
        _mm512_store_ps(arr + k, v);
    }
}
inline void applyPauliYFloatExternal(std::complex<float> *arr,
                                     const size_t num_qubits,
                                     const size_t rev_wire) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m512 v0 = _mm512_load_ps(arr + i0);
        const __m512 v1 = _mm512_load_ps(arr + i1);
        _mm512_store_ps(arr + i0,
                        Util::productImagS(v1, _mm512_set1_ps(-1.0F)));
        _mm512_store_ps(arr + i1, Util::productImagS(v0));
    }
}

template <size_t rev_wire> static void applyPauliYDoubleInternalOp(__m512d &v) {
    if constexpr (rev_wire == 0) {
        const auto factor =
            _mm512_setr_pd(1.0L, -1.0L, -1.0L, 1.0L, 1.0L, -1.0L, -1.0L, 1.0L);
        v = _mm512_mul_pd(v, factor);
        const auto shuffle_idx = _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
        v = _mm512_permutexvar_pd(shuffle_idx, v);
    } else if (rev_wire == 1) {
        const auto factor =
            _mm512_setr_pd(1.0L, -1.0L, 1.0L, -1.0L, -1.0L, 1.0L, -1.0L, 1.0L);
        const auto shuffle_idx = _mm512_set_epi64(2, 3, 0, 1, 6, 7, 4, 5);
        v = _mm512_permutexvar_pd(shuffle_idx, v);
        v = _mm512_mul_pd(v, factor);
    }
}

template <size_t rev_wire>
void applyPauliYDoubleInternal(std::complex<double> *arr,
                               const size_t num_qubits) {
    for (size_t k = 0; k < (1U << num_qubits); k += 4) {
        __m512d v = _mm512_load_pd(arr + k);
        applyPauliYDoubleInternalOp<rev_wire>(v);
        _mm512_store_pd(arr + k, v);
    }
}
inline void applyPauliYDoubleExternal(std::complex<double> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m512d v0 = _mm512_load_pd(arr + i0);
        const __m512d v1 = _mm512_load_pd(arr + i1);

        _mm512_store_pd(arr + i0,
                        Util::productImagD(v1, _mm512_set1_pd(-1.0L)));
        _mm512_store_pd(arr + i1, Util::productImagD(v0));
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX512
