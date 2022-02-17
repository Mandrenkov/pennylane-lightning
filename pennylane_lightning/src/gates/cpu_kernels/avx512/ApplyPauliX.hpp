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
 * Defines PauliX gate
 */
#pragma once
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
/// @cond DEV
template <size_t rev_wire>
inline void applyPauliXFloatInternalOp(__m512 &v) {
    if constexpr (rev_wire == 0) {
        v = _mm512_permute_ps(v, 0B01001110);
    } else if (rev_wire == 1) {
        const auto shuffle_idx = _mm512_set_epi32(
            11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
        v = _mm512_permutexvar_ps(shuffle_idx, v);
    } else if (rev_wire == 2) {
        const auto shuffle_idx = _mm512_set_epi32(
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
        v = _mm512_permutexvar_ps(shuffle_idx, v);
    }
}
template <size_t rev_wire>
void applyPauliXFloatInternal(std::complex<float> *arr,
                                     const size_t num_qubits) {
    for (size_t k = 0; k < (1U << num_qubits);
         k += step_for_complex_precision<float>) {
        __m512 v = _mm512_load_ps(arr + k);
        applyPauliXFloatInternalOp<rev_wire>(v);
        _mm512_store_ps(arr + k, v);
    }
}

void applyPauliXFloatExternal(std::complex<float> *arr,
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
        _mm512_store_ps(arr + i0, v1);
        _mm512_store_ps(arr + i1, v0);
    }
}

template <size_t rev_wire>
inline static void applyPauliXDoubleInternalOp(__m512d &v) {
    if constexpr (rev_wire == 0) {
        const auto shuffle_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        v = _mm512_permutexvar_pd(shuffle_idx, v);
    } else if (rev_wire == 1) {
        const auto shuffle_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        v = _mm512_permutexvar_pd(shuffle_idx, v);
    }
}

template <size_t rev_wire>
static void applyPauliXDoubleInternal(std::complex<double> *arr,
                                      const size_t num_qubits) {
    for (size_t k = 0; k < (1U << num_qubits);
         k += step_for_complex_precision<double>) {
        __m512d v = _mm512_load_pd(arr + k);
        applyPauliXDoubleInternalOp<rev_wire>(v);
        _mm512_store_pd(arr + k, v);
    }
}

void applyPauliXDoubleExternal(std::complex<double> *arr,
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
        _mm512_store_pd(arr + i0, v1);
        _mm512_store_pd(arr + i1, v0);
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX512
