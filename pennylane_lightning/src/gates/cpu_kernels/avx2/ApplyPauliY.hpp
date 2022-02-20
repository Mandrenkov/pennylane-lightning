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
#include "AVX2Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX2 {
/// @cond DEV

template <size_t rev_wire> void applyPauliYFloatInternalOp(__m256 &v) {
    if constexpr (rev_wire == 0) {
        const auto factor =
            _mm256_setr_ps(1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F,
                           1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F);
        v = _mm256_permute_ps(v, 0B00011011);
        v = _mm256_mul_ps(v, factor);
    } else if (rev_wire == 1) {
        const auto factor =
            _mm256_setr_ps(1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F,
                           1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
        const auto shuffle_idx = _mm256_set_epi32(10, 11, 8, 9, 14, 15, 12, 13,
                                                  2, 3, 0, 1, 6, 7, 4, 5);
        v = _mm256_permutexvar_ps(shuffle_idx, v);
        v = _mm256_mul_ps(v, factor);
    } else if (rev_wire == 2) {
        const auto factor =
            _mm256_setr_ps(1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F,
                           -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
        const auto shuffle_idx = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1, 14,
                                                  15, 12, 13, 10, 11, 8, 9);
        v = _mm256_permutexvar_ps(shuffle_idx, v);
        v = _mm256_mul_ps(v, factor);
    }
}

template <size_t rev_wire>
void applyPauliYFloatInternal(std::complex<float> *arr,
                              const size_t num_qubits) {
    for (size_t k = 0; k < (1U << num_qubits);
         k += step_for_complex_precision<float>) {
        __m256 v = _mm256_load_ps(arr + k);
        applyPauliYFloatInternalOp<rev_wire>(v);
        _mm256_store_ps(arr + k, v);
    }
}
inline void applyPauliYFloatExternal(std::complex<float> *arr,
                                     const size_t num_qubits,
                                     const size_t rev_wire) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    const auto minus_imag_prod = Util::ProdPureImag<float>(-1.0);
    const auto plus_imag_prod = Util::ProdPureImag<float>(1.0);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256 v0 = _mm256_load_ps(arr + i0);
        const __m256 v1 = _mm256_load_ps(arr + i1);
        _mm256_store_ps(arr + i0, minus_imag_prod.product(v1));
        _mm256_store_ps(arr + i1, plus_imag_prod.product(v0));
    }
}

template <size_t rev_wire> static void applyPauliYDoubleInternalOp(__m256d &v) {
    if constexpr (rev_wire == 0) {
        const auto factor =
            _mm256_setr_pd(1.0L, -1.0L, -1.0L, 1.0L, 1.0L, -1.0L, -1.0L, 1.0L);
        v = _mm256_mul_pd(v, factor);
        const auto shuffle_idx = _mm256_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
        v = _mm256_permutexvar_pd(shuffle_idx, v);
    } else if (rev_wire == 1) {
        const auto factor =
            _mm256_setr_pd(1.0L, -1.0L, 1.0L, -1.0L, -1.0L, 1.0L, -1.0L, 1.0L);
        const auto shuffle_idx = _mm256_set_epi64(2, 3, 0, 1, 6, 7, 4, 5);
        v = _mm256_permutexvar_pd(shuffle_idx, v);
        v = _mm256_mul_pd(v, factor);
    }
}

template <size_t rev_wire>
void applyPauliYDoubleInternal(std::complex<double> *arr,
                               const size_t num_qubits) {
    for (size_t k = 0; k < (1U << num_qubits); k += 4) {
        __m256d v = _mm256_load_pd(arr + k);
        applyPauliYDoubleInternalOp<rev_wire>(v);
        _mm256_store_pd(arr + k, v);
    }
}
inline void applyPauliYDoubleExternal(std::complex<double> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    const auto minus_imag_prod = Util::ProdPureImag<double>(-1.0);
    const auto plus_imag_prod = Util::ProdPureImag<double>(1.0);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256d v0 = _mm256_load_pd(arr + i0);
        const __m256d v1 = _mm256_load_pd(arr + i1);

        _mm256_store_pd(arr + i0, minus_imag_prod.product(v1));
        _mm256_store_pd(arr + i1, plus_imag_prod.product(v0));
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX2
