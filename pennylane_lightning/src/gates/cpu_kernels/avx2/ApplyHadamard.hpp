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
 * Defines Hadamard gate
 */
#pragma once
#include "AVX2Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX2 {
/// @cond DEV
template <size_t rev_wire>
void applyHadamardFloatInternal(std::complex<float> *arr,
                                const size_t num_qubits) {
    constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<float>();

    const __m256 mat_offdiag = _mm256_set1_ps(isqrt2);

    __m256 mat_diag;
    if constexpr (rev_wire == 0) {
        mat_diag = _mm256_setr4_ps(isqrt2, isqrt2, -isqrt2, -isqrt2);

    } else if (rev_wire == 1) {
        mat_diag = _mm256_setr_ps(
            isqrt2, isqrt2, isqrt2, isqrt2, -isqrt2, -isqrt2, -isqrt2, -isqrt2,
            isqrt2, isqrt2, isqrt2, isqrt2, -isqrt2, -isqrt2, -isqrt2, -isqrt2);
    } else { // rev_wire == 2
        mat_diag =
            _mm256_setr_ps(isqrt2, isqrt2, isqrt2, isqrt2, isqrt2, isqrt2,
                           isqrt2, isqrt2, -isqrt2, -isqrt2, -isqrt2, -isqrt2,
                           -isqrt2, -isqrt2, -isqrt2, -isqrt2);
    }

    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<float>) {
        __m256 v = _mm256_load_ps(arr + k);
        if constexpr (rev_wire == 0) {
            const __m256 w_diag = _mm256_mul_ps(v, mat_diag);
            const __m256 v_off = _mm256_permute_ps(v, 0B01'00'11'10);
            const __m256 w_offdiag = _mm256_mul_ps(v_off, mat_offdiag);

            v = _mm256_add_ps(w_diag, w_offdiag);
        } else if (rev_wire == 1) {
            const __m256 w_diag = _mm256_mul_ps(v, mat_diag);
            const __m256 v_off = _mm256_permutexvar_ps(
                _mm256_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6,
                                 5, 4),
                v);
            const __m256 w_offdiag = _mm256_mul_ps(v_off, mat_offdiag);

            v = _mm256_add_ps(w_diag, w_offdiag);
        } else { // rev_wire == 2
            const __m256 w_diag = _mm256_mul_ps(v, mat_diag);
            const __m256 v_off = _mm256_permutexvar_ps(
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10,
                                 9, 8),
                v);

            const __m256 w_offdiag = _mm256_mul_ps(v_off, mat_offdiag);
            v = _mm256_add_ps(w_diag, w_offdiag);
        }
        _mm256_store_ps(arr + k, v);
    }
}
inline void applyHadamardFloatExternal(std::complex<float> *arr,
                                       const size_t num_qubits,
                                       const size_t rev_wire) {
    constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<float>();

    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256 v0 = _mm256_load_ps(arr + i0);
        const __m256 v1 = _mm256_load_ps(arr + i1);

        // w0 = u00 * v0
        __m256 w0 = _mm256_mul_ps(v0, _mm256_set1_ps(isqrt2));

        // w0 +=  u01 * v1
        w0 = _mm256_add_ps(w0, _mm256_mul_ps(v1, _mm256_set1_ps(isqrt2)));

        // w1 = u11 * v1
        __m256 w1 = _mm256_mul_ps(v1, _mm256_set1_ps(-isqrt2));

        // w1 +=  u10 * v0
        w1 = _mm256_add_ps(w1, _mm256_mul_ps(v0, _mm256_set1_ps(isqrt2)));

        _mm256_store_ps(arr + i0, w0);
        _mm256_store_ps(arr + i1, w1);
    }
}

template <size_t rev_wire>
void applyHadamardDoubleInternal(std::complex<double> *arr,
                                 const size_t num_qubits) {
    constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<double>();

    const __m256d mat_offdiag = _mm256_set1_pd(isqrt2);

    __m256d mat_diag;
    if constexpr (rev_wire == 0) {
        mat_diag = _mm256_setr4_pd(isqrt2, isqrt2, -isqrt2, -isqrt2);
    } else if (rev_wire == 1) {
        mat_diag = _mm256_setr_pd(isqrt2, isqrt2, isqrt2, isqrt2, -isqrt2,
                                  -isqrt2, -isqrt2, -isqrt2);
    }
    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<double>) {
        __m256d v = _mm256_load_pd(arr + k);
        if constexpr (rev_wire == 0) {
            const __m256d w_diag = _mm256_mul_pd(v, mat_diag);
            const __m256d v_off = _mm256_permutex_pd(v, 0B01'00'11'10);
            const __m256d w_offdiag = _mm256_mul_pd(v_off, mat_offdiag);

            v = _mm256_add_pd(w_diag, w_offdiag);
        } else if (rev_wire == 1) {
            const __m256d w_diag = _mm256_mul_pd(v, mat_diag);
            const __m256d v_off = _mm256_permutexvar_pd(
                _mm256_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), v);
            const __m256d w_offdiag = _mm256_mul_pd(v_off, mat_offdiag);

            v = _mm256_add_pd(w_diag, w_offdiag);
        }
        _mm256_store_pd(arr + k, v);
    }
}
inline void applyHadamardDoubleExternal(std::complex<double> *arr,
                                        const size_t num_qubits,
                                        const size_t rev_wire) {
    constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<double>();

    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256d v0 = _mm256_load_pd(arr + i0);
        const __m256d v1 = _mm256_load_pd(arr + i1);

        // w0 = u00 * v0
        __m256d w0 = _mm256_mul_pd(v0, _mm256_set1_pd(isqrt2));

        // w0 +=  u01 * v1
        w0 = _mm256_add_pd(w0, _mm256_mul_pd(v1, _mm256_set1_pd(isqrt2)));

        // w1 = u11 * v1
        __m256d w1 = _mm256_mul_pd(v1, _mm256_set1_pd(-isqrt2));

        // w1 +=  u10 * v0
        w1 = _mm256_add_pd(w1, _mm256_mul_pd(v0, _mm256_set1_pd(isqrt2)));

        _mm256_store_pd(arr + i0, w0);
        _mm256_store_pd(arr + i1, w1);
    }
}

/// @endcond
} // namespace Pennylane::Gates::AVX2
