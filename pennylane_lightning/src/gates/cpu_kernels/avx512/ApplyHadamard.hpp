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
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
/// @cond DEV
template <size_t rev_wire>
void applyHadamardFloatInternal(std::complex<float> *arr,
                                       const size_t num_qubits) {
    constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<float>();

    const __m512 mat_offdiag = _mm512_set1_ps(isqrt2);

    __m512 mat_diag;
    if constexpr (rev_wire == 0) {
        mat_diag = _mm512_setr4_ps(isqrt2, isqrt2, -isqrt2, -isqrt2);

    } else if (rev_wire == 1) {
        mat_diag = _mm512_setr_ps(
            isqrt2, isqrt2, isqrt2, isqrt2,
            -isqrt2, -isqrt2, -isqrt2, -isqrt2,
            isqrt2, isqrt2, isqrt2, isqrt2,
            -isqrt2, -isqrt2, -isqrt2, -isqrt2
        );
    } else { // rev_wire == 2
        mat_diag = _mm512_setr_ps(
            isqrt2, isqrt2, isqrt2, isqrt2,
            isqrt2, isqrt2, isqrt2, isqrt2,
            -isqrt2, -isqrt2, -isqrt2, -isqrt2,
            -isqrt2, -isqrt2, -isqrt2, -isqrt2
        );
    }

    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<float>) {
        __m512 v = _mm512_load_ps(arr + k);
        if constexpr (rev_wire == 0) {
            const __m512 w_diag = _mm512_mul_ps(v, mat_diag);
            const __m512 v_off = _mm512_permute_ps(v, 0B01'00'11'10);
            const __m512 w_offdiag = _mm512_mul_ps(v_off, mat_offdiag);

            v = _mm512_add_ps(w_diag, w_offdiag);
        } else if (rev_wire == 1) {
            const __m512 w_diag = _mm512_mul_ps(v, mat_diag);
            const __m512 v_off = _mm512_permutexvar_ps(
                _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0,
                                 7, 6, 5, 4),
                v);
            const __m512 w_offdiag = _mm512_mul_ps(v_off, mat_offdiag);

            v = _mm512_add_ps(w_diag, w_offdiag);
        } else { // rev_wire == 2
            const __m512 w_diag = _mm512_mul_ps(v, mat_diag);
            const __m512 v_off = _mm512_permutexvar_ps(
                _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11,
                                 10, 9, 8),
                v);

            const __m512 w_offdiag = _mm512_mul_ps(v_off, mat_offdiag);
            v = _mm512_add_ps(w_diag, w_offdiag);
        }
        _mm512_store_ps(arr + k, v);
    }
}
void applyHadamardFloatExternal(std::complex<float> *arr,
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

        const __m512 v0 = _mm512_load_ps(arr + i0);
        const __m512 v1 = _mm512_load_ps(arr + i1);

        // w0 = u00 * v0
        __m512 w0 = _mm512_mul_ps(v0, _mm512_set1_ps(isqrt2));

        // w0 +=  u01 * v1
        w0 = _mm512_add_ps(w0, _mm512_mul_ps(v1, _mm512_set1_ps(isqrt2)));

        // w1 = u11 * v1
        __m512 w1 = _mm512_mul_ps(v1, _mm512_set1_ps(-isqrt2));

        // w1 +=  u10 * v0
        w1 = _mm512_add_ps(w1, _mm512_mul_ps(v0, _mm512_set1_ps(isqrt2)));

        _mm512_store_ps(arr + i0, w0);
        _mm512_store_ps(arr + i1, w1);
    }
}

template <size_t rev_wire>
void applyHadamardDoubleInternal(std::complex<double> *arr,
                                        const size_t num_qubits) {
    constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<double>();

    const __m512d mat_offdiag = _mm512_set1_pd(isqrt2);

    __m512d mat_diag;
    if constexpr (rev_wire == 0) {
        mat_diag = _mm512_setr4_pd(isqrt2, isqrt2, -isqrt2, -isqrt2);
    } else if (rev_wire == 1) {
        mat_diag = _mm512_setr_pd(
            isqrt2, isqrt2, isqrt2, isqrt2,
            -isqrt2, -isqrt2, -isqrt2, -isqrt2
        );
    } 
    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<double>) {
        __m512d v = _mm512_load_pd(arr + k);
        if constexpr (rev_wire == 0) {
            const __m512d w_diag = _mm512_mul_pd(v, mat_diag);
            const __m512d v_off = _mm512_permutex_pd(v, 0B01'00'11'10);
            const __m512d w_offdiag = _mm512_mul_pd(v_off, mat_offdiag);

            v = _mm512_add_pd(w_diag, w_offdiag);
        } else if (rev_wire == 1) {
            const __m512d w_diag = _mm512_mul_pd(v, mat_diag);
            const __m512d v_off = _mm512_permutexvar_pd(
                _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4),
                v);
            const __m512d w_offdiag = _mm512_mul_pd(v_off, mat_offdiag);

            v = _mm512_add_pd(w_diag, w_offdiag);
        }  
        _mm512_store_pd(arr + k, v);
    }
}
void applyHadamardDoubleExternal(std::complex<double> *arr,
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

        const __m512d v0 = _mm512_load_pd(arr + i0);
        const __m512d v1 = _mm512_load_pd(arr + i1);

        // w0 = u00 * v0
        __m512d w0 = _mm512_mul_pd(v0, _mm512_set1_pd(isqrt2));

        // w0 +=  u01 * v1
        w0 = _mm512_add_pd(w0, _mm512_mul_pd(v1, _mm512_set1_pd(isqrt2)));

        // w1 = u11 * v1
        __m512d w1 = _mm512_mul_pd(v1, _mm512_set1_pd(-isqrt2));

        // w1 +=  u10 * v0
        w1 = _mm512_add_pd(w1, _mm512_mul_pd(v0, _mm512_set1_pd(isqrt2)));

        _mm512_store_pd(arr + i0, w0);
        _mm512_store_pd(arr + i1, w1);
    }
}

/// @endcond
} // namespace Pennylane::Gates::AVX512
