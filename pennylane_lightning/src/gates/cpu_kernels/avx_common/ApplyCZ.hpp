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
 * Defines CZ gate
 */
#pragma once
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
/// @cond DEV
inline void applyCZFloatInternalInternal(std::complex<float> *arr,
                                         size_t num_qubits, size_t rev_wire0,
                                         size_t rev_wire1) {
    __m512 parity;

    // clang-format off
    switch(rev_wire0 ^ rev_wire1) {
    /* Possible values are (max_rev_wire, min_rev_wire) =
     *     {(1, 0) = 1, (2, 0) = 2, (2, 1) = 3}
     * */
    case 1: // (1,0)
        parity = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F);
        break;
    case 2: // (2,0)
        parity = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F);
        break;
    case 3: // (2,1)
        parity = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, -1.0F, -1.0F, -1.0F);
        break;
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    for (size_t n = 0; n < exp2(num_qubits);
         n += step_for_complex_precision<float>) {
        __m512 v = _mm512_load_ps(arr + n);
        _mm512_store_ps(arr + n, _mm512_mul_ps(v, parity));
    }
}

inline void applyCZFloatInternalExternal(std::complex<float> *arr,
                                         size_t num_qubits, size_t rev_wire0,
                                         size_t rev_wire1) {
    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t max_rev_wire_shift = (static_cast<size_t>(1U) << rev_wire_max);
    const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
    const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

    const __m512 parity = Internal::paritySInternal(rev_wire_min);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m512 v1 = _mm512_load_ps(arr + i1);

        _mm512_store_ps(arr + i1, _mm512_mul_ps(v1, parity));
    }
}

inline void applyCZFloatExternalExternal(std::complex<float> *arr,
                                         const size_t num_qubits,
                                         const size_t rev_wire0,
                                         const size_t rev_wire1) {
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t parity_low = fillTrailingOnes(rev_wire_min);
    const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
    const size_t parity_middle =
        fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<float>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        __m512 v = _mm512_load_ps(arr + i11); // 11
        v = _mm512_mul_ps(v, _mm512_set1_ps(-1.0F));
        _mm512_store_ps(arr + i11, v);
    }
}

inline void applyCZDoubleInternalInternal(std::complex<double> *arr,
                                          size_t num_qubits,
                                          [[maybe_unused]] size_t rev_wire0,
                                          [[maybe_unused]] size_t rev_wire1) {

    // Only rev_wires = (0, 1) is allowed
    __m512d parity =
        _mm512_setr_pd(1.0L, 1.0L, 1.0L, 1.0L, 1.0L, 1.0L, -1.0L, -1.0L);

    for (size_t n = 0; n < exp2(num_qubits);
         n += step_for_complex_precision<double>) {
        __m512d v = _mm512_load_pd(arr + n);
        _mm512_store_pd(arr + n, _mm512_mul_pd(v, parity));
    }
}

inline void applyCZDoubleInternalExternal(std::complex<double> *arr,
                                          size_t num_qubits, size_t rev_wire0,
                                          size_t rev_wire1) {
    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t max_rev_wire_shift = (static_cast<size_t>(1U) << rev_wire_max);
    const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
    const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

    const __m512d parity = Internal::parityDInternal(rev_wire_min);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m512d v1 = _mm512_load_pd(arr + i1);

        _mm512_store_pd(arr + i1, _mm512_mul_pd(v1, parity));
    }
}

inline void applyCZDoubleExternalExternal(std::complex<double> *arr,
                                          const size_t num_qubits,
                                          const size_t rev_wire0,
                                          const size_t rev_wire1) {
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

    const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
    const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

    const size_t parity_low = fillTrailingOnes(rev_wire_min);
    const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
    const size_t parity_middle =
        fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<double>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const __m512d v11 = _mm512_load_pd(arr + i11); // 11

        _mm512_store_pd(arr + i11, _mm512_mul_pd(v11, _mm512_set1_pd(-1.0L)));
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX512
