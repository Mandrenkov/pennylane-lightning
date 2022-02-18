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
 * Defines PauliZ gate
 */
#pragma once
#include "AVX512Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX512 {
/// @cond DEV
inline void applyPauliZFloatInternal(std::complex<float> *arr,
                                     const size_t num_qubits,
                                     const size_t rev_wire) {
    __m512 factor;
    // clang-format off
    switch (rev_wire) {
    case 0:
        factor = _mm512_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, -1.0F, -1.0F);
        break;
    case 1:
        factor = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, -1.0F, -1.0F, -1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, -1.0F, -1.0F, -1.0F);
        break;
    case 2:
        factor = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, -1.0F, -1.0F, -1.0F,
                                -1.0F, -1.0F, -1.0F, -1.0F);
        break;
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<float>) {
        __m512 v = _mm512_load_ps(arr + k);
        v = _mm512_mul_ps(v, factor);
        _mm512_store_ps(arr + k, v);
    }
}
inline void applyPauliZFloatExternal(std::complex<float> *arr,
                                     const size_t num_qubits,
                                     const size_t rev_wire) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        __m512 v1 = _mm512_load_ps(arr + i1);
        v1 = _mm512_mul_ps(v1, _mm512_set1_ps(-1.0F));
        _mm512_store_ps(arr + i1, v1);
    }
}

inline void applyPauliZDoubleInternal(std::complex<double> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire) {
    __m512d factor;
    switch (rev_wire) {
    case 0:
        factor =
            _mm512_setr_pd(1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F);
        break;
    case 1:
        factor =
            _mm512_setr_pd(1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F);
        break;
    default:
        PL_UNREACHABLE;
    }
    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<double>) {
        __m512d v = _mm512_load_pd(arr + k);
        v = _mm512_mul_pd(v, factor);
        _mm512_store_pd(arr + k, v);
    }
}

inline void applyPauliZDoubleExternal(std::complex<double> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        __m512d v1 = _mm512_load_pd(arr + i1);
        v1 = _mm512_mul_pd(v1, _mm512_set1_pd(-1.0F));
        _mm512_store_pd(arr + i1, v1);
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX512
