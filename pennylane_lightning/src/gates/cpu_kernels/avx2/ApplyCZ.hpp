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
#include "AVX2Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX2 {
/// @cond DEV
inline void applyCZFloatInternalInternal(std::complex<float> *arr,
                                         size_t num_qubits) {
    __m256 parity;

    // rev_wires must be (0, 1)
    // clang-format off
    parity = _mm256_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                            1.0F, 1.0F, -1.0F, -1.0F);
    // clang-format on
    auto* p = reinterpret_cast<float*>(arr);
    for (size_t n = 0; n < exp2(num_qubits);
         n += step_for_complex_precision<float>) {
        __m256 v = _mm256_load_ps(p + 2*n);
        _mm256_store_ps(p + 2*n, _mm256_mul_ps(v, parity));
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

    const __m256 parity = Internal::paritySInternal(rev_wire_min);

    auto* p = reinterpret_cast<float*>(arr);

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m256 v1 = _mm256_load_ps(p + 2*i1);

        _mm256_store_ps(p + 2*i1, _mm256_mul_ps(v1, parity));
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

    auto* p = reinterpret_cast<float*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<float>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        __m256 v = _mm256_load_ps(p + 2*i11); // 11
        v = _mm256_mul_ps(v, _mm256_set1_ps(-1.0F));
        _mm256_store_ps(p + 2*i11, v);
    }
}

inline void applyCZDoubleInternalExternal(std::complex<double> *arr,
                                          size_t num_qubits,
                                          size_t rev_wire_max) {
    // rev_wire_min must be 0
    const size_t max_rev_wire_shift = (static_cast<size_t>(1U) << rev_wire_max);
    const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
    const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

    const __m256d parity = Internal::parityDInternal();

    auto* p = reinterpret_cast<double*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 =
            ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
        const size_t i1 = i0 | max_rev_wire_shift;

        const __m256d v1 = _mm256_load_pd(p + 2*i1);

        _mm256_store_pd(p + 2*i1, _mm256_mul_pd(v1, parity));
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

    auto* p = reinterpret_cast<double*>(arr);
    for (size_t k = 0; k < exp2(num_qubits - 2);
         k += step_for_complex_precision<double>) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const __m256d v11 = _mm256_load_pd(p + 2*i11); // 11

        _mm256_store_pd(p + 2*i11, _mm256_mul_pd(v11, _mm256_set1_pd(-1.0L)));
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX2
