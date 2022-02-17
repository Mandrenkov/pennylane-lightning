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
 * Defines common utility functions for AVX512
 */
#pragma once
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::Gates::AVX512::Util {

template <class PrecisionT> struct AVX512Intrinsic;

template <> struct AVX512Intrinsic<float> { using Type = __m512; };

template <> struct AVX512Intrinsic<double> { using Type = __m512d; };

template <class PrecisionT>
using AVX512IntrinsicType = typename AVX512Intrinsic<PrecisionT>::Type;

template <typename T> struct ImagFactor {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "T must be float or double.");
};

template <> struct ImagFactor<float> {
    // NOLINTNEXTLINE(hicpp-avoid-c-arrays)
    alignas(64) constexpr static float value[16] = {
        -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F,
        -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F,
    };
};

template <> struct ImagFactor<double> {
    // NOLINTNEXTLINE(hicpp-avoid-c-arrays)
    alignas(64) constexpr static double value[8] = {
        -1.0L, 1.0L, -1.0L, 1.0L, -1.0L, 1.0L, -1.0L, 1.0L,
    };
};

/**
 * @brief Calculate val * 1j * factor
 *
 * @param val Complex values arranged in [i7, r7, ..., i0, r0] where
 * each complex values are r0 + 1j*i0, ...
 * @param imag_val Value to product. We product 1j*imag_val to val.
 */
inline __m512 productImagS(__m512 val, __m512 factor) {
    __m512 prod_shuffled =
        _mm512_permute_ps(_mm512_mul_ps(val, factor), 0B10110001);
    return _mm512_mul_ps(prod_shuffled,
                         _mm512_load_ps(&ImagFactor<float>::value));
}

/**
 * @brief Calculate val * 1j
 *
 * @param val Complex values arranged in [i7, r7, ..., i0, r0] where
 * each complex values are r0 + 1j*i0, ...
 */
inline __m512 productImagS(__m512 val) {
    __m512 prod_shuffled = _mm512_permute_ps(val, 0B10110001);
    return _mm512_mul_ps(prod_shuffled,
                         _mm512_load_ps(&ImagFactor<float>::value));
}

/**
 * @brief Calculate val * 1j * factor
 *
 * @param val Complex values arranged in [i3, r3, ..., i0, r0] where
 * each complex values are r0 + 1j*i0, ...
 * @param imag_val Value to product. We product 1j*imag_val to val.
 */
inline __m512d productImagD(__m512d val, __m512d factor) {
    __m512d prod_shuffled =
        _mm512_permutex_pd(_mm512_mul_pd(val, factor), 0B10110001);
    return _mm512_mul_pd(prod_shuffled,
                         _mm512_load_pd(&ImagFactor<double>::value));
}

/**
 * @brief Calculate val * 1j
 *
 * @param val Complex values arranged in [i3, r3, ..., i0, r0] where
 * each complex values are r0 + 1j*i0, ...
 * @param imag_val Value to product. We product 1j*imag_val to val.
 */
inline __m512d productImagD(__m512d val) {
    __m512d prod_shuffled = _mm512_permutex_pd(val, 0B10110001);
    return _mm512_mul_pd(prod_shuffled,
                         _mm512_load_pd(&ImagFactor<double>::value));
}


} // namespace Pennylane::Gates::AVX512::Util

namespace Pennylane::Gates::AVX512 {
template <typename T>
[[maybe_unused]] constexpr static size_t step_for_complex_precision = 64 / sizeof(T) / 2;

// function aliases
[[maybe_unused]] constexpr auto& fillLeadingOnes = Pennylane::Util::fillLeadingOnes;
[[maybe_unused]] constexpr auto& fillTrailingOnes = Pennylane::Util::fillTrailingOnes;
[[maybe_unused]] constexpr auto& exp2 = Pennylane::Util::exp2;

} // namespace Pennylane::Gates::AVX512
