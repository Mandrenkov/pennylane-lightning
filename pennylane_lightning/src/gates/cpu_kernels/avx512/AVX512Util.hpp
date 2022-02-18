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

template <class PrecisionT>
struct AVX512Intrinsic {
    static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>,
            "Data type for AVX512 must be float or double");
};

template <> struct AVX512Intrinsic<float> { using Type = __m512; };

template <> struct AVX512Intrinsic<double> { using Type = __m512d; };

template <class PrecisionT>
using AVX512IntrinsicType = typename AVX512Intrinsic<PrecisionT>::Type;

inline
AVX512IntrinsicType<float> load(const float* p) {
    return _mm512_load_ps(p);
}
inline
AVX512IntrinsicType<float> load(const std::complex<float>* p) {
    return _mm512_load_ps(p);
}

inline void store(std::complex<float>* p,
                  AVX512IntrinsicType<float> value) {
    _mm512_store_ps(p, value);
}

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
        _mm512_permute_pd(_mm512_mul_pd(val, factor), 0B01'01'01'01);
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
    __m512d prod_shuffled = _mm512_permute_pd(val, 0B01'01'01'01);
    return _mm512_mul_pd(prod_shuffled,
                         _mm512_load_pd(&ImagFactor<double>::value));
}


/**
 * @brief Simple class for easing product between complex values
 * packed in AVX512 datatype and a pure imaginary value
 */
template<typename T>
struct ProdPureImag;

template <>
struct ProdPureImag<float> {
    __m512 factor_;

    explicit ProdPureImag(float value) {
        factor_ = _mm512_setr4_ps(-value, value, -value, value);
    }

    ProdPureImag(float value1, float value2) {
        // clang-format off
        factor_ = _mm512_setr_ps(-value1, value1, -value1, value1,
                                 -value1, value1, -value1, value1,
                                 -value2, value2, -value2, value2,
                                 -value2, value2, -value2, value2);
        // clang-format on
    }

    [[nodiscard]] inline auto product(__m512 val) const -> __m512 {
        const auto prod_shuffled = _mm512_permute_ps(val, 0B10'11'00'01);
        return _mm512_mul_ps(prod_shuffled, factor_);
    }
};

template <>
struct ProdPureImag<double> {
    __m512d factor_;

    explicit ProdPureImag(double value) {
        factor_ = _mm512_setr4_pd(-value, value, -value, value);
    }

    ProdPureImag(double value1, double value2) {
        // clang-format off
        factor_ = _mm512_setr_pd(-value1, value1, -value1, value1,
                                 -value1, value1, -value1, value1);
        // clang-format on
    }

    [[nodiscard]] inline auto product(__m512d val) const -> __m512d {
        const auto prod_shuffled = _mm512_permute_pd(val, 0B01'01'01'01);
        return _mm512_mul_pd(prod_shuffled, factor_);
    }
};

} // namespace Pennylane::Gates::AVX512::Util

namespace Pennylane::Gates::AVX512 {
template <typename T>
[[maybe_unused]] constexpr size_t step_for_complex_precision = 64 / sizeof(T) /
                                                               2;

// function aliases
[[maybe_unused]] constexpr static auto &fillLeadingOnes =
    Pennylane::Util::fillLeadingOnes;
[[maybe_unused]] constexpr static auto &fillTrailingOnes =
    Pennylane::Util::fillTrailingOnes;
[[maybe_unused]] constexpr static auto &exp2 = Pennylane::Util::exp2;

} // namespace Pennylane::Gates::AVX512

/// @cond DEV
namespace Pennylane::Gates::AVX512::Internal {
inline __m512 paritySInternal(size_t rev_wire) {
    // clang-format off
    switch(rev_wire) {
    case 0:
        return _mm512_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                              1.0F, 1.0F, -1.0F, -1.0F,
                              1.0F, 1.0F, -1.0F, -1.0F,
                              1.0F, 1.0F, -1.0F, -1.0F);
    case 1:
        return _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                              -1.0F, -1.0F, -1.0F, -1.0F,
                              1.0F, 1.0F, 1.0F, 1.0F,
                              -1.0F,- 1.0F, -1.0F, -1.0F);
    case 2:
        return _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                              1.0F, 1.0F, 1.0F, 1.0F,
                              -1.0F, -1.0F, -1.0F, -1.0F,
                              -1.0F,- 1.0F, -1.0F, -1.0F);
    }
    // clang-format on
    PL_UNREACHABLE;
    return _mm512_setzero();
}
inline __m512d parityDInternal(size_t rev_wire) {
    // clang-format off
    switch(rev_wire) {
    case 0:
        return _mm512_setr_pd(1.0L, 1.0L, -1.0L, -1.0L,
                              1.0L, 1.0L, -1.0L, -1.0L);
    case 1:
        return _mm512_setr_pd(1.0L, 1.0L, 1.0L, 1.0L,
                              -1.0L, -1.0L, -1.0L, -1.0L);
    }
    // clang-format on
    PL_UNREACHABLE;
    return _mm512_setzero_pd();
}
} // namespace Pennylane::Gates::AVX512::Internal
/// @endcond
