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
 * Defines common utility functions for AVX2
 */
#pragma once
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::Gates::AVX2::Util {

template <class PrecisionT> struct AVX2Intrinsic;

template <> struct AVX2Intrinsic<float> { using Type = __m256; };

template <> struct AVX2Intrinsic<double> { using Type = __m256d; };

template <class PrecisionT>
using AVX2IntrinsicType = typename AVX2Intrinsic<PrecisionT>::Type;

/**
 * @brief Simple class for easing product between complex values
 * packed in AVX2 datatype and a pure imaginary value
 */
template<typename T>
struct ProdPureImag;

template <>
struct ProdPureImag<float> {
    __m256 factor_;

    explicit ProdPureImag(float value) {
        factor_ = _mm256_setr_ps(-value, value, -value, value, 
                                 -value, value, -value, value);
    }

    [[nodiscard]] inline auto product(__m256 val) const -> __m256 {
        const auto prod_shuffled = _mm256_permute_ps(val, 0B10'11'00'01);
        return _mm256_mul_ps(prod_shuffled, factor_);
    }
};

template <>
struct ProdPureImag<double> {
    __m256d factor_;

    explicit ProdPureImag(double value) {
        factor_ = _mm256_setr_pd(-value, value, -value, value);
    }

    [[nodiscard]] inline auto product(__m256d val) const -> __m256d {
        const auto prod_shuffled = _mm256_permute_pd(val, 0B01'01'01'01);
        return _mm256_mul_pd(prod_shuffled, factor_);
    }
};
} // namespace Pennylane::Gates::AVX2::Util

namespace Pennylane::Gates::AVX2 {
template <typename T>
[[maybe_unused]] constexpr size_t step_for_complex_precision = 32 / sizeof(T) /
                                                               2;

// function aliases
[[maybe_unused]] constexpr static auto &fillLeadingOnes =
    Pennylane::Util::fillLeadingOnes;
[[maybe_unused]] constexpr static auto &fillTrailingOnes =
    Pennylane::Util::fillTrailingOnes;
[[maybe_unused]] constexpr static auto &exp2 = Pennylane::Util::exp2;

} // namespace Pennylane::Gates::AVX2

/// @cond DEV
namespace Pennylane::Gates::AVX2::Internal {
inline __m256 paritySInternal(size_t rev_wire) {
    // clang-format off
    switch(rev_wire) {
    case 0:
        return _mm256_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                              1.0F, 1.0F, -1.0F, -1.0F);
    case 1:
        return _mm256_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                              -1.0F, -1.0F, -1.0F, -1.0F);
    }
    // clang-format on
    PL_UNREACHABLE;
    return _mm256_setzero_ps();
}
inline __m256d parityDInternal() {
    return _mm256_setr_pd(1.0L, 1.0L, -1.0L, -1.0L);
}
} // namespace Pennylane::Gates::AVX2::Internal
/// @endcond
