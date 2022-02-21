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
 * Defines common AVX256 concept
 */
#pragma once
#include "BitUtil.hpp"
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::Gates::AVX2 {
template <class PrecisionT>
struct Intrinsic {
    static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>,
            "Data type for AVX256 must be float or double");
};

template <> struct Intrinsic<float> { using Type = __m256; };

template <> struct Intrinsic<double> { using Type = __m256d; };

template <class PrecisionT>
using IntrinsicType = typename Intrinsic<PrecisionT>::Type;

/**
 * @brief Simple class for easing product between complex values
 * packed in AVX256 datatype and a pure imaginary value
 */
template <typename PrecisionT>
struct ImagProd {
    static_assert(std::is_same_v<PrecisionT, float> 
            || std::is_same_v<PrecisionT, double>);
    IntrinsicType<PrecisionT> factor_;

    ImagProd() = default;

    explicit ImagProd(PrecisionT value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm256_setr_ps(-value, value, -value, value,
                                     -value, value, -value, value);
        } else {
            factor_ = _mm256_setr_pd(-value, value, -value, value);
        }
    }
    explicit ImagProd(const IntrinsicType<PrecisionT>& val) : factor_{val} {
    }

    explicit ImagProd(IntrinsicType<PrecisionT>&& val) : factor_{std::move(val)} {
    }

    ImagProd& operator*=(PrecisionT val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm256_mul_ps(factor_, _mm256_set1_ps(val));
        } else {
            factor_ = _mm256_mul_pd(factor_, _mm256_set1_pd(val));
        }
        return *this;
    }
    ImagProd& operator*=(IntrinsicType<PrecisionT> val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm256_mul_ps(factor_, val);
        } else {
            factor_ = _mm256_mul_pd(factor_, val);
        }
        return *this;
    }
    
    static auto repeat2(PrecisionT value0, PrecisionT value1) 
        -> ImagProd<PrecisionT> {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            return ImagProd<PrecisionT>
                    {_mm256_setr_ps(-value0, value0, -value1, value1,
                                    -value0, value0, -value1, value1)};
            // clang-format on
        } else {
            return ImagProd<PrecisionT>
                    {_mm256_setr_pd(-value0, value0, -value1, value1)};
        }
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto repeat4(float value0, float value1) -> ImagProd<float> {
        // clang-format off
        return ImagProd<float>{
            _mm256_setr_ps(-value0, value0, -value0, value0,
                           -value1, value1, -value1, value1)};
        // clang-format on
    }

    [[nodiscard]] inline auto product(IntrinsicType<PrecisionT> val) const {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            const auto prod_shuffled = _mm256_permute_ps(val, 0B10'11'00'01);
            return _mm256_mul_ps(prod_shuffled, factor_);
        } else if (std::is_same_v<PrecisionT, double>) {
            const auto prod_shuffled = _mm256_permute_pd(val, 0B01'01);
            return _mm256_mul_pd(prod_shuffled, factor_);
        }
    }
};
/**
 * @brief Simple class for easing product between complex values
 * packed in AVX256 datatype and a real value
 */
template <typename PrecisionT>
struct RealProd {
    static_assert(std::is_same_v<PrecisionT, float> 
            || std::is_same_v<PrecisionT, double>);
    IntrinsicType<PrecisionT> factor_;

    RealProd() = default;

    explicit RealProd(PrecisionT value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm256_set1_ps(value);
        } else {
            factor_ = _mm256_set1_pd(value);
        }
    }

    explicit RealProd(const IntrinsicType<PrecisionT>& val) : factor_{val} {
    }

    explicit RealProd(IntrinsicType<PrecisionT>&& val) : factor_{std::move(val)} {
    }

    RealProd& operator*=(PrecisionT val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm256_mul_ps(factor_, _mm256_set1_ps(val));
        } else {
            factor_ = _mm256_mul_pd(factor_, _mm256_set1_pd(val));
        }
        return *this;
    }

    RealProd& operator*=(IntrinsicType<PrecisionT> val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm256_mul_ps(factor_, val);
        } else {
            factor_ = _mm256_mul_pd(factor_, val);
        }
        return *this;
    }
    
    static auto repeat2(PrecisionT value0, PrecisionT value1)
        -> RealProd<PrecisionT> {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            return RealProd<PrecisionT>
                    {_mm256_setr_ps(value0, value0, value1, value1,
                                    value0, value0, value1, value1)};
            // clang-format on
        } else { // double
            return RealProd<PrecisionT>
                    {_mm256_setr_pd(value0, value0, value1, value1)};
        }
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto repeat4(PrecisionT value0, PrecisionT value1) -> RealProd<float> {
        // clang-format off
        return RealProd<float>{_mm256_setr_ps(value0, value0, value0, value0,
                                              value1, value1, value1, value1)};

        // clang-format on
    }

    static auto setr2(PrecisionT value0, PrecisionT value1)
        -> RealProd<PrecisionT> {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            return RealProd<PrecisionT>
                    {_mm256_setr_ps(value0, value1, value0, value1,
                                    value0, value1, value0, value1)};
            // clang-format on
        } else { // double
            return RealProd<PrecisionT>
                    {_mm256_setr_pd(value0, value1, value0, value1)};
        }
    }

    static auto setr4(PrecisionT value0, PrecisionT value1,
                      PrecisionT value2, PrecisionT value3)
        -> RealProd<PrecisionT> {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            return RealProd<PrecisionT>
                    {_mm256_setr_ps(value0, value1, value2, value3,
                                    value0, value1, value2, value3)};
            // clang-format on
        } else { // double
            return RealProd<PrecisionT>
                    {_mm256_setr_pd(value0, value1, value2, value3)};
        }
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto setr8(PrecisionT value0, PrecisionT value1, 
                      PrecisionT value2, PrecisionT value3,
                      PrecisionT value4, PrecisionT value5,
                      PrecisionT value6, PrecisionT value7)
        -> RealProd<float> {
        // clang-format off
        return RealProd<PrecisionT>
                {_mm256_setr_ps(value0, value1, value2, value3,
                                value4, value5, value6, value7)};
        // clang-format on
    }

    [[nodiscard]] inline auto product(IntrinsicType<PrecisionT> val) const {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_mul_ps(val, factor_);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_mul_pd(val, factor_);
        }
    }
};

inline __m256 paritySInternal(size_t rev_wire) {
    // clang-format off
    switch(rev_wire) {
    case 0:
        return _mm256_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                              1.0F, 1.0F, -1.0F, -1.0F);
    case 1:
        return _mm256_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                              -1.0F, -1.0F, -1.0F, -1.0F);
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    return _mm256_setzero_ps();
}
inline __m256d parityDInternal(size_t rev_wire) {
    // clang-format off
    switch(rev_wire) {
    case 0:
        return _mm256_setr_pd(1.0, 1.0, -1.0, -1.0);
    case 1:
        return _mm256_setr_pd(1.0, 1.0, 1.0, 1.0);
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    return _mm256_setzero_pd();
}
template <typename T>
[[maybe_unused]] constexpr size_t step_for_complex_precision = 
                32 / sizeof(T) / 2;
} // namespace Pennylane::Gates::AVX2

namespace Pennylane::Gates::AVX {
template <typename T>
struct AVX2Concept {
    using PrecisionT = T;
    using IntrinsicType = AVX2::IntrinsicType<PrecisionT>;
    constexpr static size_t step_for_complex_precision = AVX2::step_for_complex_precision<PrecisionT>;
    constexpr static size_t internal_wires = Pennylane::Util::constLog2PerfectPower(step_for_complex_precision);

    PL_FORCE_INLINE
    static auto load(const std::complex<PrecisionT>* p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_load_ps(reinterpret_cast<const PrecisionT*>(p));
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_load_pd(reinterpret_cast<const PrecisionT*>(p));
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static void store(std::complex<PrecisionT>* p, IntrinsicType value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            _mm256_store_ps(reinterpret_cast<PrecisionT*>(p), value);
        } else if (std::is_same_v<PrecisionT, double>) {
            _mm256_store_pd(reinterpret_cast<PrecisionT*>(p), value);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }
    
    PL_FORCE_INLINE
    static auto product(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_mul_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_mul_pd(v0, v1);
        }else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto add(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_add_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_add_pd(v0, v1);
        }else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

    using ImagProd = AVX2::ImagProd<PrecisionT>;
    using RealProd = AVX2::RealProd<PrecisionT>;

    PL_FORCE_INLINE
    static auto internalParity(const size_t rev_wire) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return AVX2::paritySInternal(rev_wire);
        } else if (std::is_same_v<PrecisionT, double>) {
            return AVX2::parityDInternal(rev_wire);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }
    
    template<size_t rev_wire>
    static auto internalSwap(IntrinsicType v) {
        static_assert(rev_wire < internal_wires);
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if constexpr (rev_wire == 0) {
                return _mm256_permute_ps(v, 0B01'00'11'10);
            } else {// rev_wire == 1
                return _mm256_permutevar8x32_ps(
                    v,
                    _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4)); // NOLINT(readability-magic-numbers)
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_permute4x64_pd(v, 0B01'00'11'10);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }
};
} // namespace Pennylane::Gates::AVX

