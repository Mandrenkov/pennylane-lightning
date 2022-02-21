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
 * Defines common AVX512 concept
 */
#pragma once
#include "BitUtil.hpp"
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::Gates::AVX512 {
template <class PrecisionT>
struct Intrinsic {
    static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>,
            "Data type for AVX512 must be float or double");
};

template <> struct Intrinsic<float> { using Type = __m512; };

template <> struct Intrinsic<double> { using Type = __m512d; };

template <class PrecisionT>
using IntrinsicType = typename Intrinsic<PrecisionT>::Type;

/**
 * @brief Simple class for easing product between complex values
 * packed in AVX512 datatype and a pure imaginary value
 */
template <typename PrecisionT>
struct ImagProd {
    static_assert(std::is_same_v<PrecisionT, float> 
            || std::is_same_v<PrecisionT, double>);
    IntrinsicType<PrecisionT> factor_;

    ImagProd() = default;

    explicit ImagProd(PrecisionT value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm512_setr4_ps(-value, value, -value, value);
        } else {
            factor_ = _mm512_setr4_pd(-value, value, -value, value);
        }
    }

    explicit ImagProd(const IntrinsicType<PrecisionT>& val) : factor_{val} {
    }

    explicit ImagProd(IntrinsicType<PrecisionT>&& val) : factor_{std::move(val)} {
    }

    ImagProd& operator*=(PrecisionT val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm512_mul_ps(factor_, _mm512_set1_ps(val));
        } else {
            factor_ = _mm512_mul_pd(factor_, _mm512_set1_pd(val));
        }
        return *this;
    }
    ImagProd& operator*=(IntrinsicType<PrecisionT> val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm512_mul_ps(factor_, val);
        } else {
            factor_ = _mm512_mul_pd(factor_, val);
        }
        return *this;
    }
    
    static auto repeat2(PrecisionT value0, PrecisionT value1) 
        -> ImagProd<PrecisionT> {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            return ImagProd<PrecisionT>{
                          _mm512_setr_ps(-value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1)};
            // clang-format on
        } else {
            // clang-format off
            return ImagProd<PrecisionT>{
                          _mm512_setr_pd(-value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1)};
            // clang-format on
        }
    }

    static auto repeat4(PrecisionT value0, PrecisionT value1) {
        // clang-format off
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return ImagProd<PrecisionT>{
                          _mm512_setr_ps(-value0, value0, -value0, value0,
                                         -value1, value1, -value1, value1,
                                         -value0, value0, -value0, value0,
                                         -value1, value1, -value1, value1)};
        } else {
            return ImagProd<PrecisionT>{
                          _mm512_setr_pd(-value0, value0, -value0, value0,
                                         -value1, value1, -value1, value1)};
        }
        // clang-format on
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto repeat8(float value0, float value1) {
        // clang-format off
        ImagProd<float> obj{
            _mm512_setr_ps(-value0, value0, -value0, value0,
                           -value0, value0, -value0, value0,
                           -value1, value1, -value1, value1,
                           -value1, value1, -value1, value1)};
        return obj;
        // clang-format on
    }

    [[nodiscard]] inline auto product(IntrinsicType<PrecisionT> val) const {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            const auto prod_shuffled = _mm512_permute_ps(val, 0B10'11'00'01);
            return _mm512_mul_ps(prod_shuffled, factor_);
        } else if (std::is_same_v<PrecisionT, double>) {
            const auto prod_shuffled = _mm512_permute_pd(val, 0B01010101);
            return _mm512_mul_pd(prod_shuffled, factor_);
        }
    }
};
/**
 * @brief Simple class for easing product between complex values
 * packed in AVX512 datatype and a real value
 */
template <typename PrecisionT>
struct RealProd {
    static_assert(std::is_same_v<PrecisionT, float> 
            || std::is_same_v<PrecisionT, double>);
    IntrinsicType<PrecisionT> factor_;

    RealProd() = default;

    explicit RealProd(PrecisionT value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm512_set1_ps(value);
        } else {
            factor_ = _mm512_set1_pd(value);
        }
    }

    explicit RealProd(const IntrinsicType<PrecisionT>& val) : factor_{val} {
    }

    explicit RealProd(IntrinsicType<PrecisionT>&& val) : factor_{std::move(val)} {
    }

    RealProd& operator*=(PrecisionT val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm512_mul_ps(factor_, _mm512_set1_ps(val));
        } else {
            factor_ = _mm512_mul_pd(factor_, _mm512_set1_pd(val));
        }
        return *this;
    }

    RealProd& operator*=(IntrinsicType<PrecisionT> val) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            factor_ = _mm512_mul_ps(factor_, val);
        } else {
            factor_ = _mm512_mul_pd(factor_, val);
        }
        return *this;
    }

    static auto repeat2(PrecisionT value0, PrecisionT value1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            return RealProd<PrecisionT>{
                _mm512_setr_ps(value0, value0, value1, value1,
                               value0, value0, value1, value1,
                               value0, value0, value1, value1,
                               value0, value0, value1, value1)};
            // clang-format on
        } else {
            return RealProd<PrecisionT>{
                          _mm512_setr_pd(value0, value0, value1, value1,
                                         value0, value0, value1, value1)};
        }
    }

    static auto repeat4(PrecisionT value0, PrecisionT value1) {
        // clang-format off
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return RealProd<PrecisionT>{
                          _mm512_setr_ps(value0, value0, value0, value0,
                                         value1, value1, value1, value1,
                                         value0, value0, value0, value0,
                                         value1, value1, value1, value1)};
        } else {
            return RealProd<PrecisionT>{
                          _mm512_setr_pd(value0, value0, value0, value0,
                                         value1, value1, value1, value1)};
        }
        // clang-format on
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto repeat8(float value0, float value1) {
        // clang-format off
        return RealProd<float> {
            _mm512_setr_ps(value0, value0, value0, value0,
                           value0, value0, value0, value0,
                           value1, value1, value1, value1,
                           value1, value1, value1, value1)};
        // clang-format on
    }

    static auto setr2(PrecisionT value0, PrecisionT value1)
        -> RealProd<PrecisionT> {
        // clang-format off
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return RealProd<PrecisionT>
                    {_mm512_setr_ps(value0, value1, value0, value1,
                                    value0, value1, value0, value1,
                                    value0, value1, value0, value1,
                                    value0, value1, value0, value1)};
        } else { // double
            return RealProd<PrecisionT>
                    {_mm512_setr_pd(value0, value1, value0, value1,
                                    value0, value1, value0, value1)};
        }
        // clang-format on
    }

    static auto setr4(PrecisionT value0, PrecisionT value1,
                      PrecisionT value2, PrecisionT value3)
        -> RealProd<PrecisionT> {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            return RealProd<PrecisionT>
                    {_mm512_setr_ps(value0, value1, value2, value3,
                                    value0, value1, value2, value3,
                                    value0, value1, value2, value3,
                                    value0, value1, value2, value3)};
            // clang-format on
        } else { // double
            return RealProd<PrecisionT>
                    {_mm512_setr_pd(value0, value1, value2, value3,
                                    value0, value1, value2, value3)};
        }
    }

    // clang-format off
    static auto setr8(PrecisionT value0, PrecisionT value1, 
                      PrecisionT value2, PrecisionT value3,
                      PrecisionT value4, PrecisionT value5,
                      PrecisionT value6, PrecisionT value7)
        -> RealProd<PrecisionT> {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return RealProd<PrecisionT>
                    {_mm512_setr_ps(value0, value1, value2, value3,
                                    value4, value5, value6, value7,
                                    value0, value1, value2, value3,
                                    value4, value5, value6, value7)};
        } else { // double
            return RealProd<PrecisionT>
                    {_mm512_setr_pd(value0, value1, value2, value3,
                                    value4, value5, value6, value7)};
        }
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto setr16(PrecisionT value0, PrecisionT value1, 
                       PrecisionT value2, PrecisionT value3,
                       PrecisionT value4, PrecisionT value5,
                       PrecisionT value6, PrecisionT value7,
                       PrecisionT value8, PrecisionT value9,
                       PrecisionT value10, PrecisionT value11,
                       PrecisionT value12, PrecisionT value13,
                       PrecisionT value14, PrecisionT value15)
        -> RealProd<float> {
        return RealProd<PrecisionT>
            {_mm512_setr_ps(value0,  value1,  value2,  value3,
                            value4,  value5,  value6,  value7,
                            value8,  value9,  value10, value11,
                            value12, value13, value14, value15)};
    }
    // clang-format on

    [[nodiscard]] inline auto product(IntrinsicType<PrecisionT> val) const {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_mul_ps(val, factor_);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_mul_pd(val, factor_);
        }
    }
};

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
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    return _mm512_setzero();
}
inline __m512d parityDInternal(size_t rev_wire) {
    // clang-format off
    switch(rev_wire) {
    case 0:
        return _mm512_setr_pd(1.0, 1.0, -1.0, -1.0,
                              1.0, 1.0, -1.0, -1.0);
    case 1:
        return _mm512_setr_pd(1.0, 1.0, 1.0, 1.0,
                              -1.0, -1.0, -1.0, -1.0);
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    return _mm512_setzero_pd();
}
template <typename T>
[[maybe_unused]] constexpr size_t step_for_complex_precision = 
                64 / sizeof(T) / 2;
} // namespace Pennylane::Gates::AVX512

namespace Pennylane::Gates::AVX {
template <typename T>
struct AVX512Concept {
    using PrecisionT = T;
    using IntrinsicType = AVX512::IntrinsicType<PrecisionT>;
    constexpr static size_t step_for_complex_precision = AVX512::step_for_complex_precision<PrecisionT>;
    constexpr static size_t internal_wires = Pennylane::Util::constLog2PerfectPower(step_for_complex_precision);

    PL_FORCE_INLINE
    static auto load(std::complex<PrecisionT>* p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_load_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_load_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static void store(std::complex<PrecisionT>* p, IntrinsicType value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            _mm512_store_ps(p, value);
        } else if (std::is_same_v<PrecisionT, double>) {
            _mm512_store_pd(p, value);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }
    
    PL_FORCE_INLINE
    static auto product(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_mul_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_mul_pd(v0, v1);
        }else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto add(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_add_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_add_pd(v0, v1);
        }else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

    using ImagProd = AVX512::ImagProd<PrecisionT>;
    using RealProd = AVX512::RealProd<PrecisionT>;

    PL_FORCE_INLINE
    static auto internalParity(const size_t rev_wire) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return AVX512::paritySInternal(rev_wire);
        } else if (std::is_same_v<PrecisionT, double>) {
            return AVX512::parityDInternal(rev_wire);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }
    
    template<size_t rev_wire>
    static auto internalSwap(IntrinsicType v) {
        static_assert(rev_wire < internal_wires);
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if constexpr (rev_wire == 0) {
                return _mm512_permute_ps(v, 0B01'00'11'10);
            } else if (rev_wire == 1) {
                return _mm512_permutexvar_ps(
                    // NOLINTNEXTLINE(readability-magic-numbers)
                    _mm512_set_epi32(11, 10,  9,  8,
                    // NOLINTNEXTLINE(readability-magic-numbers)
                                     15, 14, 13, 12,
                    // NOLINTNEXTLINE(readability-magic-numbers)
                                      3,  2,  1,  0,
                    // NOLINTNEXTLINE(readability-magic-numbers)
                                      7,  6,  5,  4), v);
                // NOLINT(readability-magic-numbers)
            } else { // rev_wire == 2
                return _mm512_permutexvar_ps(
                    // NOLINTNEXTLINE(readability-magic-numbers)
                    _mm512_set_epi32( 7,  6,  5,  4,  3,  2, 1, 0,
                    // NOLINTNEXTLINE(readability-magic-numbers)
                                     15, 14, 13, 12, 11, 10, 9, 8),
                    v); 
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if constexpr(rev_wire == 0) {
                return _mm512_permutex_pd(v, 0B01'00'11'10);
            } else { // rev_wire == 1
                return _mm512_permutexvar_pd(
                    _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), v); // NOLINT(readability-magic-numbers)
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

};
} // namespace Pennylane::Gates::AVX
