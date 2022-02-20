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
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::Gates::AVX512::Internal {

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
    
    static auto repeat2(PrecisionT value0, PrecisionT value1) {
        ImagProd<PrecisionT> obj;
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            obj.factor_ = _mm512_setr_ps(-value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1);
            // clang-format on
        } else {
            // clang-format off
            obj.factor_ = _mm512_setr_pd(-value0, value0, -value1, value1,
                                         -value0, value0, -value1, value1);
            // clang-format on
        }
        return obj;
    }

    static auto repeat4(PrecisionT value0, PrecisionT value1) {
        // clang-format off
        ImagProd<PrecisionT> obj;
        if constexpr (std::is_same_v<PrecisionT, float>) {
            obj.factor_ = _mm512_setr_ps(-value0, value0, -value0, value0,
                                         -value1, value1, -value1, value1,
                                         -value0, value0, -value0, value0,
                                         -value1, value1, -value1, value1);
        } else {
            obj.factor_ = _mm512_setr_pd(-value0, value0, -value0, value0,
                                         -value1, value1, -value1, value1);
        }
        return obj;
        // clang-format on
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto repeat8(float value0, float value1) {
        // clang-format off
        ImagProd<float> obj;
        obj.factor_ = _mm512_setr_ps(-value0, value0, -value0, value0,
                                     -value0, value0, -value0, value0,
                                     -value1, value1, -value1, value1,
                                     -value1, value1, -value1, value1);
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
    
    static auto repeat2(PrecisionT value0, PrecisionT value1) {
        RealProd<PrecisionT> obj;
        if constexpr (std::is_same_v<PrecisionT, float>) {
            // clang-format off
            obj.factor_ = _mm512_setr_ps(value0, value0, value1, value1,
                                         value0, value0, value1, value1,
                                         value0, value0, value1, value1,
                                         value0, value0, value1, value1);
            // clang-format on
        } else {
            obj.factor_ = _mm512_setr_pd(value0, value0, value1, value1,
                                         value0, value0, value1, value1);
        }
        return obj;
    }


    static auto repeat4(PrecisionT value0, PrecisionT value1) {
        // clang-format off
        RealProd<PrecisionT> obj;
        if constexpr (std::is_same_v<PrecisionT, float>) {
            obj.factor_ = _mm512_setr_ps(value0, value0, value0, value0,
                                         value1, value1, value1, value1,
                                         value0, value0, value0, value0,
                                         value1, value1, value1, value1);
        } else {
            obj.factor_ = _mm512_setr_pd(value0, value0, value0, value0,
                                         value1, value1, value1, value1);
        }
        return obj;
        // clang-format on
    }

    template<typename T = PrecisionT,
             std::enable_if_t<std::is_same_v<T, float>, bool> = true> // only enable for float
    static auto repeat8(float value0, float value1) {
        // clang-format off
        RealProd<float> obj;
        obj.factor_ = _mm512_setr_ps(value0, value0, value0, value0,
                                     value0, value0, value0, value0,
                                     value1, value1, value1, value1,
                                     value1, value1, value1, value1);
        return obj;
        // clang-format on
    }

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
        return _mm512_setr_pd(1.0L, 1.0L, -1.0L, -1.0L,
                              1.0L, 1.0L, -1.0L, -1.0L);
    case 1:
        return _mm512_setr_pd(1.0L, 1.0L, 1.0L, 1.0L,
                              -1.0L, -1.0L, -1.0L, -1.0L);
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    return _mm512_setzero_pd();
}
} // namespace Pennylane::Gates::AVX512::Internal

namespace Pennylane::Gates::AVX512 {
template <typename T>
[[maybe_unused]] constexpr size_t step_for_complex_precision = 
                64 / sizeof(T) / 2;

// function aliases
[[maybe_unused]] constexpr static auto &fillLeadingOnes =
    Pennylane::Util::fillLeadingOnes;
[[maybe_unused]] constexpr static auto &fillTrailingOnes =
    Pennylane::Util::fillTrailingOnes;
[[maybe_unused]] constexpr static auto &exp2 = Pennylane::Util::exp2;


template <typename PrecisionT>
struct AVX512Concept {
    using IntrinsicType = Internal::IntrinsicType<PrecisionT>;
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
            return _mm512_store_ps(p, value);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_store_pd(p, value);
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

    using ImagProd = Internal::ImagProd<PrecisionT>;
    using RealProd = Internal::RealProd<PrecisionT>;

    PL_FORCE_INLINE
    static auto internalParity(const size_t rev_wire) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return Internal::paritySInternal(rev_wire);
        } else if (std::is_same_v<PrecisionT, double>) {
            return Internal::parityDInternal(rev_wire);
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
                    _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12,
                                     3, 2, 1, 0, 7, 6, 5, 4),
                    v);
            } else { // rev_wire == 2
                return _mm512_permutexvar_ps(
                    _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,
                                     15, 14, 13, 12, 11, 10, 9, 8),
                    v);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if constexpr(rev_wire == 0) {
                return _mm512_permutex_pd(v, 0B01'00'11'10);
            } else { // rev_wire == 1
                return _mm512_permutexvar_pd(
                    _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), v);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || std::is_same_v<PrecisionT, double>);
        }
    }

};
} // namespace Pennylane::Gates::AVX512

