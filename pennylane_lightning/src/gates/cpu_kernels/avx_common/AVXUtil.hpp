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
 * Defines common utility functions for all AVX
 */
#pragma once
#include "AVX2Concept.hpp"
#include "AVX512Concept.hpp"
#include "BitUtil.hpp"
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <cstdlib>

namespace Pennylane::Gates::AVX {
// function aliases
[[maybe_unused]] constexpr static auto &fillLeadingOnes =
    Pennylane::Util::fillLeadingOnes;
[[maybe_unused]] constexpr static auto &fillTrailingOnes =
    Pennylane::Util::fillTrailingOnes;
[[maybe_unused]] constexpr static auto &exp2 = Pennylane::Util::exp2;

// clang-format off
constexpr __m256i setr256i(int32_t  e0, int32_t  e1, int32_t  e2, int32_t  e3,
		                   int32_t  e4, int32_t  e5, int32_t  e6, int32_t  e7) {
    return __m256i{(int64_t(e1) << 32) | e0,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e3) << 32) | e2,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e5) << 32) | e4,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e7) << 32) | e6}; // NOLINT(hicpp-signed-bitwise)
}
constexpr __m512i setr512i(int32_t  e0, int32_t  e1, int32_t  e2, int32_t  e3,
		                   int32_t  e4, int32_t  e5, int32_t  e6, int32_t  e7, 
		                   int32_t  e8, int32_t  e9, int32_t e10, int32_t e11, 
		                   int32_t e12, int32_t e13, int32_t e14, int32_t e15) {
    return __m512i{(int64_t(e1) << 32)  |  e0,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e3) << 32)  |  e2,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e5) << 32)  |  e4,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e7) << 32)  |  e6,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e9) << 32)  |  e8,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e11) << 32) | e10,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e13) << 32) | e12,  // NOLINT(hicpp-signed-bitwise)
                   (int64_t(e15) << 32) | e14}; // NOLINT(hicpp-signed-bitwise)
}
//clang-format on
constexpr __m512i setr512i(int64_t  e0, int64_t  e1, int64_t  e2, int64_t  e3,
		                   int64_t  e4, int64_t  e5, int64_t  e6, int64_t  e7) {
    return __m512i{e0, e1, e2, e3, e4, e5, e6, e7};
}

template<typename PrecisionT, size_t packed_size>
struct AVXIntrinsic{
	static_assert((sizeof(PrecisionT) * packed_size == 32) || (sizeof(PrecisionT) * packed_size == 64));
};

template<>
struct AVXIntrinsic<float, 8> {
	// AVX2
	using Type = __m256;
};
template<>
struct AVXIntrinsic<double, 4> {
	// AVX512
	using Type = __m256d;
};
template<>
struct AVXIntrinsic<float, 16> {
	// AVX512
	using Type = __m512;
};
template<>
struct AVXIntrinsic<double, 8> {
	// AVX512
	using Type = __m512d;
};
template<typename T, size_t size>
using AVXIntrinsicType = typename AVXIntrinsic<T, size>::Type;

template<size_t mask_size>
struct Mask {
    static_assert(mask_size == 4 || mask_size == 8 || mask_size == 16);
    std::array<bool, mask_size> data = {0,};

    constexpr bool& operator[](size_t idx) {
        return data[idx];
    }
    constexpr bool operator[](size_t idx) const {
        return data[idx];
    }
};

template<size_t mask_size>
constexpr Mask<mask_size> operator~(const Mask<mask_size>& rhs) {
    Mask<mask_size> res;

    for(size_t i = 0; i < mask_size; i++) {
        res[i] = ~rhs.data[i];
    }
}

template<class PrecisionT, size_t packed_size>
struct AVXConcept;

template<>
struct AVXConcept<float, 16> {
    using Type = AVX512Concept<float>;
};
template<>
struct AVXConcept<double, 8> {
    using Type = AVX512Concept<double>;
};

template<>
struct AVXConcept<float, 8> {
    using Type = AVX2Concept<float>;
};
template<>
struct AVXConcept<double, 4> {
    using Type = AVX2Concept<double>;
};

template<class PrecisionT, size_t packed_size>
using AVXConceptType = typename AVXConcept<PrecisionT, packed_size>::Type;

template<typename PrecisionT, size_t packed_size, typename Func>
static auto toParity(Func&& func) -> decltype(auto) {
    std::array<PrecisionT, packed_size>
        data = {};
    for(size_t idx = 0; idx < packed_size / 2; idx++) {
        data[2*idx] = static_cast<PrecisionT>(1.0) - 2*static_cast<PrecisionT>(func(idx));
        data[2*idx + 1] = static_cast<PrecisionT>(1.0) - 2*static_cast<PrecisionT>(func(idx));
    }
    return AVXConceptType<PrecisionT, packed_size>::loadu(data.data());
}
template<typename PrecisionT, size_t packed_size, typename Func>
static auto setValueOneTwo(Func&& func, PrecisionT value1, PrecisionT value2)
    -> decltype(auto) {
    std::array<PrecisionT, packed_size>
        data = {};
    for(size_t idx = 0; idx < packed_size / 2; idx++) {
        data[2*idx] = static_cast<PrecisionT>(1.0) - 2*static_cast<PrecisionT>(func(idx));
        data[2*idx + 1] = static_cast<PrecisionT>(1.0) - 2*static_cast<PrecisionT>(func(idx));
    }
    return AVXConceptType<PrecisionT, packed_size>::loadu(data.data());
}

/**
 * @brief one or minus one parity for reverse wire in packed data.
 */
template<typename PrecisionT, size_t packed_size>
struct InternalParity;

template<>
struct InternalParity<float, 8> {
    // AVX2 with float
    constexpr static auto create(size_t rev_wire) -> AVXIntrinsicType<float, 8> {
        // clang-format off
        switch(rev_wire) {
        case 0:
            return __m256{1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F};
        case 1:
            return __m256{1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F};
        default:
            PL_UNREACHABLE;
        }
        // clang-format on
        return __m256{
            0.0F,
        };
    }
};
template <> struct InternalParity<double, 4> {
    // AVX2 with double
    constexpr static auto create(size_t rev_wire)
        -> AVXIntrinsicType<double, 4> {
        // clang-format off
        switch(rev_wire) {
        case 0:
            return __m256d{1.0, 1.0, -1.0, -1.0};
        case 1:
            return __m256d{1.0, 1.0, 1.0, 1.0};
        default:
            PL_UNREACHABLE;
        }
        // clang-format on
        return __m256d{
            0.0,
        };
    }
};
template <> struct InternalParity<float, 16> {
    // AVX2 with float
    constexpr static auto create(size_t rev_wire)
        -> AVXIntrinsicType<float, 16> {
        // clang-format off
        switch(rev_wire) {
        case 0:
            return __m512{1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F,
                          1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F};
        case 1:
            return __m512{1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
                          1.0F, 1.0F, 1.0F, 1.0F, -1.0F,- 1.0F, -1.0F, -1.0F};
        case 2:
            return __m512{ 1.0F,  1.0F,  1.0F,  1.0F,
                           1.0F,  1.0F,  1.0F,  1.0F,
                          -1.0F, -1.0F, -1.0F, -1.0F,
                          -1.0F,- 1.0F, -1.0F, -1.0F};
        default:
            PL_UNREACHABLE;
        }
        // clang-format on
        return __m512{
            0,
        };
    }
};

template <> struct InternalParity<double, 8> {
    // AVX2 with float
    constexpr static auto create(size_t rev_wire)
        -> AVXIntrinsicType<double, 8> {
        // clang-format off
        switch(rev_wire) {
        case 0:
            return __m512d{1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
        case 1:
            return __m512d{1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};
        default:
            PL_UNREACHABLE;
        }
        // clang-format on
        return __m512d{
            0,
        };
    }
};

template <typename PrecisionT, size_t packed_size>
constexpr auto internalParity(size_t rev_wire)
    -> AVXIntrinsicType<PrecisionT, packed_size> {
    return InternalParity<PrecisionT, packed_size>::create(rev_wire);
}

/**
 * @brief Factor that is applied to the intrinsic type for product of
 * pure imaginary value.
 */
template <typename PrecisionT, size_t packed_size> struct ImagFactor;

template <> struct ImagFactor<float, 8> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 8> {
        return __m256{-val, val, -val, val, -val, val, -val, val};
    };
};
template <> struct ImagFactor<double, 4> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 4> {
        return __m256d{-val, val, -val, val};
    };
};
template <> struct ImagFactor<float, 16> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 16> {
        return __m512{-val, val, -val, val, -val, val, -val, val,
                      -val, val, -val, val, -val, val, -val, val};
    };
};
template <> struct ImagFactor<double, 8> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 8> {
        return __m512d{-val, val, -val, val, -val, val, -val, val};
    };
};
template <typename PrecisionT, size_t packed_size>
constexpr auto imagFactor(PrecisionT val = 1.0) {
    return ImagFactor<PrecisionT, packed_size>::create(val);
}

template <typename PrecisionT, size_t packed_size> struct Set1;

template <> struct Set1<float, 8> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 8> {
        return __m256{val, val, val, val, val, val, val, val};
    }
};
template <> struct Set1<float, 16> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 16> {
        return __m512{val, val, val, val, val, val, val, val,
                      val, val, val, val, val, val, val, val};
    }
};
template <> struct Set1<double, 4> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 4> {
        return __m256d{val, val, val, val};
    }
};
template <> struct Set1<double, 8> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 8> {
        return __m512d{val, val, val, val, val, val, val, val};
    }
};

template <typename PrecisionT, size_t packed_size>
constexpr auto set1(PrecisionT val) {
    return Set1<PrecisionT, packed_size>::create(val);
}
} // namespace Pennylane::Gates::AVX
