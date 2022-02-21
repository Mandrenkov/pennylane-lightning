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

template<size_t packed_bytes>
struct PackedInteger {
    static_assert(packed_bytes == 32 || packed_bytes == 64);
};
template<>
struct PackedInteger<32> {
    using Type = __m256i;
    static Type setr(int e7, int e6, int e5, int e4, int e3, int e2,
              int e1, int e0) {
        return _mm256_setr_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
    }
    static Type setr(int64_t e3, int64_t e2, int64_t e1, int64_t e0) {
        return _mm256_setr_epi64x(e3, e2, e1, e0);
    }
};
template<>
struct PackedInteger<64> {
    using Type = __m512i;

    static Type setr(int64_t e7, int64_t e6, int64_t e5, int64_t e4,
                     int64_t e3, int64_t e2, int64_t e1, int64_t e0) {
        return _mm512_setr_epi64(e7, e6, e5, e4, e3, e2, e1, e0);
    }
    static Type setr(int32_t e15, int32_t e14, int32_t e13, int32_t e12,
                     int32_t e11, int32_t e10, int32_t  e9, int32_t  e8,
                     int32_t  e7, int32_t  e6, int32_t  e5, int32_t  e4,
                     int32_t  e3, int32_t  e2, int32_t  e1, int32_t  e0) {
        return _mm512_setr_epi32(e15, e14, e13, e12, e11, e10, e9, e8, e7,
                                 e6, e5, e4, e3, e2, e1, e0);
    }

};

template<size_t packed_bytes>
using PackedIntegerType = typename PackedInteger<packed_bytes>::Type;

} // namespace Pennylane::Gates::AVX
