
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
 * Defines permutation of AVX intrinsics
 */
#pragma once
#include "AVXUtil.hpp"

#include <immintrin.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace Pennylane::Gates::AVX {

/// @cond DEV
namespace Internal {
/**
 * @brief Custom bubble sort. Let's use this until we have constexpr 
 * std::sort in C++20.
 */
template<typename iterator>
constexpr void bubble_sort(iterator begin, iterator end) {
    bool swapped = false;
    auto n = std::distance(begin, end);
    do {
        swapped = false;
        for(typename std::iterator_traits<iterator>::difference_type idx = 0;
                idx < n - 1; idx++) {
            if (*(begin + idx) > *(begin + idx + 1)) {
                const auto tmp = *(begin + idx + 1);
                *(begin+idx+1) = *(begin+idx);
                *(begin+idx) = tmp;
                swapped = true;
            }
        }
    } while(swapped);
}
} // namespace Internal
/// @endcond

template<uint8_t e0, uint8_t e1, uint8_t e2, uint8_t e3>
struct Permute4 {
    constexpr static size_t size = 4;
    constexpr static std::array<uint8_t, 4> permutation{e0, e1, e2, e3};
};
template<uint8_t e0, uint8_t e1, uint8_t e2, uint8_t e3,
         uint8_t e4, uint8_t e5, uint8_t e6, uint8_t e7>
struct Permute8 {
    constexpr static size_t size = 8;
    constexpr static std::array<uint8_t, 8> permutation{e0, e1, e2, e3, e4, e5, e6, e7};
};
template<uint8_t e0, uint8_t e1, uint8_t e2, uint8_t e3,
         uint8_t e4, uint8_t e5, uint8_t e6, uint8_t e7,
         uint8_t e8, uint8_t e9, uint8_t e10, uint8_t e11,
         uint8_t e12, uint8_t e13, uint8_t e14, uint8_t e15>
struct Permute16 {
    constexpr static size_t size = 16;
    constexpr static std::array<uint8_t, 16> permutation{e0, e1, e2, e3, e4, e5, e6, e7,
                                                         e8, e9, e10, e11, e12, e13, e14, e15};
};

template<size_t permutation_size>
struct Permute {
    static_assert(permutation_size == 4 || permutation_size == 8 || permutation_size == 16);
};

template<>
struct Permute<4> {
    using Type = Permute4<0, 1, 2, 3>;
};

template<>
struct Permute<8> {
    using Type = Permute8<0, 1, 2, 3, 4, 5, 6, 7>; //NOLINT(readability-magic-numbers)
};

template<>
struct Permute<16> {
    //NOLINTNEXTLINE(readability-magic-numbers)
    using Type = Permute16<0, 1,  2,  3,  4,  5,  6,  7,
                           8, 9, 10, 11, 12, 13, 14, 15>; //NOLINT(readability-magic-numbers)
};

template<typename Permute, size_t rev_wire, typename Enable = void>
struct Flip;
template<typename Permute, size_t rev_wire>
struct Flip<Permute, rev_wire, std::enable_if_t<Permute::size == 4>> {
    static_assert(rev_wire == 0);
    using Type = Permute4<Permute::permutation[2], Permute::permutation[3],
                          Permute::permutation[0], Permute::permutation[1]>;
};
template<typename Permute>
struct Flip<Permute, 0, std::enable_if_t<Permute::size == 8>> {
    using Type = Permute8<Permute::permutation[2], Permute::permutation[3],
                          Permute::permutation[0], Permute::permutation[1],
                          Permute::permutation[6], Permute::permutation[7],
                          Permute::permutation[4], Permute::permutation[5]>;
};
template<typename Permute>
struct Flip<Permute, 1, std::enable_if_t<Permute::size == 8>> {
    using Type = Permute8<Permute::permutation[4], Permute::permutation[5],
                          Permute::permutation[6], Permute::permutation[7],
                          Permute::permutation[0], Permute::permutation[1],
                          Permute::permutation[2], Permute::permutation[3]>;
};
template<typename Permute>
struct Flip<Permute, 0, std::enable_if_t<Permute::size == 16>> {
    using Type = Permute16<Permute::permutation[2],  Permute::permutation[3],
                           Permute::permutation[0],  Permute::permutation[1],
                           Permute::permutation[6],  Permute::permutation[7],
                           Permute::permutation[4],  Permute::permutation[5],
                           Permute::permutation[10], Permute::permutation[11],
                           Permute::permutation[8],  Permute::permutation[9],
                           Permute::permutation[14], Permute::permutation[15],
                           Permute::permutation[12], Permute::permutation[13]>;
};
template<typename Permute>
struct Flip<Permute, 1, std::enable_if_t<Permute::size == 16>> {
    using Type = Permute16<Permute::permutation[4],  Permute::permutation[5],
                           Permute::permutation[6],  Permute::permutation[7],
                           Permute::permutation[0],  Permute::permutation[1],
                           Permute::permutation[2],  Permute::permutation[3],
                           Permute::permutation[12], Permute::permutation[13],
                           Permute::permutation[14],  Permute::permutation[15],
                           Permute::permutation[8], Permute::permutation[9],
                           Permute::permutation[10], Permute::permutation[11]>;
};
template<typename Permute>
struct Flip<Permute, 2, std::enable_if_t<Permute::size == 16>> {
    using Type = Permute16<Permute::permutation[8],  Permute::permutation[9],
                           Permute::permutation[10], Permute::permutation[11],
                           Permute::permutation[12], Permute::permutation[13],
                           Permute::permutation[14], Permute::permutation[15],
                           Permute::permutation[0],  Permute::permutation[1],
                           Permute::permutation[2],  Permute::permutation[3],
                           Permute::permutation[4],  Permute::permutation[5],
                           Permute::permutation[6],  Permute::permutation[7]>;
};

/**
 * @brief Define permutation for exchanging real and imaginary parts
 */
template<typename Permute, typename Enable = void>
struct SwapRealImag;
template<typename Permute>
struct SwapRealImag<Permute, std::enable_if_t<Permute::size == 4>> {
    using Type = Permute4<Permute::permutation[1], Permute::permutation[0],
                          Permute::permutation[3], Permute::permutation[2]>;
};
template<typename Permute>
struct SwapRealImag<Permute, std::enable_if_t<Permute::size == 8>> {
    using Type = Permute8<Permute::permutation[1], Permute::permutation[0],
                          Permute::permutation[3], Permute::permutation[2],
                          Permute::permutation[5], Permute::permutation[4],
                          Permute::permutation[7], Permute::permutation[6]>;
};
template<typename Permute>
struct SwapRealImag<Permute, std::enable_if_t<Permute::size == 16>> {
    using Type = Permute16<Permute::permutation[1],  Permute::permutation[0],
                           Permute::permutation[3],  Permute::permutation[2],
                           Permute::permutation[5],  Permute::permutation[4],
                           Permute::permutation[7],  Permute::permutation[6],
                           Permute::permutation[9],  Permute::permutation[8],
                           Permute::permutation[11], Permute::permutation[10],
                           Permute::permutation[13], Permute::permutation[12],
                           Permute::permutation[15], Permute::permutation[14]>;
};


/**
 * @brief Check whether the permutation is within 128bit lane.
 */
template<typename PrecisionT, class Permute>
constexpr bool isWithinLane() {
    constexpr size_t size_within_lane = 16 / sizeof(PrecisionT);
    constexpr auto permutation = Permute::permutation;

    std::array<uint32_t, size_within_lane> lane = {0,};
    for(size_t i = 0; i < size_within_lane; i++) {
        lane[i] = permutation[i];
    }
    {
        auto lane2 = lane;
        Internal::bubble_sort(lane2.begin(), lane2.end());
        for(size_t i = 0; i < size_within_lane; i++) {
            if(lane2[i] != i) {
                return false;
            }
        }
    }

    for(size_t k = 0; k < permutation.size(); k+= size_within_lane) {
        for(size_t idx = 0; idx < size_within_lane; idx++) {
            if (lane[idx] + k != permutation[idx + k]) {
                return false;
            }
        }
    }
    return true;
}

template<class Permute>
constexpr uint8_t getPermutation2x() {
    static_assert(Permute::size >= 2);
    constexpr auto permutation = Permute::permutation;

    uint32_t res = 0U;
    for(int idx = 1; idx >= 0; idx--) {
        res <<= 1U;
        res |= (permutation[idx] << 1);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return (res << 6U) | (res << 4U) | (res << 2U) | res;
}

template<class Permute>
constexpr uint8_t getPermutation4x() {
    static_assert(Permute::size >= 4);
    constexpr auto permutation = Permute::permutation;

    uint8_t res = 0;
    for(int idx = 3; idx >= 0; idx--) {
        res <<= 2U;
        res |= (permutation[idx] & 3U);
    }
    return res;
}

template<class Permute>
__m256i getPermutation8x256i() {
    static_assert(Permute::size == 8);
    constexpr auto& permutation = Permute::permutation;
    return _mm256_setr_epi32(permutation[0], permutation[1],
                             permutation[2], permutation[3],
                             permutation[4], permutation[5],
                             permutation[6], permutation[7]);
}
template<class Permute>
__m512i getPermutation8x512i() {
    static_assert(Permute::size == 8);
    constexpr auto& permutation = Permute::permutation;
    return _mm512_setr_epi64(permutation[0], permutation[1],
                             permutation[2], permutation[3],
                             permutation[4], permutation[5],
                             permutation[6], permutation[7]);
}

template<class Permute>
__m512i getPermutation16x512i() {
    static_assert(Permute::size == 16);
    constexpr auto& permutation = Permute::permutation;
    return _mm512_setr_epi32(permutation[0], permutation[1],
                             permutation[2], permutation[3],
                             permutation[4], permutation[5],
                             permutation[6], permutation[7],
                             permutation[8], permutation[9],
                             permutation[10], permutation[11],
                             permutation[12], permutation[13],
                             permutation[14], permutation[15]);
}

template<class Permute>
auto permuteIntrinsic(__m256 v) -> __m256 {
    // AVX2 float 
    if constexpr (isWithinLane<float, Permute>()) {
        constexpr uint8_t permute_val = getPermutation4x<Permute>();
        return _mm256_permute_ps(v, permute_val);
    } else {
        return _mm256_permutevar8x32_ps(v, getPermutation8x256i<Permute>());
    }
}
template<class Permute>
auto permuteIntrinsic(__m256d v) -> __m256d {
    // AVX2 double
    if constexpr (isWithinLane<double, Permute>()) {
        constexpr uint8_t permute_val = getPermutation2x<Permute>();
        return _mm256_permute_pd(v, permute_val);
    } else {
        constexpr uint8_t permute_val = getPermutation4x<Permute>();
        return _mm256_permute4x64_pd(v, permute_val);
    }
}
template<class Permute>
auto permuteIntrinsic(__m512 v) -> __m512 {
    if constexpr (isWithinLane<float, Permute>()) {
        constexpr uint8_t permute_val = getPermutation4x<Permute>();
        return _mm512_permute_ps(v, permute_val);
    } else {
        return _mm512_permutexvar_ps(getPermutation16x512i<Permute>(), v);
    }
}
template<class Permute>
auto permuteIntrinsic(__m512d v) -> __m512d {
    if constexpr (isWithinLane<double, Permute>()) {
        constexpr uint8_t permute_val = getPermutation2x<Permute>();
        return _mm512_permute_pd(v, permute_val);
    } else {
        return _mm512_permutexvar_pd(getPermutation8x512i<Permute>(), v);
    }
}
} // namespace Pennylane::Gates::AVX
