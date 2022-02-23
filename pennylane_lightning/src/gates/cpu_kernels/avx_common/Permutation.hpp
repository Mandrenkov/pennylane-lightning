
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

namespace Pennylane::Gates::AVX::Permutation {

/// @cond DEV
namespace Internal {
/**
 * @brief Custom bubble sort. Let's use this until we have constexpr
 * std::sort in C++20.
 */
template <typename iterator>
constexpr void bubble_sort(iterator begin, iterator end) {
    bool swapped = false;
    auto n = std::distance(begin, end);
    do {
        swapped = false;
        for (typename std::iterator_traits<iterator>::difference_type idx = 0;
             idx < n - 1; idx++) {
            if (*(begin + idx) > *(begin + idx + 1)) {
                const auto tmp = *(begin + idx + 1);
                *(begin + idx + 1) = *(begin + idx);
                *(begin + idx) = tmp;
                swapped = true;
            }
        }
    } while (swapped);
}
} // namespace Internal
/// @endcond

/**
 * @brief Maintain permutation related data in a compile time.
 *
 * TODO: This must be cleaned in C++20.
 */
template <typename PrecisionT, size_t packed_size> struct CompiledPermutation {
    using PrecisionT_ = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;

    const bool within_lane_;
    const int imm8_ = 0;
    __m256i permute256_ = {
        0,
    };
    __m512i permute512_ = {
        0,
    };

    constexpr CompiledPermutation(bool within_lane, uint8_t imm8)
        : within_lane_{within_lane}, imm8_{imm8} {}

    constexpr CompiledPermutation(bool within_lane, __m256i permute)
        : within_lane_{within_lane}, permute256_{permute} {}

    constexpr CompiledPermutation(bool within_lane, __m512i permute)
        : within_lane_{within_lane}, permute512_{permute} {}
};

template <bool within_lane, int imm8, class PrecisionT, size_t packed_size>
auto permute(const CompiledPermutation<PrecisionT, packed_size> &perm,
             const AVXIntrinsicType<PrecisionT, packed_size> &v) {
    if constexpr (within_lane) {
        if constexpr (sizeof(PrecisionT) * packed_size == 32) {
            if constexpr (std::is_same_v<float, PrecisionT>) {
                return _mm256_permute_ps(v, imm8);
            } else {
                constexpr static int imm8_trunc =
                    imm8 % 16; // to suppress error
                return _mm256_permute_pd(v, imm8_trunc);
            }
        } else if (sizeof(PrecisionT) * packed_size == 64) {
            if constexpr (std::is_same_v<float, PrecisionT>) {
                return _mm512_permute_ps(v, imm8);
            } else {
                return _mm512_permute_pd(v, imm8);
            }
        }
    } else {
        if constexpr (sizeof(PrecisionT) * packed_size == 32) {
            // AVX2
            if constexpr (std::is_same_v<PrecisionT, float>) {
                return _mm256_permutevar8x32_ps(v, perm.permute256_);
            } else if (std::is_same_v<PrecisionT, double>) {
                return _mm256_permute4x64_pd(v, imm8);
            }
        } else if (sizeof(PrecisionT) * packed_size == 64) {
            // AVX512
            if constexpr (std::is_same_v<PrecisionT, float>) {
                return _mm512_permutexvar_ps(perm.permute512_, v);
            } else if (std::is_same_v<PrecisionT, double>) {
                return _mm512_permutexvar_pd(perm.permute512_, v);
            }
        }
    }
}

template <typename PrecisionT, size_t size>
constexpr bool isWithinLane(const std::array<uint8_t, size> &permutation) {
    constexpr size_t size_within_lane = 16 / sizeof(PrecisionT);

    std::array<uint32_t, size_within_lane> lane = {
        0,
    };
    for (size_t i = 0; i < size_within_lane; i++) {
        lane[i] = permutation[i];
    }
    {
        auto lane2 = lane;
        Internal::bubble_sort(lane2.begin(), lane2.end());
        for (size_t i = 0; i < size_within_lane; i++) {
            if (lane2[i] != i) {
                return false;
            }
        }
    }

    for (size_t k = 0; k < permutation.size(); k += size_within_lane) {
        for (size_t idx = 0; idx < size_within_lane; idx++) {
            if (lane[idx] + k != permutation[idx + k]) {
                return false;
            }
        }
    }
    return true;
}

template <size_t size>
constexpr uint8_t
getPermutation2x(const std::array<uint8_t, size> &permutation) {
    uint8_t res = static_cast<uint8_t>(permutation[1] << 1U) | permutation[0];
    // NOLINTNEXTLINE(readability-magic-numbers, hicpp-signed-bitwise)
    return (res << 6U) | (res << 4U) | (res << 2U) | res;
}

template <size_t size>
constexpr uint8_t
getPermutation4x(const std::array<uint8_t, size> &permutation) {
    uint8_t res = 0;
    for (int idx = 3; idx >= 0; idx--) {
        res <<= 2U;
        res |= (permutation[idx]);
    }
    return res;
}
// clang-format off
constexpr __m256i getPermutation8x256i(const std::array<uint8_t, 8>& permutation) {
    return setr256i(permutation[0], permutation[1], // NOLINT(readability-magic-numbers)
				    permutation[2], permutation[3], // NOLINT(readability-magic-numbers)
				    permutation[4], permutation[5], // NOLINT(readability-magic-numbers)
				    permutation[6], permutation[7]);// NOLINT(readability-magic-numbers)
}
constexpr __m512i getPermutation8x512i(const std::array<uint8_t, 8>& permutation) {
    return setr512i(permutation[0], permutation[1], // NOLINT(readability-magic-numbers)
                    permutation[2], permutation[3], // NOLINT(readability-magic-numbers)
                    permutation[4], permutation[5], // NOLINT(readability-magic-numbers)
                    permutation[6], permutation[7]);// NOLINT(readability-magic-numbers)
}

constexpr __m512i getPermutation16x512i(const std::array<uint8_t, 16>& permutation) {
    return setr512i( permutation[0],  permutation[1],  // NOLINT(readability-magic-numbers)
                     permutation[2],  permutation[3],  // NOLINT(readability-magic-numbers)
                     permutation[4],  permutation[5],  // NOLINT(readability-magic-numbers)
                     permutation[6],  permutation[7],  // NOLINT(readability-magic-numbers)
                     permutation[8],  permutation[9],  // NOLINT(readability-magic-numbers)
                    permutation[10], permutation[11],  // NOLINT(readability-magic-numbers)
                    permutation[12], permutation[13],  // NOLINT(readability-magic-numbers)
                    permutation[14], permutation[15]); // NOLINT(readability-magic-numbers)
}
// clang-format on

template <typename PrecisionT, size_t packed_size>
constexpr auto
compilePermutation(const std::array<uint8_t, packed_size> &permutation)
    -> CompiledPermutation<PrecisionT, packed_size> {
    bool within_lane = isWithinLane<PrecisionT>(permutation);

    if (within_lane) {
        // within lane
        if constexpr (std::is_same_v<PrecisionT, float>) {
            int imm8 = getPermutation4x(permutation);
            return CompiledPermutation<PrecisionT, packed_size>(within_lane,
                                                                imm8);
        } else if constexpr (std::is_same_v<PrecisionT, double>) {
            int permute_val = getPermutation2x(permutation);
            return CompiledPermutation<PrecisionT, packed_size>(within_lane,
                                                                permute_val);
        }
    } else {
        // across lane
        if constexpr (sizeof(PrecisionT) * packed_size == 32) {
            // AVX2
            if constexpr (std::is_same_v<PrecisionT, float>) {
                return CompiledPermutation<PrecisionT, packed_size>(
                    within_lane, getPermutation8x256i(permutation));
            } else if (std::is_same_v<PrecisionT, double>) {
                return CompiledPermutation<PrecisionT, packed_size>(
                    within_lane, getPermutation4x(permutation));
            }
        } else if (sizeof(PrecisionT) * packed_size == 64) {
            // AVX512
            if constexpr (std::is_same_v<PrecisionT, float>) {
                return CompiledPermutation<PrecisionT, packed_size>(
                    within_lane, getPermutation16x512i(permutation));
            } else if (std::is_same_v<PrecisionT, double>) {
                return CompiledPermutation<PrecisionT, packed_size>(
                    within_lane, getPermutation8x512i(permutation));
            }
        }
    }
}

template <size_t packed_size>
constexpr auto identity() -> std::array<uint8_t, packed_size> {
    std::array<uint8_t, packed_size> res = {
        0,
    };
    for (uint8_t i = 0; i < packed_size; i++) {
        res[i] = i;
    }
    return res;
}

template <size_t packed_size>
constexpr auto flip(const std::array<uint8_t, packed_size> &perm,
                    size_t rev_wire) -> std::array<uint8_t, packed_size> {
    std::array<uint8_t, packed_size> res = {
        0,
    };

    for (uint8_t k = 0; k < packed_size / 2; k++) {
        res[2 * k + 0] = perm[2 * (k ^ (1U << rev_wire)) + 0];
        res[2 * k + 1] = perm[2 * (k ^ (1U << rev_wire)) + 1];
    }
    return res;
}
template <size_t packed_size>
constexpr auto swapRealImag(const std::array<uint8_t, packed_size> &perm)
    -> std::array<uint8_t, packed_size> {
    std::array<uint8_t, packed_size> res = {
        0,
    };

    for (uint8_t k = 0; k < packed_size / 2; k++) {
        res[2 * k + 0] = perm[2 * k + 1];
        res[2 * k + 1] = perm[2 * k + 0];
    }
    return res;
}
} // namespace Pennylane::Gates::AVX::Permutation
