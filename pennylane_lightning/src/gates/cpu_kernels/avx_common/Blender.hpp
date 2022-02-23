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
 * Defines utility functions for all AVX blend functions
 */
#pragma once
#include "AVXUtil.hpp"

#include <immintrin.h>

namespace Pennylane::Gates::AVX {

template <typename PrecisionT, size_t num_packed> struct Blender {
    static_assert(sizeof(PrecisionT) == -1,
                  "Unsupported type and/or packed size.");
};

template <> struct Blender<float, 8> {
    // AVX2 with float
    int imm8_ = 0;

    constexpr explicit Blender(const Mask<8> &mask) {
        imm8_ = 0;
        for (uint8_t i = 0; i < 8; i++) {
            imm8_ |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
    }

    template <typename T> __m256 blend(T &&a, T &&b) const {
        return _mm256_blend_ps(std::forward<T>(a), std::forward<T>(b), imm8_);
    }
};

template <> struct Blender<double, 4> {
    // AVX2 with double
    int imm8_ = 0;
    constexpr explicit Blender(const Mask<4> &mask) {
        imm8_ = 0;
        for (uint8_t i = 0; i < 4; i++) {
            imm8_ |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
    }
    template <typename T> __m256d blend(T &&a, T &&b) const {
        return _mm256_blend_pd(std::forward<T>(a), std::forward<T>(b), imm8_);
    }
};
template <> struct Blender<float, 16> {
    // AVX512 with float
    __mmask16 k_;

    explicit Blender(const Mask<16> &mask) {
        uint16_t m = 0;
        for (uint8_t i = 0; i < 16; i++) {
            m |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
        k_ = m;
    }
    template <typename T> __m512 blend(T &&a, T &&b) const {
        return _mm512_mask_blend_ps(k_, std::forward<T>(a), std::forward<T>(b));
    }
};

template <> struct Blender<double, 8> {
    // AVX512 with double
    __mmask8 k_;

    explicit Blender(const Mask<8> &mask) {
        uint8_t m = 0;
        for (uint8_t i = 0; i < 8; i++) {
            m |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
        k_ = m;
    }
    template <typename T> __m512d blend(T &&a, T &&b) const {
        return _mm512_mask_blend_pd(k_, std::forward<T>(a), std::forward<T>(b));
    }
};
} // namespace Pennylane::Gates::AVX
