// Copyright 2021 Xanadu Quantum Technologies Inc.

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
 * Defines kernel functions with less memory (and fast)
 */
#pragma once

#include "BitUtil.hpp"
#include "Error.hpp"
#include "GateImplementationsLM.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "Macros.hpp"

#include <immintrin.h>

#include <complex>
#include <vector>

namespace Pennylane::Gates {

namespace Internal {

template <class PrecisionT> struct AVX512Intrinsic;

template <> struct AVX512Intrinsic<float> { using Type = __m512; };

template <> struct AVX512Intrinsic<double> { using Type = __m512d; };

template <class PrecisionT>
using AVX512IntrinsicType = typename AVX512Intrinsic<PrecisionT>::Type;

constexpr uint8_t parity(size_t n, size_t rev_wire) {
    return static_cast<uint8_t>((n >> rev_wire) & 1U);
}
constexpr uint8_t parity(size_t n, size_t rev_wire0, size_t rev_wire1) {
    return static_cast<uint8_t>((n >> rev_wire0) & 1U) ^
           static_cast<uint8_t>((n >> rev_wire1) & 1U);
}

inline __m512 parityS(size_t n, size_t rev_wire) {
    return _mm512_setr_ps(parity(n + 0, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 0, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 1, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 1, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 2, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 2, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 3, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 3, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 4, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 4, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 5, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 5, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 6, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 6, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 7, rev_wire) ? -1.0F : 1.0F,
                          parity(n + 7, rev_wire) ? -1.0F : 1.0F);
}
inline __m512 parityS(size_t n, size_t rev_wire0, size_t rev_wire1) {
    return _mm512_setr_ps(parity(n + 0, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 0, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 1, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 1, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 2, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 2, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 3, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 3, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 4, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 4, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 5, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 5, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 6, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 6, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 7, rev_wire0, rev_wire1) ? -1.0F : 1.0F,
                          parity(n + 7, rev_wire0, rev_wire1) ? -1.0F : 1.0F);
}

inline __m512d parityD(size_t n, size_t rev_wire) {
    return _mm512_setr_pd(parity(n + 0, rev_wire) ? -1.0L : 1.0L,
                          parity(n + 0, rev_wire) ? -1.0L : 1.0L,
                          parity(n + 1, rev_wire) ? -1.0L : 1.0L,
                          parity(n + 1, rev_wire) ? -1.0L : 1.0L,
                          parity(n + 2, rev_wire) ? -1.0L : 1.0L,
                          parity(n + 2, rev_wire) ? -1.0L : 1.0L,
                          parity(n + 3, rev_wire) ? -1.0L : 1.0L,
                          parity(n + 3, rev_wire) ? -1.0L : 1.0L);
}

inline __m512d parityD(size_t n, size_t rev_wire0, size_t rev_wire1, 
        const __m512i ones_epi64) {
    const auto indices = _mm512_setr_epi64(n+0, n+0, n+1, n+1, n+2, n+2, n+3, n+3);
    auto parities = _mm512_xor_epi64(_mm512_srli_epi64(indices, rev_wire0),
                                     _mm512_srli_epi64(indices, rev_wire1));
    parities = _mm512_and_epi64(parities, ones_epi64);

    if constexpr (use_avx512dq) {
        parities = _mm512_sub_epi64(ones_epi64,
                                    _mm512_slli_epi64(parities, 1));
        return _mm512_cvtepi64_pd(parities);
    }
    const auto mask = _mm512_cmp_epi64_mask(parities,
                                            ones_epi64,
                                            _MM_CMPINT_NE);
    return _mm512_mask_mov_pd(_mm512_set1_pd(-1.0), mask, _mm512_set1_pd(1.0));
}

template<typename T>
struct ImagFactor {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, 
            "T must be float or double.");
};

template <>
struct ImagFactor<float> {
    // NOLINTNEXTLINE(hicpp-avoid-c-arrays)
    alignas(64) constexpr static float value[16] = {
        -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F,
        -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F,
    };
};

template <>
struct ImagFactor<double> {
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
    return _mm512_mul_ps(prod_shuffled, _mm512_load_ps(&ImagFactor<float>::value));
}

/**
 * @brief Calculate val * 1j
 *
 * @param val Complex values arranged in [i7, r7, ..., i0, r0] where
 * each complex values are r0 + 1j*i0, ... 
 */
inline __m512 productImagS(__m512 val) {
    __m512 prod_shuffled = _mm512_permute_ps(val, 0B10110001);
    return _mm512_mul_ps(prod_shuffled, _mm512_load_ps(&ImagFactor<float>::value));
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
        _mm512_permutex_pd(_mm512_mul_pd(val, factor), 0B10110001);
    return _mm512_mul_pd(prod_shuffled, _mm512_load_pd(&ImagFactor<double>::value));
}

/**
 * @brief Calculate val * 1j
 *
 * @param val Complex values arranged in [i3, r3, ..., i0, r0] where
 * each complex values are r0 + 1j*i0, ... 
 * @param imag_val Value to product. We product 1j*imag_val to val.
 */
inline __m512d productImagD(__m512d val) {
    __m512d prod_shuffled =_mm512_permutex_pd(val, 0B10110001);
    return _mm512_mul_pd(prod_shuffled, _mm512_load_pd(&ImagFactor<double>::value));
}


} // namespace Internal

class GateImplementationsAVX512 {
  private:
    /* Alias utility functions */
    static constexpr auto fillLeadingOnes = Util::fillLeadingOnes;
    static constexpr auto fillTrailingOnes = Util::fillTrailingOnes;

  public:
    constexpr static KernelType kernel_id = KernelType::AVX512;
    constexpr static std::string_view name = "AVX512";
    constexpr static uint32_t data_alignment_in_bytes = 64;

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,
        GateOperation::RZ,
        GateOperation::IsingZZ,
    };

    constexpr static std::array<GeneratorOperation, 0> implemented_generators =
        {};

  private:
    template <size_t rev_wire>
    inline static void applyPauliXFloatInternalOp(__m512& v) {
        if constexpr (rev_wire == 0) {
            v = _mm512_permute_ps(v, 0B01001110);
        } else if (rev_wire == 1) {
            const auto shuffle_idx = _mm512_set_epi32(
                11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
            v = _mm512_permutexvar_ps(shuffle_idx, v);
        } else if (rev_wire == 2) {
            const auto shuffle_idx = _mm512_set_epi32(
                7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
            v = _mm512_permutexvar_ps(shuffle_idx, v);
        }
    }
    template <size_t rev_wire>
    static void applyPauliXFloatInternal(std::complex<float> *arr,
                                           const size_t num_qubits) {
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(float) / 2;
        for (size_t k = 0; k < (1U << num_qubits); k += step) {
            __m512 v = _mm512_load_ps(arr + k);
            applyPauliXFloatInternalOp<rev_wire>(v);
            _mm512_store_ps(arr + k, v);
        }
    }
    static void applyPauliXFloatExternal(std::complex<float> *arr,
                                         const size_t num_qubits,
                                         const size_t rev_wire) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(float) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k += step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512 v0 = _mm512_load_ps(arr + i0);
            const __m512 v1 = _mm512_load_ps(arr + i1);
            _mm512_store_ps(arr + i0, v1);
            _mm512_store_ps(arr + i1, v0);
        }
    }

    template <size_t rev_wire>
    inline static void applyPauliXDoubleInternalOp(__m512d& v) {
        if constexpr (rev_wire == 0) {
            const auto shuffle_idx =
                _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
            v = _mm512_permutexvar_pd(shuffle_idx, v);
        } else if (rev_wire == 1) {
            const auto shuffle_idx =
                _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
            v = _mm512_permutexvar_pd(shuffle_idx, v);
        }
    }

    template <size_t rev_wire>
    static void applyPauliXDoubleInternal(std::complex<double> *arr,
                                          const size_t num_qubits) {
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(double) / 2;
        for (size_t k = 0; k < (1U << num_qubits); k += step) {
            __m512d v = _mm512_load_pd(arr + k);
            applyPauliXDoubleInternalOp<rev_wire>(v);
            _mm512_store_pd(arr + k, v);
        }
    }
    static void applyPauliXDoubleExternal(std::complex<double> *arr,
                                          const size_t num_qubits,
                                          const size_t rev_wire) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(double) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k += step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512d v0 = _mm512_load_pd(arr + i0);
            const __m512d v1 = _mm512_load_pd(arr + i1);
            _mm512_store_pd(arr + i0, v1);
            _mm512_store_pd(arr + i1, v0);
        }
    }

    template <size_t rev_wire>
    static void applyPauliYFloatInternalOp(__m512& v) {
        if constexpr (rev_wire == 0) {
            const auto factor = _mm512_setr_ps(
                    1.0F, -1.0F, -1.0F, 1.0F,
                    1.0F, -1.0F, -1.0F, 1.0F,
                    1.0F, -1.0F, -1.0F, 1.0F,
                    1.0F, -1.0F, -1.0F, 1.0F);
            v = _mm512_permute_ps(v, 0B00011011);
            v = _mm512_mul_ps(v, factor);
        } else if (rev_wire == 1) {
            const auto factor = _mm512_setr_ps(
                    1.0F, -1.0F, 1.0F, -1.0F,
                    -1.0F, 1.0F, -1.0F, 1.0F,
                    1.0F, -1.0F, 1.0F, -1.0F,
                    -1.0F, 1.0F, -1.0F, 1.0F);
            const auto shuffle_idx = _mm512_set_epi32(
                10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5);
            v = _mm512_permutexvar_ps(shuffle_idx, v);
            v = _mm512_mul_ps(v, factor);
        } else if (rev_wire == 2) {
            const auto factor = _mm512_setr_ps(
                    1.0F, -1.0F, 1.0F, -1.0F,
                    1.0F, -1.0F, 1.0F, -1.0F,
                    -1.0F, 1.0F, -1.0F, 1.0F,
                    -1.0F, 1.0F, -1.0F, 1.0F);
            const auto shuffle_idx = _mm512_set_epi32(
                6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9);
            v = _mm512_permutexvar_ps(shuffle_idx, v);
            v = _mm512_mul_ps(v, factor);
        }
    }

    template <size_t rev_wire>
    static void applyPauliYFloatInternal(std::complex<float> *arr,
                                         const size_t num_qubits) {
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(float) / 2;
        for (size_t k = 0; k < (1U << num_qubits); k += step) {
            __m512 v = _mm512_load_ps(arr + k);
            applyPauliYFloatInternalOp<rev_wire>(v);
            _mm512_store_ps(arr + k, v);
        }
    }
    static void applyPauliYFloatExternal(std::complex<float> *arr,
                                           const size_t num_qubits,
                                           const size_t rev_wire) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(float) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k += step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512 v0 = _mm512_load_ps(arr + i0);
            const __m512 v1 = _mm512_load_ps(arr + i1);
            _mm512_store_ps(arr + i0, Internal::productImagS(v1, _mm512_set1_ps(-1.0F)));
            _mm512_store_ps(arr + i1, Internal::productImagS(v0));
        }
    }

    template <size_t rev_wire>
    inline static void applyPauliYDoubleInternalOp(__m512d& v) {
        if constexpr (rev_wire == 0) {
            const auto factor = _mm512_setr_pd(
                1.0L, -1.0L, -1.0L, 1.0L,
                1.0L, -1.0L, -1.0L, 1.0L
            );
            v = _mm512_mul_pd(v, factor);
            const auto shuffle_idx =
                _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
            v = _mm512_permutexvar_pd(shuffle_idx, v);
        } else if (rev_wire == 1) {
            const auto factor = _mm512_setr_pd(
                1.0L, -1.0L, 1.0L, -1.0L,
                -1.0L, 1.0L, -1.0L, 1.0L
            );
            const auto shuffle_idx =
                _mm512_set_epi64(2, 3, 0, 1, 6, 7, 4, 5);
            v = _mm512_permutexvar_pd(shuffle_idx, v);
            v = _mm512_mul_pd(v, factor);
        }
    }

    template <size_t rev_wire>
    static void applyPauliYDoubleInternal(std::complex<double> *arr,
                                            const size_t num_qubits) {
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(double) / 2;
        for (size_t k = 0; k < (1U << num_qubits); k += step) {
            __m512d v = _mm512_load_pd(arr + k);
            applyPauliYDoubleInternalOp<rev_wire>(v);
            _mm512_store_pd(arr + k, v);
        }
    }

    static void applyPauliYDoubleExternal(std::complex<double> *arr,
                                          const size_t num_qubits,
                                          const size_t rev_wire) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        constexpr static auto step =
            data_alignment_in_bytes / sizeof(double) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k += step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512d v0 = _mm512_load_pd(arr + i0);
            const __m512d v1 = _mm512_load_pd(arr + i1);

            _mm512_store_pd(arr + i0, Internal::productImagD(v1, _mm512_set1_pd(-1.0L)));
            _mm512_store_pd(arr + i1, Internal::productImagD(v0));
        }
    }

  public:
    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                applyPauliXFloatInternal<0>(arr, num_qubits);
                return;
            case 1:
                applyPauliXFloatInternal<1>(arr, num_qubits);
                return;
            case 2:
                applyPauliXFloatInternal<2>(arr, num_qubits);
                return;
            default:
                applyPauliXFloatExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                applyPauliXDoubleInternal<0>(arr, num_qubits);
                return;
            case 1:
                applyPauliXDoubleInternal<1>(arr, num_qubits);
                return;
            default:
                applyPauliXDoubleExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliY(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                applyPauliYFloatInternal<0>(arr, num_qubits);
                return;
            case 1:
                applyPauliYFloatInternal<1>(arr, num_qubits);
                return;
            case 2:
                applyPauliYFloatInternal<2>(arr, num_qubits);
                return;
            default:
                applyPauliYFloatExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliY(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                applyPauliYDoubleInternal<0>(arr, num_qubits);
                return;
            case 1:
                applyPauliYDoubleInternal<1>(arr, num_qubits);
                return;
            default:
                applyPauliYDoubleExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
    /* Version iterate over all indices */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
            } else {
                const size_t rev_wire = num_qubits - wires[0] - 1;

                __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
                const float isin =
                    inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
                __m512 imag_sin_factor = _mm512_set_ps(
                    -isin, isin, -isin, isin, -isin, isin, -isin, isin, -isin,
                    isin, -isin, isin, -isin, isin, -isin, isin);
                for (size_t n = 0; n < (1U << num_qubits); n += 8) {
                    __m512 coeffs = _mm512_load_ps(arr + n);
                    __m512 prod_cos = _mm512_mul_ps(real_cos_factor, coeffs);

                    __m512 imag_sin_parity = _mm512_mul_ps(
                        imag_sin_factor, Internal::parityS(n, rev_wire));
                    __m512 prod_sin = _mm512_mul_ps(coeffs, imag_sin_parity);

                    __m512 prod = _mm512_add_ps(
                        prod_cos, _mm512_permute_ps(prod_sin, 0B10110001));
                    _mm512_store_ps(arr + n, prod);
                }
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
            } else {
                const size_t rev_wire = num_qubits - wires[0] - 1;
                __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));

                const double isin =
                    inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
                __m512d imag_sin_factor = _mm512_set_pd(
                    -isin, isin, -isin, isin, -isin, isin, -isin, isin);
                for (size_t n = 0; n < (1U << num_qubits); n += 4) {
                    __m512d coeffs = _mm512_load_pd(arr + n);
                    __m512d prod_cos = _mm512_mul_pd(real_cos_factor, coeffs);

                    __m512d imag_sin_parity = _mm512_mul_pd(
                        imag_sin_factor, Internal::parityD(n, rev_wire));
                    __m512d prod_sin = _mm512_mul_pd(coeffs, imag_sin_parity);

                    __m512d prod = _mm512_add_pd(
                        prod_cos, _mm512_permutex_pd(prod_sin, 0B10110001));
                    _mm512_store_pd(arr + n, prod);
                }
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float or double is supported.");
        }
    }
    /* Version iterate over all indices */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 2);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyIsingZZ(arr, num_qubits, wires,
                                                    inverse, angle);
            } else {
                const size_t rev_wire0 = num_qubits - wires[0] - 1;
                const size_t rev_wire1 = num_qubits - wires[1] - 1;
                __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
                const float isin =
                    inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
                __m512 imag_sin_factor = _mm512_set_ps(
                    -isin, isin, -isin, isin, -isin, isin, -isin, isin, -isin,
                    isin, -isin, isin, -isin, isin, -isin, isin);
                for (size_t n = 0; n < (1U << num_qubits); n += 8) {
                    __m512 coeffs = _mm512_load_ps(arr + n);
                    __m512 prod_cos = _mm512_mul_ps(real_cos_factor, coeffs);

                    __m512 imag_sin_parity = _mm512_mul_ps(
                        imag_sin_factor,
                        Internal::parityS(n, rev_wire0, rev_wire1));
                    __m512 prod_sin = _mm512_mul_ps(coeffs, imag_sin_parity);

                    __m512 prod = _mm512_add_ps(
                        prod_cos, _mm512_permute_ps(prod_sin, 0B10110001));
                    _mm512_store_ps(arr + n, prod);
                }
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyIsingZZ(arr, num_qubits, wires,
                                                    inverse, angle);
            } else {
                const size_t rev_wire0 = num_qubits - wires[0] - 1;
                const size_t rev_wire1 = num_qubits - wires[1] - 1;
                __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));

                const auto ones = _mm512_set1_epi64(1U);

                const double isin =
                    inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
                __m512d imag_sin_factor = _mm512_set_pd(
                    -isin, isin, -isin, isin, -isin, isin, -isin, isin);
                for (size_t n = 0; n < (1U << num_qubits); n += 4) {
                    __m512d coeffs = _mm512_load_pd(arr + n);
                    __m512d prod_cos = _mm512_mul_pd(real_cos_factor, coeffs);

                    __m512d imag_sin_parity = _mm512_mul_pd(
                        imag_sin_factor,
                        Internal::parityD(n, rev_wire0, rev_wire1, ones));
                    __m512d prod_sin = _mm512_mul_pd(coeffs, imag_sin_parity);

                    __m512d prod = _mm512_add_pd(
                        prod_cos, _mm512_permutex_pd(prod_sin, 0B10110001));
                    _mm512_store_pd(arr + n, prod);
                }
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float or double is supported.");
        }
    }
};
} // namespace Pennylane::Gates
