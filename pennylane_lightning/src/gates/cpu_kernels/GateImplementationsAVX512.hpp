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
 * Defines kernel functions with AVX512F and AVX512DQ
 */
#pragma once

#include "avx512/ApplyHadamard.hpp"
#include "avx512/ApplyPauliX.hpp"
#include "avx512/ApplyPauliY.hpp"
#include "avx512/ApplyPauliZ.hpp"
#include "avx512/ApplyS.hpp"

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

/// @cond DEV
namespace Internal {
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
    }
    // clang-format on
    PL_UNREACHABLE;
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
    }
    // clang-format on
    PL_UNREACHABLE;
    return _mm512_setzero_pd();
}

} // namespace Internal
/// @endcond

class GateImplementationsAVX512 {
  private:
    /* Alias utility functions */
    static constexpr auto fillLeadingOnes = Util::fillLeadingOnes;
    static constexpr auto fillTrailingOnes = Util::fillTrailingOnes;

  public:
    constexpr static KernelType kernel_id = KernelType::AVX512;
    constexpr static std::string_view name = "AVX512";
    constexpr static uint32_t data_alignment_in_bytes = 64;

    template <typename T>
    constexpr static size_t
        step_for_complex_precision = data_alignment_in_bytes / sizeof(T) / 2;

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX, GateOperation::PauliY,  GateOperation::PauliZ,
        GateOperation::Hadamard, GateOperation::S,
        GateOperation::RZ,     GateOperation::Rot, GateOperation::IsingZZ,
    };

    constexpr static std::array<GeneratorOperation, 0> implemented_generators =
        {};

  private:
    template <class ParamT>
    static void
    applyRZFloatInternal(std::complex<float> *arr, const size_t num_qubits,
                         const size_t rev_wire, [[maybe_unused]] bool inverse,
                         ParamT angle) {
        const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
        const __m512 imag_sin_factor =
            _mm512_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin,
                          -isin, isin, -isin, isin, -isin, isin, -isin, isin);

        const __m512 imag_sin_parity =
            _mm512_mul_ps(imag_sin_factor, Internal::paritySInternal(rev_wire));

        for (size_t n = 0; n < (1U << num_qubits); n += 8) {
            __m512 coeffs = _mm512_load_ps(arr + n);
            __m512 prod_cos = _mm512_mul_ps(real_cos_factor, coeffs);

            __m512 prod_sin = _mm512_mul_ps(coeffs, imag_sin_parity);

            __m512 prod = _mm512_add_ps(
                prod_cos, _mm512_permute_ps(prod_sin, 0B10110001));
            _mm512_store_ps(arr + n, prod);
        }
    }
    template <class ParamT>
    static void
    applyRZFloatExternal(std::complex<float> *arr, const size_t num_qubits,
                         const size_t rev_wire, [[maybe_unused]] bool inverse,
                         ParamT angle) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
        const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        constexpr static auto step =
            data_alignment_in_bytes / sizeof(float) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k += step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512 v0 = _mm512_load_ps(arr + i0);
            const __m512 v1 = _mm512_load_ps(arr + i1);

            const auto v0_cos = _mm512_mul_ps(v0, real_cos_factor);
            const auto v0_isin =
                Internal::productImagS(v0, _mm512_set1_ps(isin));

            const auto v1_cos = _mm512_mul_ps(v1, real_cos_factor);
            const auto v1_isin =
                Internal::productImagS(v1, _mm512_set1_ps(-isin));

            _mm512_store_ps(arr + i0, _mm512_add_ps(v0_cos, v0_isin));
            _mm512_store_ps(arr + i1, _mm512_add_ps(v1_cos, v1_isin));
        }
    }

    template <class ParamT>
    static void
    applyRZDoubleInternal(std::complex<double> *arr, const size_t num_qubits,
                          const size_t rev_wire, [[maybe_unused]] bool inverse,
                          ParamT angle) {
        const double isin =
            inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
        const __m512d imag_sin_factor =
            _mm512_set_pd(-isin, isin, -isin, isin, -isin, isin, -isin, isin);

        const __m512d imag_sin_parity =
            _mm512_mul_pd(imag_sin_factor, Internal::parityDInternal(rev_wire));

        for (size_t n = 0; n < (1U << num_qubits); n += 4) {
            __m512d coeffs = _mm512_load_pd(arr + n);
            __m512d prod_cos = _mm512_mul_pd(real_cos_factor, coeffs);

            __m512d prod_sin = _mm512_mul_pd(coeffs, imag_sin_parity);

            __m512d prod = _mm512_add_pd(
                prod_cos, _mm512_permutex_pd(prod_sin, 0B10110001));
            _mm512_store_pd(arr + n, prod);
        }
    }
    template <class ParamT>
    static void
    applyRZDoubleExternal(std::complex<double> *arr, const size_t num_qubits,
                          const size_t rev_wire, [[maybe_unused]] bool inverse,
                          ParamT angle) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
        const double isin =
            inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        constexpr static auto step =
            data_alignment_in_bytes / sizeof(double) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k += step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512d v0 = _mm512_load_pd(arr + i0);
            const __m512d v1 = _mm512_load_pd(arr + i1);

            const auto v0_cos = _mm512_mul_pd(v0, real_cos_factor);
            const auto v0_isin =
                Internal::productImagD(v0, _mm512_set1_pd(isin));

            const auto v1_cos = _mm512_mul_pd(v1, real_cos_factor);
            const auto v1_isin =
                Internal::productImagD(v1, _mm512_set1_pd(-isin));

            _mm512_store_pd(arr + i0, _mm512_add_pd(v0_cos, v0_isin));
            _mm512_store_pd(arr + i1, _mm512_add_pd(v1_cos, v1_isin));
        }
    }

    template <class ParamT>
    static void applyIsingZZFloatInternalInternal(std::complex<float> *arr,
                                                  size_t num_qubits,
                                                  size_t rev_wire0,
                                                  size_t rev_wire1,
                                                  bool inverse, ParamT angle) {

        const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        __m512 parity;

        // clang-format off
        switch(rev_wire0 ^ rev_wire1) {
        /* Possible values are (max_rev_wire, min_rev_wire) =
         *     {(1, 0) = 1, (2, 0) = 2, (2, 1) = 3}
         * */
        case 1: // (1,0)
            parity = _mm512_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                                    -1.0F, -1.0F, 1.0F, 1.0F,
                                    1.0F, 1.0F, -1.0F, -1.0F,
                                    -1.0F, -1.0F, 1.0F, 1.0F);
            break;
        case 2: // (2,0)
            parity = _mm512_setr_ps(1.0F, 1.0F, -1.0F, -1.0F,
                                    1.0F, 1.0F, -1.0F, -1.0F,
                                    -1.0F, -1.0F, 1.0F, 1.0F,
                                    -1.0F, -1.0F, 1.0F, 1.0F);
            break;
        case 3: // (2,1)
            parity = _mm512_setr_ps(1.0F, 1.0F, 1.0F, 1.0F,
                                    -1.0F, -1.0F, -1.0F, -1.0F,
                                    -1.0F, -1.0F, -1.0F, -1.0F,
                                    1.0F, 1.0F, 1.0F, 1.0F);
            break;
        default:
            PL_UNREACHABLE;
        }
        // clang-format on
        const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
        const __m512 imag_sin_factor =
            _mm512_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin,
                          -isin, isin, -isin, isin, -isin, isin, -isin, isin);
        const __m512 imag_sin_parity = _mm512_mul_ps(imag_sin_factor, parity);

        for (size_t n = 0; n < (1U << num_qubits); n += 8) {
            __m512 coeffs = _mm512_load_ps(arr + n);
            __m512 prod_cos = _mm512_mul_ps(real_cos_factor, coeffs);

            __m512 prod_sin = _mm512_mul_ps(coeffs, imag_sin_parity);

            __m512 prod = _mm512_add_ps(
                prod_cos, _mm512_permute_ps(prod_sin, 0B10110001));
            _mm512_store_ps(arr + n, prod);
        }
    }
    template <class ParamT>
    static void applyIsingZZFloatInternalExternal(std::complex<float> *arr,
                                                  size_t num_qubits,
                                                  size_t rev_wire0,
                                                  size_t rev_wire1,
                                                  bool inverse, ParamT angle) {
        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t max_rev_wire_shift =
            (static_cast<size_t>(1U) << rev_wire_max);
        const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
        const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

        const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));
        const __m512 imag_sin_factor =
            _mm512_set_ps(-isin, isin, -isin, isin, -isin, isin, -isin, isin,
                          -isin, isin, -isin, isin, -isin, isin, -isin, isin);

        const __m512 imag_sin_parity0 = _mm512_mul_ps(
            imag_sin_factor, Internal::paritySInternal(rev_wire_min));
        const __m512 imag_sin_parity1 =
            _mm512_mul_ps(imag_sin_parity0, _mm512_set1_ps(-1.0L));

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k += 8) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const __m512 v0 = _mm512_load_ps(arr + i0);
            const __m512 v1 = _mm512_load_ps(arr + i1);

            __m512 prod_cos0 = _mm512_mul_ps(real_cos_factor, v0);
            __m512 prod_sin0 = _mm512_mul_ps(v0, imag_sin_parity0);

            __m512 prod0 = _mm512_add_ps(
                prod_cos0, _mm512_permute_ps(prod_sin0, 0B10110001));

            __m512 prod_cos1 = _mm512_mul_ps(real_cos_factor, v1);
            __m512 prod_sin1 = _mm512_mul_ps(v1, imag_sin_parity1);

            __m512 prod1 = _mm512_add_ps(
                prod_cos1, _mm512_permute_ps(prod_sin1, 0B10110001));

            _mm512_store_ps(arr + i0, prod0);
            _mm512_store_ps(arr + i1, prod1);
        }
    }

    template <class ParamT>
    static void applyIsingZZFloatExternalExternal(std::complex<float> *arr,
                                                  const size_t num_qubits,
                                                  const size_t rev_wire0,
                                                  const size_t rev_wire1,
                                                  bool inverse, ParamT angle) {
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const float isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const __m512 real_cos_factor = _mm512_set1_ps(std::cos(angle / 2));

        for (size_t k = 0; k < Util::exp2(num_qubits - 2); k += 8) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            __m512 v = _mm512_load_ps(arr + i00); // 00
            __m512 prod_cos = _mm512_mul_ps(real_cos_factor, v);
            __m512 prod_isin = Internal::productImagS(v, _mm512_set1_ps(isin));
            _mm512_store_ps(arr + i00, _mm512_add_ps(prod_cos, prod_isin));

            v = _mm512_load_ps(arr + i01); // 01
            prod_cos = _mm512_mul_ps(real_cos_factor, v);
            prod_isin = Internal::productImagS(v, _mm512_set1_ps(-isin));
            _mm512_store_ps(arr + i01, _mm512_add_ps(prod_cos, prod_isin));

            v = _mm512_load_ps(arr + i10); // 10
            prod_cos = _mm512_mul_ps(real_cos_factor, v);
            prod_isin = Internal::productImagS(v, _mm512_set1_ps(-isin));
            _mm512_store_ps(arr + i10, _mm512_add_ps(prod_cos, prod_isin));

            v = _mm512_load_ps(arr + i11); // 11
            prod_cos = _mm512_mul_ps(real_cos_factor, v);
            prod_isin = Internal::productImagS(v, _mm512_set1_ps(isin));
            _mm512_store_ps(arr + i11, _mm512_add_ps(prod_cos, prod_isin));
        }
    }

    template <class ParamT>
    static void applyIsingZZDoubleInternalInternal(
        std::complex<double> *arr, size_t num_qubits,
        [[maybe_unused]] size_t rev_wire0, [[maybe_unused]] size_t rev_wire1,
        bool inverse, ParamT angle) {

        // Only rev_wires = (0, 1) is allowed

        const double isin =
            inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const __m512d parity =
            _mm512_setr_pd(1.0L, 1.0L, -1.0L, -1.0L, -1.0L, -1.0L, 1.0L, 1.0L);
        const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
        const __m512d imag_sin_factor =
            _mm512_set_pd(-isin, isin, -isin, isin, -isin, isin, -isin, isin);
        const __m512d imag_sin_parity = _mm512_mul_pd(imag_sin_factor, parity);

        for (size_t n = 0; n < (1U << num_qubits);
             n += step_for_complex_precision<double>) {
            __m512d coeffs = _mm512_load_pd(arr + n);
            __m512d prod_cos = _mm512_mul_pd(real_cos_factor, coeffs);

            __m512d prod_sin = _mm512_mul_pd(coeffs, imag_sin_parity);

            __m512d prod = _mm512_add_pd(
                prod_cos, _mm512_permutex_pd(prod_sin, 0B10110001));
            _mm512_store_pd(arr + n, prod);
        }
    }
    template <class ParamT>
    static void applyIsingZZDoubleInternalExternal(std::complex<double> *arr,
                                                   size_t num_qubits,
                                                   size_t rev_wire0,
                                                   size_t rev_wire1,
                                                   bool inverse, ParamT angle) {
        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t max_rev_wire_shift =
            (static_cast<size_t>(1U) << rev_wire_max);
        const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
        const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

        const double isin =
            inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));
        const __m512d imag_sin_factor =
            _mm512_set_pd(-isin, isin, -isin, isin, -isin, isin, -isin, isin);

        const __m512d imag_sin_parity0 = _mm512_mul_pd(
            imag_sin_factor, Internal::parityDInternal(rev_wire_min));
        const __m512d imag_sin_parity1 =
            _mm512_mul_pd(imag_sin_parity0, _mm512_set1_pd(-1.0L));

        for (size_t k = 0; k < Util::exp2(num_qubits - 1);
             k += step_for_complex_precision<double>) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const __m512d v0 = _mm512_load_pd(arr + i0);
            const __m512d v1 = _mm512_load_pd(arr + i1);

            __m512d prod_cos0 = _mm512_mul_pd(real_cos_factor, v0);
            __m512d prod_sin0 = _mm512_mul_pd(v0, imag_sin_parity0);

            __m512d prod0 = _mm512_add_pd(
                prod_cos0, _mm512_permutex_pd(prod_sin0, 0B10110001));

            __m512d prod_cos1 = _mm512_mul_pd(real_cos_factor, v1);
            __m512d prod_sin1 = _mm512_mul_pd(v1, imag_sin_parity1);

            __m512d prod1 = _mm512_add_pd(
                prod_cos1, _mm512_permutex_pd(prod_sin1, 0B10110001));

            _mm512_store_pd(arr + i0, prod0);
            _mm512_store_pd(arr + i1, prod1);
        }
    }

    template <class ParamT>
    static void applyIsingZZDoubleExternalExternal(std::complex<double> *arr,
                                                   const size_t num_qubits,
                                                   const size_t rev_wire0,
                                                   const size_t rev_wire1,
                                                   bool inverse, ParamT angle) {
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const double isin =
            inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const __m512d real_cos_factor = _mm512_set1_pd(std::cos(angle / 2));

        for (size_t k = 0; k < Util::exp2(num_qubits - 2);
             k += step_for_complex_precision<double>) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const __m512d v00 = _mm512_load_pd(arr + i00); // 00
            const __m512d v01 = _mm512_load_pd(arr + i01); // 01
            const __m512d v10 = _mm512_load_pd(arr + i10); // 10
            const __m512d v11 = _mm512_load_pd(arr + i11); // 11

            const __m512d prod_cos00 = _mm512_mul_pd(real_cos_factor, v00);
            const __m512d prod_isin00 =
                Internal::productImagD(v00, _mm512_set1_pd(isin));

            const __m512d prod_cos01 = _mm512_mul_pd(real_cos_factor, v01);
            const __m512d prod_isin01 =
                Internal::productImagD(v01, _mm512_set1_pd(-isin));

            const __m512d prod_cos10 = _mm512_mul_pd(real_cos_factor, v10);
            const __m512d prod_isin10 =
                Internal::productImagD(v10, _mm512_set1_pd(-isin));

            const __m512d prod_cos11 = _mm512_mul_pd(real_cos_factor, v11);
            const __m512d prod_isin11 =
                Internal::productImagD(v11, _mm512_set1_pd(isin));

            _mm512_store_pd(arr + i00, _mm512_add_pd(prod_cos00, prod_isin00));
            _mm512_store_pd(arr + i01, _mm512_add_pd(prod_cos01, prod_isin01));
            _mm512_store_pd(arr + i10, _mm512_add_pd(prod_cos10, prod_isin10));
            _mm512_store_pd(arr + i11, _mm512_add_pd(prod_cos11, prod_isin11));
        }
    }

  public:
    /*
    static void applySingleQubitOp(std::complex<float> *arr,
                                   const size_t num_qubits,
                                   const std::complex<float> *matrix,
                                   const size_t wire, bool inverse = false) {
        const size_t rev_wire = num_qubits - wire - 1;

        if (num_qubits < 3) {
            GateImplementationsLM::applySingleQubitOp(arr, num_qubits, matrix,
                                                      wire, inverse);
            return;
        }

        switch (rev_wire) {
        case 0:
            AVX512::applySingleQubitOpInternal<0>(arr, num_qubits, matrix, inverse);
            return;
        case 1:
            AVX512::applySingleQubitOpInternal<1>(arr, num_qubits, matrix, inverse);
            return;
        case 2:
            AVX512::applySingleQubitOpInternal<2>(arr, num_qubits, matrix, inverse);
            return;
        default:
            AVX512::applySingleQubitOpExternal(arr, num_qubits, rev_wire, matrix,
                                       inverse);
            return;
        }
    }
    static void applySingleQubitOp(std::complex<double> *arr,
                                   const size_t num_qubits,
                                   const std::complex<double> *matrix,
                                   const size_t wire, bool inverse = false) {
        const size_t rev_wire = num_qubits - wire - 1;

        if (num_qubits < 2) {
            GateImplementationsLM::applySingleQubitOp(arr, num_qubits, matrix,
                                                      wire, inverse);
            return;
        }

        switch (rev_wire) {
        case 0:
            AVX512::applySingleQubitOpInternal<0>(arr, num_qubits, matrix, inverse);
            return;
        case 1:
            AVX512::applySingleQubitOpInternal<1>(arr, num_qubits, matrix, inverse);
            return;
        default:
            AVX512::applySingleQubitOpExternal(arr, num_qubits, rev_wire, matrix,
                                       inverse);
            return;
        }
    }
    */

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
                AVX512::applyPauliXFloatInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX512::applyPauliXFloatInternal<1>(arr, num_qubits);
                return;
            case 2:
                AVX512::applyPauliXFloatInternal<2>(arr, num_qubits);
                return;
            default:
                AVX512::applyPauliXFloatExternal(arr, num_qubits, rev_wire);
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
                AVX512::applyPauliXDoubleInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX512::applyPauliXDoubleInternal<1>(arr, num_qubits);
                return;
            default:
                AVX512::applyPauliXDoubleExternal(arr, num_qubits, rev_wire);
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
                AVX512::applyPauliYFloatInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX512::applyPauliYFloatInternal<1>(arr, num_qubits);
                return;
            case 2:
                AVX512::applyPauliYFloatInternal<2>(arr, num_qubits);
                return;
            default:
                AVX512::applyPauliYFloatExternal(arr, num_qubits, rev_wire);
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
                AVX512::applyPauliYDoubleInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX512::applyPauliYDoubleInternal<1>(arr, num_qubits);
                return;
            default:
                AVX512::applyPauliYDoubleExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliZ(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 3) {
                AVX512::applyPauliZFloatInternal(arr, num_qubits, rev_wire);
                return;
            }
            AVX512::applyPauliZFloatExternal(arr, num_qubits, rev_wire);
            return;
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliZ(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 2) {
                AVX512::applyPauliZDoubleInternal(arr, num_qubits, rev_wire);
                return;
            }
            AVX512::applyPauliZDoubleExternal(arr, num_qubits, rev_wire);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr,
                       const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyS(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX512::applySFloatInternal<0>(arr, num_qubits, inverse);
                return;
            case 1:
                AVX512::applySFloatInternal<1>(arr, num_qubits, inverse);
                return;
            case 2:
                AVX512::applySFloatInternal<2>(arr, num_qubits, inverse);
                return;
            default:
                AVX512::applySFloatExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyS(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX512::applySDoubleInternal<0>(arr, num_qubits, inverse);
                return;
            case 1:
                AVX512::applySDoubleInternal<1>(arr, num_qubits, inverse);
                return;
            default:
                AVX512::applySDoubleExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyHadamard(arr, num_qubits, wires, inverse);
                return ;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch(rev_wire) {
            case 0:
                AVX512::applyHadamardFloatInternal<0>(arr, num_qubits);
                return ;
            case 1:
                AVX512::applyHadamardFloatInternal<1>(arr, num_qubits);
                return ;
            case 2:
                AVX512::applyHadamardFloatInternal<2>(arr, num_qubits);
                return ;
            default:
                AVX512::applyHadamardFloatExternal(arr, num_qubits, rev_wire);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyHadamard(arr, num_qubits, wires, inverse);
                return ;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch(rev_wire) {
            case 0:
                AVX512::applyHadamardDoubleInternal<0>(arr, num_qubits);
                return ;
            case 1:
                AVX512::applyHadamardDoubleInternal<1>(arr, num_qubits);
                return ;
            default:
                AVX512::applyHadamardDoubleExternal(arr, num_qubits, rev_wire);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            } // else
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 3) {
                applyRZFloatInternal(arr, num_qubits, rev_wire, inverse, angle);
                return;
            }
            applyRZFloatExternal(arr, num_qubits, rev_wire, inverse, angle);
            return;
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            } // else
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 2) {
                applyRZDoubleInternal(arr, num_qubits, rev_wire, inverse,
                                      angle);
                return;
            }
            applyRZDoubleExternal(arr, num_qubits, rev_wire, inverse, angle);
            return;
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        assert(wires.size() == 1);

        const auto rotMat =
            (inverse) ? Gates::getRot<PrecisionT>(-omega, -theta, -phi)
                      : Gates::getRot<PrecisionT>(phi, theta, omega);

        applySingleQubitOp(arr, num_qubits, rotMat.data(), wires[0]);
    }

    /* Two-qubit gates*/
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
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
            if (rev_wire0 < 3 && rev_wire1 < 3) {
                applyIsingZZFloatInternalInternal(arr, num_qubits, rev_wire0,
                                                  rev_wire1, inverse, angle);
                return;
            } else if (std::min(rev_wire0, rev_wire1) < 3) {
                applyIsingZZFloatInternalExternal(arr, num_qubits, rev_wire0,
                                                  rev_wire1, inverse, angle);
                return;
            } else {
                applyIsingZZFloatExternalExternal(arr, num_qubits, rev_wire0,
                                                  rev_wire1, inverse, angle);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyIsingZZ(arr, num_qubits, wires,
                                                    inverse, angle);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                applyIsingZZDoubleInternalInternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1, inverse, angle);
                return;
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                applyIsingZZDoubleInternalExternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1, inverse, angle);
                return;
            } else {
                applyIsingZZDoubleExternalExternal(arr, num_qubits, rev_wire0,
                                                   rev_wire1, inverse, angle);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }
};
} // namespace Pennylane::Gates
