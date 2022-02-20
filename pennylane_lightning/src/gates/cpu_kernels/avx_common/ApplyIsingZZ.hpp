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
 * Defines [] gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX {

template <typename PrecisionT, template <typename> typename AVXConcept>
struct ApplyIsingZZ {
    using PrecisionAVXConcept = AVXConcept<PrecisionT>;
    using RealProd = typename AVXConcept<PrecisionT>::RealProd;
    using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;

    template <class ParamT>
    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                               size_t num_qubits,
                               [[maybe_unused]] size_t rev_wire0,
                               [[maybe_unused]] size_t rev_wire1,
                               bool inverse, ParamT angle) {
        // This function is allowed for AVX512 and AVX2 with float


        const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const auto parity = PrecisionAVXConcept::product(
                PrecisionAVXConcept::internalParity(rev_wire0),
                PrecisionAVXConcept::internalParity(rev_wire1));
        const auto real_cos_factor = RealProd(std::cos(angle / 2));
        auto imag_sin_factor = ImagProd(isin);

        imag_sin_factor *= parity;

        for (size_t n = 0; n < exp2(num_qubits);
             n += PrecisionAVXConcept::step_for_complex_precision) {
            const auto v = PrecisionAVXConcept::load(arr + n);

            const auto prod_cos = real_cos_factor.product(v);
            const auto prod_sin = imag_sin_factor.product(v);

            const auto w = PrecisionAVXConcept::add(prod_cos, prod_sin);
            PrecisionAVXConcept::store(arr + n, w);
        }
    }
    template <class ParamT>
    static void
    applyInternalExternal(std::complex<PrecisionT> *arr, size_t num_qubits,
                          size_t rev_wire0, size_t rev_wire1,
                          bool inverse, ParamT angle) {
        using PrecisionAVXConcept = AVXConcept<PrecisionT>;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t max_rev_wire_shift = (static_cast<size_t>(1U) << rev_wire_max);
        const size_t max_wire_parity = fillTrailingOnes(rev_wire_max);
        const size_t max_wire_parity_inv = fillLeadingOnes(rev_wire_max + 1);

        const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const auto real_cos_factor = RealProd(std::cos(angle / 2));
        const auto imag_sin_factor = ImagProd(isin);

        auto imag_sin_parity0 = imag_sin_factor;
        imag_sin_parity0 *= PrecisionAVXConcept::internalParity(rev_wire_min);
        auto imag_sin_parity1 = imag_sin_parity0;
        imag_sin_parity1 *= -1.0;

        for (size_t k = 0; k < exp2(num_qubits - 1);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto prod_cos0 = real_cos_factor.product(v0);
            const auto prod_sin0 = imag_sin_parity0.product(v0);

            const auto prod_cos1 = real_cos_factor.product(v1);
            const auto prod_sin1 = imag_sin_parity1.product(v1);

            PrecisionAVXConcept::store(arr + i0, PrecisionAVXConcept::add(prod_cos0, prod_sin0));
            PrecisionAVXConcept::store(arr + i1, PrecisionAVXConcept::add(prod_cos1, prod_sin1));
        }
    }

    template <class ParamT>
    static void applyExternalExternal(std::complex<PrecisionT> *arr,
                                                   const size_t num_qubits,
                                                   const size_t rev_wire0,
                                                   const size_t rev_wire1,
                                                   bool inverse,
                                                   ParamT angle) {
        using PrecisionAVXConcept = AVXConcept<PrecisionT>;

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const double isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        const auto real_cos_factor = RealProd(std::cos(angle / 2));
        const auto p_isin_prod = ImagProd(isin);
        const auto m_isin_prod = ImagProd(-isin);

        for (size_t k = 0; k < exp2(num_qubits - 2);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const auto v00 = PrecisionAVXConcept::load(arr + i00); // 00
            const auto v01 = PrecisionAVXConcept::load(arr + i01); // 01
            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            const auto v11 = PrecisionAVXConcept::load(arr + i11); // 11

            const auto prod_cos00 = real_cos_factor.product(v00);
            const auto prod_isin00 = p_isin_prod.product(v00);

            const auto prod_cos01 = real_cos_factor.product(v01);
            const auto prod_isin01 = m_isin_prod.product(v01);

            const auto prod_cos10 = real_cos_factor.product(v10);
            const auto prod_isin10 = m_isin_prod.product(v10);

            const auto prod_cos11 = real_cos_factor.product(v11);
            const auto prod_isin11 = p_isin_prod.product(v11);

            PrecisionAVXConcept::store(arr + i00, PrecisionAVXConcept::add(prod_cos00, prod_isin00));
            PrecisionAVXConcept::store(arr + i01, PrecisionAVXConcept::add(prod_cos01, prod_isin01));
            PrecisionAVXConcept::store(arr + i10, PrecisionAVXConcept::add(prod_cos10, prod_isin10));
            PrecisionAVXConcept::store(arr + i11, PrecisionAVXConcept::add(prod_cos11, prod_isin11));
        }
    }
};
} // namespace Pennylane::Gates::AVX
