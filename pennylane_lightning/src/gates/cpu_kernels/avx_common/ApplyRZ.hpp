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
 * Defines RZ gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX {
template<typename PrecisionT, template<typename> class AVXConcept>
struct ApplyRZ {
    using PrecisionAVXConcept = AVXConcept<PrecisionT>;
    using RealProd = typename AVXConcept<PrecisionT>::RealProd;
    using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;
     
    template <size_t rev_wire, class ParamT>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              [[maybe_unused]] bool inverse,
                              ParamT angle) {
        const PrecisionT isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        const auto real_cos_factor = RealProd(std::cos(angle / 2));
        auto imag_sin_factor = ImagProd(isin);
        imag_sin_factor *= PrecisionAVXConcept::internalParity(rev_wire);
    
        for (size_t n = 0; n < (1U << num_qubits);
            n += PrecisionAVXConcept::step_for_complex_precision) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            PrecisionAVXConcept::store(arr + n, 
                    PrecisionAVXConcept::add(real_cos_factor.product(v),
                                             imag_sin_factor.product(v)));
        }
    }

    template <class ParamT>
    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const size_t rev_wire,
                              [[maybe_unused]] bool inverse,
                              ParamT angle) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const auto real_cos_factor = RealProd(std::cos(angle / 2));
        const PrecisionT isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        const auto plus_isin_prod = ImagProd(isin);
        const auto minus_isin_prod = ImagProd(-isin);

        for (size_t k = 0; k < exp2(num_qubits - 1);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto v0_cos = real_cos_factor.product(v0);
            const auto v0_isin = plus_isin_prod.product(v0);

            const auto v1_cos = real_cos_factor.product(v1);
            const auto v1_isin = minus_isin_prod.product(v1);

            PrecisionAVXConcept::store(arr + i0, PrecisionAVXConcept::add(v0_cos, v0_isin));
            PrecisionAVXConcept::store(arr + i1, PrecisionAVXConcept::add(v1_cos, v1_isin));
        }
    }
};
} // namespace Pennylane::Gates::AVX
