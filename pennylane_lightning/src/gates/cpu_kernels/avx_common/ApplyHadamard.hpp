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
 * Defines Hadamard gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX {
/// @cond DEV

template <typename PrecisionT, template <typename> typename AVXConcept,
         size_t rev_wire>
struct HadamardDiag {
    static_assert(sizeof(PrecisionT) == -1, "Given re_wire is not supported");
};
template <typename PrecisionT, template <typename> typename AVXConcept>
struct HadamardDiag<PrecisionT, AVXConcept, 0> {
    typename AVXConcept<PrecisionT>::RealProd
    static create() {
        using RealProd = typename AVXConcept<PrecisionT>::RealProd;
        constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<PrecisionT>();
        return RealProd::repeat2(isqrt2, -isqrt2);
    }
};
template <typename PrecisionT, template <typename> typename AVXConcept>
struct HadamardDiag<PrecisionT, AVXConcept, 1> {
    typename AVXConcept<PrecisionT>::RealProd
    static create() {
        using RealProd = typename AVXConcept<PrecisionT>::RealProd;
        constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<PrecisionT>();
        return RealProd::repeat4(isqrt2, -isqrt2);
    }
};
template <typename PrecisionT, template <typename> typename AVXConcept>
struct HadamardDiag<PrecisionT, AVXConcept, 2> {
    typename AVXConcept<PrecisionT>::RealProd
    static create() {
        using RealProd = typename AVXConcept<PrecisionT>::RealProd;
        constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<PrecisionT>();
        return RealProd::repeat8(isqrt2, -isqrt2);
    }
};

template <typename PrecisionT, template <typename> typename AVXConcept, int rev_wire>
auto createHadamardDiag() {
    return HadamardDiag<PrecisionT, AVXConcept, rev_wire>::create();
}

template <typename PrecisionT, template <typename> typename AVXConcept>
struct ApplyHadamard {
    using PrecisionAVXConcept = AVXConcept<PrecisionT>;
    using RealProd = typename AVXConcept<PrecisionT>::RealProd;
    using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;

    template <size_t rev_wire>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits) {
        constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<PrecisionT>();

        const auto mat_diag = createHadamardDiag<PrecisionT, AVXConcept, rev_wire>();
        const auto mat_offdiag = RealProd(isqrt2);

        for (size_t k = 0; k < exp2(num_qubits);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const auto v = PrecisionAVXConcept::load(arr + k);

            const auto w_diag = mat_diag.product(v);
            const auto v_offdiag = PrecisionAVXConcept::template internalSwap<rev_wire>(v);
            const auto w_offdiag = mat_offdiag.product(v_offdiag);
            PrecisionAVXConcept::store(arr + k,
                    PrecisionAVXConcept::add(w_diag, w_offdiag));
        }
    }
    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const size_t rev_wire) {
        constexpr auto isqrt2 = Pennylane::Util::INVSQRT2<PrecisionT>();

        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const auto p_isqrt2_prod = RealProd(isqrt2);
        const auto m_isqrt2_prod = RealProd(-isqrt2);

        for (size_t k = 0; k < exp2(num_qubits - 1);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto w0 = PrecisionAVXConcept::add(
                    p_isqrt2_prod.product(v0),
                    p_isqrt2_prod.product(v1));

            const auto w1 = PrecisionAVXConcept::add(
                    p_isqrt2_prod.product(v0),
                    m_isqrt2_prod.product(v1));

            PrecisionAVXConcept::store(arr + i0, w0);
            PrecisionAVXConcept::store(arr + i1, w1);
        }
    }
};
} // namespace Pennylane::Gates::AVX
