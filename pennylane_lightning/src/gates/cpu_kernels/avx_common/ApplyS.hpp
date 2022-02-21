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
 * Defines S gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"
#include "Permutation.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX {
/// @cond DEV
namespace Internal {
template <int perm_size, int rev_wire>
struct PauliSInternalPerm;

template <>
struct PauliSInternalPerm<4, 0> {
    // rev_wire == 0
    using Type = Permute4<0,1,3,2>;
};
template <>
struct PauliSInternalPerm<8, 0> {
    // rev_wire == 0
    using Type = Permute8<0, 1, 3, 2, 4, 5, 7, 6>;
};
template <>
struct PauliSInternalPerm<8, 1> {
    // rev_wire == 0
    using Type = Permute8<0, 1, 2, 3, 5, 4, 7, 6>;
};
template <>
struct PauliSInternalPerm<16, 0> {
    // rev_wire == 0
    using Type = Permute16<0, 1,  3,  2,  4,  5,  7,  6,
                           8, 9, 11, 10, 12, 13, 15, 14>;
};
template <>
struct PauliSInternalPerm<16, 1> {
    // rev_wire == 0
    using Type = Permute16<0, 1, 2, 3, 5, 4, 7, 6,
                           8, 9, 10, 11, 13, 12, 15, 14>;
};
template <>
struct PauliSInternalPerm<16, 2> {
    // rev_wire == 0
    using Type = Permute16<0, 1, 2, 3, 4, 5, 6, 7,
                          9, 8, 11, 10, 13, 12, 15, 14>;
};


template <typename RealProd, int rev_wire>
struct ApplySInternalFactor;

template <typename RealProd>
struct ApplySInternalFactor<RealProd, 0> {
    static auto create() {
        return RealProd::setr4(1.0F, 1.0F, -1.0F, 1.0F);
    }
};
template <typename RealProd>
struct ApplySInternalFactor<RealProd, 1> {
    static auto create() { 
        return RealProd::setr8(1.0F, 1.0F, 1.0F, 1.0F,
                           -1.0F, 1.0F, -1.0F, 1.0F);
    }
};
template <typename RealProd>
struct ApplySInternalFactor<RealProd, 2> {
    static auto create() {
        return RealProd::setr16(1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F,
                                -1.0F, 1.0F, -1.0F, 1.0F,
                                -1.0F, 1.0F, -1.0F, 1.0F);
    }
};

template <typename RealProd, int rev_wire>
auto createInternalFactor() {
    return ApplySInternalFactor<RealProd, rev_wire>::create();
}
} // namespace Internal
/// @endcond
template<typename PrecisionT, template<typename> class AVXConcept>
struct ApplyS {
    using PrecisionAVXConcept = AVXConcept<PrecisionT>;
    using RealProd = typename AVXConcept<PrecisionT>::RealProd;
    using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;


    template <int rev_wire>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              bool inverse) {

        auto factor = Internal::createInternalFactor<RealProd, rev_wire>();
        using Perm = typename Internal::PauliSInternalPerm<
            sizeof(typename PrecisionAVXConcept::IntrinsicType) / sizeof(PrecisionT),
            rev_wire>::Type;

        if(inverse) {
            factor *= -1.0;
        }

        for (size_t k = 0; k < (1U << num_qubits);
            k += PrecisionAVXConcept::step_for_complex_precision) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            const auto w = permuteIntrinsic<Perm>(v);
            PrecisionAVXConcept::store(arr + k, factor.product(w));
        }
    }

    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const size_t rev_wire,
                              bool inverse) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    
        const auto imag_factor = ImagProd(inverse?-1.0:1.0);
        for (size_t k = 0; k < exp2(num_qubits - 1);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v1 = PrecisionAVXConcept::load(arr + i1);
            PrecisionAVXConcept::store(arr + i1, imag_factor.product(v1));
        }
    }
};
} // namespace Pennylane::Gates::AVX
