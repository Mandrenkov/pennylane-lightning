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
 * Defines SWAP gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Blender.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX {

template <size_t num_packed, size_t xor_wire> struct PermuteSWAP;

template <size_t num_packed> struct PermuteSWAP<num_packed, 1> {
    using Type = typename Swap<typename Permute<num_packed>::Type, 0, 1>::Type;
};
template <size_t num_packed> struct PermuteSWAP<num_packed, 2> {
    using Type = typename Swap<typename Permute<num_packed>::Type, 0, 2>::Type;
};
template <size_t num_packed> struct PermuteSWAP<num_packed, 3> {
    using Type = typename Swap<typename Permute<num_packed>::Type, 1, 2>::Type;
};

template <typename PrecisionT, template <typename> typename AVXConcept>
struct ApplySWAP {
    using PrecisionAVXConcept = AVXConcept<PrecisionT>;
    using RealProd = typename AVXConcept<PrecisionT>::RealProd;
    using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;

    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t rev_wire0,
                                      size_t rev_wire1) {
        const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
        const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

        switch (min_rev_wire ^ max_rev_wire) {
        case 1: // (0, 1)
            break;
        case 2: // (0, 2)
            break;
        case 3: // (1, 2)
            break;
        }

        for (size_t n = 0; n < exp2(num_qubits);
             n += PrecisionAVXConcept::step_for_complex_precision) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            PrecisionAVXConcept::store(arr + n, permuteIntrinsic<SWAP>(v));
        }
    }

    static void applyInternalExternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t rev_wire0,
                                      size_t rev_wire1) {
        const size_t min_rev_wire = std::min(rev_wire0, rev_wire1);
        const size_t max_rev_wire = std::max(rev_wire0, rev_wire1);

        const size_t max_rev_wire_shift =
            (static_cast<size_t>(1U) << max_rev_wire);
        const size_t max_wire_parity = fillTrailingOnes(max_rev_wire);
        const size_t max_wire_parity_inv = fillLeadingOnes(max_rev_wire + 1);

        Mask<2 * PrecisionAVXConcept::step_for_complex_precision> mask;
        for (size_t i = 0; i < mask.size(); i++) {
            if ((i & (1U << min_rev_wire)) != 0) {
                mask[i] = 1U;
            } else {
                mask[i] = 0U;
            }
        }
        const auto blender =
            Blender<PrecisionT,
                    2 * PrecisionAVXConcept::step_for_complex_precision>(mask);

        for (size_t k = 0; k < exp2(num_qubits - 1);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            PrecisionAVXConcept::store(arr + i0, blender.blend(v1, v0));
            PrecisionAVXConcept::store(arr + i1, blender.blend(v0, v1));
        }
    }

    static void applyExternalExternal(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire0,
                                      const size_t rev_wire1) {
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const auto minus_one = RealProd(-1.0);

        for (size_t k = 0; k < exp2(num_qubits - 2);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;

            const auto v01 = PrecisionAVXConcept::load(arr + i01); // 01
            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            PrecisionAVXConcept::store(arr + i10, v01);
            PrecisionAVXConcept::store(arr + i01, v10);
        }
    }
};
} // namespace Pennylane::Gates::AVX
