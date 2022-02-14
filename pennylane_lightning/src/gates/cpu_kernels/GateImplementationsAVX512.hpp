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
#include "GateOperation.hpp"
#include "GateImplementationsLM.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"

#include <immintrin.h>

#include <complex>
#include <vector>

namespace Pennylane::Gates {

namespace Internal {


template <class PrecisionT>
struct AVX512Intrinsic;

template <>
struct AVX512Intrinsic<float> {
    using Type = __m512;
};

template <>
struct AVX512Intrinsic<double> {
    using Type = __m512d;
};

template <class PrecisionT>
using AVX512IntrinsicType = typename AVX512Intrinsic<PrecisionT>::Type;


template <class PrecisionT, size_t rev_wire>
auto permuteInternal(AVX512IntrinsicType<PrecisionT> v) -> AVX512IntrinsicType<PrecisionT> {
    // Permute internal data of v after grouping two(complex number)
    if constexpr (std::is_same_v<PrecisionT, float>) {
        if constexpr (rev_wire == 0) {
            return _mm512_permute_ps(v, 0B11100100);
        }
        if constexpr (rev_wire == 1) {
            const static auto shuffle_idx = _mm512_set_epi32(11, 10, 9, 8, 15, 14 , 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
            return _mm512_permutexvar_ps(shuffle_idx, v);
        }
        if constexpr (rev_wire == 2) {
            const static auto shuffle_idx = _mm512_set_epi32(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8);
            return _mm512_permutexvar_ps(shuffle_idx, v);
        }
    }
    if constexpr (std::is_same_v<PrecisionT, double>) {
        if constexpr (rev_wire == 0) {
            const static auto shuffle_idx = _mm512_set_epi64(5,4,7,6,1,0,3,2);
            return _mm512_permutexvar_pd(shuffle_idx, v);
        }
        if constexpr (rev_wire == 1) {
            const static auto shuffle_idx = _mm512_set_epi64(3,2,1,0,7,6,5,4);
            return _mm512_permutexvar_pd(shuffle_idx, v);
        }
    }
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
        GateOperation::PauliX};

    constexpr static std::array<GeneratorOperation, 0> implemented_generators = {};

  private:
    template <typename rev_wire>
    static void applyPauliX_float_internal(std::complex<float> *arr,
                                           const size_t num_qubits) {
        constexpr static auto step = data_alignment_in_bytes / sizeof(float) / 2;
        for (size_t k = 0; k < (1U << num_qubits); k += step) {
            const __m512 v = _mm512_load_ps(arr + k);
            _mm512_store_ps(arr+k, Internal::permuteInternal<float, rev_wire>(v));
        }
    }
    static void applyPauliX_float_external(std::complex<float> *arr,
                                     const size_t num_qubits,
                                     const size_t rev_wire) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        constexpr static auto step = data_alignment_in_bytes / sizeof(float) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k+=step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512 v0 = _mm512_load_ps(arr + i0);
            const __m512 v1 = _mm512_load_ps(arr + i1);
            _mm512_store_ps(arr+i0, v1);
            _mm512_store_ps(arr+i1, v0);
        }
    }

    template <typename rev_wire>
    static void applyPauliX_double_internal(std::complex<float> *arr,
                                            const size_t num_qubits) {
        constexpr static auto step = data_alignment_in_bytes / sizeof(double) / 2;
        for (size_t k = 0; k < (1U << num_qubits); k += step) {
            const __m512d v = _mm512_load_pd(arr + k);
            _mm512_store_pd(arr+k, Internal::permuteInternal<double, rev_wire>(v));
        }
    }
    static void applyPauliX_double_external(std::complex<double> *arr,
                                            const size_t num_qubits,
                                            const size_t rev_wire) {
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        constexpr static auto step = data_alignment_in_bytes / sizeof(double) / 2;
        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k+=step) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const __m512 v0 = _mm512_load_pd(arr + i0);
            const __m512 v1 = _mm512_load_pd(arr + i1);
            _mm512_store_pd(arr+i0, v1);
            _mm512_store_pd(arr+i1, v0);
        }
    }

  public:
    template <class PrecisionT>
    void applyPauliX(std::complex<PrecisionT> *arr,
                     const size_t num_qubits,
                     const std::vector<size_t> &wires,
                     [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires, inverse);
                return ;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch(rev_wire) {
            case 0:
                applyPauliX_float_internal<0>(arr, num_qubits);
                return ;
            case 1:
                applyPauliX_float_internal<1>(arr, num_qubits);
                return ; 
            case 2:
                applyPauliX_float_internal<2>(arr, num_qubits);
                return ;
            default:
                applyPauliX_float_external(arr, num_qubits, rev_wire);
                return ;
            }
        } else if (std::is_same_v<PrecisionT, double>){
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires, inverse);
                return ;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch(rev_wire) {
            case 0:
                applyPauliX_double_internal<0>(arr, num_qubits);
                return ;
            case 1:
                applyPauliX_double_internal<1>(arr, num_qubits);
                return ; 
            default:
                applyPauliX_double_external(arr, num_qubits, rev_wire);
                return ;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> || 
                    std::is_same_v<PrecisionT, double>);
        }
    }

};
} // namespace Pennylane::Gates
