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
 * Defines applySingleQubitOp for AVX
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX {

template<typename PrecisionT, template <typename> class AVXConcept>
struct SingleQubitOpProd {
    using PrecisionAVXConcept = AVXConcept<PrecisionT>;
    using RealProd = typename AVXConcept<PrecisionT>::RealProd;
    using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;

    RealProd diag_real;
    ImagProd diag_imag;
    RealProd offdiag_real;
    ImagProd offdiag_imag;
};

template<typename PrecisionT, template <typename> class AVXConcept, size_t rev_wire>
struct SingleQubitOpProdCreate {
    static_assert(sizeof(PrecisionT) == -1, "Given rev_wire is not supported.");
};

template<typename PrecisionT, template <typename> class AVXConcept>
struct SingleQubitOpProdCreate<PrecisionT, AVXConcept, 0> {
    static SingleQubitOpProd<PrecisionT, AVXConcept> create(bool inverse, const std::complex<PrecisionT>* matrix) {
        using RealProd = typename AVXConcept<PrecisionT>::RealProd;
        using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;

        SingleQubitOpProd<PrecisionT, AVXConcept> res;
        // rev_wire == 0
        if (inverse) {
            res.diag_real = RealProd::repeat2(real(matrix[0]), real(matrix[3]));
            res.diag_imag = ImagProd::repeat2(-imag(matrix[0]), -imag(matrix[3]));
            res.offdiag_real = RealProd::repeat2(real(matrix[2]), real(matrix[1]));
            res.offdiag_imag = ImagProd::repeat2(-imag(matrix[2]), -imag(matrix[1]));
        } else {
            res.diag_real = RealProd::repeat2(real(matrix[0]), real(matrix[3]));
            res.diag_imag = ImagProd::repeat2(imag(matrix[0]), imag(matrix[3]));
            res.offdiag_real = RealProd::repeat2(real(matrix[1]), real(matrix[2]));
            res.offdiag_imag = ImagProd::repeat2(imag(matrix[1]), imag(matrix[2]));
        }
        return res;
    }
};
template<typename PrecisionT, template <typename> class AVXConcept>
struct SingleQubitOpProdCreate<PrecisionT, AVXConcept, 1> {
    static SingleQubitOpProd<PrecisionT, AVXConcept> create(bool inverse, const std::complex<PrecisionT>* matrix) {
        using RealProd = typename AVXConcept<PrecisionT>::RealProd;
        using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;

        SingleQubitOpProd<PrecisionT, AVXConcept> res;
        // rev_wire == 1
        if (inverse) {
            res.diag_real = RealProd::repeat4(real(matrix[0]), real(matrix[3]));
            res.diag_imag = ImagProd::repeat4(-imag(matrix[0]), -imag(matrix[3]));
            res.offdiag_real = RealProd::repeat4(real(matrix[2]), real(matrix[1]));
            res.offdiag_imag = ImagProd::repeat4(-imag(matrix[2]), -imag(matrix[1]));
        } else {
            res.diag_real = RealProd::repeat4(real(matrix[0]), real(matrix[3]));
            res.diag_imag = ImagProd::repeat4(imag(matrix[0]), imag(matrix[3]));
            res.offdiag_real = RealProd::repeat4(real(matrix[1]), real(matrix[2]));
            res.offdiag_imag = ImagProd::repeat4(imag(matrix[1]), imag(matrix[2]));
        }
        return res;
    }
};

template<typename PrecisionT, template <typename> class AVXConcept>
struct SingleQubitOpProdCreate<PrecisionT, AVXConcept, 2> {
    static SingleQubitOpProd<PrecisionT, AVXConcept> create(bool inverse, const std::complex<PrecisionT>* matrix) {
        using RealProd = typename AVXConcept<PrecisionT>::RealProd;
        using ImagProd = typename AVXConcept<PrecisionT>::ImagProd;

        SingleQubitOpProd<PrecisionT, AVXConcept> res;
        // rev_wire == 2
        if (inverse) {
            res.diag_real = RealProd::repeat8(real(matrix[0]), real(matrix[3]));
            res.diag_imag = ImagProd::repeat8(-imag(matrix[0]), -imag(matrix[3]));
            res.offdiag_real = RealProd::repeat8(real(matrix[2]), real(matrix[1]));
            res.offdiag_imag = ImagProd::repeat8(-imag(matrix[2]), -imag(matrix[1]));
        } else {
            res.diag_real = RealProd::repeat8(real(matrix[0]), real(matrix[3]));
            res.diag_imag = ImagProd::repeat8(imag(matrix[0]), imag(matrix[3]));
            res.offdiag_real = RealProd::repeat8(real(matrix[1]), real(matrix[2]));
            res.offdiag_imag = ImagProd::repeat8(imag(matrix[1]), imag(matrix[2]));
        }
        return res;
    }
};

template<typename PrecisionT, template <typename> class AVXConcept, size_t rev_wire>
auto createSingleQubitOpProd(bool inverse, const std::complex<PrecisionT>* matrix) {
    return SingleQubitOpProdCreate<PrecisionT, AVXConcept, rev_wire>::create(inverse, matrix);
}


template<typename PrecisionT, template <typename> class AVXConcept>
struct ApplySingleQubitOp {
    template <size_t rev_wire>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::complex<PrecisionT> *matrix,
                              bool inverse = false) {
        using PrecisionAVXConcept = AVXConcept<PrecisionT>;
        const auto [diag_real, diag_imag, offdiag_real, offdiag_imag] 
            = createSingleQubitOpProd<PrecisionT, AVXConcept, rev_wire>(inverse, matrix);

        for (size_t k = 0; k < exp2(num_qubits);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            const auto w_diag = PrecisionAVXConcept::add(diag_real.product(v),
                                                   diag_imag.product(v));

            const auto v_off = PrecisionAVXConcept::template internalSwap<rev_wire>(v);

            const auto w_offdiag = PrecisionAVXConcept::add(
                offdiag_real.product(v_off),
                offdiag_imag.product(v_off));

            PrecisionAVXConcept::store(arr + k, AVXConcept<PrecisionT>::add(w_diag, w_offdiag));
        }
    }

    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const size_t rev_wire,
                              const std::complex<PrecisionT> *matrix,
                              bool inverse = false) {
        using PrecisionAVXConcept = AVXConcept<PrecisionT>;
        using RealProd = typename PrecisionAVXConcept::RealProd;
        using ImagProd = typename PrecisionAVXConcept::ImagProd;

        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        std::complex<PrecisionT> u00;
        std::complex<PrecisionT> u01;
        std::complex<PrecisionT> u10;
        std::complex<PrecisionT> u11;

        if (inverse) {
            u00 = std::conj(matrix[0]);
            u01 = std::conj(matrix[2]);
            u10 = std::conj(matrix[1]);
            u11 = std::conj(matrix[3]);
        } else {
            u00 = matrix[0];
            u01 = matrix[1];
            u10 = matrix[2];
            u11 = matrix[3];
        }

        const auto u00_real_prod = RealProd(real(u00));
        const auto u00_imag_prod = ImagProd(imag(u00));

        const auto u01_real_prod = RealProd(real(u01));
        const auto u01_imag_prod = ImagProd(imag(u01));

        const auto u10_real_prod = RealProd(real(u10));
        const auto u10_imag_prod = ImagProd(imag(u10));
        
        const auto u11_real_prod = RealProd(real(u11));
        const auto u11_imag_prod = ImagProd(imag(u11));

        for (size_t k = 0; k < exp2(num_qubits - 1);
             k += PrecisionAVXConcept::step_for_complex_precision) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            // w0 = u00 * v0 + u01 * v1
            const auto w0_real = PrecisionAVXConcept::add(
                    u00_real_prod.product(v0),
                    u01_real_prod.product(v1));
            const auto w0_imag = PrecisionAVXConcept::add(
                    u00_imag_prod.product(v0),
                    u01_imag_prod.product(v1));

            // w1 = u11 * v1 + u10 * v0
            const auto w1_real = PrecisionAVXConcept::add(
                    u11_real_prod.product(v1),
                    u10_real_prod.product(v0));
            const auto w1_imag = PrecisionAVXConcept::add(
                    u11_imag_prod.product(v1),
                    u10_imag_prod.product(v0));

            PrecisionAVXConcept::store(arr + i0, PrecisionAVXConcept::add(w0_real, w0_imag));
            PrecisionAVXConcept::store(arr + i1, PrecisionAVXConcept::add(w1_real, w1_imag));
        }
    }
};
/// @endcond
} // namespace Pennylane::Gates::AVX
