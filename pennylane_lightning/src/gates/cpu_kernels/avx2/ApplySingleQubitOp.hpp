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
 * Defines applySingleQubitOp
 */
#pragma once
#include "AVX2Util.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <complex>

namespace Pennylane::Gates::AVX2 {
/// @cond DEV
template <size_t rev_wire>
void applySingleQubitOpInternal(std::complex<float> *arr,
                                const size_t num_qubits,
                                const std::complex<float> *matrix,
                                bool inverse = false) {
    __m256 diag_real;
    __m256 diag_imag;
    __m256 offdiag_real;
    __m256 offdiag_imag;

    if constexpr (rev_wire == 0) {
        if (inverse) {
            diag_real = _mm256_setr4_ps(real(matrix[0]), real(matrix[0]),
                                        real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr4_ps(-imag(matrix[0]), -imag(matrix[0]),
                                        -imag(matrix[3]), -imag(matrix[3]));

            offdiag_real = _mm256_setr4_ps(real(matrix[2]), real(matrix[2]),
                                           real(matrix[1]), real(matrix[1]));

            offdiag_imag = _mm256_setr4_ps(-imag(matrix[2]), -imag(matrix[2]),
                                           -imag(matrix[1]), -imag(matrix[1]));

        } else {
            diag_real = _mm256_setr4_ps(real(matrix[0]), real(matrix[0]),
                                        real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr4_ps(imag(matrix[0]), imag(matrix[0]),
                                        imag(matrix[3]), imag(matrix[3]));

            offdiag_real = _mm256_setr4_ps(real(matrix[1]), real(matrix[1]),
                                           real(matrix[2]), real(matrix[2]));

            offdiag_imag = _mm256_setr4_ps(imag(matrix[1]), imag(matrix[1]),
                                           imag(matrix[2]), imag(matrix[2]));
        }
    } else if (rev_wire == 1) {
        // clang-format off
        if (inverse) {
            diag_real = _mm256_setr_ps(
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr_ps(
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[3]), -imag(matrix[3]),
                -imag(matrix[3]), -imag(matrix[3]),
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[3]), -imag(matrix[3]),
                -imag(matrix[3]), -imag(matrix[3]));

            offdiag_real = _mm256_setr_ps(
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]));

            offdiag_imag = _mm256_setr_ps(
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[1]), -imag(matrix[1]),
                -imag(matrix[1]), -imag(matrix[1]),
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[1]), -imag(matrix[1]),
                -imag(matrix[1]), -imag(matrix[1]));
        } else {
            diag_real = _mm256_setr_ps(
                real(matrix[0]), real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]), real(matrix[0]),
                real(matrix[3]), real(matrix[3]), real(matrix[3]),
                real(matrix[3]));

            diag_imag = _mm256_setr_ps(
                imag(matrix[0]), imag(matrix[0]), imag(matrix[0]),
                imag(matrix[0]), imag(matrix[3]), imag(matrix[3]),
                imag(matrix[3]), imag(matrix[3]), imag(matrix[0]),
                imag(matrix[0]), imag(matrix[0]), imag(matrix[0]),
                imag(matrix[3]), imag(matrix[3]), imag(matrix[3]),
                imag(matrix[3]));

            offdiag_real = _mm256_setr_ps(
                real(matrix[1]), real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]), real(matrix[1]),
                real(matrix[2]), real(matrix[2]), real(matrix[2]),
                real(matrix[2]));

            offdiag_imag = _mm256_setr_ps(
                imag(matrix[1]), imag(matrix[1]), imag(matrix[1]),
                imag(matrix[1]), imag(matrix[2]), imag(matrix[2]),
                imag(matrix[2]), imag(matrix[2]), imag(matrix[1]),
                imag(matrix[1]), imag(matrix[1]), imag(matrix[1]),
                imag(matrix[2]), imag(matrix[2]), imag(matrix[2]),
                imag(matrix[2]));
        }
        // clang-format on
    } else { // rev_wire == 2
        // clang-format off
        if (inverse) {
            diag_real = _mm256_setr_ps(
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr_ps(
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[3]), -imag(matrix[3]),
                -imag(matrix[3]), -imag(matrix[3]),
                -imag(matrix[3]), -imag(matrix[3]),
                -imag(matrix[3]), -imag(matrix[3]));

            offdiag_real = _mm256_setr_ps(
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]));

            offdiag_imag = _mm256_setr_ps(
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[1]), -imag(matrix[1]),
                -imag(matrix[1]), -imag(matrix[1]),
                -imag(matrix[1]), -imag(matrix[1]),
                -imag(matrix[1]), -imag(matrix[1]));
        } else {
            diag_real = _mm256_setr_ps(
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr_ps(
                imag(matrix[0]), imag(matrix[0]),
                imag(matrix[0]), imag(matrix[0]),
                imag(matrix[0]), imag(matrix[0]),
                imag(matrix[0]), imag(matrix[0]),
                imag(matrix[3]), imag(matrix[3]),
                imag(matrix[3]), imag(matrix[3]),
                imag(matrix[3]), imag(matrix[3]),
                imag(matrix[3]), imag(matrix[3]));

            offdiag_real = _mm256_setr_ps(
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]));

            offdiag_imag = _mm256_setr_ps(
                imag(matrix[1]), imag(matrix[1]),
                imag(matrix[1]), imag(matrix[1]),
                imag(matrix[1]), imag(matrix[1]),
                imag(matrix[1]), imag(matrix[1]),
                imag(matrix[2]), imag(matrix[2]),
                imag(matrix[2]), imag(matrix[2]),
                imag(matrix[2]), imag(matrix[2]),
                imag(matrix[2]), imag(matrix[2]));
        }
        // clang-format on
    }

    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<float>) {
        __m256 v = _mm256_load_ps(arr + k);
        if constexpr (rev_wire == 0) {
            __m256 w_diag = _mm256_add_ps(
                _mm256_mul_ps(v, diag_real),
                Util::productImagS(v, diag_imag)); // can optimize more?

            __m256 v_off = _mm256_permute_ps(v, 0B01'00'11'10);

            __m256 w_offdiag = _mm256_add_ps(
                _mm256_mul_ps(v_off, offdiag_real),
                Util::productImagS(v_off,
                                   offdiag_imag)); // can optimize more?

            v = _mm256_add_ps(w_diag, w_offdiag);
        } else if (rev_wire == 1) {
            __m256 w_diag = _mm256_add_ps(
                _mm256_mul_ps(v, diag_real),
                Util::productImagS(v, diag_imag)); // can optimize more?

            __m256 v_off = _mm256_permutexvar_ps(
                _mm256_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6,
                                 5, 4),
                v);

            __m256 w_offdiag = _mm256_add_ps(
                _mm256_mul_ps(v_off, offdiag_real),
                Util::productImagS(v_off,
                                   offdiag_imag)); // can optimize more?

            v = _mm256_add_ps(w_diag, w_offdiag);
        } else { // rev_wire == 2
            __m256 w_diag = _mm256_add_ps(
                _mm256_mul_ps(v, diag_real),
                Util::productImagS(v, diag_imag)); // can optimize more?

            __m256 v_off = _mm256_permutexvar_ps(
                _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10,
                                 9, 8),
                v);

            __m256 w_offdiag = _mm256_add_ps(
                _mm256_mul_ps(v_off, offdiag_real),
                Util::productImagS(v_off,
                                   offdiag_imag)); // can optimize more?

            v = _mm256_add_ps(w_diag, w_offdiag);
        }
        _mm256_store_ps(arr + k, v);
    }
}

inline void applySingleQubitOpExternal(std::complex<float> *arr,
                                       const size_t num_qubits,
                                       const size_t rev_wire,
                                       const std::complex<float> *matrix,
                                       bool inverse = false) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    std::complex<float> u00;
    std::complex<float> u01;
    std::complex<float> u10;
    std::complex<float> u11;

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

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<float>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256 v0 = _mm256_load_ps(arr + i0);
        const __m256 v1 = _mm256_load_ps(arr + i1);

        // w0 = u00 * v0
        __m256 w0_real = _mm256_mul_ps(v0, _mm256_set1_ps(real(u00)));
        __m256 w0_imag = Util::productImagS(v0, _mm256_set1_ps(imag(u00)));

        // w0 +=  u01 * v1
        w0_real = _mm256_add_ps(w0_real,
                                _mm256_mul_ps(v1, _mm256_set1_ps(real(u01))));
        w0_imag = _mm256_add_ps(
            w0_imag, Util::productImagS(v1, _mm256_set1_ps(imag(u01))));

        // w1 = u11 * v1
        __m256 w1_real = _mm256_mul_ps(v1, _mm256_set1_ps(real(u11)));
        __m256 w1_imag = Util::productImagS(v1, _mm256_set1_ps(imag(u11)));

        // w1 +=  u10 * v0
        w1_real = _mm256_add_ps(w1_real,
                                _mm256_mul_ps(v0, _mm256_set1_ps(real(u10))));
        w1_imag = _mm256_add_ps(
            w1_imag, Util::productImagS(v0, _mm256_set1_ps(imag(u10))));

        _mm256_store_ps(arr + i0, _mm256_add_ps(w0_real, w0_imag));
        _mm256_store_ps(arr + i1, _mm256_add_ps(w1_real, w1_imag));
    }
}

template <size_t rev_wire>
static void applySingleQubitOpInternal(std::complex<double> *arr,
                                       const size_t num_qubits,
                                       const std::complex<double> *matrix,
                                       bool inverse = false) {
    __m256d diag_real;
    __m256d diag_imag;
    __m256d offdiag_real;
    __m256d offdiag_imag;

    if constexpr (rev_wire == 0) {
        if (inverse) {
            diag_real = _mm256_setr4_pd(real(matrix[0]), real(matrix[0]),
                                        real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr4_pd(-imag(matrix[0]), -imag(matrix[0]),
                                        -imag(matrix[3]), -imag(matrix[3]));

            offdiag_real = _mm256_setr4_pd(real(matrix[2]), real(matrix[2]),
                                           real(matrix[1]), real(matrix[1]));

            offdiag_imag = _mm256_setr4_pd(-imag(matrix[2]), -imag(matrix[2]),
                                           -imag(matrix[1]), -imag(matrix[1]));

        } else {
            diag_real = _mm256_setr4_pd(real(matrix[0]), real(matrix[0]),
                                        real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr4_pd(imag(matrix[0]), imag(matrix[0]),
                                        imag(matrix[3]), imag(matrix[3]));

            offdiag_real = _mm256_setr4_pd(real(matrix[1]), real(matrix[1]),
                                           real(matrix[2]), real(matrix[2]));

            offdiag_imag = _mm256_setr4_pd(imag(matrix[1]), imag(matrix[1]),
                                           imag(matrix[2]), imag(matrix[2]));
        }
    } else { // rev_wire == 1
        // clang-format off
        if (inverse) {
            diag_real = _mm256_setr_pd(
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr_pd(
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[0]), -imag(matrix[0]),
                -imag(matrix[3]), -imag(matrix[3]),
                -imag(matrix[3]), -imag(matrix[3]));

            offdiag_real = _mm256_setr_pd(
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]));

            offdiag_imag = _mm256_setr_pd(
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[2]), -imag(matrix[2]),
                -imag(matrix[1]), -imag(matrix[1]),
                -imag(matrix[1]), -imag(matrix[1]));
        } else {
            diag_real = _mm256_setr_pd(
                real(matrix[0]), real(matrix[0]),
                real(matrix[0]), real(matrix[0]),
                real(matrix[3]), real(matrix[3]),
                real(matrix[3]), real(matrix[3]));

            diag_imag = _mm256_setr_pd(
                imag(matrix[0]), imag(matrix[0]),
                imag(matrix[0]), imag(matrix[0]),
                imag(matrix[3]), imag(matrix[3]),
                imag(matrix[3]), imag(matrix[3]));

            offdiag_real = _mm256_setr_pd(
                real(matrix[1]), real(matrix[1]),
                real(matrix[1]), real(matrix[1]),
                real(matrix[2]), real(matrix[2]),
                real(matrix[2]), real(matrix[2]));

            offdiag_imag = _mm256_setr_pd(
                imag(matrix[1]), imag(matrix[1]),
                imag(matrix[1]), imag(matrix[1]),
                imag(matrix[2]), imag(matrix[2]),
                imag(matrix[2]), imag(matrix[2]));
        }
        // clang-format on
    }

    for (size_t k = 0; k < exp2(num_qubits);
         k += step_for_complex_precision<double>) {
        __m256d v = _mm256_load_pd(arr + k);
        if constexpr (rev_wire == 0) {
            __m256d w_diag = _mm256_add_pd(
                _mm256_mul_pd(v, diag_real),
                Util::productImagD(v, diag_imag)); // can optimize more?

            __m256d v_off = _mm256_permutex_pd(v, 0B01'00'11'10);

            __m256d w_offdiag = _mm256_add_pd(
                _mm256_mul_pd(v_off, offdiag_real),
                Util::productImagD(v_off,
                                   offdiag_imag)); // can optimize more?

            v = _mm256_add_pd(w_diag, w_offdiag);
        } else { // rev_wire == 1
            __m256d w_diag = _mm256_add_pd(
                _mm256_mul_pd(v, diag_real),
                Util::productImagD(v, diag_imag)); // can optimize more?

            __m256d v_off = _mm256_permutexvar_pd(
                _mm256_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), v);

            __m256d w_offdiag = _mm256_add_pd(
                _mm256_mul_pd(v_off, offdiag_real),
                Util::productImagD(v_off,
                                   offdiag_imag)); // can optimize more?

            v = _mm256_add_pd(w_diag, w_offdiag);
        }
        _mm256_store_pd(arr + k, v);
    }
}

inline void applySingleQubitOpExternal(std::complex<double> *arr,
                                       const size_t num_qubits,
                                       const size_t rev_wire,
                                       const std::complex<double> *matrix,
                                       bool inverse = false) {
    const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
    const size_t wire_parity = fillTrailingOnes(rev_wire);
    const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

    std::complex<double> u00;
    std::complex<double> u01;
    std::complex<double> u10;
    std::complex<double> u11;

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

    for (size_t k = 0; k < exp2(num_qubits - 1);
         k += step_for_complex_precision<double>) {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;

        const __m256d v0 = _mm256_load_pd(arr + i0);
        const __m256d v1 = _mm256_load_pd(arr + i1);

        // w0 = u00 * v0
        __m256d w0_real = _mm256_mul_pd(v0, _mm256_set1_pd(real(u00)));
        __m256d w0_imag = Util::productImagD(v0, _mm256_set1_pd(imag(u00)));

        // w0 +=  u01 * v1
        w0_real = _mm256_add_pd(w0_real,
                                _mm256_mul_pd(v1, _mm256_set1_pd(real(u01))));
        w0_imag = _mm256_add_pd(
            w0_imag, Util::productImagD(v1, _mm256_set1_pd(imag(u01))));

        // w1 = u11 * v1
        __m256d w1_real = _mm256_mul_pd(v1, _mm256_set1_pd(real(u11)));
        __m256d w1_imag = Util::productImagD(v1, _mm256_set1_pd(imag(u11)));

        // w1 +=  u10 * v0
        w1_real = _mm256_add_pd(w1_real,
                                _mm256_mul_pd(v0, _mm256_set1_pd(real(u10))));
        w1_imag = _mm256_add_pd(
            w1_imag, Util::productImagD(v0, _mm256_set1_pd(imag(u10))));

        _mm256_store_pd(arr + i0, _mm256_add_pd(w0_real, w0_imag));
        _mm256_store_pd(arr + i1, _mm256_add_pd(w1_real, w1_imag));
    }
}
/// @endcond
} // namespace Pennylane::Gates::AVX2
