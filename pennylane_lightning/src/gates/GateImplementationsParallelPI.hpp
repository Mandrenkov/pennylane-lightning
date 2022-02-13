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
 * Defines gate operations with precomputed indices
 */
#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include "GateImplementationsPI.hpp"
#include "GateOperation.hpp"
#include "GateUtil.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "Macros.hpp"
#include "PauliGenerator.hpp"
#include "Util.hpp"

#include <complex>
#include <vector>

namespace Pennylane::Gates {
/**
 * @brief Kernel functions for gate operations with precomputed indices
 *
 * For given wires, we first compute the indices the gate applies to and use
 * the computed indices to apply the operation.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * */
class GateImplementationsParallelPI
    : public PauliGenerator<GateImplementationsParallelPI> {
  public:
    constexpr static KernelType kernel_id = KernelType::ParallelPI;
    constexpr static std::string_view name = "ParallelPI";
    constexpr static uint32_t data_alignment_in_bytes = 1;

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,     GateOperation::PauliY, GateOperation::PauliZ,
        GateOperation::Hadamard,   GateOperation::S,      GateOperation::T,
        GateOperation::RX,         GateOperation::RY,     GateOperation::RZ,
        GateOperation::PhaseShift,
    };
    constexpr static std::array<GeneratorOperation, 0> implemented_generators =
        {};

    /* Single qubit operators */
    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyPauliX(arr, num_qubits, wires, inverse);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
            }
        }
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyPauliY(arr, num_qubits, wires, inverse);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                std::complex<PrecisionT> v0 = shiftedState[indices[0]];
                shiftedState[indices[0]] =
                    std::complex<PrecisionT>{shiftedState[indices[1]].imag(),
                                             -shiftedState[indices[1]].real()};
                shiftedState[indices[1]] =
                    std::complex<PrecisionT>{-v0.imag(), v0.real()};
            }
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyPauliZ(arr, num_qubits, wires, inverse);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                shiftedState[indices[1]] = -shiftedState[indices[1]];
            }
        }
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyHadamard(arr, num_qubits, wires,
                                                 inverse);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;

                const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
                const std::complex<PrecisionT> v1 = shiftedState[indices[1]];

                shiftedState[indices[0]] =
                    Util::INVSQRT2<PrecisionT>() * (v0 + v1);
                shiftedState[indices[1]] =
                    Util::INVSQRT2<PrecisionT>() * (v0 - v1);
            }
        }
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyS(arr, num_qubits, wires, inverse);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> shift =
            (inverse) ? -Util::IMAG<PrecisionT>() : Util::IMAG<PrecisionT>();

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                shiftedState[indices[1]] *= shift;
            }
        }
    }

    template <class PrecisionT>
    static void applyT(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyT(arr, num_qubits, wires, inverse);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> shift =
            (inverse) ? std::conj(std::exp(std::complex<PrecisionT>(
                            0, static_cast<PrecisionT>(M_PI / 4))))
                      : std::exp(std::complex<PrecisionT>(
                            0, static_cast<PrecisionT>(M_PI / 4)));

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                shiftedState[indices[1]] *= shift;
            }
        }
    }

    /* Single qubit operators with a parameter */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                ParamT angle) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyPhaseShift(arr, num_qubits, wires,
                                                   inverse, angle);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> s =
            inverse ? std::conj(std::exp(std::complex<PrecisionT>(0, angle)))
                    : std::exp(std::complex<PrecisionT>(0, angle));

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                shiftedState[indices[1]] *= s;
            }
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyRX(arr, num_qubits, wires, inverse,
                                           angle);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
                const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
                shiftedState[indices[0]] =
                    c * v0 +
                    js * std::complex<PrecisionT>{-v1.imag(), v1.real()};
                shiftedState[indices[1]] =
                    js * std::complex<PrecisionT>{-v0.imag(), v0.real()} +
                    c * v1;
            }
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyRY(arr, num_qubits, wires, inverse,
                                           angle);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
                const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
                shiftedState[indices[0]] = c * v0 - s * v1;
                shiftedState[indices[1]] = s * v0 + c * v1;
            }
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        assert(wires.size() == 1);
        if (num_qubits < 4) {
            GateImplementationsPI::applyRZ(arr, num_qubits, wires, inverse,
                                           angle);
            return;
        }
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>(std::cos(angle / 2), -std::sin(angle / 2));
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>(std::cos(angle / 2), std::sin(angle / 2));
        const std::complex<PrecisionT> shift1 =
            (inverse) ? std::conj(first) : first;
        const std::complex<PrecisionT> shift2 =
            (inverse) ? std::conj(second) : second;

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n += 8) {
            PL_UNROLL_LOOP
            for (size_t i = 0; i < 8; i++) {
                const size_t externalIndex = externalIndices[n + i];
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                shiftedState[indices[0]] *= shift1;
                shiftedState[indices[1]] *= shift2;
            }
        }
    }
};
} // namespace Pennylane::Gates
