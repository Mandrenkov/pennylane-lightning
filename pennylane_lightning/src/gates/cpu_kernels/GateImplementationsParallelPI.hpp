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
    constexpr static uint32_t packed_bytes = 4;

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,  GateOperation::PauliY,
        GateOperation::PauliZ,  GateOperation::Hadamard,
        GateOperation::S,       GateOperation::T,
        GateOperation::RX,      GateOperation::RY,
        GateOperation::RZ,      GateOperation::PhaseShift,
        GateOperation::IsingXX, GateOperation::IsingYY,
        GateOperation::IsingZZ,
    };
    constexpr static std::array<GeneratorOperation, 0> implemented_generators =
        {};

    /* Single qubit operators */
    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] =
                std::complex<PrecisionT>{shiftedState[indices[1]].imag(),
                                         -shiftedState[indices[1]].real()};
            shiftedState[indices[1]] =
                std::complex<PrecisionT>{-v0.imag(), v0.real()};
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] = -shiftedState[indices[1]];
        }
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

#pragma omp parallel for
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = Util::INVSQRT2<PrecisionT>() * (v0 + v1);
            shiftedState[indices[1]] = Util::INVSQRT2<PrecisionT>() * (v0 - v1);
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
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
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
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
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
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= s;
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
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] =
                c * v0 + js * std::complex<PrecisionT>{-v1.imag(), v1.real()};
            shiftedState[indices[1]] =
                js * std::complex<PrecisionT>{-v0.imag(), v0.real()} + c * v1;
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
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
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
        for (size_t n = 0; n < externalIndices.size(); n++) {
            const size_t externalIndex = externalIndices[n];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }

    /* Two qubit operators with a parameter */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXX(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexPrecisionT = std::complex<PrecisionT>;
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

#pragma omp parallel for
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const auto v0 = shiftedState[indices[0]];
            const auto v1 = shiftedState[indices[1]];
            const auto v2 = shiftedState[indices[2]];
            const auto v3 = shiftedState[indices[3]];

            shiftedState[indices[0]] = ComplexPrecisionT{
                cr * real(v0) + sj * imag(v3), cr * imag(v0) - sj * real(v3)};
            shiftedState[indices[1]] = ComplexPrecisionT{
                cr * real(v1) + sj * imag(v2), cr * imag(v1) - sj * real(v2)};
            shiftedState[indices[2]] = ComplexPrecisionT{
                cr * real(v2) + sj * imag(v1), cr * imag(v2) - sj * real(v1)};
            shiftedState[indices[3]] = ComplexPrecisionT{
                cr * real(v3) + sj * imag(v0), cr * imag(v3) - sj * real(v0)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingYY(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexPrecisionT = std::complex<PrecisionT>;
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

#pragma omp parallel for
        for (size_t n = 0; n < externalIndices.size(); n++) {
            const size_t externalIndex = externalIndices[n];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const auto v0 = shiftedState[indices[0]];
            const auto v1 = shiftedState[indices[1]];
            const auto v2 = shiftedState[indices[2]];
            const auto v3 = shiftedState[indices[3]];

            shiftedState[indices[0]] = ComplexPrecisionT{
                cr * real(v0) - sj * imag(v3), cr * imag(v0) + sj * real(v3)};
            shiftedState[indices[1]] = ComplexPrecisionT{
                cr * real(v1) + sj * imag(v2), cr * imag(v1) - sj * real(v2)};
            shiftedState[indices[2]] = ComplexPrecisionT{
                cr * real(v2) + sj * imag(v1), cr * imag(v2) - sj * real(v1)};
            shiftedState[indices[3]] = ComplexPrecisionT{
                cr * real(v3) - sj * imag(v0), cr * imag(v3) + sj * real(v0)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        assert(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

#pragma omp parallel for
        for (size_t k = 0; k < externalIndices.size(); k++) {
            const size_t externalIndex = externalIndices[k];
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            shiftedState[indices[0]] *= shifts[0];
            shiftedState[indices[1]] *= shifts[1];
            shiftedState[indices[2]] *= shifts[1];
            shiftedState[indices[3]] *= shifts[0];
        }
    }
};
} // namespace Pennylane::Gates
