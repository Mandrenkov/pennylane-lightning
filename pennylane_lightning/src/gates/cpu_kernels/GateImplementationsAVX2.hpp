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
 * Defines kernel functions with AVX2
 */
#pragma once

// General implementations
#include "avx_common/AVX2Concept.hpp"
#include "avx_common/ApplySingleQubitOp.hpp"
#include "avx_common/ApplyIsingZZ.hpp"
#include "avx_common/ApplyHadamard.hpp"
/*
#include "avx2/ApplyHadamard.hpp"
#include "avx2/ApplyPauliX.hpp"
#include "avx2/ApplyPauliY.hpp"
#include "avx2/ApplyPauliZ.hpp"
#include "avx2/ApplyRZ.hpp"
#include "avx2/ApplyS.hpp"
*/

#include "BitUtil.hpp"
#include "Error.hpp"
#include "GateImplementationsLM.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "Macros.hpp"

#include <immintrin.h>

#include <complex>
#include <vector>

namespace Pennylane::Gates {

class GateImplementationsAVX2 {
  private:
    template <typename T>
    using AVXConcept = AVX::AVX2Concept<T>;

  public:
    constexpr static KernelType kernel_id = KernelType::AVX2;
    constexpr static std::string_view name = "AVX2";
    constexpr static uint32_t data_alignment_in_bytes = 32;

    constexpr static std::array implemented_gates = {
        /*
        GateOperation::PauliX,
        GateOperation::PauliY,
        GateOperation::PauliZ,
        */
        GateOperation::Hadamard,
        /*
        GateOperation::S,
        */
        /* T, RX, RY, PhaseShift, SWAP, IsingXX, IsingYY */
        /*
        GateOperation::RZ,
        */
        GateOperation::Rot,
        /*
        GateOperation::CZ,
        */
        GateOperation::IsingZZ,
    };

    constexpr static std::array<GeneratorOperation, 0> implemented_generators =
        {};

  private:
  public:
    template <typename PrecisionT>
    static void applySingleQubitOp(std::complex<PrecisionT> *arr,
                                   const size_t num_qubits,
                                   const std::complex<PrecisionT> *matrix,
                                   const size_t wire, bool inverse = false) {
        const size_t rev_wire = num_qubits - wire - 1;

        using SingleQubitOpProdAVX2 = AVX::ApplySingleQubitOp<PrecisionT, AVXConcept>;

        if (num_qubits < AVXConcept<PrecisionT>::internal_wires) {
            GateImplementationsLM::applySingleQubitOp(arr, num_qubits, matrix,
                                                      wire, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            switch (rev_wire) {
            case 0:
                SingleQubitOpProdAVX2::template applyInternal<0>(arr, num_qubits,
                                                                   matrix, inverse);
                return;
            case 1:
                SingleQubitOpProdAVX2::template applyInternal<1>(arr, num_qubits, matrix,
                                                      inverse);
                return;
            default:
                SingleQubitOpProdAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                       matrix, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (rev_wire == 0) {
                SingleQubitOpProdAVX2::template applyInternal<0>(arr, num_qubits,
                        matrix, inverse);
            } else {
                SingleQubitOpProdAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                       matrix, inverse);
            }
        }
    }

    /*
    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX2::applyPauliXFloatInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX2::applyPauliXFloatInternal<1>(arr, num_qubits);
                return;
            default:
                AVX2::applyPauliXFloatExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX2::applyPauliXDoubleInternal<0>(arr, num_qubits);
                return;
            default:
                AVX2::applyPauliXDoubleExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliY(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX2::applyPauliYFloatInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX2::applyPauliYFloatInternal<1>(arr, num_qubits);
                return;
            default:
                AVX2::applyPauliYFloatExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliY(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX2::applyPauliYDoubleInternal(arr, num_qubits);
                return;
            default:
                AVX2::applyPauliYDoubleExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliZ(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 2) {
                AVX2::applyPauliZFloatInternal(arr, num_qubits, rev_wire);
                return;
            }
            AVX2::applyPauliZFloatExternal(arr, num_qubits, rev_wire);
            return;
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire == 0) {
                AVX2::applyPauliZDoubleInternal(arr, num_qubits);
                return;
            }
            AVX2::applyPauliZDoubleExternal(arr, num_qubits, rev_wire);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyS(arr, num_qubits, wires, inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX2::applySFloatInternal<0>(arr, num_qubits, inverse);
                return;
            case 1:
                AVX2::applySFloatInternal<1>(arr, num_qubits, inverse);
                return;
            default:
                AVX2::applySFloatExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            switch (rev_wire) {
            case 0:
                AVX2::applySDoubleInternal(arr, num_qubits, inverse);
                return;
            default:
                AVX2::applySDoubleExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
    */
    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        using ApplyHadamardAVX2 = AVX::ApplyHadamard<PrecisionT, AVXConcept>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyHadamard(arr, num_qubits, wires,
                                                     inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyHadamardAVX2::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyHadamardAVX2::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyHadamardAVX2::applyExternal(arr, num_qubits, rev_wire);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;
            if(rev_wire == 0) {
                ApplyHadamardAVX2::template applyInternal<0>(arr, num_qubits);
            } else {
                ApplyHadamardAVX2::applyExternal(arr, num_qubits, rev_wire);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
    /*
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 1);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            } // else
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 2) {
                AVX2::applyRZFloatInternal(arr, num_qubits, rev_wire, inverse,
                                             angle);
                return;
            }
            AVX2::applyRZFloatExternal(arr, num_qubits, rev_wire, inverse,
                                         angle);
            return;
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire == 0) {
                AVX2::applyRZDoubleInternal(arr, num_qubits, inverse, angle);
                return;
            }
            AVX2::applyRZDoubleExternal(arr, num_qubits, rev_wire, inverse,
                                          angle);
            return;
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }
    */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        assert(wires.size() == 1);

        const auto rotMat =
            (inverse) ? Gates::getRot<PrecisionT>(-omega, -theta, -phi)
                      : Gates::getRot<PrecisionT>(phi, theta, omega);

        applySingleQubitOp(arr, num_qubits, rotMat.data(), wires[0]);
    }
    /* Two-qubit gates*/
        /*
    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        if constexpr (std::is_same_v<PrecisionT, float>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
            if (rev_wire0 < 2 && rev_wire1 < 2) {
                AVX2::applyCZFloatInternalInternal(arr, num_qubits);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                AVX2::applyCZFloatInternalExternal(arr, num_qubits, rev_wire0,
                                                     rev_wire1);
            } else {
                AVX2::applyCZFloatExternalExternal(arr, num_qubits, rev_wire0,
                                                     rev_wire1);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (std::min(rev_wire0, rev_wire1) == 0) {
                AVX2::applyCZDoubleInternalExternal(
                        arr, num_qubits,
                        std::max(rev_wire0, rev_wire1));
            } else {
                AVX2::applyCZDoubleExternalExternal(arr, num_qubits,
                                                      rev_wire0, rev_wire1);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }
    */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        assert(wires.size() == 2);

        using ApplyIsingZZAVX2 = AVX::ApplyIsingZZ<PrecisionT, AVXConcept>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyIsingZZ(arr, num_qubits, wires,
                                                    inverse, angle);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplyIsingZZAVX2::applyInternalInternal(arr, num_qubits, 
                        rev_wire0, rev_wire1, inverse, angle);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                ApplyIsingZZAVX2::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (std::min(rev_wire0, rev_wire1) == 0) {
                ApplyIsingZZAVX2::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX2::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                              std::is_same_v<PrecisionT, double>,
                          "Only float and double are supported.");
        }
    }
};
} // namespace Pennylane::Gates
