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
 * Defines kernel functions with AVX512F and AVX512DQ
 */
#pragma once
/*
#include "avx512/ApplyCZ.hpp"
#include "avx512/ApplyHadamard.hpp"
#include "avx512/ApplyIsingZZ.hpp"
#include "avx512/ApplyPauliY.hpp"
#include "avx512/ApplyPauliZ.hpp"
#include "avx512/ApplyRZ.hpp"
#include "avx512/ApplyS.hpp"
*/
#include "avx_common/ApplyPauliX.hpp"
#include "avx_common/ApplyHadamard.hpp"
#include "avx_common/ApplySingleQubitOp.hpp"
#include "avx_common/ApplyIsingZZ.hpp"
#include "avx_common/AVX512Concept.hpp"

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

class GateImplementationsAVX512 {
  private:
    template <typename T>
    using AVXConcept = AVX::AVX512Concept<T>;

  public:
    constexpr static KernelType kernel_id = KernelType::AVX512;
    constexpr static std::string_view name = "AVX512";
    constexpr static uint32_t data_alignment_in_bytes = 64;

    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,
        /*
        GateOperation::PauliY,
        GateOperation::PauliZ,
        */
        GateOperation::Hadamard,
        /*
        GateOperation::S,
        */
        /* T, RX, RY, PhaseShift, SWAP, IsingXX, IsingYY */
        //GateOperation::RZ,
        GateOperation::Rot,
        /*
        GateOperation::CZ,
        */
        GateOperation::IsingZZ,
    };

    constexpr static std::array<GeneratorOperation, 0> implemented_generators =
        {};

    template <typename PrecisionT>
    static void applySingleQubitOp(std::complex<PrecisionT> *arr,
                                   const size_t num_qubits,
                                   const std::complex<PrecisionT> *matrix,
                                   const size_t wire, bool inverse = false) {
        const size_t rev_wire = num_qubits - wire - 1;

        using SingleQubitOpProdAVX512 = AVX::ApplySingleQubitOp<PrecisionT, AVXConcept>;

        if (num_qubits < AVXConcept<PrecisionT>::internal_wires) {
            GateImplementationsLM::applySingleQubitOp(arr, num_qubits, matrix,
                                                      wire, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            switch (rev_wire) {
            case 0:
                SingleQubitOpProdAVX512::template applyInternal<0>(arr, num_qubits,
                                                                   matrix, inverse);
                return;
            case 1:
                SingleQubitOpProdAVX512::template applyInternal<1>(arr, num_qubits, matrix,
                                                      inverse);
                return;
            case 2:
                SingleQubitOpProdAVX512::template applyInternal<2>(arr, num_qubits, matrix,
                                                      inverse);
                return;
            default:
                SingleQubitOpProdAVX512::applyExternal(arr, num_qubits, rev_wire,
                                                       matrix, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            switch (rev_wire) {
            case 0:
                SingleQubitOpProdAVX512::template applyInternal<0>(arr, num_qubits,
                        matrix, inverse);
                return;
            case 1:
                SingleQubitOpProdAVX512::template applyInternal<1>(arr, num_qubits, matrix,
                                                      inverse);
                return;
            default:
                SingleQubitOpProdAVX512::applyExternal(arr, num_qubits, rev_wire,
                                                       matrix, inverse);
                return;
            }
        }
    }
    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliXAVX512 = AVX::ApplyPauliX<PrecisionT, AVXConcept>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyPauliX(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyPauliXAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliXAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            case 2:
                ApplyPauliXAVX512::template applyInternal<2>(arr, num_qubits);
                return;
            default:
                ApplyPauliXAVX512::applyExternal(arr, num_qubits, rev_wire);
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
                ApplyPauliXAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyPauliXAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyPauliXAVX512::applyExternal(arr, num_qubits, rev_wire);
                return;
            }
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    /*
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
                AVX512::applyPauliYFloatInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX512::applyPauliYFloatInternal<1>(arr, num_qubits);
                return;
            case 2:
                AVX512::applyPauliYFloatInternal<2>(arr, num_qubits);
                return;
            default:
                AVX512::applyPauliYFloatExternal(arr, num_qubits, rev_wire);
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
                AVX512::applyPauliYDoubleInternal<0>(arr, num_qubits);
                return;
            case 1:
                AVX512::applyPauliYDoubleInternal<1>(arr, num_qubits);
                return;
            default:
                AVX512::applyPauliYDoubleExternal(arr, num_qubits, rev_wire);
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

            if (rev_wire < 3) {
                AVX512::applyPauliZFloatInternal(arr, num_qubits, rev_wire);
                return;
            }
            AVX512::applyPauliZFloatExternal(arr, num_qubits, rev_wire);
            return;
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyPauliZ(arr, num_qubits, wires,
                                                   inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 2) {
                AVX512::applyPauliZDoubleInternal(arr, num_qubits, rev_wire);
                return;
            }
            AVX512::applyPauliZDoubleExternal(arr, num_qubits, rev_wire);
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
            if (num_qubits < 3) {
                GateImplementationsLM::applyS(arr, num_qubits, wires, inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX512::applySFloatInternal<0>(arr, num_qubits, inverse);
                return;
            case 1:
                AVX512::applySFloatInternal<1>(arr, num_qubits, inverse);
                return;
            case 2:
                AVX512::applySFloatInternal<2>(arr, num_qubits, inverse);
                return;
            default:
                AVX512::applySFloatExternal(arr, num_qubits, rev_wire, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyS(arr, num_qubits, wires, inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                AVX512::applySDoubleInternal<0>(arr, num_qubits, inverse);
                return;
            case 1:
                AVX512::applySDoubleInternal<1>(arr, num_qubits, inverse);
                return;
            default:
                AVX512::applySDoubleExternal(arr, num_qubits, rev_wire,
                                             inverse);
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

        using ApplyHadamardAVX512 = AVX::ApplyHadamard<PrecisionT, AVXConcept>;
        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyHadamard(arr, num_qubits, wires,
                                                     inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyHadamardAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyHadamardAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            case 2:
                ApplyHadamardAVX512::template applyInternal<2>(arr, num_qubits);
                return;
            default:
                ApplyHadamardAVX512::applyExternal(arr, num_qubits, rev_wire);
            }

        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyHadamard(arr, num_qubits, wires,
                                                     inverse);
                return;
            }
            const size_t rev_wire = num_qubits - wires[0] - 1;

            switch (rev_wire) {
            case 0:
                ApplyHadamardAVX512::template applyInternal<0>(arr, num_qubits);
                return;
            case 1:
                ApplyHadamardAVX512::template applyInternal<1>(arr, num_qubits);
                return;
            default:
                ApplyHadamardAVX512::applyExternal(arr, num_qubits, rev_wire);
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
            if (num_qubits < 3) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            } // else
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 3) {
                AVX512::applyRZFloatInternal(arr, num_qubits, rev_wire, inverse,
                                             angle);
                return;
            }
            AVX512::applyRZFloatExternal(arr, num_qubits, rev_wire, inverse,
                                         angle);
            return;
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyRZ(arr, num_qubits, wires, inverse,
                                               angle);
                return;
            } // else
            const size_t rev_wire = num_qubits - wires[0] - 1;

            if (rev_wire < 2) {
                AVX512::applyRZDoubleInternal(arr, num_qubits, rev_wire,
                                              inverse, angle);
                return;
            }
            AVX512::applyRZDoubleExternal(arr, num_qubits, rev_wire, inverse,
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
            if (num_qubits < 3) {
                GateImplementationsLM::applyCZ(arr, num_qubits, wires, inverse);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
            if (rev_wire0 < 3 && rev_wire1 < 3) {
                AVX512::applyCZFloatInternalInternal(arr, num_qubits, rev_wire0,
                                                     rev_wire1);
            } else if (std::min(rev_wire0, rev_wire1) < 3) {
                AVX512::applyCZFloatInternalExternal(arr, num_qubits, rev_wire0,
                                                     rev_wire1);
            } else {
                AVX512::applyCZFloatExternalExternal(arr, num_qubits, rev_wire0,
                                                     rev_wire1);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyCZ(arr, num_qubits, wires, inverse);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                AVX512::applyCZDoubleInternalInternal(arr, num_qubits,
                                                      rev_wire0, rev_wire1);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                AVX512::applyCZDoubleInternalExternal(arr, num_qubits,
                                                      rev_wire0, rev_wire1);
            } else {
                AVX512::applyCZDoubleExternalExternal(arr, num_qubits,
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

        using ApplyIsingZZAVX512 = AVX::ApplyIsingZZ<PrecisionT, AVXConcept>;

        if constexpr (std::is_same_v<PrecisionT, float>) {
            if (num_qubits < 3) {
                GateImplementationsLM::applyIsingZZ(arr, num_qubits, wires,
                                                    inverse, angle);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit
            if (rev_wire0 < 3 && rev_wire1 < 3) {
                ApplyIsingZZAVX512::applyInternalInternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else if (std::min(rev_wire0, rev_wire1) < 3) {
                ApplyIsingZZAVX512::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX512::applyExternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (num_qubits < 2) {
                GateImplementationsLM::applyIsingZZ(arr, num_qubits, wires,
                                                    inverse, angle);
                return;
            }
            const size_t rev_wire0 = num_qubits - wires[1] - 1;
            const size_t rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

            if (rev_wire0 < 2 && rev_wire1 < 2) {
                ApplyIsingZZAVX512::applyInternalInternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else if (std::min(rev_wire0, rev_wire1) < 2) {
                ApplyIsingZZAVX512::applyInternalExternal(
                    arr, num_qubits, rev_wire0, rev_wire1, inverse, angle);
            } else {
                ApplyIsingZZAVX512::applyExternalExternal(
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
