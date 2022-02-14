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
 */
#include "DispatchKeys.hpp"
#include "GateOperation.hpp"
#include "KernelType.hpp"

#include <functional>
#include <unordered_map>

namespace Pennylane {

inline auto larger_than(size_t size) {
    return [=](size_t num_qubits) { return num_qubits > size; };
}
inline auto larger_than_equal_to(size_t size) {
    return [=](size_t num_qubits) { return num_qubits >= size; };
}
inline auto less_than(size_t size) {
    return [=](size_t num_qubits) { return num_qubits < size; };
}
inline auto less_than_equal_to(size_t size) {
    return [=](size_t num_qubits) { return num_qubits <= size; };
}
inline auto in_between_closed(size_t l1, size_t l2) {
    return [=](size_t num_qubits) {
        return (l1 <= num_qubits) && (num_qubits <= l2);
    };
}

class DefaultKernelsForStateVector {
  private:
    const static inline std::unordered_map<CPUMemoryModel,
                                           std::vector<Gates::KernelType>>
        allowed_kernels{
            {CPUMemoryModel::Unaligned,
             {Gates::KernelType::LM, Gates::KernelType::PI,
              Gates::KernelType::ParallelLM, Gates::KernelType::ParallelPI}},
            {CPUMemoryModel::Aligned256,
             {Gates::KernelType::LM, Gates::KernelType::PI,
              Gates::KernelType::ParallelLM, Gates::KernelType::ParallelPI}},
            {CPUMemoryModel::Aligned512,
             {Gates::KernelType::LM, Gates::KernelType::PI,
              Gates::KernelType::ParallelLM, Gates::KernelType::ParallelPI}},
        };

    std::unordered_map<
        Gates::GateOperation,
        std::vector<std::tuple<uint32_t, std::function<bool(size_t)>,
                               Gates::KernelType>>>
        kernel_map_;

    void registerThreadsDefaultGates() {
        using Gates::GateOperation;
        auto &instance = *this;
        auto all_qubit_numbers = [](size_t num_qubits) { return true; };
        instance.assignKernelForGate(GateOperation::PauliX, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::PauliY, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::PauliZ, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::Hadamard, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::S, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::T, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::RX, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::RY, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::RZ, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::PhaseShift, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::Rot, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::ControlledPhaseShift,
                                     Threading::All, CPUMemoryModel::All,
                                     all_qubit_numbers, Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CNOT, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CY, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);

        instance.assignKernelForGate(GateOperation::CZ, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::SWAP, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);

        instance.assignKernelForGate(GateOperation::IsingXX, Threading::All,
                                     CPUMemoryModel::All, less_than(12),
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(
            GateOperation::IsingXX, Threading::All, CPUMemoryModel::All,
            in_between_closed(12, 20), Gates::KernelType::PI);
        instance.assignKernelForGate(GateOperation::IsingXX, Threading::All,
                                     CPUMemoryModel::All, larger_than(20),
                                     Gates::KernelType::LM);

        instance.assignKernelForGate(GateOperation::IsingYY, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::IsingZZ, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CRX, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CRY, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CRZ, Threading::All,
                                     CPUMemoryModel::All, all_qubit_numbers,
                                     Gates::KernelType::LM);
    }

  public:
    static struct AllThreading {
    } all_threading;
    static struct AllMemoryModel {
    } all_memory_model;

    void assignKernelForGate(Gates::GateOperation gate, Threading threading,
                             CPUMemoryModel memory_model,
                             std::function<bool(size_t)> num_qubits_creterion,
                             Gates::KernelType kernel) {
        kernel_map_[gate].emplace_back({toDispatchKey(threading, memory_model),
                                        std::move(num_qubits_creterion),
                                        kernel});
    }

    void assignKernelForGate(Gates::GateOperation gate,
                             [[maybe_unused]] AllThreading dummy,
                             CPUMemoryModel memory_model,
                             std::function<bool(size_t)> num_qubits_creterion,
                             Gates::KernelType kernel) {
        Util::for_each_enum<Threading>(
            [](Threading threading) {
            kernel_map_[gate].emplace_back(
                {toDispatchKey(threading, memory_model),
                 std::move(num_qubits_creterion), kernel});
        }
    }

    void assignKernelForGate(Gates::GateOperation gate, Threading threading,
                             [[maybe_unused]] AllMemoryModel dummy,
                             std::function<bool(size_t)> num_qubits_creterion,
                             Gates::KernelType kernel) {
        Util::for_each_enum<CPUMemoryModel>(
            [](CPUMemoryModel memory_model) {
            kernel_map_[gate].emplace_back(
                {toDispatchKey(threading, memory_model),
                 std::move(num_qubits_creterion), kernel});
        }
    }

    static auto getInstance() -> DefaultKernelsForStateVector & {
        static DefaultKernelsForStateVector instance;

        return instance;
    }

    auto getKernelMap(Threading threading, CPUMemoryModel memory_model,
                      size_t num_qubits)
        -> std::unordered_map<Gates::GateOperation, Gates::KernelType> {
        uint32_t dispatch_key = toDispatchKey(threading, memory_model);

        std::unordered_map<Gates::GateOperation, Gates::KernelType>
            kernel_for_gates;

        for (auto gate = Gates::GateOperation::BEGIN;
             gate != Gates::GateOperation::END;
             gate = static_cast<Gates::GateOperation>(
                 static_cast<uint32_t>(gate) + 1)) {

            uint32_t max_dispatch_key = 0;
            Gates::KernelType kernel;
            std::for_each(kernel_map_[gate].cbegin(), kernel_map_[gate].cend(),
                          [&](const auto &t) {
                              if (((std::get<0>(t) & dispatch_key) != 0) &&
                                  std::get<1>(t)(num_qubits) &&
                                  max_dispatch_key < std::get<0>(t)) {
                                  max_dispatch_key = std::get<0>(t);
                                  kernel = std::get<2>(t);
                              }
                          });
            if (max_dispatch_key == 0) {
                throw std::range_error("Cannot find registered kernel for a "
                                       "dispatch key and number of qubits.");
            }
            kernel_for_gates[gate] = kernel;
        }
        return kernel_for_gates;
    }
};
} // namespace Pennylane
