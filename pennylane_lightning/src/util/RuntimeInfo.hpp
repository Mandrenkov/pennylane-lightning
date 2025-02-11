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
 * Runtime information based on cpuid
 */
#pragma once
#include <bitset>

namespace Pennylane::Util {
/**
 * @brief This class is only usable in x86 or x86_64 architecture.
 */
class RuntimeInfo {
  private:
    /// @cond DEV
    struct InternalRuntimeInfo {
        InternalRuntimeInfo();

        std::bitset<32> f_1_ecx{};
        std::bitset<32> f_1_edx{};
        std::bitset<32> f_7_ebx{};
        std::bitset<32> f_7_ecx{};
    };
    /// @endcond

    static const inline InternalRuntimeInfo internal_runtime_info_;

  public:
    static inline bool AVX() {
        // NOLINTNEXTLINE(readability-magic-numbers)
        return internal_runtime_info_.f_1_ecx[28];
    }
    static inline bool AVX2() {
        // NOLINTNEXTLINE(readability-magic-numbers)
        return internal_runtime_info_.f_7_ebx[5];
    }
    static inline bool AVX512F() {
        // NOLINTNEXTLINE(readability-magic-numbers)
        return internal_runtime_info_.f_7_ebx[16];
    }
};
} // namespace Pennylane::Util
