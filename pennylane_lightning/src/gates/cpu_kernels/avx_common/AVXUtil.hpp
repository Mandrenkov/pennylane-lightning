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
 * Defines common utility functions for all AVX
 */
#pragma once
#include "BitUtil.hpp"
#include "Util.hpp"

#include <cstdlib>

namespace Pennylane::Gates::AVX {
// function aliases
[[maybe_unused]] constexpr static auto &fillLeadingOnes =
    Pennylane::Util::fillLeadingOnes;
[[maybe_unused]] constexpr static auto &fillTrailingOnes =
    Pennylane::Util::fillTrailingOnes;
[[maybe_unused]] constexpr static auto &exp2 = Pennylane::Util::exp2;
} // namespace Pennylane::Gates::AVX
