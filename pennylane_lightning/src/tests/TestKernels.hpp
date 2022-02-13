#pragma once
/**
 * @file
 * We define test kernels. Note that kernels not registered to
 * AvailableKernels can be also tested by adding it to here.
 */
#include "GateImplementationsLM.hpp"
#include "GateImplementationsPI.hpp"
#include "Macros.hpp"
#if PL_USE_OMP
#include "GateImplementationsParallelLM.hpp"
#include "GateImplementationsParallelPI.hpp"
#endif

#include "TypeList.hpp"

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI,
#if PL_USE_OMP
                              Pennylane::Gates::GateImplementationsParallelLM,
                              Pennylane::Gates::GateImplementationsParallelPI,
#endif
                              void>;
