#pragma once
/**
 * @file
 * We define test kernels. Note that kernels not registered to
 * AvailableKernels can be also tested by adding it to here.
 */
#include "Macros.hpp"
#include "TypeList.hpp"

#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"
#if PL_USE_OMP
#include "cpu_kernels/GateImplementationsParallelLM.hpp"
#include "cpu_kernels/GateImplementationsParallelPI.hpp"
#endif

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI,
#if PL_USE_OMP
                              Pennylane::Gates::GateImplementationsParallelLM,
                              Pennylane::Gates::GateImplementationsParallelPI,
#endif
                              void>;
