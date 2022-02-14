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
#if PL_USE_AVX512F
#include "cpu_kernels/GateImplementationsAVX512.hpp"
#endif

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI,
#if PL_USE_OMP
                              Pennylane::Gates::GateImplementationsParallelLM,
                              Pennylane::Gates::GateImplementationsParallelPI,
#endif
#if PL_USE_AVX512F
                              Pennylane::Gates::GateImplementationsAVX512,
#endif
                              void>;
