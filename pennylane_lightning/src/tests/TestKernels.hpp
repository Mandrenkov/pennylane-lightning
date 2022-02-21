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
#endif
#if PL_USE_AVX512F && PL_USE_AVX512DQ
#include "cpu_kernels/GateImplementationsAVX512.hpp"
#endif
#if PL_USE_AVX2
#include "cpu_kernels/GateImplementationsAVX2.hpp"
#endif

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI,
#if PL_USE_OMP
                              Pennylane::Gates::GateImplementationsParallelLM,
#endif
#if PL_USE_AVX512F && PL_USE_AVX512DQ
                              Pennylane::Gates::GateImplementationsAVX512,
#endif
#if PL_USE_AVX2
                              Pennylane::Gates::GateImplementationsAVX2,
#endif
                              void>;
