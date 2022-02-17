#include "TestHelpers.hpp"
#include "TestKernels.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file Test_GateImplementations_Nonparam.cpp
 *
 * This file contains tests for non-parameterized gates. List of such
 * gates are [PauliX, PauliY, PauliZ, Hadamard, S, T, CNOT, SWAP, CZ, Toffoli,
 * CSWAP].
 */
using namespace Pennylane;
using namespace Pennylane::Gates;
using namespace Pennylane::Util;

namespace {
using std::vector;
}

template<typename TypeList>
std::string kernelsToString() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        if constexpr (!std::is_same_v<typename TypeList::Next, void>) {
            return std::string(TypeList::Type::name) + ", " +
                kernelsToString<typename TypeList::Next>();
        }
        return std::string(TypeList::Type::name);
    }
}

/* Type transformation */
template <Gates::GateOperation gate_op, typename TypeList>
struct KernelsImplementingGateHelper {
    using Type = std::conditional_t<
        array_has_elt(TypeList::Type::implemented_gates, gate_op),
        typename PrependToTypeList<typename TypeList::Type, 
            typename KernelsImplementingGateHelper<gate_op, typename TypeList::Next>
            ::Type>::Type,
        typename KernelsImplementingGateHelper<gate_op, typename TypeList::Next>::Type
    >;
};
template <Gates::GateOperation gate_op>
struct KernelsImplementingGateHelper<gate_op, void> {
    using Type = void;
};

template <Gates::GateOperation gate_op>
struct KernelsImplementingGate {
    using Type = typename KernelsImplementingGateHelper<gate_op, TestKernels>::Type;
};

template<Gates::GateOperation gate_op, typename PrecisionT, typename ParamT,
    typename GateImplementation>
auto applyGate(TestVector<std::complex<PrecisionT>> ini, 
               size_t num_qubits,
               const std::vector<size_t>& wires,
               bool inverse,
               const std::vector<ParamT>& params)
    -> TestVector<std::complex<PrecisionT>> {
    callGateOps(
        GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation, gate_op>::value,
        ini.data(), 
        num_qubits, wires, inverse, params);
    return ini;
}

template<Gates::GateOperation gate_op, typename PrecisionT, typename ParamT,
    typename Kernels, size_t... I> 
auto applyGateForImplemetingKernels(
        const TestVector<std::complex<PrecisionT>>& ini, 
        size_t num_qubits,
        const std::vector<size_t>& wires,
        bool inverse,
        const std::vector<ParamT>& params,
        [[maybe_unused]] std::index_sequence<I...> dummy) { 
    return std::make_tuple(
        applyGate<gate_op, PrecisionT, ParamT, getNthType<Kernels, I>>(
            ini, num_qubits, wires, inverse, params)...
    );
}

template<Gates::GateOperation gate_op, typename PrecisionT, typename ParamT,
    class RandomEngine>
void testApplyGate(RandomEngine& re, size_t num_qubits) {
    const auto ini = createRandomState<PrecisionT>(re, num_qubits);

    using Kernels = typename KernelsImplementingGate<gate_op>::Type;

    INFO("Kernels implementing " << lookup(Constant::gate_names, gate_op)
            << " are " << kernelsToString<Kernels>());

    INFO("PrecisionT, ParamT = " << PrecisionToName<PrecisionT>::value 
            << ", " << PrecisionToName<ParamT>::value);
    
    if constexpr (gate_op != GateOperation::Matrix) {
        const auto wires = createWires(gate_op, num_qubits);
        const auto params = createParams<ParamT>(gate_op);

        const auto results = 
            Util::tuple_to_array(applyGateForImplemetingKernels<gate_op, PrecisionT, ParamT, Kernels>(
                    ini, num_qubits, wires, false, params, std::make_index_sequence<length<Kernels>()>()));

        for(size_t i = 0; i < results.size() - 1; i++) {
            REQUIRE(results[i] == PLApprox(results[i+1]).margin(1e-7));
        }
    }
}

template<size_t gate_idx, typename PrecisionT, typename ParamT,
    class RandomEngine>
void testAllGatesIter(RandomEngine& re, size_t max_num_qubits) {
    if constexpr (gate_idx < static_cast<size_t>(GateOperation::END)) {
        constexpr static auto gate_op = static_cast<GateOperation>(gate_idx);

        size_t min_num_qubits = array_has_elt(Constant::multi_qubit_gates,
                gate_op)?1:lookup(Constant::gate_wires, gate_op);
        for (size_t num_qubits = min_num_qubits;
                num_qubits < max_num_qubits;
                num_qubits ++) {
            testApplyGate<gate_op, PrecisionT, ParamT>(re, num_qubits);
        }
        testAllGatesIter<gate_idx+1, PrecisionT, ParamT>(re, max_num_qubits);
    }
}

template<typename PrecisionT, typename ParamT, class RandomEngine>
void testAllGates(RandomEngine& re, size_t max_num_qubits) {
    testAllGatesIter<0, PrecisionT, ParamT>(re, max_num_qubits);
}

TEMPLATE_TEST_CASE("Test all kernels give the same results",
                   "[Test_GateImplementations_CompareKernels]",
                   float, double) {
    std::mt19937 re{1337};
    testAllGates<TestType, TestType>(re, 6);
}