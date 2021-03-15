// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <set>
#include <cmath>

#include "Apply.hpp"
#include "Gates.hpp"
#include "StateVector.hpp"
#include "Util.hpp"

using std::set;
using std::string;
using std::unique_ptr;
using std::vector;

vector<unsigned int> Pennylane::getIndicesAfterExclusion(const vector<unsigned int>& indicesToExclude, const unsigned int qubits) {
    set<unsigned int> indices;
    for (unsigned int i = 0; i < qubits; i++) {
        indices.insert(indices.end(), i);
    }
    for (const unsigned int& excludedIndex : indicesToExclude) {
        indices.erase(excludedIndex);
    }
    return vector<unsigned int>(indices.begin(), indices.end());
}

vector<size_t> Pennylane::generateBitPatterns(const vector<unsigned int>& qubitIndices, const unsigned int qubits) {
    vector<size_t> indices;
    indices.reserve(exp2(qubitIndices.size()));
    indices.push_back(0);
    for (int i = qubitIndices.size() - 1; i >= 0; i--) {
        size_t value = maxDecimalForQubit(qubitIndices[i], qubits);
        size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.push_back(indices[j] + value);
        }
    }
    return indices;
}

void Pennylane::constructAndApplyOperation(
    StateVector& state,
    const string& opLabel,
    const vector<unsigned int>& opWires,
    const vector<double>& opParams,
    const unsigned int qubits
) {
    unique_ptr<AbstractGate> gate = constructGate(opLabel, opParams);
    if (gate->numQubits != opWires.size())
        throw std::invalid_argument(string("The gate of type ") + opLabel + " requires " + std::to_string(gate->numQubits) + " wires, but " + std::to_string(opWires.size()) + " were supplied");

    vector<size_t> internalIndices = generateBitPatterns(opWires, qubits);

    vector<unsigned int> externalWires = getIndicesAfterExclusion(opWires, qubits);
    vector<size_t> externalIndices = generateBitPatterns(externalWires, qubits);

    gate->applyKernel(state, internalIndices, externalIndices);
}

void Pennylane::apply(
    StateVector& state,
    const vector<string>& ops,
    const vector<vector<unsigned int> >& wires,
    const vector<vector<double> >& params,
    const unsigned int qubits
) {
    if (qubits <= 0)
        throw std::invalid_argument("Must specify one or more qubits");

    size_t expectedLength = exp2(qubits);
    if (state.length != expectedLength)
        throw std::invalid_argument(string("Input state vector length (") + std::to_string(state.length) + ") does not match the given number of qubits " + std::to_string(qubits));

    size_t numOperations = ops.size();
    if (numOperations != wires.size() || numOperations != params.size())
        throw std::invalid_argument("Invalid arguments: number of operations, wires, and parameters must all be equal");

    for (int i = 0; i < numOperations; i++) {
        constructAndApplyOperation(state, ops[i], wires[i], params[i], qubits);
    }

}

vector<double> Pennylane::adjointJacobian(
    StateVector& phi,
    const vector<string>& observables,
    const vector<vector<unsigned int> >& obsWires,
    const vector<vector<double> >& obsParams,
    const vector<string>& operations,
    const vector<vector<unsigned int> >& opWires,
    const vector<vector<double> >& opParams,
    const vector<int>& trainableParams,
    int paramNumber
) {
    vector<StateVector> lambdas;
    size_t numOperations = operations.size();
    size_t numObservables = observables.size();
    size_t trainableParamNumber = trainableParams.size() - 1;

    for (int i = 0; i < numObservables; i++) {
        StateVector state = phi;
        Pennylane:constructAndApplyOperation(
            state,
            observables[i],
            obsWires[i],
            obsParams[i],
            obsWires[i].size()
        );
        lambdas.push_back(state);
    }

    for (vector<const string>::reverse_iterator opIt = operations.rbegin(); opIt != operations.rend(); ++opIt) {
        int i = std::distance(opIt, operations.rbegin());

        if ((*opIt != "QubitStateVector") && (*opIt != "BasisState")) {
            Pennylane:constructAndApplyOperation(
                phi,
                *opIt,
                opWires[i],
                opParams[i],
                opWires[i].size()
            );

            // if (paramNumber in trainableParams) {
            //     d_op_matrix = ...
            //     mu = applyUnitary(state, matInv, opWires, qubits)

            //     calculate jacColumn
            //     trainableParamNumber--;
            // }
            // paramNumber--;

            for (vector<StateVector>::iterator lamIt = lambdas.begin(); lamIt != lambdas.end(); ++lamIt) {
                StateVector state = *lamIt;
                int j = std::distance(lamIt, lambdas.begin());
                // useInverse = true;

                Pennylane:constructAndApplyOperation(
                state,
                *opIt,
                opWires[i],
                opParams[i],
                opWires[i].size()
                // useInverse
            );
            lambdas[j] = state;
            }
        }
    }

    // return jac;
}
