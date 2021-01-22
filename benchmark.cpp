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
/**
 * @file
 * Basic benchmarking file that allows applying StronglyEntangling layers to a state.
 */
#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>
#include "pennylane_lightning/src/lightning_qubit.hpp"

using std::vector;
using std::string;

#include <random>

constexpr double PI = 3.14159265358;

float random_param(){
    std::mt19937 rng( std::random_device{}() );
    std::uniform_real_distribution<> dist(0, 2*PI);
    return dist(rng);
}

void strongly_entangling(vector<string> &ops, vector<vector<int>> &orig_wires, const vector<int> &layer_wires,
        vector<vector<float>> & params){

    // Rotations
    for(int i : layer_wires) {
        ops.push_back("Rot");
        vector<int> w = {i};
        orig_wires.push_back(w);
        vector<float> p = {random_param(), random_param(), random_param()};
        params.push_back(p);
    }

    // CNOTs
    for(int i=0; i<layer_wires.size()-1; ++i) {
        ops.push_back("CNOT");
        vector<int> w = {layer_wires.at(i), layer_wires.at(i+1)};
        orig_wires.push_back(w);
        vector<float> p = {};
        params.push_back(p);
    }

    // CNOT for the cycle
    ops.push_back("CNOT");
    vector<int> w = {layer_wires.at(layer_wires.size()-1), layer_wires.at(0)};
    orig_wires.push_back(w);
    vector<float> p = {};
    params.push_back(p);
}

int main(){
    
    bool output = false;

    // Benchmark params
    const int qubits = 20;
    const int num_layers = 10;

    // Prep 0 state
    VectorXcd state(int(std::pow(2, qubits)));
    state(0) = 1;

    // Init operation params
    vector<string> ops = {};
    vector<vector<int>> wires = {};
    vector<vector<float>> params = {};

    // The wires strongly entangling layers should act on
    vector<int> entangling_wires = {8,9,3,2,1,0,5,6,4};

    // Call the layers multiple times
    for (int i=0; i<num_layers; ++i){
        strongly_entangling(ops, wires, entangling_wires, params);
    }

    // Apply the operations
    apply (state, ops, wires, params, qubits);

    // Outputs (optional)
    if (output == true){
        std::cout << "Ops: ";
        for(string i : ops) {
            std::cout << i << "   ";
        }

        std::cout << std::endl << "Wires: ";
        for(const vector<int> & i : wires) {
            std:: cout << "New vec: ";
            for(int j: i) {
                std::cout << j << "   ";
            }
        }
        std::cout << std::endl << "Params: ";
        for(const vector<float> & i : params) {
            std:: cout << "New vec: ";
            for(float j: i) {
                std::cout << j << "   ";
            }
        }

        std::cout << state;
    }

    return 0;
}
