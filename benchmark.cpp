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
 *
 * It uses the Google Benchmark library to run benchmarks.
 */
#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>
#include "pennylane_lightning/src/lightning_qubit.hpp"

// Include Google benchmark
#include <benchmark/benchmark.h>

using std::vector;
using std::string;

#include <random>

constexpr double PI = 3.14159265358;

/**
* Generates a random floating point number.
*
* @return the random float
*/
float random_param(){
    return static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/2*PI));
}

/**
* Applies a single strongly entangling layer and updates the provided operations, wires and parameters.
*
* @param ops a vector of the original operations that is appended by applying a layer
* @param orig_wires a vector of the original operation wires that is appended by applying a layer
* @param layer_wires a vector of wires for the operations applied in this layer
* @param params a vector of parameters corresponding to the operations specified in ops
*/
void strongly_entangling_layer(vector<string> &ops, vector<vector<int>> &orig_wires, const vector<int> &layer_wires,
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

/**
* Applies strongly entangling layers.
*
* @param output determines whether or not further information about the layers should be outputted
* @param ops a vector of the original operations that is appended by applying a layer
* @param qubits the number of qubits in the overall system
* @param num_layers the number of layers to be applied
* @param entangling_wires the wires to apply the entangling layers to
*/
void apply_strongly_entangling_layers(bool output, const int qubits, const int num_layers, vector<int>
        entangling_wires){

    // Prep 0 state
    VectorXcd state(int(std::pow(2, qubits)));
    state(0) = 1;

    // Init operation params
    vector<string> ops = {};
    vector<vector<int>> wires = {};
    vector<vector<float>> params = {};

    // Call the layers multiple times
    for (int i=0; i<num_layers; ++i){
        strongly_entangling_layer(ops, wires, entangling_wires, params);
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
}

// --------------------------------------
// Benchmark functions with pre-defined parameters
// Consecutive entangling wiring

void benchmark_5_qubits_consecutive(){

    // Benchmark params
    const int qubits = 5;
    const int num_layers = 30;

    // The consecutive wires strongly entangling layers should act on
    vector<int> entangling_wires = {0,1,2,3,4};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

void benchmark_10_qubits_consecutive(){

    // Benchmark params
    const int qubits = 10;
    const int num_layers = 20;

    // The consecutive wires strongly entangling layers should act on
    vector<int> entangling_wires = {0,1,2,3,4,5,6,7,8,9};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

void benchmark_15_qubits_consecutive(){

    // Benchmark params
    const int qubits = 15;
    const int num_layers = 10;

    // The consecutive wires strongly entangling layers should act on
    vector<int> entangling_wires = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

void benchmark_20_qubits_consecutive(){

    // Benchmark params
    const int qubits = 20;
    const int num_layers = 1;

    // The consecutive wires strongly entangling layers should act on
    vector<int> entangling_wires = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

// Pseudorandom entangling wiring
void benchmark_5_qubits(){

    // Benchmark params
    const int qubits = 5;
    const int num_layers = 30;

    // The wires strongly entangling layers should act on
    vector<int> entangling_wires = {3,0,1,2,4};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

void benchmark_10_qubits(){

    // Benchmark params
    const int qubits = 10;
    const int num_layers = 20;

    // The wires strongly entangling layers should act on
    vector<int> entangling_wires = {3,0,1,2,4, 5,8,7,9};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

void benchmark_15_qubits(){

    // Benchmark params
    const int qubits = 15;
    const int num_layers = 10;

    // The wires strongly entangling layers should act on
    vector<int> entangling_wires = {3,0,1,2,4, 5,8,7,9, 12, 11, 14};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

void benchmark_20_qubits(){

    // Benchmark params
    const int qubits = 20;
    const int num_layers = 1;

    // The wires strongly entangling layers should act on
    vector<int> entangling_wires = {3,0,1,2,4, 5,8,7,9, 12, 11, 14, 15, 13, 17, 18, 19};

    bool output = false;
    apply_strongly_entangling_layers(output, qubits, num_layers, entangling_wires);
}

// --------------------------------------
// Auxiliary benchmark functions needed with Google Benchmark

// Consecutive entangling wiring
// 5 qubits
static void BM_5QubitStronglyEntanglingConsecutive(benchmark::State& state) {
  for (auto _ : state)
    benchmark_5_qubits();
}

// 10 qubits
static void BM_10QubitStronglyEntanglingConsecutive(benchmark::State& state) {
  for (auto _ : state)
    benchmark_10_qubits();
}

// 15 qubits
static void BM_15QubitStronglyEntanglingConsecutive(benchmark::State& state) {
  for (auto _ : state)
    benchmark_15_qubits();
}

// 20 qubits
static void BM_20QubitStronglyEntanglingConsecutive(benchmark::State& state) {
  for (auto _ : state)
    benchmark_20_qubits();
}

// Register the functions as benchmarks
BENCHMARK(BM_5QubitStronglyEntanglingConsecutive);
BENCHMARK(BM_10QubitStronglyEntanglingConsecutive);
BENCHMARK(BM_15QubitStronglyEntanglingConsecutive);
BENCHMARK(BM_20QubitStronglyEntanglingConsecutive);

// Pseudorandom entangling wiring
// 5 qubits
static void BM_5QubitStronglyEntangling(benchmark::State& state) {
  for (auto _ : state)
    benchmark_5_qubits();
}

// 10 qubits
static void BM_10QubitStronglyEntangling(benchmark::State& state) {
  for (auto _ : state)
    benchmark_10_qubits();
}

// 15 qubits
static void BM_15QubitStronglyEntangling(benchmark::State& state) {
  for (auto _ : state)
    benchmark_15_qubits();
}

// 20 qubits
static void BM_20QubitStronglyEntangling(benchmark::State& state) {
  for (auto _ : state)
    benchmark_20_qubits();
}

// Register the functions as benchmarks
BENCHMARK(BM_5QubitStronglyEntangling);
BENCHMARK(BM_10QubitStronglyEntangling);
BENCHMARK(BM_15QubitStronglyEntangling);
BENCHMARK(BM_20QubitStronglyEntangling);

BENCHMARK_MAIN();
