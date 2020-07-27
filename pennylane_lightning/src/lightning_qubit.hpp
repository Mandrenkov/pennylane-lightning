#pragma once

#include <iostream>
#include <vector>
#include <string>
#include "tensor.h"

#include "lightning_qubit.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pybind11/stl.h"

namespace py = pybind11;

using std::vector;
using std::string;
using namespace qflex;

using std::array;
using std::complex;


const double SQRT2INV = 0.7071067811865475;

vector<int> calc_perm(vector<int> perm, int qubits) {
    for (int j = 0; j < qubits; j++) {
        if (count(perm.begin(), perm.end(), j) == 0) {
        perm.push_back(j);
        }
    }
    return perm;
}

/*
Gate_1q get_gate_1q(const string &gate_name, const vector<float> &params) {
    Gate_1q op;

    if (params.empty()){
        pfunc_1q f = OneQubitOps.at(gate_name);
        op = (*f)();
    }
    else if (params.size() == 1){
        pfunc_1q_one_param f = OneQubitOpsOneParam.at(gate_name);
        op = (*f)(params[0]);
    }
    else if (params.size() == 3){
        pfunc_1q_three_params f = OneQubitOpsThreeParams.at(gate_name);
        op = (*f)(params[0], params[1], params[2]);
    }
    return op;
}


Gate_2q get_gate_2q(const string &gate_name, const vector<float> &params) {
    Gate_2q op;

    if (params.empty()) {
        pfunc_2q f = TwoQubitOps.at(gate_name);
        op = (*f)();
    }
    return op;
}
*/

qflex::Tensor tensordot_aux(
			  qflex::Tensor &a, qflex::Tensor &b,
			  const std::vector<size_t> &axes_a,
			  const std::vector<size_t> &axes_b
			  )
{
  if(axes_a.size() != axes_b.size()){
    exit(1);
  }
  const std::vector<std::string>& indices_a = a.get_indices();
  const std::vector<std::string>& indices_b = b.get_indices();
  std::vector<size_t> dimensions_a = a.get_dimensions();
  std::vector<size_t> dimensions_b = b.get_dimensions();

  size_t dim = 1;
  for (int i = 0; i < axes_a.size(); i++){
    dimensions_a[axes_a[i]] = 1;
  }
  for (int i = 0; i < axes_b.size(); i++){
    dimensions_b[axes_b[i]] = 1;
  }
  for (int i = 0; i < dimensions_a.size(); i++){
    dim *= dimensions_a[i];
  }
  for (int i = 0; i < dimensions_b.size(); i++){
    dim *= dimensions_b[i];
  }

  qflex::Tensor res({"x"},{dim});
  for (int i = 0; i < a.get_dimensions().size(); i++){
    if (indices_a[i] == indices_b[i]){
      a.rename_index(indices_a[i], indices_b[i] + "1");
    }
  }

  for(int i = 0; i < axes_a.size(); i++){
    if (indices_a[axes_a[i]] != indices_b[axes_b[i]])
      a.rename_index(indices_a[axes_a[i]], indices_b[axes_b[i]]);
  }

  std::vector<qflex::s_type> scratch(a.size() > b.size() ? a.size() : b.size());
  qflex::multiply(a, b, res, scratch.data());

  const std::vector<size_t> & dimensions = res.get_dimensions();
  /*
  local_size_ = 1;
  for (int i = 0; i < dimensions.size(); i++){
    local_size_ *= dimensions[i];
  }
  */
  return res;
}

vector<std::complex<float>> array_to_vector(py::array xs){
    py::buffer_info info = xs.request();
    auto ptr = static_cast<double *>(info.ptr);

    int n = 1;
    for (auto r: info.shape) {
      n *= r;
    }

    return vector<std::complex<float>>(ptr, ptr+n);

}

py::array_t<std::complex<float>> apply_2q(
    py::array_t<std::complex<float>> in_state,
    vector<string> ops,
    vector<vector<int>> wires,
    vector<vector<double>> params
    ) {

    for (int j = 0; j < in_state.size(); j++) {
        py::print(*(in_state.data()+j));
    }
    std::vector<std::string> letters = {"a", "b"};
    std::vector<size_t> dims = {2, 2};

    std::vector<std::complex<float>> state;
    state.assign(in_state.mutable_data(), in_state.mutable_data()+in_state.size());

    // std::vector<std::complex<float>> state(in_state.data());
    Tensor state_tensor(letters, dims, state);

    for (int j = 0; j < state_tensor.size(); j++) {
        py::print(*(state_tensor.data()+j));
    }

    // PauliX
    std::vector<std::string> pauli_letters = {"c", "d"};
    std::vector<size_t> pauli_dims = {2, 2};

    Tensor PauliX(pauli_letters, pauli_dims, {0, 1, 1, 0});

    std::vector<size_t> axes_a = {0};
    std::vector<size_t> axes_b = {1};

    auto out_state = tensordot_aux(state_tensor, PauliX, axes_a, axes_b);

    /*
    State_2q state_tensor = TensorMap<State_2q>(state.data(), 2, 2);
    State_2q evolved_tensor = state_tensor;

    for (int i = 0; i < ops.size(); i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        vector<float> p = params[i];
        State_2q tensor_contracted;

        if (w.size() == 1) {
            Gate_1q op_1q = get_gate_1q(op_string, p);
            Pairs_1q pairs_1q = {Pairs(1, w[0])};
            tensor_contracted = op_1q.contract(evolved_tensor, pairs_1q);
        }
        else if (w.size() == 2) {
            Gate_2q op_2q = get_gate_2q(op_string, p);
            Pairs_2q pairs_2q = {Pairs(2, w[0]), Pairs(3, w[1])};
            tensor_contracted = op_2q.contract(evolved_tensor, pairs_2q);
        }

        auto perm = calc_perm(w, qubits);
        evolved_tensor = tensor_contracted.shuffle(perm);
    }

    auto out_state = Map<VectorXcd> (evolved_tensor.data(), 4, 1);
    */
    // Pointer to the data
    auto v = new std::vector<complex<float>>(out_state.data(), out_state.data()+out_state.size());
    auto capsule = py::capsule(out_state.data(), [](void *v) { delete reinterpret_cast<std::vector<int>*>(v); });

    py::print(out_state.size(), out_state.data());
    //py::print(v->size(), v->data());
    auto py_array = py::array(v->size(), v->data(), capsule);
    return py_array;
}
