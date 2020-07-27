#include "pybind11/stl.h"
#include <pybind11/stl_bind.h>
#include "lightning_qubit.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::complex<float>>);

PYBIND11_MODULE(lightning_qubit_ops, m)
{

    py::bind_vector<std::vector<std::complex<float>>>(m, "VectorComplex", py::buffer_protocol());

    m.doc() = "lightning.qubit apply() method using Eigen";
    m.def("apply_2q", apply_2q, "lightning.qubit 2-qubit apply() method");
}
