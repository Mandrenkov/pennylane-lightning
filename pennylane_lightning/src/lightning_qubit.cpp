//#include "pybind11/stl.h"
#include <pybind11/numpy.h>
#include "lightning_qubit.hpp"
//#include <pybind11/pybind11.h>


PYBIND11_MODULE(lightning_qubit, m)
{
    m.doc() = "lightning.qubit apply() method using Eigen";
    m.def("apply_2q", apply_2q, "lightning.qubit 2-qubit apply() method");
}
