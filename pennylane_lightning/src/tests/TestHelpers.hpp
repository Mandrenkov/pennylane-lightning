#include <complex>
#include <vector>

#include "Util.hpp"

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool isApproxEqual(
    const std::vector<Data_t> &data1, const std::vector<Data_t> &data2,
    const typename Data_t::value_type eps =
        std::numeric_limits<typename Data_t::value_type>::epsilon() * 100) {
    if (data1.size() != data2.size())
        return false;

    for (size_t i = 0; i < data1.size(); i++) {
        if (data1[i].real() != Approx(data2[i].real()).epsilon(eps) ||
            data1[i].imag() != Approx(data2[i].imag()).epsilon(eps)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool
isApproxEqual(const Data_t &data1, const Data_t &data2,
              const typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    if (data1.real() != Approx(data2.real()).epsilon(eps) ||
        data1.imag() != Approx(data2.imag()).epsilon(eps)) {
        return false;
    }
    return true;
}

/**
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t>
void scaleVector(std::vector<std::complex<Data_t>> &data,
                 std::complex<Data_t> scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief Naive commutator operation generator for testing-use only.
 *
 * @tparam Data_t
 * @param left_op
 * @param right_op
 * @param tpose
 * @return std::vector<std::complex<Data_t>>
 */
template <class Data_t>
std::vector<std::complex<Data_t>>
commutator(const std::vector<std::complex<Data_t>> &left_op,
           const std::vector<std::complex<Data_t>> &right_op,
           bool tpose = false) {
    using namespace Pennylane::Util;
    std::vector<std::complex<Data_t>> out1(4, {0, 0});
    std::vector<std::complex<Data_t>> out2(
        4, {0, 0}); // To be replaced with use of BLAS beta
    MatMatProd<Data_t>(left_op.data(), right_op.data(), out1.data(), 2, 2, 2,
                       tpose);
    MatMatProd<Data_t>(right_op.data(), left_op.data(), out2.data(), 2, 2, 2,
                       tpose);
    std::transform(out2.cbegin(), out2.cend(), out2.begin(), std::negate<>{});
    std::transform(out1.begin(), out1.end(), out2.begin(), out1.begin(),
                   std::plus<std::complex<Data_t>>());
    return out1;
}