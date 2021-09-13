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
 * Contains uncategorised utility functions.
 */
#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <type_traits>

#include <iostream>

#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
#include <cblas.h>
#define USE_CBLAS ;
#else
#endif

namespace Pennylane {

namespace Util {

/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam T Precision of complex value and result.
 * @tparam U Precision of real value.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr std::complex<T> ConstMult(U a, std::complex<T> b) {
    return {a * b.real(), a * b.imag()};
}

/**
 * @brief Compile-time scalar complex times complex.
 *
 * @tparam T Precision of complex value `a` and result.
 * @tparam U Precision of complex value `b`.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr std::complex<T> ConstMult(std::complex<U> a,
                                                  std::complex<T> b) {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
template <class T, class U = T>
inline static constexpr std::complex<T> ConstMultConj(std::complex<U> a,
                                                      std::complex<T> b) {
    return {a.real() * b.real() + a.imag() * b.imag(),
            -a.imag() * b.real() + a.real() * b.imag()};
}

template <class T, class U = T>
inline static constexpr std::complex<T> ConstSum(std::complex<U> a,
                                                 std::complex<T> b) {
    return a + b;
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class T> inline static constexpr std::complex<T> ONE() {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class T> inline static constexpr std::complex<T> ZERO() {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class T> inline static constexpr std::complex<T> IMAG() {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class T> inline static constexpr T SQRT2() {
    if constexpr (std::is_same_v<T, float>) {
        return {0x1.6a09e6p+0f};
    } else {
        return {0x1.6a09e667f3bcdp+0};
    }
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class T> inline static constexpr T INVSQRT2() {
    return {1 / SQRT2<T>()};
}

/**
 * Calculates 2^n for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n
 */
inline size_t exp2(const size_t &n) { return static_cast<size_t>(1) << n; }

/**
 * @brief
 *
 * @param value
 * @return size_t
 */
inline size_t log2(size_t value) {
    return static_cast<size_t>(std::log2(value));
}

/**
 * Calculates the decimal value for a qubit, assuming a big-endian convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline size_t maxDecimalForQubit(size_t qubitIndex, size_t qubits) {
    assert(qubitIndex < qubits);
    return exp2(qubits - qubitIndex - 1);
}

/**
 * @brief Returns the number of wires supported by a given qubit gate.
 *
 * @tparam T Floating point precision type.
 * @param data Gate matrix data.
 * @return size_t Number of wires.
 */
template <class T> inline size_t dimSize(const std::vector<T> &data) {
    const size_t s = data.size();
    const size_t s_sqrt = std::sqrt(s);

    if (s < 4)
        throw std::invalid_argument("The dataset must be at least 2x2.");
    if (((s == 0) || (s & (s - 1))))
        throw std::invalid_argument("The dataset must be a power of 2");
    if (s_sqrt * s_sqrt != s)
        throw std::invalid_argument("The dataset must be a perfect square");

    return log2(sqrt(data.size()));
}

/**
 * @brief Calculates the inner-product using the best available method.
 *
 * @tparam T Floating point precision type.
 * @param data_1 Complex data array 1.
 * @param data_2 Complex data array 2.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T>
std::complex<T> innerProd(const std::complex<T> *data_1,
                          const std::complex<T> *data_2,
                          const size_t data_size) {
    std::complex<T> result(0, 0);

#ifdef USE_CBLAS
    if constexpr (std::is_same_v<T, float>)
        cblas_cdotu_sub(data_size, data_1, 1, data_2, 1, &result);
    else if constexpr (std::is_same_v<T, double>)
        cblas_zdotu_sub(data_size, data_1, 1, data_2, 1, &result);
#else
    result = std::inner_product(
        data_1, data_1 + data_size, data_2, std::complex<T>(), ConstSum<T>,
        static_cast<std::complex<T> (*)(std::complex<T>, std::complex<T>)>(
            &ConstMult<T>));
#endif
    return result;
}

/**
 * @brief Calculates the inner-product using the best available method with the
 * first dataset conjugated.
 *
 * @tparam T Floating point precision type.
 * @param data_1 Complex data array 1; conjugated before application.
 * @param data_2 Complex data array 2.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T>
std::complex<T> innerProdC(const std::complex<T> *data_1,
                           const std::complex<T> *data_2,
                           const size_t data_size) {
    std::complex<T> result(0, 0);

#ifdef USE_CBLAS
    if constexpr (std::is_same_v<T, float>)
        cblas_cdotc_sub(data_size, data_1, 1, data_2, 1, &result);
    else if constexpr (std::is_same_v<T, double>)
        cblas_zdotc_sub(data_size, data_1, 1, data_2, 1, &result);
#else
    result =
        std::inner_product(data_1, data_1 + data_size, data_2,
                           std::complex<T>(), ConstSum<T>, ConstMultConj<T>);
#endif
    return result;
}

template <class T>
inline std::complex<T> innerProd(const std::vector<std::complex<T>> &data_1,
                                 const std::vector<std::complex<T>> &data_2) {
    return innerProd(data_1.data(), data_2.data(), data_1.size());
}

template <class T>
inline std::complex<T> innerProdC(const std::vector<std::complex<T>> &data_1,
                                  const std::vector<std::complex<T>> &data_2) {
    return innerProdC(data_1.data(), data_2.data(), data_1.size());
}

/**
 * @brief Utility method for performing complex matrix-vector multiplication as
 * C[n] = A[m][n] * B[n]. Offloads to BLAS if enabled, otherwise defaults to
 * naive O(n^2) multiplication kernel. Transposes indices of matrix A if flag is
 * set.
 *
 * @tparam T
 * @param mat_left Left matrix, A in row-major format.
 * @param vec_right Right vector, B.
 * @param vec_out Output vector, C.
 * @param m Row-size of matrix A.
 * @param n Column size of matrix A, vector B, and output vector C.
 * @param transpose Transpose indices of matrix A prior to calculating C index
 * value.
 * @return std::complex<T>
 */
template <class T>
void MatVecProd(const std::complex<T> *mat_left,
                const std::complex<T> *vec_right, std::complex<T> *vec_out,
                size_t m, size_t n, bool transpose = false) {
    const auto alpha = ONE<T>();
    const auto beta = ZERO<T>();

#ifdef USE_CBLAS
    const auto tp = (transpose) ? CblasTrans : CblasNoTrans;
    if constexpr (std::is_same_v<T, float>)
        cblas_cgemv(CblasRowMajor, tp, m, n, &alpha, mat_left, std::max(1ul, m),
                    vec_right, 1, &beta, vec_out, 1);
    else if constexpr (std::is_same_v<T, double>)
        cblas_zgemv(CblasRowMajor, tp, m, n, &alpha, mat_left, std::max(1ul, m),
                    vec_right, 1, &beta, vec_out, 1);
#else // If not using BLAS, perform naive multiplication ops;
    std::fill_n(vec_out, n, beta);
    if (!transpose) {
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                vec_out[col] += mat_left[row * n + col] * vec_right[col];
            }
        }
    } else {
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                vec_out[col] += mat_left[col * m + row] * vec_right[col];
            }
        }
    }
#endif
}

template <class T>
std::complex<T> MatVecProd(const std::complex<T> &mat_left,
                           const std::complex<T> *vec_right,
                           std::complex<T> *vec_out, size_t m, size_t n,
                           bool transpose = false);

/**
 * @brief Utility method for performing complex matrix-matrix multiplication as
 * C[m][n] = A[m][k] * B[k][n]. Offloads to BLAS if enabled, otherwise defaults
 * to naive O(n^3) multiplication kernel. Transposes indices of matrix B if flag
 * is set.
 *
 * @tparam T Complex precision. `float` and `double` are supported.
 * @param mat_left Left matrix, A in row-major format. Rows and cols are of size
 * m * k. Indices transposed if requested.
 * @param mat_right Right matrix, B in row-major format. Rows and cols are of
 * size k * n;
 * @param mat_out Output matrix, C in row-major format. Rows and cols are of
 * size m * n.
 * @param m Row-size of matrix A, and output matrix C.
 * @param n Column size of matrix B, and output matrix C.
 * @param k Column size of matrix A, and row size of matrix B.
 * @param transpose Indicates whether indices of matrix B are to be transposed
 * before calculating C index.
 * @return std::complex<T>
 */
template <class T>
void MatMatProd(const std::complex<T> *mat_left,
                const std::complex<T> *mat_right, std::complex<T> *mat_out,
                size_t m, size_t n, size_t k, bool transpose = false) {
    const auto alpha = ONE<T>();
    const auto beta = ZERO<T>();

#ifdef USE_CBLAS
    std::cout << "I GOT HERE!!!" << std::endl;

    const auto tp = (transpose) ? CblasTrans : CblasNoTrans;

    if constexpr (std::is_same_v<T, float>)
        cblas_cgemm(CblasRowMajor, tp, CblasNoTrans, m, n, k, &alpha, mat_left,
                    std::max(1ul, k), mat_right, std::max(1ul, n), &beta,
                    mat_out, std::max(1ul, n));
    else if constexpr (std::is_same_v<T, double>)
        cblas_zgemm(CblasRowMajor, tp, CblasNoTrans, m, n, k, &alpha, mat_left,
                    std::max(1ul, k), mat_right, std::max(1ul, n), &beta,
                    mat_out, std::max(1ul, n));
#else // If not using BLAS, perform naive multiplication ops;
    std::fill_n(mat_out, m * n, beta);
    if (!transpose) {
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                for (size_t inner = 0; inner < k; inner++) {
                    mat_out[row * n + col] +=
                        mat_left[row * k + inner] * mat_right[inner * n + col];
                }
            }
        }
    } else {
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                for (size_t inner = 0; inner < k; inner++) {
                    mat_out[row * n + col] +=
                        mat_left[inner * m + row] * mat_right[inner * n + col];
                }
            }
        }
    }
#endif
}

template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    os << '[';
    for (size_t i = 0; i < vec.size(); i++) {
        os << vec[i] << ",";
    }
    os << ']';
    return os;
}

template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::set<T> &s) {
    os << '{';
    for (const auto &e : s) {
        os << e << ",";
    }
    os << '}';
    return os;
}

template <class T> std::vector<T> linspace(T start, T end, size_t num_points) {
    std::vector<T> data(num_points);
    T step = (end - start) / (num_points - 1);
    for (size_t i = 0; i < num_points; i++) {
        data[i] = start + (step * i);
    }
    return data;
}

/**
 * @brief Exception for functions that are not yet implemented.
 *
 */
class NotImplementedException : public std::logic_error {
  public:
    explicit NotImplementedException(const std::string &fname)
        : std::logic_error(std::string("Function is not implemented. ") +
                           fname){};
};

} // namespace Util
} // namespace Pennylane
