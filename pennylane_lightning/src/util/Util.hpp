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
 * @file Util.hpp
 * Contains uncategorised utility functions.
 */
#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace Pennylane::Util {
/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(U a, std::complex<T> b)
    -> std::complex<T> {
    return {a * b.real(), a * b.imag()};
}

/**
 * @brief Compile-time scalar complex times complex.
 *
 * @tparam U Precision of complex value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
template <class T, class U = T>
inline static constexpr auto ConstMultConj(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() + a.imag() * b.imag(),
            -a.imag() * b.real() + a.real() * b.imag()};
}

/**
 * @brief Compile-time scalar complex summation.
 *
 * @tparam T Precision of complex value `a` and result.
 * @tparam U Precision of complex value `b`.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstSum(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return a + b;
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class T> inline static constexpr auto ONE() -> std::complex<T> {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class T> inline static constexpr auto ZERO() -> std::complex<T> {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class T> inline static constexpr auto IMAG() -> std::complex<T> {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class T> inline static constexpr auto SQRT2() -> T {
    if constexpr (std::is_same_v<T, float>) {
        return 0x1.6a09e6p+0F; // NOLINT: To be replaced in C++20
    } else {
        return 0x1.6a09e667f3bcdp+0; // NOLINT: To be replaced in C++20
    }
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class T> inline static constexpr auto INVSQRT2() -> T {
    return {1 / SQRT2<T>()};
}

/**
 * @brief Calculates 2^n for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n
 */
inline auto exp2(const size_t &n) -> size_t {
    return static_cast<size_t>(1) << n;
}

/**
 * @brief Log2 calculation.
 *
 * @param value Value to calculate for.
 * @return size_t
 */
inline auto log2(size_t value) -> size_t {
    return static_cast<size_t>(std::log2(value));
}

/**
 * @brief Calculates the decimal value for a qubit, assuming a big-endian
 * convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline auto maxDecimalForQubit(size_t qubitIndex, size_t qubits) -> size_t {
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
template <class T> inline auto dimSize(const std::vector<T> &data) -> size_t {
    const size_t s = data.size();
    const auto s_sqrt = static_cast<size_t>(std::floor(std::sqrt(s)));

    if (s < 4) {
        throw std::invalid_argument("The dataset must be at least 2x2");
    }
    if (((s == 0) || (s & (s - 1)))) {
        throw std::invalid_argument("The dataset must be a power of 2");
    }
    if (s_sqrt * s_sqrt != s) {
        throw std::invalid_argument("The dataset must be a perfect square");
    }

    return static_cast<size_t>(log2(s_sqrt));
}

/**
 * @brief Streaming operator for vector data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param vec Vector data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::vector<T> &vec)
    -> std::ostream & {
    os << '[';
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            os << vec[i] << ", ";
        }
        os << vec.back();
    }
    os << ']';
    return os;
}

/**
 * @brief Streaming operator for set data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param s Set data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::set<T> &s)
    -> std::ostream & {
    os << '{';
    for (const auto &e : s) {
        os << e << ",";
    }
    os << '}';
    return os;
}

/**
 * @brief Define linearly spaced data [start, end]
 *
 * @tparam T Data type.
 * @param start Start position.
 * @param end End position.
 * @param num_points Number of data-points in range.
 * @return std::vector<T>
 */
template <class T>
auto linspace(T start, T end, size_t num_points) -> std::vector<T> {
    std::vector<T> data(num_points);
    T step = (end - start) / (num_points - 1);
    for (size_t i = 0; i < num_points; i++) {
        data[i] = start + (step * i);
    }
    return data;
}

/**
 * @brief Determines the indices that would sort an array.
 *
 * @tparam T Vector data type.
 * @param arr Array to be inspected.
 * @param length Size of the array
 * @return a vector with indices that would sort the array.
 */
template <typename T>
inline auto sorting_indices(const T *arr, size_t length)
    -> std::vector<size_t> {
    std::vector<size_t> indices(length);
    iota(indices.begin(), indices.end(), 0);

    // indices will be sorted in accordance to the array provided.
    sort(indices.begin(), indices.end(),
         [&arr](size_t i1, size_t i2) { return arr[i1] < arr[i2]; });

    return indices;
}

/**
 * @brief Determines the indices that would sort a vector.
 *
 * @tparam T Array data type.
 * @param vec Vector to be inspected.
 * @return a vector with indices that would sort the vector.
 */
template <typename T>
inline auto sorting_indices(const std::vector<T> &vec) -> std::vector<size_t> {
    return sorting_indices(vec.data(), vec.size());
}

/**
 * @brief Determines the transposed index of a tensor stored linearly.
 *  This function assumes each axis will have a length of 2 (|0>, |1>).
 *
 * @param ind index after transposition.
 * @param new_axes new axes distribution.
 * @return unsigned int with the new transposed index.
 */
inline auto transposed_state_index(size_t ind,
                                   const std::vector<size_t> &new_axes)
    -> size_t {
    size_t new_index = 0;
    for (size_t axis : new_axes) {
        new_index += (ind % 2) << axis;
        ind /= 2;
    }
    return new_index;
}

/**
 * @brief Template for the transposition of state tensors,
 * axes are assumed to have a length of 2 (|0>, |1>).
 *
 * @tparam T Tensor data type.
 * @param tensor Tensor to be transposed.
 * @param new_axes new axes distribution.
 * @return Transposed Tensor.
 */
template <typename T>
auto transpose_state_tensor(const std::vector<T> &tensor,
                            const std::vector<size_t> &new_axes)
    -> std::vector<T> {
    std::vector<T> transposed_tensor(tensor.size());
    for (size_t ind = 0; ind < tensor.size(); ind++) {
        transposed_tensor[transposed_state_index(ind, new_axes)] = tensor[ind];
    }
    return transposed_tensor;
}

/**
 * @brief Exception for functions that are not yet implemented.
 *
 */
class NotImplementedException : public std::logic_error {
  public:
    /**
     * @brief Construct a NotImplementedException exception object.
     *
     * @param fname Function name to indicate not implemented.
     */
    explicit NotImplementedException(const std::string &fname)
        : std::logic_error(std::string("Function is not implemented. ") +
                           fname){};
};

// Enable until C++20 support is explicitly allowed
template <class T> struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

/**
 * @brief Chunk the data using the requested chunk size.
 *
 * @tparam Container STL container type
 * @tparam T Data-type of STL container
 * @param data Data to chunk
 * @param chunk_size Chunk size to use.
 * @return Container<Container<T>> Container of {Containers of data with sizes
 * chunk_size}
 */
template <template <typename...> class Container, typename T>
auto chunkDataSize(const Container<T> &data, std::size_t chunk_size)
    -> Container<Container<T>> {
    Container<Container<T>> output;
    for (std::size_t chunk = 0; chunk < data.size(); chunk += chunk_size) {
        const auto chunk_end = std::min(data.size(), chunk + chunk_size);
        output.emplace_back(data.begin() + chunk, data.begin() + chunk_end);
    }
    return output;
}

/** Chunk the data into the requested number of chunks */

/**
 * @brief Chunk the data into the requested number of chunks.
 *
 * @tparam Container STL container type
 * @tparam T Data-type of STL container
 * @param data Data to chunk
 * @param num_chunks Chunk size to use.
 * @return Container<Container<T>> Container of num_chunks {Containers of data}
 */
template <template <typename...> class Container, typename T>
auto chunkData(const Container<T> &data, std::size_t num_chunks)
    -> Container<Container<T>> {
    const auto rem = data.size() % num_chunks;
    const auto div = static_cast<std::size_t>(data.size() / num_chunks);
    if (!div) { // Match chunks to available work
        return chunkDataSize(data, 1);
    }
    if (rem) { // We have an uneven split; ensure fair distribution
        auto output =
            chunkDataSize(Container<T>{data.begin(), data.end() - rem}, div);
        auto output_rem =
            chunkDataSize(Container<T>{data.end() - rem, data.end()}, div);
        for (std::size_t idx = 0; idx < output_rem.size(); idx++) {
            output[idx].insert(output[idx].end(), output_rem[idx].begin(),
                               output_rem[idx].end());
        }
        return output;
    }
    return chunkDataSize(data, div);
}

/**
 * @brief For lookup from any array of pair whose first elements are
 * GateOperation.
 *
 * As Util::lookup can be used in constexpr context, this function is redundant
 * (by the standard). But GCC 9 still does not accept Util::lookup in constexpr
 * some cases.
 */
template <auto op, class T, size_t size>
constexpr auto
static_lookup(const std::array<std::pair<decltype(op), T>, size> &arr) -> T {
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<0>(arr[idx]) == op) {
            return std::get<1>(arr[idx]);
        }
    }
    return T{};
}

// type alias
template <class T> using remove_cvref_t = typename remove_cvref<T>::type;

template <typename T> struct remove_complex { using type = T; };
template <typename T> struct remove_complex<std::complex<T>> {
    using type = T;
};
template <typename T> using remove_complex_t = typename remove_complex<T>::type;

template <typename T> struct is_complex : std::false_type {};

template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

} // namespace Pennylane::Util
