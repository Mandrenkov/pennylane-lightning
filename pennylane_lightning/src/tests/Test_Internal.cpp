#include "CreateAllWires.hpp"
#include "GateImplementationsPI.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <random>

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/**
 * We test internal functions for test suite.
 */

using namespace Pennylane;
using Pennylane::Gates::GateImplementationsPI;

TEMPLATE_TEST_CASE("Approx", "[Test_Internal]", float, double) {
    using PrecisionT = TestType;
    using ComplexPrecisionT = std::complex<PrecisionT>;

    const auto margin = PrecisionT{0.00015};

    SECTION("vector{1.0, 1.0*I} approx vector{1.0001, 0.9999*I} with margin "
            "0.00015") {
        const std::vector<ComplexPrecisionT> test1{
            ComplexPrecisionT{1.0, 0.0},
            ComplexPrecisionT{0.0, 1.0},
        };
        const std::vector<ComplexPrecisionT> test2{
            ComplexPrecisionT{1.0001, 0.0},
            ComplexPrecisionT{0.0, 0.9999},
        };
        REQUIRE(test1 == approx(test2).margin(margin));
    }
    SECTION("vector{1.0, 1.0*I} does not approx vector{1.0002, 0.9998*I} with "
            "margin 0.00015") {
        const std::vector<ComplexPrecisionT> test1{
            ComplexPrecisionT{1.0, 0.0},
            ComplexPrecisionT{0.0, 1.0},
        };
        const std::vector<ComplexPrecisionT> test2{
            ComplexPrecisionT{1.0002, 0.0},
            ComplexPrecisionT{0.0, 0.9998},
        };
        REQUIRE(test1 != approx(test2).margin(margin));
    }
    SECTION("vector{1.0, 1.0*I} does not approx vector{1.0I, 1.0} with margin "
            "0.00015") {
        const std::vector<ComplexPrecisionT> test1{
            ComplexPrecisionT{1.0, 0.0},
            ComplexPrecisionT{0.0, 1.0},
        };
        const std::vector<ComplexPrecisionT> test2{
            ComplexPrecisionT{0.0, 1.0},
            ComplexPrecisionT{1.0, 0.0},
        };
        REQUIRE(test1 != approx(test2).margin(margin));
    }
}

TEMPLATE_TEST_CASE("createProductState", "[Test_Internal]", float, double) {
    using PrecisionT = TestType;

    SECTION("createProductState(\"+-0\") == |+-0> ") {
        const auto st = createProductState<PrecisionT>("+-0");

        auto expected = createZeroState<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);

        REQUIRE(st == approx(expected).margin(PrecisionT{1e-7}));
    }
    SECTION("createProductState(\"+-0\") == |+-1> ") {
        const auto st = createProductState<PrecisionT>("+-0");

        auto expected = createZeroState<PrecisionT>(3);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {0}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {1}, false);
        GateImplementationsPI::applyHadamard(expected.data(), 3, {1}, false);

        GateImplementationsPI::applyPauliX(expected.data(), 3, {2}, false);

        REQUIRE(st != approx(expected).margin(PrecisionT{1e-7}));
    }
}

size_t binomialCeff(size_t n, size_t r) {
    size_t num = 1;
    size_t dem = 1;
    for (size_t k = 0; k < r; k++) {
        num *= (n - k);
    }
    for (size_t k = 1; k <= r; k++) {
        dem *= k;
    }
    return num / dem;
}

size_t permSize(size_t n, size_t r) {
    size_t res = 1;
    for (size_t k = 0; k < r; k++) {
        res *= (n - k);
    }
    return res;
}

/**
 * @brief Test create all wires
 */
TEST_CASE("createAllWires", "[Test_Internal]") {
    SECTION("order = false") {
        const std::vector<std::pair<size_t, size_t>> test_pairs{
            {4, 2},  {8, 3},  {12, 1}, {12, 2}, {12, 3},  {12, 4},  {12, 5},
            {12, 6}, {12, 7}, {12, 8}, {12, 9}, {12, 10}, {12, 11}, {12, 12}};

        for (const auto &[n, r] : test_pairs) {
            std::vector<std::set<size_t>> vec;
            auto v = CombinationGenerator(n, r).all_perms();

            REQUIRE(v.size() == binomialCeff(n, r));
            for (const auto &perm : v) {
                REQUIRE(perm.size() == r);
                vec.emplace_back(perm.begin(), perm.end());
            }

            std::sort(v.begin(), v.end(),
                      [](const std::vector<size_t> &v1,
                         const std::vector<size_t> &v2) {
                          return std::lexicographical_compare(
                              v1.begin(), v1.end(), v2.begin(), v2.end());
                      }); // sort lexicographically
            for (size_t i = 0; i < v.size() - 1; i++) {
                REQUIRE(v[i] != v[i + 1]); // all combinations must be different
            }
        }
    }
    SECTION("order = true") {
        const std::vector<std::pair<size_t, size_t>> test_pairs{
            {4, 2}, {8, 3}, {12, 1}, {12, 2}, {12, 3}, {12, 4}, {12, 5}};

        for (const auto &[n, r] : test_pairs) {
            auto v = PermutationGenerator(n, r).all_perms();

            REQUIRE(v.size() == permSize(n, r));
            for (const auto &perm : v) {
                REQUIRE(perm.size() == r);
            }

            std::sort(v.begin(), v.end(),
                      [](const std::vector<size_t> &v1,
                         const std::vector<size_t> &v2) {
                          return std::lexicographical_compare(
                              v1.begin(), v1.end(), v2.begin(), v2.end());
                      }); // sort lexicographically
            for (size_t i = 0; i < v.size() - 1; i++) {
                REQUIRE(v[i] != v[i + 1]); // all permutations must be different
            }
        }
    }
}
