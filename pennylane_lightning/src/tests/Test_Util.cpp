#include <algorithm>
#include <complex>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Gates.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

using namespace Pennylane;

/**
 * @brief This tests the compile-time calculation of a given scalar
 * multiplication.
 */
TEMPLATE_TEST_CASE("Util::ConstMult", "[Util]", float, double) {
    constexpr TestType r_val = 0.679;
    constexpr std::complex<TestType> c0_val{1.321, -0.175};
    constexpr std::complex<TestType> c1_val{0.579, 1.334};

    SECTION("Real times Complex") {
        constexpr std::complex<TestType> result =
            Util::ConstMult(r_val, c0_val);
        const std::complex<TestType> expected = r_val * c0_val;
        CHECK(isApproxEqual(result, expected));
    }
    SECTION("Complex times Complex") {
        constexpr std::complex<TestType> result =
            Util::ConstMult(c0_val, c1_val);
        const std::complex<TestType> expected = c0_val * c1_val;
        CHECK(isApproxEqual(result, expected));
    }
}

TEMPLATE_TEST_CASE("Constant values", "[Util]", float, double) {
    SECTION("One") {
        CHECK(Util::ONE<TestType>() == std::complex<TestType>{1, 0});
    }
    SECTION("Zero") {
        CHECK(Util::ZERO<TestType>() == std::complex<TestType>{0, 0});
    }
    SECTION("Imag") {
        CHECK(Util::IMAG<TestType>() == std::complex<TestType>{0, 1});
    }
    SECTION("Sqrt2") {
        CHECK(Util::SQRT2<TestType>() == std::sqrt(static_cast<TestType>(2)));
    }
    SECTION("Inverse Sqrt2") {
        CHECK(Util::INVSQRT2<TestType>() ==
              static_cast<TestType>(1 / std::sqrt(2)));
    }
}

TEMPLATE_TEST_CASE("Utility math functions", "[Util]", float, double) {
    SECTION("exp2: 2^n") {
        for (size_t i = 0; i < 10; i++) {
            CHECK(Util::exp2(i) == static_cast<size_t>(std::pow(2, i)));
        }
    }
    SECTION("maxDecimalForQubit") {
        for (size_t num_qubits = 0; num_qubits < 4; num_qubits++) {
            for (size_t index = 0; index < num_qubits; index++) {
                CHECK(Util::maxDecimalForQubit(index, num_qubits) ==
                      static_cast<size_t>(std::pow(2, num_qubits - index - 1)));
            }
        }
    }
    SECTION("dimSize") {
        using namespace Catch::Matchers;
        for (size_t i = 0; i < 64; i++) {
            std::vector<size_t> data(i);
            TestType rem;
            TestType f2 = std::modf(sqrt(i), &rem);
            if (i < 4) {
                CHECK_THROWS_AS(Util::dimSize(data), std::invalid_argument);
                CHECK_THROWS_WITH(
                    Util::dimSize(data),
                    Contains("The dataset must be at least 2x2."));
            } else if (rem != 0.0 && i >= 4 && (i & (i - 1))) {
                CHECK_THROWS_AS(Util::dimSize(data), std::invalid_argument);
                CHECK_THROWS_WITH(Util::dimSize(data),
                                  Contains("The dataset must be a power of 2"));
            } else if (std::sqrt(i) * std::sqrt(i) != i) {
                CHECK_THROWS_AS(Util::dimSize(data), std::invalid_argument);
                CHECK_THROWS_WITH(
                    Util::dimSize(data),
                    Contains("The dataset must be a perfect square"));
            } else {
                CHECK(Util::dimSize(data) == std::log2(std::sqrt(i)));
            }
        }
    }
    SECTION("innerProd") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                std::vector<std::complex<double>> data1(1UL << i, {1, 1});
                std::vector<std::complex<double>> data2(1UL << i, {1, 1});
                std::complex<double> expected_result(0, 1UL << (i + 1));
                std::complex<double> result = Util::innerProd(data1, data2);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<double>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::vector<std::complex<double>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::complex<double> expected_result(0, -0.312985152368);
            std::complex<double> result = Util::innerProd(data1, data2);
            CHECK(isApproxEqual(result, expected_result));
        }
    }
    SECTION("innerProdC") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                std::vector<std::complex<double>> data1(1UL << i, {1, 1});
                std::vector<std::complex<double>> data2(1UL << i, {1, 1});
                std::complex<double> expected_result(1UL << (i + 1), 0);
                std::complex<double> result = Util::innerProdC(data1, data2);
                CAPTURE(result);
                CAPTURE(expected_result);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<double>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::vector<std::complex<double>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::complex<double> expected_result(0, -4.40916e-7);
            std::complex<double> result = Util::innerProdC(data1, data2);
            CAPTURE(result);
            CAPTURE(expected_result);
            CHECK(isApproxEqual(result, expected_result, 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("Matrix & vector operations", "[Util1]", float, double) {
    using namespace Pennylane::Gates;
    const auto X = getPauliX<TestType>();
    const auto Y = getPauliY<TestType>();
    const auto Z = getPauliZ<TestType>();
    const std::vector<std::complex<TestType>> Zero2x2{
        {0, 0}, {0, 0}, {0, 0}, {0, 0}};

    SECTION("Matrix-Matrix product") {
        SECTION("Pauli commutation") {
            SECTION("[X,X]=0") {
                auto out = commutator<TestType>(X, X);
                CHECK(isApproxEqual(out, Zero2x2));
            }
            SECTION("[Y,Y]=0") {
                auto out = commutator<TestType>(Y, Y);
                CHECK(isApproxEqual(out, Zero2x2));
            }
            SECTION("[Z,Z]=0") {
                auto out = commutator<TestType>(Z, Z);
                CHECK(isApproxEqual(out, Zero2x2));
            }
            SECTION("[X,Y]=2iZ") {
                auto out = commutator<TestType>(X, Y);
                std::vector<std::complex<TestType>> expected{
                    {0, 2}, {0, 0}, {0, 0}, {0, -2}};
                CHECK(isApproxEqual(out, expected));
            }
            SECTION("[Y,Z]=2iX") {
                auto out = commutator<TestType>(Y, Z);
                std::vector<std::complex<TestType>> expected{
                    {0, 0}, {0, 2}, {0, 2}, {0, 0}};
                CHECK(isApproxEqual(out, expected));
            }
            SECTION("[Z,X]=2iY") {
                auto out = commutator<TestType>(Z, X);
                std::vector<std::complex<TestType>> expected{
                    {0, 0}, {2, 0}, {-2, 0}, {0, 0}};
                CHECK(isApproxEqual(out, expected));
            }
        }
        SECTION("Random 3x4 and 4x2") {
            const std::vector<std::complex<TestType>> A3x4{
                {0.0476912, 0.0336896}, {0.785569, 0.868045},
                {0.267464, 0.656553},   {0.966149, 0.00257674},
                {0.581249, 0.613336},   {0.467911, 0.15134},
                {0.718083, 0.537794},   {0.371861, 0.441774},
                {0.546396, 0.352677},   {0.877694, 0.73446},
                {0.164514, 0.732967},   {0.259599, 0.365028}};
            const std::vector<std::complex<TestType>> B4x2{
                {0.53859, 0.343218},  {0.536729, 0.0228276},
                {0.338171, 0.713998}, {0.240874, 0.369791},
                {0.110366, 0.73901},  {0.763877, 0.106989},
                {0.563794, 0.736961}, {0.053594, 0.895969}};
            std::vector<std::complex<TestType>> output(3 * 2);
            std::vector<std::complex<TestType>> expected(
                {{-0.252873, 1.87254},
                 {0.0765927, 1.91468},
                 {-0.281376, 2.02824},
                 {
                     0.469818,
                     1.39643,
                 },
                 {-0.700518, 1.85211},
                 {-0.0408593, 1.53289}});
            MatMatProd(A3x4.data(), B4x2.data(), output.data(), 3, 2, 4);
            CHECK(isApproxEqual(output, expected, 1e-5));
        }
        SECTION("Random 4x4 and 4x2") {
            const std::vector<std::complex<TestType>> A4x4{
                {0.844266099318979, 0.6435217821896528},
                {0.7245510554214545, 0.5398643297200929},
                {0.8515969820553153, 0.017860598475019085},
                {0.4285185717017146, 0.3579994337901722},
                {0.9056574960862331, 0.36204890986255567},
                {0.17136617271550825, 0.5043975156015124},
                {0.5342444768255232, 0.8406079240126088},
                {0.644076345832169, 0.17921178292317608},
                {0.5750392704576959, 0.9680758671245733},
                {0.11834501666908315, 0.5720110030647187},
                {0.5744255941594201, 0.7225813717016034},
                {0.9499531206577758, 0.4701123797122495},
                {0.034787257667178606, 0.13477560677380418},
                {0.23675294007412684, 0.9306514931007688},
                {0.9506086357518193, 0.7060265560209127},
                {0.7653010831637865, 0.48712247844599244}};
            const std::vector<std::complex<TestType>> B4x2{
                {0.7933644759827247, 0.7642858531389018},
                {0.8140302824169507, 0.0620299318947215},
                {0.41922872577802606, 0.03634713887254426},
                {0.5534947153220708, 0.5252953112921657},
                {0.7084185813792376, 0.11697741795763017},
                {0.016507953181337243, 0.27874726572431907},
                {0.24928717144721446, 0.17366959745365107},
                {0.014182860071624948, 0.44981381963403044}};
            std::vector<std::complex<TestType>> output(4 * 2);
            std::vector<std::complex<TestType>> outputT(2 * 4);
            std::vector<std::complex<TestType>> expec_AB{
                {1.1079546334615138, 1.6844056603915414},
                {0.6189115942832295, 1.6911375160015765},
                {0.9048887426984651, 2.011633044177458},
                {0.2476924096352795, 1.1751479357868748},
                {0.22272666675304664, 2.312892183695239},
                {-0.21684664078285298, 1.8085013509366943},
                {0.6870400344127316, 1.3979797099102185},
                {-0.727238238781853, 1.379131573721164}};
            std::vector<std::complex<TestType>> expec_ATB{
                {0.8238854138149615, 2.1332161798033766},
                {0.6379482748188227, 1.4461766495992965},
                {0.1300496534968253, 1.8919430843339797},
                {-0.1865435644378397, 1.0157340966123198},
                {1.2921603805763313, 1.9570388126738312},
                {0.05021879012128949, 1.4229300136416951},
                {1.054012266984468, 1.4085780993397},
                {0.26535313455923293, 1.3792356388042593}};

            MatMatProd(A4x4.data(), B4x2.data(), output.data(), 4, 2, 4);
            MatMatProd(A4x4.data(), B4x2.data(), outputT.data(), 4, 2, 4, true);
            CHECK(isApproxEqual(output, expec_AB));
            CHECK(isApproxEqual(outputT, expec_ATB));
        }
    }
    SECTION("Matrix-Vector product") {}
}