// Copyright 2020 Xanadu Quantum Technologies Inc.

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
 * \rst
 * Contains tensor representations of supported gates in ``lightning.qubit``.
 * \endrst
 */
#pragma once

#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <map>
#include <string>

#include "typedefs.hpp"
using Eigen::Matrix2cd;
using std::string;
using std::vector;

using mvp_pfunc_Xq = Matrix2cd(*)();
using mvp_pfunc_Xq_one_param = Matrix2cd(*)(const double&);
using mvp_pfunc_Xq_three_params = Matrix2cd(*)(const double&, const double&, const double&);


/**
* Generates the identity gate.
*
* @return the identity tensor
*/
inline Matrix2cd MVPIdentity() {
    Matrix2cd X;
    X << 1, 0, 0, 1;
    return X;
}

/**
* Generates the X gate.
*
* @return the X tensor
*/
inline Matrix2cd MVPX() {
    Matrix2cd X;
    X << 0, 1, 1, 0;
    return X;
}

/**
* Generates the Y gate.
*
* @return the Y tensor
*/
inline Matrix2cd MVPY() {
    Matrix2cd Y;
    Y << 0, NEGATIVE_IMAG, IMAG, 0;
    return Y;
}

/**
* Generates the Z gate.
*
* @return the Z tensor
*/
inline Matrix2cd MVPZ() {
    Matrix2cd Z;
    Z << 1, 0, 0, -1;
    return Z;
}

/**
* Generates the H gate.
*
* @return the H tensor
*/
inline Matrix2cd MVPH() {
    Matrix2cd H;
    H << 1/SQRT_2, 1/SQRT_2, 1/SQRT_2, -1/SQRT_2;
    return H;
}

/**
* Generates the S gate.
*
* @return the S tensor
*/
inline Matrix2cd MVPS() {
    Matrix2cd S;
    S << 1, 0, 0, IMAG;
    return S;
}

/**
* Generates the T gate.
*
* @return the T tensor
*/
inline Matrix2cd MVPT() {
    Matrix2cd T;

    const std::complex<double> exponent(0, M_PI/4);
    T << 1, 0, 0, std::pow(M_E, exponent);
    return T;
}

/**
* Generates the X rotation gate.
*
* @param parameter the rotation angle
* @return the RX tensor
*/
inline Matrix2cd MVPRX(const double& parameter) {
    Matrix2cd RX;

    const std::complex<double> c (std::cos(parameter / 2), 0);
    const std::complex<double> js (0, std::sin(-parameter / 2));

    RX << c, js, js, c;
    return RX;
}

/**
* Generates the Y rotation gate.
*
* @param parameter the rotation angle
* @return the RY tensor
*/
inline Matrix2cd MVPRY(const double& parameter) {
    Matrix2cd RY;

    const double c = std::cos(parameter / 2);
    const double s = std::sin(parameter / 2);

    RY << c, -s, s, c;
    return RY;
}

/**
* Generates the Z rotation gate.
*
* @param parameter the rotation angle
* @return the RZ tensor
*/
inline Matrix2cd MVPRZ(const double& parameter) {
    Matrix2cd RZ;

    const std::complex<double> exponent(0, -parameter/2);
    const std::complex<double> exponent_second(0, parameter/2);
    const std::complex<double> first = std::pow(M_E, exponent);
    const std::complex<double> second = std::pow(M_E, exponent_second);

    RZ << first, 0, 0, second;
    return RZ;
}

/**
* Generates the phase-shift gate.
*
* @param parameter the phase shift
* @return the phase-shift tensor
*/
inline Matrix2cd MVPPhaseShift(const double& parameter) {
    Matrix2cd PhaseShift;

    const std::complex<double> exponent(0, parameter);
    const std::complex<double> shift = std::pow(M_E, exponent);

    PhaseShift << 1, 0, 0, shift;
    return PhaseShift;
}

/**
* Generates the arbitrary single qubit rotation gate.
*
* The rotation is achieved through three separate rotations:
* \f$R(\phi, \theta, \omega)= RZ(\omega)RY(\theta)RZ(\phi)\f$.
*
* @param phi the first rotation angle
* @param theta the second rotation angle
* @param omega the third rotation angle
* @return the rotation tensor
*/
inline Matrix2cd MVPRot(const double& phi, const double& theta, const double& omega) {
    Matrix2cd Rot;

    const std::complex<double> e00(0, (-phi - omega)/2);
    const std::complex<double> e10(0, (-phi + omega)/2);
    const std::complex<double> e01(0, (phi - omega)/2);
    const std::complex<double> e11(0, (phi + omega)/2);

    const std::complex<double> exp00 = std::pow(M_E, e00);
    const std::complex<double> exp10 = std::pow(M_E, e10);
    const std::complex<double> exp01 = std::pow(M_E, e01);
    const std::complex<double> exp11 = std::pow(M_E, e11);

    const double c = std::cos(theta / 2);
    const double s = std::sin(theta / 2);

    Rot << exp00 * c, -exp01 * s, exp10 * s, exp11 * c;

    return Rot;
}


// Defining the operation maps
const std::map<std::string, mvp_pfunc_Xq> MVPOneQubitOps = {
    {"Identity", MVPIdentity},
    {"PauliX", MVPX},
    {"PauliY", MVPY},
    {"PauliZ", MVPZ},
    {"Hadamard", MVPH},
    {"S", MVPS},
    {"T", MVPT}
};

const std::map<std::string, mvp_pfunc_Xq_one_param> MVPOneQubitOpsOneParam = {
    {"RX", MVPRX},
    {"RY", MVPRY},
    {"RZ", MVPRZ},
    {"PhaseShift", MVPPhaseShift}
};

const std::map<std::string, mvp_pfunc_Xq_three_params> MVPOneQubitOpsThreeParams = {
    {"Rot", MVPRot}
};
