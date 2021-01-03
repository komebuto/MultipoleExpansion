#ifndef __COORDINATE_H_
#define __COORDINATE_H_

#include <iostream>
#include <complex>
#include <string>
#include <cmath>
#include <Eigen/Eigen>
#include "mytype.h"

namespace coordinate {
template<typename T>
class VectorField3 : public Eigen::Matrix<T, 3, 1> {
public:
    virtual std::string vectype_str();
};

template<typename T>
class VectorPolar3;

template<typename T>
class VectorCartesian3 : public Eigen::Matrix<T, 3, 1> {
    using Base_Vector = Eigen::Matrix<T, 3, 1>;
public:
    using Base_Vector::Matrix;
    VectorCartesian3(VectorPolar3<T>& polar);

    Eigen::Matrix<T, 1, 3> t() { return Eigen::Matrix<T, 1, 3>((*this)[0], (*this)[1], (*this)[2]); }
    std::string vectype_str() {
        std::string res = "Cartesian3" + mytype::type_str<T>();
        return res;
    };
};

template<typename T>
class VectorPolar3 : public Eigen::Matrix<T, 3, 1> {
    using Base_Vector = Eigen::Matrix<T, 3, 1>;
public:
    using Base_Vector::Matrix;
    VectorPolar3(VectorCartesian3<T>& cartesian);

    Eigen::Matrix<T, 1, 3> t() { return Eigen::Matrix<T, 1, 3>((*this)[0], (*this)[1], (*this)[2]); }
    std::string vectype_str() {
        std::string res = "Polar3" + mytype::type_str<T>();
        return res;
    };
};

template<typename T>
VectorCartesian3<T>::VectorCartesian3(VectorPolar3<T>& polar)
    : Base_Vector::Matrix(polar[0] * sin(polar[1]) * cos(polar[2]),
                          polar[0] * sin(polar[1]) * sin(polar[2]),
                          polar[0] * cos(polar[1])) {}

template<> inline
VectorCartesian3<std::complex<double>>::VectorCartesian3(VectorPolar3<std::complex<double>>& polar)
    : Base_Vector::Matrix(
        std::complex<double> (
        polar[0].real() * std::sin(polar[1].real()) * std::cos(polar[2].real()),
        polar[0].imag() * std::sin(polar[1].imag()) * std::cos(polar[2].imag())
        ),
        std::complex<double> (
        polar[0].real() * std::sin(polar[1].real()) * std::sin(polar[2].real()),
        polar[0].imag() * std::sin(polar[1].imag()) * std::sin(polar[2].imag())
        ),
        std::complex<double> (
        polar[0].real() * std::cos(polar[1].real()),
        polar[0].imag() * std::cos(polar[1].imag())
        )
    ) {}

template<typename T>
T hypot3(T x1, T x2, T x3) {
    return sqrt(x1*x1 + x2*x2 + x3*x3);
}

template<typename T>
VectorPolar3<T>::VectorPolar3(VectorCartesian3<T>& cartesian)
    : Base_Vector::Matrix(hypot3(cartesian[0], cartesian[1], cartesian[2]),
                          atan2(hypot(cartesian[0], cartesian[1]), cartesian[2]),
                          atan2(cartesian[1], cartesian[0])) {}

template<typename T>
std::ostream& operator<<(std::ostream& os, VectorCartesian3<T>& vec) {
    os << vec.vectype_str() << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
    return os;
}

template<> inline
std::ostream& operator<<(std::ostream& os, VectorCartesian3<std::complex<double>>& vec) {
    os << vec.vectype_str() << "("  << vec[0].real() << "+" << vec[0].imag() << "i"
                            << ", " << vec[1].real() << "+" << vec[1].imag() << "i"
                            << ", " << vec[2].real() << "+" << vec[2].imag() << "i)";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, VectorCartesian3<T>&& vec) {
    os << vec;
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, VectorPolar3<T>& vec) {
    os << vec.vectype_str() << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
    return os;
}

template<> inline
std::ostream& operator<<(std::ostream& os, VectorPolar3<std::complex<double>>& vec) {
    os << vec.vectype_str() << "("  << vec[0].real() << "+" << vec[0].imag() << "i"
                            << ", " << vec[1].real() << "+" << vec[1].imag() << "i"
                            << ", " << vec[2].real() << "+" << vec[2].imag() << "i)";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, VectorPolar3<T>&& vec) {
    os << vec;
    return os;
}

using VectorCartesian3d  = VectorCartesian3<double>;
using VectorPolar3d      = VectorPolar3<double>;
using VectorCartesian3cd = VectorCartesian3<std::complex<double>>;
using VectorPolar3cd     = VectorPolar3<std::complex<double>>;

} // namespace coordinate

#endif // __COORDINATE_H_