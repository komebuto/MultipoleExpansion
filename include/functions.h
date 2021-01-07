#ifndef __FUNCTIONS_H_
#define __FUNCTIONS_H_

#include <complex>
#include <vector>
#include <utility>
#include <eigen3/Eigen/Eigen>
#include "coordinate.h"

namespace special {
using coordinate::VectorPolar3cd;
using coordinate::VectorPolar3d;
// combine all of spherical Bessel/Hankel functions above
// p = 1 -> jn, 2 -> yn, 3 -> h1n, 4 -> h2n
std::complex<double> spherical_bessel_zn(int p, int n, double x, bool derivative=false);

// Riccati Bessel funcitions
// p = 1 -> x jn(x), 2 -> -x yn(x), 3 -> x h1n(x), 4 -> x h2n(x)
// https://en.wikipedia.org/wiki/Bessel_function#Riccati%E2%80%93Bessel_functions:_Sn,_Cn,_%CE%BEn,_%CE%B6n
std::complex<double> riccati_bessel_zn(int p, int n, double x, bool derivative=false);

// from index to lm
void legendre_lm(const size_t index, int& l, int& m);

Eigen::Vector3cd vector_spherical_harmonics(
                                    int p, 
                                    int tau, char sigma, int l, int m,
                                    double k, double r, double theta, double phi);

Eigen::Vector3cd curl_vector_spherical_harmonics(
                                    int p, 
                                    int tau, char sigma, int l, int m,
                                    double k, double r, double theta, double phi);
} // namespace special

#endif // __FUNCTIONS_H_
