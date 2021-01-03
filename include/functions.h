#ifndef __FUNCTIONS_H_
#define __FUNCTIONS_H_

#include <complex>
#include <vector>
#include <Eigen/Eigen>
#include "coordinate.h"

namespace special {
using coordinate::VectorPolar3cd;
using coordinate::VectorPolar3d;
// combine all of spherical Bessel/Hankel functions above
// p = 1 -> jn, 2 -> yn, 3 -> h1n, 4 -> h2n
std::complex<double> zn(int p, int n, double x, bool derivative=false);
/*
VectorPolar3cd vector_spherical_harmonics(int p, 
                                          int tau, char sigma, int l, int m,
                                          double k, VectorPolar3d point);
VectorPolar3cd curl_vector_spherical_harmonics(int p, 
                                               int tau, char sigma, int l, int m,
                                               double k, VectorPolar3d point);
*/

void legendre_lm(const size_t index, int& l, int& m);

Eigen::Vector3cd vector_spherical_harmonics(
                                    int p, 
                                    int tau, char sigma, int l, int m,
                                    double k, double r, double theta, double phi);

Eigen::Vector3cd curl_vector_spherical_harmonics(
                                    int p, 
                                    int tau, char sigma, int l, int m,
                                    double k, double r, double theta, double phi);

struct VecSphIndex { int p, tau; char sigma; int l, m; };
} // namespace special

#endif // __FUNCTIONS_H_
