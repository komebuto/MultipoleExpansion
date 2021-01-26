#ifndef __TMATRIX_H_
#define __TMATRIX_H_

#include <complex>
#include "functions.h"

namespace Mie {
std::complex<double> coef_an(int n, std::complex<double> k0, std::complex<double> k1, double r);
std::complex<double> coef_bn(int n, std::complex<double> k0, std::complex<double> k1, double r);
}

namespace TMatrix {
struct TmatrixIndex{int tau; char sigma; int l, m; };
struct VecSphIndex { int p, tau; char sigma; int l, m; };
bool operator<(const VecSphIndex& n1, const VecSphIndex& n2);
bool operator>(const VecSphIndex& n1, const VecSphIndex& n2);
bool operator<=(const VecSphIndex& n1, const VecSphIndex& n2);
bool operator>=(const VecSphIndex& n1, const VecSphIndex& n2);
bool operator==(const VecSphIndex& n1, const VecSphIndex& n2);
bool operator!=(const VecSphIndex& n1, const VecSphIndex& n2);

// p1, tau1, sig1, l1, m1 / p2, tau2, sig2, l2, m2
using Indexes = std::pair<VecSphIndex, VecSphIndex>;
    
struct Parameters {
    Indexes indexes;
    double  k0, k1; // k0 : wavenumber of medium, k1 : wavenumber of particle
    double  at;     // surface integrate at (r0(radius) for sphere, x0/y0/z0 for rectangular)
    enum class Ax {X, Y, Z} ax; // axes verticle to surface integrated (only for rectangular)
};

std::complex<double> intSphere(Indexes indexes, double k0, double k1, double r);
std::complex<double> intRectangular(Indexes indexes, double k0, double k1, 
                                    double wx, double wy, double wz);

std::complex<double> Mie_coef_an(int n, std::complex<double> k0, std::complex<double> k1, double r);
std::complex<double> Mie_coef_bn(int n, std::complex<double> k0, std::complex<double> k1, double r);
std::complex<double> T_sph_element(TmatrixIndex n, std::complex<double> k0, std::complex<double> k1, double r);
}

namespace DDA {
constexpr double EPS0 = 8.8541878128E-12;
std::vector<std::vector<std::complex<double>>> G(double k, double x, double y, double z);
std::complex<double> aE(double k, std::complex<double> a1, double eps0=EPS0);
std::complex<double> aM(double k, std::complex<double> b1);
}

#endif // __TMATRIX_H_