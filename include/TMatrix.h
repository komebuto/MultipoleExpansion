#ifndef __TMATRIX_H_
#define __TMATRIX_H_

#include <complex>
#include "functions.h"

namespace TMatrix {
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
    double  k;
    double  at; // surface integrate at (r0(radius) for sphere, x0/y0/z0 for rectangular)
    enum class Ax {X, Y, Z} ax; // axes verticle to surface integrated (only for rectangular)
};

std::complex<double> intSphere(Indexes indexes, double k, double r);
std::complex<double> intRectangular(Indexes indexes, double k, 
                                    double wx, double wy, double wz);
}

#endif // __TMATRIX_H_