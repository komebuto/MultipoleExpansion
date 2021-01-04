#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <cubature/cubature.h>
#include "functions.h"
#include "coordinate.h"
#include "TMatrix.h"

using namespace std;
using namespace special;
using namespace Eigen;
using namespace coordinate;

namespace TMatrix {
bool operator<(const VecSphIndex& n1, const VecSphIndex& n2) {
    return     n1.p    < n2.p      //  1  <  2
           || n1.tau   < n2.tau    //  1  <  2
           || n1.sigma < n2.sigma  // 'e' < 'o'
           || n1.l     < n2.l 
           || n1.m     < n2.m;
}

bool operator>(const VecSphIndex& n1, const VecSphIndex& n2) {
    return n2 < n1;
}

bool operator<=(const VecSphIndex& n1, const VecSphIndex& n2) {
    return !(n1 > n2);
}

bool operator>=(const VecSphIndex& n1, const VecSphIndex& n2) {
    return !(n1 < n2);
}

bool operator==(const VecSphIndex& n1, const VecSphIndex& n2) {
    return !(n1 < n2) && !(n1 > n2);
}

bool operator!=(const VecSphIndex& n1, const VecSphIndex& n2) {
    return !(n1 == n2);
}

/*
* integrate n_vec・integrand_vec on Surface
*/

Eigen::Vector3cd integrand_vec(Indexes &index, double k, double r, double theta, double phi) {
    auto   n1 = index.first;
    auto   n2 = index.second;

    Eigen::Vector3cd M1, cM1, M2, cM2;
    M1  = special::vector_spherical_harmonics(n1.p, n1.tau, n1.sigma, n1.l, n1.m, k, r, theta, phi);
    cM1 = special::curl_vector_spherical_harmonics(n1.p, n1.tau, n1.sigma, n1.l, n1.m, k, r, theta, phi);
    M2  = special::vector_spherical_harmonics(n2.p, n2.tau, n2.sigma, n2.l, n2.m, k, r, theta, phi);
    cM2 = special::curl_vector_spherical_harmonics(n2.p, n2.tau, n2.sigma, n2.l, n2.m, k, r, theta, phi);
    return cM1.cross(M2) + M1.cross(cM2);
}


// surface integrate on sphere
// n_vec = er
// var = {theta, phi}
int integrand_sph(unsigned int ndim, const double *var, void *fdata, unsigned int fdim, double *fval) {
    double k = ((Parameters*)fdata)->k;
    double r = ((Parameters*)fdata)->at;
    Indexes id = ((Parameters*)fdata)->indexes;

    auto V = integrand_vec(id, k, r, var[0], var[1]);

    fval[0] = V[0].real() * r*r * std::sin(var[0]);
    fval[1] = V[1].imag() * r*r * std::sin(var[0]);
    return 0;
}

// surface integrate on rectangle
// n_vec = ex, ey, ez
int integrand_rec(unsigned int ndim, const double *var, void *fdata, unsigned int fdim, double *fval) {
    double  k  = ((Parameters*)fdata)->k;
    double  x, y, z;
    Indexes id = ((Parameters*)fdata)->indexes;

    switch (((Parameters*)fdata)->ax) {
    case Parameters::Ax::X: // rectangular vertical to x axis / var {y, z}
        x = ((Parameters*)fdata)->at; y = var[0]; z = var[1]; break;
    case Parameters::Ax::Y: // rectangular vertical to y axis / var {z, x}
        x = var[1]; y = ((Parameters*)fdata)->at; z = var[0]; break;
    case Parameters::Ax::Z: // rectangular vertical to z axis / var {x, y}
        x = var[0]; y = var[1]; z = ((Parameters*)fdata)->at; break;        
    }

    double r, theta, phi;
    tie(r, theta, phi) = cart2pol(x, y, z);
    
    auto V = integrand_vec(id, k, r, theta, phi);

    complex<double> vx, vy, vz;
    tie(vx, vy, vz) = pol2cart(V[0], V[1], V[2]);

    switch (((Parameters*)fdata)->ax) {
    case Parameters::Ax::X: // rectangular vertical to x axis / var {y, z}
        fval[0] = vx.real(); fval[1] = vx.imag(); return 0;
    case Parameters::Ax::Y: // rectangular vertical to y axis / var {z, x}
        fval[0] = vy.real(); fval[1] = vy.imag(); return 0;
    case Parameters::Ax::Z: // rectangular vertical to z axis / var {x, y}
        fval[0] = vz.real(); fval[1] = vz.imag(); return 0;
    }
    return 1;
}

complex<double> intSphere(Indexes indexes, double k, double r) {
    constexpr unsigned   fdim        = 2; // 返り値の次元
    constexpr unsigned   vardim      = 2; // 変数の数 2 = {theta, phi}
    constexpr size_t     maxEval     = 0;
    constexpr double     reqAbsError = 1e-4;
    constexpr double     reqRelError = 1e-4;
    constexpr error_norm norm        = ERROR_INDIVIDUAL;
    constexpr double varmin[] = {0,    0};
    constexpr double varmax[] = {M_PI, 2*M_PI};
    double val[fdim], err[fdim];

    Parameters prm{indexes, k, r};
    
    hcubature(fdim, integrand_sph, &prm, 
              vardim, varmin, varmax, 
              maxEval, reqAbsError, reqRelError, norm, 
              val, err);
    return complex<double>{val[0],val[1]};
}

complex<double> intRectangular(Indexes indexes, double k, double wx, double wy, double wz) {
    constexpr unsigned   fdim        = 2;
    constexpr unsigned   vardim      = 2;
    constexpr size_t     maxEval     = 0;
    constexpr double     reqAbsError = 1e-4;
    constexpr double     reqRelError = 1e-4;
    constexpr error_norm norm        = ERROR_INDIVIDUAL;

    double varmin[vardim], varmax[vardim];
    double val[fdim], err[fdim];
    double res[fdim] = {0, 0};

    auto f = [&](double at, Parameters::Ax axes, int sign) {
                    Parameters prm{indexes, k, at, axes};
                    hcubature(fdim, integrand_rec, &prm, 
                              vardim, varmin, varmax, 
                              maxEval, reqAbsError, reqRelError, norm, 
                              val, err);
                    res[0] += sign * val[0];
                    res[1] += sign * val[1];
             };
    
    f(wx/2,  Parameters::Ax::X, +1);
    f(-wx/2, Parameters::Ax::X, -1);
    f(wy/2,  Parameters::Ax::Y, +1);
    f(-wy/2, Parameters::Ax::Y, -1);
    f(wz/2,  Parameters::Ax::Z, +1);
    f(-wz/2, Parameters::Ax::Z, -1);

    return complex<double>{res[0], res[1]};
}
} // namespace TMatrix