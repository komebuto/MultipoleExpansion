#include <iostream>
#include <cmath>
#include <Eigen/Eigen>
#include <cubature/cubature.h>
#include "functions.h"
#include "coordinate.h"

using namespace std;
using namespace special;
using namespace Eigen;
using namespace coordinate;

struct Indexes {
    pair<VecSphIndex, VecSphIndex> n12;
};

/*
* integrate n_vec・integrand_vec on Surface
*/

Eigen::Vector3cd integrand_vec(Indexes &index, double k, double r, double theta, double phi) {
    auto   n1 = index.n12.first;
    auto   n2 = index.n12.second;

    Eigen::Vector3cd M1, cM1, M2, cM2;
    M1  = special::vector_spherical_harmonics(n1.p, n1.tau, n1.sigma, n1.l, n1.m, k, r, theta, phi);
    cM1 = special::curl_vector_spherical_harmonics(n1.p, n1.tau, n1.sigma, n1.l, n1.m, k, r, theta, phi);
    M1  = special::vector_spherical_harmonics(n2.p, n2.tau, n2.sigma, n2.l, n2.m, k, r, theta, phi);
    cM1 = special::curl_vector_spherical_harmonics(n2.p, n2.tau, n2.sigma, n2.l, n2.m, k, r, theta, phi);
    return cM1.cross(M2) + M1.cross(cM2);
}
// integrand

struct Parameters {
    Indexes indexes;
    double  k;
    double  r;
    double  wx, wy, wz;
};

// ndim = 2, x = {theta, phi}, fdim = 2, f[0]: real, f[1]: imag
int Q_integrand_sph(unsigned int ndim, const double *x, void *fdata, unsigned int fdim, double *fval) {
    double k = ((Parameters*)fdata)->k;
    double r = ((Parameters*)fdata)->r;
    Indexes id = ((Parameters*)fdata)->indexes;
    auto V = integrand_vec(id, k, r, x[0], x[1]);
    fval[0] = V[0].real() * r*r * std::sin(x[0]);
    fval[1] = V[1].imag() * r*r * std::sin(x[0]);
}

int Q_integrand_rec(unsigned int ndim, const double *x, void *fdata, unsigned int fdim, double *fval) {
    double  k  = ((Parameters*)fdata)->k;
    double  wx = ((Parameters*)fdata)->wx;
    double  wy = ((Parameters*)fdata)->wy;
    double  wz = ((Parameters*)fdata)->wz;
    Indexes id = ((Parameters*)fdata)->indexes;
    VectorCartesian3d point{x[0], x[1], x[2]}; // x, y, z
}

/*
//typedef int (*integrand)(unsigned int ndim, const double *x, void *, unsigned int fdim, double *fval)
int r_Qintegrand_vec(unsigned int ndim, const double *x, void *fdata, unsigned int fdim, double *fval) {
    
    auto vec = Qintegrand_vec(param);
    return vr(vec);
}

int intSphere(double r, double &val, double &err) {
    constexpr unsigned   fdim        = 2;
    constexpr unsigned   dim         = 2;
    constexpr size_t     maxEval     = 0;
    constexpr double     reqAbsError = 1e-4;
    constexpr double     reqRelError = 1e-4;
    constexpr error_norm norm        = ERROR_INDIVIDUAL;
    constexpr double xmin[] = {0,    0};
    constexpr double xmax[] = {M_PI, 2*M_PI};

    hcubature(fdim, )
}

int intSphere(double r, double thetamin, double thetamax, double phimin, double phimax,
              double &val, double &err) {

    hcubature(fdim, , &r, 
              dim, xmin, xmax,
              maxEval, reqAbsError, reqRelError, norm,
              &val, &err);
}
*/