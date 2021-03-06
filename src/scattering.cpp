#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <cubature/cubature.h>
#include <complex_bessel.h>
#include "functions.h"
#include "coordinate.h"
#include "scattering.h"

using namespace std;
using namespace std::complex_literals;
using namespace special;
using namespace Eigen;
using namespace coordinate;

namespace Mie {
/*
 Mie coefficient a
 Absorption and Scattering of Light by Small Particles

 n  : index of vector spherical harmonics 
 k0 : wave number of medium
 k1 : wave number of scatterer
 r  : radius of sphere scatterer
*/
complex<double> coef_an(int n, complex<double> k0, complex<double> k1, double r) {
    auto m = k1 / k0;
    auto x = k0 * r;

    // numerator
    complex<double> num = m * riccati_bessel_zn(1, n, m*x) * riccati_bessel_zn(1, n, x, true) 
                            - riccati_bessel_zn(1, n, x)   * riccati_bessel_zn(1, n, m*x, true);
    // denominator
    complex<double> den = m * riccati_bessel_zn(1, n, m*x) * riccati_bessel_zn(3, n, x, true) 
                            - riccati_bessel_zn(3, n, x)   * riccati_bessel_zn(1, n, m*x, true);
    return num / den;
}

/*
 Mie coefficient b
 Absorption and Scattering of Light by Small Particles

 n  : index of vector spherical harmonics 
 k0 : wave number of medium
 k1 : wave number of scatterer
 r  : radius of sphere scatterer
*/
complex<double> coef_bn(int n, complex<double> k0, complex<double> k1, double r) {
    auto m = k1 / k0;
    auto x = k0 * r;

    // numerator
    complex<double> num =     riccati_bessel_zn(1, n, m*x) * riccati_bessel_zn(1, n, x, true) 
                        - m * riccati_bessel_zn(1, n, x)   * riccati_bessel_zn(1, n, m*x, true);
    // denominator
    complex<double> den =     riccati_bessel_zn(1, n, m*x) * riccati_bessel_zn(3, n, x, true) 
                        - m * riccati_bessel_zn(3, n, x)   * riccati_bessel_zn(1, n, m*x, true);
    return num / den;
}
} // namespace Mie

namespace TMatrix {
bool operator<(const VecSphIndex& n1, const VecSphIndex& n2) {
    return    n1.tau   < n2.tau    //  1  <  2
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

// element of Tmatrix of sphere of radius a
// k0 : wave number of medium
// k1 : wave number of particle
complex<double> T_sph_element(TmatrixIndex n, complex<double> k0, complex<double> k1, double r) {
    if (n.m != 1) return complex<double>{};
    if (n.tau == 1) return - Mie::coef_bn(n.l, k0, k1, r);
    else if (n.tau == 2) return - Mie::coef_an(n.l, k0, k1, r);
    cerr << "T_shp_element(): error: argument n.tau must be 1, or 2" << endl;
    return 0;
}

/*
* integrate n_vec・integrand_vec on Surface
  k0 : wave number of medium
  k1 : wave number of particle
*/

vector<complex<double>> cross(vector<complex<double>>& a, vector<complex<double>>& b) {
    vector<complex<double>> c(3);
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[2] * b[0];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

vector<complex<double>> integrand_vec(Indexes &index, double k0, double k1, double r, double theta, double phi) {
    auto   n1 = index.first;
    auto   n2 = index.second;

    auto M1  = special::vector_spherical_harmonics(n1.p, n1.tau, n1.sigma, n1.l, n1.m, k0, r, theta, phi);
    auto cM1 = special::curl_vector_spherical_harmonics(n1.p, n1.tau, n1.sigma, n1.l, n1.m, k0, r, theta, phi);
    auto M2  = special::vector_spherical_harmonics(n2.p, n2.tau, n2.sigma, n2.l, n2.m, k1, r, theta, phi);
    auto cM2 = special::curl_vector_spherical_harmonics(n2.p, n2.tau, n2.sigma, n2.l, n2.m, k1, r, theta, phi);
    
    auto v1 = cross(cM1, M2);
    auto v2 = cross(M1, cM2);
    vector<complex<double>> res{v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]};
    return res;
}


// surface integrate on sphere
// n_vec = er
// var = {theta, phi}
int integrand_sph(unsigned int ndim, const double *var, void *fdata, unsigned int fdim, double *fval) {
    double k0 = ((Parameters*)fdata)->k0;
    double k1 = ((Parameters*)fdata)->k1;
    double r = ((Parameters*)fdata)->at;
    Indexes id = ((Parameters*)fdata)->indexes;

    auto V = integrand_vec(id, k0, k1, r, var[0], var[1]);

    fval[0] = V[0].real() * r*r * std::sin(var[0]);
    fval[1] = V[1].imag() * r*r * std::sin(var[0]);
    return 0;
}

// surface integrate on rectangle
// n_vec = ex, ey, ez
int integrand_rec(unsigned int ndim, const double *var, void *fdata, unsigned int fdim, double *fval) {
    double  k0 = ((Parameters*)fdata)->k0;
    double  k1 = ((Parameters*)fdata)->k1;
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
    
    auto V = integrand_vec(id, k0, k1, r, theta, phi);

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

complex<double> intSphere(Indexes indexes, double k0, double k1, double r) {
    constexpr unsigned   fdim        = 2; // 返り値の次元
    constexpr unsigned   vardim      = 2; // 変数の数 2 = {theta, phi}
    constexpr size_t     maxEval     = 0;
    constexpr double     reqAbsError = 1e-4;
    constexpr double     reqRelError = 1e-4;
    constexpr error_norm norm        = ERROR_INDIVIDUAL;
    constexpr double varmin[] = {0,    0};
    constexpr double varmax[] = {M_PI, 2*M_PI};
    double val[fdim], err[fdim];

    Parameters prm{indexes, k0, k1, r};
    
    hcubature(fdim, integrand_sph, &prm, 
              vardim, varmin, varmax, 
              maxEval, reqAbsError, reqRelError, norm, 
              val, err);
    return complex<double>{val[0],val[1]};
}

complex<double> intRectangular(Indexes indexes, double k0, double k1, double wx, double wy, double wz) {
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
                    Parameters prm{indexes, k0, k1, at, axes};
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

namespace DDA {
// Descrete dipole approximation
// Physical Review B 82, 045404 (2010)

double hypot3(double x, double y, double z) { 
    return sqrt(x*x + y*y + z*z);
}

complex<double> exp(complex<double> z) {
    return cos(real(z)) + 1i * sin(imag(z));
}

vector<vector<complex<double>>> G(double k, double x, double y, double z) {
    vector<vector<complex<double>>> res{{x*x, x*y, x*z},
                                        {y*x, y*y, y*z},
                                        {z*x, z*y, z*z}};
    auto R = hypot3(x, y, z);
    auto term1 = 1/R + 1/(k*R*R) - 1/(k*k*R*R*R);
    auto term2 = -1/R - 3i/(k*R*R) + 3/(k*k*R*R*R);
    auto term3 = exp(1i*k*R/(4*M_PI));
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        res[i][j] = res[i][j] * term2 / (R*R);
        if (i == j) res[i][j] += term1;
        res[i][j] *= term3;
    }
    return res;
}

complex<double> aE(double k, complex<double> a1, double eps0) {
    return 6.0i*M_PI*eps0*a1/(k*k*k);
}

complex<double> aM(double k, complex<double> b1) {
    return 6.0i*M_PI*b1/(k*k*k);
}
} // namespace DDA