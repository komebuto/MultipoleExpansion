#include <cmath>
#include <complex>
#include <iostream>
#include <exception>
#include <vector>
#include <complex>
#include <map>
#include <eigen3/Eigen/Eigen>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_legendre.h>
#include "coordinate.h"

using namespace std;
using coordinate::VectorPolar3d;
using coordinate::VectorPolar3cd;

namespace special {
constexpr double EPS = 1e-10;

double spherical_bessel_jn(int n, double x, bool derivative) {
    if (derivative) {
        if (n == 0) return -spherical_bessel_jn(1, x, false);
        else        return (n*spherical_bessel_jn(n-1, x, false) - (n+1)*spherical_bessel_jn(n+1, x, false))/(2*n+1);
    }

    if (x < EPS) {
        if (n == 0) return 1.0;
        else        return 0.0;
    }

    return gsl_sf_bessel_Jnu(n + 0.5, x) * sqrt(0.5*M_PI/x);
}

double spherical_bessel_yn(int n, double x, bool derivative) {
    if (derivative) {
        if (n == 0) return -spherical_bessel_yn(1, x, false);
        else        return (n*spherical_bessel_yn(n-1, x, false) - (n+1)*spherical_bessel_yn(n+1, x, false))/(2*n+1);
    }
    return gsl_sf_bessel_Ynu(n + 0.5, x) * sqrt(0.5*M_PI/x);
}

complex<double> spherical_bessel_h1n(int n, double x, bool derivative) {
    return complex<double>{spherical_bessel_jn(n, x, derivative), spherical_bessel_yn(n, x, derivative)};
}

complex<double> spherical_bessel_h2n(int n, double x, bool derivative) {
    return complex<double>{spherical_bessel_jn(n, x, derivative), -spherical_bessel_yn(n, x, derivative)};
}

/*
  Spherical-Bessel Functions

  int p: the kind of spherical bessel fucntion (1 -> jn, 2 -> yn, 3 -> h1n, 4 -> h2n)
*/
complex<double> spherical_bessel_zn(int p, int n, double x, bool derivative) {
    switch (p)
    {
    case 1: return complex<double>{spherical_bessel_jn(n,x,derivative),0};
    case 2: return complex<double>{spherical_bessel_yn(n,x,derivative),0};
    case 3: return spherical_bessel_h1n(n,x,derivative);
    case 4: return spherical_bessel_h2n(n,x,derivative);
    }
    
    cerr << __FILE__ << ":" << __LINE__ << ": spherical_bessel_zn(): ERROR:" 
         << "argument p = " << p << " must be >= 1 and <= 4" << endl;
    throw;
}

// return spherical_bessel_zn(x)/x
complex<double> zn_div(int p, int n, double x) {
    if (n == 0) return complex<double>{0,0};
    return (spherical_bessel_zn(p, n-1, x, false) + spherical_bessel_zn(p, n+1, x, false)) / static_cast<double>(2*n+1);
}

// return 1/x * d(x*zn(x))/dx
complex<double> dzn_div(int p, int n, double x) {
    return zn_div(p, n, x) + spherical_bessel_zn(p, n, x, true);
}

/*
  Riccati-Bessel Functions
*/
complex<double> riccati_bessel_zn(int p, int n, double x, bool derivative) {
    if (derivative) {
        switch (p) {
        case 1: case 3: case 4: // riccati bessel function of first / third/ fourth kind
            return spherical_bessel_zn(p, n, x, false) + x * spherical_bessel_zn(p, n, x, true);
        case 2:
            return - spherical_bessel_zn(p, n, x, false) - x * spherical_bessel_zn(p, n, x, true);
        }
    }
    switch (p) {
    case 1: case 3: case 4:
        return x * spherical_bessel_zn(p, n, x, false);
    case 2:
        return - x * spherical_bessel_zn(p, n, x, false);
    }
}

constexpr long long fact[] = {1, 1, 2, 6, 24, 120, 720, 5040, 
                              40320, 362880, 3628800, 39916800, 
                              479001600, 6227020800, 87178291200, 
                              1307674368000, 20922789888000,
                              355687428096000, 6402373705728000,
                              121645100408832000, 2432902008176640000};
double factor(int k) {
    if (k <= 20) return fact[k];
    else         return k*factor(k-1);
}

double Plm(int l, int m, double x) {
    if (l < abs(m)) return 0;
    else if (m < 0) return (m%2 ? -1.0: +1.0) * factor(l+m) / factor(l-m) * Plm(l, -m, x);
    return gsl_sf_legendre_Plm(l, m, x);
}

// index -> (l, m)
void legendre_lm(const size_t index, int& l, int& m) {
    for (l = 0;; ++l) {
        m = index - l*(l+1)/2;
        if (m < l+1) break;
    }
}

// return d[Plm(std::cos(theta))]/d[theta]
double Plm_deriv_alt(int l, int m, double theta) {
    double cosine = std::cos(theta);
    return 0.5 * (Plm(l, m+1, cosine) - (l + m) * (l - m + 1) * Plm(l, m-1, cosine));
}

// return m * Plm(x) / sqrt(1 - x^2)
double m_Plm_div(int l, int m, double x) {
    return -0.5 * (Plm(l+1, m+1, x) + (l-m+1)*(l-m+2)*Plm(l+1, m-1, x));
}

Eigen::Vector3cd M1e(int p, int l, int m, double k, double r, double theta, double phi) {
    auto vr     = complex<double>{0.0, 0.0};
    auto vtheta = -1.0 * spherical_bessel_zn(p, l, k*r, false) * m_Plm_div(l, m, std::cos(theta)) * std::sin(m*phi);
    auto vphi   = -1.0 * spherical_bessel_zn(p, l, k*r, false) * Plm_deriv_alt(l, m, theta)  * std::cos(m*phi);
    return Eigen::Vector3cd{vr, vtheta, vphi};
}

Eigen::Vector3cd M1o(int p, int l, int m, double k, double r, double theta, double phi) {
    auto vr     = complex<double>{0.0, 0.0};
    auto vtheta =        spherical_bessel_zn(p, l, k*r, false) * m_Plm_div(l, m, std::cos(theta)) * std::cos(m*phi);
    auto vphi   = -1.0 * spherical_bessel_zn(p, l, k*r, false) * Plm_deriv_alt(l, m, theta)  * std::sin(m*phi);
    return Eigen::Vector3cd{vr, vtheta, vphi};
}

Eigen::Vector3cd M2e(int p, int l, int m, double k, double r, double theta, double phi) {
    auto vr     = static_cast<double>(l*(l+1)) 
                        * zn_div(p, l, k*r) * Plm(l, m, std::cos(theta))      * std::cos(m*phi);
    auto vtheta =        dzn_div(p, l, k*r) * Plm_deriv_alt(l, m, theta) * std::cos(m*phi);
    auto vphi   = -1.0 * dzn_div(p, l, k*r) * m_Plm_div(l, m, std::cos(theta))* std::sin(m*phi);
    return Eigen::Vector3cd{vr, vtheta, vphi};
}

Eigen::Vector3cd M2o(int p, int l, int m, double k, double r, double theta, double phi) {
    auto vr     = static_cast<double>(l*(l+1)) 
                 * zn_div(p, l, k*r) * Plm(l, m, std::cos(theta))      * std::sin(m*phi);
    auto vtheta = dzn_div(p, l, k*r) * Plm_deriv_alt(l, m, theta) * std::sin(m*phi);
    auto vphi   = dzn_div(p, l, k*r) * m_Plm_div(l, m, std::cos(theta))* std::cos(m*phi);
    return Eigen::Vector3cd{vr, vtheta, vphi};
}

Eigen::Vector3cd vector_spherical_harmonics(int p, 
                                          int tau, char sigma, int l, int m,
                                          double k, double r, double theta, double phi) {
    switch (tau) {
    case 1:
        if (sigma == 'e')      return M1e(p, l, m, k, r, theta, phi);
        else if (sigma == 'o') return M1o(p, l, m, k, r, theta, phi);
        else {
            cerr << __FILE__ << ":" << __LINE__ << ": vector_spherical_harmonics(): ERROR:" 
                 << "argument sigma = " << sigma << " must be either 'e' or 'o'" << endl;
            throw;
        }
    case 2:
        if (sigma == 'e')      return M2e(p, l, m, k, r, theta, phi); 
        else if (sigma == 'o') return M2o(p, l, m, k, r, theta, phi); 
        else {
            cerr << __FILE__ << ":" << __LINE__ << ": vector_spherical_harmonics(): ERROR:" 
                 << "argument sigma = " << sigma << " must be either 'e' or 'o'" << endl;
            throw;
        }
    default:
        cerr << __FILE__ << ":" << __LINE__ << ": vector_spherical_harmonics(): ERROR:" 
             << "argument tau = " << tau << " must be either 1 or 2" << endl;
        throw;  
    }
}

Eigen::Vector3cd curl_vector_spherical_harmonics(int p, 
                                               int tau, char sigma, int l, int m,
                                               double k, double r, double theta, double phi) {
    return k * vector_spherical_harmonics(p, 3-tau, sigma, l, m, k, r, theta, phi);
}

} // namespace special
