#include <math.h>
#include <complex>
#include <iostream>
#include <cubature/cubature.h>
#include "integrate.h"

using namespace std;
using namespace std::complex_literals;

namespace integrate {
double EPS = 1e-10;
complex<double> sphere(Func f, void* fdata, 
                       double thetamin, double thetamax, 
                       double phimin, double phimax) {

    integrand f_integrand = [&](unsigned ndim, const double *var, void *fdata, 
                                unsigned fdim, double *fval) {
        // var = {theta, phi}
        auto res = f(var[0], var[1], fdata);
        fval[0] = res.real() * std::sin(var[0]);  
        fval[1] = res.imag() * std::sin(var[0]);
        return 0;
    };

    const double varmin[] = {thetamin, phimin}, varmax[] = {thetamax, phimax};
    constexpr int vardim = 2, fdim = 2;
    constexpr size_t maxEval = 0; // 0 = until Error is below reqAbsError or reqRelError
    constexpr double reqAbsError = 1e-4, reqRelError = 1e-4;
    constexpr error_norm norm = ERROR_INDIVIDUAL;
    double val[fdim], err[fdim];

    hcubature(fdim, f_integrand, fdata, 
              vardim, varmin, varmax, 
              maxEval, reqAbsError, reqRelError, norm, 
              val, err); 
    if (val[0] < -EPS) cerr << "val[0] = " << val[0] << endl;
    return val[0] + 1.0i*val[1];
}

complex<double> sphere_v(Func f, void* fdata, double thetamin, double thetamax, double phimin, double phimax) {
    
    integrand_v f_integrand_v = [&](unsigned ndim, size_t npts, const double *var, void *fdata, 
                                    unsigned fdim, double *fval) {
        // var = {theta, phi}
    #pragma omp parallel for
        for (unsigned i = 0; i < npts; ++i) {
            auto res = f(var[i*ndim + 0], var[i*ndim + 1], fdata);
            fval[i*fdim + 0] = res.real() * std::sin(var[i*ndim + 0]);
            fval[i*fdim + 1] = res.imag() * std::sin(var[i*ndim + 0]);
        }
        return 0;
    };

    const double varmin[] = {thetamin, phimin}, varmax[] = {thetamax, phimax};
    constexpr int vardim = 2, fdim = 2;
    constexpr size_t maxEval = 0; // 0 = until Error is below reqAbsError or reqRelError
    constexpr double reqAbsError = 1e-4, reqRelError = 1e-4;
    constexpr error_norm norm = ERROR_INDIVIDUAL;
    double val[fdim], err[fdim];

    hcubature_v(fdim, f_integrand_v, fdata, 
                vardim, varmin, varmax, 
                maxEval, reqAbsError, reqRelError, norm, 
                val, err); 
    return val[0] + 1.0i*val[1];
}
} // namespace integrate