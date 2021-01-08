#include <math.h>
#include <complex>
#include <cubature/cubature.h>
#include "integrate.h"

using namespace std;
using namespace std::complex_literals;

namespace integrate {
complex<double> sphere(Func f, void* fdata, double thetamin, double thetamax, double phimin, double phimax) {
    
    auto f_integrand = [&](unsigned ndim, const double *var, void *fdata, unsigned fdim, double *fval) {
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
    return val[0] + 1.0i*val[1];
}
} // namespace integrate