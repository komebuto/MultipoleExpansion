#include <math.h>
#include <cubature/cubature.h>

//typedef int (*integrand)(unsigned int ndim, const double *x, void *, unsigned int fdim, double *fval)

using function3 = double(*)(double, double, double);



int intSphere(integrand func, 
              double r, double thetamin, double thetamax, double phimin, double phimax,
              double &val, double &err) {
    constexpr unsigned   fdim        = 1;
    constexpr unsigned   dim         = 2;
    constexpr size_t     maxEval     = 0;
    constexpr double     reqAbsError = 1e-4;
    constexpr double     reqRelError = 1e-4;
    constexpr error_norm norm        = ERROR_INDIVIDUAL;

    const double xmin[] = {thetamin, phimin};
    const double xmax[] = {thetamax, phimax};

    hcubature(fdim, func, &r, 
              dim, xmin, xmax,
              maxEval, reqAbsError, reqRelError, norm,
              &val, &err);
}