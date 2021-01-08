#ifndef __INTEGRATE_H_
#define __INTEGRATE_H_

#include <complex>
#include <math.h>
#include <functional>

namespace integrate {
using Func = std::function<std::complex<double>(double theta, double phi, void *fdata)>;

// integration on sphere
std::complex<double> sphere(Func f, void *fdata=nullptr, 
                            double thetamin=0, double thetamax=M_PI, 
                            double phimin=0, double phimax=2*M_PI);
                            
// vectorized version (using OpenMP)
std::complex<double> sphere_v(Func f, void *fdata=nullptr, 
                              double thetamin=0, double thetamax=M_PI, 
                              double phimin=0, double phimax=2*M_PI);
}

#endif // __INTEGRATE_H_