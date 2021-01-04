#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cubature/cubature.h>
#include <cmath>
#include <chrono>
#include <tuple>
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>

#include "functions.h"
#include "coordinate.h"
#include "TMatrix.h"

using namespace std;
using namespace Eigen;
using namespace coordinate;
using namespace special;
using namespace TMatrix;

namespace test {
constexpr int count_lm(int lmax) { return (lmax + 3) * lmax / 2 + 1;}
constexpr int LMMAX = count_lm(10); // assuming lmax is <= 10
constexpr double EPS = 1e-8;

int M1dotM2(unsigned dim, const double *x, void *fdata,
            unsigned fdim, double *fval) {
    VecSphIndex n1 = ((Parameters*)fdata)->indexes.first;
    VecSphIndex n2 = ((Parameters*)fdata)->indexes.second;
    int p     = n1.p;
    int q     = n2.p;
    int tau1  = n1.tau;
    int tau2  = n2.tau;
    char sig1 = n1.sigma;
    char sig2 = n2.sigma;
    int l1    = n1.l;
    int l2    = n2.l;
    int m1    = n1.m;
    int m2    = n2.m;
    double k  = ((Parameters*)fdata)->k;
    double r  = ((Parameters*)fdata)->at;

    double theta = x[0];
    double phi   = x[1];

    auto M1 = special::vector_spherical_harmonics(p, tau1, sig1, l1, m1, k, r, theta, phi);
    auto M2 = special::vector_spherical_harmonics(q, tau2, sig2, l2, m2, k, r, theta, phi);

    complex<double> z = M1.transpose() * M2;
    fval[0] = z.real() * std::sin(theta);
    fval[1] = z.imag() * std::sin(theta);
    return 0;
}

// vectorized version
int M1dotM2_v(unsigned dim, size_t npts, const double *x, void *fdata,
              unsigned fdim, double *fval) {
    VecSphIndex n1 = ((Parameters*)fdata)->indexes.first;
    VecSphIndex n2 = ((Parameters*)fdata)->indexes.second;
    int p     = n1.p;
    int q     = n2.p;
    int tau1  = n1.tau;
    int tau2  = n2.tau;
    char sig1 = n1.sigma;
    char sig2 = n2.sigma;
    int l1    = n1.l;
    int l2    = n2.l;
    int m1    = n1.m;
    int m2    = n2.m;
    double k  = ((Parameters*)fdata)->k;
    double r  = ((Parameters*)fdata)->at;
#pragma omp parallel for
    for (unsigned i = 0; i< npts; ++i) { // evaluate the integrand for npts points
        double theta = x[i*dim + 0];
        double phi   = x[i*dim + 1];
        VectorPolar3d point(r, theta, phi);
        auto M1 = special::vector_spherical_harmonics(p, tau1, sig1, l1, m1, k, r, theta, phi);
        auto M2 = special::vector_spherical_harmonics(q, tau2, sig2, l2, m2, k, r, theta, phi);
        complex<double> z = M1.transpose() * M2;
        fval[i*fdim + 0] = z.real() * std::sin(theta);
        fval[i*fdim + 1] = z.imag() * std::sin(theta);
    }
    return 0;
}

constexpr char sigs[] = {'e', 'o'};

void id2prm(int id, int lmax, int& p, int& tau, char& sig, int& l, int& m) {
    int lmsize = count_lm(lmax);
    int k   = id % lmsize; id /= lmsize;
    legendre_lm(k, l, m);
    sig = sigs[id % 2]; id /= 2;
    tau = id % 2 + 1; id /= 2;
    p   = id % 2 + 1;
}

void print_indexes(const Indexes &ids) {
    cout << ids.first.p     << ", " << ids.second.p     << " / "
         << ids.first.tau   << ", " << ids.second.tau   << " / "
         << ids.first.sigma << ", " << ids.second.sigma << " / "
         << ids.first.l     << ", " << ids.second.l     << " / "
         << ids.first.m     << ", " << ids.second.m     << endl;
}

void test_vector_spherical(bool vectorized = false) {
    constexpr double xmin[] = {0, 0}, xmax[] = {M_PI, 2*M_PI};
    constexpr double k = 1.0, r = 1.0;
    constexpr int    fdim = 2, ndim = 2;
    double val[LMMAX*LMMAX*64][fdim], err[LMMAX*LMMAX*64][fdim];
    int lmax;
    cout << "lmax <= 5: ";
    cin >> lmax;
    int a = count_lm(lmax);
//#pragma omp parallel for
    for (int id1 = 0; id1 < 8*a; ++id1) 
//#pragma omp parallel for
    for (int id2 = 0; id2 < 8*a; ++id2) {
        double valtmp[fdim], errtmp[fdim];
        int p, q, tau1, tau2, l1, l2, m1, m2;
        char sig1, sig2;
        id2prm(id1, lmax, p, tau1, sig1, l1, m1);
        id2prm(id2, lmax, q, tau2, sig2, l2, m2);
        Indexes indexes = make_pair(VecSphIndex{p, tau1, sig1, l1, m1}, VecSphIndex{q, tau2, sig2, l2, m2});
        //print_indexes(indexes);
        Parameters prm {indexes, k, r};
        if (vectorized) hcubature_v(fdim, M1dotM2_v, &prm, ndim, xmin, xmax, 0, 1e-4, 1e-4, ERROR_INDIVIDUAL, valtmp, errtmp);
        else hcubature(fdim, M1dotM2, &prm, ndim, xmin, xmax, 0, 1e-4, 1e-4, ERROR_INDIVIDUAL, valtmp, errtmp);
    
        if (!(prm.indexes.first == prm.indexes.second) && valtmp[0] > 1e-4) {
            print_indexes(prm.indexes);
            cerr << "wrong! value: " << valtmp[0] << endl;
        }
        
    }
}

pair<int, int> writeoutQ(int lmax) {
    constexpr double k  = 1.0;
    constexpr double r0 = 1.0;
    int cntfinite = 0, cntall = 0;
    for (int id1 = 0; id1 < 8*count_lm(lmax); ++id1)
    for (int id2 = 0; id2 < 8*count_lm(lmax); ++id2) {
        ++cntall;
        int p1, p2, tau1, tau2, l1, l2, m1, m2;
        char sig1, sig2;
        id2prm(id1, lmax, p1, tau1, sig1, l1, m1);
        id2prm(id2, lmax, p2, tau2, sig2, l2, m2);
        Indexes idx = make_pair(VecSphIndex{p1,tau1,sig1,l1,m1}, VecSphIndex{p2,tau2,sig2,l2,m2});
        auto Q = intSphere(idx, k, r0);
        if (abs(Q) > EPS) {
            print_indexes(idx);
            cout << Q << endl;
            ++cntfinite;
        }
    }
    return make_pair(cntall, cntfinite);
}

} // namespace test

void print_aspolar(VectorPolar3d p) {
    cout << p;
}

void print_ascart(VectorCartesian3d p) {
    cout << p;
}

int main() {
    auto s = chrono::system_clock::now();
    test::test_vector_spherical(true);
    auto e = chrono::system_clock::now();
    cout << "test_vector_sperical(): "
         << chrono::duration_cast<chrono::milliseconds>(e-s).count() 
         << " msec." << endl;

    cout << "lmax: ";
    int lmax;
    cin  >> lmax;
    auto cnt = test::writeoutQ(lmax);
    cout << "\ncntall: " << cnt.first << ", cntfinite: " << cnt.second << endl;
    cout << "cnt_diagnol: " << 8*test::count_lm(lmax) << endl;
}