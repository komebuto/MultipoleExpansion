#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <cubature/cubature.h>
#include <cmath>
#include <chrono>
#include <tuple>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>
#include <matplotlibcpp.h>
#include <functional>
#include <complex_bessel.h>

using namespace std::complex_literals;

#include "functions.h"
#include "coordinate.h"
#include "scattering.h"
#include "integrate.h"

using namespace std;
using namespace Eigen;
using namespace coordinate;
using namespace special;
using namespace TMatrix;
using namespace sp_bessel;
namespace plt = matplotlibcpp;

namespace test {
const string TESTOUTDIR = "../test/out";
constexpr int count_lm(int lmax) { return (lmax + 3) * lmax / 2 + 1;}
constexpr int LMMAX = count_lm(10); // assuming lmax is <= 10
constexpr double EPS = 1e-8;
constexpr char sigs[] = {'e', 'o'};

complex<double> M1dotM2(double theta, double phi, void *fdata) {
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
    double k0 = ((Parameters*)fdata)->k0;
    double r  = ((Parameters*)fdata)->at;

    auto M1 = special::vector_spherical_harmonics(p, tau1, sig1, l1, m1, k0, r, theta, phi);
    auto M2 = special::vector_spherical_harmonics(q, tau2, sig2, l2, m2, k0, r, theta, phi);
    
    return M1.transpose() * M2;
}

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
    constexpr double k = 1.0, r = 1.0;
    int lmax;
    cout << "lmax <= 5: ";
    cin >> lmax;
    int a = count_lm(lmax);
//#pragma omp parallel for
    for (int id1 = 0; id1 < 8*a; ++id1) 
//#pragma omp parallel for
    for (int id2 = 0; id2 < 8*a; ++id2) {
        complex<double> val;
        int p, q, tau1, tau2, l1, l2, m1, m2;
        char sig1, sig2;
        id2prm(id1, lmax, p, tau1, sig1, l1, m1);
        id2prm(id2, lmax, q, tau2, sig2, l2, m2);
        Indexes indexes = make_pair(VecSphIndex{p, tau1, sig1, l1, m1}, VecSphIndex{q, tau2, sig2, l2, m2});
        //print_indexes(indexes);
        Parameters prm {indexes, k, 0, r};
        if (vectorized) val = integrate::sphere_v(M1dotM2, &prm);
        else            val = integrate::sphere(M1dotM2, &prm);
        //if (abs(v.real() - valtmp[0]) > 1e-5) cout << "different: " << valtmp[0] << ", " << v << endl;
        if (!(prm.indexes.first == prm.indexes.second) && abs(val) > 1e-4) {
            print_indexes(prm.indexes);
            cerr << "wrong! value: " << val << endl;
        }
        
    }
}

Eigen::MatrixXcd Qpq(int p, int q, int lmax, double k0, double k1, double r) {
    Eigen::MatrixXcd Q(4*count_lm(lmax), 4*count_lm(lmax));
    for (int id1 = 0; id1 < 4*count_lm(lmax); ++id1)
    for (int id2 = 0; id2 < 4*count_lm(lmax); ++id2) {
        int _, tau1, tau2, l1, l2, m1, m2;
        char sig1, sig2;
        id2prm(id1, lmax, _, tau1, sig1, l1, m1);
        id2prm(id2, lmax, _, tau2, sig2, l2, m2);
        Indexes idx = make_pair(VecSphIndex{p,tau1,sig1,l1,m1}, VecSphIndex{q,tau2,sig2,l2,m2});
        Q(id1, id2) = intSphere(idx, k0, k1, r);
    }
    return Q;
}

pair<int, int> writeoutQ(int lmax, string fname = "Qtmp.csv") {
    std::ofstream file(TESTOUTDIR + "/" + fname);
    constexpr double k  = 1.0;
    constexpr double r0 = 1.0;
    int cntfinite = 0, cntall = 0;
    for (int id1 = 0; id1 < 8*count_lm(lmax); ++id1) {
        for (int id2 = 0; id2 < 8*count_lm(lmax); ++id2) {
            ++cntall;
            int p1, p2, tau1, tau2, l1, l2, m1, m2;
            char sig1, sig2;
            id2prm(id1, lmax, p1, tau1, sig1, l1, m1);
            id2prm(id2, lmax, p2, tau2, sig2, l2, m2);
            Indexes idx = make_pair(VecSphIndex{p1,tau1,sig1,l1,m1}, VecSphIndex{p2,tau2,sig2,l2,m2});
            auto Q = intSphere(idx, k, k, r0);
            file << Q.real() << ",";
            if (abs(Q) > EPS) { ++cntfinite; }
            ++cntall;
        }
        file << "\n";
    }
}

Eigen::MatrixXcd Tsph(int lmax, double k0, double k1, double r) {
    Eigen::MatrixXcd T(4*count_lm(lmax), 4*count_lm(lmax));
    for (int id1 = 0; id1 < 4*count_lm(lmax); ++id1)
    for (int id2 = 0; id2 < 4*count_lm(lmax); ++id2) {
        if (id1 != id2) {
            T(id1, id2) = 0;
            continue;
        }
        int _, tau, l, m;
        char sig;
        id2prm(id1, lmax, _, tau, sig, l, m);
        auto idx = TmatrixIndex{tau, sig, l, m};
        T(id1, id2) = TMatrix::T_sph_element(idx, k0, k1, r);
    }
    return T;
}

void checkTsph(int lmax, double k0, double k1, double r) {
    auto Q11 = Qpq(1, 1, lmax, k0, k1, r);
    auto Q31 = Qpq(3, 1, lmax, k0, k1, r);
    auto   T = Tsph(lmax, k0, k1, r);
    cout << Q11.cols() << ", " << Q31.cols() << ", " << T.cols() << endl;
    cout << Q11.rows() << ", " << Q31.rows() << ", " << T.rows() << endl;
    Eigen::Matrix<complex<double>, Dynamic, Dynamic> Z = T*Q31 + Q11;
    for (int i = 0; i < 4*count_lm(lmax); ++i)
    for (int j = 0; j < 4*count_lm(lmax); ++j) {
        if (abs(Z(i,j)) > EPS) {
            int _, tau1, l1, m1, tau2, l2, m2;
            char sig1, sig2;
            id2prm(i, lmax, _, tau1, sig1, l1, m1);
            id2prm(j, lmax, _, tau2, sig2, l2, m2);
            cout << "tau: " << tau1 << ", " << tau2 << endl;
            cout << "sig: " << sig1 << ", " << sig2 << endl;
            cout << "l  : " << l1   << ", " << l2   << endl;
            cout << "m  : " << m1   << ", " << m2   << endl;
            cout << "\t Z: " << Z(i, j) << endl;
        }
    }
}

double Qext(double x, int lmax) {
    complex<double> res{};
    auto k0  = 1.;
    auto k1 = 1.33 + 1e-8*1i;
    for (int i = 1; i <= lmax; ++i) {
        auto an = Mie::coef_an(i, k0, k1, x/k0);
        auto bn = Mie::coef_bn(i, k0, k1, x/k0);
        res += static_cast<double>(2*i + 1) * (an + bn);
    }
    return res.real()/(x*x);
}
} // namespace test

void print_aspolar(VectorPolar3d p) {
    cout << p;
}

void print_ascart(VectorCartesian3d p) {
    cout << p;
}

int main() {
    int n = 1;
    cout << sph_besselJ(n, 1.0+1.0i) << endl;
    double minx = 1, maxx = 40;
    double dx   = 0.1;
    vector<double> xs;
    vector<double> Qs;
    int lmax = 50;
    for (double x = minx; x <= maxx; x += dx) {
        xs.push_back(x);
        Qs.push_back(test::Qext(x, lmax));
    }
    plt::scatter(xs, Qs);
    plt::save("../test/out/Qext.png");


/*    
    cout << "lmax: ";
    cin >> lmax;
    test::checkTsph(lmax, 1, 10, 1);
*/
    auto s = chrono::system_clock::now();
    test::test_vector_spherical(true);
    auto e = chrono::system_clock::now();
    cout << "test_vector_sperical(vectorize=true): "
         << chrono::duration_cast<chrono::milliseconds>(e-s).count() 
         << " msec." << endl;
    s = chrono::system_clock::now();
    test::test_vector_spherical(false);
    e = chrono::system_clock::now();
    cout << "test_vector_sperical(vectorize=false): "
         << chrono::duration_cast<chrono::milliseconds>(e-s).count() 
         << " msec." << endl;
    
/*
    cout << "lmax: ";
    cin  >> lmax;
    auto cnt = test::writeoutQ(lmax);
    cout << "\ncntall: " << cnt.first << ", cntfinite: " << cnt.second << endl;
    cout << "cnt_diagnol: " << 8*test::count_lm(lmax) << endl;
*/    
}