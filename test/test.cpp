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
#include <sstream>

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
    
    auto res = M1[0]*M2[0] + M1[1]*M2[1] + M1[2]*M2[2];

    if (n1 == n2 && res.real() < -EPS) {
        cout << "res = " << res << endl;
        cout << "M[0]*M[0] = " << M1[0]*M2[0] << endl;
    }
    return res;
}

int fact(int nupper, int nlower) {
    int res = 1;
    for (int i = nlower; i <= nupper; ++i) {
        if (i == 0) continue;
        res *= i;
    }
    return res;
}

complex<double> pow2(complex<double> k) {
    return k * k;
}

complex<double> M1dotM2_correct(const Indexes& idx, double k, double r) {
    auto n1 = idx.first, n2 = idx.second;
    if (n1 != n2) return 0;
    int p1 = n1.p, p2 = n2.p, tau = n1.tau, l = n1.l, m = n1.m;
    double same = (1 + (m==0))*2*M_PI/(2*l+1)*fact(l+m,l-m)*l*(l+1);
    if (tau == 1) {
        return same * spherical_bessel_zn(p1, l, k, r) 
                    * spherical_bessel_zn(p2, l, k, r);
    } 
    else if (tau == 2) {
        double ld = static_cast<double>(l);
        return same / (2*ld + 1) * ( 
            (ld + 1) * spherical_bessel_zn(p1,l-1,k,r) * spherical_bessel_zn(p2,l-1,k,r)
          + ld * spherical_bessel_zn(p1,l+1,k,r) * spherical_bessel_zn(p2,l+1,k,r)
        );
    }
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
        auto correct = M1dotM2_correct(indexes, k, r); 
        if (abs(val - correct) > 1e-4) {
            print_indexes(prm.indexes);
            cerr << "wrong! value: " << val;
            cerr << ", correct: " << correct << endl;
        }
        //if (abs(v.real() - valtmp[0]) > 1e-5) cout << "different: " << valtmp[0] << ", " << v << endl;
        /*if (!(prm.indexes.first == prm.indexes.second) && abs(val) > 1e-4) {
            print_indexes(prm.indexes);
            cerr << "wrong! value: " << val << endl;
        }*/
        
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
    return make_pair(cntall, cntfinite);
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

void dda() {
    constexpr double lda = 1053E-9;
    constexpr double kair = 2*M_PI/lda;    
    constexpr double ksi  = 3.5581 * kair;

    constexpr int nx = 10000, ny = 10000;
    constexpr double dx = 350E-9, dy=350E-9;
    constexpr double SL = dx * dy;

    constexpr double rmin = 10E-9, rmax = 350E-9, dr = 10E-9; 
    vector<vector<complex<double>>> G_(3,vector<complex<double>>(3,0));
    #pragma omp parallel for
    for (int xi = -nx; xi <= nx; ++xi) 
    for (int yi = -ny; yi <= ny; ++yi) {
        if (xi == 0 && yi == 0) continue;
        double x = xi*dx, y = yi*dy;
        auto tmp = DDA::G(kair, x, y, 0);
        for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            G_[i][j] = G_[i][j] + tmp[i][j];
        }
    }

    for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
        cout << G_[i][j] << ", ";
    }
    cout << endl;
    }

    vector<double> rs;
    vector<double> t2;

    #pragma omp parallel for
    for (double r = rmin; r <= rmax; r += dr) {
        auto a1 = Mie::coef_an(1,kair,ksi,r);
        auto b1 = Mie::coef_bn(1,kair,ksi,r);
        auto ae = DDA::aE(kair, a1);
        auto am = DDA::aM(kair, b1);
        auto aEeff = 1.0/(DDA::EPS0/ae - kair*kair*G_[0][0]);
        auto aMeff = 1.0/(1.0/am - kair*kair*G_[1][1]);

        auto t = 1.0 + 1i*kair/(2*SL)*(aEeff + aMeff);
        rs.push_back(r);
        t2.push_back(
            real(t)*real(t) + imag(t)*imag(t)
        );
    }

    plt::plot(rs,t2);
    plt::save("../test/out/dda_t.png");
}

int main() {
    dda();
    /*
    int n = 1;
    cout << sph_besselJ(n, 1.0+1.0i) << endl;
    double minx = 1, maxx = 40;
    vector<double> Xs;
    vector<double> Qs;
    int lmax = 100;
    for (double x = minx; x <= maxx; x += 0.1) {
        xs.push_back(x);
        Qs.push_back(test::Qext(x, lmax));
    }
    plt::scatter(xs, Qs);
    plt::save("../test/out/Qext.png");
*/

/*    
    cout << "lmax: ";
    cin >> lmax;
    test::checkTsph(lmax, 1, 10, 1);
*/
/*
    auto s = chrono::system_clock::now();
    test::test_vector_spherical(false);
    auto e = chrono::system_clock::now();
    cout << "test_vector_sperical(vectorize=true): "
         << chrono::duration_cast<chrono::milliseconds>(e-s).count() 
         << " msec." << endl;
    */
   /*
    vector<double> rs;
    vector<double> exts;
    vector<complex<double>> transmission, reflection;

    constexpr double nsi = 3.5581, nair = 1.0; // refractive index of Si and air
    constexpr double lda = 1053; // wavelength in [nm]
    constexpr double ksi = 2*M_PI*nsi/lda, kair = 2*M_PI*nair/lda;
    constexpr double minr = 10, maxr = 2000, dr = 1; // radius of spherical scatterer in [nm]
    
    for (double r = minr; r <= maxr; r += dr) {
        rs.push_back(r);
        double ext = 0;
        for (int n = 1; n <= 100; ++n) {
            auto an = Mie::coef_an(n, kair, ksi, r);
            auto bn = Mie::coef_bn(n, kair, ksi, r);
            ext += 2*M_PI/(kair*kair) * (2*n + 1) * real(an + bn);
        }
        exts.push_back(ext/(r*r*M_PI));
    }

    plt::plot(rs, exts);
    plt::save("../test/out/extinctions.png");
    plt::close();
*/

    
    /*
    double dtheta = 0.01*M_PI, dphi = 0.01*M_PI;
    int lmax = 3;
    double k = 1., r = 10;
    vector<Vector3cd> Ms;
    for (int id = 0; id < test::count_lm(lmax) * 4; ++id){
        vector<vector<double>> xs, ys, zs, Rs;
        int p, tau, l, m; char sigma;
        for (double theta = 0; theta <= M_PI; theta += dtheta) {
            vector<double> xrow, yrow, zrow, Rrow;
            for (double phi   = 0; phi   <= 2*M_PI; phi   += dphi) {
                test::id2prm(id, lmax, p, tau, sigma, l, m);

                auto M = vector_spherical_harmonics(p, tau, sigma, l, m, 
                                                    k, r, theta, phi);
                auto R = sqrt(abs(M[0])*abs(M[0]) + abs(M[1])*abs(M[1]) + abs(M[2])*abs(M[2]));
                double x = R * sin(theta) * cos(phi);
                double y = R * sin(theta) * sin(phi);
                double z = R * cos(theta);
                xrow.push_back(x);
                yrow.push_back(y);
                zrow.push_back(z);
                Rrow.push_back(R);
            }
            xs.push_back(xrow);
            ys.push_back(yrow);
            zs.push_back(zrow);
            Rs.push_back(Rrow);
        }
           
        ostringstream text;
                      text << "phis_"
                           << "p_" << p << "_" 
                           << "tau_" << tau << "_" 
                           << "sigma_" << sigma << "_" 
                           << "l_" << l << "_"
                           << "m_" << m;
        
        plt::plot_surface(xs, ys, zs);
        plt::subplots_adjust();
        plt::title(text.str());
        plt::save("../test/out/" + text.str());
    }
*/
/*
    cout << "lmax: ";
    cin  >> lmax;
    auto cnt = test::writeoutQ(lmax);
    cout << "\ncntall: " << cnt.first << ", cntfinite: " << cnt.second << endl;
    cout << "cnt_diagnol: " << 8*test::count_lm(lmax) << endl;
*/    
}