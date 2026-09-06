// Exact vs projected apply on an affine HEX8 grid, for a nonlinear material.
//
// This is the case the technique exists for: the exact kernel evaluates the
// material tangent at every one of eight quadrature points, the projected one
// averages it once.  The projection is a genuine approximation here, so the
// comparison reports a relative difference rather than expecting agreement.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include "sfem_base.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "kernel_math.hpp"
#include MATERIAL_INEXACT_HEADER

// A linear material's exact apply takes no state: its tangent does not depend
// on one.  The projected kernel always takes a state, because it builds the
// tangent from it, so the two signatures differ and the bench must know which.
extern "C" int EXACT_APPLY(
        const ptrdiff_t, const ptrdiff_t, idx_t **const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const,
        const double, const double,
#ifdef EXACT_TAKES_STATE
        const ptrdiff_t, const double *const, const double *const, const double *const,
#endif
        const ptrdiff_t, const double *const, const double *const, const double *const,
        const ptrdiff_t, double *const, double *const, double *const);

int main(int argc, char **argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 5;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const double mu = 2.3333333333333335, lmbda = 2.2;
    std::printf("threads %d, best of %d, HEX8\n\n", threads, repeats);
    std::printf("%10s %10s %10s %14s %14s %9s %12s\n",
                "elements", "nodes", "ndof", "exact MDOF/s", "proj. MDOF/s", "ratio", "rel. diff");
    for (int n : {8, 16, 24, 32, 40}) {
        const int nn = n + 1;
        const ptrdiff_t nnodes = (ptrdiff_t)nn * nn * nn;
        const ptrdiff_t nelements = (ptrdiff_t)n * n * n;
        const ptrdiff_t ndof = 3 * nnodes;
        const double hh = 1.0 / n;
        std::vector<std::vector<idx_t>> ev(8, std::vector<idx_t>(nelements));
        auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
        ptrdiff_t e = 0;
        for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i, ++e) {
            ev[0][e]=nid(i,j,k);     ev[1][e]=nid(i+1,j,k);   ev[2][e]=nid(i+1,j+1,k); ev[3][e]=nid(i,j+1,k);
            ev[4][e]=nid(i,j,k+1);   ev[5][e]=nid(i+1,j,k+1); ev[6][e]=nid(i+1,j+1,k+1); ev[7][e]=nid(i,j+1,k+1);
        }
        std::vector<idx_t*> evp; for (auto &r : ev) evp.push_back(r.data());
        std::vector<std::vector<geom_t>> adj(9, std::vector<geom_t>(nelements, 0));
        std::vector<geom_t> det(nelements, hh*hh*hh);
        for (ptrdiff_t i = 0; i < nelements; ++i) { adj[0][i]=hh*hh; adj[4][i]=hh*hh; adj[8][i]=hh*hh; }

        // Seeded from position, not from node index.  Indexing by node number
        // makes the field jump between neighbours in y and z -- one index step
        // is a whole row -- which drives the deformation gradient far past
        // anything physical and inverts elements, and a hyperelastic tangent is
        // not defined there.  A smooth field of the coordinates is what the
        // projection error is meant to be measured on.
        std::vector<double> ux(nnodes), uy(nnodes), uz(nnodes), hx(nnodes), hy(nnodes), hz(nnodes);
        for (int k = 0; k < nn; ++k) for (int j = 0; j < nn; ++j) for (int i = 0; i < nn; ++i) {
            const ptrdiff_t v = nid(i, j, k);
            const double x = i * hh, y = j * hh, z = k * hh;
            ux[v] = 0.02 * std::sin(3.0*x + 1.0*y + 0.5*z);
            uy[v] = 0.02 * std::sin(1.0*x + 3.0*y + 1.5*z);
            uz[v] = 0.02 * std::sin(0.5*x + 1.5*y + 3.0*z);
            hx[v] = 0.05 * std::sin(2.0*x + 0.7*y + 1.1*z);
            hy[v] = 0.05 * std::sin(0.7*x + 2.0*y + 1.3*z);
            hz[v] = 0.05 * std::sin(1.1*x + 1.3*y + 2.0*z);
        }
        std::vector<double> ax(nnodes,0), ay(nnodes,0), az(nnodes,0), bx(nnodes,0), by(nnodes,0), bz(nnodes,0);
        auto run_exact = [&] {
            std::fill(ax.begin(),ax.end(),0.0); std::fill(ay.begin(),ay.end(),0.0); std::fill(az.begin(),az.end(),0.0);
            EXACT_APPLY(nelements, nnodes, evp.data(),
                adj[0].data(),adj[1].data(),adj[2].data(),adj[3].data(),adj[4].data(),
                adj[5].data(),adj[6].data(),adj[7].data(),adj[8].data(), det.data(),
                lmbda, mu,
#ifdef EXACT_TAKES_STATE
                1, ux.data(), uy.data(), uz.data(),
#endif
                1, hx.data(), hy.data(), hz.data(), 1, ax.data(), ay.data(), az.data());
        };
        auto run_proj = [&] {
            std::fill(bx.begin(),bx.end(),0.0); std::fill(by.begin(),by.end(),0.0); std::fill(bz.begin(),bz.end(),0.0);
            sfem::codegen::PROJECTED_APPLY<double, geom_t>(nelements, nnodes, evp.data(),
                adj[0].data(),adj[1].data(),adj[2].data(),adj[3].data(),adj[4].data(),
                adj[5].data(),adj[6].data(),adj[7].data(),adj[8].data(), det.data(),
                lmbda, mu, 1, ux.data(), uy.data(), uz.data(),
                1, hx.data(), hy.data(), hz.data(), 1, bx.data(), by.data(), bz.data());
        };
        run_exact(); run_proj();
        double num = 0, den = 0;
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            num += std::fabs(ax[i]-bx[i])+std::fabs(ay[i]-by[i])+std::fabs(az[i]-bz[i]);
            den += std::fabs(ax[i])+std::fabs(ay[i])+std::fabs(az[i]);
        }
        auto best = [&](auto &&fn) {
            double top = 0;
            for (int r = 0; r < repeats; ++r) {
                auto t0 = std::chrono::steady_clock::now(); fn();
                auto t1 = std::chrono::steady_clock::now();
                top = std::max(top, (double)ndof / std::chrono::duration<double>(t1-t0).count() * 1e-6);
            }
            return top;
        };
        const double ex = best(run_exact), pr = best(run_proj);
        std::printf("%10ld %10ld %10ld %14.2f %14.2f %9.2fx %12.2e\n",
                    (long)nelements, (long)nnodes, (long)ndof, ex, pr, pr/ex, num/den);
    }
    return 0;
}
