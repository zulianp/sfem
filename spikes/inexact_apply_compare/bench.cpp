// Exact vs projected apply: agreement, then throughput, on a TET4 grid.
//
// Reports MDOF/s against the degree-of-freedom count and the thread count,
// swept over refinement so that the reader can see where it saturates rather
// than being handed one number from an arbitrary size.
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
#include "linear_elasticity_tet4_inexact_apply_inline.hpp"

extern "C" int linear_elasticity_tet4_apply_affine_mesh_soa(
        const ptrdiff_t, const ptrdiff_t, idx_t **const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const,
        const double, const double,
        const ptrdiff_t, const double *const, const double *const, const double *const,
        const ptrdiff_t, double *const, double *const, double *const);

struct Mesh {
    ptrdiff_t nelements = 0, nnodes = 0;
    std::vector<std::vector<idx_t>> ev;
    std::vector<idx_t *> evp;
    std::vector<std::vector<geom_t>> adj;
    std::vector<geom_t> det;
};

static Mesh build(int n) {
    Mesh m;
    const int nn = n + 1;
    const double h = 1.0 / n;
    m.nnodes = (ptrdiff_t)nn * nn * nn;
    m.nelements = (ptrdiff_t)n * n * n * 6;
    m.ev.assign(4, std::vector<idx_t>(m.nelements));
    std::vector<std::vector<geom_t>> pts(3, std::vector<geom_t>(m.nnodes));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    for (int k = 0; k < nn; ++k) for (int j = 0; j < nn; ++j) for (int i = 0; i < nn; ++i) {
        pts[0][nid(i,j,k)] = i * h; pts[1][nid(i,j,k)] = j * h; pts[2][nid(i,j,k)] = k * h;
    }
    static const int tets[6][4] = {{0,1,3,7},{0,1,5,7},{0,4,5,7},{0,4,6,7},{0,2,6,7},{0,2,3,7}};
    m.adj.assign(9, std::vector<geom_t>(m.nelements, 0));
    m.det.assign(m.nelements, 0);
    ptrdiff_t e = 0;
    for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
        for (int t = 0; t < 6; ++t, ++e) {
            idx_t v[4];
            for (int c = 0; c < 4; ++c) {
                const int b = tets[t][c];
                v[c] = nid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
            }
            double J[9];
            for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
                J[d*3+c] = (double)pts[d][v[c+1]] - (double)pts[d][v[0]];
            double D = J[0]*(J[4]*J[8]-J[5]*J[7]) - J[1]*(J[3]*J[8]-J[5]*J[6]) + J[2]*(J[3]*J[7]-J[4]*J[6]);
            if (D < 0) { std::swap(v[1], v[2]);
                for (int c = 0; c < 3; ++c) for (int d = 0; d < 3; ++d)
                    J[d*3+c] = (double)pts[d][v[c+1]] - (double)pts[d][v[0]];
                D = -D; }
            for (int c = 0; c < 4; ++c) m.ev[c][e] = v[c];
            m.adj[0][e]=J[4]*J[8]-J[5]*J[7]; m.adj[1][e]=J[2]*J[7]-J[1]*J[8]; m.adj[2][e]=J[1]*J[5]-J[2]*J[4];
            m.adj[3][e]=J[5]*J[6]-J[3]*J[8]; m.adj[4][e]=J[0]*J[8]-J[2]*J[6]; m.adj[5][e]=J[2]*J[3]-J[0]*J[5];
            m.adj[6][e]=J[3]*J[7]-J[4]*J[6]; m.adj[7][e]=J[1]*J[6]-J[0]*J[7]; m.adj[8][e]=J[0]*J[4]-J[1]*J[3];
            m.det[e] = D;
        }
    for (auto &r : m.ev) m.evp.push_back(r.data());
    return m;
}

int main(int argc, char **argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 5;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const double mu = 2.3333333333333335, lmbda = 2.2;
    std::printf("threads %d, best of %d, linear elasticity TET4\n\n", threads, repeats);
    std::printf("%10s %12s %12s %14s %14s %9s %12s\n",
                "elements", "nodes", "ndof", "exact MDOF/s", "proj. MDOF/s", "ratio", "rel. diff");
    for (int n : {8, 16, 24, 32, 40}) {
        Mesh m = build(n);
        const ptrdiff_t ndof = 3 * m.nnodes;
        std::vector<double> hx(m.nnodes), hy(m.nnodes), hz(m.nnodes), zero(m.nnodes, 0.0);
        for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
            hx[i] = std::sin(0.5 + 0.125*i)*0.05;
            hy[i] = std::sin(1.5 + 0.125*i)*0.05;
            hz[i] = std::sin(2.5 + 0.125*i)*0.05;
        }
        std::vector<double> ax(m.nnodes,0), ay(m.nnodes,0), az(m.nnodes,0);
        std::vector<double> bx(m.nnodes,0), by(m.nnodes,0), bz(m.nnodes,0);
        auto run_exact = [&] {
            std::fill(ax.begin(),ax.end(),0.0); std::fill(ay.begin(),ay.end(),0.0); std::fill(az.begin(),az.end(),0.0);
            linear_elasticity_tet4_apply_affine_mesh_soa(m.nelements, m.nnodes, m.evp.data(),
                m.adj[0].data(),m.adj[1].data(),m.adj[2].data(),m.adj[3].data(),m.adj[4].data(),
                m.adj[5].data(),m.adj[6].data(),m.adj[7].data(),m.adj[8].data(), m.det.data(),
                lmbda, mu, 1, hx.data(), hy.data(), hz.data(), 1, ax.data(), ay.data(), az.data());
        };
        auto run_proj = [&] {
            std::fill(bx.begin(),bx.end(),0.0); std::fill(by.begin(),by.end(),0.0); std::fill(bz.begin(),bz.end(),0.0);
            sfem::codegen::linear_elasticity_tet4_apply_inexact_affine_mesh_soa_impl<double, geom_t>(
                m.nelements, m.nnodes, m.evp.data(),
                m.adj[0].data(),m.adj[1].data(),m.adj[2].data(),m.adj[3].data(),m.adj[4].data(),
                m.adj[5].data(),m.adj[6].data(),m.adj[7].data(),m.adj[8].data(), m.det.data(),
                lmbda, mu, 1, zero.data(), zero.data(), zero.data(),
                1, hx.data(), hy.data(), hz.data(), 1, bx.data(), by.data(), bz.data());
        };
        run_exact(); run_proj();
        double num = 0, den = 0;
        for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
            num += std::fabs(ax[i]-bx[i])+std::fabs(ay[i]-by[i])+std::fabs(az[i]-bz[i]);
            den += std::fabs(ax[i])+std::fabs(ay[i])+std::fabs(az[i]);
        }
        auto best = [&](void (*)(), auto &&fn) {
            double top = 0;
            for (int r = 0; r < repeats; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                fn();
                auto t1 = std::chrono::steady_clock::now();
                const double s = std::chrono::duration<double>(t1 - t0).count();
                top = std::max(top, (double)ndof / s * 1e-6);
            }
            return top;
        };
        const double e = best(nullptr, run_exact);
        const double p = best(nullptr, run_proj);
        std::printf("%10ld %12ld %12ld %14.2f %14.2f %9.2fx %12.2e\n",
                    (long)m.nelements, (long)m.nnodes, (long)ndof, e, p, p / e, num / den);
    }
    return 0;
}
