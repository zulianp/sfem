// How the projected apply deviates as the displacement is warped, at fixed mesh.
//
//     u(x) = L x + w * N(x)
//
// `L` is a fixed, mild linear map and `N` a smooth nonlinear field; `w` is the
// warp severity, swept upward.  The deformation gradient of the linear part is
// constant, so at w = 0 the tangent does not vary over an element and the P0
// projection is *exact* -- the deviation there must be round-off.  That makes
// w = 0 a control on far more than the projection: a linear displacement is
// reproduced exactly by a quadratic element only if the nodal values land on
// the right nodes, so a nonzero deviation at w = 0 on TET10 is a node-ordering
// fault rather than an approximation.
//
// Validity is checked rather than assumed: the stored tangent is built through
// log(det F), so a non-positive Jacobian anywhere in the mesh makes an entry
// non-finite.  Such a row is reported as invalid rather than measured.  The
// actual range of det F is computed alongside in Python, where the basis
// gradients are available, and printed by the driver script.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include "sfem_base.hpp"
#include "kernel_math.hpp"
#include MATERIAL_INEXACT_HEADER
#include "element_mesh.inc"

#define TANGENT_COMPONENTS 45

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

// A mild shear plus stretch: constant gradient, det > 0.
static const double L[3][3] = {{ 0.030, 0.020, -0.010},
                               {-0.015, 0.025,  0.012},
                               { 0.008, -0.018, 0.022}};
static const double N_PH[3][3] = {{3.0, 1.0, 0.5}, {1.0, 3.0, 1.5}, {0.5, 1.5, 3.0}};
static const double H_PH[3][3] = {{2.0, 0.7, 1.1}, {0.7, 2.0, 1.3}, {1.1, 1.3, 2.0}};

int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 16;
    const double mu = 2.3333333333333335, lmbda = 2.2;
    Mesh m = build(n);
    const ptrdiff_t ndof = 3 * m.nnodes, ecount = m.nelements, cstride = ecount + 64;

    std::vector<double> ux(m.nnodes), uy(m.nnodes), uz(m.nnodes);
    std::vector<double> hx(m.nnodes), hy(m.nnodes), hz(m.nnodes);
    // The increment is smooth by default.  With -DRANDOM_INCREMENT it is white
    // noise instead, which matters for what the reported ratio means: for a
    // smooth increment the assembled (H h)_p is a discrete second derivative, so
    // neighbouring elements cancel and the denominator collapses, while the
    // per-element projection errors keep independent signs and do not cancel.
    // A ratio of a non-cancelling numerator to a cancelling denominator need not
    // converge even when the projection does.  White noise removes the
    // cancellation and makes the ratio a fair measure of the projection.
#ifdef RANDOM_INCREMENT
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
        hx[v] = dist(gen); hy[v] = dist(gen); hz[v] = dist(gen);
    }
#else
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
        const double x = m.px[v], y = m.py[v], z = m.pz[v];
        hx[v] = 0.05*std::sin(H_PH[0][0]*x + H_PH[0][1]*y + H_PH[0][2]*z);
        hy[v] = 0.05*std::sin(H_PH[1][0]*x + H_PH[1][1]*y + H_PH[1][2]*z);
        hz[v] = 0.05*std::sin(H_PH[2][0]*x + H_PH[2][1]*y + H_PH[2][2]*z);
    }
#endif
    std::vector<double> S64((size_t)cstride*TANGENT_COMPONENTS);
    std::vector<double> ax(m.nnodes), ay(m.nnodes), az(m.nnodes);
    std::vector<double> bx(m.nnodes), by(m.nnodes), bz(m.nnodes);

    std::printf("%s, %s, n=%d, %ld elements, %ld nodes, ndof %ld\n\n",
                MATERIAL_LABEL, ELEMENT_NAME, n, (long)m.nelements, (long)m.nnodes, (long)ndof);
    std::printf("  u(x) = L x + w N(x);  w = 0 makes the tangent constant, so the\n"
                "  projection is exact there and the deviation must be round-off.\n\n");
    std::printf("%10s | %12s | %14s\n", "warp w", "rel. dev.", "tangent");

    for (double w : {0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6}) {
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            const double x = m.px[v], y = m.py[v], z = m.pz[v];
            const double p[3] = {x, y, z};
            double lin[3] = {0, 0, 0};
            for (int c = 0; c < 3; ++c)
                for (int d = 0; d < 3; ++d) lin[c] += L[c][d]*p[d];
            ux[v] = lin[0] + w*0.02*std::sin(N_PH[0][0]*x + N_PH[0][1]*y + N_PH[0][2]*z);
            uy[v] = lin[1] + w*0.02*std::sin(N_PH[1][0]*x + N_PH[1][1]*y + N_PH[1][2]*z);
            uz[v] = lin[2] + w*0.02*std::sin(N_PH[2][0]*x + N_PH[2][1]*y + N_PH[2][2]*z);
        }
        std::fill(ax.begin(), ax.end(), 0.0); std::fill(ay.begin(), ay.end(), 0.0);
        std::fill(az.begin(), az.end(), 0.0);
        EXACT_APPLY(m.nelements, m.nnodes, m.evp.data(),
            m.adj[0].data(),m.adj[1].data(),m.adj[2].data(),m.adj[3].data(),m.adj[4].data(),
            m.adj[5].data(),m.adj[6].data(),m.adj[7].data(),m.adj[8].data(), m.det.data(),
            lmbda, mu,
#ifdef EXACT_TAKES_STATE
            1, ux.data(), uy.data(), uz.data(),
#endif
            1, hx.data(), hy.data(), hz.data(), 1, ax.data(), ay.data(), az.data());

        sfem::codegen::TANGENT_KERNEL<double, geom_t, double>(
            m.nelements, m.evp.data(),
            m.adj[0].data(),m.adj[1].data(),m.adj[2].data(),m.adj[3].data(),m.adj[4].data(),
            m.adj[5].data(),m.adj[6].data(),m.adj[7].data(),m.adj[8].data(), m.det.data(),
            lmbda, mu, 1, ux.data(), uy.data(), uz.data(), 1, cstride, S64.data());
        std::fill(bx.begin(), bx.end(), 0.0); std::fill(by.begin(), by.end(), 0.0);
        std::fill(bz.begin(), bz.end(), 0.0);
        sfem::codegen::STORED_APPLY<double, double>(
            m.nelements, m.evp.data(), 1, cstride, S64.data(),
            1, hx.data(), hy.data(), hz.data(), 1, bx.data(), by.data(), bz.data());

        double num = 0, den = 0;
        for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
            num += std::fabs(ax[i]-bx[i]) + std::fabs(ay[i]-by[i]) + std::fabs(az[i]-bz[i]);
            den += std::fabs(ax[i]) + std::fabs(ay[i]) + std::fabs(az[i]);
        }
        // A finite tangent everywhere means the Jacobian stayed positive: the
        // tangent is built through log(det F).
        bool finite = true;
        for (size_t k = 0; k < S64.size(); ++k)
            if (!std::isfinite(S64[k])) { finite = false; break; }
        if (!finite) std::printf("%10.4f | %12s | %14s\n", w, "--", "det F <= 0");
        else         std::printf("%10.4f | %12.4e | %14s\n", w, num / den, "finite");
    }
    return 0;
}
