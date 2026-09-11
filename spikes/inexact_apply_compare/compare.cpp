// Compare the generated inexact apply against the exact one on a TET4 grid.
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "sfem_base.hpp"

#include "kernel_math.hpp"
#include "linear_elasticity_tet4_inexact_apply_inline.hpp"

#include "generated_abi.inc"

//: `Sbar`'s component count for a three-dimensional element.
#define TANGENT_COMPONENTS 45

extern "C" int linear_elasticity_tet4_apply_a_msoa(
        const int,
        const ptrdiff_t, const ptrdiff_t, idx_t **const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const, const geom_t *const, const geom_t *const,
        const geom_t *const,
        const double, const double,
        const ptrdiff_t, const double *const, const double *const, const double *const,
        const ptrdiff_t, double *const, double *const, double *const);

int main() {
    const int n = 6;
    const int nn = n + 1;
    const ptrdiff_t nnodes = (ptrdiff_t)nn * nn * nn;
    const ptrdiff_t nelements = (ptrdiff_t)n * n * n * 6;
    const double h = 1.0 / n;
    std::vector<std::vector<idx_t>> ev(4, std::vector<idx_t>(nelements));
    std::vector<std::vector<geom_t>> pts(3, std::vector<geom_t>(nnodes));
    auto nid = [&](int i, int j, int k) { return (idx_t)((k * nn + j) * nn + i); };
    for (int k = 0; k < nn; ++k) for (int j = 0; j < nn; ++j) for (int i = 0; i < nn; ++i) {
        pts[0][nid(i,j,k)] = i * h; pts[1][nid(i,j,k)] = j * h; pts[2][nid(i,j,k)] = k * h;
    }
    static const int tets[6][4] = {{0,1,3,7},{0,1,5,7},{0,4,5,7},{0,4,6,7},{0,2,6,7},{0,2,3,7}};
    std::vector<std::vector<geom_t>> adj(9, std::vector<geom_t>(nelements, 0));
    std::vector<geom_t> det(nelements, 0);
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
            for (int c = 0; c < 4; ++c) ev[c][e] = v[c];
            adj[0][e] = J[4]*J[8]-J[5]*J[7]; adj[1][e] = J[2]*J[7]-J[1]*J[8]; adj[2][e] = J[1]*J[5]-J[2]*J[4];
            adj[3][e] = J[5]*J[6]-J[3]*J[8]; adj[4][e] = J[0]*J[8]-J[2]*J[6]; adj[5][e] = J[2]*J[3]-J[0]*J[5];
            adj[6][e] = J[3]*J[7]-J[4]*J[6]; adj[7][e] = J[1]*J[6]-J[0]*J[7]; adj[8][e] = J[0]*J[4]-J[1]*J[3];
            det[e] = D;
        }
    std::vector<idx_t*> evp; for (auto &r : ev) evp.push_back(r.data());

    const double mu = 2.3333333333333335, lmbda = 2.2;
    std::vector<double> hx(nnodes), hy(nnodes), hz(nnodes), zero(nnodes, 0.0);
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        hx[i] = std::sin(0.5 + 0.125 * i) * 0.05;
        hy[i] = std::sin(1.5 + 0.125 * i) * 0.05;
        hz[i] = std::sin(2.5 + 0.125 * i) * 0.05;
    }
    std::vector<double> ax(nnodes,0), ay(nnodes,0), az(nnodes,0);
    std::vector<double> bx(nnodes,0), by(nnodes,0), bz(nnodes,0);

    // `Sbar` is 45 numbers per element in three dimensions, stored one
    // component-plane at a time.  The padding keeps each plane's start clear of
    // the previous plane's tail for the vector loads.
    const ptrdiff_t tangent_component_stride = nelements + 64;

    linear_elasticity_tet4_apply_a_msoa(SFEM_CODEGEN_F64, nelements, nnodes, evp.data(),
        adj[0].data(),adj[1].data(),adj[2].data(),adj[3].data(),adj[4].data(),
        adj[5].data(),adj[6].data(),adj[7].data(),adj[8].data(), det.data(),
        lmbda, mu, 1, hx.data(), hy.data(), hz.data(), 1, ax.data(), ay.data(), az.data());

    // The inexact apply is two kernels, not one: the projected tangent `Sbar`
    // is assembled once per state, and the apply reads it instead of the
    // geometry and the state.  That split is the whole point of the
    // construction -- see the README -- so the comparison performs both, and
    // what it compares is still one exact apply against one inexact one.
    //
    // `linear_elasticity`'s tangent does not depend on the state, so the
    // assembly is handed a zero state and still produces the same `Sbar`.
    std::vector<double> tangent((size_t)tangent_component_stride * TANGENT_COMPONENTS);
    sfem::codegen::linear_elasticity_tet4_inexact_apply_tangent_a_msoa_impl<double, geom_t, double, 16>(
        nelements,
        adj[0].data(),adj[1].data(),adj[2].data(),adj[3].data(),adj[4].data(),
        adj[5].data(),adj[6].data(),adj[7].data(),adj[8].data(), det.data(),
        lmbda, mu,
        tangent_component_stride, tangent.data());

    sfem::codegen::linear_elasticity_tet4_inexact_apply_stored_a_msoa_impl<double, double, 16>(
        nelements, evp.data(),
        tangent_component_stride, tangent.data(),
        1, hx.data(), hy.data(), hz.data(),
        1, bx.data(), by.data(), bz.data());

    double num = 0, den = 0;
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        num += std::fabs(ax[i]-bx[i]) + std::fabs(ay[i]-by[i]) + std::fabs(az[i]-bz[i]);
        den += std::fabs(ax[i]) + std::fabs(ay[i]) + std::fabs(az[i]);
    }
    std::printf("nelements %ld  nnodes %ld  ndof %ld\n", (long)nelements, (long)nnodes, (long)(3*nnodes));
    std::printf("exact   l1 = %.15e\n", den);
    std::printf("inexact l1 = %.15e\n", [&]{ double s=0; for (ptrdiff_t i=0;i<nnodes;++i) s+=std::fabs(bx[i])+std::fabs(by[i])+std::fabs(bz[i]); return s; }());
    std::printf("relative l1 difference = %.3e\n", num / den);
    return num / den < 1e-12 ? 0 : 1;
}
