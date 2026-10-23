#ifndef SFEM_SDF_SAMPLING_HPP
#define SFEM_SDF_SAMPLING_HPP

#include <math.h>
#include <stddef.h>

#include "sfem_base.hpp"

// Sampling of a signed distance field on a regular grid by its interpolating cubic B-spline.
//
// The interpolant is twice continuously differentiable, so the field's gradient (a contact
// normal) and Hessian (a contact curvature) do not jump between cells, as a trilinear
// interpolant's gradient does. It equals the samples at the grid nodes, reproduces linear
// fields exactly and converges as h^4 on smooth ones.
//
// The grid is given as its B-spline coefficients (see sdf_bspline_coefficients), padded by one
// layer on every side: `n` is the number of samples per axis, `origin` and `delta` place
// sample 0 and the spacing, `stride` is the padded array's, and `coeffs` points at the
// coefficient of sample (0, 0, 0), so indices -1 and n are valid. A point is inside the grid
// while its cell is, i.e. 0 <= floor((p - origin) / delta) <= n - 2 on every axis, the rule
// the trilinear sampling used.

// The gap every sampler reports where a point lies outside the grid: large, so the contact
// constraint there stays inactive.
#define SFEM_SDF_UNREACHABLE 10000

// The cell of p and its local coordinate in [0, 1) per axis; false outside the grid.
SFEM_HOST_DEVICE SFEM_INLINE bool sdf_cell(const ptrdiff_t* const SFEM_RESTRICT n,
                                           const geom_t* const SFEM_RESTRICT    origin,
                                           const geom_t* const SFEM_RESTRICT    delta,
                                           const real_t                         x,
                                           const real_t                         y,
                                           const real_t                         z,
                                           ptrdiff_t* const SFEM_RESTRICT       cell,
                                           real_t* const SFEM_RESTRICT          t) {
    const real_t p[3] = {x, y, z};
    for (int d = 0; d < 3; ++d) {
        const real_t    g = (p[d] - (real_t)origin[d]) / (real_t)delta[d];
        const ptrdiff_t i = (ptrdiff_t)floor(g);
        if (i < 0 || i + 1 >= n[d]) {
            return false;
        }
        cell[d] = i;
        t[d]    = g - (real_t)i;
    }
    return true;
}

// Cubic B-spline weights of the four nodes i - 1 .. i + 2 at local coordinate t, and their
// first and second derivatives in t.
SFEM_HOST_DEVICE SFEM_INLINE void sdf_bspline_weights(const real_t                t,
                                                      real_t* const SFEM_RESTRICT w,
                                                      real_t* const SFEM_RESTRICT dw,
                                                      real_t* const SFEM_RESTRICT ddw) {
    const real_t t2 = t * t;
    const real_t t3 = t2 * t;
    const real_t s  = 1 - t;

    w[0] = s * s * s / 6;
    w[1] = (3 * t3 - 6 * t2 + 4) / 6;
    w[2] = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6;
    w[3] = t3 / 6;

    dw[0] = -s * s / 2;
    dw[1] = real_t(1.5) * t2 - 2 * t;
    dw[2] = real_t(-1.5) * t2 + t + real_t(0.5);
    dw[3] = t2 / 2;

    ddw[0] = s;
    ddw[1] = 3 * t - 2;
    ddw[2] = 1 - 3 * t;
    ddw[3] = t;
}

// The field at p, and with ORDER >= 1 its gradient, with ORDER >= 2 its Hessian packed
// [xx, xy, xz, yy, yz, zz], both in physical coordinates. False, and nothing written, outside
// the grid.
template <int ORDER>
SFEM_HOST_DEVICE SFEM_INLINE bool sdf_sample(const ptrdiff_t* const SFEM_RESTRICT n,
                                             const ptrdiff_t* const SFEM_RESTRICT stride,
                                             const geom_t* const SFEM_RESTRICT    origin,
                                             const geom_t* const SFEM_RESTRICT    delta,
                                             const geom_t* const SFEM_RESTRICT    coeffs,
                                             const real_t                         x,
                                             const real_t                         y,
                                             const real_t                         z,
                                             real_t* const SFEM_RESTRICT          value,
                                             real_t* const SFEM_RESTRICT          gradient,
                                             real_t* const SFEM_RESTRICT          hessian) {
    ptrdiff_t cell[3];
    real_t    t[3];
    if (!sdf_cell(n, origin, delta, x, y, z, cell, t)) {
        return false;
    }

    real_t w[3][4], dw[3][4], ddw[3][4];
    for (int d = 0; d < 3; ++d) {
        sdf_bspline_weights(t[d], w[d], dw[d], ddw[d]);
    }

    const geom_t* const base = &coeffs[(cell[0] - 1) * stride[0] + (cell[1] - 1) * stride[1] + (cell[2] - 1) * stride[2]];

    real_t f = 0;
    real_t g[3] = {0, 0, 0};
    real_t H[6] = {0, 0, 0, 0, 0, 0};
    for (int c = 0; c < 4; ++c) {
        for (int b = 0; b < 4; ++b) {
            for (int a = 0; a < 4; ++a) {
                const real_t v = (real_t)base[a * stride[0] + b * stride[1] + c * stride[2]];
                f += w[0][a] * w[1][b] * w[2][c] * v;
                if (ORDER >= 1) {
                    g[0] += dw[0][a] * w[1][b] * w[2][c] * v;
                    g[1] += w[0][a] * dw[1][b] * w[2][c] * v;
                    g[2] += w[0][a] * w[1][b] * dw[2][c] * v;
                }
                if (ORDER >= 2) {
                    H[0] += ddw[0][a] * w[1][b] * w[2][c] * v;
                    H[1] += dw[0][a] * dw[1][b] * w[2][c] * v;
                    H[2] += dw[0][a] * w[1][b] * dw[2][c] * v;
                    H[3] += w[0][a] * ddw[1][b] * w[2][c] * v;
                    H[4] += w[0][a] * dw[1][b] * dw[2][c] * v;
                    H[5] += w[0][a] * w[1][b] * ddw[2][c] * v;
                }
            }
        }
    }

    *value = f;
    if (ORDER >= 1) {
        for (int d = 0; d < 3; ++d) {
            gradient[d] = g[d] / (real_t)delta[d];
        }
    }
    if (ORDER >= 2) {
        const real_t dx = delta[0], dy = delta[1], dz = delta[2];
        hessian[0]      = H[0] / (dx * dx);
        hessian[1]      = H[1] / (dx * dy);
        hessian[2]      = H[2] / (dx * dz);
        hessian[3]      = H[3] / (dy * dy);
        hessian[4]      = H[4] / (dy * dz);
        hessian[5]      = H[5] / (dz * dz);
    }
    return true;
}

// The interpolating cubic B-spline coefficients of the samples of a grid with `n` points per
// axis (`samples` with strides `sample_stride`), written to a coefficient array with strides
// `coeff_stride` whose (0, 0, 0) entry is `coeffs`: the n nodes per axis and the pad nodes -1
// and n. Beyond its ends each grid line is extended by point reflection through its end
// sample, so linear fields are reproduced exactly up to the boundary. Separable recursive
// filtering (Unser, Aldroubi and Eden 1993), in double, once per grid line.
int sdf_bspline_coefficients(const ptrdiff_t* const SFEM_RESTRICT n,
                             const ptrdiff_t* const SFEM_RESTRICT sample_stride,
                             const geom_t* const SFEM_RESTRICT    samples,
                             const ptrdiff_t* const SFEM_RESTRICT coeff_stride,
                             geom_t* const SFEM_RESTRICT          coeffs);

#endif  // SFEM_SDF_SAMPLING_HPP
