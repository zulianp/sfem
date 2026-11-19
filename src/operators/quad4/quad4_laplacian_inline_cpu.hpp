#ifndef QUAD4_LAPLACIAN_INLINE_CPU_H
#define QUAD4_LAPLACIAN_INLINE_CPU_H

#include "sfem_base.hpp"

static SFEM_INLINE void quad4_laplacian_apply_micro(const idx_t *const SFEM_RESTRICT  ev,
                                                    geom_t **const SFEM_RESTRICT      points,
                                                    const real_t *const SFEM_RESTRICT u,
                                                    real_t *const SFEM_RESTRICT       element_vector) {
    static constexpr real_t q     = real_t(0.57735026918962576450914878050195746);
    static constexpr real_t qp[2] = {-q, q};

    const real_t x[4]  = {real_t(points[0][ev[0]]),
                          real_t(points[0][ev[1]]),
                          real_t(points[0][ev[2]]),
                          real_t(points[0][ev[3]])};
    const real_t y[4]  = {real_t(points[1][ev[0]]),
                          real_t(points[1][ev[1]]),
                          real_t(points[1][ev[2]]),
                          real_t(points[1][ev[3]])};
    const real_t ue[4] = {u[ev[0]], u[ev[1]], u[ev[2]], u[ev[3]]};

    element_vector[0] = 0;
    element_vector[1] = 0;
    element_vector[2] = 0;
    element_vector[3] = 0;

    for (int iy = 0; iy < 2; ++iy) {
        const real_t eta = qp[iy];
        for (int ix = 0; ix < 2; ++ix) {
            const real_t xi = qp[ix];

            const real_t dndxi[4] = {-(real_t(1) - eta) * real_t(0.25),
                                      (real_t(1) - eta) * real_t(0.25),
                                      (real_t(1) + eta) * real_t(0.25),
                                      -(real_t(1) + eta) * real_t(0.25)};
            const real_t dndeta[4] = {-(real_t(1) - xi) * real_t(0.25),
                                       -(real_t(1) + xi) * real_t(0.25),
                                       (real_t(1) + xi) * real_t(0.25),
                                       (real_t(1) - xi) * real_t(0.25)};

            real_t dx_dxi  = 0;
            real_t dx_deta = 0;
            real_t dy_dxi  = 0;
            real_t dy_deta = 0;
            for (int a = 0; a < 4; ++a) {
                dx_dxi += x[a] * dndxi[a];
                dx_deta += x[a] * dndeta[a];
                dy_dxi += y[a] * dndxi[a];
                dy_deta += y[a] * dndeta[a];
            }

            const real_t det     = dx_dxi * dy_deta - dx_deta * dy_dxi;
            const real_t inv_det = real_t(1) / det;

            real_t gx[4];
            real_t gy[4];
            real_t gux = 0;
            real_t guy = 0;
            for (int a = 0; a < 4; ++a) {
                gx[a] = (dy_deta * dndxi[a] - dy_dxi * dndeta[a]) * inv_det;
                gy[a] = (-dx_deta * dndxi[a] + dx_dxi * dndeta[a]) * inv_det;
                gux += gx[a] * ue[a];
                guy += gy[a] * ue[a];
            }

            for (int a = 0; a < 4; ++a) {
                element_vector[a] += det * (gx[a] * gux + gy[a] * guy);
            }
        }
    }
}

static SFEM_INLINE void quad4_laplacian_diag_micro(const idx_t *const SFEM_RESTRICT ev,
                                                   geom_t **const SFEM_RESTRICT     points,
                                                   real_t *const SFEM_RESTRICT      element_diag) {
    static constexpr real_t q     = real_t(0.57735026918962576450914878050195746);
    static constexpr real_t qp[2] = {-q, q};

    const real_t x[4] = {real_t(points[0][ev[0]]),
                         real_t(points[0][ev[1]]),
                         real_t(points[0][ev[2]]),
                         real_t(points[0][ev[3]])};
    const real_t y[4] = {real_t(points[1][ev[0]]),
                         real_t(points[1][ev[1]]),
                         real_t(points[1][ev[2]]),
                         real_t(points[1][ev[3]])};

    element_diag[0] = 0;
    element_diag[1] = 0;
    element_diag[2] = 0;
    element_diag[3] = 0;

    for (int iy = 0; iy < 2; ++iy) {
        const real_t eta = qp[iy];
        for (int ix = 0; ix < 2; ++ix) {
            const real_t xi = qp[ix];

            const real_t dndxi[4] = {-(real_t(1) - eta) * real_t(0.25),
                                      (real_t(1) - eta) * real_t(0.25),
                                      (real_t(1) + eta) * real_t(0.25),
                                      -(real_t(1) + eta) * real_t(0.25)};
            const real_t dndeta[4] = {-(real_t(1) - xi) * real_t(0.25),
                                       -(real_t(1) + xi) * real_t(0.25),
                                       (real_t(1) + xi) * real_t(0.25),
                                       (real_t(1) - xi) * real_t(0.25)};

            real_t dx_dxi  = 0;
            real_t dx_deta = 0;
            real_t dy_dxi  = 0;
            real_t dy_deta = 0;
            for (int a = 0; a < 4; ++a) {
                dx_dxi += x[a] * dndxi[a];
                dx_deta += x[a] * dndeta[a];
                dy_dxi += y[a] * dndxi[a];
                dy_deta += y[a] * dndeta[a];
            }

            const real_t det     = dx_dxi * dy_deta - dx_deta * dy_dxi;
            const real_t inv_det = real_t(1) / det;

            for (int a = 0; a < 4; ++a) {
                const real_t gx = (dy_deta * dndxi[a] - dy_dxi * dndeta[a]) * inv_det;
                const real_t gy = (-dx_deta * dndxi[a] + dx_dxi * dndeta[a]) * inv_det;
                element_diag[a] += det * (gx * gx + gy * gy);
            }
        }
    }
}

#endif  // QUAD4_LAPLACIAN_INLINE_CPU_H
