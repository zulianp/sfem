#include "sfem_dss_bxyze_spectral_laplacian.hpp"

#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_structured.hpp"

#include "hex8_laplacian_inline_cpu.hpp"
#include "lagrange.hpp"
#include "line_quadrature.hpp"
#include "sfem_macros.hpp"
#include "sshex8.hpp"
#include "lagrange_hex_laplacian_inline.hpp"

#include <algorithm>

namespace {

bool valid_shape(const int level, const ptrdiff_t nelements, const int element_block_size) {
    return (level == 2 || level == 4 || level == 8) && nelements >= 0 && element_block_size > 0;
}

ptrdiff_t local_offset(const int n, const int element_block_size, const int ix, const int iy, const int iz) {
    return (ptrdiff_t)element_block_size *
           ((ptrdiff_t)ix + (ptrdiff_t)n * ((ptrdiff_t)iy + (ptrdiff_t)n * (ptrdiff_t)iz));
}

bool select_quadrature(const int order, const int q, const scalar_t *&qx, const scalar_t *&qw) {
    switch (order) {
        case 2: {
            if (line_q6_n != q) return false;
            qx = line_q6_x;
            qw = line_q6_w;
            return true;
        }
        case 4: {
            if (line_q12_n != q) return false;
            qx = line_q12_x;
            qw = line_q12_w;
            return true;
        }
        case 8: {
            if (line_q24_n != q) return false;
            qx = line_q24_x;
            qw = line_q24_w;
            return true;
        }
        default: {
            return false;
        }
    }
}

template <int order>
int lagrange_volumetric_apply_tpl(const ptrdiff_t                  nelements,
                                  const int                        element_block_size,
                                  const idx_t *const *const        elements,
                                  const geom_t *const *const       points,
                                  const real_t *const SFEM_RESTRICT x_bxyze,
                                  real_t *const SFEM_RESTRICT       y_bxyze) {
    SFEM_TRACE_SCOPE("lagrange_volumetric_apply_tpl");

    static const int N  = order + 1;
    static const int N2 = N * N;
    static const int N3 = N2 * N;
    static const int Q  = 3 * order;

    if (!valid_shape(order, nelements, element_block_size) || !elements || !points || !x_bxyze || !y_bxyze ||
        x_bxyze == y_bxyze) {
        return SFEM_FAILURE;
    }

    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(order, nelements, element_block_size);
    if (nvalues < 0) {
        return SFEM_FAILURE;
    }

    std::fill(y_bxyze, y_bxyze + nvalues, (real_t)0);
    if (nelements == 0) {
        return SFEM_SUCCESS;
    }

    scalar_t S[Q * N] = {0};
    scalar_t D[Q * N] = {0};

    const scalar_t *qx{nullptr};
    const scalar_t *qw{nullptr};
    if (!select_quadrature(order, Q, qx, qw)) {
        return SFEM_FAILURE;
    }

    lagrange_eval<scalar_t>(order, Q, qx, S);
    lagrange_diff_eval<scalar_t>(order, Q, qx, D);

    const int hex8_corners[8] = {
            sshex8_lidx(order, 0, 0, 0),
            sshex8_lidx(order, order, 0, 0),
            sshex8_lidx(order, order, order, 0),
            sshex8_lidx(order, 0, order, 0),
            sshex8_lidx(order, 0, 0, order),
            sshex8_lidx(order, order, 0, order),
            sshex8_lidx(order, order, order, order),
            sshex8_lidx(order, 0, order, order)};

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];
    const int           B = element_block_size;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const ptrdiff_t eb   = e / B;
        const int       lane = (int)(e - eb * B);
        const ptrdiff_t base = (ptrdiff_t)B * (ptrdiff_t)N3 * eb + lane;

        scalar_t      element_u[N3];
        accumulator_t element_vector[N3] = {0};
        scalar_t      fff[6];
        scalar_t      lx[8], ly[8], lz[8];

        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    const int       v   = iz * N2 + iy * N + ix;
                    const ptrdiff_t off = base + local_offset(N, B, ix, iy, iz);
                    element_u[v]        = (scalar_t)x_bxyze[off];
                }
            }
        }

        for (int v = 0; v < 8; ++v) {
            const idx_t gid = elements[hex8_corners[v]][e];
            lx[v]           = (scalar_t)x[gid];
            ly[v]           = (scalar_t)y[gid];
            lz[v]           = (scalar_t)z[gid];
        }

        hex8_fff(lx, ly, lz, 0.5, 0.5, 0.5, fff);
        lagrange_hex_laplacian_apply<N, Q, scalar_t>(S, D, fff, qw, element_u, element_vector);

        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    const int       v   = iz * N2 + iy * N + ix;
                    const ptrdiff_t off = base + local_offset(N, B, ix, iy, iz);
                    y_bxyze[off]        = (real_t)element_vector[v];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int dispatch_lagrange_volumetric_apply(const int                         level,
                                       const ptrdiff_t                   nelements,
                                       const int                         element_block_size,
                                       const idx_t *const *const         elements,
                                       const geom_t *const *const        points,
                                       const real_t *const SFEM_RESTRICT x_bxyze,
                                       real_t *const SFEM_RESTRICT       y_bxyze) {
    switch (level) {
        case 2:
            return lagrange_volumetric_apply_tpl<2>(nelements, element_block_size, elements, points, x_bxyze, y_bxyze);
        case 4:
            return lagrange_volumetric_apply_tpl<4>(nelements, element_block_size, elements, points, x_bxyze, y_bxyze);
        case 8:
            return lagrange_volumetric_apply_tpl<8>(nelements, element_block_size, elements, points, x_bxyze, y_bxyze);
        default:
            return SFEM_FAILURE;
    }
}

}  // namespace

int sfem_dss_bxyze_lagrange_spectral_laplacian_volumetric_apply(const int                  level,
                                                                const ptrdiff_t            nelements,
                                                                const int                  element_block_size,
                                                                const idx_t *const *const  elements,
                                                                const geom_t *const *const points,
                                                                const real_t *const        x_bxyze,
                                                                real_t *const              y_bxyze) {
    return dispatch_lagrange_volumetric_apply(level, nelements, element_block_size, elements, points, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_lagrange_spectral_laplacian_apply(const int                  level,
                                                     const ptrdiff_t            nx,
                                                     const ptrdiff_t            ny,
                                                     const ptrdiff_t            nz,
                                                     const int                  element_block_size,
                                                     const idx_t *const *const  elements,
                                                     const geom_t *const *const points,
                                                     const real_t *const        x_bxyze,
                                                     real_t *const              y_bxyze) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        return SFEM_FAILURE;
    }

    const ptrdiff_t nelements = nx * ny * nz;
    int err = dispatch_lagrange_volumetric_apply(level, nelements, element_block_size, elements, points, x_bxyze, y_bxyze);
    if (err != SFEM_SUCCESS) {
        return err;
    }

    return sfem_dss_bxyze_structured_inplace(level, nx, ny, nz, element_block_size, y_bxyze);
}
