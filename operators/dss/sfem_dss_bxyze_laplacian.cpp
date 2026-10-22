#include "sfem_dss_bxyze_laplacian.hpp"

#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_structured.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int SFEM_DSS_LAPLACIAN_MAX_LEVEL = 8;

using CartesianLayout = sfem_dss_bxyze_laplacian_cartesian_layout_t;

bool valid_shape(const int       level,
                 const ptrdiff_t nx,
                 const ptrdiff_t ny,
                 const ptrdiff_t nz,
                 const int       element_block_size) {
    return level >= 1 && level <= SFEM_DSS_LAPLACIAN_MAX_LEVEL && nx > 0 && ny > 0 && nz > 0 && element_block_size > 0;
}

bool valid_layout(const CartesianLayout &layout) {
    if (!valid_shape(layout.level, layout.nx, layout.ny, layout.nz, layout.element_block_size) ||
        layout.execution_block_size <= 0 || layout.execution_block_size > layout.element_block_size) {
        return false;
    }

    const int       n       = layout.level + 1;
    const ptrdiff_t ne      = layout.nx * layout.ny * layout.nz;
    const int       bsave   = layout.element_block_size;
    const int       bexec   = layout.execution_block_size;
    const ptrdiff_t n3      = (ptrdiff_t)n * (ptrdiff_t)n * (ptrdiff_t)n;
    const ptrdiff_t nblocks = (ne + bsave - 1) / bsave;
    const int       chunks  = (bsave + bexec - 1) / bexec;

    return layout.n == n && layout.nelements == ne && layout.block_stride == (ptrdiff_t)bsave * n3 &&
           layout.execution_stride == (ptrdiff_t)bexec * n3 && layout.nblocks == nblocks &&
           layout.chunks_per_block == chunks && layout.ntasks == nblocks * (ptrdiff_t)chunks;
}

ptrdiff_t local_offset(const int n, const int element_block_size, const int ix, const int iy, const int iz) {
    return (ptrdiff_t)element_block_size *
           ((ptrdiff_t)ix + (ptrdiff_t)n * ((ptrdiff_t)iy + (ptrdiff_t)n * (ptrdiff_t)iz));
}

template <typename Scalar>
bool valid_metrics(const Scalar hx, const Scalar hy, const Scalar hz) {
    return hx > (Scalar)0 && hy > (Scalar)0 && hz > (Scalar)0;
}

template <typename Scalar>
void legendre_and_derivative(const int n, const Scalar z, Scalar &p, Scalar &dp) {
    Scalar p0 = (Scalar)1;
    Scalar p1 = z;

    if (n == 0) {
        p  = p0;
        dp = (Scalar)0;
        return;
    }

    if (n == 1) {
        p  = p1;
        dp = (Scalar)1;
        return;
    }

    for (int k = 2; k <= n; ++k) {
        const Scalar pk = (((Scalar)(2 * k - 1)) * z * p1 - ((Scalar)(k - 1)) * p0) / (Scalar)k;
        p0              = p1;
        p1              = pk;
    }

    p  = p1;
    dp = ((Scalar)n) * (z * p1 - p0) / (z * z - (Scalar)1);
}

template <typename Scalar>
bool gauss_legendre_unit(const int q, std::vector<Scalar> &points, std::vector<Scalar> &weights) {
    if (q <= 0) {
        return false;
    }

    points.assign((size_t)q, (Scalar)0);
    weights.assign((size_t)q, (Scalar)0);

    const Scalar pi  = (Scalar)3.141592653589793238462643383279502884L;
    const int    mid = (q + 1) / 2;

    for (int i = 0; i < mid; ++i) {
        Scalar z = std::cos(pi * ((Scalar)i + (Scalar)0.75) / ((Scalar)q + (Scalar)0.5));
        Scalar p = (Scalar)0;
        Scalar d = (Scalar)0;

        for (int it = 0; it < 32; ++it) {
            legendre_and_derivative(q, z, p, d);
            const Scalar dz = p / d;
            z -= dz;
            if (std::abs(dz) <= (Scalar)64 * std::numeric_limits<Scalar>::epsilon()) {
                break;
            }
        }

        legendre_and_derivative(q, z, p, d);

        const Scalar w = (Scalar)1 / (((Scalar)1 - z * z) * d * d);
        points[(size_t)i]             = ((Scalar)1 - z) * (Scalar)0.5;
        points[(size_t)(q - 1 - i)]   = ((Scalar)1 + z) * (Scalar)0.5;
        weights[(size_t)i]           = w;
        weights[(size_t)(q - 1 - i)] = w;
    }

    return true;
}

template <typename Scalar>
void lagrange_equispaced_eval(const int                  level,
                              const std::vector<Scalar> &points,
                              std::vector<Scalar>       &values,
                              std::vector<Scalar>       &derivatives) {
    const int n = level + 1;
    const int q = (int)points.size();

    values.assign((size_t)q * (size_t)n, (Scalar)0);
    derivatives.assign((size_t)q * (size_t)n, (Scalar)0);

    std::vector<Scalar> nodes((size_t)n);
    for (int i = 0; i < n; ++i) {
        nodes[(size_t)i] = (Scalar)i / (Scalar)level;
    }

    for (int iq = 0; iq < q; ++iq) {
        const Scalar x = points[(size_t)iq];

        for (int i = 0; i < n; ++i) {
            Scalar li = (Scalar)1;
            for (int m = 0; m < n; ++m) {
                if (m != i) {
                    li *= (x - nodes[(size_t)m]) / (nodes[(size_t)i] - nodes[(size_t)m]);
                }
            }

            Scalar dli = (Scalar)0;
            for (int m = 0; m < n; ++m) {
                if (m == i) {
                    continue;
                }

                Scalar term = (Scalar)1 / (nodes[(size_t)i] - nodes[(size_t)m]);
                for (int r = 0; r < n; ++r) {
                    if (r != i && r != m) {
                        term *= (x - nodes[(size_t)r]) / (nodes[(size_t)i] - nodes[(size_t)r]);
                    }
                }

                dli += term;
            }

            values[(size_t)iq * (size_t)n + (size_t)i]      = li;
            derivatives[(size_t)iq * (size_t)n + (size_t)i] = dli;
        }
    }
}

template <typename Scalar>
bool make_1d_mass_stiffness(const int level, std::vector<Scalar> &mass, std::vector<Scalar> &stiffness) {
    const int n = level + 1;
    const int q = n;

    std::vector<Scalar> points;
    std::vector<Scalar> weights;
    std::vector<Scalar> values;
    std::vector<Scalar> derivatives;

    if (!gauss_legendre_unit(q, points, weights)) {
        return false;
    }

    lagrange_equispaced_eval(level, points, values, derivatives);

    mass.assign((size_t)n * (size_t)n, (Scalar)0);
    stiffness.assign((size_t)n * (size_t)n, (Scalar)0);

    for (int iq = 0; iq < q; ++iq) {
        const Scalar w = weights[(size_t)iq];
        for (int i = 0; i < n; ++i) {
            const Scalar vi = values[(size_t)iq * (size_t)n + (size_t)i];
            const Scalar di = derivatives[(size_t)iq * (size_t)n + (size_t)i];

            for (int j = 0; j < n; ++j) {
                const Scalar vj = values[(size_t)iq * (size_t)n + (size_t)j];
                const Scalar dj = derivatives[(size_t)iq * (size_t)n + (size_t)j];

                mass[(size_t)i * (size_t)n + (size_t)j] += w * vi * vj;
                stiffness[(size_t)i * (size_t)n + (size_t)j] += w * di * dj;
            }
        }
    }

    return true;
}

template <typename Scalar>
void zero_chunk(const int       n,
                const int       storage_bsize,
                const int       lane_begin,
                const int       lane_count,
                Scalar *const SFEM_RESTRICT y) {
    for (int iz = 0; iz < n; ++iz) {
        for (int iy = 0; iy < n; ++iy) {
            for (int ix = 0; ix < n; ++ix) {
                Scalar *const SFEM_RESTRICT dst = &y[local_offset(n, storage_bsize, ix, iy, iz) + lane_begin];

#pragma omp simd
                for (int b = 0; b < lane_count; ++b) {
                    dst[b] = (Scalar)0;
                }
            }
        }
    }
}

template <typename Scalar>
void apply_tensor3(const int             n,
                   const int             storage_bsize,
                   const int             execution_bsize,
                   const int             lane_begin,
                   const int             lane_count,
                   const Scalar *const   a0,
                   const Scalar *const   a1,
                   const Scalar *const   a2,
                   const Scalar          scale,
                   const Scalar *const SFEM_RESTRICT x,
                   Scalar *const SFEM_RESTRICT       y,
                   Scalar *const SFEM_RESTRICT       scratch0,
                   Scalar *const SFEM_RESTRICT       scratch1) {
    for (int iz = 0; iz < n; ++iz) {
        for (int iy = 0; iy < n; ++iy) {
            for (int ox = 0; ox < n; ++ox) {
                Scalar *const SFEM_RESTRICT dst = &scratch0[local_offset(n, execution_bsize, ox, iy, iz)];

#pragma omp simd
                for (int b = 0; b < lane_count; ++b) {
                    dst[b] = (Scalar)0;
                }

                for (int ix = 0; ix < n; ++ix) {
                    const Scalar        c   = a0[(size_t)ox * (size_t)n + (size_t)ix];
                    const Scalar *const src = &x[local_offset(n, storage_bsize, ix, iy, iz) + lane_begin];

#pragma omp simd
                    for (int b = 0; b < lane_count; ++b) {
                        dst[b] += c * src[b];
                    }
                }
            }
        }
    }

    for (int iz = 0; iz < n; ++iz) {
        for (int oy = 0; oy < n; ++oy) {
            for (int ox = 0; ox < n; ++ox) {
                Scalar *const SFEM_RESTRICT dst = &scratch1[local_offset(n, execution_bsize, ox, oy, iz)];

#pragma omp simd
                for (int b = 0; b < lane_count; ++b) {
                    dst[b] = (Scalar)0;
                }

                for (int iy = 0; iy < n; ++iy) {
                    const Scalar        c   = a1[(size_t)oy * (size_t)n + (size_t)iy];
                    const Scalar *const src = &scratch0[local_offset(n, execution_bsize, ox, iy, iz)];

#pragma omp simd
                    for (int b = 0; b < lane_count; ++b) {
                        dst[b] += c * src[b];
                    }
                }
            }
        }
    }

    for (int oz = 0; oz < n; ++oz) {
        for (int oy = 0; oy < n; ++oy) {
            for (int ox = 0; ox < n; ++ox) {
                Scalar *const SFEM_RESTRICT dst = &y[local_offset(n, storage_bsize, ox, oy, oz) + lane_begin];

                for (int iz = 0; iz < n; ++iz) {
                    const Scalar        c   = scale * a2[(size_t)oz * (size_t)n + (size_t)iz];
                    const Scalar *const src = &scratch1[local_offset(n, execution_bsize, ox, oy, iz)];

#pragma omp simd
                    for (int b = 0; b < lane_count; ++b) {
                        dst[b] += c * src[b];
                    }
                }
            }
        }
    }
}

template <typename Scalar>
int cartesian_volumetric_apply(const CartesianLayout      &layout,
                               const Scalar               hx,
                               const Scalar               hy,
                               const Scalar               hz,
                               const Scalar               kappa,
                               const Scalar *const SFEM_RESTRICT x_bxyze,
                               Scalar *const SFEM_RESTRICT       y_bxyze) {
    if (!valid_layout(layout) || !valid_metrics(hx, hy, hz) || !x_bxyze || !y_bxyze || x_bxyze == y_bxyze) {
        return SFEM_FAILURE;
    }

    const int       level            = layout.level;
    const int       n                = layout.n;
    const int       bsave            = layout.element_block_size;
    const int       bexec            = layout.execution_block_size;
    const ptrdiff_t block_stride     = layout.block_stride;
    const ptrdiff_t execution_stride = layout.execution_stride;
    const int       chunks_per_block = layout.chunks_per_block;
    const ptrdiff_t ntasks           = layout.ntasks;

    std::vector<Scalar> mass;
    std::vector<Scalar> stiffness;
    if (!make_1d_mass_stiffness(level, mass, stiffness)) {
        return SFEM_FAILURE;
    }

    const Scalar cx = kappa * hy * hz / hx;
    const Scalar cy = kappa * hx * hz / hy;
    const Scalar cz = kappa * hx * hy / hz;

#pragma omp parallel
    {
        std::vector<Scalar> scratch0((size_t)execution_stride);
        std::vector<Scalar> scratch1((size_t)execution_stride);

#pragma omp for schedule(static)
        for (ptrdiff_t task = 0; task < ntasks; ++task) {
            const ptrdiff_t eb    = task / chunks_per_block;
            const int       chunk = (int)(task - eb * chunks_per_block);
            const int       lane_begin = chunk * bexec;
            const int       lane_count = std::min(bexec, bsave - lane_begin);

            const Scalar *const SFEM_RESTRICT xb = &x_bxyze[block_stride * eb];
            Scalar *const SFEM_RESTRICT       yb = &y_bxyze[block_stride * eb];

            zero_chunk(n, bsave, lane_begin, lane_count, yb);

            apply_tensor3(n,
                          bsave,
                          bexec,
                          lane_begin,
                          lane_count,
                          stiffness.data(),
                          mass.data(),
                          mass.data(),
                          cx,
                          xb,
                          yb,
                          scratch0.data(),
                          scratch1.data());
            apply_tensor3(n,
                          bsave,
                          bexec,
                          lane_begin,
                          lane_count,
                          mass.data(),
                          stiffness.data(),
                          mass.data(),
                          cy,
                          xb,
                          yb,
                          scratch0.data(),
                          scratch1.data());
            apply_tensor3(n,
                          bsave,
                          bexec,
                          lane_begin,
                          lane_count,
                          mass.data(),
                          mass.data(),
                          stiffness.data(),
                          cz,
                          xb,
                          yb,
                          scratch0.data(),
                          scratch1.data());
        }
    }

    return SFEM_SUCCESS;
}

}  // namespace

int sfem_dss_bxyze_laplacian_cartesian_create_layout(
        const int                                                level,
        const ptrdiff_t                                          nx,
        const ptrdiff_t                                          ny,
        const ptrdiff_t                                          nz,
        const int                                                element_block_size,
        const int                                                execution_block_size,
        sfem_dss_bxyze_laplacian_cartesian_layout_t *const layout) {
    if (!layout || !valid_shape(level, nx, ny, nz, element_block_size) || execution_block_size <= 0 ||
        execution_block_size > element_block_size) {
        return SFEM_FAILURE;
    }

    const ptrdiff_t max_size = std::numeric_limits<ptrdiff_t>::max();
    if (nx > max_size / ny || nx * ny > max_size / nz) {
        return SFEM_FAILURE;
    }

    const int err = sfem_dss_bxyze_create_layout(
            level, nx * ny * nz, element_block_size, execution_block_size, layout);
    if (err != SFEM_SUCCESS) {
        return err;
    }

    layout->nx = nx;
    layout->ny = ny;
    layout->nz = nz;

    return SFEM_SUCCESS;
}

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_layout(
        const sfem_dss_bxyze_laplacian_cartesian_layout_t *const layout,
        const real_t                                             hx,
        const real_t                                             hy,
        const real_t                                             hz,
        const real_t                                             kappa,
        const real_t *const                                      x_bxyze,
        real_t *const                                            y_bxyze) {
    if (!layout) {
        return SFEM_FAILURE;
    }

    return cartesian_volumetric_apply<real_t>(*layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_apply_layout(
        const sfem_dss_bxyze_laplacian_cartesian_layout_t *const layout,
        const real_t                                             hx,
        const real_t                                             hy,
        const real_t                                             hz,
        const real_t                                             kappa,
        const real_t *const                                      x_bxyze,
        real_t *const                                            y_bxyze) {
    if (!layout) {
        return SFEM_FAILURE;
    }

    const int err = sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_layout(
            layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    return sfem_dss_bxyze_structured_inplace(
            layout->level, layout->nx, layout->ny, layout->nz, layout->element_block_size, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply(const int           level,
                                                        const ptrdiff_t     nx,
                                                        const ptrdiff_t     ny,
                                                        const ptrdiff_t     nz,
                                                        const int           element_block_size,
                                                        const real_t        hx,
                                                        const real_t        hy,
                                                        const real_t        hz,
                                                        const real_t        kappa,
                                                        const real_t *const x_bxyze,
                                                        real_t *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, element_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    return sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_layout(&layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_blocked(const int           level,
                                                                const ptrdiff_t     nx,
                                                                const ptrdiff_t     ny,
                                                                const ptrdiff_t     nz,
                                                                const int           element_block_size,
                                                                const int           execution_block_size,
                                                                const real_t        hx,
                                                                const real_t        hy,
                                                                const real_t        hz,
                                                                const real_t        kappa,
                                                                const real_t *const x_bxyze,
                                                                real_t *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, execution_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    return sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_layout(&layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_apply(const int           level,
                                             const ptrdiff_t     nx,
                                             const ptrdiff_t     ny,
                                             const ptrdiff_t     nz,
                                             const int           element_block_size,
                                             const real_t        hx,
                                             const real_t        hy,
                                             const real_t        hz,
                                             const real_t        kappa,
                                             const real_t *const x_bxyze,
                                             real_t *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, element_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    return sfem_dss_bxyze_laplacian_cartesian_apply_layout(&layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_apply_blocked(const int           level,
                                                     const ptrdiff_t     nx,
                                                     const ptrdiff_t     ny,
                                                     const ptrdiff_t     nz,
                                                     const int           element_block_size,
                                                     const int           execution_block_size,
                                                     const real_t        hx,
                                                     const real_t        hy,
                                                     const real_t        hz,
                                                     const real_t        kappa,
                                                     const real_t *const x_bxyze,
                                                     real_t *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, execution_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    return sfem_dss_bxyze_laplacian_cartesian_apply_layout(&layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_real32(const int          level,
                                                               const ptrdiff_t    nx,
                                                               const ptrdiff_t    ny,
                                                               const ptrdiff_t    nz,
                                                               const int          element_block_size,
                                                               const float        hx,
                                                               const float        hy,
                                                               const float        hz,
                                                               const float        kappa,
                                                               const float *const x_bxyze,
                                                               float *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, element_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    return cartesian_volumetric_apply<float>(layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_apply_real32(const int          level,
                                                    const ptrdiff_t    nx,
                                                    const ptrdiff_t    ny,
                                                    const ptrdiff_t    nz,
                                                    const int          element_block_size,
                                                    const float        hx,
                                                    const float        hy,
                                                    const float        hz,
                                                    const float        kappa,
                                                    const float *const x_bxyze,
                                                    float *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, element_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    const int vol_err = cartesian_volumetric_apply<float>(layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
    if (vol_err != SFEM_SUCCESS) {
        return vol_err;
    }

    return sfem_dss_bxyze_structured_inplace_real32(
            layout.level, layout.nx, layout.ny, layout.nz, layout.element_block_size, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_real64(const int           level,
                                                               const ptrdiff_t     nx,
                                                               const ptrdiff_t     ny,
                                                               const ptrdiff_t     nz,
                                                               const int           element_block_size,
                                                               const double        hx,
                                                               const double        hy,
                                                               const double        hz,
                                                               const double        kappa,
                                                               const double *const x_bxyze,
                                                               double *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, element_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    return cartesian_volumetric_apply<double>(layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
}

int sfem_dss_bxyze_laplacian_cartesian_apply_real64(const int           level,
                                                    const ptrdiff_t     nx,
                                                    const ptrdiff_t     ny,
                                                    const ptrdiff_t     nz,
                                                    const int           element_block_size,
                                                    const double        hx,
                                                    const double        hy,
                                                    const double        hz,
                                                    const double        kappa,
                                                    const double *const x_bxyze,
                                                    double *const       y_bxyze) {
    CartesianLayout layout;
    const int err = sfem_dss_bxyze_laplacian_cartesian_create_layout(
            level, nx, ny, nz, element_block_size, element_block_size, &layout);

    if (err != SFEM_SUCCESS) {
        return err;
    }

    const int vol_err = cartesian_volumetric_apply<double>(layout, hx, hy, hz, kappa, x_bxyze, y_bxyze);
    if (vol_err != SFEM_SUCCESS) {
        return vol_err;
    }

    return sfem_dss_bxyze_structured_inplace_real64(
            layout.level, layout.nx, layout.ny, layout.nz, layout.element_block_size, y_bxyze);
}
