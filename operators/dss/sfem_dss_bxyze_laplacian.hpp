#pragma once

#include "sfem_dss_bxyze.hpp"

#include <stddef.h>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif

#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

/*
 * Finite-element Cartesian Laplacian on scalar fields stored in BXYZE layout.
 *
 * The structured element ordering is:
 *   e = ex + nx * (ey + ny * ez)
 *
 * The cell-wise storage layout is:
 *   bxyze[b + B * (x + N * (y + N * (z + N * eb)))]
 *
 * with:
 *   N  = level + 1
 *   B  = element_block_size
 *   eb = e / B
 *   b  = e % B
 *
 * The kernels assume equispaced Lagrange nodes on [0, 1] in each coordinate
 * and a Cartesian axis-aligned element map with sizes hx, hy, hz.
 *
 * The volumetric kernels apply only the element-local tensor-product operator.
 * The cartesian_apply kernels apply the volumetric operator and then perform
 * structured DSS on the output, giving the continuous assembled action in
 * duplicated BXYZE storage.
 */
typedef struct sfem_dss_bxyze_laplacian_cartesian_layout : sfem_dss_bxyze_layout_t {
    ptrdiff_t nx;
    ptrdiff_t ny;
    ptrdiff_t nz;
} sfem_dss_bxyze_laplacian_cartesian_layout_t;

int sfem_dss_bxyze_laplacian_cartesian_create_layout(int level,
                                                     ptrdiff_t nx,
                                                     ptrdiff_t ny,
                                                     ptrdiff_t nz,
                                                     int element_block_size,
                                                     int execution_block_size,
                                                     sfem_dss_bxyze_laplacian_cartesian_layout_t *layout);

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_layout(
        const sfem_dss_bxyze_laplacian_cartesian_layout_t *const layout,
        real_t hx,
        real_t hy,
        real_t hz,
        real_t kappa,
        const real_t *const x_bxyze,
        real_t *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_apply_layout(
        const sfem_dss_bxyze_laplacian_cartesian_layout_t *const layout,
        real_t hx,
        real_t hy,
        real_t hz,
        real_t kappa,
        const real_t *const x_bxyze,
        real_t *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply(int level,
                                                        ptrdiff_t nx,
                                                        ptrdiff_t ny,
                                                        ptrdiff_t nz,
                                                        int element_block_size,
                                                        real_t hx,
                                                        real_t hy,
                                                        real_t hz,
                                                        real_t kappa,
                                                        const real_t *const x_bxyze,
                                                        real_t *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_blocked(int level,
                                                                ptrdiff_t nx,
                                                                ptrdiff_t ny,
                                                                ptrdiff_t nz,
                                                                int element_block_size,
                                                                int execution_block_size,
                                                                real_t hx,
                                                                real_t hy,
                                                                real_t hz,
                                                                real_t kappa,
                                                                const real_t *const x_bxyze,
                                                                real_t *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_apply(int level,
                                             ptrdiff_t nx,
                                             ptrdiff_t ny,
                                             ptrdiff_t nz,
                                             int element_block_size,
                                             real_t hx,
                                             real_t hy,
                                             real_t hz,
                                             real_t kappa,
                                             const real_t *const x_bxyze,
                                             real_t *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_apply_blocked(int level,
                                                     ptrdiff_t nx,
                                                     ptrdiff_t ny,
                                                     ptrdiff_t nz,
                                                     int element_block_size,
                                                     int execution_block_size,
                                                     real_t hx,
                                                     real_t hy,
                                                     real_t hz,
                                                     real_t kappa,
                                                     const real_t *const x_bxyze,
                                                     real_t *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_real32(int level,
                                                               ptrdiff_t nx,
                                                               ptrdiff_t ny,
                                                               ptrdiff_t nz,
                                                               int element_block_size,
                                                               float hx,
                                                               float hy,
                                                               float hz,
                                                               float kappa,
                                                               const float *const x_bxyze,
                                                               float *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_apply_real32(int level,
                                                    ptrdiff_t nx,
                                                    ptrdiff_t ny,
                                                    ptrdiff_t nz,
                                                    int element_block_size,
                                                    float hx,
                                                    float hy,
                                                    float hz,
                                                    float kappa,
                                                    const float *const x_bxyze,
                                                    float *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_volumetric_apply_real64(int level,
                                                               ptrdiff_t nx,
                                                               ptrdiff_t ny,
                                                               ptrdiff_t nz,
                                                               int element_block_size,
                                                               double hx,
                                                               double hy,
                                                               double hz,
                                                               double kappa,
                                                               const double *const x_bxyze,
                                                               double *const y_bxyze);

int sfem_dss_bxyze_laplacian_cartesian_apply_real64(int level,
                                                    ptrdiff_t nx,
                                                    ptrdiff_t ny,
                                                    ptrdiff_t nz,
                                                    int element_block_size,
                                                    double hx,
                                                    double hy,
                                                    double hz,
                                                    double kappa,
                                                    const double *const x_bxyze,
                                                    double *const y_bxyze);
