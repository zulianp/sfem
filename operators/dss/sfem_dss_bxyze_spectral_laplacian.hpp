#pragma once

#include "sfem_base.hpp"

#include <stddef.h>

/*
 * Cell-wise BXYZE wrapper around SFEM's existing Lagrange spectral-hex
 * Laplacian microkernel.
 *
 * This path keeps the local element operation identical to
 * SpectralElementLaplacian's Lagrange branch:
 *
 *   lagrange_hex_laplacian_apply<N, Q, scalar_t>(S, D, fff, qw, xe, ye)
 *
 * but replaces global gather/scatter with:
 *
 *   BXYZE local pack -> same microkernel -> BXYZE local unpack -> DSS
 *
 * It implements the scalar execution variant: BXYZE storage can use B > 1, but
 * the microkernel is still called for one element at a time (B_exec = 1).
 */
int sfem_dss_bxyze_lagrange_spectral_laplacian_volumetric_apply(int level,
                                                                ptrdiff_t nelements,
                                                                int element_block_size,
                                                                const idx_t *const *const elements,
                                                                const geom_t *const *const points,
                                                                const real_t *const x_bxyze,
                                                                real_t *const y_bxyze);

int sfem_dss_bxyze_lagrange_spectral_laplacian_apply(int level,
                                                     ptrdiff_t nx,
                                                     ptrdiff_t ny,
                                                     ptrdiff_t nz,
                                                     int element_block_size,
                                                     const idx_t *const *const elements,
                                                     const geom_t *const *const points,
                                                     const real_t *const x_bxyze,
                                                     real_t *const y_bxyze);
