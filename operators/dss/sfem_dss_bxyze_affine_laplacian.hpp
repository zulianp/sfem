#pragma once

#include "sfem_dss_bxyze.hpp"

struct sfem_dss_bxyze_laplacian_affine_volumetric;
typedef int (*sfem_dss_bxyze_laplacian_affine_volumetric_kernel_t)(
        const sfem_dss_bxyze_laplacian_affine_volumetric *const op,
        const scalar_t *const S,
        const scalar_t *const D,
        const scalar_t *const qw,
        const scalar_t *const geometry,
        const real_t *const x_bxyze,
        real_t *const y_bxyze,
        scalar_t *const workspace);

/* Prepared execution metadata. Treat as read-only after a successful setup;
 * rerun setup to change the layout or quadrature. Does not own any buffers.
 */
typedef struct sfem_dss_bxyze_laplacian_affine_volumetric {
    sfem_dss_bxyze_layout_t layout;
    int Q;
    ptrdiff_t workspace_stride;
    sfem_dss_bxyze_laplacian_affine_volumetric_kernel_t kernel;
} sfem_dss_bxyze_laplacian_affine_volumetric_t;

/* Preallocated scalar_t values per OpenMP thread, including a separation gap.
 * gx, gy, gz / weighted fluxes (Q^3 * B_exec each), then shared gradient and
 * transpose scratch. Both contractions reuse the same scratch allocation.
 * Allocate omp_get_max_threads() slices before apply (one without OpenMP).
 * Supports p=2,4,8; Q=p+1 or 3*p; B_exec=1,2,4,8,16; full storage blocks.
 * Returns -1 for unsupported dimensions or inconsistent layout metadata.
 */
ptrdiff_t sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(
        const sfem_dss_bxyze_layout_t *const layout, const int Q);

/* Validate dimensions, copy the layout, compute workspace_stride and select
 * the N/Q/B_exec/B_save specialization once. Storage sizes 1,2,4,8,16,32,64
 * use constant strides; other valid sizes retain the generic storage path.
 * No allocation or basis/geometry setup.
 * On failure, clears op so it cannot retain an earlier kernel.
 */
int sfem_dss_bxyze_laplacian_affine_volumetric_setup(
        const sfem_dss_bxyze_layout_t *const layout,
        const int Q,
        sfem_dss_bxyze_laplacian_affine_volumetric_t *const op);

/* Affine cellwise Laplacian: y_e = G^T [w_q H_e] G x_e, overwriting y_bxyze.
 * Requires a successful setup. No dimension dispatch or workspace sizing here.
 * Allocate omp_get_max_threads() * op->workspace_stride scalar_t values before
 * apply (one slice without OpenMP), for the thread count used by each call.
 * Each thread reuses its workspace slice for all its batches. No allocations
 * or scratch resets. The output contains local nodal contributions, before DSS.
 * qw contains Q one-dimensional weights. Geometry is [block][6][B_save], ordered
 * H00,H01,H02,H11,H12,H22, with H = det(J) J^-1 J^-T (no quadrature weights).
 * Returns SFEM_SUCCESS on completion, including an empty mesh, or SFEM_FAILURE
 * for missing required pointers / setup. Inputs, output and workspace must be
 * disjoint. Each output node is written once; no prior zeroing is needed.
 */
int sfem_dss_bxyze_laplacian_affine_volumetric_apply(
        const sfem_dss_bxyze_laplacian_affine_volumetric_t *const op,
        const scalar_t *const S,
        const scalar_t *const D,
        const scalar_t *const qw,
        const scalar_t *const geometry,
        const real_t *const x_bxyze,
        real_t *const y_bxyze,
        scalar_t *const workspace);
