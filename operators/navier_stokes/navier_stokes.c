#include "navier_stokes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#include "tet10_navier_stokes.h"
#include "tri6_navier_stokes.h"

void navier_stokes_assemble_value_aos(const enum ElemType               element_type,
                                      const ptrdiff_t                   nelements,
                                      const ptrdiff_t                   nnodes,
                                      idx_t **const SFEM_RESTRICT       elems,
                                      geom_t **const SFEM_RESTRICT      xyz,
                                      const real_t                      nu,
                                      const real_t                      rho,
                                      const real_t *const SFEM_RESTRICT u,
                                      real_t *const SFEM_RESTRICT       value) {
    SFEM_ERROR("IMPLEMENT ME!\n");
}

void navier_stokes_assemble_gradient_aos(const enum ElemType               element_type,
                                         const ptrdiff_t                   nelements,
                                         const ptrdiff_t                   nnodes,
                                         idx_t **const SFEM_RESTRICT       elems,
                                         geom_t **const SFEM_RESTRICT      xyz,
                                         const real_t                      nu,
                                         const real_t                      rho,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT       values) {
    SFEM_ERROR("IMPLEMENT ME!\n");
}

void navier_stokes_assemble_hessian_aos(const enum ElemType                element_type,
                                        const ptrdiff_t                    nelements,
                                        const ptrdiff_t                    nnodes,
                                        idx_t **const SFEM_RESTRICT        elems,
                                        geom_t **const SFEM_RESTRICT       xyz,
                                        const real_t                       nu,
                                        const real_t                       rho,
                                        const count_t *const SFEM_RESTRICT rowptr,
                                        const idx_t *const SFEM_RESTRICT   colidx,
                                        real_t *const SFEM_RESTRICT        values) {
    SFEM_ERROR("IMPLEMENT ME!\n");
}

void navier_stokes_apply_aos(const enum ElemType               element_type,
                             const ptrdiff_t                   nelements,
                             const ptrdiff_t                   nnodes,
                             idx_t **const SFEM_RESTRICT       elems,
                             geom_t **const SFEM_RESTRICT      xyz,
                             const real_t                      nu,
                             const real_t                      rho,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT       values) {
    SFEM_ERROR("IMPLEMENT ME!\n");
}

void navier_stokes_mixed_explict_momentum_tentative(const enum ElemType          element_type,
                                                    const ptrdiff_t              nelements,
                                                    const ptrdiff_t              nnodes,
                                                    idx_t **const                elems,
                                                    geom_t **const               points,
                                                    const real_t                 dt,
                                                    const real_t                 nu,
                                                    const real_t                 convonoff,
                                                    real_t **const SFEM_RESTRICT vel,
                                                    real_t **const SFEM_RESTRICT f) {
    switch (element_type) {
        case TRI6: {
            tri6_explict_momentum_tentative(nelements, nnodes, elems, points, dt, nu, convonoff, vel, f);
            break;
        }
        case TET10: {
            tet10_explict_momentum_tentative(nelements, nnodes, elems, points, dt, nu, convonoff, vel, f);
            break;
        }
        default: {
            assert(0);
            fprintf(stderr, "Not implemented %s:%d", __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);
            break;
        }
    }
}

void navier_stokes_mixed_divergence(const enum ElemType          high_order,
                                    const enum ElemType          low_order,
                                    const ptrdiff_t              nelements,
                                    const ptrdiff_t              nnodes,
                                    idx_t **const                elems,
                                    geom_t **const               points,
                                    const real_t                 dt,
                                    const real_t                 rho,
                                    const real_t                 nu,
                                    real_t **const SFEM_RESTRICT vel,
                                    real_t *const SFEM_RESTRICT  f) {
    switch (high_order) {
        case TRI6: {
            assert(low_order == TRI3);
            tri3_tri6_divergence(nelements, nnodes, elems, points, dt, rho, nu, vel, f);
            break;
        }
        case TET10: {
            tet4_tet10_divergence(nelements, nnodes, elems, points, dt, rho, nu, vel, f);
            break;
        }
        default: {
            assert(0);
            fprintf(stderr, "Not implemented %s:%d", __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);
            break;
        }
    }
}

void navier_stokes_mixed_correction(const enum ElemType          high_order,
                                    const enum ElemType          low_order,
                                    const ptrdiff_t              nelements,
                                    const ptrdiff_t              nnodes,
                                    idx_t **const                elems,
                                    geom_t **const               points,
                                    const real_t                 dt,
                                    const real_t                 rho,
                                    real_t *const SFEM_RESTRICT  p,
                                    real_t **const SFEM_RESTRICT values) {
    switch (high_order) {
        case TRI6: {
            tri6_tri3_correction(nelements, nnodes, elems, points, dt, rho, p, values);
            break;
        }
        case TET10: {
            tet10_tet4_correction(nelements, nnodes, elems, points, dt, rho, p, values);
            break;
        }
        default: {
            SFEM_ERROR("Not implemented %s:%d", __FILE__, __LINE__);
            break;
        }
    }
}

void navier_stokes_momentum_lhs_scalar_crs(const enum ElemType                element_type,
                                           const ptrdiff_t                    nelements,
                                           const ptrdiff_t                    nnodes,
                                           idx_t **const                      elems,
                                           geom_t **const                     points,
                                           const real_t                       dt,
                                           const real_t                       nu,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT   colidx,
                                           real_t *const SFEM_RESTRICT        values) {
    switch (element_type) {
        case TRI6: {
            tri6_momentum_lhs_scalar_crs(nelements, nnodes, elems, points, dt, nu, rowptr, colidx, values);
            break;
        }
        case TET10: {
            tet10_momentum_lhs_scalar_crs(nelements, nnodes, elems, points, dt, nu, rowptr, colidx, values);
            break;
        }
        default: {
            SFEM_ERROR("Not implemented %s:%d", __FILE__, __LINE__);
            break;
        }
    }
}
