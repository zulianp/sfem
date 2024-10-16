#include "cu_linear_elasticity.h"

#include "cu_hex8_linear_elasticity.h"
#include "cu_macro_tet4_linear_elasticity.h"
#include "cu_tet10_linear_elasticity.h"
#include "cu_tet4_linear_elasticity.h"

#include <mpi.h>
#include <stdio.h>

extern int cu_linear_elasticity_apply(const enum ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t stride,  // Stride for elements and fff
                                      const idx_t *const SFEM_RESTRICT elements,
                                      const void *const SFEM_RESTRICT jacobian_adjugate,
                                      const void *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t mu,
                                      const real_t lambda,
                                      const enum RealType real_type,
                                      const real_t *const d_x,
                                      real_t *const d_y,
                                      void *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_linear_elasticity_apply(nelements,
                                                   stride,
                                                   elements,
                                                   jacobian_adjugate,
                                                   jacobian_determinant,
                                                   mu,
                                                   lambda,
                                                   real_type,
                                                   3,
                                                   d_x,
                                                   &d_x[1],
                                                   &d_x[2],
                                                   3,
                                                   d_y,
                                                   &d_y[1],
                                                   &d_y[2],
                                                   stream);
        }
        case MACRO_TET4: {
            return cu_macro_tet4_linear_elasticity_apply(nelements,
                                                         stride,
                                                         elements,
                                                         jacobian_adjugate,
                                                         jacobian_determinant,
                                                         mu,
                                                         lambda,
                                                         real_type,
                                                         3,
                                                         d_x,
                                                         &d_x[1],
                                                         &d_x[2],
                                                         3,
                                                         d_y,
                                                         &d_y[1],
                                                         &d_y[2],
                                                         stream);
        }
        case TET10: {
            return cu_tet10_linear_elasticity_apply(nelements,
                                                    stride,
                                                    elements,
                                                    jacobian_adjugate,
                                                    jacobian_determinant,
                                                    mu,
                                                    lambda,
                                                    real_type,
                                                    3,
                                                    d_x,
                                                    &d_x[1],
                                                    &d_x[2],
                                                    3,
                                                    d_y,
                                                    &d_y[1],
                                                    &d_y[2],
                                                    stream);
        }
        case HEX8: {
            return cu_affine_hex8_linear_elasticity_apply(nelements,
                                                          stride,
                                                          elements,
                                                          jacobian_adjugate,
                                                          jacobian_determinant,
                                                          mu,
                                                          lambda,
                                                          real_type,
                                                          3,
                                                          d_x,
                                                          &d_x[1],
                                                          &d_x[2],
                                                          3,
                                                          d_y,
                                                          &d_y[1],
                                                          &d_y[2],
                                                          stream);
        }

        default: {
            fprintf(stderr,
                    "Invalid element type %d\n (%s %s:%d)",
                    element_type,
                    __FUNCTION__,
                    __FILE__,
                    __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
}

extern int cu_linear_elasticity_diag(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t stride,  // Stride for elements and fff
                                     const idx_t *const SFEM_RESTRICT elements,
                                     const void *const SFEM_RESTRICT jacobian_adjugate,
                                     const void *const SFEM_RESTRICT jacobian_determinant,
                                     const real_t mu,
                                     const real_t lambda,
                                     const enum RealType real_type,
                                     real_t *const d_t,
                                     void *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_linear_elasticity_diag(nelements,
                                                  stride,
                                                  elements,
                                                  jacobian_adjugate,
                                                  jacobian_determinant,
                                                  mu,
                                                  lambda,
                                                  real_type,
                                                  3,
                                                  d_t,
                                                  &d_t[1],
                                                  &d_t[2],
                                                  stream);
        }
        case MACRO_TET4: {
            return cu_macro_tet4_linear_elasticity_diag(nelements,
                                                        stride,
                                                        elements,
                                                        jacobian_adjugate,
                                                        jacobian_determinant,
                                                        mu,
                                                        lambda,
                                                        real_type,
                                                        3,
                                                        d_t,
                                                        &d_t[1],
                                                        &d_t[2],
                                                        stream);
        }
        case TET10: {
            return cu_tet10_linear_elasticity_diag(nelements,
                                                   stride,
                                                   elements,
                                                   jacobian_adjugate,
                                                   jacobian_determinant,
                                                   mu,
                                                   lambda,
                                                   real_type,
                                                   3,
                                                   d_t,
                                                   &d_t[1],
                                                   &d_t[2],
                                                   stream);
        }
        default: {
            fprintf(stderr,
                    "Invalid element type %d\n (%s %s:%d)",
                    element_type,
                    __FUNCTION__,
                    __FILE__,
                    __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
}
