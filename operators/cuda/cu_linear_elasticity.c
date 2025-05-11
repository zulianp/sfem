#include "cu_linear_elasticity.h"

#include "sfem_base.h"
#include "sfem_defs.h"

#include "cu_hex8_linear_elasticity.h"
#include "cu_macro_tet4_linear_elasticity.h"
#include "cu_tet10_linear_elasticity.h"
#include "cu_tet4_linear_elasticity.h"

#include <mpi.h>
#include <stdio.h>

extern int cu_linear_elasticity_apply(const enum ElemType             element_type,
                                      const ptrdiff_t                 nelements,
                                      idx_t **const SFEM_RESTRICT     elements,
                                      const ptrdiff_t                 jacobian_stride,
                                      const void *const SFEM_RESTRICT jacobian_adjugate,
                                      const void *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t                    mu,
                                      const real_t                    lambda,
                                      const enum RealType             real_type,
                                      const real_t *const             d_x,
                                      real_t *const                   d_y,
                                      void                           *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_linear_elasticity_apply(nelements,
                                                   elements,
                                                   jacobian_stride,
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
                                                         elements,
                                                         jacobian_stride,
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
                                                    elements,
                                                    jacobian_stride,
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
                                                          elements,
                                                          jacobian_stride,
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
            SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type);
            return SFEM_FAILURE;
        }
    }
}

extern int cu_linear_elasticity_diag(const enum ElemType             element_type,
                                     const ptrdiff_t                 nelements,
                                     idx_t **const SFEM_RESTRICT     elements,
                                     const ptrdiff_t                 jacobian_stride,
                                     const void *const SFEM_RESTRICT jacobian_adjugate,
                                     const void *const SFEM_RESTRICT jacobian_determinant,
                                     const real_t                    mu,
                                     const real_t                    lambda,
                                     const enum RealType             real_type,
                                     real_t *const                   d_t,
                                     void                           *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_linear_elasticity_diag(nelements,
                                                  elements,
                                                  jacobian_stride,
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
                                                        elements,
                                                        jacobian_stride,
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
                                                   elements,
                                                   jacobian_stride,
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
            SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_linear_elasticity_bsr(const enum ElemType                element_type,
                             const ptrdiff_t                    nelements,
                             idx_t **const SFEM_RESTRICT        elements,
                             const ptrdiff_t                    jacobian_stride,
                             const void *const SFEM_RESTRICT    jacobian_adjugate,
                             const void *const SFEM_RESTRICT    jacobian_determinant,
                             const real_t                       mu,
                             const real_t                       lambda,
                             const enum RealType                real_type,
                             const count_t *const SFEM_RESTRICT rowptr,
                             const idx_t *const SFEM_RESTRICT   colidx,
                             void *const SFEM_RESTRICT          values,
                             void                              *stream) {
    switch (element_type) {
        case HEX8: {
            return cu_affine_hex8_linear_elasticity_bsr(nelements,
                                                        elements,
                                                        jacobian_stride,
                                                        jacobian_adjugate,
                                                        jacobian_determinant,
                                                        mu,
                                                        lambda,
                                                        real_type,
                                                        rowptr,
                                                        colidx,
                                                        values,
                                                        stream);
        }
        default: {
            SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_linear_elasticity_block_diag_sym_aos(const enum ElemType             element_type,
                                            const ptrdiff_t                 nelements,
                                            idx_t **const SFEM_RESTRICT     elements,
                                            const ptrdiff_t                 jacobian_stride,
                                            const void *const SFEM_RESTRICT jacobian_adjugate,
                                            const void *const SFEM_RESTRICT jacobian_determinant,
                                            const real_t                    mu,
                                            const real_t                    lambda,
                                            const enum RealType             real_type,
                                            void *const                     out,
                                            void                           *stream) {
    switch (element_type) {
        case HEX8: {
            return cu_affine_hex8_linear_elasticity_block_diag_sym(nelements,
                                                                   elements,
                                                                   jacobian_stride,
                                                                   jacobian_adjugate,
                                                                   jacobian_determinant,
                                                                   mu,
                                                                   lambda,
                                                                   6,
                                                                   real_type,
                                                                   // Offset for AoS to SoA style function
                                                                   out + 0 * real_type_size(real_type),
                                                                   out + 1 * real_type_size(real_type),
                                                                   out + 2 * real_type_size(real_type),
                                                                   out + 3 * real_type_size(real_type),
                                                                   out + 4 * real_type_size(real_type),
                                                                   out + 5 * real_type_size(real_type),
                                                                   stream);
        }
        default: {
            SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type);
            return SFEM_FAILURE;
        }
    }
}
