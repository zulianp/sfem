#include "cu_linear_elasticity.hpp"

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "cu_hex8_linear_elasticity.hpp"
#include "cu_macro_tet4_linear_elasticity.hpp"
#include "cu_sshex8_linear_elasticity.hpp"
#include "cu_tet10_linear_elasticity.hpp"
#include "cu_tet4_linear_elasticity.hpp"

#include <mpi.h>
#include <stdio.h>

extern int cu_linear_elasticity_apply(const smesh::ElemType           element_type,
                                      const ptrdiff_t                 nelements,
                                      idx_t **const SFEM_RESTRICT     elements,
                                      const ptrdiff_t                 jacobian_stride,
                                      const void *const SFEM_RESTRICT jacobian_adjugate,
                                      const void *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t                    mu,
                                      const real_t                    lambda,
                                      const enum smesh::PrimitiveType real_type,
                                      const real_t *const             d_x,
                                      real_t *const                   d_y,
                                      void                           *stream) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        return cu_affine_sshex8_linear_elasticity_apply(level,
                                                        nelements,
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

    switch (element_type) {
        case smesh::TET4: {
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
        case smesh::MACRO_TET4: {
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
        case smesh::TET10: {
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
        case smesh::HEX8: {
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

extern int cu_linear_elasticity_diag(const smesh::ElemType           element_type,
                                     const ptrdiff_t                 nelements,
                                     idx_t **const SFEM_RESTRICT     elements,
                                     const ptrdiff_t                 jacobian_stride,
                                     const void *const SFEM_RESTRICT jacobian_adjugate,
                                     const void *const SFEM_RESTRICT jacobian_determinant,
                                     const real_t                    mu,
                                     const real_t                    lambda,
                                     const enum smesh::PrimitiveType real_type,
                                     real_t *const                   d_t,
                                     void                           *stream) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        return cu_affine_sshex8_linear_elasticity_diag(level,
                                                       nelements,
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

    switch (element_type) {
        case smesh::TET4: {
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
        case smesh::MACRO_TET4: {
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
        case smesh::TET10: {
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

int cu_linear_elasticity_bsr(const smesh::ElemType              element_type,
                             const ptrdiff_t                    nelements,
                             idx_t **const SFEM_RESTRICT        elements,
                             const ptrdiff_t                    jacobian_stride,
                             const void *const SFEM_RESTRICT    jacobian_adjugate,
                             const void *const SFEM_RESTRICT    jacobian_determinant,
                             const real_t                       mu,
                             const real_t                       lambda,
                             const enum smesh::PrimitiveType    real_type,
                             const count_t *const SFEM_RESTRICT rowptr,
                             const idx_t *const SFEM_RESTRICT   colidx,
                             void *const SFEM_RESTRICT          values,
                             void                              *stream) {
    switch (element_type) {
        case smesh::HEX8: {
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

int cu_linear_elasticity_block_diag_sym_aos(const smesh::ElemType           element_type,
                                            const ptrdiff_t                 nelements,
                                            idx_t **const SFEM_RESTRICT     elements,
                                            const ptrdiff_t                 jacobian_stride,
                                            const void *const SFEM_RESTRICT jacobian_adjugate,
                                            const void *const SFEM_RESTRICT jacobian_determinant,
                                            const real_t                    mu,
                                            const real_t                    lambda,
                                            const enum smesh::PrimitiveType real_type,
                                            void *const                     out,
                                            void                           *stream) {
    size_t nbytes = 0;
    if (real_type == smesh::SMESH_DEFAULT) {
        nbytes = sizeof(real_t);
    } else {
        nbytes = smesh::num_bytes(real_type);
    }

    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        return cu_affine_sshex8_linear_elasticity_block_diag_sym(level,
                                                                 nelements,
                                                                 elements,
                                                                 jacobian_stride,
                                                                 jacobian_adjugate,
                                                                 jacobian_determinant,
                                                                 mu,
                                                                 lambda,
                                                                 6,
                                                                 real_type,
                                                                 (char *)out + 0 * nbytes,
                                                                 (char *)out + 1 * nbytes,
                                                                 (char *)out + 2 * nbytes,
                                                                 (char *)out + 3 * nbytes,
                                                                 (char *)out + 4 * nbytes,
                                                                 (char *)out + 5 * nbytes,
                                                                 stream);
    }

    switch (element_type) {
        case smesh::HEX8: {
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
                                                                   (char *)out + 0 * nbytes,
                                                                   (char *)out + 1 * nbytes,
                                                                   (char *)out + 2 * nbytes,
                                                                   (char *)out + 3 * nbytes,
                                                                   (char *)out + 4 * nbytes,
                                                                   (char *)out + 5 * nbytes,
                                                                   stream);
        }
        default: {
            SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type);
            return SFEM_FAILURE;
        }
    }
}
