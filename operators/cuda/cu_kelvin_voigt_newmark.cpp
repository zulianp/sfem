#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "cu_kelvin_voigt_newmark.hpp"
#include "cu_hex8_kelvin_voigt_newmark.hpp"
#include "cu_sshex8_kelvin_voigt_newmark.hpp"

#include <mpi.h>
#include <stdio.h>

int cu_kelvin_voigt_newmark_apply(const smesh::ElemType             element_type,
                                      const ptrdiff_t                 nelements,
                                      idx_t **const SFEM_RESTRICT     elements,
                                      const ptrdiff_t                 jacobian_stride,
                                      const void *const SFEM_RESTRICT jacobian_adjugate,
                                      const void *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t                    k,
                                      const real_t                    K,
                                      const real_t                    eta,
                                      const real_t                    rho,
                                      const real_t                    dt,
                                      const real_t                    gamma,
                                      const real_t                    beta,
                                      const enum smesh::PrimitiveType             real_type,
                                      const real_t *const             d_x,
                                      const real_t *const             d_v,
                                      const real_t *const             d_a,
                                      real_t *const                   d_y,
                                      void                           *stream) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        return cu_affine_sshex8_kelvin_voigt_newmark_apply(level,
                                                           nelements,
                                                           elements,
                                                           jacobian_stride,
                                                           jacobian_adjugate,
                                                           jacobian_determinant,
                                                           k,
                                                           K,
                                                           eta,
                                                           rho,
                                                           dt,
                                                           gamma,
                                                           beta,
                                                           real_type,
                                                           3,
                                                           d_x,
                                                           &d_x[1],
                                                           &d_x[2],
                                                           d_v,
                                                           &d_v[1],
                                                           &d_v[2],
                                                           d_a,
                                                           &d_a[1],
                                                           &d_a[2],
                                                           3,
                                                           d_y,
                                                           &d_y[1],
                                                           &d_y[2],
                                                           stream);
    }

    switch (element_type) {
        case smesh::HEX8: {
            return cu_affine_hex8_kelvin_voigt_newmark_apply(nelements,
                                                          elements,
                                                          jacobian_stride,
                                                          jacobian_adjugate,
                                                          jacobian_determinant,
                                                          k,
                                                          K,
                                                          eta,
                                                          rho,
                                                          real_type,
                                                          3,
                                                          d_x,
                                                          &d_x[1],
                                                          &d_x[2],
                                                          d_v,
                                                          &d_v[1],
                                                          &d_v[2],
                                                          d_a,
                                                          &d_a[1],
                                                          &d_a[2],
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

int cu_kelvin_voigt_newmark_diag(const smesh::ElemType             element_type,
                                 const ptrdiff_t                 nelements,
                                 idx_t **const SFEM_RESTRICT     elements,
                                 const ptrdiff_t                 jacobian_stride,
                                 const void *const SFEM_RESTRICT jacobian_adjugate,
                                 const void *const SFEM_RESTRICT jacobian_determinant,
                                 const real_t                    k,
                                 const real_t                    K,
                                 const real_t                    eta,
                                 const real_t                    rho,
                                 const real_t                    dt,
                                 const real_t                    gamma,
                                 const real_t                    beta,
                                 const enum smesh::PrimitiveType real_type,
                                 real_t *const                   d_t,
                                 void                           *stream) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        return cu_affine_sshex8_kelvin_voigt_newmark_diag(level,
                                                          nelements,
                                                          elements,
                                                          jacobian_stride,
                                                          jacobian_adjugate,
                                                          jacobian_determinant,
                                                          k,
                                                          K,
                                                          eta,
                                                          rho,
                                                          dt,
                                                          gamma,
                                                          beta,
                                                          real_type,
                                                          3,
                                                          d_t,
                                                          &d_t[1],
                                                          &d_t[2],
                                                          stream);
    }

    SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type);
    return SFEM_FAILURE;
}
