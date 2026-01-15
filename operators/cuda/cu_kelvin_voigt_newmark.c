#include "sfem_base.h"
#include "sfem_defs.h"

#include "cu_hex8_kelvin_voigt_newmark.h"

#include <mpi.h>
#include <stdio.h>

extern int cu_kelvin_voigt_newmark_apply(const enum ElemType             element_type,
                                      const ptrdiff_t                 nelements,
                                      idx_t **const SFEM_RESTRICT     elements,
                                      const ptrdiff_t                 jacobian_stride,
                                      const void *const SFEM_RESTRICT jacobian_adjugate,
                                      const void *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t                    k,
                                      const real_t                    K,
                                      const real_t                    eta,
                                      const real_t                    rho,
                                      const enum RealType             real_type,
                                      const real_t *const             d_x,
                                      const real_t *const             d_v,
                                      const real_t *const             d_a,
                                      real_t *const                   d_y,
                                      void                           *stream) {
    switch (element_type) {
        case HEX8: {
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
