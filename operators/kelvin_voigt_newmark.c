#include "kelvin_voigt_newmark.h"

#include "hex8_kelvin_voigt_newmark.h"

#include <assert.h>
#include <stdio.h>

//////////////////////////////////////////////// SOA //////////////////////////////////////////////////////

int kelvin_voigt_newmark_apply_adjugate_soa(const enum ElemType                   element_type,
                                            const ptrdiff_t                       nelements,
                                            const ptrdiff_t                       nnodes,
                                            idx_t **const SFEM_RESTRICT           elements,
                                            const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                            const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                            const real_t                          dt,
                                            const real_t                          gamma,
                                            const real_t                          beta,
                                            const real_t                          k,
                                            const real_t                          K,
                                            const real_t                          eta,
                                            const real_t                          rho,
                                            const real_t *const SFEM_RESTRICT     u,
                                            real_t *const SFEM_RESTRICT           values) {
    switch (element_type) {
        case HEX8: {
            return affine_hex8_kelvin_voigt_newmark_lhs_apply(nelements,
                                                              nnodes,
                                                              elements,
                                                              jacobian_adjugate,
                                                              jacobian_determinant,
                                                              dt,
                                                              gamma,
                                                              beta,
                                                              k,
                                                              K,
                                                              eta,
                                                              rho,
                                                              1,
                                                              &u[0],
                                                              &u[1],
                                                              &u[2],
                                                              1,
                                                              &values[0],
                                                              &values[1],
                                                              &values[2]);
        }
        default: {
            SFEM_ERROR("linear_elasticity_apply_adjugate_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int kelvin_voigt_newmark_gradient_soa(const enum ElemType                   element_type,
                                      const ptrdiff_t                       nelements,
                                      const ptrdiff_t                       nnodes,
                                      idx_t **const SFEM_RESTRICT           elements,
                                      const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                      const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t                          k,
                                      const real_t                          K,
                                      const real_t                          eta,
                                      const real_t                          rho,
                                      const real_t *const SFEM_RESTRICT     u,
                                      const real_t *const SFEM_RESTRICT     vx,
                                      const real_t *const SFEM_RESTRICT     vy,
                                      const real_t *const SFEM_RESTRICT     vz,
                                      const real_t *const SFEM_RESTRICT     ax,
                                      const real_t *const SFEM_RESTRICT     ay,
                                      const real_t *const SFEM_RESTRICT     az,
                                      real_t *const SFEM_RESTRICT           values) {
    // return linear_elasticity_apply_aos(element_type, nelements, nnodes, elements, points, mu, lambda, u, values);
    switch (element_type) {
        case HEX8: {
            return affine_hex8_kelvin_voigt_newmark_gradient(nelements,
                                                             nnodes,
                                                             elements,
                                                             jacobian_adjugate,
                                                             jacobian_determinant,
                                                             k,
                                                             K,
                                                             eta,
                                                             rho,
                                                             1,
                                                             &u[0],
                                                             &u[1],
                                                             &u[2],
                                                             vx,
                                                             vy,
                                                             vz,
                                                             ax,
                                                             ay,
                                                             az,
                                                             1,
                                                             &values[0],
                                                             &values[1],
                                                             &values[2]);
        }
        default: {
            SFEM_ERROR("linear_elasticity_apply_adjugate_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

//////////////////////////////////////////////// AOS //////////////////////////////////////////////////////

int kelvin_voigt_newmark_apply_adjugate_aos(const enum ElemType                   element_type,
                                            const ptrdiff_t                       nelements,
                                            const ptrdiff_t                       nnodes,
                                            idx_t **const SFEM_RESTRICT           elements,
                                            const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                            const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                            const real_t                          dt,
                                            const real_t                          gamma,
                                            const real_t                          beta,
                                            const real_t                          k,
                                            const real_t                          K,
                                            const real_t                          eta,
                                            const real_t                          rho,
                                            const real_t *const SFEM_RESTRICT     u,
                                            real_t *const SFEM_RESTRICT           values) {
    switch (element_type) {
        case HEX8: {
            return affine_hex8_kelvin_voigt_newmark_lhs_apply(nelements,
                                                              nnodes,
                                                              elements,
                                                              jacobian_adjugate,
                                                              jacobian_determinant,
                                                              dt,
                                                              gamma,
                                                              beta,
                                                              k,
                                                              K,
                                                              eta,
                                                              rho,
                                                              3,
                                                              &u[0],
                                                              &u[1],
                                                              &u[2],
                                                              3,
                                                              &values[0],
                                                              &values[1],
                                                              &values[2]);
        }
        default: {
            SFEM_ERROR("linear_elasticity_apply_adjugate_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int kelvin_voigt_newmark_gradient_aos(const enum ElemType                   element_type,
                                      const ptrdiff_t                       nelements,
                                      const ptrdiff_t                       nnodes,
                                      idx_t **const SFEM_RESTRICT           elements,
                                      const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                      const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                      const real_t                          k,
                                      const real_t                          K,
                                      const real_t                          eta,
                                      const real_t                          rho,
                                      const real_t *const SFEM_RESTRICT     u,
                                      const real_t *const SFEM_RESTRICT     vx,
                                      const real_t *const SFEM_RESTRICT     vy,
                                      const real_t *const SFEM_RESTRICT     vz,
                                      const real_t *const SFEM_RESTRICT     ax,
                                      const real_t *const SFEM_RESTRICT     ay,
                                      const real_t *const SFEM_RESTRICT     az,
                                      real_t *const SFEM_RESTRICT           values) {
    switch (element_type) {
        case HEX8: {
            return affine_hex8_kelvin_voigt_newmark_gradient(nelements,
                                                             nnodes,
                                                             elements,
                                                             jacobian_adjugate,
                                                             jacobian_determinant,
                                                             k,
                                                             K,
                                                             eta,
                                                             rho,
                                                             3,
                                                             &u[0],
                                                             &u[1],
                                                             &u[2],
                                                             vx,
                                                             vy,
                                                             vz,
                                                             ax,
                                                             ay,
                                                             az,
                                                             3,
                                                             &values[0],
                                                             &values[1],
                                                             &values[2]);
        }
        default: {
            SFEM_ERROR("kelvin_voigt_newmark_gradient_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int kelvin_voigt_newmark_assemble_diag_aos(const enum ElemType          element_type,
                                           const ptrdiff_t              nelements,
                                           const ptrdiff_t              nnodes,
                                           idx_t **const SFEM_RESTRICT  elems,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const real_t                 beta,
                                           const real_t                 gamma,
                                           const real_t                 dt,
                                           const real_t                 k,
                                           const real_t                 K,
                                           const real_t                 eta,
                                           const real_t                 rho,
                                           real_t *const SFEM_RESTRICT  values) {
    switch (element_type) {
        case HEX8: {
            return affine_hex8_kelvin_voigt_newmark_diag(
                    nelements, nnodes, elems, xyz, beta, gamma, dt, k, K, eta, rho, 3, &values[0], &values[1], &values[2]);
        }
        default: {
            SFEM_ERROR("kelvin_voigt_newmark_assemble_diag_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

// int kelvin_voigt_newmark_crs_sym(const enum ElemType                element_type,
//                                  const ptrdiff_t                    nelements,
//                                  const ptrdiff_t                    nnodes,
//                                  idx_t **const SFEM_RESTRICT        elements,
//                                  geom_t **const SFEM_RESTRICT       points,
//                                  const real_t                       beta,
//                                  const real_t                       gamma,
//                                  const real_t                       dt,
//                                  const real_t                       k,
//                                  const real_t                       K,
//                                  const real_t                       eta,
//                                  const real_t                       rho,
//                                  const count_t *const SFEM_RESTRICT rowptr,
//                                  const idx_t *const SFEM_RESTRICT   colidx,
//                                  const ptrdiff_t block_stride,  // stride of the block matrix to interchange SoA and AoS.
//                                  real_t **const SFEM_RESTRICT block_diag,
//                                  real_t **const SFEM_RESTRICT block_offdiag) {
//     switch (element_type) {
//         case HEX8: {
//             return affine_hex8_kelvin_voigt_newmark_crs_sym(nelements,
//                                                             nnodes,
//                                                             elements,
//                                                             points,
//                                                             beta,
//                                                             gamma,
//                                                             dt,
//                                                             k,
//                                                             K,
//                                                             eta,
//                                                             rho,
//                                                             rowptr,
//                                                             colidx,
//                                                             block_stride,
//                                                             block_diag,
//                                                             block_offdiag);
//         }
//         default: {
//             SFEM_ERROR("kelvin_voigt_newmark_crs_sym not implemented for type %s\n", type_to_string(element_type));
//         }
//     }

//     return SFEM_FAILURE;
// }

int kelvin_voigt_newmark_block_diag_sym(const enum ElemType          element_type,
                                        const ptrdiff_t              nelements,
                                        const ptrdiff_t              nnodes,
                                        idx_t **const SFEM_RESTRICT  elements,
                                        geom_t **const SFEM_RESTRICT points,
                                        const real_t                 beta,
                                        const real_t                 gamma,
                                        const real_t                 dt,
                                        const real_t                 k,
                                        const real_t                 K,
                                        const real_t                 eta,
                                        const real_t                 rho,
                                        const ptrdiff_t              out_stride,
                                        real_t *const                out0,
                                        real_t *const                out1,
                                        real_t *const                out2,
                                        real_t *const                out3,
                                        real_t *const                out4,
                                        real_t *const                out5) {
    switch (element_type) {
        case HEX8: {
            return affine_hex8_kelvin_voigt_newmark_block_diag_sym(nelements,
                                                                   nnodes,
                                                                   elements,
                                                                   points,
                                                                   beta,
                                                                   gamma,
                                                                   dt,
                                                                   k,
                                                                   K,
                                                                   eta,
                                                                   rho,
                                                                   out_stride,
                                                                   out0,
                                                                   out1,
                                                                   out2,
                                                                   out3,
                                                                   out4,
                                                                   out5);
        }
        default: {
            SFEM_ERROR("kelvin_voigt_newmark_block_diag_sym not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}
