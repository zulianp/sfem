#include "kelvin_voigt_newmark.h"

#include "hex8_kelvin_voigt_newmark.h"

#include <assert.h>
#include <stdio.h>


//////////////////////////////////////////////// SOA //////////////////////////////////////////////////////

// int kelvin_voigt_newmark_apply_aos(const enum ElemType               element_type,
//                                    const ptrdiff_t                   nelements,
//                                    const ptrdiff_t                   nnodes,
//                                    idx_t **const SFEM_RESTRICT       elements,
//                                    geom_t **const SFEM_RESTRICT      points,
//                                    const real_t *const SFEM_RESTRICT u,
//                                    real_t *const SFEM_RESTRICT       values) {
//     switch (element_type) {
//         case HEX8: {
//             return hex8_kelvin_voigt_newmark_apply(
//                     nelements, nnodes, elements, points, 3, &u[0], &u[1], &u[2], 3, &values[0], &values[1], &values[2]);
//         }
//         default: {
//             SFEM_ERROR("kelvin_voigt_newmark_apply_aos not implemented for type %s\n", type_to_string(element_type));
//         }
//     }

//     return SFEM_FAILURE;
// }




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
                                         const real_t *const SFEM_RESTRICT     u,
                                         const real_t *const SFEM_RESTRICT     v,
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
                                                                1,
                                                                &u[0],
                                                                &u[1],
                                                                &u[2],
                                                                &v[0],
                                                                &v[1],
                                                                &v[2],
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
                                         const real_t *const SFEM_RESTRICT     u,
                                         const real_t *const SFEM_RESTRICT     v,
                                         real_t *const SFEM_RESTRICT           values){
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
                                                                3,
                                                                &u[0],
                                                                &u[1],
                                                                &u[2],
                                                                &v[0],
                                                                &v[1],
                                                                &v[2],
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


