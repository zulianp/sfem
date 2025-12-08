#include "mooney_rivlin_visco.h"
#include "hex8_mooney_rivlin_visco.h"
#include <stdio.h>

int mooney_rivlin_visco_gradient_aos(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     const real_t *const SFEM_RESTRICT u,
                                     real_t *const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_gradient(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                3, &u[0], &u[1], &u[2], 
                3, &out[0], &out[1], &out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_gradient_aos not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_gradient_soa(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     real_t **const SFEM_RESTRICT u,
                                     real_t **const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_gradient(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                1, u[0], u[1], u[2], 
                1, out[0], out[1], out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_gradient_soa not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_update_history_aos(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           const real_t *const SFEM_RESTRICT u) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_update_history(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, new_history, 
                3, &u[0], &u[1], &u[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_update_history_aos not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_update_history_soa(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           real_t **const SFEM_RESTRICT u) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_update_history(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, new_history, 
                1, u[0], u[1], u[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_update_history_soa not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_bsr(const enum ElemType element_type,
                            const ptrdiff_t nelements,
                            const ptrdiff_t nnodes,
                            idx_t **const SFEM_RESTRICT elements,
                            geom_t **const SFEM_RESTRICT points,
                            const real_t C10,
                            const real_t K,
                            const real_t C01,
                            const real_t dt,
                            const int num_prony_terms,
                            const real_t *const SFEM_RESTRICT g,
                            const real_t *const SFEM_RESTRICT tau,
                            const ptrdiff_t history_stride,
                            const real_t *const SFEM_RESTRICT history,
                            const ptrdiff_t u_stride,
                            const real_t *const SFEM_RESTRICT ux,
                            const real_t *const SFEM_RESTRICT uy,
                            const real_t *const SFEM_RESTRICT uz,
                            const count_t *const SFEM_RESTRICT rowptr,
                            const idx_t *const SFEM_RESTRICT colidx,
                            real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case HEX8: {
            // Fixed: added u_stride parameter
            return hex8_mooney_rivlin_visco_bsr(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                u_stride, ux, uy, uz, 
                1, // out_stride (for values? No, values is packed. Wait, let's check signature)
                values, // values is flattened array
                1, // row_stride
                rowptr, colidx);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_bsr not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_hessian_diag_aos(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_hessian_diag(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                3, &u[0], &u[1], &u[2], 
                3, &out[0], &out[1], &out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_hessian_diag_aos not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_hessian_diag_soa(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         real_t **const SFEM_RESTRICT u,
                                         real_t **const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_hessian_diag(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                1, u[0], u[1], u[2], 
                1, out[0], out[1], out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_hessian_diag_soa not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

#include <stdio.h>

int mooney_rivlin_visco_gradient_aos(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     const real_t *const SFEM_RESTRICT u,
                                     real_t *const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_gradient(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                3, &u[0], &u[1], &u[2], 
                3, &out[0], &out[1], &out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_gradient_aos not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_gradient_soa(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     real_t **const SFEM_RESTRICT u,
                                     real_t **const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_gradient(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                1, u[0], u[1], u[2], 
                1, out[0], out[1], out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_gradient_soa not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_update_history_aos(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           const real_t *const SFEM_RESTRICT u) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_update_history(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, new_history, 
                3, &u[0], &u[1], &u[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_update_history_aos not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_update_history_soa(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           real_t **const SFEM_RESTRICT u) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_update_history(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, new_history, 
                1, u[0], u[1], u[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_update_history_soa not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_bsr(const enum ElemType element_type,
                            const ptrdiff_t nelements,
                            const ptrdiff_t nnodes,
                            idx_t **const SFEM_RESTRICT elements,
                            geom_t **const SFEM_RESTRICT points,
                            const real_t C10,
                            const real_t K,
                            const real_t C01,
                            const real_t dt,
                            const int num_prony_terms,
                            const real_t *const SFEM_RESTRICT g,
                            const real_t *const SFEM_RESTRICT tau,
                            const ptrdiff_t history_stride,
                            const real_t *const SFEM_RESTRICT history,
                            const ptrdiff_t u_stride,
                            const real_t *const SFEM_RESTRICT ux,
                            const real_t *const SFEM_RESTRICT uy,
                            const real_t *const SFEM_RESTRICT uz,
                            const count_t *const SFEM_RESTRICT rowptr,
                            const idx_t *const SFEM_RESTRICT colidx,
                            real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case HEX8: {
            // Fixed: added u_stride parameter
            return hex8_mooney_rivlin_visco_bsr(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                u_stride, ux, uy, uz, 
                1, // out_stride (for values? No, values is packed. Wait, let's check signature)
                values, // values is flattened array
                1, // row_stride
                rowptr, colidx);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_bsr not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_hessian_diag_aos(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_hessian_diag(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                3, &u[0], &u[1], &u[2], 
                3, &out[0], &out[1], &out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_hessian_diag_aos not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

int mooney_rivlin_visco_hessian_diag_soa(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         real_t **const SFEM_RESTRICT u,
                                         real_t **const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_hessian_diag(
                nelements, 1, nnodes, elements, points, 
                C10, K, C01, dt, num_prony_terms, g, tau, 
                history_stride, history, 
                1, u[0], u[1], u[2], 
                1, out[0], out[1], out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_hessian_diag_soa not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}
