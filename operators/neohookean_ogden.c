#include "neohookean_ogden.h"

#include "hex8_neohookean_ogden.h"
#include "tet10_neohookean_ogden.h"
#include "tet4_neohookean_ogden.h"

#include <mpi.h>
#include <stdio.h>

//////////////////////////
// Structure of arrays
//////////////////////////

int neohookean_ogden_value_soa(const enum ElemType          element_type,
                               const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 mu,
                               const real_t                 lambda,
                               real_t **const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT  value) {
    SFEM_ERROR("IMPLEMENT ME");
    return SFEM_FAILURE;
}

int neohookean_ogden_gradient_soa(const enum ElemType          element_type,
                                  const ptrdiff_t              nelements,
                                  const ptrdiff_t              nnodes,
                                  idx_t **const SFEM_RESTRICT  elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t                 mu,
                                  const real_t                 lambda,
                                  real_t **const SFEM_RESTRICT u,
                                  real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_gradient(
                    nelements, nnodes, elements, points, mu, lambda, 1, u[0], u[1], u[2], 1, values[0], values[1], values[2]);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_gradient_soa not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int neohookean_ogden_apply_soa(const enum ElemType          element_type,
                               const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 mu,
                               const real_t                 lambda,
                               real_t **const SFEM_RESTRICT u,
                               real_t **const SFEM_RESTRICT h,
                               real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TET4: {
            tet4_neohookean_ogden_apply(nelements,
                                        nnodes,
                                        elements,
                                        points,
                                        mu,
                                        lambda,
                                        1,
                                        u[0],
                                        u[1],
                                        u[2],
                                        1,
                                        h[0],
                                        h[1],
                                        h[2],
                                        1,
                                        values[0],
                                        values[1],
                                        values[2]);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_apply_soa not implemented for type %s\n", type_to_string(element_type));
        }
    }
    return SFEM_FAILURE;
}

int neohookean_ogden_hessian_soa(const enum ElemType                element_type,
                                 const ptrdiff_t                    nelements,
                                 const ptrdiff_t                    nnodes,
                                 idx_t **const SFEM_RESTRICT        elements,
                                 geom_t **const SFEM_RESTRICT       points,
                                 const real_t                       mu,
                                 const real_t                       lambda,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT   colidx,
                                 real_t **const SFEM_RESTRICT       values) {
    SFEM_ERROR("IMPLEMENT ME");
    return SFEM_FAILURE;
}

//////////////////////////
// Array of structures
//////////////////////////

int neohookean_ogden_value_aos(const enum ElemType               element_type,
                               const ptrdiff_t                   nelements,
                               const ptrdiff_t                   nnodes,
                               idx_t **const SFEM_RESTRICT       elements,
                               geom_t **const SFEM_RESTRICT      points,
                               const real_t                      mu,
                               const real_t                      lambda,
                               const real_t *const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT       value) {
    SFEM_ERROR("IMPLEMENT ME");
    return SFEM_FAILURE;
}

int neohookean_ogden_apply_aos(const enum ElemType               element_type,
                               const ptrdiff_t                   nelements,
                               const ptrdiff_t                   nnodes,
                               idx_t **const SFEM_RESTRICT       elements,
                               geom_t **const SFEM_RESTRICT      points,
                               const real_t                      mu,
                               const real_t                      lambda,
                               const real_t *const SFEM_RESTRICT u,
                               const real_t *const SFEM_RESTRICT h,
                               real_t *const SFEM_RESTRICT       values) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_apply(nelements,
                                               nnodes,
                                               elements,
                                               points,
                                               mu,
                                               lambda,
                                               3,
                                               &u[0],
                                               &u[1],
                                               &u[2],
                                               3,
                                               &h[0],
                                               &h[1],
                                               &h[2],
                                               3,
                                               &values[0],
                                               &values[1],
                                               &values[2]);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_apply_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int neohookean_ogden_gradient_aos(const enum ElemType               element_type,
                                  const ptrdiff_t                   nelements,
                                  const ptrdiff_t                   nnodes,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t                      mu,
                                  const real_t                      lambda,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT       values) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_gradient(nelements,
                                                  nnodes,
                                                  elements,
                                                  points,
                                                  mu,
                                                  lambda,
                                                  3,
                                                  &u[0],
                                                  &u[1],
                                                  &u[2],
                                                  3,
                                                  &values[0],
                                                  &values[1],
                                                  &values[2]);
        }
        case TET10: {
            return tet10_neohookean_ogden_gradient(nelements,
                                                   1,
                                                   nnodes,
                                                   elements,
                                                   points,
                                                   mu,
                                                   lambda,
                                                   3,
                                                   &u[0],
                                                   &u[1],
                                                   &u[2],
                                                   3,
                                                   &values[0],
                                                   &values[1],
                                                   &values[2]);
        }
        case HEX8: {
            return hex8_neohookean_ogden_gradient(nelements,
                                                  1,
                                                  nnodes,
                                                  elements,
                                                  points,
                                                  mu,
                                                  lambda,
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
            SFEM_ERROR("neohookean_ogden_gradient_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }
    return SFEM_FAILURE;
}

int neohookean_ogden_hessian_aos(const enum ElemType                element_type,
                                 const ptrdiff_t                    nelements,
                                 const ptrdiff_t                    nnodes,
                                 idx_t **const SFEM_RESTRICT        elements,
                                 geom_t **const SFEM_RESTRICT       points,
                                 const real_t                       mu,
                                 const real_t                       lambda,
                                 const real_t *const SFEM_RESTRICT  u,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT   colidx,
                                 real_t *const SFEM_RESTRICT        values) {
    SFEM_ERROR("IMPLEMENT ME");
    return SFEM_FAILURE;
}

int neohookean_ogden_diag_aos(const enum ElemType               element_type,
                              const ptrdiff_t                   nelements,
                              const ptrdiff_t                   stride,
                              const ptrdiff_t                   nnodes,
                              idx_t **const SFEM_RESTRICT       elements,
                              geom_t **const SFEM_RESTRICT      points,
                              const real_t                      mu,
                              const real_t                      lambda,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT       values) {
    switch (element_type) {
        case HEX8: {
            return hex8_neohookean_ogden_elasticity_diag(nelements,
                                                         stride,
                                                         nnodes,
                                                         elements,
                                                         points,
                                                         mu,
                                                         lambda,
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
            SFEM_ERROR("neohookean_ogden_diag_aos not implemented for type %s\n", type_to_string(element_type));
            return SFEM_FAILURE;
        }
    }
}

int neohookean_ogden_partial_assembly_diag(const enum ElemType               element_type,
                                           const ptrdiff_t                   nelements,
                                           const ptrdiff_t                   stride,
                                           idx_t **const SFEM_RESTRICT       elements,
                                           geom_t **const SFEM_RESTRICT      points,
                                           const real_t                      mu,
                                           const real_t                      lambda,
                                           const ptrdiff_t                   u_stride,
                                           const real_t *const SFEM_RESTRICT ux,
                                           const real_t *const SFEM_RESTRICT uy,
                                           const real_t *const SFEM_RESTRICT uz,
                                           const ptrdiff_t                   out_stride,
                                           real_t *const SFEM_RESTRICT       outx,
                                           real_t *const SFEM_RESTRICT       outy,
                                           real_t *const SFEM_RESTRICT       outz) {

                                            
    switch (element_type) {
        case HEX8: {
            return hex8_neohookean_ogden_partial_assembly_diag(
                    nelements, stride, elements, points, mu, lambda, 3, ux, uy, uz, 3, outx, outy, outz);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_partial_assembly_diag not implemented for type %s\n", type_to_string(element_type));
            return SFEM_FAILURE;
        }
    }
}

int neohookean_ogden_hessian_partial_assembly(const enum ElemType                  element_type,
                                              const ptrdiff_t                      nelements,
                                              const ptrdiff_t                      stride,
                                              idx_t **const SFEM_RESTRICT          elements,
                                              geom_t **const SFEM_RESTRICT         points,
                                              const real_t                         mu,
                                              const real_t                         lambda,
                                              const ptrdiff_t                      u_stride,
                                              const real_t *const SFEM_RESTRICT    ux,
                                              const real_t *const SFEM_RESTRICT    uy,
                                              const real_t *const SFEM_RESTRICT    uz,
                                              metric_tensor_t *const SFEM_RESTRICT partial_assembly) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_hessian_partial_assembly(
                    nelements, stride, elements, points, mu, lambda, u_stride, ux, uy, uz, partial_assembly);
        }
        case TET10: {
            return tet10_neohookean_ogden_hessian_partial_assembly(
                    nelements, stride, elements, points, mu, lambda, u_stride, ux, uy, uz, partial_assembly);
        }
        case HEX8: {
            return hex8_neohookean_ogden_hessian_partial_assembly(
                    nelements, stride, elements, points, mu, lambda, u_stride, ux, uy, uz, partial_assembly);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_hessian_partial_assembly not implemented for type %s\n", type_to_string(element_type));
        }
    }
    return SFEM_FAILURE;
}

int neohookean_ogden_partial_assembly_apply(const enum ElemType                        element_type,
                                            const ptrdiff_t                            nelements,
                                            const ptrdiff_t                            stride,
                                            idx_t **const SFEM_RESTRICT                elements,
                                            const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                            const ptrdiff_t                            h_stride,
                                            const real_t *const                        hx,
                                            const real_t *const                        hy,
                                            const real_t *const                        hz,
                                            const ptrdiff_t                            out_stride,
                                            real_t *const                              outx,
                                            real_t *const                              outy,
                                            real_t *const                              outz) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_partial_assembly_apply(
                    nelements, stride, elements, partial_assembly, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
        }
        case TET10: {
            return tet10_neohookean_ogden_partial_assembly_apply(
                    nelements, stride, elements, partial_assembly, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
        }
        case HEX8: {
            return hex8_neohookean_ogden_partial_assembly_apply(
                    nelements, stride, elements, partial_assembly, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_partial_assembly_apply not implemented for type %s\n", type_to_string(element_type));
        }
    }
    return SFEM_FAILURE;
}

int neohookean_ogden_compressed_partial_assembly_apply(const enum ElemType                     element_type,
                                                       const ptrdiff_t                         nelements,
                                                       const ptrdiff_t                         stride,
                                                       idx_t **const SFEM_RESTRICT             elements,
                                                       const compressed_t *const SFEM_RESTRICT partial_assembly,
                                                       const scaling_t *const SFEM_RESTRICT    scaling,
                                                       const ptrdiff_t                         h_stride,
                                                       const real_t *const                     hx,
                                                       const real_t *const                     hy,
                                                       const real_t *const                     hz,
                                                       const ptrdiff_t                         out_stride,
                                                       real_t *const                           outx,
                                                       real_t *const                           outy,
                                                       real_t *const                           outz) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_compressed_partial_assembly_apply(
                    nelements, stride, elements, partial_assembly, scaling, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
        }
        case TET10: {
            return tet10_neohookean_ogden_compressed_partial_assembly_apply(
                    nelements, stride, elements, partial_assembly, scaling, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
        }
        case HEX8: {
            return hex8_neohookean_ogden_compressed_partial_assembly_apply(
                    nelements, stride, elements, partial_assembly, scaling, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_compressed_partial_assembly_apply not implemented for type %s\n",
                       type_to_string(element_type));
        }
    }
    return SFEM_FAILURE;
}

int neohookean_ogden_objective_aos(const enum ElemType               element_type,
                                   const ptrdiff_t                   nelements,
                                   const ptrdiff_t                   stride,
                                   const ptrdiff_t                   nnodes,
                                   idx_t **const SFEM_RESTRICT       elements,
                                   geom_t **const SFEM_RESTRICT      points,
                                   const real_t                      mu,
                                   const real_t                      lambda,
                                   const real_t *const SFEM_RESTRICT u,
                                   const int                         is_element_wise,
                                   real_t *const SFEM_RESTRICT       out) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_objective(
                    nelements, stride, nnodes, elements, points, mu, lambda, 3, &u[0], &u[1], &u[2], is_element_wise, out);
        }
        case TET10: {
            return tet10_neohookean_ogden_objective(
                    nelements, stride, nnodes, elements, points, mu, lambda, 3, &u[0], &u[1], &u[2], is_element_wise, out);
        }
        case HEX8: {
            return hex8_neohookean_ogden_objective(
                    nelements, stride, nnodes, elements, points, mu, lambda, 3, &u[0], &u[1], &u[2], is_element_wise, out);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_objective_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }
    return SFEM_FAILURE;
}

int neohookean_ogden_objective_steps_aos(const enum ElemType               element_type,
                                         const ptrdiff_t                   nelements,
                                         const ptrdiff_t                   stride,
                                         const ptrdiff_t                   nnodes,
                                         idx_t **const SFEM_RESTRICT       elements,
                                         geom_t **const SFEM_RESTRICT      points,
                                         const real_t                      mu,
                                         const real_t                      lambda,
                                         const real_t *const SFEM_RESTRICT u,
                                         const real_t *const SFEM_RESTRICT inc,
                                         const int                         nsteps,
                                         const real_t *const               steps,
                                         real_t *const SFEM_RESTRICT       out) {
    switch (element_type) {
        case TET4: {
            return tet4_neohookean_ogden_objective_steps(nelements,
                                                         stride,
                                                         nnodes,
                                                         elements,
                                                         points,
                                                         mu,
                                                         lambda,
                                                         3,
                                                         &u[0],
                                                         &u[1],
                                                         &u[2],
                                                         3,
                                                         &inc[0],
                                                         &inc[1],
                                                         &inc[2],
                                                         nsteps,
                                                         steps,
                                                         out);
        }
        case TET10: {
            return tet10_neohookean_ogden_objective_steps(nelements,
                                                          stride,
                                                          nnodes,
                                                          elements,
                                                          points,
                                                          mu,
                                                          lambda,
                                                          3,
                                                          &u[0],
                                                          &u[1],
                                                          &u[2],
                                                          3,
                                                          &inc[0],
                                                          &inc[1],
                                                          &inc[2],
                                                          nsteps,
                                                          steps,
                                                          out);
        }
        case HEX8: {
            return hex8_neohookean_ogden_objective_steps(nelements,
                                                         stride,
                                                         nnodes,
                                                         elements,
                                                         points,
                                                         mu,
                                                         lambda,
                                                         3,
                                                         &u[0],
                                                         &u[1],
                                                         &u[2],
                                                         3,
                                                         &inc[0],
                                                         &inc[1],
                                                         &inc[2],
                                                         nsteps,
                                                         steps,
                                                         out);
        }
        default: {
            SFEM_ERROR("neohookean_ogden_objective_steps_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }
    return SFEM_FAILURE;
}