#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"
#include "tet4_neohookean_ogden_inline_cpu.h"

// #ifndef NDEBUG
// #include "tet4_neohookean.h"
// #endif

// int tet4_neohookean_ogden_value(const ptrdiff_t nelements,
//                                 const ptrdiff_t nnodes,
//                                 idx_t **const SFEM_RESTRICT elements,
//                                 geom_t **const SFEM_RESTRICT points,
//                                 const real_t mu,
//                                 const real_t lambda,
//                                 const ptrdiff_t u_stride,
//                                 const real_t *const ux,
//                                 const real_t *const uy,
//                                 const real_t *const uz,
//                                 real_t *const SFEM_RESTRICT value) {
//     SFEM_UNUSED(nnodes);

//     const geom_t *const x = points[0];
//     const geom_t *const y = points[1];
//     const geom_t *const z = points[2];

//     real_t acc = 0;
// #pragma omp parallel for reduction(+ : acc)
//     for (ptrdiff_t i = 0; i < nelements; ++i) {
//         idx_t ev[4];
//         scalar_t element_ux[4];
//         scalar_t element_uy[4];
//         scalar_t element_uz[4];

// #pragma unroll(4)
//         for (int v = 0; v < 4; ++v) {
//             ev[v] = elements[v][i];
//         }

//         for (int enode = 0; enode < 4; ++enode) {
//             idx_t dof = ev[enode] * u_stride;
//             element_ux[enode] = ux[dof];
//             element_uy[enode] = uy[dof];
//             element_uz[enode] = uz[dof];
//         }

//         real_t element_scalar = 0;
//         tet4_neohookean_ogden_value_points(  // Model parameters
//                 mu,
//                 lambda,
//                 // X-coordinates
//                 x[ev[0]],
//                 x[ev[1]],
//                 x[ev[2]],
//                 x[ev[3]],
//                 // Y-coordinates
//                 y[ev[0]],
//                 y[ev[1]],
//                 y[ev[2]],
//                 y[ev[3]],
//                 // Z-coordinates
//                 z[ev[0]],
//                 z[ev[1]],
//                 z[ev[2]],
//                 z[ev[3]],
//                 element_ux,
//                 element_uy,
//                 element_uz,
//                 // output vector
//                 &element_scalar);

//         acc += element_scalar;
//     }

//     *value += acc;
//     return 0;
// }

int tet4_neohookean_ogden_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t mu,
                                const real_t lambda,
                                const ptrdiff_t u_stride,
                                const real_t *const ux,
                                const real_t *const uy,
                                const real_t *const uz,
                                const ptrdiff_t h_stride,
                                const real_t *const hx,
                                const real_t *const hy,
                                const real_t *const hz,
                                const ptrdiff_t out_stride,
                                real_t *const outx,
                                real_t *const outy,
                                real_t *const outz)

{
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        scalar_t element_ux[4];
        scalar_t element_uy[4];
        scalar_t element_uz[4];

        scalar_t element_hx[4];
        scalar_t element_hy[4];
        scalar_t element_hz[4];

        accumulator_t element_outx[4];
        accumulator_t element_outy[4];
        accumulator_t element_outz[4];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_ux[v] = ux[ev[v] * u_stride];
            element_uy[v] = uy[ev[v] * u_stride];
            element_uz[v] = uz[ev[v] * u_stride];
        }

        for (int v = 0; v < 4; ++v) {
            element_hx[v] = hx[ev[v] * h_stride];
            element_hy[v] = hy[ev[v] * h_stride];
            element_hz[v] = hz[ev[v] * h_stride];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
                                x[ev[1]],
                                x[ev[2]],
                                x[ev[3]],
                                // Y-coordinates
                                y[ev[0]],
                                y[ev[1]],
                                y[ev[2]],
                                y[ev[3]],
                                // Z-coordinates
                                z[ev[0]],
                                z[ev[1]],
                                z[ev[2]],
                                z[ev[3]],
                                // Output
                                jacobian_adjugate,
                                &jacobian_determinant);

        tet4_neohookean_hessian_apply_adj(jacobian_adjugate,
                                          jacobian_determinant,
                                          mu,
                                          lambda,
                                          element_ux,
                                          element_uy,
                                          element_uz,
                                          element_hx,
                                          element_hy,
                                          element_hz,
                                          element_outx,
                                          element_outy,
                                          element_outz);

        for (int edof_i = 0; edof_i < 4; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return 0;
}

int tet4_neohookean_ogden_gradient(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const SFEM_RESTRICT elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t mu,
                                   const real_t lambda,
                                   const ptrdiff_t u_stride,
                                   const real_t *const ux,
                                   const real_t *const uy,
                                   const real_t *const uz,
                                   const ptrdiff_t out_stride,
                                   real_t *const outx,
                                   real_t *const outy,
                                   real_t *const outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        scalar_t element_ux[4];
        scalar_t element_uy[4];
        scalar_t element_uz[4];

        accumulator_t element_outx[4];
        accumulator_t element_outy[4];
        accumulator_t element_outz[4];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_ux[v] = ux[ev[v] * u_stride];
            element_uy[v] = uy[ev[v] * u_stride];
            element_uz[v] = uz[ev[v] * u_stride];
        }

        tet4_adjugate_and_det_s(x[ev[0]],
                                x[ev[1]],
                                x[ev[2]],
                                x[ev[3]],
                                // Y-coordinates
                                y[ev[0]],
                                y[ev[1]],
                                y[ev[2]],
                                y[ev[3]],
                                // Z-coordinates
                                z[ev[0]],
                                z[ev[1]],
                                z[ev[2]],
                                z[ev[3]],
                                // Output
                                jacobian_adjugate,
                                &jacobian_determinant);

        tet4_neohookean_gradient_adj(jacobian_adjugate,
                                     jacobian_determinant,
                                     mu,
                                     lambda,
                                     element_ux,
                                     element_uy,
                                     element_uz,
                                     element_outx,
                                     element_outy,
                                     element_outz);

        for (int edof_i = 0; edof_i < 4; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

// #ifndef NDEBUG
//     printf("TESTING NEO\n");
//     real_t *test_input = calloc(nnodes * 3, sizeof(real_t));
//     for (ptrdiff_t i = 0; i < nnodes; i++) {
//         test_input[i * 3] = ux[i * u_stride];
//         test_input[i * 3 + 1] = uy[i * u_stride];
//         test_input[i * 3 + 2] = uz[i * u_stride];
//     }

//     real_t *test_values = calloc(nnodes * 3, sizeof(real_t));
//     neohookean_assemble_gradient(
//             nelements, nnodes, elements, points, mu, lambda, test_input, test_values);

//     for (ptrdiff_t i = 0; i < nnodes; i++) {
//         assert(fabs(test_values[i * 3] - outx[i * out_stride]) < 1e-10);
//         assert(fabs(test_values[i * 3 + 1] - outy[i * out_stride]) < 1e-10);
//         assert(fabs(test_values[i * 3 + 2] - outz[i * out_stride]) < 1e-10);
//     }

//     free(test_input);
//     free(test_values);
// #endif

    return 0;
}

// int tet4_neohookean_ogden_diag(const ptrdiff_t nelements,
//                                const ptrdiff_t nnodes,
//                                idx_t **const SFEM_RESTRICT elements,
//                                geom_t **const SFEM_RESTRICT points,
//                                const real_t mu,
//                                const real_t lambda,
//                                const ptrdiff_t u_stride,
//                                const real_t *const ux,
//                                const real_t *const uy,
//                                const real_t *const uz,
//                                const ptrdiff_t out_stride,
//                                real_t *const outx,
//                                real_t *const outy,
//                                real_t *const outz) {
//     SFEM_UNUSED(nnodes);

//     const geom_t *const x = points[0];
//     const geom_t *const y = points[1];
//     const geom_t *const z = points[2];

// #pragma omp parallel for
//     for (ptrdiff_t i = 0; i < nelements; ++i) {
//         idx_t ev[4];

//         accumulator_t element_outx[4];
//         accumulator_t element_outy[4];
//         accumulator_t element_outz[4];

//         scalar_t jacobian_adjugate[9];
//         scalar_t jacobian_determinant = 0;

// #pragma unroll(4)
//         for (int v = 0; v < 4; ++v) {
//             ev[v] = elements[v][i];
//         }

//         tet4_adjugate_and_det_s(x[ev[0]],
//                                 x[ev[1]],
//                                 x[ev[2]],
//                                 x[ev[3]],
//                                 // Y-coordinates
//                                 y[ev[0]],
//                                 y[ev[1]],
//                                 y[ev[2]],
//                                 y[ev[3]],
//                                 // Z-coordinates
//                                 z[ev[0]],
//                                 z[ev[1]],
//                                 z[ev[2]],
//                                 z[ev[3]],
//                                 // Output
//                                 jacobian_adjugate,
//                                 &jacobian_determinant);

//         tet4_neohookean_ogden_diag_adj(mu,
//                                        lambda,
//                                        jacobian_adjugate,
//                                        jacobian_determinant,
//                                        // Output
//                                        element_outx,
//                                        element_outy,
//                                        element_outz);

//         for (int edof_i = 0; edof_i < 4; ++edof_i) {
//             const ptrdiff_t idx = ev[edof_i] * out_stride;

// #pragma omp atomic update
//             outx[idx] += element_outx[edof_i];

// #pragma omp atomic update
//             outy[idx] += element_outy[edof_i];

// #pragma omp atomic update
//             outz[idx] += element_outz[edof_i];
//         }
//     }

//     return 0;
// }

// int tet4_neohookean_ogden_hessian(const ptrdiff_t nelements,
//                                   const ptrdiff_t nnodes,
//                                   idx_t **const SFEM_RESTRICT elements,
//                                   geom_t **const SFEM_RESTRICT points,
//                                   const real_t mu,
//                                   const real_t lambda,
//                                   const ptrdiff_t u_stride,
//                                   const real_t *const ux,
//                                   const real_t *const uy,
//                                   const real_t *const uz,
//                                   const count_t *const SFEM_RESTRICT rowptr,
//                                   const idx_t *const SFEM_RESTRICT colidx,
//                                   real_t *const SFEM_RESTRICT values) {
//     SFEM_UNUSED(nnodes);

//     const geom_t *const x = points[0];
//     const geom_t *const y = points[1];
//     const geom_t *const z = points[2];

// #pragma omp parallel for
//     for (ptrdiff_t i = 0; i < nelements; ++i) {
//         idx_t ev[4];
//         accumulator_t element_matrix[(4 * 3) * (4 * 3)];

// #pragma unroll(4)
//         for (int v = 0; v < 4; ++v) {
//             ev[v] = elements[v][i];
//         }

//         scalar_t jacobian_adjugate[9];
//         scalar_t jacobian_determinant = 0;
//         tet4_adjugate_and_det_s(x[ev[0]],
//                                 x[ev[1]],
//                                 x[ev[2]],
//                                 x[ev[3]],
//                                 // Y-coordinates
//                                 y[ev[0]],
//                                 y[ev[1]],
//                                 y[ev[2]],
//                                 y[ev[3]],
//                                 // Z-coordinates
//                                 z[ev[0]],
//                                 z[ev[1]],
//                                 z[ev[2]],
//                                 z[ev[3]],
//                                 // Output
//                                 jacobian_adjugate,
//                                 &jacobian_determinant);

//         tet4_neohookean_ogden_hessian_adj  // Fastest on M1
//                                            // tet4_neohookean_ogden_hessian_adj_less_registers //
//                                            // Slightly slower on M1
//                 (mu, lambda, jacobian_adjugate, jacobian_determinant, element_matrix);

//         tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
//     }

//     return 0;
// }
