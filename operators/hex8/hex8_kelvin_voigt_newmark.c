#include "hex8_kelvin_voigt_newmark.h"

#include "hex8_inline_cpu.h"
#include "line_quadrature.h"
#include "hex8_kelvin_voigt_newmark_inline_cpu.h"
#include "hex8_laplacian_inline_cpu.h"

#include <assert.h>
#include <stdio.h>

// HAOYU

// int hex8_kelvin_voigt_newmark_lhs_apply(const ptrdiff_t              nelements,
//                                     const ptrdiff_t              nnodes,
//                                     idx_t **const SFEM_RESTRICT  elements,
//                                     geom_t **const SFEM_RESTRICT points,
//                                     const ptrdiff_t              in_stride,
//                                     // unified interface for both SoA and AoS
//                                     const real_t *const SFEM_RESTRICT ux,
//                                     const real_t *const SFEM_RESTRICT uy,
//                                     const real_t *const SFEM_RESTRICT uz,
//                                     const ptrdiff_t                   out_stride,
//                                     real_t *const SFEM_RESTRICT       outx,
//                                     real_t *const SFEM_RESTRICT       outy,
//                                     real_t *const SFEM_RESTRICT       outz) {
//     const geom_t *const x = points[0];
//     const geom_t *const y = points[1];
//     const geom_t *const z = points[2];

//     int             n_qp = line_q2_n;
//     const scalar_t *qx   = line_q2_x;
//     const scalar_t *qw   = line_q2_w;

// #pragma omp parallel for
//     for (ptrdiff_t e = 0; e < nelements; ++e) {
//         idx_t ev[8];

//         for(int v = 0;v < 8; v++) {
//         	ev[v] = elements[v][e];
//         }

//         // Example
//         const ptrdiff_t idx = ev[0] * in_stride;
//         scalar_t ux0 = ux[idx];
//         scalar_t uy0 = uy[idx];
//         scalar_t uz0 = uz[idx];

//         SFEM_ERROR("IMPLEMENT ME!");
//     }

//     return SFEM_SUCCESS;
// }




int affine_hex8_kelvin_voigt_newmark_lhs_apply(const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,

                                 const real_t *const          g_jacobian_adjugate,
                                 const real_t *const          g_jacobian_determinant,

                                 const real_t                 dt,
                                 const real_t                 gamma,
                                 const real_t                 beta, 

                                 const real_t                 k,
                                 const real_t                 K,
                                 const real_t                 eta,

                                 const ptrdiff_t              u_stride,
                                 const real_t *const          ux,
                                 const real_t *const          uy,
                                 const real_t *const          uz,
                                 const ptrdiff_t              out_stride,
                                 real_t *const                outx,
                                 real_t *const                outy,
                                 real_t *const                outz) {
    SFEM_UNUSED(nnodes);


    int             n_qp = line_q3_n;
    const scalar_t *qx   = line_q3_x;
    const scalar_t *qw   = line_q3_w;


#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];


        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = g_jacobian_determinant[i];

        for (int d = 0; d < 9; d++) {
            jacobian_adjugate[d] = g_jacobian_adjugate[i * 9 + d];
        }


        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }


        for (int d = 0; d < 8; d++) {
            element_outx[d] = 0;
            element_outy[d] = 0;
            element_outz[d] = 0;
        }

        // hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_kelvin_voigt_newmark_lhs_apply_adj(k,
                                                     K,
                                                     eta,
                                                     dt,
                                                     gamma,
                                                     beta,
                                                     jacobian_adjugate,
                                                     jacobian_determinant,
                                                     qx[kx],
                                                     qx[ky],
                                                     qx[kz],
                                                     qw[kx] * qw[ky] * qw[kz],
                                                     element_ux,
                                                     element_uy,
                                                     element_uz,
                                                     element_outx,
                                                     element_outy,
                                                     element_outz);
                }
            }
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}



// int affine_hex8_kelvin_voigt_newmark_bsr(const ptrdiff_t                    nelements,
//                                       const ptrdiff_t                    nnodes,
//                                       idx_t **const SFEM_RESTRICT        elements,
//                                       geom_t **const SFEM_RESTRICT       points,
//                                       const real_t                       mu,
//                                       const real_t                       lambda,
//                                       const count_t *const SFEM_RESTRICT rowptr,
//                                       const idx_t *const SFEM_RESTRICT   colidx,
//                                       real_t *const SFEM_RESTRICT        values) {
//     SFEM_UNUSED(nnodes);

//     const geom_t *const x = points[0];
//     const geom_t *const y = points[1];
//     const geom_t *const z = points[2];

//     // int SFEM_HEX8_QUADRATURE_ORDER = 2;
//     // SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
//     // // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

//     // int n_qp = line_q3_n;
//     // const scalar_t *qx = line_q3_x;
//     // const scalar_t *qw = line_q3_w;
//     // if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
//     //     n_qp = line_q2_n;
//     //     qx = line_q2_x;
//     //     qw = line_q2_w;
//     // } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
//     //     n_qp = line_q6_n;
//     //     qx = line_q6_x;
//     //     qw = line_q6_w;
//     // }

// #pragma omp parallel
//     {
//         scalar_t element_matrix[(3 * 8) * (3 * 8)];
// #pragma omp for
//         for (ptrdiff_t i = 0; i < nelements; ++i) {
//             idx_t ev[8];

//             scalar_t lx[8];
//             scalar_t ly[8];
//             scalar_t lz[8];

//             for (int v = 0; v < 8; ++v) {
//                 ev[v] = elements[v][i];
//             }

//             for (int d = 0; d < 8; d++) {
//                 lx[d] = x[ev[d]];
//                 ly[d] = y[ev[d]];
//                 lz[d] = z[ev[d]];
//             }

//             scalar_t jacobian_adjugate[9];
//             scalar_t jacobian_determinant;
//             hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

//             hex8_linear_elasticity_matrix(mu, lambda, jacobian_adjugate, jacobian_determinant, element_matrix);

//             hex8_local_to_global_bsr3(ev, element_matrix, rowptr, colidx, values);
//         }
//     }

//     return SFEM_SUCCESS;
// }



// //  F(x, x', x'') = 0
// int hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t              nelements,
//                                        const ptrdiff_t              nnodes,
//                                        idx_t **const SFEM_RESTRICT  elements,
//                                        geom_t **const SFEM_RESTRICT points,
//                                        // unified interface for both SoA and AoS
//                                        const ptrdiff_t in_stride,
//                                        // Displacement
//                                        const real_t *const SFEM_RESTRICT u_oldx,
//                                        const real_t *const SFEM_RESTRICT u_oldy,
//                                        const real_t *const SFEM_RESTRICT u_oldz,
//                                        // Velocity
//                                        const real_t *const SFEM_RESTRICT v_oldx,
//                                        const real_t *const SFEM_RESTRICT v_oldy,
//                                        const real_t *const SFEM_RESTRICT v_oldz,
//                                        // Accleration
//                                        const real_t *const SFEM_RESTRICT a_oldx,
//                                        const real_t *const SFEM_RESTRICT a_oldy,
//                                        const real_t *const SFEM_RESTRICT a_oldz,
//                                        // Current input
//                                        const real_t *const SFEM_RESTRICT ux,
//                                        const real_t *const SFEM_RESTRICT uy,
//                                        const real_t *const SFEM_RESTRICT uz,
//                                        // Output
//                                        const ptrdiff_t             out_stride,
//                                        real_t *const SFEM_RESTRICT outx,
//                                        real_t *const SFEM_RESTRICT outy,
//                                        real_t *const SFEM_RESTRICT outz) {
//     const geom_t *const x = points[0];
//     const geom_t *const y = points[1];
//     const geom_t *const z = points[2];

//     int             n_qp = line_q2_n;
//     const scalar_t *qx   = line_q2_x;
//     const scalar_t *qw   = line_q2_w;

// #pragma omp parallel for
//     for (ptrdiff_t e = 0; e < nelements; ++e) {
//         // TODO
//         SFEM_ERROR("IMPLEMENT ME!");
//     }

//     return SFEM_SUCCESS;
// }

void newmark_increment_update(
                                const real_t dt,         
                                const real_t beta,       
                                const real_t gamma,      
                                
                                const real_t *const ux,        
                                const real_t *const uy,     
                                const real_t *const uz,  

                                real_t *const vx,      
                                real_t *const vy,       
                                real_t *const vz,        
                                real_t *const ax,        
                                real_t *const ay,        
                                real_t *const az,        

                                const real_t *const u_oldx,
                                const real_t *const u_oldy,
                                const real_t *const u_oldz,
                                const real_t *const v_oldx,
                                const real_t *const v_oldy,
                                const real_t *const v_oldz,
                                const real_t *const a_oldx,
                                const real_t *const a_oldy,
                                const real_t *const a_oldz,

                                const ptrdiff_t nnodes,
                                const ptrdiff_t stride) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        const ptrdiff_t idx = i * stride;

        
        ax[idx] = (ux[idx] - u_oldx[idx] - dt*v_oldx[idx] - (dt*dt/2)*(1-2*beta)*a_oldx[idx]) / (beta*dt*dt);
        ay[idx] = (uy[idx] - u_oldy[idx] - dt*v_oldy[idx] - (dt*dt/2)*(1-2*beta)*a_oldy[idx]) / (beta*dt*dt);
        az[idx] = (uz[idx] - u_oldz[idx] - dt*v_oldz[idx] - (dt*dt/2)*(1-2*beta)*a_oldz[idx]) / (beta*dt*dt);

        vx[idx] = v_oldx[idx] + dt*((1-gamma)*a_oldx[idx] + gamma*ax[idx]);
        vy[idx] = v_oldy[idx] + dt*((1-gamma)*a_oldy[idx] + gamma*ay[idx]);
        vz[idx] = v_oldz[idx] + dt*((1-gamma)*a_oldz[idx] + gamma*az[idx]);
    }
}


// int affine_hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t              nelements,
//                                  const ptrdiff_t              nnodes,
//                                  idx_t **const SFEM_RESTRICT  elements,

//                                  const real_t *const          g_jacobian_adjugate,
//                                  const real_t *const          g_jacobian_determinant,

//                                  const real_t                 dt,
//                                  const real_t                 gamma,
//                                  const real_t                 beta, 

//                                  const real_t                 k,
//                                  const real_t                 K,
//                                  const real_t                 eta,
//                                  const real_t                 rho,

//                                  const ptrdiff_t              u_stride,

//                                  const real_t *const          u_oldx,
//                                  const real_t *const          u_oldy,
//                                  const real_t *const          u_oldz,
//                                  const real_t *const          v_oldx,
//                                  const real_t *const          v_oldy,
//                                  const real_t *const          v_oldz,
//                                  const real_t *const          a_oldx,
//                                  const real_t *const          a_oldy,
//                                  const real_t *const          a_oldz,

//                                  const real_t *const          ux,
//                                  const real_t *const          uy,
//                                  const real_t *const          uz,

//                                  const ptrdiff_t              out_stride,
//                                  real_t *const                outx,
//                                  real_t *const                outy,
//                                  real_t *const                outz) {
//     SFEM_UNUSED(nnodes);

//     int             n_qp = line_q3_n;
//     const scalar_t *qx   = line_q3_x;
//     const scalar_t *qw   = line_q3_w;

//     real_t *const   vx = malloc(nnodes * sizeof(real_t));
//     real_t *const   vy = malloc(nnodes * sizeof(real_t));
//     real_t *const   vz = malloc(nnodes * sizeof(real_t));
//     real_t *const   ax = malloc(nnodes * sizeof(real_t));
//     real_t *const   ay = malloc(nnodes * sizeof(real_t));
//     real_t *const   az = malloc(nnodes * sizeof(real_t));

//     newmark_increment_update(dt, beta, gamma, ux, uy, uz, vx, vy, vz, ax, ay, az, 
//                     u_oldx, u_oldy, u_oldz, v_oldx, v_oldy, v_oldz, a_oldx, a_oldy, a_oldz, nnodes, u_stride);

// #pragma omp parallel for
//     for (ptrdiff_t i = 0; i < nelements; ++i) {
//         idx_t    ev[8];
//         scalar_t element_ux[8];
//         scalar_t element_uy[8];
//         scalar_t element_uz[8];

//         scalar_t element_vx[8];
//         scalar_t element_vy[8];
//         scalar_t element_vz[8];

//         scalar_t element_ax[8];
//         scalar_t element_ay[8];
//         scalar_t element_az[8];

//         accumulator_t element_outx[8];
//         accumulator_t element_outy[8];
//         accumulator_t element_outz[8];

//         scalar_t jacobian_adjugate[9];
//         scalar_t jacobian_determinant = g_jacobian_determinant[i];

//         for (int d = 0; d < 9; d++) {
//             jacobian_adjugate[d] = g_jacobian_adjugate[i * 9 + d];
//         }

//         for (int v = 0; v < 8; ++v) {
//             ev[v] = elements[v][i];
//         }

//         for (int v = 0; v < 8; ++v) {
//             const ptrdiff_t idx = ev[v] * u_stride;
//             element_ux[v]       = ux[idx];
//             element_uy[v]       = uy[idx];
//             element_uz[v]       = uz[idx];
//             element_vx[v]       = vx[idx];
//             element_vy[v]       = vy[idx];
//             element_vz[v]       = vz[idx];
//             element_ax[v]       = ax[idx];
//             element_ay[v]       = ay[idx];
//             element_az[v]       = az[idx];
//         }

//         for (int d = 0; d < 8; d++) {
//             element_outx[d] = 0;
//             element_outy[d] = 0;
//             element_outz[d] = 0;
//         }

//         // hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

//         for (int kz = 0; kz < n_qp; kz++) {
//             for (int ky = 0; ky < n_qp; ky++) {
//                 for (int kx = 0; kx < n_qp; kx++) {
//                     hex8_kelvin_voigt_newmark_gradient_adj(k,
//                                                      K,
//                                                      eta,
//                                                      jacobian_adjugate,
//                                                      jacobian_determinant,
//                                                      qx[kx],
//                                                      qx[ky],
//                                                      qx[kz],
//                                                      qw[kx] * qw[ky] * qw[kz],
//                                                      element_ux,
//                                                      element_uy,
//                                                      element_uz,
//                                                      element_vx,
//                                                      element_vy,
//                                                      element_vz,
//                                                      element_ax,
//                                                      element_ay,
//                                                      element_az,
//                                                      element_outx,
//                                                      element_outy,
//                                                      element_outz);
//             }
//         }

//         for (int edof_i = 0; edof_i < 8; edof_i++) {
//             const ptrdiff_t idx = ev[edof_i] * out_stride;

// #pragma omp atomic update
//             outx[idx] += element_outx[edof_i];

// #pragma omp atomic update
//             outy[idx] += element_outy[edof_i];

// #pragma omp atomic update
//             outz[idx] += element_outz[edof_i];
//         }
//     }

//     return SFEM_SUCCESS;

// }


int affine_hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,

                                 const real_t *const          g_jacobian_adjugate,
                                 const real_t *const          g_jacobian_determinant,

                                 const real_t                 k,
                                 const real_t                 K,
                                 const real_t                 eta,

                                 const ptrdiff_t              u_stride,

                                 const real_t *const          ux,
                                 const real_t *const          uy,
                                 const real_t *const          uz,
                                 const real_t *const          vx,
                                 const real_t *const          vy,
                                 const real_t *const          vz,

                                 const ptrdiff_t              out_stride,
                                 real_t *const                outx,
                                 real_t *const                outy,
                                 real_t *const                outz) {
    SFEM_UNUSED(nnodes);

    int             n_qp = line_q3_n;
    const scalar_t *qx   = line_q3_x;
    const scalar_t *qw   = line_q3_w;


#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];

        scalar_t element_vx[8];
        scalar_t element_vy[8];
        scalar_t element_vz[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = g_jacobian_determinant[i];

        for (int d = 0; d < 9; d++) {
            jacobian_adjugate[d] = g_jacobian_adjugate[i * 9 + d];
        }

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
            element_vx[v]       = vx[idx];
            element_vy[v]       = vy[idx];
            element_vz[v]       = vz[idx];
        }

        for (int d = 0; d < 8; d++) {
            element_outx[d] = 0;
            element_outy[d] = 0;
            element_outz[d] = 0;
        }

        // hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_kelvin_voigt_newmark_gradient_adj(k,
                                                     K,
                                                     eta,
                                                     jacobian_adjugate,
                                                     jacobian_determinant,
                                                     qx[kx],
                                                     qx[ky],
                                                     qx[kz],
                                                     qw[kx] * qw[ky] * qw[kz],
                                                     element_ux,
                                                     element_uy,
                                                     element_uz,
                                                     element_vx,
                                                     element_vy,
                                                     element_vz,
                                                     element_outx,
                                                     element_outy,
                                                     element_outz);
            }
        }
        }
        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
        }

    return SFEM_SUCCESS;

}