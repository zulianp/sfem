#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"
#include "tet4_neohookean_ogden_inline_cpu.h"
#include "tet4_partial_assembly_neohookean_inline.h"

static void neohookean_gradient_ref(const real_t                      mu,
                                    const real_t                      lambda,
                                    const real_t                      px0,
                                    const real_t                      px1,
                                    const real_t                      px2,
                                    const real_t                      px3,
                                    const real_t                      py0,
                                    const real_t                      py1,
                                    const real_t                      py2,
                                    const real_t                      py3,
                                    const real_t                      pz0,
                                    const real_t                      pz1,
                                    const real_t                      pz2,
                                    const real_t                      pz3,
                                    const real_t *const SFEM_RESTRICT u,
                                    real_t *const SFEM_RESTRICT       element_vector) {
    // FLOATING POINT OPS!
    //  - Result: 12*ADD + 12*ASSIGNMENT + 48*MUL
    //  - Subexpressions: 49*ADD + 5*DIV + LOG + 151*MUL + 13*NEG + 50*SUB
    const real_t x0    = pz0 - pz3;
    const real_t x1    = -x0;
    const real_t x2    = py0 - py2;
    const real_t x3    = -x2;
    const real_t x4    = px0 - px1;
    const real_t x5    = -1.0 / 6.0 * x4;
    const real_t x6    = py0 - py3;
    const real_t x7    = -x6;
    const real_t x8    = pz0 - pz2;
    const real_t x9    = -x8;
    const real_t x10   = py0 - py1;
    const real_t x11   = -x10;
    const real_t x12   = px0 - px2;
    const real_t x13   = -1.0 / 6.0 * x12;
    const real_t x14   = pz0 - pz1;
    const real_t x15   = -x14;
    const real_t x16   = px0 - px3;
    const real_t x17   = -1.0 / 6.0 * x16;
    const real_t x18   = -x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 - x15 * x17 * x3 - x5 * x7 * x9;
    const real_t x19   = x0 * x12 - x16 * x8;
    const real_t x20   = x2 * x4;
    const real_t x21   = x12 * x6;
    const real_t x22   = x10 * x16;
    const real_t x23   = x4 * x6;
    const real_t x24   = x10 * x12;
    const real_t x25   = x16 * x2;
    const real_t x26   = 1.0 / (x0 * x20 - x0 * x24 + x14 * x21 - x14 * x25 + x22 * x8 - x23 * x8);
    const real_t x27   = u[3] * x26;
    const real_t x28   = x0 * x4;
    const real_t x29   = x14 * x16;
    const real_t x30   = -x28 + x29;
    const real_t x31   = u[6] * x26;
    const real_t x32   = -x12 * x14 + x4 * x8;
    const real_t x33   = u[9] * x26;
    const real_t x34   = -x19 + x28 - x29 - x32;
    const real_t x35   = u[0] * x26;
    const real_t x36   = x19 * x27 + x30 * x31 + x32 * x33 + x34 * x35;
    const real_t x37   = x20 - x24;
    const real_t x38   = -x37;
    const real_t x39   = u[10] * x26;
    const real_t x40   = x21 - x25;
    const real_t x41   = -x40;
    const real_t x42   = u[4] * x26;
    const real_t x43   = -x22 + x23;
    const real_t x44   = u[7] * x26;
    const real_t x45   = x22 - x23 + x37 + x40;
    const real_t x46   = u[1] * x26;
    const real_t x47   = x38 * x39 + x41 * x42 + x43 * x44 + x45 * x46;
    const real_t x48   = x10 * x8 - x14 * x2;
    const real_t x49   = -x48;
    const real_t x50   = u[11] * x26;
    const real_t x51   = x0 * x2 - x6 * x8;
    const real_t x52   = -x51;
    const real_t x53   = u[5] * x26;
    const real_t x54   = x0 * x10;
    const real_t x55   = x14 * x6;
    const real_t x56   = x54 - x55;
    const real_t x57   = u[8] * x26;
    const real_t x58   = x48 + x51 - x54 + x55;
    const real_t x59   = u[2] * x26;
    const real_t x60   = x49 * x50 + x52 * x53 + x56 * x57 + x58 * x59;
    const real_t x61   = x47 * x60;
    const real_t x62   = x39 * x49 + x42 * x52 + x44 * x56 + x46 * x58;
    const real_t x63   = x38 * x50 + x41 * x53 + x43 * x57 + x45 * x59 + 1;
    const real_t x64   = x62 * x63;
    const real_t x65   = x61 - x64;
    const real_t x66   = x27 * x41 + x31 * x43 + x33 * x38 + x35 * x45;
    const real_t x67   = x19 * x53 + x30 * x57 + x32 * x50 + x34 * x59;
    const real_t x68   = x62 * x67;
    const real_t x69   = x19 * x42 + x30 * x44 + x32 * x39 + x34 * x46 + 1;
    const real_t x70   = x60 * x69;
    const real_t x71   = x27 * x52 + x31 * x56 + x33 * x49 + x35 * x58 + 1;
    const real_t x72   = x47 * x67;
    const real_t x73   = x36 * x61 - x36 * x64 + x63 * x69 * x71 + x66 * x68 - x66 * x70 - x71 * x72;
    const real_t x74   = 1.0 / x73;
    const real_t x75   = mu * x74;
    const real_t x76   = lambda * x74 * log(x73);
    const real_t x77   = mu * x36 - x65 * x75 + x65 * x76;
    const real_t x78   = x26 * x34;
    const real_t x79   = x68 - x70;
    const real_t x80   = mu * x66 - x75 * x79 + x76 * x79;
    const real_t x81   = x26 * x45;
    const real_t x82   = x63 * x69 - x72;
    const real_t x83   = mu * x71 - x75 * x82 + x76 * x82;
    const real_t x84   = x26 * x58;
    const real_t x85   = -x36 * x63 + x66 * x67;
    const real_t x86   = mu * x62 - x75 * x85 + x76 * x85;
    const real_t x87   = x36 * x60 - x67 * x71;
    const real_t x88   = mu * x47 - x75 * x87 + x76 * x87;
    const real_t x89   = -x60 * x66 + x63 * x71;
    const real_t x90   = mu * x69 - x75 * x89 + x76 * x89;
    const real_t x91   = -x47 * x71 + x62 * x66;
    const real_t x92   = mu * x67 - x75 * x91 + x76 * x91;
    const real_t x93   = x36 * x47 - x66 * x69;
    const real_t x94   = mu * x60 - x75 * x93 + x76 * x93;
    const real_t x95   = -x36 * x62 + x69 * x71;
    const real_t x96   = mu * x63 - x75 * x95 + x76 * x95;
    const real_t x97   = x26 * x41;
    const real_t x98   = x19 * x26;
    const real_t x99   = x26 * x52;
    const real_t x100  = x26 * x43;
    const real_t x101  = x26 * x30;
    const real_t x102  = x26 * x56;
    const real_t x103  = x26 * x38;
    const real_t x104  = x26 * x32;
    const real_t x105  = x26 * x49;
    element_vector[0]  = x18 * (x77 * x78 + x80 * x81 + x83 * x84);
    element_vector[1]  = x18 * (x78 * x90 + x81 * x88 + x84 * x86);
    element_vector[2]  = x18 * (x78 * x92 + x81 * x96 + x84 * x94);
    element_vector[3]  = x18 * (x77 * x98 + x80 * x97 + x83 * x99);
    element_vector[4]  = x18 * (x86 * x99 + x88 * x97 + x90 * x98);
    element_vector[5]  = x18 * (x92 * x98 + x94 * x99 + x96 * x97);
    element_vector[6]  = x18 * (x100 * x80 + x101 * x77 + x102 * x83);
    element_vector[7]  = x18 * (x100 * x88 + x101 * x90 + x102 * x86);
    element_vector[8]  = x18 * (x100 * x96 + x101 * x92 + x102 * x94);
    element_vector[9]  = x18 * (x103 * x80 + x104 * x77 + x105 * x83);
    element_vector[10] = x18 * (x103 * x88 + x104 * x90 + x105 * x86);
    element_vector[11] = x18 * (x103 * x96 + x104 * x92 + x105 * x94);
}

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

int tet4_neohookean_ogden_apply(const ptrdiff_t              nelements,
                                const ptrdiff_t              nnodes,
                                idx_t **const SFEM_RESTRICT  elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t                 mu,
                                const real_t                 lambda,
                                const ptrdiff_t              u_stride,
                                const real_t *const          ux,
                                const real_t *const          uy,
                                const real_t *const          uz,
                                const ptrdiff_t              h_stride,
                                const real_t *const          hx,
                                const real_t *const          hy,
                                const real_t *const          hz,
                                const ptrdiff_t              out_stride,
                                real_t *const                outx,
                                real_t *const                outy,
                                real_t *const                outz)

{
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[4];
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
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        for (int v = 0; v < 4; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
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

#if 0  // Old implementation
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
#else  // New partial assembly implementation
       // FUTURE: Preprocessing (once per linearization)
        scalar_t F[9] = {0};
        tet4_F(jacobian_adjugate, jacobian_determinant, element_ux, element_uy, element_uz, F);
        scalar_t S_ikmn[TET4_S_IKMN_SIZE] = {0};
        tet4_S_ikmn_neohookean(jacobian_adjugate, jacobian_determinant, F, mu, lambda, 1, S_ikmn);

        // FUTURE: Processing (each apply)
        scalar_t *inc_grad = F;
        tet4_ref_inc_grad(element_hx, element_hy, element_hz, inc_grad);
        tet4_apply_S_ikmn(S_ikmn, inc_grad, element_outx, element_outy, element_outz);


#ifndef NDEBUG
        scalar_t test_outx[4] = {0};
        scalar_t test_outy[4] = {0};
        scalar_t test_outz[4] = {0};
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
                                          test_outx,
                                          test_outy,
                                          test_outz);

        for (int k = 0; k < 4; k++) {
            scalar_t diffx = test_outx[k] - element_outx[k];
            scalar_t diffy = test_outy[k] - element_outy[k];
            scalar_t diffz = test_outz[k] - element_outz[k];

            assert(fabs(diffx) < 1e-8);
            assert(fabs(diffy) < 1e-8);
            assert(fabs(diffz) < 1e-8);
        }
#endif  // NDEBUG
#endif

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

int tet4_neohookean_ogden_gradient(const ptrdiff_t                   nelements,
                                   const ptrdiff_t                   nnodes,
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
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[4];
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
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        tet4_adjugate_and_det_s(
                // X-coordinates
                x[ev[0]],
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

#ifndef NDEBUG

        scalar_t test_u[3 * 4]      = {0};
        scalar_t test_vector[3 * 4] = {0};

        for (int k = 0; k < 4; k++) {
            test_u[0 + k * 3] = element_ux[k];
            test_u[1 + k * 3] = element_uy[k];
            test_u[2 + k * 3] = element_uz[k];
        }

        neohookean_gradient_ref(mu,
                                lambda,
                                x[ev[0]],
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
                                test_u,
                                test_vector);

        for (int k = 0; k < 4; k++) {
            scalar_t diffx = test_vector[0 + k * 3] - element_outx[k];
            scalar_t diffy = test_vector[1 + k * 3] - element_outy[k];
            scalar_t diffz = test_vector[2 + k * 3] - element_outz[k];

#ifndef NDEBUG
            if (fabs(diffx) >= 1e-12 || fabs(diffy) >= 1e-12 || fabs(diffz) >= 1e-12) {
                fprintf(stderr, "%d)\n", k);
                fprintf(stderr, "x: %g - %g = %g\n", test_vector[0 + k * 3], element_outx[k], diffx);
                fprintf(stderr, "y: %g - %g = %g\n", test_vector[1 + k * 3], element_outy[k], diffy);
                fprintf(stderr, "z: %g - %g = %g\n", test_vector[2 + k * 3], element_outz[k], diffz);
                fflush(stderr);
            }
#endif

            assert(diffx == diffx);
            assert(diffy == diffy);
            assert(diffz == diffz);

            assert(fabs(diffx) < 1e-12);
            assert(fabs(diffy) < 1e-12);
            assert(fabs(diffz) < 1e-12);
        }

#endif  // NDEBUG

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

int tet4_neohookean_ogden_diag(const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 mu,
                               const real_t                 lambda,
                               const ptrdiff_t              u_stride,
                               const real_t *const          ux,
                               const real_t *const          uy,
                               const real_t *const          uz,
                               const ptrdiff_t              out_stride,
                               real_t *const                outx,
                               real_t *const                outy,
                               real_t *const                outz) {
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

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
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

        // tet4_neohookean_ogden_diag_adj(jacobian_adjugate,
        //                                jacobian_determinant,
        //                                mu,
        //                                lambda,
        //                                element_ux,
        //                                element_uy,
        //                                element_uz,
        //                                // Output
        //                                element_outx,
        //                                element_outy,
        //                                element_outz);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
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

int tet4_neohookean_ogden_hessian(const ptrdiff_t                   nelements,
                                  const ptrdiff_t                   nnodes,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t                      mu,
                                  const real_t                      lambda,
                                  const ptrdiff_t                   u_stride,
                                  const real_t *const SFEM_RESTRICT ux,
                                  const real_t *const SFEM_RESTRICT uy,
                                  const real_t *const SFEM_RESTRICT uz,
                                  count_t *const SFEM_RESTRICT      rowptr,
                                  idx_t *const SFEM_RESTRICT        colidx,
                                  real_t *const SFEM_RESTRICT       values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for  // nowait
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[4];
        scalar_t      element_ux[4];
        scalar_t      element_uy[4];
        scalar_t      element_uz[4];
        accumulator_t element_matrix[(4 * 3) * (4 * 3)];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        neohookean_ogden_hessian_points(
                // X-coordinates
                x[ev[0]],
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
                // Model parameters
                mu,
                lambda,
                // element dispalcement
                element_ux,
                element_uy,
                element_uz,
                // output matrix
                element_matrix);

        tet4_local_to_global_vec3(ev, element_matrix, rowptr, colidx, values);
    }

    return 0;
}
