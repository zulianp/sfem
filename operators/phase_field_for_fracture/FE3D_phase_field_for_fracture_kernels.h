// Automatically generate code for phase_field_for_fracture
#include "math.h"
#include "sfem_base.h"
#include "sfem_vec.h"

#ifndef SFEM_CUDA_INLINE
#define SFEM_CUDA_INLINE SFEM_INLINE __device__ __host__ 
#endif 

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_value(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_scalar) {
    // TODO COST
    element_scalar[0] =
        det_jac * ref_vol *
        ((1.0 / 2.0) * Gc *
             (ls * (pow(s_grad_phase[0], 2) + pow(s_grad_phase[1], 2) + pow(s_grad_phase[2], 2)) +
              pow(s_phase, 2) / ls) +
         pow(1 - s_phase, 2) *
             ((1.0 / 2.0) * lambda * pow(s_grad_disp[0] + s_grad_disp[4] + s_grad_disp[8], 2) +
              mu * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[4], 2) + pow(s_grad_disp[8], 2) +
                    2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[3], 2) +
                    2 * pow((1.0 / 2.0) * s_grad_disp[2] + (1.0 / 2.0) * s_grad_disp[6], 2) +
                    2 * pow((1.0 / 2.0) * s_grad_disp[5] + (1.0 / 2.0) * s_grad_disp[7], 2))));
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_gradient(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = Gc * ls;
    const real_t x1 = (1.0 / 2.0) * lambda;
    const real_t x2 = det_jac * ref_vol;
    const real_t x3 = pow(1 - s_phase, 2);
    const real_t x4 = test_grad[1] * x3;
    const real_t x5 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x6 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x7 = test_grad[2] * x3;
    const real_t x8 = mu * x7;
    const real_t x9 = 2 * s_grad_disp[0];
    const real_t x10 = 2 * s_grad_disp[4];
    const real_t x11 = 2 * s_grad_disp[8];
    const real_t x12 = x1 * (x10 + x11 + x9);
    const real_t x13 = test_grad[0] * x3;
    const real_t x14 = s_grad_disp[5] + s_grad_disp[7];
    element_vector[0] =
        x2 *
        (s_grad_phase[0] * test_grad[0] * x0 + s_grad_phase[1] * test_grad[1] * x0 +
         s_grad_phase[2] * test_grad[2] * x0 +
         test_fun *
             (Gc * s_phase / ls +
              (2 * s_phase - 2) *
                  (mu * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[4], 2) + pow(s_grad_disp[8], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[3], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[2] + (1.0 / 2.0) * s_grad_disp[6], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[5] + (1.0 / 2.0) * s_grad_disp[7], 2)) +
                   x1 * pow(s_grad_disp[0] + s_grad_disp[4] + s_grad_disp[8], 2))));
    element_vector[1] = x2 * (x13 * (mu * x9 + x12) + x4 * x5 + x6 * x8);
    element_vector[2] = x2 * (x13 * x5 + x14 * x8 + x4 * (mu * x10 + x12));
    element_vector[3] = x2 * (mu * x13 * x6 + mu * x14 * x4 + x7 * (mu * x11 + x12));
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_hessian(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_matrix) {
    // TODO COST
    const real_t x0 = Gc * ls;
    const real_t x1 = test_grad[0] * trial_grad[0];
    const real_t x2 = test_grad[1] * trial_grad[1];
    const real_t x3 = test_grad[2] * trial_grad[2];
    const real_t x4 = 2 * mu;
    const real_t x5 = det_jac * ref_vol;
    const real_t x6 = 2 * s_phase - 2;
    const real_t x7 = test_grad[1] * x6;
    const real_t x8 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x9 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x10 = test_grad[2] * x6;
    const real_t x11 = mu * x10;
    const real_t x12 = 2 * s_grad_disp[0];
    const real_t x13 = 2 * s_grad_disp[4];
    const real_t x14 = 2 * s_grad_disp[8];
    const real_t x15 = (1.0 / 2.0) * lambda * (x12 + x13 + x14);
    const real_t x16 = mu * x12 + x15;
    const real_t x17 = test_grad[0] * x6;
    const real_t x18 = trial_fun * x5;
    const real_t x19 = s_grad_disp[5] + s_grad_disp[7];
    const real_t x20 = mu * x13 + x15;
    const real_t x21 = mu * x9;
    const real_t x22 = mu * x19;
    const real_t x23 = mu * x14 + x15;
    const real_t x24 = test_fun * x6;
    const real_t x25 = trial_grad[1] * x24;
    const real_t x26 = trial_grad[2] * x24;
    const real_t x27 = trial_grad[0] * x24;
    const real_t x28 = pow(1 - s_phase, 2);
    const real_t x29 = mu * x28;
    const real_t x30 = x2 * x29;
    const real_t x31 = x29 * x3;
    const real_t x32 = lambda + x4;
    const real_t x33 = x1 * x28;
    const real_t x34 = lambda * x28;
    const real_t x35 = test_grad[1] * trial_grad[0];
    const real_t x36 = test_grad[0] * x29;
    const real_t x37 = test_grad[2] * x34;
    const real_t x38 = test_grad[0] * x34;
    const real_t x39 = mu * x33;
    const real_t x40 = x28 * x32;
    const real_t x41 = test_grad[1] * trial_grad[2];
    const real_t x42 = test_grad[2] * x29;
    element_matrix[0] =
        x5 * (test_fun * trial_fun *
                  (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[4] + s_grad_disp[8], 2) +
                   x4 * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[4], 2) + pow(s_grad_disp[8], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[3], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[2] + (1.0 / 2.0) * s_grad_disp[6], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[5] + (1.0 / 2.0) * s_grad_disp[7], 2))) +
              x0 * x1 + x0 * x2 + x0 * x3);
    element_matrix[1] = x18 * (x11 * x9 + x16 * x17 + x7 * x8);
    element_matrix[2] = x18 * (x11 * x19 + x17 * x8 + x20 * x7);
    element_matrix[3] = x18 * (x10 * x23 + x17 * x21 + x22 * x7);
    element_matrix[4] = x5 * (x16 * x27 + x21 * x26 + x25 * x8);
    element_matrix[5] = x5 * (x30 + x31 + x32 * x33);
    element_matrix[6] = x5 * (trial_grad[1] * x36 + x34 * x35);
    element_matrix[7] = x5 * (trial_grad[0] * x37 + trial_grad[2] * x36);
    element_matrix[8] = x5 * (x20 * x25 + x22 * x26 + x27 * x8);
    element_matrix[9] = x5 * (trial_grad[1] * x38 + x29 * x35);
    element_matrix[10] = x5 * (x2 * x40 + x31 + x39);
    element_matrix[11] = x5 * (trial_grad[1] * x37 + x29 * x41);
    element_matrix[12] = x5 * (x21 * x27 + x22 * x25 + x23 * x26);
    element_matrix[13] = x5 * (trial_grad[0] * x42 + trial_grad[2] * x38);
    element_matrix[14] = x5 * (trial_grad[1] * x42 + x34 * x41);
    element_matrix[15] = x5 * (x3 * x40 + x30 + x39);
}
// Split evaliations

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_gradient_u(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = pow(1 - s_phase, 2);
    const real_t x1 = test_grad[1] * x0;
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x3 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x4 = test_grad[2] * x0;
    const real_t x5 = mu * x4;
    const real_t x6 = 2 * s_grad_disp[0];
    const real_t x7 = 2 * s_grad_disp[4];
    const real_t x8 = 2 * s_grad_disp[8];
    const real_t x9 = (1.0 / 2.0) * lambda * (x6 + x7 + x8);
    const real_t x10 = test_grad[0] * x0;
    const real_t x11 = det_jac * ref_vol;
    const real_t x12 = s_grad_disp[5] + s_grad_disp[7];
    element_vector[0] = x11 * (x1 * x2 + x10 * (mu * x6 + x9) + x3 * x5);
    element_vector[1] = x11 * (x1 * (mu * x7 + x9) + x10 * x2 + x12 * x5);
    element_vector[2] = x11 * (mu * x1 * x12 + mu * x10 * x3 + x4 * (mu * x8 + x9));
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_gradient_c(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = Gc * ls;
    element_vector[0] =
        det_jac * ref_vol *
        (s_grad_phase[0] * test_grad[0] * x0 + s_grad_phase[1] * test_grad[1] * x0 +
         s_grad_phase[2] * test_grad[2] * x0 +
         test_fun *
             (Gc * s_phase / ls +
              (2 * s_phase - 2) *
                  ((1.0 / 2.0) * lambda * pow(s_grad_disp[0] + s_grad_disp[4] + s_grad_disp[8], 2) +
                   mu *
                       (pow(s_grad_disp[0], 2) + pow(s_grad_disp[4], 2) + pow(s_grad_disp[8], 2) +
                        2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[3], 2) +
                        2 * pow((1.0 / 2.0) * s_grad_disp[2] + (1.0 / 2.0) * s_grad_disp[6], 2) +
                        2 * pow((1.0 / 2.0) * s_grad_disp[5] + (1.0 / 2.0) * s_grad_disp[7], 2)))));
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_hessian_cc(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_matrix) {
    // TODO COST
    const real_t x0 = Gc * ls;
    element_matrix[0] =
        det_jac * ref_vol *
        (test_fun * trial_fun *
             (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[4] + s_grad_disp[8], 2) +
              2 * mu *
                  (pow(s_grad_disp[0], 2) + pow(s_grad_disp[4], 2) + pow(s_grad_disp[8], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[3], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[2] + (1.0 / 2.0) * s_grad_disp[6], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[5] + (1.0 / 2.0) * s_grad_disp[7], 2))) +
         test_grad[0] * trial_grad[0] * x0 + test_grad[1] * trial_grad[1] * x0 +
         test_grad[2] * trial_grad[2] * x0);
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_hessian_uu(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_matrix) {
    // TODO COST
    const real_t x0 = pow(1 - s_phase, 2);
    const real_t x1 = mu * x0;
    const real_t x2 = test_grad[1] * x1;
    const real_t x3 = trial_grad[1] * x2;
    const real_t x4 = test_grad[2] * x1;
    const real_t x5 = trial_grad[2] * x4;
    const real_t x6 = lambda + 2 * mu;
    const real_t x7 = test_grad[0] * x0;
    const real_t x8 = trial_grad[0] * x7;
    const real_t x9 = det_jac * ref_vol;
    const real_t x10 = test_grad[1] * x0;
    const real_t x11 = lambda * trial_grad[0];
    const real_t x12 = mu * x7;
    const real_t x13 = test_grad[2] * x0;
    const real_t x14 = lambda * trial_grad[1];
    const real_t x15 = mu * x8;
    const real_t x16 = lambda * trial_grad[2];
    element_matrix[0] = x9 * (x3 + x5 + x6 * x8);
    element_matrix[1] = x9 * (trial_grad[1] * x12 + x10 * x11);
    element_matrix[2] = x9 * (trial_grad[2] * x12 + x11 * x13);
    element_matrix[3] = x9 * (trial_grad[0] * x2 + x14 * x7);
    element_matrix[4] = x9 * (trial_grad[1] * x10 * x6 + x15 + x5);
    element_matrix[5] = x9 * (trial_grad[2] * x2 + x13 * x14);
    element_matrix[6] = x9 * (trial_grad[0] * x4 + x16 * x7);
    element_matrix[7] = x9 * (trial_grad[1] * x4 + x10 * x16);
    element_matrix[8] = x9 * (trial_grad[2] * x13 * x6 + x15 + x3);
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_hessian_uc(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_matrix) {
    // TODO COST
    const real_t x0 = test_fun * (2 * s_phase - 2);
    const real_t x1 = trial_grad[1] * x0;
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x3 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x4 = trial_grad[2] * x0;
    const real_t x5 = mu * x4;
    const real_t x6 = 2 * s_grad_disp[0];
    const real_t x7 = 2 * s_grad_disp[4];
    const real_t x8 = 2 * s_grad_disp[8];
    const real_t x9 = (1.0 / 2.0) * lambda * (x6 + x7 + x8);
    const real_t x10 = trial_grad[0] * x0;
    const real_t x11 = det_jac * ref_vol;
    const real_t x12 = s_grad_disp[5] + s_grad_disp[7];
    element_matrix[0] = x11 * (x1 * x2 + x10 * (mu * x6 + x9) + x3 * x5);
    element_matrix[1] = x11 * (x1 * (mu * x7 + x9) + x10 * x2 + x12 * x5);
    element_matrix[2] = x11 * (mu * x1 * x12 + mu * x10 * x3 + x4 * (mu * x8 + x9));
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_hessian_cu(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    real_t* SFEM_RESTRICT element_matrix) {
    // TODO COST
    const real_t x0 = 2 * s_phase - 2;
    const real_t x1 = test_grad[1] * x0;
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x3 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x4 = test_grad[2] * x0;
    const real_t x5 = mu * x4;
    const real_t x6 = 2 * s_grad_disp[0];
    const real_t x7 = 2 * s_grad_disp[4];
    const real_t x8 = 2 * s_grad_disp[8];
    const real_t x9 = (1.0 / 2.0) * lambda * (x6 + x7 + x8);
    const real_t x10 = test_grad[0] * x0;
    const real_t x11 = det_jac * ref_vol * trial_fun;
    const real_t x12 = s_grad_disp[5] + s_grad_disp[7];
    element_matrix[0] = x11 * (x1 * x2 + x10 * (mu * x6 + x9) + x3 * x5);
    element_matrix[1] = x11 * (x1 * (mu * x7 + x9) + x10 * x2 + x12 * x5);
    element_matrix[2] = x11 * (mu * x1 * x12 + mu * x10 * x3 + x4 * (mu * x8 + x9));
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_apply(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    const real_t* SFEM_RESTRICT increment,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = 2 * s_phase - 2;
    const real_t x1 = test_grad[1] * x0;
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x3 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x4 = test_grad[2] * x0;
    const real_t x5 = mu * x4;
    const real_t x6 = 2 * s_grad_disp[0];
    const real_t x7 = 2 * s_grad_disp[4];
    const real_t x8 = 2 * s_grad_disp[8];
    const real_t x9 = (1.0 / 2.0) * lambda * (x6 + x7 + x8);
    const real_t x10 = mu * x6 + x9;
    const real_t x11 = test_grad[0] * x0;
    const real_t x12 = det_jac * ref_vol;
    const real_t x13 = increment[1] * x12;
    const real_t x14 = s_grad_disp[5] + s_grad_disp[7];
    const real_t x15 = mu * x7 + x9;
    const real_t x16 = increment[2] * x12;
    const real_t x17 = mu * x3;
    const real_t x18 = mu * x14;
    const real_t x19 = mu * x8 + x9;
    const real_t x20 = increment[3] * x12;
    const real_t x21 = Gc * ls;
    const real_t x22 = test_grad[0] * trial_grad[0];
    const real_t x23 = test_grad[1] * trial_grad[1];
    const real_t x24 = test_grad[2] * trial_grad[2];
    const real_t x25 = 2 * mu;
    const real_t x26 = increment[0] * x12;
    const real_t x27 = pow(1 - s_phase, 2);
    const real_t x28 = lambda * x27;
    const real_t x29 = test_grad[1] * trial_grad[0];
    const real_t x30 = mu * x27;
    const real_t x31 = test_grad[0] * x30;
    const real_t x32 = test_grad[2] * x28;
    const real_t x33 = x23 * x30;
    const real_t x34 = x24 * x30;
    const real_t x35 = lambda + x25;
    const real_t x36 = x22 * x27;
    const real_t x37 = test_fun * x0;
    const real_t x38 = trial_grad[1] * x37;
    const real_t x39 = trial_grad[2] * x37;
    const real_t x40 = trial_grad[0] * x37;
    const real_t x41 = test_grad[0] * x28;
    const real_t x42 = test_grad[1] * trial_grad[2];
    const real_t x43 = mu * x36;
    const real_t x44 = x27 * x35;
    const real_t x45 = test_grad[2] * x30;
    element_vector[0] =
        trial_fun * x13 * (x1 * x2 + x10 * x11 + x3 * x5) +
        trial_fun * x16 * (x1 * x15 + x11 * x2 + x14 * x5) +
        trial_fun * x20 * (x1 * x18 + x11 * x17 + x19 * x4) +
        x26 *
            (test_fun * trial_fun *
                 (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[4] + s_grad_disp[8], 2) +
                  x25 * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[4], 2) + pow(s_grad_disp[8], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[3], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[2] + (1.0 / 2.0) * s_grad_disp[6], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[5] + (1.0 / 2.0) * s_grad_disp[7], 2))) +
             x21 * x22 + x21 * x23 + x21 * x24);
    element_vector[1] = x13 * (x33 + x34 + x35 * x36) + x16 * (trial_grad[1] * x31 + x28 * x29) +
                        x20 * (trial_grad[0] * x32 + trial_grad[2] * x31) +
                        x26 * (x10 * x40 + x17 * x39 + x2 * x38);
    element_vector[2] = x13 * (trial_grad[1] * x41 + x29 * x30) + x16 * (x23 * x44 + x34 + x43) +
                        x20 * (trial_grad[1] * x32 + x30 * x42) +
                        x26 * (x15 * x38 + x18 * x39 + x2 * x40);
    element_vector[3] = x13 * (trial_grad[0] * x45 + trial_grad[2] * x41) +
                        x16 * (trial_grad[1] * x45 + x28 * x42) + x20 * (x24 * x44 + x33 + x43) +
                        x26 * (x17 * x40 + x18 * x38 + x19 * x39);
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_apply_uu(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    const real_t* SFEM_RESTRICT increment,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = pow(1 - s_phase, 2);
    const real_t x1 = test_grad[1] * x0;
    const real_t x2 = lambda * trial_grad[0];
    const real_t x3 = test_grad[0] * x0;
    const real_t x4 = mu * x3;
    const real_t x5 = det_jac * ref_vol;
    const real_t x6 = increment[1] * x5;
    const real_t x7 = test_grad[2] * x0;
    const real_t x8 = increment[2] * x5;
    const real_t x9 = mu * x0;
    const real_t x10 = test_grad[1] * x9;
    const real_t x11 = trial_grad[1] * x10;
    const real_t x12 = test_grad[2] * x9;
    const real_t x13 = trial_grad[2] * x12;
    const real_t x14 = lambda + 2 * mu;
    const real_t x15 = trial_grad[0] * x3;
    const real_t x16 = increment[0] * x5;
    const real_t x17 = lambda * trial_grad[1];
    const real_t x18 = mu * x15;
    const real_t x19 = lambda * trial_grad[2];
    element_vector[0] = x16 * (x11 + x13 + x14 * x15) + x6 * (trial_grad[1] * x4 + x1 * x2) +
                        x8 * (trial_grad[2] * x4 + x2 * x7);
    element_vector[1] = x16 * (trial_grad[0] * x10 + x17 * x3) +
                        x6 * (trial_grad[1] * x1 * x14 + x13 + x18) +
                        x8 * (trial_grad[2] * x10 + x17 * x7);
    element_vector[2] = x16 * (trial_grad[0] * x12 + x19 * x3) +
                        x6 * (trial_grad[1] * x12 + x1 * x19) +
                        x8 * (trial_grad[2] * x14 * x7 + x11 + x18);
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_apply_cc(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    const real_t* SFEM_RESTRICT increment,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = Gc * ls;
    element_vector[0] =
        det_jac * increment[0] * ref_vol *
        (test_fun * trial_fun *
             (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[4] + s_grad_disp[8], 2) +
              2 * mu *
                  (pow(s_grad_disp[0], 2) + pow(s_grad_disp[4], 2) + pow(s_grad_disp[8], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[3], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[2] + (1.0 / 2.0) * s_grad_disp[6], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[5] + (1.0 / 2.0) * s_grad_disp[7], 2))) +
         test_grad[0] * trial_grad[0] * x0 + test_grad[1] * trial_grad[1] * x0 +
         test_grad[2] * trial_grad[2] * x0);
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_apply_uc(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    const real_t* SFEM_RESTRICT increment,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = test_fun * (2 * s_phase - 2);
    const real_t x1 = trial_grad[1] * x0;
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x3 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x4 = trial_grad[2] * x0;
    const real_t x5 = mu * x4;
    const real_t x6 = 2 * s_grad_disp[0];
    const real_t x7 = 2 * s_grad_disp[4];
    const real_t x8 = 2 * s_grad_disp[8];
    const real_t x9 = (1.0 / 2.0) * lambda * (x6 + x7 + x8);
    const real_t x10 = trial_grad[0] * x0;
    const real_t x11 = det_jac * increment[0] * ref_vol;
    const real_t x12 = s_grad_disp[5] + s_grad_disp[7];
    element_vector[0] = x11 * (x1 * x2 + x10 * (mu * x6 + x9) + x3 * x5);
    element_vector[1] = x11 * (x1 * (mu * x7 + x9) + x10 * x2 + x12 * x5);
    element_vector[2] = x11 * (mu * x1 * x12 + mu * x10 * x3 + x4 * (mu * x8 + x9));
}

SFEM_CUDA_INLINE static void FE3D_phase_field_for_fracture_apply_cu(
    // material parameters
    const real_t mu,
    const real_t lambda,
    const real_t Gc,
    const real_t ls,
    const real_t ref_vol,
    const real_t det_jac,
    const real_t s_phase,
    const real_t* SFEM_RESTRICT s_grad_phase,
    const real_t* SFEM_RESTRICT s_grad_disp,
    const real_t test_fun,
    const real_t* SFEM_RESTRICT test_grad,
    const real_t trial_fun,
    const real_t* SFEM_RESTRICT trial_grad,
    const real_t* SFEM_RESTRICT increment,
    real_t* SFEM_RESTRICT element_vector) {
    // TODO COST
    const real_t x0 = 2 * s_phase - 2;
    const real_t x1 = test_grad[1] * x0;
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[3]);
    const real_t x3 = s_grad_disp[2] + s_grad_disp[6];
    const real_t x4 = test_grad[2] * x0;
    const real_t x5 = mu * x4;
    const real_t x6 = 2 * s_grad_disp[0];
    const real_t x7 = 2 * s_grad_disp[4];
    const real_t x8 = 2 * s_grad_disp[8];
    const real_t x9 = (1.0 / 2.0) * lambda * (x6 + x7 + x8);
    const real_t x10 = test_grad[0] * x0;
    const real_t x11 = det_jac * ref_vol * trial_fun;
    const real_t x12 = s_grad_disp[5] + s_grad_disp[7];
    element_vector[0] = increment[0] * x11 * (x1 * x2 + x10 * (mu * x6 + x9) + x3 * x5) +
                        increment[1] * x11 * (x1 * (mu * x7 + x9) + x10 * x2 + x12 * x5) +
                        increment[2] * x11 * (mu * x1 * x12 + mu * x10 * x3 + x4 * (mu * x8 + x9));
}
