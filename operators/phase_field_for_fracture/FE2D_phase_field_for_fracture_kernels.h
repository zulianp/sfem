// Automatically generate code for phase_field_for_fracture
#include "math.h"
#include "sfem_base.h"
#include "sfem_vec.h"

#define SFEM_CUDA_INLINE SFEM_INLINE __device__ __host__ 

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_value(
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
             (ls * (pow(s_grad_phase[0], 2) + pow(s_grad_phase[1], 2)) + pow(s_phase, 2) / ls) +
         pow(1 - s_phase, 2) *
             ((1.0 / 2.0) * lambda * pow(s_grad_disp[0] + s_grad_disp[3], 2) +
              mu * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[3], 2) +
                    2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[2], 2))));
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_gradient(
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
    const real_t x5 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x6 = 2 * s_grad_disp[0];
    const real_t x7 = 2 * s_grad_disp[3];
    const real_t x8 = x1 * (x6 + x7);
    const real_t x9 = test_grad[0] * x3;
    element_vector[0] =
        x2 *
        (s_grad_phase[0] * test_grad[0] * x0 + s_grad_phase[1] * test_grad[1] * x0 +
         test_fun *
             (Gc * s_phase / ls +
              (2 * s_phase - 2) *
                  (mu * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[3], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[2], 2)) +
                   x1 * pow(s_grad_disp[0] + s_grad_disp[3], 2))));
    element_vector[1] = x2 * (x4 * x5 + x9 * (mu * x6 + x8));
    element_vector[2] = x2 * (x4 * (mu * x7 + x8) + x5 * x9);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_hessian(
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
    const real_t x3 = 2 * mu;
    const real_t x4 = det_jac * ref_vol;
    const real_t x5 = 2 * s_phase - 2;
    const real_t x6 = test_grad[1] * x5;
    const real_t x7 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x8 = 2 * s_grad_disp[0];
    const real_t x9 = 2 * s_grad_disp[3];
    const real_t x10 = (1.0 / 2.0) * lambda * (x8 + x9);
    const real_t x11 = mu * x8 + x10;
    const real_t x12 = test_grad[0] * x5;
    const real_t x13 = trial_fun * x4;
    const real_t x14 = mu * x9 + x10;
    const real_t x15 = test_fun * x5;
    const real_t x16 = trial_grad[1] * x15;
    const real_t x17 = trial_grad[0] * x15;
    const real_t x18 = pow(1 - s_phase, 2);
    const real_t x19 = mu * x18;
    const real_t x20 = lambda + x3;
    const real_t x21 = x1 * x18;
    const real_t x22 = lambda * x18;
    const real_t x23 = test_grad[1] * trial_grad[0];
    const real_t x24 = test_grad[0] * trial_grad[1];
    element_matrix[0] =
        x4 * (test_fun * trial_fun *
                  (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[3], 2) +
                   x3 * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[3], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[2], 2))) +
              x0 * x1 + x0 * x2);
    element_matrix[1] = x13 * (x11 * x12 + x6 * x7);
    element_matrix[2] = x13 * (x12 * x7 + x14 * x6);
    element_matrix[3] = x4 * (x11 * x17 + x16 * x7);
    element_matrix[4] = x4 * (x19 * x2 + x20 * x21);
    element_matrix[5] = x4 * (x19 * x24 + x22 * x23);
    element_matrix[6] = x4 * (x14 * x16 + x17 * x7);
    element_matrix[7] = x4 * (x19 * x23 + x22 * x24);
    element_matrix[8] = x4 * (mu * x21 + x18 * x2 * x20);
}
// Split evaliations

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_gradient_u(
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
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x3 = 2 * s_grad_disp[0];
    const real_t x4 = 2 * s_grad_disp[3];
    const real_t x5 = (1.0 / 2.0) * lambda * (x3 + x4);
    const real_t x6 = test_grad[0] * x0;
    const real_t x7 = det_jac * ref_vol;
    element_vector[0] = x7 * (x1 * x2 + x6 * (mu * x3 + x5));
    element_vector[1] = x7 * (x1 * (mu * x4 + x5) + x2 * x6);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_gradient_c(
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
         test_fun * (Gc * s_phase / ls +
                     (2 * s_phase - 2) *
                         ((1.0 / 2.0) * lambda * pow(s_grad_disp[0] + s_grad_disp[3], 2) +
                          mu * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[3], 2) +
                                2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[2],
                                        2)))));
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_hessian_cc(
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
             (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[3], 2) +
              2 * mu *
                  (pow(s_grad_disp[0], 2) + pow(s_grad_disp[3], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[2], 2))) +
         test_grad[0] * trial_grad[0] * x0 + test_grad[1] * trial_grad[1] * x0);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_hessian_uu(
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
    const real_t x0 = mu * trial_grad[1];
    const real_t x1 = pow(1 - s_phase, 2);
    const real_t x2 = test_grad[1] * x1;
    const real_t x3 = lambda + 2 * mu;
    const real_t x4 = test_grad[0] * x1;
    const real_t x5 = trial_grad[0] * x4;
    const real_t x6 = det_jac * ref_vol;
    const real_t x7 = trial_grad[0] * x2;
    element_matrix[0] = x6 * (x0 * x2 + x3 * x5);
    element_matrix[1] = x6 * (lambda * x7 + x0 * x4);
    element_matrix[2] = x6 * (lambda * trial_grad[1] * x4 + mu * x7);
    element_matrix[3] = x6 * (mu * x5 + trial_grad[1] * x2 * x3);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_hessian_uc(
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
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x3 = 2 * s_grad_disp[0];
    const real_t x4 = 2 * s_grad_disp[3];
    const real_t x5 = (1.0 / 2.0) * lambda * (x3 + x4);
    const real_t x6 = trial_grad[0] * x0;
    const real_t x7 = det_jac * ref_vol;
    element_matrix[0] = x7 * (x1 * x2 + x6 * (mu * x3 + x5));
    element_matrix[1] = x7 * (x1 * (mu * x4 + x5) + x2 * x6);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_hessian_cu(
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
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x3 = 2 * s_grad_disp[0];
    const real_t x4 = 2 * s_grad_disp[3];
    const real_t x5 = (1.0 / 2.0) * lambda * (x3 + x4);
    const real_t x6 = test_grad[0] * x0;
    const real_t x7 = det_jac * ref_vol * trial_fun;
    element_matrix[0] = x7 * (x1 * x2 + x6 * (mu * x3 + x5));
    element_matrix[1] = x7 * (x1 * (mu * x4 + x5) + x2 * x6);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_apply(
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
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x3 = 2 * s_grad_disp[0];
    const real_t x4 = 2 * s_grad_disp[3];
    const real_t x5 = (1.0 / 2.0) * lambda * (x3 + x4);
    const real_t x6 = mu * x3 + x5;
    const real_t x7 = test_grad[0] * x0;
    const real_t x8 = det_jac * ref_vol;
    const real_t x9 = increment[1] * x8;
    const real_t x10 = mu * x4 + x5;
    const real_t x11 = increment[2] * x8;
    const real_t x12 = Gc * ls;
    const real_t x13 = test_grad[0] * trial_grad[0];
    const real_t x14 = test_grad[1] * trial_grad[1];
    const real_t x15 = 2 * mu;
    const real_t x16 = increment[0] * x8;
    const real_t x17 = pow(1 - s_phase, 2);
    const real_t x18 = lambda * x17;
    const real_t x19 = test_grad[1] * trial_grad[0];
    const real_t x20 = mu * x17;
    const real_t x21 = test_grad[0] * trial_grad[1];
    const real_t x22 = lambda + x15;
    const real_t x23 = x13 * x17;
    const real_t x24 = test_fun * x0;
    const real_t x25 = trial_grad[1] * x24;
    const real_t x26 = trial_grad[0] * x24;
    element_vector[0] =
        trial_fun * x11 * (x1 * x10 + x2 * x7) + trial_fun * x9 * (x1 * x2 + x6 * x7) +
        x16 *
            (test_fun * trial_fun *
                 (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[3], 2) +
                  x15 * (pow(s_grad_disp[0], 2) + pow(s_grad_disp[3], 2) +
                         2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[2], 2))) +
             x12 * x13 + x12 * x14);
    element_vector[1] =
        x11 * (x18 * x19 + x20 * x21) + x16 * (x2 * x25 + x26 * x6) + x9 * (x14 * x20 + x22 * x23);
    element_vector[2] = x11 * (mu * x23 + x14 * x17 * x22) + x16 * (x10 * x25 + x2 * x26) +
                        x9 * (x18 * x21 + x19 * x20);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_apply_uu(
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
    const real_t x2 = trial_grad[0] * x1;
    const real_t x3 = test_grad[0] * x0;
    const real_t x4 = mu * trial_grad[1];
    const real_t x5 = det_jac * ref_vol;
    const real_t x6 = increment[1] * x5;
    const real_t x7 = lambda + 2 * mu;
    const real_t x8 = trial_grad[0] * x3;
    const real_t x9 = increment[0] * x5;
    element_vector[0] = x6 * (lambda * x2 + x3 * x4) + x9 * (x1 * x4 + x7 * x8);
    element_vector[1] =
        x6 * (mu * x8 + trial_grad[1] * x1 * x7) + x9 * (lambda * trial_grad[1] * x3 + mu * x2);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_apply_cc(
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
             (Gc / ls + lambda * pow(s_grad_disp[0] + s_grad_disp[3], 2) +
              2 * mu *
                  (pow(s_grad_disp[0], 2) + pow(s_grad_disp[3], 2) +
                   2 * pow((1.0 / 2.0) * s_grad_disp[1] + (1.0 / 2.0) * s_grad_disp[2], 2))) +
         test_grad[0] * trial_grad[0] * x0 + test_grad[1] * trial_grad[1] * x0);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_apply_uc(
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
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x3 = 2 * s_grad_disp[0];
    const real_t x4 = 2 * s_grad_disp[3];
    const real_t x5 = (1.0 / 2.0) * lambda * (x3 + x4);
    const real_t x6 = trial_grad[0] * x0;
    const real_t x7 = det_jac * increment[0] * ref_vol;
    element_vector[0] = x7 * (x1 * x2 + x6 * (mu * x3 + x5));
    element_vector[1] = x7 * (x1 * (mu * x4 + x5) + x2 * x6);
}

SFEM_CUDA_INLINE static void FE2D_phase_field_for_fracture_apply_cu(
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
    const real_t x2 = mu * (s_grad_disp[1] + s_grad_disp[2]);
    const real_t x3 = 2 * s_grad_disp[0];
    const real_t x4 = 2 * s_grad_disp[3];
    const real_t x5 = (1.0 / 2.0) * lambda * (x3 + x4);
    const real_t x6 = test_grad[0] * x0;
    const real_t x7 = det_jac * ref_vol * trial_fun;
    element_vector[0] = increment[0] * x7 * (x1 * x2 + x6 * (mu * x3 + x5)) +
                        increment[1] * x7 * (x1 * (mu * x4 + x5) + x2 * x6);
}
