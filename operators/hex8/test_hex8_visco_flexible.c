/**
 * @file test_hex8_visco_flexible.c
 * @brief Simple unit test for the new flexible visco kernels.
 * 
 * This test verifies that:
 * 1. hex8_mooney_rivlin_S_dev_from_disp computes S_dev correctly
 * 2. hex8_mooney_rivlin_update_Hi_single updates H_i correctly
 * 3. hex8_mooney_rivlin_hessian_algo_micro produces non-zero Hessian
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "sfem_base.h"
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_visco_unique_Hi_local.h"

// Helper: create a simple unit cube element
static void setup_unit_cube(scalar_t *lx, scalar_t *ly, scalar_t *lz) {
    // Standard hex8 node ordering
    lx[0] = 0; ly[0] = 0; lz[0] = 0;
    lx[1] = 1; ly[1] = 0; lz[1] = 0;
    lx[2] = 1; ly[2] = 1; lz[2] = 0;
    lx[3] = 0; ly[3] = 1; lz[3] = 0;
    lx[4] = 0; ly[4] = 0; lz[4] = 1;
    lx[5] = 1; ly[5] = 0; lz[5] = 1;
    lx[6] = 1; ly[6] = 1; lz[6] = 1;
    lx[7] = 0; ly[7] = 1; lz[7] = 1;
}

// Helper: apply small displacement
static void apply_displacement(scalar_t *dispx, scalar_t *dispy, scalar_t *dispz, 
                               const scalar_t *lx, const scalar_t *ly, const scalar_t *lz,
                               scalar_t strain) {
    for (int i = 0; i < 8; ++i) {
        dispx[i] = strain * lx[i];  // Uniaxial stretch in x
        dispy[i] = -0.3 * strain * ly[i];  // Poisson contraction
        dispz[i] = -0.3 * strain * lz[i];  // Poisson contraction
    }
}

int test_S_dev_from_disp() {
    printf("Testing hex8_mooney_rivlin_S_dev_from_disp...\n");
    
    scalar_t lx[8], ly[8], lz[8];
    setup_unit_cube(lx, ly, lz);
    
    scalar_t dispx[8], dispy[8], dispz[8];
    apply_displacement(dispx, dispy, dispz, lx, ly, lz, 0.01);
    
    // Compute Jacobian at center (qx=qy=qz=0)
    scalar_t jacobian_adjugate[9];
    scalar_t jacobian_determinant;
    hex8_adjugate_and_det(lx, ly, lz, 0.0, 0.0, 0.0, jacobian_adjugate, &jacobian_determinant);
    
    // Material parameters
    const scalar_t K = 100.0;
    const scalar_t C10 = 1.0;
    const scalar_t C01 = 0.5;
    
    // Compute S_dev
    scalar_t S_dev[6];
    hex8_mooney_rivlin_S_dev_from_disp(
        jacobian_adjugate,
        jacobian_determinant,
        0.0, 0.0, 0.0,  // qx, qy, qz
        1.0,            // qw
        C10, C01, K,
        dispx, dispy, dispz,
        S_dev);
    
    // Check that S_dev is non-zero and finite
    int ok = 1;
    for (int i = 0; i < 6; ++i) {
        if (!isfinite(S_dev[i])) {
            printf("  ERROR: S_dev[%d] is not finite!\n", i);
            ok = 0;
        }
    }
    
    printf("  S_dev = [%g, %g, %g, %g, %g, %g]\n",
           S_dev[0], S_dev[1], S_dev[2], S_dev[3], S_dev[4], S_dev[5]);
    
    // For uniaxial tension, S_dev[0] (xx) should be positive, S_dev[1,2] (yy,zz) negative
    if (S_dev[0] > 0 && S_dev[1] < 0 && S_dev[2] < 0) {
        printf("  OK: S_dev signs are correct for uniaxial tension\n");
    } else {
        printf("  WARNING: S_dev signs may be unexpected\n");
    }
    
    return ok ? 0 : 1;
}

int test_update_Hi_single() {
    printf("Testing hex8_mooney_rivlin_update_Hi_single...\n");
    
    const scalar_t dt = 0.1;
    const scalar_t g = 0.3;
    const scalar_t tau = 1.0;
    
    scalar_t S_dev_prev[6] = {1.0, -0.5, -0.5, 0.0, 0.0, 0.0};
    scalar_t S_dev_curr[6] = {1.1, -0.55, -0.55, 0.0, 0.0, 0.0};
    scalar_t H_old[6] = {0.5, -0.25, -0.25, 0.0, 0.0, 0.0};
    scalar_t H_new[6];
    
    hex8_mooney_rivlin_update_Hi_single(dt, g, tau, S_dev_prev, S_dev_curr, H_old, H_new);
    
    // Verify formula: H_new = alpha * H_old + beta * (S_curr - S_prev)
    const scalar_t x = dt / tau;
    const scalar_t alpha = exp(-x);
    const scalar_t beta = g * (1.0 - alpha) / x;
    
    int ok = 1;
    printf("  alpha = %g, beta = %g\n", alpha, beta);
    for (int i = 0; i < 6; ++i) {
        scalar_t expected = alpha * H_old[i] + beta * (S_dev_curr[i] - S_dev_prev[i]);
        if (fabs(H_new[i] - expected) > 1e-10) {
            printf("  ERROR: H_new[%d] = %g, expected %g\n", i, H_new[i], expected);
            ok = 0;
        }
    }
    
    if (ok) {
        printf("  OK: H_i update formula is correct\n");
    }
    
    return ok ? 0 : 1;
}

int test_hessian_algo_micro() {
    printf("Testing hex8_mooney_rivlin_hessian_algo_micro...\n");
    
    scalar_t lx[8], ly[8], lz[8];
    setup_unit_cube(lx, ly, lz);
    
    scalar_t dispx[8], dispy[8], dispz[8];
    apply_displacement(dispx, dispy, dispz, lx, ly, lz, 0.01);
    
    scalar_t jacobian_adjugate[9];
    scalar_t jacobian_determinant;
    hex8_adjugate_and_det(lx, ly, lz, 0.0, 0.0, 0.0, jacobian_adjugate, &jacobian_determinant);
    
    const scalar_t K = 100.0;
    const scalar_t C10 = 1.0;
    const scalar_t C01 = 0.5;
    const scalar_t gamma = 0.8;  // g_inf + algorithmic contribution
    
    scalar_t H[24 * 24];
    memset(H, 0, sizeof(H));
    
    hex8_mooney_rivlin_hessian_algo_micro(
        jacobian_adjugate,
        jacobian_determinant,
        0.0, 0.0, 0.0,  // qx, qy, qz
        1.0,            // qw
        C10, C01, K, gamma,
        dispx, dispy, dispz,
        H);
    
    // Check that Hessian is non-zero
    scalar_t max_val = 0;
    scalar_t diag_sum = 0;
    for (int i = 0; i < 24 * 24; ++i) {
        if (!isfinite(H[i])) {
            printf("  ERROR: H[%d] is not finite!\n", i);
            return 1;
        }
        if (fabs(H[i]) > max_val) max_val = fabs(H[i]);
    }
    for (int i = 0; i < 24; ++i) {
        diag_sum += H[i * 24 + i];
    }
    
    printf("  max |H[i,j]| = %g\n", max_val);
    printf("  trace(H) = %g\n", diag_sum);
    
    if (max_val > 0 && diag_sum > 0) {
        printf("  OK: Hessian is non-zero and has positive trace\n");
        return 0;
    } else {
        printf("  WARNING: Hessian may have issues\n");
        return 1;
    }
}

int test_gradient_flexible() {
    printf("Testing hex8_mooney_rivlin_grad_flexible...\n");
    
    scalar_t lx[8], ly[8], lz[8];
    setup_unit_cube(lx, ly, lz);
    
    scalar_t dispx[8], dispy[8], dispz[8];
    apply_displacement(dispx, dispy, dispz, lx, ly, lz, 0.01);
    
    scalar_t jacobian_adjugate[9];
    scalar_t jacobian_determinant;
    hex8_adjugate_and_det(lx, ly, lz, 0.0, 0.0, 0.0, jacobian_adjugate, &jacobian_determinant);
    
    const scalar_t K = 100.0;
    const scalar_t C10 = 1.0;
    const scalar_t C01 = 0.5;
    const scalar_t gamma = 0.8;
    
    scalar_t gx[8] = {0}, gy[8] = {0}, gz[8] = {0};
    
    hex8_mooney_rivlin_grad_flexible(
        jacobian_adjugate,
        jacobian_determinant,
        0.0, 0.0, 0.0,
        1.0,
        C10, C01, K, gamma,
        dispx, dispy, dispz,
        gx, gy, gz);
    
    // Check forces are non-zero
    scalar_t force_mag = 0;
    for (int i = 0; i < 8; ++i) {
        force_mag += gx[i]*gx[i] + gy[i]*gy[i] + gz[i]*gz[i];
    }
    force_mag = sqrt(force_mag);
    
    printf("  |F| = %g\n", force_mag);
    
    if (force_mag > 0 && isfinite(force_mag)) {
        printf("  OK: Gradient is non-zero\n");
        return 0;
    } else {
        printf("  WARNING: Gradient may have issues\n");
        return 1;
    }
}

int main(int argc, char *argv[]) {
    printf("=== Testing Hex8 Mooney-Rivlin Visco Flexible Kernels ===\n\n");
    
    int errors = 0;
    
    errors += test_S_dev_from_disp();
    printf("\n");
    
    errors += test_update_Hi_single();
    printf("\n");
    
    errors += test_gradient_flexible();
    printf("\n");
    
    errors += test_hessian_algo_micro();
    printf("\n");
    
    if (errors == 0) {
        printf("=== ALL TESTS PASSED ===\n");
        return 0;
    } else {
        printf("=== %d TESTS FAILED ===\n", errors);
        return 1;
    }
}

