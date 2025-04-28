#ifndef CU_HEX8_LINEAR_ELASTICITY_MATRIX_INLINE_HPP
#define CU_HEX8_LINEAR_ELASTICITY_MATRIX_INLINE_HPP

#include "cu_hex8_inline.hpp"

template <typename scalar_t>
static inline __host__ __device__ void cu_hex8_ref_shape_fun(const int                     i,
                                                             const scalar_t                qx,
                                                             const scalar_t                qy,
                                                             const scalar_t                qz,
                                                             scalar_t *const SFEM_RESTRICT val) {
    switch (i) {
        case 0: {
            val[0] = (1 - qx) * (1 - qy) * (1 - qz);
            return;
        }
        case 1: {
            val[0] = qx * (1 - qy) * (1 - qz);
            return;
        }
        case 2: {
            val[0] = qx * qy * (1 - qz);
            return;
        }
        case 3: {
            val[0] = (1 - qx) * qy * (1 - qz);
            return;
        }
        case 4: {
            val[0] = (1 - qx) * (1 - qy) * qz;
            return;
        }
        case 5: {
            val[0] = qx * (1 - qy) * qz;
            return;
        }
        case 6: {
            val[0] = qx * qy * qz;
            return;
        }
        case 7: {
            val[0] = (1 - qx) * qy * qz;
            return;
        }
        default: {
            val[0] = 0;
            assert(false);
            return;
        }
    }
}

template <typename scalar_t>
static inline __host__ __device__ void cu_hex8_ref_shape_grad(const int                     i,
                                                              const scalar_t                qx,
                                                              const scalar_t                qy,
                                                              const scalar_t                qz,
                                                              scalar_t *const SFEM_RESTRICT val) {
    switch (i) {
        case 0: {
            const scalar_t x0 = 1 - qy;
            const scalar_t x1 = 1 - qz;
            const scalar_t x2 = 1 - qx;
            val[0]            = -x0 * x1;
            val[1]            = -x1 * x2;
            val[2]            = -x0 * x2;
            return;
        }
        case 1: {
            const scalar_t x0 = 1 - qy;
            const scalar_t x1 = 1 - qz;
            val[0]            = x0 * x1;
            val[1]            = -qx * x1;
            val[2]            = -qx * x0;
            return;
        }
        case 2: {
            const scalar_t x0 = 1 - qz;
            val[0]            = qy * x0;
            val[1]            = qx * x0;
            val[2]            = -qx * qy;
            return;
        }
        case 3: {
            const scalar_t x0 = 1 - qz;
            const scalar_t x1 = 1 - qx;
            val[0]            = -qy * x0;
            val[1]            = x0 * x1;
            val[2]            = -qy * x1;
            return;
        }
        case 4: {
            const scalar_t x0 = 1 - qy;
            const scalar_t x1 = 1 - qx;
            val[0]            = -qz * x0;
            val[1]            = -qz * x1;
            val[2]            = x0 * x1;
            return;
        }
        case 5: {
            const scalar_t x0 = 1 - qy;
            val[0]            = qz * x0;
            val[1]            = -qx * qz;
            val[2]            = qx * x0;
            return;
        }
        case 6: {
            val[0] = qy * qz;
            val[1] = qx * qz;
            val[2] = qx * qy;
            return;
        }
        case 7: {
            const scalar_t x0 = 1 - qx;
            val[0]            = -qy * qz;
            val[1]            = qz * x0;
            val[2]            = qy * x0;
            return;
        }
        default: {
            val[0] = 0;
            val[1] = 0;
            val[2] = 0;
            assert(false);
            return;
        }
    }
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_ref_shape_grads(const scalar_t                qx,
                                                               const scalar_t                qy,
                                                               const scalar_t                qz,
                                                               scalar_t *const SFEM_RESTRICT val) {
    const scalar_t x0  = 1 - qy;
    const scalar_t x1  = 1 - qz;
    const scalar_t x2  = x0 * x1;
    const scalar_t x3  = 1 - qx;
    const scalar_t x4  = x1 * x3;
    const scalar_t x5  = x0 * x3;
    const scalar_t x6  = qx * x1;
    const scalar_t x7  = qx * x0;
    const scalar_t x8  = qy * x1;
    const scalar_t x9  = qx * qy;
    const scalar_t x10 = qy * x3;
    const scalar_t x11 = qz * x0;
    const scalar_t x12 = qz * x3;
    const scalar_t x13 = qx * qz;
    const scalar_t x14 = qy * qz;
    val[0]             = -x2;
    val[1]             = -x4;
    val[2]             = -x5;
    val[3]             = x2;
    val[4]             = -x6;
    val[5]             = -x7;
    val[6]             = x8;
    val[7]             = x6;
    val[8]             = -x9;
    val[9]             = -x8;
    val[10]            = x4;
    val[11]            = -x10;
    val[12]            = -x11;
    val[13]            = -x12;
    val[14]            = x5;
    val[15]            = x11;
    val[16]            = -x13;
    val[17]            = x7;
    val[18]            = x14;
    val[19]            = x13;
    val[20]            = x9;
    val[21]            = -x14;
    val[22]            = x12;
    val[23]            = x10;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_linear_elasticity_matrix_block(const scalar_t                      mu,
                                                                         const scalar_t                      lambda,
                                                                         const scalar_t *const SFEM_RESTRICT adjugate,
                                                                         const scalar_t                      jacobian_determinant,
                                                                         const scalar_t                      qw,
                                                                         const scalar_t *const               trial_grad,
                                                                         const scalar_t *const               test_grad,
                                                                         scalar_t *const SFEM_RESTRICT       block) {
    const scalar_t x0  = 1.0 / jacobian_determinant;
    const scalar_t x1  = mu * x0;
    const scalar_t x2  = adjugate[1] * test_grad[0];
    const scalar_t x3  = test_grad[1] * x1;
    const scalar_t x4  = test_grad[2] * x1;
    const scalar_t x5  = x0 * (adjugate[4] * x3 + adjugate[7] * x4 + x1 * x2);
    const scalar_t x6  = adjugate[1] * x5;
    const scalar_t x7  = adjugate[2] * test_grad[0];
    const scalar_t x8  = x0 * (adjugate[5] * x3 + adjugate[8] * x4 + x1 * x7);
    const scalar_t x9  = adjugate[2] * x8;
    const scalar_t x10 = x0 * (lambda + 2 * mu);
    const scalar_t x11 = adjugate[0] * test_grad[0];
    const scalar_t x12 = adjugate[3] * test_grad[1];
    const scalar_t x13 = adjugate[6] * test_grad[2];
    const scalar_t x14 = x0 * (x10 * x11 + x10 * x12 + x10 * x13);
    const scalar_t x15 = adjugate[4] * x5;
    const scalar_t x16 = adjugate[5] * x8;
    const scalar_t x17 = adjugate[7] * x5;
    const scalar_t x18 = adjugate[8] * x8;
    const scalar_t x19 = jacobian_determinant * qw;
    const scalar_t x20 = lambda * x0;
    const scalar_t x21 = x0 * (x11 * x20 + x12 * x20 + x13 * x20);
    const scalar_t x22 = adjugate[4] * test_grad[1];
    const scalar_t x23 = adjugate[7] * test_grad[2];
    const scalar_t x24 = x0 * (x2 * x20 + x20 * x22 + x20 * x23);
    const scalar_t x25 = x0 * (x1 * x11 + x1 * x12 + x1 * x13);
    const scalar_t x26 = adjugate[0] * x25;
    const scalar_t x27 = x0 * (x10 * x2 + x10 * x22 + x10 * x23);
    const scalar_t x28 = adjugate[3] * x25;
    const scalar_t x29 = adjugate[6] * x25;
    const scalar_t x30 = adjugate[5] * test_grad[1];
    const scalar_t x31 = adjugate[8] * test_grad[2];
    const scalar_t x32 = x0 * (x20 * x30 + x20 * x31 + x20 * x7);
    const scalar_t x33 = x0 * (x10 * x30 + x10 * x31 + x10 * x7);

    block[0] += x19 * (trial_grad[0] * (adjugate[0] * x14 + x6 + x9) + trial_grad[1] * (adjugate[3] * x14 + x15 + x16) +
                       trial_grad[2] * (adjugate[6] * x14 + x17 + x18));
    block[1] += x19 *
                (trial_grad[0] * (adjugate[0] * x5 + adjugate[1] * x21) + trial_grad[1] * (adjugate[3] * x5 + adjugate[4] * x21) +
                 trial_grad[2] * (adjugate[6] * x5 + adjugate[7] * x21));
    block[2] += x19 *
                (trial_grad[0] * (adjugate[0] * x8 + adjugate[2] * x21) + trial_grad[1] * (adjugate[3] * x8 + adjugate[5] * x21) +
                 trial_grad[2] * (adjugate[6] * x8 + adjugate[8] * x21));
    block[3] += x19 * (trial_grad[0] * (adjugate[0] * x24 + adjugate[1] * x25) +
                       trial_grad[1] * (adjugate[3] * x24 + adjugate[4] * x25) +
                       trial_grad[2] * (adjugate[6] * x24 + adjugate[7] * x25));
    block[4] += x19 * (trial_grad[0] * (adjugate[1] * x27 + x26 + x9) + trial_grad[1] * (adjugate[4] * x27 + x16 + x28) +
                       trial_grad[2] * (adjugate[7] * x27 + x18 + x29));
    block[5] += x19 *
                (trial_grad[0] * (adjugate[1] * x8 + adjugate[2] * x24) + trial_grad[1] * (adjugate[4] * x8 + adjugate[5] * x24) +
                 trial_grad[2] * (adjugate[7] * x8 + adjugate[8] * x24));
    block[6] += x19 * (trial_grad[0] * (adjugate[0] * x32 + adjugate[2] * x25) +
                       trial_grad[1] * (adjugate[3] * x32 + adjugate[5] * x25) +
                       trial_grad[2] * (adjugate[6] * x32 + adjugate[8] * x25));
    block[7] += x19 *
                (trial_grad[0] * (adjugate[1] * x32 + adjugate[2] * x5) + trial_grad[1] * (adjugate[4] * x32 + adjugate[5] * x5) +
                 trial_grad[2] * (adjugate[7] * x32 + adjugate[8] * x5));
    block[8] += x19 * (trial_grad[0] * (adjugate[2] * x33 + x26 + x6) + trial_grad[1] * (adjugate[5] * x33 + x15 + x28) +
                       trial_grad[2] * (adjugate[8] * x33 + x17 + x29));
}

template <typename scalar_t>
static inline __device__ __host__ void cu_linear_elasticity_matrix_diag_block(const scalar_t                      mu,
                                                                              const scalar_t                      lambda,
                                                                              const scalar_t *const SFEM_RESTRICT adjugate,
                                                                              const scalar_t                jacobian_determinant,
                                                                              const scalar_t                qw,
                                                                              const scalar_t *const         test_grad,
                                                                              scalar_t *const SFEM_RESTRICT block) {
    const scalar_t x0  = adjugate[1] * test_grad[0] + adjugate[4] * test_grad[1] + adjugate[7] * test_grad[2];
    const scalar_t x1  = mu * x0;
    const scalar_t x2  = adjugate[1] * x1;
    const scalar_t x3  = adjugate[2] * test_grad[0] + adjugate[5] * test_grad[1] + adjugate[8] * test_grad[2];
    const scalar_t x4  = mu * x3;
    const scalar_t x5  = adjugate[2] * x4;
    const scalar_t x6  = lambda + 2 * mu;
    const scalar_t x7  = adjugate[0] * test_grad[0] + adjugate[3] * test_grad[1] + adjugate[6] * test_grad[2];
    const scalar_t x8  = x6 * x7;
    const scalar_t x9  = adjugate[4] * x1;
    const scalar_t x10 = adjugate[5] * x4;
    const scalar_t x11 = adjugate[7] * x1;
    const scalar_t x12 = adjugate[8] * x4;
    const scalar_t x13 = qw / jacobian_determinant;
    const scalar_t x14 = lambda * x7;
    const scalar_t x15 = lambda * x0;
    const scalar_t x16 = mu * x7;
    const scalar_t x17 = adjugate[0] * x16;
    const scalar_t x18 = x0 * x6;
    const scalar_t x19 = adjugate[3] * x16;
    const scalar_t x20 = adjugate[6] * x16;
    const scalar_t x21 = lambda * x3;
    const scalar_t x22 = x3 * x6;

    block[0] += x13 * (test_grad[0] * (adjugate[0] * x8 + x2 + x5) + test_grad[1] * (adjugate[3] * x8 + x10 + x9) +
                       test_grad[2] * (adjugate[6] * x8 + x11 + x12));
    block[1] +=
            x13 * (test_grad[0] * (adjugate[0] * x1 + adjugate[1] * x14) + test_grad[1] * (adjugate[3] * x1 + adjugate[4] * x14) +
                   test_grad[2] * (adjugate[6] * x1 + adjugate[7] * x14));
    block[2] +=
            x13 * (test_grad[0] * (adjugate[0] * x4 + adjugate[2] * x14) + test_grad[1] * (adjugate[3] * x4 + adjugate[5] * x14) +
                   test_grad[2] * (adjugate[6] * x4 + adjugate[8] * x14));
    block[3] += x13 *
                (test_grad[0] * (adjugate[0] * x15 + adjugate[1] * x16) + test_grad[1] * (adjugate[3] * x15 + adjugate[4] * x16) +
                 test_grad[2] * (adjugate[6] * x15 + adjugate[7] * x16));
    block[4] += x13 * (test_grad[0] * (adjugate[1] * x18 + x17 + x5) + test_grad[1] * (adjugate[4] * x18 + x10 + x19) +
                       test_grad[2] * (adjugate[7] * x18 + x12 + x20));
    block[5] +=
            x13 * (test_grad[0] * (adjugate[1] * x4 + adjugate[2] * x15) + test_grad[1] * (adjugate[4] * x4 + adjugate[5] * x15) +
                   test_grad[2] * (adjugate[7] * x4 + adjugate[8] * x15));
    block[6] += x13 *
                (test_grad[0] * (adjugate[0] * x21 + adjugate[2] * x16) + test_grad[1] * (adjugate[3] * x21 + adjugate[5] * x16) +
                 test_grad[2] * (adjugate[6] * x21 + adjugate[8] * x16));
    block[7] +=
            x13 * (test_grad[0] * (adjugate[1] * x21 + adjugate[2] * x1) + test_grad[1] * (adjugate[4] * x21 + adjugate[5] * x1) +
                   test_grad[2] * (adjugate[7] * x21 + adjugate[8] * x1));
    block[8] += x13 * (test_grad[0] * (adjugate[2] * x22 + x17 + x2) + test_grad[1] * (adjugate[5] * x22 + x19 + x9) +
                       test_grad[2] * (adjugate[8] * x22 + x11 + x20));
}


template<typename scalar_t>
static __device__ void cu_linear_elasticity_matrix_sym(const scalar_t                mu,
                                                       const scalar_t                lambda,
                                                       const scalar_t *SFEM_RESTRICT adjugate,
                                                       const scalar_t                jacobian_determinant,
                                                       const scalar_t *SFEM_RESTRICT trial_grad,
                                                       const scalar_t *SFEM_RESTRICT test_grad,
                                                       const scalar_t                qw,
                                                       scalar_t *const SFEM_RESTRICT element_matrix) {
    const scalar_t x0  = 1.0 / jacobian_determinant;
    const scalar_t x1  = mu * x0;
    const scalar_t x2  = adjugate[1] * test_grad[0];
    const scalar_t x3  = test_grad[1] * x1;
    const scalar_t x4  = test_grad[2] * x1;
    const scalar_t x5  = x0 * (adjugate[4] * x3 + adjugate[7] * x4 + x1 * x2);
    const scalar_t x6  = adjugate[1] * x5;
    const scalar_t x7  = adjugate[2] * test_grad[0];
    const scalar_t x8  = x0 * (adjugate[5] * x3 + adjugate[8] * x4 + x1 * x7);
    const scalar_t x9  = adjugate[2] * x8;
    const scalar_t x10 = x0 * (lambda + 2 * mu);
    const scalar_t x11 = adjugate[0] * test_grad[0];
    const scalar_t x12 = adjugate[3] * test_grad[1];
    const scalar_t x13 = adjugate[6] * test_grad[2];
    const scalar_t x14 = x0 * (x10 * x11 + x10 * x12 + x10 * x13);
    const scalar_t x15 = adjugate[4] * x5;
    const scalar_t x16 = adjugate[5] * x8;
    const scalar_t x17 = adjugate[7] * x5;
    const scalar_t x18 = adjugate[8] * x8;
    const scalar_t x19 = jacobian_determinant * qw;
    const scalar_t x20 = lambda * x0;
    const scalar_t x21 = x0 * (x11 * x20 + x12 * x20 + x13 * x20);
    const scalar_t x22 = x0 * (x1 * x11 + x1 * x12 + x1 * x13);
    const scalar_t x23 = adjugate[0] * x22;
    const scalar_t x24 = adjugate[4] * test_grad[1];
    const scalar_t x25 = adjugate[7] * test_grad[2];
    const scalar_t x26 = x0 * (x10 * x2 + x10 * x24 + x10 * x25);
    const scalar_t x27 = adjugate[3] * x22;
    const scalar_t x28 = adjugate[6] * x22;
    const scalar_t x29 = x0 * (x2 * x20 + x20 * x24 + x20 * x25);
    const scalar_t x30 = x0 * (adjugate[5] * test_grad[1] * x10 + adjugate[8] * test_grad[2] * x10 + x10 * x7);
    element_matrix[0] += x19 * (trial_grad[0] * (adjugate[0] * x14 + x6 + x9) + trial_grad[1] * (adjugate[3] * x14 + x15 + x16) +
                                trial_grad[2] * (adjugate[6] * x14 + x17 + x18));
    element_matrix[1] += x19 * (trial_grad[0] * (adjugate[0] * x5 + adjugate[1] * x21) +
                                trial_grad[1] * (adjugate[3] * x5 + adjugate[4] * x21) +
                                trial_grad[2] * (adjugate[6] * x5 + adjugate[7] * x21));
    element_matrix[2] += x19 * (trial_grad[0] * (adjugate[0] * x8 + adjugate[2] * x21) +
                                trial_grad[1] * (adjugate[3] * x8 + adjugate[5] * x21) +
                                trial_grad[2] * (adjugate[6] * x8 + adjugate[8] * x21));
    element_matrix[3] += x19 * (trial_grad[0] * (adjugate[1] * x26 + x23 + x9) + trial_grad[1] * (adjugate[4] * x26 + x16 + x27) +
                                trial_grad[2] * (adjugate[7] * x26 + x18 + x28));
    element_matrix[4] += x19 * (trial_grad[0] * (adjugate[1] * x8 + adjugate[2] * x29) +
                                trial_grad[1] * (adjugate[4] * x8 + adjugate[5] * x29) +
                                trial_grad[2] * (adjugate[7] * x8 + adjugate[8] * x29));
    element_matrix[5] += x19 * (trial_grad[0] * (adjugate[2] * x30 + x23 + x6) + trial_grad[1] * (adjugate[5] * x30 + x15 + x27) +
                                trial_grad[2] * (adjugate[8] * x30 + x17 + x28));
}

#endif  // CU_HEX8_LINEAR_ELASTICITY_MATRIX_INLINE_HPP