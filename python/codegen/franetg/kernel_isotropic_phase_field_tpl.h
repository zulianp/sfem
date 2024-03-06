#include <math.h>
#include <stddef.h>
#include <cstddef>

#ifndef SFEM_RESTRICT
typedef double real_t;
#define SFEM_INLINE inline
#define SFEM_RESTRICT restrict
#endif

#ifndef SFEM_STATIC
#define SFEM_STATIC static
#endif

#define SFEM_DIM 3
#define SFEM_CUDA_WARP_SIZE 32
#define SFEM_CUDA_BLOCK_SIZE 128
#define SFEM_NUM_ELEM_X_BLOCK 32
#define SFEM_NUM_NODES_X_ELEM 4
#define SFEM_REF_MEASURE (1. / 6)

#ifndef __global__
#define __global__
#define __shared__
#define __device__
#endif

#define SFEM_DEVICE_MK SFEM_STATIC SFEM_INLINE __device__

template <typename T>
SFEM_DEVICE_MK T warp_reduce(T v, const int width) {
    for (int i = width / 2; i >= 1; i /= 2) {
        v += __shfl_xor_sync(FULL_MASK, v, i);
    }

    return v;
}

__global__ void sub_kernel_form1_init_fe() {
    // TODO
}

__global__ void sub_kernel_form1_init_fields() {
    // TODO
}

__global__ void sub_kernel_form1(const real_t mu,
                                 const real_t lambda,
                                 const real_t Gc,
                                 const real_t ls,
                                 const ptrdiff_t nelements,
                                 // QP reference
                                 // [QP X NNODES]
                                 const real_t *const SFEM_RESTRICT shape_fun,
                                 // [DIM X NNODES]
                                 real_t **const SFEM_RESTRICT shape_grad,
                                 // Geo
                                 // [DIM * DIM X NSUBE]
                                 const geom_t **const SFEM_RESTRICT jacobian_inverse,
                                 // [NSUBE]
                                 geom_t *const SFEM_RESTRICT jacobian_determinant,
                                 // Coeffs
                                 // [DIM X NSUBE X NNODES]
                                 real_t ***const SFEM_RESTRICT u,
                                 // [NSUBE X NNODES]
                                 real_t **const SFEM_RESTRICT c,
                                 // Out
                                 // [DIM X NSUBE X NNODES]
                                 real_t ***const SFEM_RESTRICT u_values,
                                 // [NSUBE X NNODES]
                                 real_t **const SFEM_RESTRICT c_values) {
    const int node = threadIdx.x % SFEM_NUM_NODES_X_ELEM;
    const int sub_e = threadIdx.x / SFEM_NUM_ELEM_X_BLOCK;

    // Registers
    real_t physical_grad[SFEM_DIM];
    real_t eval_c_grad[SFEM_DIM];
    real_t eval_u_grad[SFEM_DIM * SFEM_DIM];

    // Accumulators for assembly results
    real_t u_element_vector[SFEM_NUM_NODES_X_ELEM];
    real_t c_element_scalar;

    // One element per block
    for (ptrdiff_t block = blockIdx.x; block < n_elements; block += blockDim.x) {
        ptrdiff_t e = block * SFEM_NUM_ELEM_X_BLOCK + sub_e;

        // TODO Se if we have to use sub_e or e
        const ptrdiff_t element = sub_e;
        const geom_t dV = jacobian_determinant[element] * SFEM_REF_MEASURE;

        {  // !!! Only for Simplex
#pragma unroll(SFEM_DIM)
            for (int d1 = 0; d1 < SFEM_DIM; d1++) {
                physical_grad[d1] = 0;

                for (int d2 = 0; d2 < SFEM_DIM; d2++) {
                    // Assume linear element (constant over element)
                    physical_grad[d1] +=
                        shape_grad[d2][node] * jacobian_inverse[d2 * SFEM_DIM + d1][element];
                }
            }
        }

        {  // Reset values to zero
#pragma unroll(SFEM_NUM_NODES_X_ELEM)
            for (int i = 0; i < SFEM_NUM_NODES_X_ELEM; i++) {
                u_element_vector[i] = 0;
            }

            c_element_scalar = 0;
        }

        for (int qp = 0; qp < N_QP; qp++) {
            const real_t sf = shape_fun[qp][node];
            const real_t eval_c = warp_reduce(sf * c[element][node], SFEM_NUM_NODES_X_ELEM);

            // Phase-field gradient
#pragma unroll(SFEM_DIM)
            for (int d1 = 0; d1 < SFEM_DIM; d1++) {
                eval_c_grad[d1] =
                    warp_reduce(physical_grad[d1] * c[element][node], SFEM_NUM_NODES_X_ELEM);
            }

            // Displacement gradient
#pragma unroll(SFEM_DIM)
            for (int d1 = 0; d1 < SFEM_DIM; d1++) {
#pragma unroll(SFEM_DIM)
                for (int d2 = 0; d2 < SFEM_DIM; d2++) {
                    eval_u_grad[d1 * SFEM_DIM + d2] = warp_reduce(
                        physical_grad[d2] * u[d1][element][node], SFEM_NUM_NODES_X_ELEM);
                }
            }

            micro_kernel_form1_u(
                mu, lambda, dV, physical_grad, eval_u_grad, eval_c, 1, u_element_vector);

            micro_kernel_form1_c(mu,
                                 lambda,
                                 Gc,
                                 ls,
                                 dV,
                                 sf,
                                 physical_grad,
                                 eval_u_grad,
                                 eval_c,
                                 eval_c_grad,
                                 1,
                                 &c_element_scalar);
        }

        for (int d = 0; d < SFEM_DIM; d++) {
            u_values[d][element][node] += u_element_vector[d];
        }

        c_values[element][node] += c_element_scalar;
    }
}
