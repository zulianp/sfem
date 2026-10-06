#ifndef LINEAR_ELASTICITY_PROTEUS_HEX8_ELEMENT_API_HPP
#define LINEAR_ELASTICITY_PROTEUS_HEX8_ELEMENT_API_HPP

#include <stddef.h>
#include "../linear_elasticity_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif

#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace sfem {
namespace codegen {


template <typename scalar_t>
struct linear_elasticity_proteus_hex8_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[4] = {scalar_t(0.78867513459481287), scalar_t(0.21132486540518708), scalar_t(0.21132486540518713), scalar_t(0.78867513459481287)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[4] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[2] = {scalar_t(0.5), scalar_t(0.5)};
        return data;
    }
};

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        linear_elasticity_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_energy_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        linear_elasticity_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_gradient_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        linear_elasticity_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_proteus_hex8_hessian_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_QP = 8;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), block_coordinate_data, 2, coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);
        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            linear_elasticity_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d(), sfem::codegen::linear_elasticity_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d(), lmbda, mu, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}


} // namespace codegen
} // namespace sfem

#endif
