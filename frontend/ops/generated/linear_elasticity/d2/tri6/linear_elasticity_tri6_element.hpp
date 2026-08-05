#ifndef LINEAR_ELASTICITY_TRI6_ELEMENT_API_HPP
#define LINEAR_ELASTICITY_TRI6_ELEMENT_API_HPP

#include <stddef.h>
#include "../linear_elasticity_d2_simplex_local.hpp"
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
struct linear_elasticity_tri6_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[36] = {scalar_t(-0.084730493093977982), scalar_t(-0.04820837781551205), scalar_t(-0.04820837781551205), scalar_t(0.1928335112620482), scalar_t(0.79548022620090564), scalar_t(0.1928335112620482), scalar_t(-0.04820837781551205), scalar_t(-0.084730493093977968), scalar_t(-0.04820837781551205), scalar_t(0.19283351126204817), scalar_t(0.19283351126204817), scalar_t(0.79548022620090564), scalar_t(-0.04820837781551205), scalar_t(-0.04820837781551205), scalar_t(-0.084730493093977968), scalar_t(0.79548022620090564), scalar_t(0.19283351126204817), scalar_t(0.19283351126204817), scalar_t(0.5176323419876725), scalar_t(-0.074803807748196505), scalar_t(-0.074803807748196505), scalar_t(0.29921523099278602), scalar_t(0.03354481152314847), scalar_t(0.29921523099278602), scalar_t(-0.074803807748196505), scalar_t(0.5176323419876725), scalar_t(-0.074803807748196505), scalar_t(0.29921523099278602), scalar_t(0.29921523099278602), scalar_t(0.03354481152314847), scalar_t(-0.074803807748196505), scalar_t(-0.074803807748196505), scalar_t(0.5176323419876725), scalar_t(0.03354481152314847), scalar_t(0.29921523099278602), scalar_t(0.29921523099278602)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[36] = {scalar_t(0.56758792732771912), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(-1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(-0.56758792732771912), scalar_t(0), scalar_t(1.3513818909915787), scalar_t(1.7837939636638596), scalar_t(-1.7837939636638596), scalar_t(-0.78379396366385956), scalar_t(0.78379396366385956), scalar_t(0), scalar_t(5.5511151231257827e-17), scalar_t(0.43241207267228082), scalar_t(-0.43241207267228082), scalar_t(-2.2673902919218341), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(2.2673902919218341), scalar_t(0), scalar_t(-2.9010854378827511), scalar_t(0.36630485403908297), scalar_t(-0.36630485403908297), scalar_t(0.63369514596091703), scalar_t(-0.63369514596091703), scalar_t(0), scalar_t(0), scalar_t(3.2673902919218341), scalar_t(-3.2673902919218341)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[36] = {scalar_t(0.56758792732771912), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(-1.3513818909915787), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(0.78379396366385956), scalar_t(-0.43241207267228082), scalar_t(0.43241207267228082), scalar_t(0), scalar_t(-0.78379396366385956), scalar_t(0), scalar_t(-0.56758792732771912), scalar_t(-1.7837939636638596), scalar_t(1.7837939636638596), scalar_t(1.3513818909915787), scalar_t(-2.2673902919218341), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(2.9010854378827511), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(-0.63369514596091703), scalar_t(-3.2673902919218341), scalar_t(3.2673902919218341), scalar_t(0), scalar_t(0.63369514596091703), scalar_t(0), scalar_t(2.2673902919218341), scalar_t(-0.36630485403908297), scalar_t(0.36630485403908297), scalar_t(-2.9010854378827511)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};
        return data;
    }
};

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_tri6_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        linear_elasticity_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_tri6_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        linear_elasticity_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_tri6_energy_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        linear_elasticity_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_tri6_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        linear_elasticity_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_tri6_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        linear_elasticity_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_tri6_gradient_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        linear_elasticity_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int linear_elasticity_tri6_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
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
            linear_elasticity_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_h_streams, block_out_streams);
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
static SFEM_INLINE int linear_elasticity_tri6_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
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
            linear_elasticity_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_h_streams, block_out_streams);
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
static SFEM_INLINE int linear_elasticity_tri6_hessian_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
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
            linear_elasticity_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::linear_elasticity_tri6_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_h_streams, block_out_streams);
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
