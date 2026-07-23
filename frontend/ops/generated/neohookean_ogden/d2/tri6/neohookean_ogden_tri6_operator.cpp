#include <cstdio>
#include <type_traits>
#include "../neohookean_ogden_d2_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <cstdlib>
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

template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *affine_geometry_stream(
        const int,
        const jacobian_t *const SFEM_RESTRICT source,
        scalar_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *affine_geometry_stream(
        const int nelems,
        const jacobian_t *const SFEM_RESTRICT source,
        scalar_t *const SFEM_RESTRICT converted,
        std::false_type) {
    #pragma omp simd
    for (int lane = 0; lane < nelems; ++lane) {
        converted[lane] = scalar_t(source[lane]);
    }
    return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename scalar_t>
struct neohookean_ogden_tri6_affine_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[18] = {scalar_t(0.22222222222222235), scalar_t(-0.11111111111111112), scalar_t(-0.11111111111111112), scalar_t(0.44444444444444448), scalar_t(0.1111111111111111), scalar_t(0.44444444444444448), scalar_t(-0.11111111111111112), scalar_t(0.22222222222222215), scalar_t(-0.11111111111111112), scalar_t(0.44444444444444453), scalar_t(0.44444444444444442), scalar_t(0.11111111111111113), scalar_t(-0.11111111111111113), scalar_t(-0.11111111111111112), scalar_t(0.22222222222222215), scalar_t(0.11111111111111116), scalar_t(0.44444444444444442), scalar_t(0.44444444444444464)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[18] = {scalar_t(-1.666666666666667), scalar_t(-0.33333333333333337), scalar_t(0), scalar_t(2.0000000000000004), scalar_t(0.66666666666666663), scalar_t(-0.66666666666666663), scalar_t(0.33333333333333315), scalar_t(1.6666666666666665), scalar_t(0), scalar_t(-1.9999999999999996), scalar_t(0.66666666666666663), scalar_t(-0.66666666666666663), scalar_t(0.33333333333333304), scalar_t(-0.33333333333333337), scalar_t(0), scalar_t(4.4408920985006262e-16), scalar_t(2.6666666666666665), scalar_t(-2.6666666666666665)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[18] = {scalar_t(-1.666666666666667), scalar_t(0), scalar_t(-0.33333333333333337), scalar_t(-0.66666666666666663), scalar_t(0.66666666666666663), scalar_t(2), scalar_t(0.33333333333333315), scalar_t(0), scalar_t(-0.33333333333333337), scalar_t(-2.6666666666666665), scalar_t(2.6666666666666665), scalar_t(2.2204460492503131e-16), scalar_t(0.33333333333333304), scalar_t(0), scalar_t(1.6666666666666665), scalar_t(-0.66666666666666663), scalar_t(0.66666666666666663), scalar_t(-1.9999999999999996)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[3] = {scalar_t(0.16666666666666666), scalar_t(0.16666666666666666), scalar_t(0.16666666666666666)};
        return data;
    }
};

template <typename scalar_t>
struct neohookean_ogden_tri6_isoparametric_reference_data {
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

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data = {
    "neohookean_ogden_tri6_tri6_objective_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    9,
    9,
    0,
    0,
    5,
    0,
    1,
    0,
    2,
    4,
    43,
    0,
    0,
    3,
    7,
    5,
    72,
    6,
    2,
    12,
    0,
    1,
    1,
    1,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri6_tri6_objective_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_tri6_tri6_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_tri6_tri6_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_tri6_tri6_objective_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_tri6_tri6_objective_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_tri6_tri6_objective_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 3;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, mu, lmbda, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_objective_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" int neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 3;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy};
        const scalar_t *const h_components[DIM] = {hx, hy};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, mu, lmbda, block_u_streams, block_value);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int neohookean_ogden_tri6_tri6_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

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
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
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

        neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, mu, lmbda, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, value);
}

extern "C" int neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy};
        const scalar_t *const h_components[DIM] = {hx, hy};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

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
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
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

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            neohookean_ogden_d2_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, mu, lmbda, block_u_streams, block_value);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int neohookean_ogden_tri6_tri6_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data = {
    "neohookean_ogden_tri6_tri6_gradient_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    11,
    22,
    1,
    0,
    0,
    0,
    1,
    0,
    2,
    13,
    61,
    0,
    0,
    9,
    10,
    5,
    72,
    6,
    2,
    12,
    0,
    12,
    12,
    12,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri6_tri6_gradient_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_tri6_tri6_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_tri6_tri6_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_tri6_tri6_gradient_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_tri6_tri6_gradient_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 3;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        neohookean_ogden_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, mu, lmbda, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

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
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
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

        neohookean_ogden_d2_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, mu, lmbda, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data = {
    "neohookean_ogden_tri6_tri6_apply_soa",
    "TRI6",
    2,
    6,
    6,
    16,
    4,
    36,
    75,
    1,
    0,
    6,
    0,
    1,
    0,
    2,
    34,
    145,
    0,
    0,
    30,
    20,
    5,
    72,
    6,
    2,
    12,
    12,
    12,
    12,
    12,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri6_tri6_apply_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_tri6_tri6_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_tri6_tri6_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_tri6_tri6_apply_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_tri6_tri6_apply_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_tri6_tri6_apply_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_tri6_tri6_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 3;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tri6_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy};
        const scalar_t *const h_components[DIM] = {hx, hy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const scalar_t *block_h_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_bsr_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_affine_mesh_soa(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_bsr_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_dia_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_affine_mesh_soa(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_dia_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_patch_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_affine_mesh_soa(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_patch_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy};
        const scalar_t *const h_components[DIM] = {hx, hy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const scalar_t *block_h_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

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
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
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

        neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_bsr_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_bsr_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_dia_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_dia_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_patch_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_tri6_tri6_patch_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static SFEM_INLINE void neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_find_cols(
        const idx_t *const SFEM_RESTRICT targets,
        const idx_t *const SFEM_RESTRICT row,
        const int lenrow,
        idx_t *const SFEM_RESTRICT ks) {
#pragma unroll(6)
    for (int d = 0; d < 6; ++d) {
        ks[d] = 0;
    }
    for (int k = 0; k < lenrow; ++k) {
#pragma unroll(6)
        for (int d = 0; d < 6; ++d) {
            ks[d] += row[k] < targets[d];
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_bsr(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    count_t entries[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    bool valid_block_graph = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        const idx_t dof_i = ev[i];
        const count_t row_begin = rowptr[dof_i];
        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {
                if (valid_block_graph) {
                    std::fprintf(stderr, "neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_bsr missing block graph entry (%ld, %ld)\n", (long)ev[i], (long)ev[j]);
                }
                entries[i * N_SHAPE + j] = row_begin;
                valid_block_graph = false;
            } else {
                entries[i * N_SHAPE + j] = row_begin + ks[j];
            }
        }
    }
    if (!valid_block_graph) return SFEM_FAILURE;
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];
            for (int bi = 0; bi < DIM; ++bi) {
                const int row = bi * N_SHAPE + i;
                for (int bj = 0; bj < DIM; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_crs(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    count_t row_begin[N_SHAPE];
    int lenrow[N_SHAPE];
    int local_col[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    bool valid_graph = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        row_begin[i] = rowptr[ev[i]];
        lenrow[i] = (int)(rowptr[ev[i] + 1] - row_begin[i]);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin[i]];
        neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow[i], ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            if (ks[j] < 0 || ks[j] >= lenrow[i] || cols[ks[j]] != ev[j]) {
                if (valid_graph) {
                    std::fprintf(stderr, "neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_crs missing graph entry (%ld, %ld)\n", (long)ev[i], (long)ev[j]);
                }
                local_col[i * N_SHAPE + j] = 0;
                valid_graph = false;
            } else {
                local_col[i * N_SHAPE + j] = (int)ks[j];
            }
        }
    }
    if (!valid_graph) return SFEM_FAILURE;
    for (int i = 0; i < N_SHAPE; ++i) {
        const count_t rb = row_begin[i];
        const int lr = lenrow[i];
        for (int j = 0; j < N_SHAPE; ++j) {
            const int lc = local_col[i * N_SHAPE + j];
            for (int bi = 0; bi < DIM; ++bi) {
                const int row = bi * N_SHAPE + i;
                scalar_t *const row_values = &values[rb * DIM * DIM + bi * lr * DIM];
                for (int bj = 0; bj < DIM; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    row_values[lc * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_dia(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const ptrdiff_t nnodes,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    ptrdiff_t diagonals[N_SHAPE * N_SHAPE];
    bool valid_diagonal_offsets = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            const int offset = (int)(ev[j] - ev[i]);
            ptrdiff_t diagonal = 0;
            while (diagonal < ndiag && diag_offsets[diagonal] != offset) ++diagonal;
            if (diagonal == ndiag) {
                if (valid_diagonal_offsets) {
                    std::fprintf(stderr, "neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_dia missing diagonal offset %d\n", offset);
                }
                diagonals[i * N_SHAPE + j] = 0;
                valid_diagonal_offsets = false;
            } else {
                diagonals[i * N_SHAPE + j] = diagonal;
            }
        }
    }
    if (!valid_diagonal_offsets) return SFEM_FAILURE;
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            const ptrdiff_t diagonal = diagonals[i * N_SHAPE + j];
            scalar_t *const block = &values[(diagonal * nnodes + ev[i]) * DIM * DIM];
            for (int bi = 0; bi < DIM; ++bi) {
                const int row = bi * N_SHAPE + i;
                for (int bj = 0; bj < DIM; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_coo(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const ptrdiff_t nnz,
        const idx_t *const SFEM_RESTRICT rows,
        const idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    ptrdiff_t entries[N_SHAPE * N_SHAPE];
    bool valid_coo_entries = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            ptrdiff_t lo = 0;
            ptrdiff_t hi = nnz;
            while (lo < hi) {
                const ptrdiff_t mid = lo + (hi - lo) / 2;
                if (rows[mid] < ev[i] || (rows[mid] == ev[i] && cols[mid] < ev[j])) lo = mid + 1;
                else hi = mid;
            }
            if (lo == nnz || rows[lo] != ev[i] || cols[lo] != ev[j]) {
                if (valid_coo_entries) {
                    std::fprintf(stderr, "neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_coo missing graph entry (%ld, %ld)\n", (long)ev[i], (long)ev[j]);
                }
                entries[i * N_SHAPE + j] = 0;
                valid_coo_entries = false;
            } else {
                entries[i * N_SHAPE + j] = lo;
            }
        }
    }
    if (!valid_coo_entries) return SFEM_FAILURE;
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];
            for (int bi = 0; bi < DIM; ++bi) {
                const int row = bi * N_SHAPE + i;
                for (int bj = 0; bj < DIM; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE void neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_coo_triplets(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const ptrdiff_t element,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int NDOFS = DIM * N_SHAPE;
    const ptrdiff_t element_offset = element * NDOFS * NDOFS;
    for (int bi = 0; bi < DIM; ++bi) {
        for (int i = 0; i < N_SHAPE; ++i) {
            const int row = bi * N_SHAPE + i;
            const idx_t global_row = ev[i] * DIM + bi;
            for (int bj = 0; bj < DIM; ++bj) {
                for (int j = 0; j < N_SHAPE; ++j) {
                    const int col = bj * N_SHAPE + j;
                    const ptrdiff_t entry = element_offset + row * NDOFS + col;
                    rows[entry] = global_row;
                    cols[entry] = ev[j] * DIM + bj;
                    values[entry] = element_matrix[row * NDOFS + col];
                }
            }
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_patch(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    count_t entries[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    for (int i = 0; i < N_SHAPE; ++i) {
        const count_t row_begin = rowptr[ev[i]];
        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            entries[i * N_SHAPE + j] = row_begin + ks[j];
        }
    }
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            scalar_t *const block = &values[entries[i * N_SHAPE + j] * DIM * DIM];
            for (int bi = 0; bi < DIM; ++bi) {
                const int row = bi * N_SHAPE + i;
                for (int bj = 0; bj < DIM; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    block[bi * DIM + bj] += element_matrix[row * (DIM * N_SHAPE) + col];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

static SFEM_INLINE idx_t neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_global_node(
        const uint16_t packed_node,
        const ptrdiff_t pack,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx) {
    const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
    return packed_node < n_contiguous ? idx_t(owned_nodes_ptr[pack] + packed_node) : ghost_idx[ghost_ptr[pack] + packed_node - n_contiguous];
}

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_discover_packed_crs_entries(
        const idx_t *const SFEM_RESTRICT ev,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        count_t *const SFEM_RESTRICT entries) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int NDOFS = DIM * N_SHAPE;
    idx_t ks[N_SHAPE];
    bool valid_graph = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        const count_t row_begin = rowptr[ev[i]];
        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {
                if (valid_graph) {
                    std::fprintf(stderr, "neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_discover_packed_crs_entries missing graph entry (%ld, %ld)\n", (long)ev[i], (long)ev[j]);
                }
                for (int bi = 0; bi < DIM; ++bi) {
                    for (int bj = 0; bj < DIM; ++bj) {
                        const int row = bi * N_SHAPE + i;
                        const int col = bj * N_SHAPE + j;
                        entries[row * NDOFS + col] = row_begin;
                    }
                }
                valid_graph = false;
            } else {
                const int local_col = (int)ks[j];
                for (int bi = 0; bi < DIM; ++bi) {
                    const count_t row_value_offset = row_begin * DIM * DIM + bi * lenrow * DIM;
                    for (int bj = 0; bj < DIM; ++bj) {
                        const int row = bi * N_SHAPE + i;
                        const int col = bj * N_SHAPE + j;
                        entries[row * NDOFS + col] = row_value_offset + local_col * DIM + bj;
                    }
                }
            }
        }
    }
    return valid_graph ? SFEM_SUCCESS : SFEM_FAILURE;
}

template <typename scalar_t>
static SFEM_INLINE void neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_packed_crs_entries(
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT entries,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    static constexpr int NDOFS = DIM * N_SHAPE;
    for (int row = 0; row < NDOFS; ++row) {
        for (int col = 0; col < NDOFS; ++col) {
#pragma omp atomic update
            values[entries[row * NDOFS + col]] += element_matrix[row * NDOFS + col];
        }
    }
}

template <typename scalar_t, typename geometry_t, int FORMAT>
static int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        const ptrdiff_t coo_nnz,
        const idx_t *const SFEM_RESTRICT coo_rows,
        const idx_t *const SFEM_RESTRICT coo_cols,
        idx_t *const SFEM_RESTRICT coo_triplet_rows,
        idx_t *const SFEM_RESTRICT coo_triplet_cols) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 1;
    static constexpr int NDOFS = DIM * N_SHAPE;
    const scalar_t *const u_components[DIM] = {ux, uy};
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::q_weight();

    int invalid_matrix_graph = 0;
#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[NDOFS * NDOFS];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        static constexpr int nelems = VECTOR_SIZE;
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const scalar_t *block_h_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < DIM; ++d) {
                block_coordinate_data[shape * DIM + d][0] = scalar_t(points[d][node]);
                block_u_data[shape * DIM + d][0] = u_components[d][node * u_stride];
            }
        }


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
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
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

        for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }

        for (int trial_component = 0; trial_component < DIM; ++trial_component) {
            for (int trial_shape = 0; trial_shape < N_SHAPE; ++trial_shape) {
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_data[stream][0] = scalar_t(0);
                    block_out_data[stream][0] = scalar_t(0);
                }
                block_h_data[trial_shape * DIM + trial_component][0] = scalar_t(1);
                neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(1, 1, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);
                const int col = trial_component * N_SHAPE + trial_shape;
                for (int test_component = 0; test_component < DIM; ++test_component) {
                    for (int test_shape = 0; test_shape < N_SHAPE; ++test_shape) {
                        const int row = test_component * N_SHAPE + test_shape;
                        element_matrix[row * NDOFS + col] = block_out_data[test_shape * DIM + test_component][0];
                    }
                }
            }
        }

        if constexpr (FORMAT == 1) {
            invalid_matrix_graph |= (neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 0) {
            invalid_matrix_graph |= (neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 2) {
            invalid_matrix_graph |= (neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_dia(ev, element_matrix, nnodes, diag_offsets, ndiag, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 3) {
            invalid_matrix_graph |= (neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_coo(ev, element_matrix, coo_nnz, coo_rows, coo_cols, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 5) {
            neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_coo_triplets(ev, element_matrix, element, coo_triplet_rows, coo_triplet_cols, values);
        } else {
            invalid_matrix_graph |= (neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_patch(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
        }
    }

    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;
}

template <typename scalar_t, typename geometry_t>
static int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_discover_impl(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        count_t *const SFEM_RESTRICT packed_element_entries
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 6;
    (void)nnodes;
    (void)max_nodes_per_pack;
    (void)n_shared_nodes;
    int invalid_matrix_graph = 0;
#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
        const ptrdiff_t e_start = pack * n_elements_per_pack;
        const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
        for (ptrdiff_t element = e_start; element < e_end; ++element) {
            idx_t ev[N_SHAPE];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                ev[shape] = neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_global_node(elements[shape][element], pack, owned_nodes_ptr, ghost_ptr, ghost_idx);
            }
            count_t *const entries = &packed_element_entries[element * (DIM * N_SHAPE) * (DIM * N_SHAPE)];
            invalid_matrix_graph |= (neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_discover_packed_crs_entries<scalar_t>(ev, rowptr, colidx, entries) != SFEM_SUCCESS);
        }
    }
    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;
}

template <typename scalar_t, typename geometry_t>
static int neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_fill_impl(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t mu,
        const scalar_t lmbda,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT packed_element_entries,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 6;
    static constexpr int VECTOR_SIZE = 1;
    static constexpr int NDOFS = DIM * N_SHAPE;
    (void)nnodes;
    (void)n_shared_nodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::neohookean_ogden_tri6_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = (scalar_t *)std::malloc((size_t)DIM * (size_t)max_nodes_per_pack * sizeof(scalar_t));
        scalar_t *const SFEM_RESTRICT pack_u = (scalar_t *)std::malloc((size_t)DIM * (size_t)max_nodes_per_pack * sizeof(scalar_t));

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geometry_t *const coordinate_components[DIM] = {x, y};
            const scalar_t *const u_components[DIM] = {ux, uy};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geometry_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                scalar_t element_matrix[NDOFS * NDOFS];
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                static constexpr int nelems = VECTOR_SIZE;
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t packed_node = elements[shape][element];
                    for (int d = 0; d < DIM; ++d) {
                        block_coordinate_data[shape * DIM + d][0] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        block_u_data[shape * DIM + d][0] = pack_u[d * max_nodes_per_pack + packed_node];
                    }
                }


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
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
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

            for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
                element_matrix[entry] = scalar_t(0);
            }

            for (int trial_component = 0; trial_component < DIM; ++trial_component) {
                for (int trial_shape = 0; trial_shape < N_SHAPE; ++trial_shape) {
                    for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                        block_h_data[stream][0] = scalar_t(0);
                        block_out_data[stream][0] = scalar_t(0);
                    }
                    block_h_data[trial_shape * DIM + trial_component][0] = scalar_t(1);
                    neohookean_ogden_d2_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(1, 1, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, mu, lmbda, block_u_streams, block_h_streams, block_out_streams);
                    const int col = trial_component * N_SHAPE + trial_shape;
                    for (int test_component = 0; test_component < DIM; ++test_component) {
                        for (int test_shape = 0; test_shape < N_SHAPE; ++test_shape) {
                            const int row = test_component * N_SHAPE + test_shape;
                            element_matrix[row * NDOFS + col] = block_out_data[test_shape * DIM + test_component][0];
                        }
                    }
                }
            }

            const count_t *const entries = &packed_element_entries[element * NDOFS * NDOFS];
            neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_scatter_packed_crs_entries(element_matrix, entries, values);
            }
        }
        std::free(pack_coordinates);
        std::free(pack_u);
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tri6_tri6_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 0>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 0>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 2>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, nullptr, nullptr, values, diag_offsets, ndiag, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 2>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, nullptr, nullptr, values, diag_offsets, ndiag, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_coo_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t nnz,
        const idx_t *const SFEM_RESTRICT rows,
        const idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 3>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, nullptr, nullptr, values, nullptr, 0, nnz, rows, cols, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_coo_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t nnz,
        const idx_t *const SFEM_RESTRICT rows,
        const idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 3>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, nullptr, nullptr, values, nullptr, 0, nnz, rows, cols, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 5>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, nullptr, nullptr, values, nullptr, 0, 0, nullptr, nullptr, rows, cols);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 5>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, nullptr, nullptr, values, nullptr, 0, 0, nullptr, nullptr, rows, cols);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_patch_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 4>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_patch_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 4>(nelements, nnodes, elements, points, mu, lmbda, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_crs_packed_one_pass_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT packed_element_entries,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_fill_impl<double, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, mu, lmbda, u_stride, ux, uy, packed_element_entries, values);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_crs_packed_one_pass_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT packed_element_entries,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_fill_impl<float, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, mu, lmbda, u_stride, ux, uy, packed_element_entries, values);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_crs_packed_two_pass_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const double lmbda,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        count_t *const SFEM_RESTRICT packed_element_entries,
        double *const SFEM_RESTRICT values
) {
    const int graph_status = sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_discover_impl<double, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, rowptr, colidx, packed_element_entries);
    if (graph_status != SFEM_SUCCESS) return graph_status;
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_fill_impl<double, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, mu, lmbda, u_stride, ux, uy, packed_element_entries, values);
}

extern "C" int neohookean_ogden_tri6_tri6_hessian_crs_packed_two_pass_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const float lmbda,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        count_t *const SFEM_RESTRICT packed_element_entries,
        float *const SFEM_RESTRICT values
) {
    const int graph_status = sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_discover_impl<float, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, rowptr, colidx, packed_element_entries);
    if (graph_status != SFEM_SUCCESS) return graph_status;
    return sfem::codegen::neohookean_ogden_tri6_tri6_hessian_isoparametric_mesh_soa_packed_fill_impl<float, geom_t>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, mu, lmbda, u_stride, ux, uy, packed_element_entries, values);
}
