#include <cstdio>
#include <type_traits>
#include "../neohookean_ogden_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <cstdlib>
#include "../../../packed_thread_scratch.hpp"
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
struct neohookean_ogden_proteus_hex27_affine_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
        return data;
    }
};

template <typename scalar_t>
struct neohookean_ogden_proteus_hex27_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[12] = {scalar_t(0.80134602936993082), scalar_t(0.25844425285419081), scalar_t(-0.059790282224121687), scalar_t(0.22778407679095203), scalar_t(0.88441289000295209), scalar_t(-0.11219696679390417), scalar_t(-0.11219696679390401), scalar_t(0.88441289000295198), scalar_t(0.22778407679095214), scalar_t(-0.05979028222412186), scalar_t(0.25844425285419081), scalar_t(0.80134602936993082)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[12] = {scalar_t(-2.7222726231881049), scalar_t(3.4445452463762103), scalar_t(-0.72227262318810515), scalar_t(-1.6799620871697125), scalar_t(1.359924174339425), scalar_t(0.32003791283028749), scalar_t(-0.32003791283028749), scalar_t(-1.359924174339425), scalar_t(1.6799620871697125), scalar_t(0.72227262318810492), scalar_t(-3.4445452463762098), scalar_t(2.7222726231881049)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data = {
    "neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa",
    "PROTEUS_HEX27",
    3,
    64,
    27,
    16,
    4,
    19,
    21,
    0,
    0,
    10,
    0,
    1,
    0,
    6,
    5,
    70,
    11056,
    21024,
    4,
    12,
    10,
    24,
    4,
    2,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

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
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

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
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, value);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

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

        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};
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
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
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

            neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

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

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_packed_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
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

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_packed_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
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

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

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
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

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

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

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
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};
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

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

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

            neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

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

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_packed_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

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

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29], block_u_data[30], block_u_data[31], block_u_data[32], block_u_data[33], block_u_data[34], block_u_data[35], block_u_data[36], block_u_data[37], block_u_data[38], block_u_data[39], block_u_data[40], block_u_data[41], block_u_data[42], block_u_data[43], block_u_data[44], block_u_data[45], block_u_data[46], block_u_data[47], block_u_data[48], block_u_data[49], block_u_data[50], block_u_data[51], block_u_data[52], block_u_data[53], block_u_data[54], block_u_data[55], block_u_data[56], block_u_data[57], block_u_data[58], block_u_data[59], block_u_data[60], block_u_data[61], block_u_data[62], block_u_data[63], block_u_data[64], block_u_data[65], block_u_data[66], block_u_data[67], block_u_data[68], block_u_data[69], block_u_data[70], block_u_data[71], block_u_data[72], block_u_data[73], block_u_data[74], block_u_data[75], block_u_data[76], block_u_data[77], block_u_data[78], block_u_data[79], block_u_data[80]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

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

                    neohookean_ogden_d3_tensor_product_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data = {
    "neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa",
    "PROTEUS_HEX27",
    3,
    64,
    27,
    16,
    4,
    35,
    76,
    1,
    0,
    0,
    0,
    1,
    0,
    6,
    31,
    139,
    21988,
    31956,
    22,
    16,
    10,
    24,
    4,
    2,
    81,
    0,
    81,
    81,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

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
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

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
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

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

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_two_pass_affine_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_two_pass_affine_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

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
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

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

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

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

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
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

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
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

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
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

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
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

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data = {
    "neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa",
    "PROTEUS_HEX27",
    3,
    64,
    27,
    16,
    4,
    190,
    387,
    1,
    0,
    10,
    0,
    1,
    0,
    6,
    129,
    615,
    21988,
    31956,
    120,
    69,
    10,
    24,
    4,
    2,
    81,
    81,
    81,
    81,
    81,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

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
        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

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
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

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

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_two_pass_affine_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_two_pass_affine_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
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
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_bsr_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_bsr_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_float(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

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
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

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

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

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

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();
    static constexpr int N_QP_1D = 4;
    static constexpr int N_SHAPE_1D = 3;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
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
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }

    scalar_t *const out_components[DIM] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < DIM; ++d) {
            const scalar_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            scalar_t sum = scalar_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_bsr_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_bsr_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}


namespace sfem {
namespace codegen {

static SFEM_INLINE void neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_find_cols(
        const idx_t *const SFEM_RESTRICT targets,
        const idx_t *const SFEM_RESTRICT row,
        const int lenrow,
        idx_t *const SFEM_RESTRICT ks) {
    for (int d = 0; d < 27; ++d) {
        ks[d] = 0;
    }
    for (int k = 0; k < lenrow; ++k) {
        for (int d = 0; d < 27; ++d) {
            ks[d] += row[k] < targets[d];
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE int neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_bsr(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 27;
    count_t entries[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    bool valid_block_graph = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        const idx_t dof_i = ev[i];
        const count_t row_begin = rowptr[dof_i];
        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {
                if (valid_block_graph) {
                    std::fprintf(stderr, "neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_bsr missing block graph entry (%ld, %ld)\n", (long)ev[i], (long)ev[j]);
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

template <typename scalar_t, typename geometry_t, int FORMAT>
static int neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_assemble_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
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
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int VECTOR_SIZE = 1;
    static constexpr int NDOFS = DIM * N_SHAPE;
    (void)nnodes;
    const scalar_t *const u_components[DIM] = {ux, uy, uz};
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();

    int invalid_matrix_graph = 0;
#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[NDOFS * NDOFS];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        static constexpr int nelems = VECTOR_SIZE;
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
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
        scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
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

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

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
                neohookean_ogden_d3_tensor_product_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(1, 1, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
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
            invalid_matrix_graph |= (neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 0) {
            invalid_matrix_graph |= (neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 2) {
            invalid_matrix_graph |= (neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_dia(ev, element_matrix, nnodes, diag_offsets, ndiag, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 3) {
            invalid_matrix_graph |= (neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_coo(ev, element_matrix, coo_nnz, coo_rows, coo_cols, values) != SFEM_SUCCESS);
        } else if constexpr (FORMAT == 5) {
            neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_coo_triplets(ev, element_matrix, element, coo_triplet_rows, coo_triplet_cols, values);
        } else {
            invalid_matrix_graph |= (neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_scatter_patch(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
        }
    }

    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_proteus_hex27_proteus_hex27_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_proteus_hex27_proteus_hex27_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
