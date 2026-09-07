#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#if defined(__has_include)
#if __has_include("smesh_types.hpp")
#include "smesh_types.hpp"
#endif
#endif

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>

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
struct mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_affine_reference_data {
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
struct mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data {
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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[81],
        const double *const SFEM_RESTRICT previous[81],
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        double *const SFEM_RESTRICT output[81]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_residual_block<double, 64, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<double>::q_weight_1d(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[81],
        const float *const SFEM_RESTRICT previous[81],
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        float *const SFEM_RESTRICT output[81]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_residual_block<float, 64, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<float>::q_weight_1d(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_affine_mesh_soa_impl(
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
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u0,
        const scalar_t *const SFEM_RESTRICT u1,
        const scalar_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u0_old,
        const scalar_t *const SFEM_RESTRICT u1_old,
        const scalar_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u0_out,
        scalar_t *const SFEM_RESTRICT u1_out,
        scalar_t *const SFEM_RESTRICT u2_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        const scalar_t *const current_components[N_FIELDS] = {u0, u1, u2};
        const scalar_t *const previous_components[N_FIELDS] = {u0_old, u1_old, u2_old};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 81; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const jacobian_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[10][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current, block_previous, eta_b, eta_s, newmark_velocity_alpha, block_output);

        scalar_t *const output_components[N_FIELDS] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_affine_mesh_soa(
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
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_affine_mesh_soa_float(
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
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u0,
        const scalar_t *const SFEM_RESTRICT u1,
        const scalar_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u0_old,
        const scalar_t *const SFEM_RESTRICT u1_old,
        const scalar_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u0_out,
        scalar_t *const SFEM_RESTRICT u1_out,
        scalar_t *const SFEM_RESTRICT u2_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const scalar_t *const current_components[N_FIELDS] = {u0, u1, u2};
        const scalar_t *const previous_components[N_FIELDS] = {u0_old, u1_old, u2_old};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 81; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current, block_previous, eta_b, eta_s, newmark_velocity_alpha, block_output);

        scalar_t *const output_components[N_FIELDS] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[81],
        const double *const SFEM_RESTRICT previous[81],
        const double *const SFEM_RESTRICT direction[81],
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        double *const SFEM_RESTRICT output[81]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_jacobian_action_block<double, 64, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<double>::q_weight_1d(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[81],
        const float *const SFEM_RESTRICT previous[81],
        const float *const SFEM_RESTRICT direction[81],
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        float *const SFEM_RESTRICT output[81]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_jacobian_action_block<float, 64, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<float>::q_weight_1d(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_affine_mesh_soa_impl(
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
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u0,
        const scalar_t *const SFEM_RESTRICT u1,
        const scalar_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u0_old,
        const scalar_t *const SFEM_RESTRICT u1_old,
        const scalar_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u0_direction,
        const scalar_t *const SFEM_RESTRICT u1_direction,
        const scalar_t *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u0_out,
        scalar_t *const SFEM_RESTRICT u1_out,
        scalar_t *const SFEM_RESTRICT u2_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        const scalar_t *const current_components[N_FIELDS] = {u0, u1, u2};
        const scalar_t *const previous_components[N_FIELDS] = {u0_old, u1_old, u2_old};
        const scalar_t *const direction_components[N_FIELDS] = {u0_direction, u1_direction, u2_direction};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 81; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const jacobian_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[10][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current, block_previous, block_direction, eta_b, eta_s, newmark_velocity_alpha, block_output);

        scalar_t *const output_components[N_FIELDS] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_affine_mesh_soa(
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
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_affine_mesh_soa_float(
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
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u0,
        const scalar_t *const SFEM_RESTRICT u1,
        const scalar_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u0_old,
        const scalar_t *const SFEM_RESTRICT u1_old,
        const scalar_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u0_direction,
        const scalar_t *const SFEM_RESTRICT u1_direction,
        const scalar_t *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u0_out,
        scalar_t *const SFEM_RESTRICT u1_out,
        scalar_t *const SFEM_RESTRICT u2_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const scalar_t *const current_components[N_FIELDS] = {u0, u1, u2};
        const scalar_t *const previous_components[N_FIELDS] = {u0_old, u1_old, u2_old};
        const scalar_t *const direction_components[N_FIELDS] = {u0_direction, u1_direction, u2_direction};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 81; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_tensor_product_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current, block_previous, block_direction, eta_b, eta_s, newmark_velocity_alpha, block_output);

        scalar_t *const output_components[N_FIELDS] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, direction + 0, direction + 1, direction + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, direction + 0, direction + 1, direction + 2, 3, output + 0, output + 1, output + 2);
}
