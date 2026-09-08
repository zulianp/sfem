#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_local.hpp"
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

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
        const int,
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
        const int nelems,
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT converted,
        std::false_type) {
    #pragma omp simd
    for (int lane = 0; lane < nelems; ++lane) {
        converted[lane] = s_t(source[lane]);
    }
    return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_affine_reference_data {
    static const s_t *shape_1d() {
        static const s_t data[4] = {s_t(0.78867513459481287), s_t(0.21132486540518708), s_t(0.21132486540518713), s_t(0.78867513459481287)};
        return data;
    }
    static const s_t *grad_1d() {
        static const s_t data[4] = {s_t(-1), s_t(1), s_t(-1), s_t(1)};
        return data;
    }
    static const s_t *q_weight_1d() {
        static const s_t data[2] = {s_t(0.5), s_t(0.5)};
        return data;
    }
};

template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data {
    static const s_t *shape_1d() {
        static const s_t data[4] = {s_t(0.78867513459481287), s_t(0.21132486540518708), s_t(0.21132486540518713), s_t(0.78867513459481287)};
        return data;
    }
    static const s_t *grad_1d() {
        static const s_t data[4] = {s_t(-1), s_t(1), s_t(-1), s_t(1)};
        return data;
    }
    static const s_t *q_weight_1d() {
        static const s_t data[2] = {s_t(0.5), s_t(0.5)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT previous[8],
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        double *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_residual_block<double, 4, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<double>::q_weight_1d(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT previous[8],
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        float *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_residual_block<float, 4, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<float>::q_weight_1d(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_affine_reference_data<s_t>::shape_1d();
    const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_affine_reference_data<s_t>::grad_1d();
    const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_affine_reference_data<s_t>::q_weight_1d();
    const idx_t *const SFEM_RESTRICT field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t bcurrent[NC * NS][VS];
        s_t bprevious[NC * NS][VS];
        s_t boutput[NC * NS][VS];
        const s_t *const current_components[NC] = {u0, u1};
        const s_t *const previous_components[NC] = {u0_old, u1_old};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    bcurrent[stream][lane] = current_components[field][node * current_stride];
                    bprevious[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 8; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                boutput[stream][lane] = s_t(0);
            }
        }

        const g_t *const affine_geometry_sources[5] = {g_jacobian_adjugate0 + evb, g_jacobian_adjugate1 + evb, g_jacobian_adjugate2 + evb, g_jacobian_adjugate3 + evb, g_jacobian_determinant0 + evb};
        s_t baffine_geometry_data[5][VS];
        const s_t *bageom_streams[5];
        for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
            bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *badjugate[4];
        for (int component = 0; component < 4; ++component) {
            badjugate[component] = bageom_streams[component];
        }

        mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(nelems, 0, bageom_streams[4], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bprevious, eta_b, eta_s, newmark_velocity_alpha, boutput);

        s_t *const output_components[NC] = {u0_out, u1_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    const idx_t *const SFEM_RESTRICT field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t bcoordinates[2 * NS][VS];
        s_t badjugate_data[4][NQ * VS];
        s_t bdeterminant[NQ * VS];
        s_t bcurrent[NC * NS][VS];
        s_t bprevious[NC * NS][VS];
        s_t boutput[NC * NS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const s_t *const current_components[NC] = {u0, u1};
        const s_t *const previous_components[NC] = {u0_old, u1_old};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    bcurrent[stream][lane] = current_components[field][node * current_stride];
                    bprevious[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 8; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                boutput[stream][lane] = s_t(0);
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

        const s_t *const badjugate[4] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};

        mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(nelems, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bprevious, eta_b, eta_s, newmark_velocity_alpha, boutput);

        s_t *const output_components[NC] = {u0_out, u1_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa(
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
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_float(
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
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, output + 0, output + 1);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, output + 0, output + 1);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT previous[8],
        const double *const SFEM_RESTRICT direction[8],
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        double *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_jacobian_action_block<double, 4, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<double>::q_weight_1d(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT previous[8],
        const float *const SFEM_RESTRICT direction[8],
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        float *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_jacobian_action_block<float, 4, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<float>::q_weight_1d(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT u0_direction,
        const s_t *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_affine_reference_data<s_t>::shape_1d();
    const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_affine_reference_data<s_t>::grad_1d();
    const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_affine_reference_data<s_t>::q_weight_1d();
    const idx_t *const SFEM_RESTRICT field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t bcurrent[NC * NS][VS];
        s_t bprevious[NC * NS][VS];
        s_t bdirection[NC * NS][VS];
        s_t boutput[NC * NS][VS];
        const s_t *const current_components[NC] = {u0, u1};
        const s_t *const previous_components[NC] = {u0_old, u1_old};
        const s_t *const direction_components[NC] = {u0_direction, u1_direction};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    bcurrent[stream][lane] = current_components[field][node * current_stride];
                    bprevious[stream][lane] = previous_components[field][node * previous_stride];
                    bdirection[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 8; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                boutput[stream][lane] = s_t(0);
            }
        }

        const g_t *const affine_geometry_sources[5] = {g_jacobian_adjugate0 + evb, g_jacobian_adjugate1 + evb, g_jacobian_adjugate2 + evb, g_jacobian_adjugate3 + evb, g_jacobian_determinant0 + evb};
        s_t baffine_geometry_data[5][VS];
        const s_t *bageom_streams[5];
        for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
            bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *badjugate[4];
        for (int component = 0; component < 4; ++component) {
            badjugate[component] = bageom_streams[component];
        }

        mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(nelems, 0, bageom_streams[4], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bprevious, bdirection, eta_b, eta_s, newmark_velocity_alpha, boutput);

        s_t *const output_components[NC] = {u0_out, u1_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT u0_direction,
        const s_t *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out
) {
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    const idx_t *const SFEM_RESTRICT field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t bcoordinates[2 * NS][VS];
        s_t badjugate_data[4][NQ * VS];
        s_t bdeterminant[NQ * VS];
        s_t bcurrent[NC * NS][VS];
        s_t bprevious[NC * NS][VS];
        s_t bdirection[NC * NS][VS];
        s_t boutput[NC * NS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const s_t *const current_components[NC] = {u0, u1};
        const s_t *const previous_components[NC] = {u0_old, u1_old};
        const s_t *const direction_components[NC] = {u0_direction, u1_direction};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    bcurrent[stream][lane] = current_components[field][node * current_stride];
                    bprevious[stream][lane] = previous_components[field][node * previous_stride];
                    bdirection[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 8; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                boutput[stream][lane] = s_t(0);
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

        const s_t *const badjugate[4] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};

        mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(nelems, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bprevious, bdirection, eta_b, eta_s, newmark_velocity_alpha, boutput);

        s_t *const output_components[NC] = {u0_out, u1_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = field_elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
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
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
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
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
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
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
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
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}
