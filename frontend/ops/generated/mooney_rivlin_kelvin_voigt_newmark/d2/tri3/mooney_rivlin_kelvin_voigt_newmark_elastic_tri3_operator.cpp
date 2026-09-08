#include <cstdio>
#include <type_traits>
#include "../mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_local.hpp"
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

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *affine_geometry_stream(
        const int,
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *affine_geometry_stream(
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
struct mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_affine_reference_data {
    static const s_t *shape() {
        static const s_t data[3] = {s_t(0.33333333333333343), s_t(0.33333333333333331), s_t(0.33333333333333331)};
        return data;
    }
    static const s_t *grad_ref_x() {
        static const s_t data[3] = {s_t(-1), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_y() {
        static const s_t data[3] = {s_t(-1), s_t(0), s_t(1)};
        return data;
    }
    static const s_t *q_weight() {
        static const s_t data[1] = {s_t(0.5)};
        return data;
    }
};

template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data {
    static const s_t *shape() {
        static const s_t data[3] = {s_t(0.33333333333333343), s_t(0.33333333333333331), s_t(0.33333333333333331)};
        return data;
    }
    static const s_t *grad_ref_x() {
        static const s_t data[3] = {s_t(-1), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_y() {
        static const s_t data[3] = {s_t(-1), s_t(0), s_t(1)};
        return data;
    }
    static const s_t *q_weight() {
        static const s_t data[1] = {s_t(0.5)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data = {
    "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa",
    "TRI3",
    2,
    1,
    3,
    16,
    1,
    19,
    18,
    0,
    0,
    9,
    0,
    0,
    0,
    2,
    11,
    46,
    0,
    0,
    10,
    14,
    5,
    6,
    1,
    2,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics(void) {
    return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_affine_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_affine_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_isoparametric_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const s_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const s_t *const SFEM_RESTRICT hx,
        const s_t *const SFEM_RESTRICT hy,
        const int nsteps,
        const s_t *const SFEM_RESTRICT steps,
        s_t *const SFEM_RESTRICT value
) {
    static constexpr int NC = 2;
    static constexpr int NQ = 1;
    static constexpr int NS = 3;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        idx_t ev[VS * NS];
        s_t block_u_data[NS * NC][VS];
        s_t block_u_base_data[NS * NC][VS];
        s_t block_h_data[NS * NC][VS];
        s_t block_value[VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evbegin + lane];
            }
        }

        const s_t *const u_components[NC] = {ux, uy};
        const s_t *const h_components[NC] = {hx, hy};
        const s_t *block_u_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    block_u_base_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        s_t block_jacobian_adjugate0_data[VS];
        const s_t *const block_jacobian_adjugate0 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate1_data[VS];
        const s_t *const block_jacobian_adjugate1 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate2_data[VS];
        const s_t *const block_jacobian_adjugate2 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate3_data[VS];
        const s_t *const block_jacobian_adjugate3 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_determinant0_data[VS];
        const s_t *const block_jacobian_determinant0 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<g_t, s_t>());

        for (int step = 0; step < nsteps; ++step) {
            const s_t alpha = steps[step];
            for (int shape = 0; shape < NS; ++shape) {
                for (int d = 0; d < NC; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * NC + d][lane] = block_u_base_data[shape * NC + d][lane] + alpha * block_h_data[shape * NC + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = s_t(0);
            }

            mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_tri3_objective_block<s_t, NQ, NS, VS>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, affine_q_weight, lmbda, mu, block_u_streams, block_value);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
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
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
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
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data = {
    "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa",
    "TRI3",
    2,
    1,
    3,
    16,
    1,
    30,
    45,
    0,
    0,
    4,
    0,
    0,
    0,
    2,
    12,
    79,
    0,
    0,
    8,
    11,
    5,
    6,
    1,
    2,
    6,
    0,
    6,
    6,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics(void) {
    return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const s_t *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT outx,
        s_t *const SFEM_RESTRICT outy
) {
    static constexpr int NC = 2;
    static constexpr int NQ = 1;
    static constexpr int NS = 3;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        idx_t ev[VS * NS];
        s_t block_u_data[NS * NC][VS];
        s_t block_out_data[NS * NC][VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evbegin + lane];
            }
        }
        const s_t *const u_components[NC] = {ux, uy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    block_u_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = s_t(0);
            }
        }

        const s_t *block_u_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        s_t *block_out_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }
        s_t block_jacobian_adjugate0_data[VS];
        const s_t *const block_jacobian_adjugate0 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate1_data[VS];
        const s_t *const block_jacobian_adjugate1 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate2_data[VS];
        const s_t *const block_jacobian_adjugate2 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate3_data[VS];
        const s_t *const block_jacobian_adjugate3 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_determinant0_data[VS];
        const s_t *const block_jacobian_determinant0 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<g_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_tri3_gradient_block<s_t, NQ, NS, VS>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, affine_q_weight, lmbda, mu, block_u_streams, block_out_streams);

        s_t *const out_components[NC] = {outx, outy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += block_out_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data = {
    "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa",
    "TRI3",
    2,
    1,
    3,
    16,
    1,
    33,
    53,
    0,
    0,
    4,
    0,
    0,
    0,
    2,
    23,
    90,
    0,
    0,
    19,
    16,
    5,
    6,
    1,
    2,
    6,
    6,
    6,
    6,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics(void) {
    return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const s_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const s_t *const SFEM_RESTRICT hx,
        const s_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT outx,
        s_t *const SFEM_RESTRICT outy
) {
    static constexpr int NC = 2;
    static constexpr int NQ = 1;
    static constexpr int NS = 3;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        idx_t ev[VS * NS];
        s_t block_u_data[NS * NC][VS];
        s_t block_h_data[NS * NC][VS];
        s_t block_out_data[NS * NC][VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evbegin + lane];
            }
        }
        const s_t *const u_components[NC] = {ux, uy};
        const s_t *const h_components[NC] = {hx, hy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    block_u_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = s_t(0);
            }
        }

        const s_t *block_u_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const s_t *block_h_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        s_t *block_out_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }
        s_t block_jacobian_adjugate0_data[VS];
        const s_t *const block_jacobian_adjugate0 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate1_data[VS];
        const s_t *const block_jacobian_adjugate1 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate2_data[VS];
        const s_t *const block_jacobian_adjugate2 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_adjugate3_data[VS];
        const s_t *const block_jacobian_adjugate3 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<g_t, s_t>());
        s_t block_jacobian_determinant0_data[VS];
        const s_t *const block_jacobian_determinant0 = affine_geometry_stream<s_t, g_t, VS>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<g_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_tri3_apply_block<s_t, NQ, NS, VS>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, affine_q_weight, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

        s_t *const out_components[NC] = {outx, outy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += block_out_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
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
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
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
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}
