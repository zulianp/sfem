#include <cstdio>
#include <type_traits>
#include "../mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_local.hpp"
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
SFEM_INLINE const s_t *ageom_stream(
        const int,
        const g_t *const RSTR source,
        s_t *const RSTR,
        std::true_type) {
    return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
        const int nelems,
        const g_t *const RSTR source,
        s_t *const RSTR converted,
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
struct mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data {
    static const s_t *shape() {
        static const s_t data[4] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25)};
        return data;
    }
    static const s_t *grad_ref_x() {
        static const s_t data[4] = {s_t(-1), s_t(1), s_t(0), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_y() {
        static const s_t data[4] = {s_t(-1), s_t(0), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_z() {
        static const s_t data[4] = {s_t(-1), s_t(0), s_t(0), s_t(1)};
        return data;
    }
    static const s_t *q_weight() {
        static const s_t data[1] = {s_t(0.16666666666666666)};
        return data;
    }
};

template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data {
    static const s_t *shape() {
        static const s_t data[4] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25)};
        return data;
    }
    static const s_t *grad_ref_x() {
        static const s_t data[4] = {s_t(-1), s_t(1), s_t(0), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_y() {
        static const s_t data[4] = {s_t(-1), s_t(0), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_z() {
        static const s_t data[4] = {s_t(-1), s_t(0), s_t(0), s_t(1)};
        return data;
    }
    static const s_t *q_weight() {
        static const s_t data[1] = {s_t(0.16666666666666666)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data = {
    "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    45,
    49,
    0,
    0,
    17,
    0,
    0,
    0,
    6,
    22,
    111,
    0,
    0,
    21,
    29,
    10,
    12,
    1,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics(void) {
    return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_affine_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_affine_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_isoparametric_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_jacobian_adjugate0,
        const g_t *const RSTR g_jacobian_adjugate1,
        const g_t *const RSTR g_jacobian_adjugate2,
        const g_t *const RSTR g_jacobian_adjugate3,
        const g_t *const RSTR g_jacobian_adjugate4,
        const g_t *const RSTR g_jacobian_adjugate5,
        const g_t *const RSTR g_jacobian_adjugate6,
        const g_t *const RSTR g_jacobian_adjugate7,
        const g_t *const RSTR g_jacobian_adjugate8,
        const g_t *const RSTR g_jacobian_determinant0,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        idx_t ev[VS * NS];
        s_t bu_data[NS * NC][VS];
        s_t bu_base_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bvalue[VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const RSTR element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evb + lane];
            }
        }

        const s_t *const u_components[NC] = {ux, uy, uz};
        const s_t *const h_components[NC] = {hx, hy, hz};
        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bu_streams[stream] = bu_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    bu_base_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                    bh_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<g_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<g_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<g_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<g_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<g_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<g_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<g_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<g_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<g_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<g_t, s_t>());

        for (int step = 0; step < nsteps; ++step) {
            const s_t alpha = steps[step];
            for (int shape = 0; shape < NS; ++shape) {
                for (int d = 0; d < NC; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        bu_data[shape * NC + d][lane] = bu_base_data[shape * NC + d][lane] + alpha * bh_data[shape * NC + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bvalue[lane] = s_t(0);
            }

            mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bvalue);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    using s_t = double;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u_base = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux, uy, uz};
            const s_t *const h_components[NC] = {hx, hy, hz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_u_base_component = pk_u_base + d * max_nodes_per_pack;
                s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                const s_t *const RSTR h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_base_component[k] = u_component[node * u_stride];
                    pk_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pk_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bu_base_data[NS * NC][VS];
                s_t bh_data[NS * NC][VS];
                s_t bvalue[VS];

                const s_t *bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[3], bu_data[4], bu_data[5], bu_data[6], bu_data[7], bu_data[8], bu_data[9], bu_data[10], bu_data[11]};

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_base_data[shape * NC + d][lane] = pk_u_base[d * max_nodes_per_pack + packed_node];
                            bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                for (int step = 0; step < nsteps; ++step) {
                    const s_t alpha = steps[step];
                    for (int shape = 0; shape < NS; ++shape) {
                        for (int d = 0; d < NC; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                bu_data[shape * NC + d][lane] = bu_base_data[shape * NC + d][lane] + alpha * bh_data[shape * NC + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        bvalue[lane] = s_t(0);
                    }

                    mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bvalue);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_objective_steps_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    using s_t = float;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u_base = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux, uy, uz};
            const s_t *const h_components[NC] = {hx, hy, hz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_u_base_component = pk_u_base + d * max_nodes_per_pack;
                s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                const s_t *const RSTR h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_base_component[k] = u_component[node * u_stride];
                    pk_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pk_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bu_base_data[NS * NC][VS];
                s_t bh_data[NS * NC][VS];
                s_t bvalue[VS];

                const s_t *bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[3], bu_data[4], bu_data[5], bu_data[6], bu_data[7], bu_data[8], bu_data[9], bu_data[10], bu_data[11]};

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_base_data[shape * NC + d][lane] = pk_u_base[d * max_nodes_per_pack + packed_node];
                            bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                for (int step = 0; step < nsteps; ++step) {
                    const s_t alpha = steps[step];
                    for (int shape = 0; shape < NS; ++shape) {
                        for (int d = 0; d < NC; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                bu_data[shape * NC + d][lane] = bu_base_data[shape * NC + d][lane] + alpha * bh_data[shape * NC + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        bvalue[lane] = s_t(0);
                    }

                    mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bvalue);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];
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

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data = {
    "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    95,
    150,
    0,
    0,
    9,
    0,
    0,
    0,
    6,
    39,
    254,
    0,
    0,
    30,
    32,
    10,
    12,
    1,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics(void) {
    return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_jacobian_adjugate0,
        const g_t *const RSTR g_jacobian_adjugate1,
        const g_t *const RSTR g_jacobian_adjugate2,
        const g_t *const RSTR g_jacobian_adjugate3,
        const g_t *const RSTR g_jacobian_adjugate4,
        const g_t *const RSTR g_jacobian_adjugate5,
        const g_t *const RSTR g_jacobian_adjugate6,
        const g_t *const RSTR g_jacobian_adjugate7,
        const g_t *const RSTR g_jacobian_adjugate8,
        const g_t *const RSTR g_jacobian_determinant0,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        idx_t ev[VS * NS];
        s_t bu_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const RSTR element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evb + lane];
            }
        }
        const s_t *const u_components[NC] = {ux, uy, uz};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    bu_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bout_data[stream][lane] = s_t(0);
            }
        }

        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bu_streams[stream] = bu_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bout_streams[stream] = bout_data[stream];
        }
        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<g_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<g_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<g_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<g_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<g_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<g_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<g_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<g_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<g_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<g_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

        s_t *const out_components[NC] = {outx, outy, outz};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += bout_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
    using s_t = double;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux, uy, uz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
    using s_t = float;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux, uy, uz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_packed_two_pass_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        double *const RSTR ghost_buf,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
    using s_t = double;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const s_t *const u_components[NC] = {ux, uy, uz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
            s_t sum = s_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_packed_two_pass_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        float *const RSTR ghost_buf,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
    using s_t = float;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const s_t *const u_components[NC] = {ux, uy, uz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
            s_t sum = s_t(0);
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

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data = {
    "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    233,
    332,
    0,
    0,
    18,
    0,
    0,
    0,
    6,
    143,
    583,
    0,
    0,
    134,
    78,
    10,
    12,
    1,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics(void) {
    return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_jacobian_adjugate0,
        const g_t *const RSTR g_jacobian_adjugate1,
        const g_t *const RSTR g_jacobian_adjugate2,
        const g_t *const RSTR g_jacobian_adjugate3,
        const g_t *const RSTR g_jacobian_adjugate4,
        const g_t *const RSTR g_jacobian_adjugate5,
        const g_t *const RSTR g_jacobian_adjugate6,
        const g_t *const RSTR g_jacobian_adjugate7,
        const g_t *const RSTR g_jacobian_adjugate8,
        const g_t *const RSTR g_jacobian_determinant0,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        idx_t ev[VS * NS];
        s_t bu_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const RSTR element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evb + lane];
            }
        }
        const s_t *const u_components[NC] = {ux, uy, uz};
        const s_t *const h_components[NC] = {hx, hy, hz};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    bu_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                    bh_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bout_data[stream][lane] = s_t(0);
            }
        }

        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bu_streams[stream] = bu_data[stream];
        }
        const s_t *bh_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bh_streams[stream] = bh_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bout_streams[stream] = bout_data[stream];
        }
        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<g_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<g_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<g_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<g_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<g_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<g_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<g_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<g_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<g_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
                nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<g_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

        s_t *const out_components[NC] = {outx, outy, outz};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += bout_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
    using s_t = double;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux, uy, uz};
            const s_t *const h_components[NC] = {hx, hy, hz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                const s_t *const RSTR h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                    pk_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pk_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bh_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                const s_t *bh_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bh_streams[stream] = bh_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
    using s_t = float;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux, uy, uz};
            const s_t *const h_components[NC] = {hx, hy, hz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                const s_t *const RSTR h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                    pk_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pk_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bh_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                const s_t *bh_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bh_streams[stream] = bh_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_packed_two_pass_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        double *const RSTR ghost_buf,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
    using s_t = double;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const s_t *const u_components[NC] = {ux, uy, uz};
            const s_t *const h_components[NC] = {hx, hy, hz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                const s_t *const RSTR h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                    pk_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pk_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bh_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                const s_t *bh_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bh_streams[stream] = bh_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
            s_t sum = s_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_apply_packed_two_pass_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        float *const RSTR ghost_buf,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
    using s_t = float;
    static constexpr int NC = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_x();
    const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_y();
    const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::grad_ref_z();
    const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            (void)n_shared_nodes;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            const s_t *const u_components[NC] = {ux, uy, uz};
            const s_t *const h_components[NC] = {hx, hy, hz};
            s_t *const out_components[NC] = {outx, outy, outz};
            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
                s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
                const s_t *const RSTR u_component = u_components[d];
                const s_t *const RSTR h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pk_u_component[k] = u_component[node * u_stride];
                    pk_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pk_u_component[n_contiguous + k] = u_component[node * u_stride];
                    pk_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
                const int nelems = (int)MIN((ptrdiff_t)VS, e_end - evb);
                s_t bu_data[NS * NC][VS];
                s_t bh_data[NS * NC][VS];
                s_t bout_data[NS * NC][VS];
                const s_t *bu_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bu_streams[stream] = bu_data[stream];
                }
                const s_t *bh_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bh_streams[stream] = bh_data[stream];
                }
                s_t *bout_streams[NS * NC];
                for (int stream = 0; stream < NS * NC; ++stream) {
                    bout_streams[stream] = bout_data[stream];
                }

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evb + lane];
                            bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
                            bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
                            bout_data[shape * NC + d][lane] = s_t(0);
                        }
                    }
                }

                s_t badj0_data[VS];
                const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate0 + evb, badj0_data, std::is_same<geom_t, s_t>());
                s_t badj1_data[VS];
                const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate1 + evb, badj1_data, std::is_same<geom_t, s_t>());
                s_t badj2_data[VS];
                const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate2 + evb, badj2_data, std::is_same<geom_t, s_t>());
                s_t badj3_data[VS];
                const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate3 + evb, badj3_data, std::is_same<geom_t, s_t>());
                s_t badj4_data[VS];
                const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate4 + evb, badj4_data, std::is_same<geom_t, s_t>());
                s_t badj5_data[VS];
                const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate5 + evb, badj5_data, std::is_same<geom_t, s_t>());
                s_t badj6_data[VS];
                const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate6 + evb, badj6_data, std::is_same<geom_t, s_t>());
                s_t badj7_data[VS];
                const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate7 + evb, badj7_data, std::is_same<geom_t, s_t>());
                s_t badj8_data[VS];
                const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_adjugate8 + evb, badj8_data, std::is_same<geom_t, s_t>());
                s_t bdet0_data[VS];
                const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
                        nelems, g_jacobian_determinant0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

                mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(nelems, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

                for (int shape = 0; shape < NS; ++shape) {
                    const uint16_t *const RSTR element_shape = elements[shape];
                    for (int d = 0; d < NC; ++d) {
                        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
                s_t *const RSTR global_out = out_components[d];
                s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
                    pk_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
                    pk_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
            s_t sum = s_t(0);
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
