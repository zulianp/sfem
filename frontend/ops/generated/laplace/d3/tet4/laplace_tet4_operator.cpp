#include <cstdio>
#include <type_traits>
#include "../laplace_d3_simplex_local.hpp"
#include "../laplace_d3_simplex_hessian.hpp"
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
struct laplace_tet4_affine_reference_data {
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
struct laplace_tet4_isoparametric_reference_data {
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

static const KernelDiagnostics laplace_tet4_objective_soa_diagnostics_data = {
    "laplace_tet4_objective_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    2,
    2,
    0,
    0,
    3,
    0,
    0,
    0,
    3,
    1,
    7,
    0,
    0,
    0,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_objective_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data;
}

extern "C" double laplace_tet4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tet4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_objective_soa",
            &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_objective_soa_float",
            &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_objective_affine_mesh_soa",
            &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_objective_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_objective_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int laplace_tet4_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_geom_metric0,
        const g_t *const SFEM_RESTRICT g_geom_metric1,
        const g_t *const SFEM_RESTRICT g_geom_metric2,
        const g_t *const SFEM_RESTRICT g_geom_metric3,
        const g_t *const SFEM_RESTRICT g_geom_metric4,
        const g_t *const SFEM_RESTRICT g_geom_metric5,
        const s_t kappa,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const s_t *const SFEM_RESTRICT hx,
        const int nsteps,
        const s_t *const SFEM_RESTRICT steps,
        s_t *const SFEM_RESTRICT value
) {
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const s_t x0 = ux[ev0 * u_stride];
        const s_t x1 = ux[ev1 * u_stride];
        const s_t x2 = ux[ev2 * u_stride];
        const s_t x3 = ux[ev3 * u_stride];
        const s_t h0 = hx[ev0 * h_stride];
        const s_t h1 = hx[ev1 * h_stride];
        const s_t h2 = hx[ev2 * h_stride];
        const s_t h3 = hx[ev3 * h_stride];
        const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
        const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
        const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
        const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
        const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
        const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
        for (int step = 0; step < nsteps; ++step) {
            const s_t alpha = steps[step];
            const s_t u0 = x0 + alpha * h0;
            const s_t u1 = x1 + alpha * h1;
            const s_t u2 = x2 + alpha * h2;
            const s_t u3 = x3 + alpha * h3;
            const s_t t0 = -u0 + u1;
            const s_t t1 = -u0 + u2;
            const s_t t2 = -u0 + u3;
            value[(ptrdiff_t)step * nelements + element] = ((s_t(1) / s_t(2)))*t0*(fff0*t0 + fff1*t1 + fff2*t2) + ((s_t(1) / s_t(2)))*t1*(fff1*t0 + fff3*t1 + fff4*t2) + ((s_t(1) / s_t(2)))*t2*(fff2*t0 + fff4*t1 + fff5*t2);
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_tet4_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_tet4_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_tet4_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_tet4_objective_steps_packed_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using s_t = double;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux};
            const s_t *const h_components[NC] = {hx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT u_component = u_components[d];
                const s_t *const SFEM_RESTRICT h_component = h_components[d];
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t x0 = pack_u_base[ev0];
                const s_t x1 = pack_u_base[ev1];
                const s_t x2 = pack_u_base[ev2];
                const s_t x3 = pack_u_base[ev3];
                const s_t h0 = pack_h[ev0];
                const s_t h1 = pack_h[ev1];
                const s_t h2 = pack_h[ev2];
                const s_t h3 = pack_h[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                for (int step = 0; step < nsteps; ++step) {
                    const s_t alpha = steps[step];
                    const s_t u0 = x0 + alpha * h0;
                    const s_t u1 = x1 + alpha * h1;
                    const s_t u2 = x2 + alpha * h2;
                    const s_t u3 = x3 + alpha * h3;
                    const s_t t0 = -u0 + u1;
                    const s_t t1 = -u0 + u2;
                    const s_t t2 = -u0 + u3;
                    value[(ptrdiff_t)step * nelements + element] = ((s_t(1) / s_t(2)))*t0*(fff0*t0 + fff1*t1 + fff2*t2) + ((s_t(1) / s_t(2)))*t1*(fff1*t0 + fff3*t1 + fff4*t2) + ((s_t(1) / s_t(2)))*t2*(fff2*t0 + fff4*t1 + fff5*t2);
                }
            }

        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_objective_steps_packed_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using s_t = float;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const s_t *const u_components[NC] = {ux};
            const s_t *const h_components[NC] = {hx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT u_component = u_components[d];
                const s_t *const SFEM_RESTRICT h_component = h_components[d];
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t x0 = pack_u_base[ev0];
                const s_t x1 = pack_u_base[ev1];
                const s_t x2 = pack_u_base[ev2];
                const s_t x3 = pack_u_base[ev3];
                const s_t h0 = pack_h[ev0];
                const s_t h1 = pack_h[ev1];
                const s_t h2 = pack_h[ev2];
                const s_t h3 = pack_h[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                for (int step = 0; step < nsteps; ++step) {
                    const s_t alpha = steps[step];
                    const s_t u0 = x0 + alpha * h0;
                    const s_t u1 = x1 + alpha * h1;
                    const s_t u2 = x2 + alpha * h2;
                    const s_t u3 = x3 + alpha * h3;
                    const s_t t0 = -u0 + u1;
                    const s_t t1 = -u0 + u2;
                    const s_t t2 = -u0 + u3;
                    value[(ptrdiff_t)step * nelements + element] = ((s_t(1) / s_t(2)))*t0*(fff0*t0 + fff1*t1 + fff2*t2) + ((s_t(1) / s_t(2)))*t1*(fff1*t0 + fff3*t1 + fff4*t2) + ((s_t(1) / s_t(2)))*t2*(fff2*t0 + fff4*t1 + fff5*t2);
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

static const KernelDiagnostics laplace_tet4_gradient_soa_diagnostics_data = {
    "laplace_tet4_gradient_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    0,
    3,
    0,
    0,
    0,
    0,
    0,
    0,
    3,
    3,
    3,
    0,
    0,
    0,
    2,
    10,
    12,
    1,
    2,
    12,
    0,
    4,
    4,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_gradient_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data;
}

extern "C" double laplace_tet4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_gradient_soa",
            &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_gradient_soa_float",
            &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_gradient_affine_mesh_soa",
            &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_gradient_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_gradient_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int laplace_tet4_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_geom_metric0,
        const g_t *const SFEM_RESTRICT g_geom_metric1,
        const g_t *const SFEM_RESTRICT g_geom_metric2,
        const g_t *const SFEM_RESTRICT g_geom_metric3,
        const g_t *const SFEM_RESTRICT g_geom_metric4,
        const g_t *const SFEM_RESTRICT g_geom_metric5,
        const s_t kappa,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT outx
) {
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const s_t u0 = ux[ev0 * u_stride];
        const s_t u1 = ux[ev1 * u_stride];
        const s_t u2 = ux[ev2 * u_stride];
        const s_t u3 = ux[ev3 * u_stride];
        const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
        const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
        const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
        const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
        const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
        const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
        const s_t t0 = -u0 + u1;
        const s_t t1 = -u0 + u2;
        const s_t t2 = -u0 + u3;
        const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
        const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
        const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
        const s_t e0 = -t3 - t4 - t5;
        #pragma omp atomic update
        outx[ev0 * out_stride] += e0;
        const s_t e1 = t3;
        #pragma omp atomic update
        outx[ev1 * out_stride] += e1;
        const s_t e2 = t4;
        #pragma omp atomic update
        outx[ev2 * out_stride] += e2;
        const s_t e3 = t5;
        #pragma omp atomic update
        outx[ev3 * out_stride] += e3;
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tet4_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_tet4_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tet4_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_tet4_gradient_packed_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using s_t = double;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const u_components[NC] = {ux};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_u[ev0];
                const s_t u1 = pack_u[ev1];
                const s_t u2 = pack_u[ev2];
                const s_t u3 = pack_u[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_gradient_packed_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using s_t = float;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const u_components[NC] = {ux};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_u[ev0];
                const s_t u1 = pack_u[ev1];
                const s_t u2 = pack_u[ev2];
                const s_t u3 = pack_u[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_gradient_packed_two_pass_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using s_t = double;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const u_components[NC] = {ux};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_u[ev0];
                const s_t u1 = pack_u[ev1];
                const s_t u2 = pack_u[ev2];
                const s_t u3 = pack_u[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            s_t sum = s_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_gradient_packed_two_pass_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using s_t = float;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const u_components[NC] = {ux};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_u[ev0];
                const s_t u1 = pack_u[ev1];
                const s_t u2 = pack_u[ev2];
                const s_t u3 = pack_u[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
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

static const KernelDiagnostics laplace_tet4_apply_soa_diagnostics_data = {
    "laplace_tet4_apply_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    0,
    3,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    3,
    3,
    0,
    0,
    0,
    2,
    10,
    12,
    1,
    2,
    0,
    12,
    4,
    4,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_apply_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data;
}

extern "C" double laplace_tet4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tet4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_apply_soa",
            &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_apply_soa_float",
            &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_apply_affine_mesh_soa",
            &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_apply_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_apply_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int laplace_tet4_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_geom_metric0,
        const g_t *const SFEM_RESTRICT g_geom_metric1,
        const g_t *const SFEM_RESTRICT g_geom_metric2,
        const g_t *const SFEM_RESTRICT g_geom_metric3,
        const g_t *const SFEM_RESTRICT g_geom_metric4,
        const g_t *const SFEM_RESTRICT g_geom_metric5,
        const s_t kappa,
        const ptrdiff_t h_stride,
        const s_t *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT outx
) {
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const s_t u0 = hx[ev0 * h_stride];
        const s_t u1 = hx[ev1 * h_stride];
        const s_t u2 = hx[ev2 * h_stride];
        const s_t u3 = hx[ev3 * h_stride];
        const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
        const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
        const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
        const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
        const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
        const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
        const s_t t0 = -u0 + u1;
        const s_t t1 = -u0 + u2;
        const s_t t2 = -u0 + u3;
        const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
        const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
        const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
        const s_t e0 = -t3 - t4 - t5;
        #pragma omp atomic update
        outx[ev0 * out_stride] += e0;
        const s_t e1 = t3;
        #pragma omp atomic update
        outx[ev1 * out_stride] += e1;
        const s_t e2 = t4;
        #pragma omp atomic update
        outx[ev2 * out_stride] += e2;
        const s_t e3 = t5;
        #pragma omp atomic update
        outx[ev3 * out_stride] += e3;
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tet4_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_tet4_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tet4_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, h_stride, hx, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_tet4_apply_packed_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using s_t = double;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const h_components[NC] = {hx};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_h[ev0];
                const s_t u1 = pack_h[ev1];
                const s_t u2 = pack_h[ev2];
                const s_t u3 = pack_h[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_apply_packed_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using s_t = float;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const h_components[NC] = {hx};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_h[ev0];
                const s_t u1 = pack_h[ev1];
                const s_t u2 = pack_h[ev2];
                const s_t u3 = pack_h[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_apply_packed_two_pass_affine_mesh_soa(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using s_t = double;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const h_components[NC] = {hx};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_h[ev0];
                const s_t u1 = pack_h[ev1];
                const s_t u2 = pack_h[ev2];
                const s_t u3 = pack_h[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
            s_t sum = s_t(0);
            for (ptrdiff_t j = begin; j < end; ++j) {
                sum += ghost_component[ghost_reduce_idx[j]];
            }
            out_components[d][dest * out_stride] += sum;
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_apply_packed_two_pass_affine_mesh_soa_float(
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
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const geom_t *const SFEM_RESTRICT g_geom_metric3,
        const geom_t *const SFEM_RESTRICT g_geom_metric4,
        const geom_t *const SFEM_RESTRICT g_geom_metric5,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using s_t = float;
    static constexpr int NC = 1;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;

    const s_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel
    {
        s_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
        s_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

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
            const s_t *const h_components[NC] = {hx};
            s_t *const out_components[NC] = {outx};
            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const s_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const s_t u0 = pack_h[ev0];
                const s_t u1 = pack_h[ev1];
                const s_t u2 = pack_h[ev2];
                const s_t u3 = pack_h[ev3];
                const s_t fff0 = kappa * s_t(g_geom_metric0[element]);
                const s_t fff1 = kappa * s_t(g_geom_metric1[element]);
                const s_t fff2 = kappa * s_t(g_geom_metric2[element]);
                const s_t fff3 = kappa * s_t(g_geom_metric3[element]);
                const s_t fff4 = kappa * s_t(g_geom_metric4[element]);
                const s_t fff5 = kappa * s_t(g_geom_metric5[element]);
                const s_t t0 = -u0 + u1;
                const s_t t1 = -u0 + u2;
                const s_t t2 = -u0 + u3;
                const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const s_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const s_t e1 = t3;
                pack_out[ev1] += e1;
                const s_t e2 = t4;
                pack_out[ev2] += e2;
                const s_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < NC; ++d) {
                s_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                s_t *const SFEM_RESTRICT global_out = out_components[d];
                s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = s_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    ghost_component[ghost_off + k] = pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = s_t(0);
                }
            }
        }
    }

    s_t *const out_components[NC] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < NC; ++d) {
            const s_t *const SFEM_RESTRICT ghost_component = ghost_buf + d * n_ghost_entries;
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

static SFEM_INLINE void laplace_tet4_hessian_isoparametric_mesh_soa_find_cols(
        const idx_t *const SFEM_RESTRICT targets,
        const idx_t *const SFEM_RESTRICT row,
        const int lenrow,
        idx_t *const SFEM_RESTRICT ks) {
#pragma unroll(4)
    for (int d = 0; d < 4; ++d) {
        ks[d] = 0;
    }
    for (int k = 0; k < lenrow; ++k) {
#pragma unroll(4)
        for (int d = 0; d < 4; ++d) {
            ks[d] += row[k] < targets[d];
        }
    }
}

template <typename s_t>
static SFEM_INLINE void laplace_tet4_hessian_isoparametric_mesh_soa_scatter_bsr(
        const idx_t *const SFEM_RESTRICT ev,
        const s_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        s_t *const SFEM_RESTRICT values) {
    static constexpr int NC = 1;
    static constexpr int NS = 4;
    count_t entries[NS * NS];
    idx_t ks[NS];
    for (int i = 0; i < NS; ++i) {
        const idx_t dof_i = ev[i];
        const count_t row_begin = rowptr[dof_i];
        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        laplace_tet4_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < NS; ++j) {
            entries[i * NS + j] = row_begin + ks[j];
        }
    }
    for (int i = 0; i < NS; ++i) {
        for (int j = 0; j < NS; ++j) {
            s_t *const block = &values[entries[i * NS + j] * NC * NC];
            for (int bi = 0; bi < NC; ++bi) {
                const int row = bi * NS + i;
                for (int bj = 0; bj < NC; ++bj) {
                    const int col = bj * NS + j;
#pragma omp atomic update
                    block[bi * NC + bj] += element_matrix[row * (NC * NS) + col];
                }
            }
        }
    }
}

template <typename s_t>
static SFEM_INLINE void laplace_tet4_hessian_isoparametric_mesh_soa_scatter_crs(
        const idx_t *const SFEM_RESTRICT ev,
        const s_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        s_t *const SFEM_RESTRICT values) {
    static constexpr int NC = 1;
    static constexpr int NS = 4;
    count_t row_begin[NS];
    int lenrow[NS];
    int local_col[NS * NS];
    idx_t ks[NS];
    for (int i = 0; i < NS; ++i) {
        row_begin[i] = rowptr[ev[i]];
        lenrow[i] = (int)(rowptr[ev[i] + 1] - row_begin[i]);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin[i]];
        laplace_tet4_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow[i], ks);
        for (int j = 0; j < NS; ++j) {
            local_col[i * NS + j] = (int)ks[j];
        }
    }
    for (int i = 0; i < NS; ++i) {
        const count_t rb = row_begin[i];
        const int lr = lenrow[i];
        for (int j = 0; j < NS; ++j) {
            const int lc = local_col[i * NS + j];
            for (int bi = 0; bi < NC; ++bi) {
                const int row = bi * NS + i;
                s_t *const row_values = &values[rb * NC * NC + bi * lr * NC];
                for (int bj = 0; bj < NC; ++bj) {
                    const int col = bj * NS + j;
#pragma omp atomic update
                    row_values[lc * NC + bj] += element_matrix[row * (NC * NS) + col];
                }
            }
        }
    }
}

template <typename s_t, typename g_t, int FORMAT>
static int laplace_tet4_hessian_isoparametric_mesh_soa_assemble_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const *const SFEM_RESTRICT points,
        const s_t kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        s_t *const SFEM_RESTRICT values,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        const ptrdiff_t coo_nnz,
        const idx_t *const SFEM_RESTRICT coo_rows,
        const idx_t *const SFEM_RESTRICT coo_cols,
        idx_t *const SFEM_RESTRICT coo_triplet_rows,
        idx_t *const SFEM_RESTRICT coo_triplet_cols) {
    static constexpr int NC = 1;
    static constexpr int ND = 3;
    static constexpr int NQ = 1;
    static constexpr int NS = 4;
    static constexpr int VS = 1;
    static constexpr int NDOFS = NC * NS;
    (void)nnodes;
    const g_t *const SFEM_RESTRICT x = points[0];
    const g_t *const SFEM_RESTRICT y = points[1];
    const g_t *const SFEM_RESTRICT z = points[2];
    const s_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
    const s_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
    const s_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
    const s_t *const isoparametric_q_weight = sfem::codegen::laplace_tet4_isoparametric_reference_data<s_t>::q_weight();

    int unsupported_matrix_format = 0;
#pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        idx_t ev[NS];
        s_t element_matrix[NDOFS * NDOFS];
        s_t block_h_data[NS * NC][VS];
        s_t block_out_data[NS * NC][VS];
        s_t block_coordinate_data[NS * ND][VS];
        static constexpr int nelems = VS;
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_adjugate4[NQ * VS];
        s_t block_jacobian_adjugate5[NQ * VS];
        s_t block_jacobian_adjugate6[NQ * VS];
        s_t block_jacobian_adjugate7[NQ * VS];
        s_t block_jacobian_adjugate8[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        const s_t *block_h_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        s_t *block_out_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < ND; ++d) {
                block_coordinate_data[shape * ND + d][0] = s_t(points[d][node]);
            }
        }


        for (int q = 0; q < NQ; ++q) {
            s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J02_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            s_t J12_values[VS];
            s_t J20_values[VS];
            s_t J21_values[VS];
            s_t J22_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
                const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
                const s_t g2 = isoparametric_grad_ref_z[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J02 = J02_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                const s_t J12 = J12_values[lane];
                const s_t J20 = J20_values[lane];
                const s_t J21 = J21_values[lane];
                const s_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VS + lane);
            }
        }

        laplace_d3_simplex_direct_hessian_reference_element_matrix<s_t, NQ, NS, VS>(block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, kappa, element_matrix);

        if constexpr (FORMAT == 1) {
            laplace_tet4_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
        } else if constexpr (FORMAT == 0) {
            laplace_tet4_hessian_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
        } else {
            unsupported_matrix_format |= 1;
        }
    }

    return unsupported_matrix_format ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tet4_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tet4_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tet4_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tet4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tet4_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tet4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tet4_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
