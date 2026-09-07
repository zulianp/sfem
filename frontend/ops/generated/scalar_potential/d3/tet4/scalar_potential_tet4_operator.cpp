#include <cstdio>
#include <type_traits>
#include "../scalar_potential_d3_simplex_local.hpp"
#include "../scalar_potential_d3_simplex_hessian.hpp"
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
struct scalar_potential_tet4_affine_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[4] = {scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[1] = {scalar_t(0.16666666666666666)};
        return data;
    }
};

template <typename scalar_t>
struct scalar_potential_tet4_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[4] = {scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[4] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[4] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[4] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[1] = {scalar_t(0.16666666666666666)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics scalar_potential_tet4_objective_soa_diagnostics_data = {
    "scalar_potential_tet4_objective_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *scalar_potential_tet4_objective_soa_diagnostics(void) {
    return &sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data;
}

extern "C" double scalar_potential_tet4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void scalar_potential_tet4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "scalar_potential_tet4_objective_soa",
            &sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "scalar_potential_tet4_objective_soa_float",
            &sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void scalar_potential_tet4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "scalar_potential_tet4_objective_affine_mesh_soa",
            &sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "scalar_potential_tet4_objective_affine_mesh_soa_float",
            &sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void scalar_potential_tet4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "scalar_potential_tet4_objective_isoparametric_mesh_soa",
            &sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "scalar_potential_tet4_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::scalar_potential_tet4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int scalar_potential_tet4_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric3,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric4,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric5,
        const scalar_t kappa,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const scalar_t x0 = ux[ev0 * u_stride];
        const scalar_t x1 = ux[ev1 * u_stride];
        const scalar_t x2 = ux[ev2 * u_stride];
        const scalar_t x3 = ux[ev3 * u_stride];
        const scalar_t h0 = hx[ev0 * h_stride];
        const scalar_t h1 = hx[ev1 * h_stride];
        const scalar_t h2 = hx[ev2 * h_stride];
        const scalar_t h3 = hx[ev3 * h_stride];
        const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
        const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
        const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
        const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
        const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
        const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            const scalar_t u0 = x0 + alpha * h0;
            const scalar_t u1 = x1 + alpha * h1;
            const scalar_t u2 = x2 + alpha * h2;
            const scalar_t u3 = x3 + alpha * h3;
            const scalar_t t0 = -u0 + u1;
            const scalar_t t1 = -u0 + u2;
            const scalar_t t2 = -u0 + u3;
            value[(ptrdiff_t)step * nelements + element] = ((scalar_t(1) / scalar_t(2)))*t0*(fff0*t0 + fff1*t1 + fff2*t2) + ((scalar_t(1) / scalar_t(2)))*t1*(fff1*t0 + fff3*t1 + fff4*t2) + ((scalar_t(1) / scalar_t(2)))*t2*(fff2*t0 + fff4*t1 + fff5*t2);
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int scalar_potential_tet4_objective_steps_affine_mesh_soa(
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
    return sfem::codegen::scalar_potential_tet4_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int scalar_potential_tet4_objective_steps_affine_mesh_soa_float(
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
    return sfem::codegen::scalar_potential_tet4_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int scalar_potential_tet4_objective_steps_packed_affine_mesh_soa(
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
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const scalar_t x0 = pack_u_base[ev0];
                const scalar_t x1 = pack_u_base[ev1];
                const scalar_t x2 = pack_u_base[ev2];
                const scalar_t x3 = pack_u_base[ev3];
                const scalar_t h0 = pack_h[ev0];
                const scalar_t h1 = pack_h[ev1];
                const scalar_t h2 = pack_h[ev2];
                const scalar_t h3 = pack_h[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    const scalar_t u0 = x0 + alpha * h0;
                    const scalar_t u1 = x1 + alpha * h1;
                    const scalar_t u2 = x2 + alpha * h2;
                    const scalar_t u3 = x3 + alpha * h3;
                    const scalar_t t0 = -u0 + u1;
                    const scalar_t t1 = -u0 + u2;
                    const scalar_t t2 = -u0 + u3;
                    value[(ptrdiff_t)step * nelements + element] = ((scalar_t(1) / scalar_t(2)))*t0*(fff0*t0 + fff1*t1 + fff2*t2) + ((scalar_t(1) / scalar_t(2)))*t1*(fff1*t0 + fff3*t1 + fff4*t2) + ((scalar_t(1) / scalar_t(2)))*t2*(fff2*t0 + fff4*t1 + fff5*t2);
                }
            }

        }
    }
    return SFEM_SUCCESS;
}

extern "C" int scalar_potential_tet4_objective_steps_packed_affine_mesh_soa_float(
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
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const scalar_t x0 = pack_u_base[ev0];
                const scalar_t x1 = pack_u_base[ev1];
                const scalar_t x2 = pack_u_base[ev2];
                const scalar_t x3 = pack_u_base[ev3];
                const scalar_t h0 = pack_h[ev0];
                const scalar_t h1 = pack_h[ev1];
                const scalar_t h2 = pack_h[ev2];
                const scalar_t h3 = pack_h[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    const scalar_t u0 = x0 + alpha * h0;
                    const scalar_t u1 = x1 + alpha * h1;
                    const scalar_t u2 = x2 + alpha * h2;
                    const scalar_t u3 = x3 + alpha * h3;
                    const scalar_t t0 = -u0 + u1;
                    const scalar_t t1 = -u0 + u2;
                    const scalar_t t2 = -u0 + u3;
                    value[(ptrdiff_t)step * nelements + element] = ((scalar_t(1) / scalar_t(2)))*t0*(fff0*t0 + fff1*t1 + fff2*t2) + ((scalar_t(1) / scalar_t(2)))*t1*(fff1*t0 + fff3*t1 + fff4*t2) + ((scalar_t(1) / scalar_t(2)))*t2*(fff2*t0 + fff4*t1 + fff5*t2);
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
static SFEM_INLINE int scalar_potential_tet4_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * SPATIAL_DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }

        const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
        const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
        const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        {
            const int q = 0;  // TET4 evaluates in closed form
            scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
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
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            scalar_potential_d3_simplex_tet4_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_value);

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

extern "C" int scalar_potential_tet4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::scalar_potential_tet4_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int scalar_potential_tet4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::scalar_potential_tet4_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int scalar_potential_tet4_objective_steps_packed_isoparametric_mesh_soa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                }
            }
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    scalar_potential_d3_simplex_tet4_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_value);

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

extern "C" int scalar_potential_tet4_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                }
            }
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = block_u_base_data[shape * N_FIELD_COMPONENTS + d][lane] + alpha * block_h_data[shape * N_FIELD_COMPONENTS + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    scalar_potential_d3_simplex_tet4_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_value);

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

static const KernelDiagnostics scalar_potential_tet4_gradient_soa_diagnostics_data = {
    "scalar_potential_tet4_gradient_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *scalar_potential_tet4_gradient_soa_diagnostics(void) {
    return &sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data;
}

extern "C" double scalar_potential_tet4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void scalar_potential_tet4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "scalar_potential_tet4_gradient_soa",
            &sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "scalar_potential_tet4_gradient_soa_float",
            &sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void scalar_potential_tet4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "scalar_potential_tet4_gradient_affine_mesh_soa",
            &sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "scalar_potential_tet4_gradient_affine_mesh_soa_float",
            &sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void scalar_potential_tet4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "scalar_potential_tet4_gradient_isoparametric_mesh_soa",
            &sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "scalar_potential_tet4_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::scalar_potential_tet4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int scalar_potential_tet4_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric3,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric4,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric5,
        const scalar_t kappa,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const scalar_t u0 = ux[ev0 * u_stride];
        const scalar_t u1 = ux[ev1 * u_stride];
        const scalar_t u2 = ux[ev2 * u_stride];
        const scalar_t u3 = ux[ev3 * u_stride];
        const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
        const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
        const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
        const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
        const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
        const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
        const scalar_t t0 = -u0 + u1;
        const scalar_t t1 = -u0 + u2;
        const scalar_t t2 = -u0 + u3;
        const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
        const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
        const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
        const scalar_t e0 = -t3 - t4 - t5;
        #pragma omp atomic update
        outx[ev0 * out_stride] += e0;
        const scalar_t e1 = t3;
        #pragma omp atomic update
        outx[ev1 * out_stride] += e1;
        const scalar_t e2 = t4;
        #pragma omp atomic update
        outx[ev2 * out_stride] += e2;
        const scalar_t e3 = t5;
        #pragma omp atomic update
        outx[ev3 * out_stride] += e3;
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int scalar_potential_tet4_gradient_affine_mesh_soa(
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
    return sfem::codegen::scalar_potential_tet4_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int scalar_potential_tet4_gradient_affine_mesh_soa_float(
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
    return sfem::codegen::scalar_potential_tet4_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, u_stride, ux, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int scalar_potential_tet4_gradient_packed_affine_mesh_soa(
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
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const scalar_t u0 = pack_u[ev0];
                const scalar_t u1 = pack_u[ev1];
                const scalar_t u2 = pack_u[ev2];
                const scalar_t u3 = pack_u[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_gradient_packed_affine_mesh_soa_float(
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
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const scalar_t u0 = pack_u[ev0];
                const scalar_t u1 = pack_u[ev1];
                const scalar_t u2 = pack_u[ev2];
                const scalar_t u3 = pack_u[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_gradient_packed_two_pass_affine_mesh_soa(
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
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const scalar_t u0 = pack_u[ev0];
                const scalar_t u1 = pack_u[ev1];
                const scalar_t u2 = pack_u[ev2];
                const scalar_t u3 = pack_u[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_gradient_packed_two_pass_affine_mesh_soa_float(
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
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

            for (ptrdiff_t element = e_start; element < e_end; ++element) {
                const uint16_t ev0 = elements[0][element];
                const uint16_t ev1 = elements[1][element];
                const uint16_t ev2 = elements[2][element];
                const uint16_t ev3 = elements[3][element];
                const scalar_t u0 = pack_u[ev0];
                const scalar_t u1 = pack_u[ev1];
                const scalar_t u2 = pack_u[ev2];
                const scalar_t u3 = pack_u[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
static SFEM_INLINE int scalar_potential_tet4_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * SPATIAL_DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        {
            const int q = 0;  // TET4 evaluates in closed form
            scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
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
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }

        scalar_potential_d3_simplex_tet4_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_out_streams);

        scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * N_FIELD_COMPONENTS + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int scalar_potential_tet4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::scalar_potential_tet4_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int scalar_potential_tet4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::scalar_potential_tet4_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int scalar_potential_tet4_gradient_packed_isoparametric_mesh_soa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_gradient_packed_isoparametric_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const u_components[N_FIELD_COMPONENTS] = {ux};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
                scalar_t block_u_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

static const KernelDiagnostics scalar_potential_tet4_apply_soa_diagnostics_data = {
    "scalar_potential_tet4_apply_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *scalar_potential_tet4_apply_soa_diagnostics(void) {
    return &sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data;
}

extern "C" double scalar_potential_tet4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void scalar_potential_tet4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "scalar_potential_tet4_apply_soa",
            &sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "scalar_potential_tet4_apply_soa_float",
            &sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void scalar_potential_tet4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "scalar_potential_tet4_apply_affine_mesh_soa",
            &sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "scalar_potential_tet4_apply_affine_mesh_soa_float",
            &sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void scalar_potential_tet4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "scalar_potential_tet4_apply_isoparametric_mesh_soa",
            &sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void scalar_potential_tet4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "scalar_potential_tet4_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::scalar_potential_tet4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int scalar_potential_tet4_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric3,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric4,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric5,
        const scalar_t kappa,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const idx_t ev0 = elements[0][element];
        const idx_t ev1 = elements[1][element];
        const idx_t ev2 = elements[2][element];
        const idx_t ev3 = elements[3][element];
        const scalar_t u0 = hx[ev0 * h_stride];
        const scalar_t u1 = hx[ev1 * h_stride];
        const scalar_t u2 = hx[ev2 * h_stride];
        const scalar_t u3 = hx[ev3 * h_stride];
        const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
        const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
        const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
        const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
        const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
        const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
        const scalar_t t0 = -u0 + u1;
        const scalar_t t1 = -u0 + u2;
        const scalar_t t2 = -u0 + u3;
        const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
        const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
        const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
        const scalar_t e0 = -t3 - t4 - t5;
        #pragma omp atomic update
        outx[ev0 * out_stride] += e0;
        const scalar_t e1 = t3;
        #pragma omp atomic update
        outx[ev1 * out_stride] += e1;
        const scalar_t e2 = t4;
        #pragma omp atomic update
        outx[ev2 * out_stride] += e2;
        const scalar_t e3 = t5;
        #pragma omp atomic update
        outx[ev3 * out_stride] += e3;
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int scalar_potential_tet4_apply_affine_mesh_soa(
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
    return sfem::codegen::scalar_potential_tet4_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int scalar_potential_tet4_apply_affine_mesh_soa_float(
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
    return sfem::codegen::scalar_potential_tet4_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, h_stride, hx, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int scalar_potential_tet4_apply_packed_affine_mesh_soa(
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
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
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
                const scalar_t u0 = pack_h[ev0];
                const scalar_t u1 = pack_h[ev1];
                const scalar_t u2 = pack_h[ev2];
                const scalar_t u3 = pack_h[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_apply_packed_affine_mesh_soa_float(
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
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
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
                const scalar_t u0 = pack_h[ev0];
                const scalar_t u1 = pack_h[ev1];
                const scalar_t u2 = pack_h[ev2];
                const scalar_t u3 = pack_h[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_apply_packed_two_pass_affine_mesh_soa(
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
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
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
                const scalar_t u0 = pack_h[ev0];
                const scalar_t u1 = pack_h[ev1];
                const scalar_t u2 = pack_h[ev2];
                const scalar_t u3 = pack_h[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_apply_packed_two_pass_affine_mesh_soa_float(
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
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_q_weight = sfem::codegen::scalar_potential_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
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
                const scalar_t u0 = pack_h[ev0];
                const scalar_t u1 = pack_h[ev1];
                const scalar_t u2 = pack_h[ev2];
                const scalar_t u3 = pack_h[ev3];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                const scalar_t fff3 = kappa * scalar_t(g_geom_metric3[element]);
                const scalar_t fff4 = kappa * scalar_t(g_geom_metric4[element]);
                const scalar_t fff5 = kappa * scalar_t(g_geom_metric5[element]);
                const scalar_t t0 = -u0 + u1;
                const scalar_t t1 = -u0 + u2;
                const scalar_t t2 = -u0 + u3;
                const scalar_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
                const scalar_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
                const scalar_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
                const scalar_t e0 = -t3 - t4 - t5;
                pack_out[ev0] += e0;
                const scalar_t e1 = t3;
                pack_out[ev1] += e1;
                const scalar_t e2 = t4;
                pack_out[ev2] += e2;
                const scalar_t e3 = t5;
                pack_out[ev3] += e3;
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
static SFEM_INLINE int scalar_potential_tet4_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx
) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * SPATIAL_DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        {
            const int q = 0;  // TET4 evaluates in closed form
            scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
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
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }

        scalar_potential_d3_simplex_tet4_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_h_streams, block_out_streams);

        scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * N_FIELD_COMPONENTS + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int scalar_potential_tet4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::scalar_potential_tet4_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int scalar_potential_tet4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::scalar_potential_tet4_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx);
}

namespace sfem {
namespace codegen {

extern "C" int scalar_potential_tet4_apply_packed_isoparametric_mesh_soa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_apply_packed_isoparametric_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    using scalar_t = double;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

extern "C" int scalar_potential_tet4_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    using scalar_t = float;
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::scalar_potential_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)SPATIAL_DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)N_FIELD_COMPONENTS * (size_t)max_nodes_per_pack);

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y, z};
            const scalar_t *const h_components[N_FIELD_COMPONENTS] = {hx};
            scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
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
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
                for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < SPATIAL_DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * SPATIAL_DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                        }
                    }
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * N_FIELD_COMPONENTS + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * N_FIELD_COMPONENTS + d][lane] = scalar_t(0);
                        }
                    }
                }


                {
                    const int q = 0;  // TET4 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
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
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
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
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                scalar_potential_d3_simplex_tet4_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * N_FIELD_COMPONENTS + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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

    scalar_t *const out_components[N_FIELD_COMPONENTS] = {outx};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
        const idx_t dest = ghost_reduce_dest[row];
        const ptrdiff_t begin = ghost_reduce_ptr[row];
        const ptrdiff_t end = ghost_reduce_ptr[row + 1];
        for (int d = 0; d < N_FIELD_COMPONENTS; ++d) {
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
