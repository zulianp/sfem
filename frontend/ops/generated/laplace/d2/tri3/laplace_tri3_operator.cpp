#include <cstdio>
#include <type_traits>
#include "../laplace_d2_simplex_local.hpp"
#include "../laplace_d2_simplex_hessian.hpp"
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
struct laplace_tri3_affine_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[3] = {scalar_t(0.33333333333333343), scalar_t(0.33333333333333331), scalar_t(0.33333333333333331)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[3] = {scalar_t(-1), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[3] = {scalar_t(-1), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[1] = {scalar_t(0.5)};
        return data;
    }
};

template <typename scalar_t>
struct laplace_tri3_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[3] = {scalar_t(0.33333333333333343), scalar_t(0.33333333333333331), scalar_t(0.33333333333333331)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[3] = {scalar_t(-1), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[3] = {scalar_t(-1), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[1] = {scalar_t(0.5)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tri3_objective_soa_diagnostics_data = {
    "laplace_tri3_objective_soa",
    "TRI3",
    2,
    1,
    3,
    16,
    1,
    1,
    2,
    0,
    0,
    2,
    0,
    0,
    0,
    2,
    1,
    5,
    0,
    0,
    0,
    3,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_objective_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data;
}

extern "C" double laplace_tri3_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tri3_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tri3_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tri3_objective_soa",
            &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tri3_objective_soa_float",
            &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tri3_objective_affine_mesh_soa",
            &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tri3_objective_affine_mesh_soa_float",
            &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tri3_objective_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tri3_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tri3_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
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
        const scalar_t x0 = ux[ev0 * u_stride];
        const scalar_t x1 = ux[ev1 * u_stride];
        const scalar_t x2 = ux[ev2 * u_stride];
        const scalar_t h0 = hx[ev0 * h_stride];
        const scalar_t h1 = hx[ev1 * h_stride];
        const scalar_t h2 = hx[ev2 * h_stride];
        const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
        const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
        const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            const scalar_t u0 = x0 + alpha * h0;
            const scalar_t u1 = x1 + alpha * h1;
            const scalar_t u2 = x2 + alpha * h2;
            const scalar_t t0 = -u0 + u1;
            const scalar_t t1 = -u0 + u2;
            value[(ptrdiff_t)step * nelements + element] = ((scalar_t(1) / scalar_t(2)))*t0*(fff0*t0 + fff1*t1) + ((scalar_t(1) / scalar_t(2)))*t1*(fff1*t0 + fff2*t1);
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_objective_steps_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_tri3_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_tri3_objective_steps_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::laplace_tri3_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_tri3_objective_steps_packed_affine_mesh_soa(
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
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_q_weight = sfem::codegen::laplace_tri3_affine_reference_data<scalar_t>::q_weight();

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
                const scalar_t x0 = pack_u_base[ev0];
                const scalar_t x1 = pack_u_base[ev1];
                const scalar_t x2 = pack_u_base[ev2];
                const scalar_t h0 = pack_h[ev0];
                const scalar_t h1 = pack_h[ev1];
                const scalar_t h2 = pack_h[ev2];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    const scalar_t u0 = x0 + alpha * h0;
                    const scalar_t u1 = x1 + alpha * h1;
                    const scalar_t u2 = x2 + alpha * h2;
                    const scalar_t t0 = -u0 + u1;
                    const scalar_t t1 = -u0 + u2;
                    value[(ptrdiff_t)step * nelements + element] = ((scalar_t(1) / scalar_t(2)))*t0*(fff0*t0 + fff1*t1) + ((scalar_t(1) / scalar_t(2)))*t1*(fff1*t0 + fff2*t1);
                }
            }

        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tri3_objective_steps_packed_affine_mesh_soa_float(
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
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_q_weight = sfem::codegen::laplace_tri3_affine_reference_data<scalar_t>::q_weight();

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
                const scalar_t x0 = pack_u_base[ev0];
                const scalar_t x1 = pack_u_base[ev1];
                const scalar_t x2 = pack_u_base[ev2];
                const scalar_t h0 = pack_h[ev0];
                const scalar_t h1 = pack_h[ev1];
                const scalar_t h2 = pack_h[ev2];
                const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
                const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
                const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    const scalar_t u0 = x0 + alpha * h0;
                    const scalar_t u1 = x1 + alpha * h1;
                    const scalar_t u2 = x2 + alpha * h2;
                    const scalar_t t0 = -u0 + u1;
                    const scalar_t t1 = -u0 + u2;
                    value[(ptrdiff_t)step * nelements + element] = ((scalar_t(1) / scalar_t(2)))*t0*(fff0*t0 + fff1*t1) + ((scalar_t(1) / scalar_t(2)))*t1*(fff1*t0 + fff2*t1);
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
static SFEM_INLINE int laplace_tri3_objective_steps_isoparametric_mesh_soa_impl(
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
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::q_weight();

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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y};

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
            const int q = 0;  // TRI3 evaluates in closed form
            scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
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

            laplace_d2_simplex_tri3_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_value);

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

extern "C" int laplace_tri3_objective_steps_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_tri3_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_tri3_objective_steps_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_tri3_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int laplace_tri3_objective_steps_packed_isoparametric_mesh_soa(
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
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::q_weight();

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y};
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
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2]};

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
                    const int q = 0;  // TRI3 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
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

                    laplace_d2_simplex_tri3_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_value);

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

extern "C" int laplace_tri3_objective_steps_packed_isoparametric_mesh_soa_float(
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
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::q_weight();

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
            const geom_t *const coordinate_components[SPATIAL_DIM] = {x, y};
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
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};

                const scalar_t *block_u_streams[N_SHAPE * N_FIELD_COMPONENTS] = {block_u_data[0], block_u_data[1], block_u_data[2]};

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
                    const int q = 0;  // TRI3 evaluates in closed form
                scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
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

                    laplace_d2_simplex_tri3_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_value);

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

static const KernelDiagnostics laplace_tri3_gradient_soa_diagnostics_data = {
    "laplace_tri3_gradient_soa",
    "TRI3",
    2,
    1,
    3,
    16,
    1,
    0,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    2,
    2,
    0,
    0,
    0,
    2,
    5,
    6,
    1,
    2,
    6,
    0,
    3,
    3,
    3,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_gradient_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data;
}

extern "C" double laplace_tri3_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tri3_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tri3_gradient_soa",
            &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tri3_gradient_soa_float",
            &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tri3_gradient_affine_mesh_soa",
            &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tri3_gradient_affine_mesh_soa_float",
            &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tri3_gradient_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tri3_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tri3_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
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
        const scalar_t u0 = ux[ev0 * u_stride];
        const scalar_t u1 = ux[ev1 * u_stride];
        const scalar_t u2 = ux[ev2 * u_stride];
        const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
        const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
        const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
        const scalar_t t0 = -u0 + u1;
        const scalar_t t1 = -u0 + u2;
        const scalar_t t2 = fff0*t0 + fff1*t1;
        const scalar_t t3 = fff1*t0 + fff2*t1;
        const scalar_t e0 = -t2 - t3;
        #pragma omp atomic update
        outx[ev0 * out_stride] += e0;
        const scalar_t e1 = t2;
        #pragma omp atomic update
        outx[ev1 * out_stride] += e1;
        const scalar_t e2 = t3;
        #pragma omp atomic update
        outx[ev2 * out_stride] += e2;
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_gradient_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tri3_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_tri3_gradient_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tri3_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, kappa, u_stride, ux, out_stride, outx);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int laplace_tri3_gradient_isoparametric_mesh_soa_impl(
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
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::q_weight();

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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y};

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
            const int q = 0;  // TRI3 evaluates in closed form
            scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
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

        laplace_d2_simplex_tri3_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_u_streams, block_out_streams);

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

extern "C" int laplace_tri3_gradient_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_tri3_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_tri3_gradient_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_tri3_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tri3_apply_soa_diagnostics_data = {
    "laplace_tri3_apply_soa",
    "TRI3",
    2,
    1,
    3,
    16,
    1,
    0,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    2,
    0,
    0,
    0,
    2,
    5,
    6,
    1,
    2,
    0,
    6,
    3,
    3,
    3,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_apply_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data;
}

extern "C" double laplace_tri3_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::laplace_tri3_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tri3_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tri3_apply_soa",
            &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tri3_apply_soa_float",
            &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tri3_apply_affine_mesh_soa",
            &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tri3_apply_affine_mesh_soa_float",
            &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tri3_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tri3_apply_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tri3_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tri3_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tri3_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
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
        const scalar_t u0 = hx[ev0 * h_stride];
        const scalar_t u1 = hx[ev1 * h_stride];
        const scalar_t u2 = hx[ev2 * h_stride];
        const scalar_t fff0 = kappa * scalar_t(g_geom_metric0[element]);
        const scalar_t fff1 = kappa * scalar_t(g_geom_metric1[element]);
        const scalar_t fff2 = kappa * scalar_t(g_geom_metric2[element]);
        const scalar_t t0 = -u0 + u1;
        const scalar_t t1 = -u0 + u2;
        const scalar_t t2 = fff0*t0 + fff1*t1;
        const scalar_t t3 = fff1*t0 + fff2*t1;
        const scalar_t e0 = -t2 - t3;
        #pragma omp atomic update
        outx[ev0 * out_stride] += e0;
        const scalar_t e1 = t2;
        #pragma omp atomic update
        outx[ev1 * out_stride] += e1;
        const scalar_t e2 = t3;
        #pragma omp atomic update
        outx[ev2 * out_stride] += e2;
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_apply_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tri3_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_tri3_apply_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric0,
        const geom_t *const SFEM_RESTRICT g_geom_metric1,
        const geom_t *const SFEM_RESTRICT g_geom_metric2,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
) {
    return sfem::codegen::laplace_tri3_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, kappa, h_stride, hx, out_stride, outx);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int laplace_tri3_apply_isoparametric_mesh_soa_impl(
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
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::q_weight();

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
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[SPATIAL_DIM] = {x, y};

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
            const int q = 0;  // TRI3 evaluates in closed form
            scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
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

        laplace_d2_simplex_tri3_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_q_weight, kappa, block_h_streams, block_out_streams);

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

extern "C" int laplace_tri3_apply_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_tri3_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_tri3_apply_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_tri3_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx);
}


namespace sfem {
namespace codegen {

static SFEM_INLINE void laplace_tri3_hessian_isoparametric_mesh_soa_find_cols(
        const idx_t *const SFEM_RESTRICT targets,
        const idx_t *const SFEM_RESTRICT row,
        const int lenrow,
        idx_t *const SFEM_RESTRICT ks) {
#pragma unroll(3)
    for (int d = 0; d < 3; ++d) {
        ks[d] = 0;
    }
    for (int k = 0; k < lenrow; ++k) {
#pragma unroll(3)
        for (int d = 0; d < 3; ++d) {
            ks[d] += row[k] < targets[d];
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void laplace_tri3_hessian_isoparametric_mesh_soa_scatter_bsr(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_SHAPE = 3;
    count_t entries[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    for (int i = 0; i < N_SHAPE; ++i) {
        const idx_t dof_i = ev[i];
        const count_t row_begin = rowptr[dof_i];
        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        laplace_tri3_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            entries[i * N_SHAPE + j] = row_begin + ks[j];
        }
    }
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            scalar_t *const block = &values[entries[i * N_SHAPE + j] * N_FIELD_COMPONENTS * N_FIELD_COMPONENTS];
            for (int bi = 0; bi < N_FIELD_COMPONENTS; ++bi) {
                const int row = bi * N_SHAPE + i;
                for (int bj = 0; bj < N_FIELD_COMPONENTS; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    block[bi * N_FIELD_COMPONENTS + bj] += element_matrix[row * (N_FIELD_COMPONENTS * N_SHAPE) + col];
                }
            }
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void laplace_tri3_hessian_isoparametric_mesh_soa_scatter_crs(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int N_SHAPE = 3;
    count_t row_begin[N_SHAPE];
    int lenrow[N_SHAPE];
    int local_col[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    for (int i = 0; i < N_SHAPE; ++i) {
        row_begin[i] = rowptr[ev[i]];
        lenrow[i] = (int)(rowptr[ev[i] + 1] - row_begin[i]);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin[i]];
        laplace_tri3_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow[i], ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            local_col[i * N_SHAPE + j] = (int)ks[j];
        }
    }
    for (int i = 0; i < N_SHAPE; ++i) {
        const count_t rb = row_begin[i];
        const int lr = lenrow[i];
        for (int j = 0; j < N_SHAPE; ++j) {
            const int lc = local_col[i * N_SHAPE + j];
            for (int bi = 0; bi < N_FIELD_COMPONENTS; ++bi) {
                const int row = bi * N_SHAPE + i;
                scalar_t *const row_values = &values[rb * N_FIELD_COMPONENTS * N_FIELD_COMPONENTS + bi * lr * N_FIELD_COMPONENTS];
                for (int bj = 0; bj < N_FIELD_COMPONENTS; ++bj) {
                    const int col = bj * N_SHAPE + j;
#pragma omp atomic update
                    row_values[lc * N_FIELD_COMPONENTS + bj] += element_matrix[row * (N_FIELD_COMPONENTS * N_SHAPE) + col];
                }
            }
        }
    }
}

template <typename scalar_t, typename geometry_t, int FORMAT>
static int laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
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
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 3;
    static constexpr int VECTOR_SIZE = 1;
    static constexpr int NDOFS = N_FIELD_COMPONENTS * N_SHAPE;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tri3_isoparametric_reference_data<scalar_t>::q_weight();

    int unsupported_matrix_format = 0;
#pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[NDOFS * NDOFS];
        scalar_t block_h_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * N_FIELD_COMPONENTS][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * SPATIAL_DIM][VECTOR_SIZE];
        static constexpr int nelems = VECTOR_SIZE;
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        const scalar_t *block_h_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * N_FIELD_COMPONENTS];
        for (int stream = 0; stream < N_SHAPE * N_FIELD_COMPONENTS; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < SPATIAL_DIM; ++d) {
                block_coordinate_data[shape * SPATIAL_DIM + d][0] = scalar_t(points[d][node]);
            }
        }


        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[SPATIAL_DIM * SPATIAL_DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
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

        laplace_d2_simplex_direct_hessian_reference_element_matrix<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, kappa, element_matrix);

        if constexpr (FORMAT == 1) {
            laplace_tri3_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
        } else if constexpr (FORMAT == 0) {
            laplace_tri3_hessian_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
        } else {
            unsupported_matrix_format |= 1;
        }
    }

    return unsupported_matrix_format ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tri3_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tri3_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 0>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tri3_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int laplace_tri3_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tri3_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
