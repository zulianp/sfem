#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../laplace_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>

#include "tet4_laplacian_inline_cpu.hpp"

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
struct laplace_tet4_affine_reference_data {
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
struct laplace_tet4_isoparametric_reference_data {
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

static const KernelDiagnostics laplace_tet4_residual_element_soa_diagnostics_data = {
    "laplace_tet4_residual_element_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    2,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    1,
    6,
    0,
    0,
    0,
    7,
    10,
    16,
    1,
    1,
    4,
    0,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_residual_element_soa",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_residual_element_soa_float",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_residual_affine_mesh_soa",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tet4_jacobian_u_u_diagnostics_data = {
    "laplace_tet4_jacobian_u_u",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    2,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    1,
    6,
    0,
    0,
    0,
    7,
    10,
    16,
    1,
    1,
    0,
    4,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_tet4_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_tet4_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_tet4_jacobian_u_u_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet4_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_u_u",
            &sfem::codegen::laplace_tet4_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_u_u_float",
            &sfem::codegen::laplace_tet4_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tet4_jacobian_action_element_soa_diagnostics_data = {
    "laplace_tet4_jacobian_action_element_soa",
    "TET4",
    3,
    1,
    4,
    16,
    1,
    2,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    1,
    6,
    0,
    0,
    0,
    7,
    10,
    16,
    1,
    1,
    0,
    4,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_action_element_soa",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int laplace_tet4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[4],
        const double kappa,
        double *const SFEM_RESTRICT output[4]
) {
    sfem::codegen::laplace_d3_simplex_residual_block<double, 1, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::q_weight(), current, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[4],
        const float kappa,
        float *const SFEM_RESTRICT output[4]
) {
    sfem::codegen::laplace_d3_simplex_residual_block<float, 1, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::q_weight(), current, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tet4_residual_affine_mesh_soa_impl(
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
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    if (current_stride == 1 && out_stride == 1) {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0];
                const scalar_t u1 = u[ev1];
                const scalar_t u2 = u[ev2];
                const scalar_t u3 = u[ev3];
                const scalar_t fff0 = scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0];
                const scalar_t u1 = u[ev1];
                const scalar_t u2 = u[ev2];
                const scalar_t u3 = u[ev3];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        }
    } else {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0 * current_stride];
                const scalar_t u1 = u[ev1 * current_stride];
                const scalar_t u2 = u[ev2 * current_stride];
                const scalar_t u3 = u[ev3 * current_stride];
                const scalar_t fff0 = scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0 * current_stride];
                const scalar_t u1 = u[ev1 * current_stride];
                const scalar_t u2 = u[ev2 * current_stride];
                const scalar_t u3 = u[ev3 * current_stride];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_residual_affine_mesh_soa(
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
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_tet4_residual_affine_mesh_soa_float(
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
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_tet4_residual_affine_mesh_soa_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    (void)DIM;
    (void)N_QP;
    (void)N_SHAPE;
    (void)nnodes;
    if (current_stride == 1 && out_stride == 1) {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                scalar_t element_vector[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0];
                const scalar_t u1 = u[ev1];
                const scalar_t u2 = u[ev2];
                const scalar_t u3 = u[ev3];
                const scalar_t x0 = fff[0] + fff[1] + fff[2];
                const scalar_t x1 = fff[1] + fff[3] + fff[4];
                const scalar_t x2 = fff[2] + fff[4] + fff[5];
                const scalar_t x3 = fff[1] * u0;
                const scalar_t x4 = fff[2] * u0;
                const scalar_t x5 = fff[4] * u0;
                element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
                element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
                element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
                    #pragma omp atomic update
                    u_out[ev0] += element_vector[0];
                    #pragma omp atomic update
                    u_out[ev1] += element_vector[1];
                    #pragma omp atomic update
                    u_out[ev2] += element_vector[2];
                    #pragma omp atomic update
                    u_out[ev3] += element_vector[3];
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0];
                const scalar_t u1 = u[ev1];
                const scalar_t u2 = u[ev2];
                const scalar_t u3 = u[ev3];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        }
    } else {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0 * current_stride];
                const scalar_t u1 = u[ev1 * current_stride];
                const scalar_t u2 = u[ev2 * current_stride];
                const scalar_t u3 = u[ev3 * current_stride];
                const scalar_t fff0 = scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0 * current_stride];
                const scalar_t u1 = u[ev1 * current_stride];
                const scalar_t u2 = u[ev2 * current_stride];
                const scalar_t u3 = u[ev3 * current_stride];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_residual_affine_mesh_soa_aos_unit(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double *const SFEM_RESTRICT u,
        double *const SFEM_RESTRICT u_out
) {
    using scalar_t = double;
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        scalar_t element_vector[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        const idx_t ev0 = elements[0][i];
        const idx_t ev1 = elements[1][i];
        const idx_t ev2 = elements[2][i];
        const idx_t ev3 = elements[3][i];
        const scalar_t u0 = u[ev0];
        const scalar_t u1 = u[ev1];
        const scalar_t u2 = u[ev2];
        const scalar_t u3 = u[ev3];
        const scalar_t x0 = fff[0] + fff[1] + fff[2];
        const scalar_t x1 = fff[1] + fff[3] + fff[4];
        const scalar_t x2 = fff[2] + fff[4] + fff[5];
        const scalar_t x3 = fff[1] * u0;
        const scalar_t x4 = fff[2] * u0;
        const scalar_t x5 = fff[4] * u0;
        element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
        element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
        element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
        element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
            #pragma omp atomic update
            u_out[ev0] += element_vector[0];
            #pragma omp atomic update
            u_out[ev1] += element_vector[1];
            #pragma omp atomic update
            u_out[ev2] += element_vector[2];
            #pragma omp atomic update
            u_out[ev3] += element_vector[3];
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_residual_affine_mesh_soa_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    (void)DIM;
    (void)N_QP;
    (void)N_SHAPE;
    (void)nnodes;
    if (current_stride == 1 && out_stride == 1) {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                scalar_t element_vector[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0];
                const scalar_t u1 = u[ev1];
                const scalar_t u2 = u[ev2];
                const scalar_t u3 = u[ev3];
                const scalar_t x0 = fff[0] + fff[1] + fff[2];
                const scalar_t x1 = fff[1] + fff[3] + fff[4];
                const scalar_t x2 = fff[2] + fff[4] + fff[5];
                const scalar_t x3 = fff[1] * u0;
                const scalar_t x4 = fff[2] * u0;
                const scalar_t x5 = fff[4] * u0;
                element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
                element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
                element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
                    #pragma omp atomic update
                    u_out[ev0] += element_vector[0];
                    #pragma omp atomic update
                    u_out[ev1] += element_vector[1];
                    #pragma omp atomic update
                    u_out[ev2] += element_vector[2];
                    #pragma omp atomic update
                    u_out[ev3] += element_vector[3];
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0];
                const scalar_t u1 = u[ev1];
                const scalar_t u2 = u[ev2];
                const scalar_t u3 = u[ev3];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        }
    } else {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0 * current_stride];
                const scalar_t u1 = u[ev1 * current_stride];
                const scalar_t u2 = u[ev2 * current_stride];
                const scalar_t u3 = u[ev3 * current_stride];
                const scalar_t fff0 = scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u[ev0 * current_stride];
                const scalar_t u1 = u[ev1 * current_stride];
                const scalar_t u2 = u[ev2 * current_stride];
                const scalar_t u3 = u[ev3 * current_stride];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_residual_affine_mesh_soa_aos_unit_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float *const SFEM_RESTRICT u,
        float *const SFEM_RESTRICT u_out
) {
    using scalar_t = float;
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        scalar_t element_vector[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        const idx_t ev0 = elements[0][i];
        const idx_t ev1 = elements[1][i];
        const idx_t ev2 = elements[2][i];
        const idx_t ev3 = elements[3][i];
        const scalar_t u0 = u[ev0];
        const scalar_t u1 = u[ev1];
        const scalar_t u2 = u[ev2];
        const scalar_t u3 = u[ev3];
        const scalar_t x0 = fff[0] + fff[1] + fff[2];
        const scalar_t x1 = fff[1] + fff[3] + fff[4];
        const scalar_t x2 = fff[2] + fff[4] + fff[5];
        const scalar_t x3 = fff[1] * u0;
        const scalar_t x4 = fff[2] * u0;
        const scalar_t x5 = fff[4] * u0;
        element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
        element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
        element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
        element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
            #pragma omp atomic update
            u_out[ev0] += element_vector[0];
            #pragma omp atomic update
            u_out[ev1] += element_vector[1];
            #pragma omp atomic update
            u_out[ev2] += element_vector[2];
            #pragma omp atomic update
            u_out[ev3] += element_vector[3];
    }
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_tet4_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
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
        const scalar_t *const current_components[N_FIELDS] = {u};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                }
            }
        }

        for (int stream = 0; stream < 4; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
                const scalar_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
                const scalar_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
                const scalar_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
                const scalar_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
                const scalar_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_simplex_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_current, kappa, block_output);

        scalar_t *const output_components[N_FIELDS] = {u_out};
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

extern "C" int laplace_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_tet4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    return laplace_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_tet4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    return laplace_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_tet4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT direction[4],
        const double kappa,
        double *const SFEM_RESTRICT output[4]
) {
    sfem::codegen::laplace_d3_simplex_jacobian_action_block<double, 1, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::q_weight(), direction, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT direction[4],
        const float kappa,
        float *const SFEM_RESTRICT output[4]
) {
    sfem::codegen::laplace_d3_simplex_jacobian_action_block<float, 1, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::q_weight(), direction, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tet4_jacobian_action_affine_mesh_soa_impl(
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
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    if (direction_stride == 1 && out_stride == 1) {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0];
                const scalar_t u1 = u_direction[ev1];
                const scalar_t u2 = u_direction[ev2];
                const scalar_t u3 = u_direction[ev3];
                const scalar_t fff0 = scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0];
                const scalar_t u1 = u_direction[ev1];
                const scalar_t u2 = u_direction[ev2];
                const scalar_t u3 = u_direction[ev3];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        }
    } else {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0 * direction_stride];
                const scalar_t u1 = u_direction[ev1 * direction_stride];
                const scalar_t u2 = u_direction[ev2 * direction_stride];
                const scalar_t u3 = u_direction[ev3 * direction_stride];
                const scalar_t fff0 = scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0 * direction_stride];
                const scalar_t u1 = u_direction[ev1 * direction_stride];
                const scalar_t u2 = u_direction[ev2 * direction_stride];
                const scalar_t u3 = u_direction[ev3 * direction_stride];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric0[i]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric1[i]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric2[i]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric3[i]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric4[i]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric5[i]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_jacobian_action_affine_mesh_soa(
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
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet4_jacobian_action_affine_mesh_soa_float(
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
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet4_jacobian_action_affine_mesh_soa_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    (void)DIM;
    (void)N_QP;
    (void)N_SHAPE;
    (void)nnodes;
    if (direction_stride == 1 && out_stride == 1) {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                scalar_t element_vector[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0];
                const scalar_t u1 = u_direction[ev1];
                const scalar_t u2 = u_direction[ev2];
                const scalar_t u3 = u_direction[ev3];
                const scalar_t x0 = fff[0] + fff[1] + fff[2];
                const scalar_t x1 = fff[1] + fff[3] + fff[4];
                const scalar_t x2 = fff[2] + fff[4] + fff[5];
                const scalar_t x3 = fff[1] * u0;
                const scalar_t x4 = fff[2] * u0;
                const scalar_t x5 = fff[4] * u0;
                element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
                element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
                element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
                    #pragma omp atomic update
                    u_out[ev0] += element_vector[0];
                    #pragma omp atomic update
                    u_out[ev1] += element_vector[1];
                    #pragma omp atomic update
                    u_out[ev2] += element_vector[2];
                    #pragma omp atomic update
                    u_out[ev3] += element_vector[3];
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0];
                const scalar_t u1 = u_direction[ev1];
                const scalar_t u2 = u_direction[ev2];
                const scalar_t u3 = u_direction[ev3];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        }
    } else {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0 * direction_stride];
                const scalar_t u1 = u_direction[ev1 * direction_stride];
                const scalar_t u2 = u_direction[ev2 * direction_stride];
                const scalar_t u3 = u_direction[ev3 * direction_stride];
                const scalar_t fff0 = scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0 * direction_stride];
                const scalar_t u1 = u_direction[ev1 * direction_stride];
                const scalar_t u2 = u_direction[ev2 * direction_stride];
                const scalar_t u3 = u_direction[ev3 * direction_stride];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_jacobian_action_affine_mesh_soa_aos_unit(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double *const SFEM_RESTRICT u_direction,
        double *const SFEM_RESTRICT u_out
) {
    using scalar_t = double;
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        scalar_t element_vector[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        const idx_t ev0 = elements[0][i];
        const idx_t ev1 = elements[1][i];
        const idx_t ev2 = elements[2][i];
        const idx_t ev3 = elements[3][i];
        const scalar_t u0 = u_direction[ev0];
        const scalar_t u1 = u_direction[ev1];
        const scalar_t u2 = u_direction[ev2];
        const scalar_t u3 = u_direction[ev3];
        const scalar_t x0 = fff[0] + fff[1] + fff[2];
        const scalar_t x1 = fff[1] + fff[3] + fff[4];
        const scalar_t x2 = fff[2] + fff[4] + fff[5];
        const scalar_t x3 = fff[1] * u0;
        const scalar_t x4 = fff[2] * u0;
        const scalar_t x5 = fff[4] * u0;
        element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
        element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
        element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
        element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
            #pragma omp atomic update
            u_out[ev0] += element_vector[0];
            #pragma omp atomic update
            u_out[ev1] += element_vector[1];
            #pragma omp atomic update
            u_out[ev2] += element_vector[2];
            #pragma omp atomic update
            u_out[ev3] += element_vector[3];
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_jacobian_action_affine_mesh_soa_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    (void)DIM;
    (void)N_QP;
    (void)N_SHAPE;
    (void)nnodes;
    if (direction_stride == 1 && out_stride == 1) {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                scalar_t element_vector[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0];
                const scalar_t u1 = u_direction[ev1];
                const scalar_t u2 = u_direction[ev2];
                const scalar_t u3 = u_direction[ev3];
                const scalar_t x0 = fff[0] + fff[1] + fff[2];
                const scalar_t x1 = fff[1] + fff[3] + fff[4];
                const scalar_t x2 = fff[2] + fff[4] + fff[5];
                const scalar_t x3 = fff[1] * u0;
                const scalar_t x4 = fff[2] * u0;
                const scalar_t x5 = fff[4] * u0;
                element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
                element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
                element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
                    #pragma omp atomic update
                    u_out[ev0] += element_vector[0];
                    #pragma omp atomic update
                    u_out[ev1] += element_vector[1];
                    #pragma omp atomic update
                    u_out[ev2] += element_vector[2];
                    #pragma omp atomic update
                    u_out[ev3] += element_vector[3];
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0];
                const scalar_t u1 = u_direction[ev1];
                const scalar_t u2 = u_direction[ev2];
                const scalar_t u3 = u_direction[ev3];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0] += e0;
                #pragma omp atomic update
                u_out[ev1] += e1;
                #pragma omp atomic update
                u_out[ev2] += e2;
                #pragma omp atomic update
                u_out[ev3] += e3;
            }
        }
    } else {
        if ((kappa) == scalar_t(1)) {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0 * direction_stride];
                const scalar_t u1 = u_direction[ev1 * direction_stride];
                const scalar_t u2 = u_direction[ev2 * direction_stride];
                const scalar_t u3 = u_direction[ev3 * direction_stride];
                const scalar_t fff0 = scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        } else {

            #pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < nelements; ++i) {
                const idx_t ev0 = elements[0][i];
                const idx_t ev1 = elements[1][i];
                const idx_t ev2 = elements[2][i];
                const idx_t ev3 = elements[3][i];
                const scalar_t u0 = u_direction[ev0 * direction_stride];
                const scalar_t u1 = u_direction[ev1 * direction_stride];
                const scalar_t u2 = u_direction[ev2 * direction_stride];
                const scalar_t u3 = u_direction[ev3 * direction_stride];
                const scalar_t metric_factor = kappa;
                const scalar_t fff0 = metric_factor * scalar_t(g_geom_metric[i * 6 + 0]);
                const scalar_t fff1 = metric_factor * scalar_t(g_geom_metric[i * 6 + 1]);
                const scalar_t fff2 = metric_factor * scalar_t(g_geom_metric[i * 6 + 2]);
                const scalar_t fff3 = metric_factor * scalar_t(g_geom_metric[i * 6 + 3]);
                const scalar_t fff4 = metric_factor * scalar_t(g_geom_metric[i * 6 + 4]);
                const scalar_t fff5 = metric_factor * scalar_t(g_geom_metric[i * 6 + 5]);
                const scalar_t grad0 = u1 - u0;
                const scalar_t grad1 = u2 - u0;
                const scalar_t grad2 = u3 - u0;
                const scalar_t x0 = fff0 + fff1 + fff2;
                const scalar_t x1 = fff1 + fff3 + fff4;
                const scalar_t x2 = fff2 + fff4 + fff5;
                const scalar_t x3 = fff1 * u0;
                const scalar_t x4 = fff2 * u0;
                const scalar_t x5 = fff4 * u0;
                const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
                const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
                const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
                const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
                #pragma omp atomic update
                u_out[ev0 * out_stride] += e0;
                #pragma omp atomic update
                u_out[ev1 * out_stride] += e1;
                #pragma omp atomic update
                u_out[ev2 * out_stride] += e2;
                #pragma omp atomic update
                u_out[ev3 * out_stride] += e3;
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int laplace_tet4_jacobian_action_affine_mesh_soa_aos_unit_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float *const SFEM_RESTRICT u_direction,
        float *const SFEM_RESTRICT u_out
) {
    using scalar_t = float;
    (void)nnodes;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        scalar_t element_vector[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        const idx_t ev0 = elements[0][i];
        const idx_t ev1 = elements[1][i];
        const idx_t ev2 = elements[2][i];
        const idx_t ev3 = elements[3][i];
        const scalar_t u0 = u_direction[ev0];
        const scalar_t u1 = u_direction[ev1];
        const scalar_t u2 = u_direction[ev2];
        const scalar_t u3 = u_direction[ev3];
        const scalar_t x0 = fff[0] + fff[1] + fff[2];
        const scalar_t x1 = fff[1] + fff[3] + fff[4];
        const scalar_t x2 = fff[2] + fff[4] + fff[5];
        const scalar_t x3 = fff[1] * u0;
        const scalar_t x4 = fff[2] * u0;
        const scalar_t x5 = fff[4] * u0;
        element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
        element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
        element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
        element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
            #pragma omp atomic update
            u_out[ev0] += element_vector[0];
            #pragma omp atomic update
            u_out[ev1] += element_vector[1];
            #pragma omp atomic update
            u_out[ev2] += element_vector[2];
            #pragma omp atomic update
            u_out[ev3] += element_vector[3];
    }
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t, bool UnitKappa>
static SFEM_INLINE int laplace_tet4_jacobian_action_packed_affine_mesh_soa_impl_kernel(
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
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric3,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric4,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric5,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int VECTOR_SIZE = 64;
    (void)nnodes;
    (void)max_nodes_per_pack;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_direction = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)max_nodes_per_pack);
        scalar_t out0[VECTOR_SIZE];
        scalar_t out1[VECTOR_SIZE];
        scalar_t out2[VECTOR_SIZE];
        scalar_t out3[VECTOR_SIZE];
        scalar_t u0[VECTOR_SIZE];
        scalar_t u1[VECTOR_SIZE];
        scalar_t u2[VECTOR_SIZE];
        scalar_t u3[VECTOR_SIZE];
        scalar_t fff0[VECTOR_SIZE];
        scalar_t fff1[VECTOR_SIZE];
        scalar_t fff2[VECTOR_SIZE];
        scalar_t fff3[VECTOR_SIZE];
        scalar_t fff4[VECTOR_SIZE];
        scalar_t fff5[VECTOR_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                pack_direction[k] = u_direction[(owned_nodes_ptr[pack] + k) * direction_stride];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                pack_direction[n_contiguous + k] = u_direction[ghosts[k] * direction_stride];
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                const scalar_t metric_factor = UnitKappa ? scalar_t(1) : kappa;

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    fff0[lane] = metric_factor * scalar_t(g_geom_metric0[element]);
                    fff1[lane] = metric_factor * scalar_t(g_geom_metric1[element]);
                    fff2[lane] = metric_factor * scalar_t(g_geom_metric2[element]);
                    fff3[lane] = metric_factor * scalar_t(g_geom_metric3[element]);
                    fff4[lane] = metric_factor * scalar_t(g_geom_metric4[element]);
                    fff5[lane] = metric_factor * scalar_t(g_geom_metric5[element]);
                }

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    u0[lane] = pack_direction[elements[0][element]];
                    u1[lane] = pack_direction[elements[1][element]];
                    u2[lane] = pack_direction[elements[2][element]];
                    u3[lane] = pack_direction[elements[3][element]];
                }

#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    tet4_laplacian_apply_fff_soa_tpl<scalar_t, scalar_t>(
                            fff0[lane], fff1[lane], fff2[lane], fff3[lane], fff4[lane], fff5[lane],
                            u0[lane], u1[lane], u2[lane], u3[lane],
                            &out0[lane], &out1[lane], &out2[lane], &out3[lane]);
                }

                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    const ptrdiff_t element = evbegin + lane;
                    pack_out[elements[0][element]] += out0[lane];
                    pack_out[elements[1][element]] += out1[lane];
                    pack_out[elements[2][element]] += out2[lane];
                    pack_out[elements[3][element]] += out3[lane];
                }
            }

            for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                u_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                u_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];
                pack_out[k] = scalar_t(0);
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                u_out[ghosts[k] * out_stride] += pack_out[n_contiguous + k];
                pack_out[n_contiguous + k] = scalar_t(0);
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_tet4_jacobian_action_packed_affine_mesh_soa_impl(
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
        const jacobian_t *const SFEM_RESTRICT g_geom_metric0,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric1,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric2,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric3,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric4,
        const jacobian_t *const SFEM_RESTRICT g_geom_metric5,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    if (kappa == scalar_t(1)) {
        return laplace_tet4_jacobian_action_packed_affine_mesh_soa_impl_kernel<scalar_t, jacobian_t, true>(
                n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack,
                elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx,
                g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5,
                kappa, direction_stride, u_direction, out_stride, u_out);
    }
    return laplace_tet4_jacobian_action_packed_affine_mesh_soa_impl_kernel<scalar_t, jacobian_t, false>(
            n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack,
            elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx,
            g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5,
            kappa, direction_stride, u_direction, out_stride, u_out);
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_jacobian_action_packed_affine_mesh_soa(
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
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_jacobian_action_packed_affine_mesh_soa_impl<double, geom_t>(
            n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack,
            elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx,
            g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5,
            kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet4_jacobian_action_packed_affine_mesh_soa_float(
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
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_jacobian_action_packed_affine_mesh_soa_impl<float, geom_t>(
            n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack,
            elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx,
            g_geom_metric0, g_geom_metric1, g_geom_metric2, g_geom_metric3, g_geom_metric4, g_geom_metric5,
            kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_tet4_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
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
        const scalar_t *const direction_components[N_FIELDS] = {u_direction};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 4; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
                const scalar_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
                const scalar_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
                const scalar_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
                const scalar_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
                const scalar_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_direction, kappa, block_output);

        scalar_t *const output_components[N_FIELDS] = {u_out};
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

extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_tet4_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return laplace_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_tet4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return laplace_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

namespace sfem {
namespace codegen {

static SFEM_INLINE void laplace_tet4_hessian_crs_isoparametric_mesh_soa_find_cols(
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

template <typename scalar_t>
static SFEM_INLINE int laplace_tet4_hessian_crs_isoparametric_mesh_soa_scatter_crs(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_SHAPE = 4;
    count_t entries[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    bool valid_graph = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        const count_t row_begin = rowptr[ev[i]];
        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        laplace_tet4_hessian_crs_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {
                if (valid_graph) {
                    std::fprintf(stderr, "laplace_tet4_hessian_crs_isoparametric_mesh_soa_scatter_crs missing graph entry (%ld, %ld)\n", (long)ev[i], (long)ev[j]);
                }
                entries[i * N_SHAPE + j] = row_begin;
                valid_graph = false;
            } else {
                entries[i * N_SHAPE + j] = row_begin + ks[j];
            }
        }
    }
    if (!valid_graph) return SFEM_FAILURE;
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
#pragma omp atomic update
            values[entries[i * N_SHAPE + j]] += element_matrix[i * N_SHAPE + j];
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int laplace_tet4_hessian_crs_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;
    static constexpr int VECTOR_SIZE = 1;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::q_weight();

    int invalid_matrix_graph = 0;
#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const ptrdiff_t evbegin = element;
        const int nelems = 1;
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[16];
        scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_STREAMS][VECTOR_SIZE];
        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t node = elements[shape][element];
            const idx_t coordinate_node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < DIM; ++d) {
                block_coordinates[shape * DIM + d][0] = scalar_t(coordinate_components[d][coordinate_node]);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            const int lane = 0;
            const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
            const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
            const scalar_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
            const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
            const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
            const scalar_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
            const scalar_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
            const scalar_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
            const scalar_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
            geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                    J00, J01, J02, J10, J11, J12, J20, J21, J22,
                    block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
        }
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        for (int entry = 0; entry < 16; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }
        for (int trial_local = 0; trial_local < 4; ++trial_local) {
            const int trial = trial_local;
            for (int stream = 0; stream < N_STREAMS; ++stream) {
                block_direction[stream][0] = scalar_t(0);
                block_output[stream][0] = scalar_t(0);
            }
            block_direction[trial][0] = scalar_t(1);
            laplace_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(1, 1, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_direction, kappa, block_output);
            for (int test_local = 0; test_local < 4; ++test_local) {
                const int test = test_local;
                element_matrix[test_local * 4 + trial_local] = block_output[test][0];
            }
        }

        invalid_matrix_graph |= (laplace_tet4_hessian_crs_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
    }

    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;
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
    return sfem::codegen::laplace_tet4_hessian_crs_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
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
    return sfem::codegen::laplace_tet4_hessian_crs_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
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
    return sfem::codegen::laplace_tet4_hessian_crs_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
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
    return sfem::codegen::laplace_tet4_hessian_crs_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, rowptr, colidx, values);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_tet4_hessian_dia_isoparametric_mesh_soa_scatter_dia(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const ptrdiff_t nnodes,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_SHAPE = 4;
    ptrdiff_t diagonals[N_SHAPE * N_SHAPE];
    bool valid_diagonal_offsets = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        for (int j = 0; j < N_SHAPE; ++j) {
            const int offset = (int)(ev[j] - ev[i]);
            ptrdiff_t diagonal = 0;
            while (diagonal < ndiag && diag_offsets[diagonal] != offset) ++diagonal;
            if (diagonal == ndiag) {
                if (valid_diagonal_offsets) {
                    std::fprintf(stderr, "laplace_tet4_hessian_dia_isoparametric_mesh_soa_scatter_dia missing diagonal offset %d\n", offset);
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
#pragma omp atomic update
            values[diagonal * nnodes + ev[i]] += element_matrix[i * N_SHAPE + j];
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int laplace_tet4_hessian_dia_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 1;
    static constexpr int N_SHAPE = 4;
    static constexpr int VECTOR_SIZE = 1;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::laplace_tet4_isoparametric_reference_data<scalar_t>::q_weight();

    int invalid_matrix_graph = 0;
#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const ptrdiff_t evbegin = element;
        const int nelems = 1;
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[N_SHAPE * N_SHAPE];
        scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_SHAPE][VECTOR_SIZE];
        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t node = elements[shape][element];
            const idx_t coordinate_node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < DIM; ++d) {
                block_coordinates[shape * DIM + d][0] = scalar_t(coordinate_components[d][coordinate_node]);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            const int lane = 0;
            const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
            const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
            const scalar_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
            const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
            const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
            const scalar_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
            const scalar_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 3];
            const scalar_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 3];
            const scalar_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * N_SHAPE + 3];
            geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                    J00, J01, J02, J10, J11, J12, J20, J21, J22,
                    block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
        }
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        for (int entry = 0; entry < N_SHAPE * N_SHAPE; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }
        static constexpr int TENSOR_STREAMS[N_SHAPE] = {0, 1, 2, 3};
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            const int tensor_trial = TENSOR_STREAMS[trial];
            for (int stream = 0; stream < N_SHAPE; ++stream) {
                block_direction[stream][0] = scalar_t(0);
                block_output[stream][0] = scalar_t(0);
            }
            block_direction[tensor_trial][0] = scalar_t(1);
            laplace_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(1, 1, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_direction, kappa, block_output);
            for (int test = 0; test < N_SHAPE; ++test) {
                const int tensor_test = TENSOR_STREAMS[test];
                element_matrix[test * N_SHAPE + trial] = block_output[tensor_test][0];
            }
        }

        invalid_matrix_graph |= (laplace_tet4_hessian_dia_isoparametric_mesh_soa_scatter_dia(ev, element_matrix, nnodes, diag_offsets, ndiag, values) != SFEM_SUCCESS);
    }

    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet4_hessian_dia_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tet4_hessian_dia_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
}

extern "C" int laplace_tet4_hessian_dia_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::laplace_tet4_hessian_dia_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, diag_offsets, ndiag, values);
}
