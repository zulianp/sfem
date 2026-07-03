#include <type_traits>
#include "../laplace_d3_simplex_local.hpp"
#include "../../geometry_kernels.hpp"
#include "../../kernel_diagnostics.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifdef _OPENMP
#include <omp.h>
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
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_residual_element_soa",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_residual_element_soa_float",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_residual_affine_mesh_soa",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
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
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_u_u",
            &sfem::codegen::laplace_tet4_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_u_u_float",
            &sfem::codegen::laplace_tet4_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
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
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_action_element_soa",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_tet4_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_tet4_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_tet4_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
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
                idx_t ev[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                #pragma unroll(4)
                for (int v = 0; v < 4; ++v) {
                    ev[v] = elements[v][i];
                }
                const scalar_t u0 = u[ev[0]];
                const scalar_t u1 = u[ev[1]];
                const scalar_t u2 = u[ev[2]];
                const scalar_t u3 = u[ev[3]];
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
                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
                    #pragma omp atomic update
                    u_out[dof_i] += element_vector[edof_i];
                }
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
        idx_t ev[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        #pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }
        const scalar_t u0 = u[ev[0]];
        const scalar_t u1 = u[ev[1]];
        const scalar_t u2 = u[ev[2]];
        const scalar_t u3 = u[ev[3]];
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
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            #pragma omp atomic update
            u_out[dof_i] += element_vector[edof_i];
        }
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
                idx_t ev[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                #pragma unroll(4)
                for (int v = 0; v < 4; ++v) {
                    ev[v] = elements[v][i];
                }
                const scalar_t u0 = u[ev[0]];
                const scalar_t u1 = u[ev[1]];
                const scalar_t u2 = u[ev[2]];
                const scalar_t u3 = u[ev[3]];
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
                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
                    #pragma omp atomic update
                    u_out[dof_i] += element_vector[edof_i];
                }
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
        idx_t ev[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        #pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }
        const scalar_t u0 = u[ev[0]];
        const scalar_t u1 = u[ev[1]];
        const scalar_t u2 = u[ev[2]];
        const scalar_t u3 = u[ev[3]];
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
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            #pragma omp atomic update
            u_out[dof_i] += element_vector[edof_i];
        }
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

        const scalar_t * block_current_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_current_streams[stream] = block_current[stream];
        }
        scalar_t * block_output_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_output_streams[stream] = block_output[stream];
        }
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_simplex_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_current_streams, kappa, block_output_streams);

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
                idx_t ev[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                #pragma unroll(4)
                for (int v = 0; v < 4; ++v) {
                    ev[v] = elements[v][i];
                }
                const scalar_t u0 = u_direction[ev[0]];
                const scalar_t u1 = u_direction[ev[1]];
                const scalar_t u2 = u_direction[ev[2]];
                const scalar_t u3 = u_direction[ev[3]];
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
                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
                    #pragma omp atomic update
                    u_out[dof_i] += element_vector[edof_i];
                }
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
        idx_t ev[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        #pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }
        const scalar_t u0 = u_direction[ev[0]];
        const scalar_t u1 = u_direction[ev[1]];
        const scalar_t u2 = u_direction[ev[2]];
        const scalar_t u3 = u_direction[ev[3]];
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
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            #pragma omp atomic update
            u_out[dof_i] += element_vector[edof_i];
        }
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
                idx_t ev[4];
                scalar_t fff[6];
                for (int k = 0; k < 6; ++k) {
                    fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
                }
                #pragma unroll(4)
                for (int v = 0; v < 4; ++v) {
                    ev[v] = elements[v][i];
                }
                const scalar_t u0 = u_direction[ev[0]];
                const scalar_t u1 = u_direction[ev[1]];
                const scalar_t u2 = u_direction[ev[2]];
                const scalar_t u3 = u_direction[ev[3]];
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
                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
                    #pragma omp atomic update
                    u_out[dof_i] += element_vector[edof_i];
                }
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
        idx_t ev[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; ++k) {
            fff[k] = scalar_t(g_geom_metric[i * 6 + k]);
        }
        #pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }
        const scalar_t u0 = u_direction[ev[0]];
        const scalar_t u1 = u_direction[ev[1]];
        const scalar_t u2 = u_direction[ev[2]];
        const scalar_t u3 = u_direction[ev[3]];
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
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            #pragma omp atomic update
            u_out[dof_i] += element_vector[edof_i];
        }
    }
    return SFEM_SUCCESS;
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

        const scalar_t * block_direction_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_direction_streams[stream] = block_direction[stream];
        }
        scalar_t * block_output_streams[N_FIELDS * N_SHAPE];
        for (int stream = 0; stream < N_FIELDS * N_SHAPE; ++stream) {
            block_output_streams[stream] = block_output[stream];
        }
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_simplex_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_direction_streams, kappa, block_output_streams);

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
