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
    static constexpr int N_QP = 1;
    static constexpr int VECTOR_SIZE = 16;
    double geom_metric_data[6][N_QP * VECTOR_SIZE];
    const double *const geom_metric[6] = {geom_metric_data[0], geom_metric_data[1], geom_metric_data[2], geom_metric_data[3], geom_metric_data[4], geom_metric_data[5]};
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const double metric_det = determinant[geometry_offset];
            const double metric_adj0 = adjugate[0][geometry_offset];
            const double metric_adj1 = adjugate[1][geometry_offset];
            const double metric_adj2 = adjugate[2][geometry_offset];
            const double metric_adj3 = adjugate[3][geometry_offset];
            const double metric_adj4 = adjugate[4][geometry_offset];
            const double metric_adj5 = adjugate[5][geometry_offset];
            const double metric_adj6 = adjugate[6][geometry_offset];
            const double metric_adj7 = adjugate[7][geometry_offset];
            const double metric_adj8 = adjugate[8][geometry_offset];
            geom_metric_data[0][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj0 + metric_adj1 * metric_adj1 + metric_adj2 * metric_adj2) / metric_det;
            geom_metric_data[1][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj3 + metric_adj1 * metric_adj4 + metric_adj2 * metric_adj5) / metric_det;
            geom_metric_data[3][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj6 + metric_adj1 * metric_adj7 + metric_adj2 * metric_adj8) / metric_det;
            geom_metric_data[2][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj3 + metric_adj4 * metric_adj4 + metric_adj5 * metric_adj5) / metric_det;
            geom_metric_data[4][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj6 + metric_adj4 * metric_adj7 + metric_adj5 * metric_adj8) / metric_det;
            geom_metric_data[5][q * VECTOR_SIZE + lane] = (metric_adj6 * metric_adj6 + metric_adj7 * metric_adj7 + metric_adj8 * metric_adj8) / metric_det;
        }
    }
    sfem::codegen::laplace_d3_simplex_residual_block<double, 1, 4, 16>(nelems, geometry_stride, geom_metric, sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::q_weight(), current, kappa, output);
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
    static constexpr int N_QP = 1;
    static constexpr int VECTOR_SIZE = 16;
    float geom_metric_data[6][N_QP * VECTOR_SIZE];
    const float *const geom_metric[6] = {geom_metric_data[0], geom_metric_data[1], geom_metric_data[2], geom_metric_data[3], geom_metric_data[4], geom_metric_data[5]};
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const float metric_det = determinant[geometry_offset];
            const float metric_adj0 = adjugate[0][geometry_offset];
            const float metric_adj1 = adjugate[1][geometry_offset];
            const float metric_adj2 = adjugate[2][geometry_offset];
            const float metric_adj3 = adjugate[3][geometry_offset];
            const float metric_adj4 = adjugate[4][geometry_offset];
            const float metric_adj5 = adjugate[5][geometry_offset];
            const float metric_adj6 = adjugate[6][geometry_offset];
            const float metric_adj7 = adjugate[7][geometry_offset];
            const float metric_adj8 = adjugate[8][geometry_offset];
            geom_metric_data[0][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj0 + metric_adj1 * metric_adj1 + metric_adj2 * metric_adj2) / metric_det;
            geom_metric_data[1][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj3 + metric_adj1 * metric_adj4 + metric_adj2 * metric_adj5) / metric_det;
            geom_metric_data[3][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj6 + metric_adj1 * metric_adj7 + metric_adj2 * metric_adj8) / metric_det;
            geom_metric_data[2][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj3 + metric_adj4 * metric_adj4 + metric_adj5 * metric_adj5) / metric_det;
            geom_metric_data[4][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj6 + metric_adj4 * metric_adj7 + metric_adj5 * metric_adj8) / metric_det;
            geom_metric_data[5][q * VECTOR_SIZE + lane] = (metric_adj6 * metric_adj6 + metric_adj7 * metric_adj7 + metric_adj8 * metric_adj8) / metric_det;
        }
    }
    sfem::codegen::laplace_d3_simplex_residual_block<float, 1, 4, 16>(nelems, geometry_stride, geom_metric, sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::q_weight(), current, kappa, output);
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
    const scalar_t *const affine_shape = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_current[0][lane] = u[ev[lane * N_SHAPE + 0] * current_stride];
            block_current[1][lane] = u[ev[lane * N_SHAPE + 1] * current_stride];
            block_current[2][lane] = u[ev[lane * N_SHAPE + 2] * current_stride];
            block_current[3][lane] = u[ev[lane * N_SHAPE + 3] * current_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
        }

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3]};
        scalar_t block_geom_metric0_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric0 + evbegin, block_geom_metric0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric1_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric1 + evbegin, block_geom_metric1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric2_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric2 + evbegin, block_geom_metric2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric3_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric3 + evbegin, block_geom_metric3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric4_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric4 + evbegin, block_geom_metric4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric5_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric5 + evbegin, block_geom_metric5_data, std::is_same<jacobian_t, scalar_t>());
        const scalar_t *const block_geom_metric[6] = {block_geom_metric0, block_geom_metric1, block_geom_metric3, block_geom_metric2, block_geom_metric4, block_geom_metric5};
        static const scalar_t cached_affine_metric_q_weight[1] = {scalar_t(1)};

        laplace_d3_simplex_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_geom_metric, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, cached_affine_metric_q_weight, block_current_streams, kappa, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 3] * out_stride] += block_output[3][scatter];
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
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_geom_metric_data[6][N_QP * VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_coordinates[0][lane] = points[0][ev[lane * N_SHAPE + 0]];
            block_coordinates[1][lane] = points[1][ev[lane * N_SHAPE + 0]];
            block_coordinates[2][lane] = points[2][ev[lane * N_SHAPE + 0]];
            block_current[0][lane] = u[ev[lane * N_SHAPE + 0] * current_stride];
            block_coordinates[3][lane] = points[0][ev[lane * N_SHAPE + 1]];
            block_coordinates[4][lane] = points[1][ev[lane * N_SHAPE + 1]];
            block_coordinates[5][lane] = points[2][ev[lane * N_SHAPE + 1]];
            block_current[1][lane] = u[ev[lane * N_SHAPE + 1] * current_stride];
            block_coordinates[6][lane] = points[0][ev[lane * N_SHAPE + 2]];
            block_coordinates[7][lane] = points[1][ev[lane * N_SHAPE + 2]];
            block_coordinates[8][lane] = points[2][ev[lane * N_SHAPE + 2]];
            block_current[2][lane] = u[ev[lane * N_SHAPE + 2] * current_stride];
            block_coordinates[9][lane] = points[0][ev[lane * N_SHAPE + 3]];
            block_coordinates[10][lane] = points[1][ev[lane * N_SHAPE + 3]];
            block_coordinates[11][lane] = points[2][ev[lane * N_SHAPE + 3]];
            block_current[3][lane] = u[ev[lane * N_SHAPE + 3] * current_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
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

        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t geometry_offset = q * VECTOR_SIZE + lane;
                const scalar_t metric_det = block_determinant[geometry_offset];
                const scalar_t metric_adj0 = block_adjugate_data[0][geometry_offset];
                const scalar_t metric_adj1 = block_adjugate_data[1][geometry_offset];
                const scalar_t metric_adj2 = block_adjugate_data[2][geometry_offset];
                const scalar_t metric_adj3 = block_adjugate_data[3][geometry_offset];
                const scalar_t metric_adj4 = block_adjugate_data[4][geometry_offset];
                const scalar_t metric_adj5 = block_adjugate_data[5][geometry_offset];
                const scalar_t metric_adj6 = block_adjugate_data[6][geometry_offset];
                const scalar_t metric_adj7 = block_adjugate_data[7][geometry_offset];
                const scalar_t metric_adj8 = block_adjugate_data[8][geometry_offset];
                block_geom_metric_data[0][geometry_offset] = (metric_adj0 * metric_adj0 + metric_adj1 * metric_adj1 + metric_adj2 * metric_adj2) / metric_det;
                block_geom_metric_data[1][geometry_offset] = (metric_adj0 * metric_adj3 + metric_adj1 * metric_adj4 + metric_adj2 * metric_adj5) / metric_det;
                block_geom_metric_data[3][geometry_offset] = (metric_adj0 * metric_adj6 + metric_adj1 * metric_adj7 + metric_adj2 * metric_adj8) / metric_det;
                block_geom_metric_data[2][geometry_offset] = (metric_adj3 * metric_adj3 + metric_adj4 * metric_adj4 + metric_adj5 * metric_adj5) / metric_det;
                block_geom_metric_data[4][geometry_offset] = (metric_adj3 * metric_adj6 + metric_adj4 * metric_adj7 + metric_adj5 * metric_adj8) / metric_det;
                block_geom_metric_data[5][geometry_offset] = (metric_adj6 * metric_adj6 + metric_adj7 * metric_adj7 + metric_adj8 * metric_adj8) / metric_det;
            }
        }

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3]};
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        const scalar_t *const block_geom_metric[6] = {block_geom_metric_data[0], block_geom_metric_data[1], block_geom_metric_data[2], block_geom_metric_data[3], block_geom_metric_data[4], block_geom_metric_data[5]};

        laplace_d3_simplex_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_geom_metric, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_current_streams, kappa, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 3] * out_stride] += block_output[3][scatter];
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
    static constexpr int N_QP = 1;
    static constexpr int VECTOR_SIZE = 16;
    double geom_metric_data[6][N_QP * VECTOR_SIZE];
    const double *const geom_metric[6] = {geom_metric_data[0], geom_metric_data[1], geom_metric_data[2], geom_metric_data[3], geom_metric_data[4], geom_metric_data[5]};
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const double metric_det = determinant[geometry_offset];
            const double metric_adj0 = adjugate[0][geometry_offset];
            const double metric_adj1 = adjugate[1][geometry_offset];
            const double metric_adj2 = adjugate[2][geometry_offset];
            const double metric_adj3 = adjugate[3][geometry_offset];
            const double metric_adj4 = adjugate[4][geometry_offset];
            const double metric_adj5 = adjugate[5][geometry_offset];
            const double metric_adj6 = adjugate[6][geometry_offset];
            const double metric_adj7 = adjugate[7][geometry_offset];
            const double metric_adj8 = adjugate[8][geometry_offset];
            geom_metric_data[0][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj0 + metric_adj1 * metric_adj1 + metric_adj2 * metric_adj2) / metric_det;
            geom_metric_data[1][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj3 + metric_adj1 * metric_adj4 + metric_adj2 * metric_adj5) / metric_det;
            geom_metric_data[3][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj6 + metric_adj1 * metric_adj7 + metric_adj2 * metric_adj8) / metric_det;
            geom_metric_data[2][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj3 + metric_adj4 * metric_adj4 + metric_adj5 * metric_adj5) / metric_det;
            geom_metric_data[4][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj6 + metric_adj4 * metric_adj7 + metric_adj5 * metric_adj8) / metric_det;
            geom_metric_data[5][q * VECTOR_SIZE + lane] = (metric_adj6 * metric_adj6 + metric_adj7 * metric_adj7 + metric_adj8 * metric_adj8) / metric_det;
        }
    }
    sfem::codegen::laplace_d3_simplex_jacobian_action_block<double, 1, 4, 16>(nelems, geometry_stride, geom_metric, sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<double>::q_weight(), direction, kappa, output);
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
    static constexpr int N_QP = 1;
    static constexpr int VECTOR_SIZE = 16;
    float geom_metric_data[6][N_QP * VECTOR_SIZE];
    const float *const geom_metric[6] = {geom_metric_data[0], geom_metric_data[1], geom_metric_data[2], geom_metric_data[3], geom_metric_data[4], geom_metric_data[5]};
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const float metric_det = determinant[geometry_offset];
            const float metric_adj0 = adjugate[0][geometry_offset];
            const float metric_adj1 = adjugate[1][geometry_offset];
            const float metric_adj2 = adjugate[2][geometry_offset];
            const float metric_adj3 = adjugate[3][geometry_offset];
            const float metric_adj4 = adjugate[4][geometry_offset];
            const float metric_adj5 = adjugate[5][geometry_offset];
            const float metric_adj6 = adjugate[6][geometry_offset];
            const float metric_adj7 = adjugate[7][geometry_offset];
            const float metric_adj8 = adjugate[8][geometry_offset];
            geom_metric_data[0][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj0 + metric_adj1 * metric_adj1 + metric_adj2 * metric_adj2) / metric_det;
            geom_metric_data[1][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj3 + metric_adj1 * metric_adj4 + metric_adj2 * metric_adj5) / metric_det;
            geom_metric_data[3][q * VECTOR_SIZE + lane] = (metric_adj0 * metric_adj6 + metric_adj1 * metric_adj7 + metric_adj2 * metric_adj8) / metric_det;
            geom_metric_data[2][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj3 + metric_adj4 * metric_adj4 + metric_adj5 * metric_adj5) / metric_det;
            geom_metric_data[4][q * VECTOR_SIZE + lane] = (metric_adj3 * metric_adj6 + metric_adj4 * metric_adj7 + metric_adj5 * metric_adj8) / metric_det;
            geom_metric_data[5][q * VECTOR_SIZE + lane] = (metric_adj6 * metric_adj6 + metric_adj7 * metric_adj7 + metric_adj8 * metric_adj8) / metric_det;
        }
    }
    sfem::codegen::laplace_d3_simplex_jacobian_action_block<float, 1, 4, 16>(nelems, geometry_stride, geom_metric, sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::laplace_tet4_isoparametric_reference_data<float>::q_weight(), direction, kappa, output);
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
    const scalar_t *const affine_shape = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::laplace_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_direction[0][lane] = u_direction[ev[lane * N_SHAPE + 0] * direction_stride];
            block_direction[1][lane] = u_direction[ev[lane * N_SHAPE + 1] * direction_stride];
            block_direction[2][lane] = u_direction[ev[lane * N_SHAPE + 2] * direction_stride];
            block_direction[3][lane] = u_direction[ev[lane * N_SHAPE + 3] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
        }

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3]};
        scalar_t block_geom_metric0_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric0 + evbegin, block_geom_metric0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric1_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric1 + evbegin, block_geom_metric1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric2_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric2 + evbegin, block_geom_metric2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric3_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric3 + evbegin, block_geom_metric3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric4_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric4 + evbegin, block_geom_metric4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_geom_metric5_data[VECTOR_SIZE];
        const scalar_t *const block_geom_metric5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_geom_metric5 + evbegin, block_geom_metric5_data, std::is_same<jacobian_t, scalar_t>());
        const scalar_t *const block_geom_metric[6] = {block_geom_metric0, block_geom_metric1, block_geom_metric3, block_geom_metric2, block_geom_metric4, block_geom_metric5};
        static const scalar_t cached_affine_metric_q_weight[1] = {scalar_t(1)};

        laplace_d3_simplex_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_geom_metric, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, cached_affine_metric_q_weight, block_direction_streams, kappa, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 3] * out_stride] += block_output[3][scatter];
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
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_geom_metric_data[6][N_QP * VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_coordinates[0][lane] = points[0][ev[lane * N_SHAPE + 0]];
            block_coordinates[1][lane] = points[1][ev[lane * N_SHAPE + 0]];
            block_coordinates[2][lane] = points[2][ev[lane * N_SHAPE + 0]];
            block_direction[0][lane] = u_direction[ev[lane * N_SHAPE + 0] * direction_stride];
            block_coordinates[3][lane] = points[0][ev[lane * N_SHAPE + 1]];
            block_coordinates[4][lane] = points[1][ev[lane * N_SHAPE + 1]];
            block_coordinates[5][lane] = points[2][ev[lane * N_SHAPE + 1]];
            block_direction[1][lane] = u_direction[ev[lane * N_SHAPE + 1] * direction_stride];
            block_coordinates[6][lane] = points[0][ev[lane * N_SHAPE + 2]];
            block_coordinates[7][lane] = points[1][ev[lane * N_SHAPE + 2]];
            block_coordinates[8][lane] = points[2][ev[lane * N_SHAPE + 2]];
            block_direction[2][lane] = u_direction[ev[lane * N_SHAPE + 2] * direction_stride];
            block_coordinates[9][lane] = points[0][ev[lane * N_SHAPE + 3]];
            block_coordinates[10][lane] = points[1][ev[lane * N_SHAPE + 3]];
            block_coordinates[11][lane] = points[2][ev[lane * N_SHAPE + 3]];
            block_direction[3][lane] = u_direction[ev[lane * N_SHAPE + 3] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
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

        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t geometry_offset = q * VECTOR_SIZE + lane;
                const scalar_t metric_det = block_determinant[geometry_offset];
                const scalar_t metric_adj0 = block_adjugate_data[0][geometry_offset];
                const scalar_t metric_adj1 = block_adjugate_data[1][geometry_offset];
                const scalar_t metric_adj2 = block_adjugate_data[2][geometry_offset];
                const scalar_t metric_adj3 = block_adjugate_data[3][geometry_offset];
                const scalar_t metric_adj4 = block_adjugate_data[4][geometry_offset];
                const scalar_t metric_adj5 = block_adjugate_data[5][geometry_offset];
                const scalar_t metric_adj6 = block_adjugate_data[6][geometry_offset];
                const scalar_t metric_adj7 = block_adjugate_data[7][geometry_offset];
                const scalar_t metric_adj8 = block_adjugate_data[8][geometry_offset];
                block_geom_metric_data[0][geometry_offset] = (metric_adj0 * metric_adj0 + metric_adj1 * metric_adj1 + metric_adj2 * metric_adj2) / metric_det;
                block_geom_metric_data[1][geometry_offset] = (metric_adj0 * metric_adj3 + metric_adj1 * metric_adj4 + metric_adj2 * metric_adj5) / metric_det;
                block_geom_metric_data[3][geometry_offset] = (metric_adj0 * metric_adj6 + metric_adj1 * metric_adj7 + metric_adj2 * metric_adj8) / metric_det;
                block_geom_metric_data[2][geometry_offset] = (metric_adj3 * metric_adj3 + metric_adj4 * metric_adj4 + metric_adj5 * metric_adj5) / metric_det;
                block_geom_metric_data[4][geometry_offset] = (metric_adj3 * metric_adj6 + metric_adj4 * metric_adj7 + metric_adj5 * metric_adj8) / metric_det;
                block_geom_metric_data[5][geometry_offset] = (metric_adj6 * metric_adj6 + metric_adj7 * metric_adj7 + metric_adj8 * metric_adj8) / metric_det;
            }
        }

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3]};
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        const scalar_t *const block_geom_metric[6] = {block_geom_metric_data[0], block_geom_metric_data[1], block_geom_metric_data[2], block_geom_metric_data[3], block_geom_metric_data[4], block_geom_metric_data[5]};

        laplace_d3_simplex_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_geom_metric, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_direction_streams, kappa, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 3] * out_stride] += block_output[3][scatter];
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
