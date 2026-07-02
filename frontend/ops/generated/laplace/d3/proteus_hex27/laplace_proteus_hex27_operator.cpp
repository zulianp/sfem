#include <type_traits>
#include "../laplace_d3_tensor_product_local.hpp"
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
struct laplace_proteus_hex27_affine_reference_data {
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
struct laplace_proteus_hex27_isoparametric_reference_data {
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

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex27_residual_element_soa_diagnostics_data = {
    "laplace_proteus_hex27_residual_element_soa",
    "PROTEUS_HEX27",
    3,
    27,
    27,
    16,
    3,
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
    18,
    3,
    1,
    27,
    0,
    27,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex27_residual_element_soa",
            &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex27_residual_element_soa_float",
            &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex27_residual_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex27_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex27_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex27_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex27_jacobian_u_u_diagnostics_data = {
    "laplace_proteus_hex27_jacobian_u_u",
    "PROTEUS_HEX27",
    3,
    27,
    27,
    16,
    3,
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
    18,
    3,
    1,
    0,
    27,
    27,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex27_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_proteus_hex27_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex27_jacobian_u_u_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex27_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex27_jacobian_u_u",
            &sfem::codegen::laplace_proteus_hex27_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex27_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex27_jacobian_u_u_float",
            &sfem::codegen::laplace_proteus_hex27_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data = {
    "laplace_proteus_hex27_jacobian_action_element_soa",
    "PROTEUS_HEX27",
    3,
    27,
    27,
    16,
    3,
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
    18,
    3,
    1,
    0,
    27,
    27,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex27_jacobian_action_element_soa",
            &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex27_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex27_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int laplace_proteus_hex27_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[27],
        const double kappa,
        double *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<double, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<double>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex27_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[27],
        const float kappa,
        float *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<float, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<float>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex27_residual_affine_mesh_soa_impl(
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
        const scalar_t kappa,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();

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
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_current[0][lane] = u[ev[lane * N_SHAPE + 0] * current_stride];
            block_current[1][lane] = u[ev[lane * N_SHAPE + 1] * current_stride];
            block_current[2][lane] = u[ev[lane * N_SHAPE + 2] * current_stride];
            block_current[3][lane] = u[ev[lane * N_SHAPE + 3] * current_stride];
            block_current[4][lane] = u[ev[lane * N_SHAPE + 4] * current_stride];
            block_current[5][lane] = u[ev[lane * N_SHAPE + 5] * current_stride];
            block_current[6][lane] = u[ev[lane * N_SHAPE + 6] * current_stride];
            block_current[7][lane] = u[ev[lane * N_SHAPE + 7] * current_stride];
            block_current[8][lane] = u[ev[lane * N_SHAPE + 8] * current_stride];
            block_current[9][lane] = u[ev[lane * N_SHAPE + 9] * current_stride];
            block_current[10][lane] = u[ev[lane * N_SHAPE + 10] * current_stride];
            block_current[11][lane] = u[ev[lane * N_SHAPE + 11] * current_stride];
            block_current[12][lane] = u[ev[lane * N_SHAPE + 12] * current_stride];
            block_current[13][lane] = u[ev[lane * N_SHAPE + 13] * current_stride];
            block_current[14][lane] = u[ev[lane * N_SHAPE + 14] * current_stride];
            block_current[15][lane] = u[ev[lane * N_SHAPE + 15] * current_stride];
            block_current[16][lane] = u[ev[lane * N_SHAPE + 16] * current_stride];
            block_current[17][lane] = u[ev[lane * N_SHAPE + 17] * current_stride];
            block_current[18][lane] = u[ev[lane * N_SHAPE + 18] * current_stride];
            block_current[19][lane] = u[ev[lane * N_SHAPE + 19] * current_stride];
            block_current[20][lane] = u[ev[lane * N_SHAPE + 20] * current_stride];
            block_current[21][lane] = u[ev[lane * N_SHAPE + 21] * current_stride];
            block_current[22][lane] = u[ev[lane * N_SHAPE + 22] * current_stride];
            block_current[23][lane] = u[ev[lane * N_SHAPE + 23] * current_stride];
            block_current[24][lane] = u[ev[lane * N_SHAPE + 24] * current_stride];
            block_current[25][lane] = u[ev[lane * N_SHAPE + 25] * current_stride];
            block_current[26][lane] = u[ev[lane * N_SHAPE + 26] * current_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
        }

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26]};
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());
        const scalar_t *const block_adjugate[9] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

        laplace_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current_streams, kappa, block_output_streams);

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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 10] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 11] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 12] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 13] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 14] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 15] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 16] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 17] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 18] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 19] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 20] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 21] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 22] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 23] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 24] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 25] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 26] * out_stride] += block_output[26][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex27_residual_affine_mesh_soa(
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
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_proteus_hex27_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex27_residual_affine_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_proteus_hex27_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex27_residual_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
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
            block_coordinates[12][lane] = points[0][ev[lane * N_SHAPE + 4]];
            block_coordinates[13][lane] = points[1][ev[lane * N_SHAPE + 4]];
            block_coordinates[14][lane] = points[2][ev[lane * N_SHAPE + 4]];
            block_current[4][lane] = u[ev[lane * N_SHAPE + 4] * current_stride];
            block_coordinates[15][lane] = points[0][ev[lane * N_SHAPE + 5]];
            block_coordinates[16][lane] = points[1][ev[lane * N_SHAPE + 5]];
            block_coordinates[17][lane] = points[2][ev[lane * N_SHAPE + 5]];
            block_current[5][lane] = u[ev[lane * N_SHAPE + 5] * current_stride];
            block_coordinates[18][lane] = points[0][ev[lane * N_SHAPE + 6]];
            block_coordinates[19][lane] = points[1][ev[lane * N_SHAPE + 6]];
            block_coordinates[20][lane] = points[2][ev[lane * N_SHAPE + 6]];
            block_current[6][lane] = u[ev[lane * N_SHAPE + 6] * current_stride];
            block_coordinates[21][lane] = points[0][ev[lane * N_SHAPE + 7]];
            block_coordinates[22][lane] = points[1][ev[lane * N_SHAPE + 7]];
            block_coordinates[23][lane] = points[2][ev[lane * N_SHAPE + 7]];
            block_current[7][lane] = u[ev[lane * N_SHAPE + 7] * current_stride];
            block_coordinates[24][lane] = points[0][ev[lane * N_SHAPE + 8]];
            block_coordinates[25][lane] = points[1][ev[lane * N_SHAPE + 8]];
            block_coordinates[26][lane] = points[2][ev[lane * N_SHAPE + 8]];
            block_current[8][lane] = u[ev[lane * N_SHAPE + 8] * current_stride];
            block_coordinates[27][lane] = points[0][ev[lane * N_SHAPE + 9]];
            block_coordinates[28][lane] = points[1][ev[lane * N_SHAPE + 9]];
            block_coordinates[29][lane] = points[2][ev[lane * N_SHAPE + 9]];
            block_current[9][lane] = u[ev[lane * N_SHAPE + 9] * current_stride];
            block_coordinates[30][lane] = points[0][ev[lane * N_SHAPE + 10]];
            block_coordinates[31][lane] = points[1][ev[lane * N_SHAPE + 10]];
            block_coordinates[32][lane] = points[2][ev[lane * N_SHAPE + 10]];
            block_current[10][lane] = u[ev[lane * N_SHAPE + 10] * current_stride];
            block_coordinates[33][lane] = points[0][ev[lane * N_SHAPE + 11]];
            block_coordinates[34][lane] = points[1][ev[lane * N_SHAPE + 11]];
            block_coordinates[35][lane] = points[2][ev[lane * N_SHAPE + 11]];
            block_current[11][lane] = u[ev[lane * N_SHAPE + 11] * current_stride];
            block_coordinates[36][lane] = points[0][ev[lane * N_SHAPE + 12]];
            block_coordinates[37][lane] = points[1][ev[lane * N_SHAPE + 12]];
            block_coordinates[38][lane] = points[2][ev[lane * N_SHAPE + 12]];
            block_current[12][lane] = u[ev[lane * N_SHAPE + 12] * current_stride];
            block_coordinates[39][lane] = points[0][ev[lane * N_SHAPE + 13]];
            block_coordinates[40][lane] = points[1][ev[lane * N_SHAPE + 13]];
            block_coordinates[41][lane] = points[2][ev[lane * N_SHAPE + 13]];
            block_current[13][lane] = u[ev[lane * N_SHAPE + 13] * current_stride];
            block_coordinates[42][lane] = points[0][ev[lane * N_SHAPE + 14]];
            block_coordinates[43][lane] = points[1][ev[lane * N_SHAPE + 14]];
            block_coordinates[44][lane] = points[2][ev[lane * N_SHAPE + 14]];
            block_current[14][lane] = u[ev[lane * N_SHAPE + 14] * current_stride];
            block_coordinates[45][lane] = points[0][ev[lane * N_SHAPE + 15]];
            block_coordinates[46][lane] = points[1][ev[lane * N_SHAPE + 15]];
            block_coordinates[47][lane] = points[2][ev[lane * N_SHAPE + 15]];
            block_current[15][lane] = u[ev[lane * N_SHAPE + 15] * current_stride];
            block_coordinates[48][lane] = points[0][ev[lane * N_SHAPE + 16]];
            block_coordinates[49][lane] = points[1][ev[lane * N_SHAPE + 16]];
            block_coordinates[50][lane] = points[2][ev[lane * N_SHAPE + 16]];
            block_current[16][lane] = u[ev[lane * N_SHAPE + 16] * current_stride];
            block_coordinates[51][lane] = points[0][ev[lane * N_SHAPE + 17]];
            block_coordinates[52][lane] = points[1][ev[lane * N_SHAPE + 17]];
            block_coordinates[53][lane] = points[2][ev[lane * N_SHAPE + 17]];
            block_current[17][lane] = u[ev[lane * N_SHAPE + 17] * current_stride];
            block_coordinates[54][lane] = points[0][ev[lane * N_SHAPE + 18]];
            block_coordinates[55][lane] = points[1][ev[lane * N_SHAPE + 18]];
            block_coordinates[56][lane] = points[2][ev[lane * N_SHAPE + 18]];
            block_current[18][lane] = u[ev[lane * N_SHAPE + 18] * current_stride];
            block_coordinates[57][lane] = points[0][ev[lane * N_SHAPE + 19]];
            block_coordinates[58][lane] = points[1][ev[lane * N_SHAPE + 19]];
            block_coordinates[59][lane] = points[2][ev[lane * N_SHAPE + 19]];
            block_current[19][lane] = u[ev[lane * N_SHAPE + 19] * current_stride];
            block_coordinates[60][lane] = points[0][ev[lane * N_SHAPE + 20]];
            block_coordinates[61][lane] = points[1][ev[lane * N_SHAPE + 20]];
            block_coordinates[62][lane] = points[2][ev[lane * N_SHAPE + 20]];
            block_current[20][lane] = u[ev[lane * N_SHAPE + 20] * current_stride];
            block_coordinates[63][lane] = points[0][ev[lane * N_SHAPE + 21]];
            block_coordinates[64][lane] = points[1][ev[lane * N_SHAPE + 21]];
            block_coordinates[65][lane] = points[2][ev[lane * N_SHAPE + 21]];
            block_current[21][lane] = u[ev[lane * N_SHAPE + 21] * current_stride];
            block_coordinates[66][lane] = points[0][ev[lane * N_SHAPE + 22]];
            block_coordinates[67][lane] = points[1][ev[lane * N_SHAPE + 22]];
            block_coordinates[68][lane] = points[2][ev[lane * N_SHAPE + 22]];
            block_current[22][lane] = u[ev[lane * N_SHAPE + 22] * current_stride];
            block_coordinates[69][lane] = points[0][ev[lane * N_SHAPE + 23]];
            block_coordinates[70][lane] = points[1][ev[lane * N_SHAPE + 23]];
            block_coordinates[71][lane] = points[2][ev[lane * N_SHAPE + 23]];
            block_current[23][lane] = u[ev[lane * N_SHAPE + 23] * current_stride];
            block_coordinates[72][lane] = points[0][ev[lane * N_SHAPE + 24]];
            block_coordinates[73][lane] = points[1][ev[lane * N_SHAPE + 24]];
            block_coordinates[74][lane] = points[2][ev[lane * N_SHAPE + 24]];
            block_current[24][lane] = u[ev[lane * N_SHAPE + 24] * current_stride];
            block_coordinates[75][lane] = points[0][ev[lane * N_SHAPE + 25]];
            block_coordinates[76][lane] = points[1][ev[lane * N_SHAPE + 25]];
            block_coordinates[77][lane] = points[2][ev[lane * N_SHAPE + 25]];
            block_current[25][lane] = u[ev[lane * N_SHAPE + 25] * current_stride];
            block_coordinates[78][lane] = points[0][ev[lane * N_SHAPE + 26]];
            block_coordinates[79][lane] = points[1][ev[lane * N_SHAPE + 26]];
            block_coordinates[80][lane] = points[2][ev[lane * N_SHAPE + 26]];
            block_current[26][lane] = u[ev[lane * N_SHAPE + 26] * current_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26]};
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current_streams, kappa, block_output_streams);

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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 10] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 11] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 12] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 13] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 14] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 15] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 16] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 17] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 18] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 19] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 20] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 21] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 22] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 23] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 24] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 25] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 26] * out_stride] += block_output[26][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex27_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex27_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex27_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT direction[27],
        const double kappa,
        double *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<double, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<double>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex27_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT direction[27],
        const float kappa,
        float *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<float, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<float>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex27_jacobian_action_affine_mesh_soa_impl(
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
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex27_affine_reference_data<scalar_t>::q_weight_1d();

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
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_direction[0][lane] = u_direction[ev[lane * N_SHAPE + 0] * direction_stride];
            block_direction[1][lane] = u_direction[ev[lane * N_SHAPE + 1] * direction_stride];
            block_direction[2][lane] = u_direction[ev[lane * N_SHAPE + 2] * direction_stride];
            block_direction[3][lane] = u_direction[ev[lane * N_SHAPE + 3] * direction_stride];
            block_direction[4][lane] = u_direction[ev[lane * N_SHAPE + 4] * direction_stride];
            block_direction[5][lane] = u_direction[ev[lane * N_SHAPE + 5] * direction_stride];
            block_direction[6][lane] = u_direction[ev[lane * N_SHAPE + 6] * direction_stride];
            block_direction[7][lane] = u_direction[ev[lane * N_SHAPE + 7] * direction_stride];
            block_direction[8][lane] = u_direction[ev[lane * N_SHAPE + 8] * direction_stride];
            block_direction[9][lane] = u_direction[ev[lane * N_SHAPE + 9] * direction_stride];
            block_direction[10][lane] = u_direction[ev[lane * N_SHAPE + 10] * direction_stride];
            block_direction[11][lane] = u_direction[ev[lane * N_SHAPE + 11] * direction_stride];
            block_direction[12][lane] = u_direction[ev[lane * N_SHAPE + 12] * direction_stride];
            block_direction[13][lane] = u_direction[ev[lane * N_SHAPE + 13] * direction_stride];
            block_direction[14][lane] = u_direction[ev[lane * N_SHAPE + 14] * direction_stride];
            block_direction[15][lane] = u_direction[ev[lane * N_SHAPE + 15] * direction_stride];
            block_direction[16][lane] = u_direction[ev[lane * N_SHAPE + 16] * direction_stride];
            block_direction[17][lane] = u_direction[ev[lane * N_SHAPE + 17] * direction_stride];
            block_direction[18][lane] = u_direction[ev[lane * N_SHAPE + 18] * direction_stride];
            block_direction[19][lane] = u_direction[ev[lane * N_SHAPE + 19] * direction_stride];
            block_direction[20][lane] = u_direction[ev[lane * N_SHAPE + 20] * direction_stride];
            block_direction[21][lane] = u_direction[ev[lane * N_SHAPE + 21] * direction_stride];
            block_direction[22][lane] = u_direction[ev[lane * N_SHAPE + 22] * direction_stride];
            block_direction[23][lane] = u_direction[ev[lane * N_SHAPE + 23] * direction_stride];
            block_direction[24][lane] = u_direction[ev[lane * N_SHAPE + 24] * direction_stride];
            block_direction[25][lane] = u_direction[ev[lane * N_SHAPE + 25] * direction_stride];
            block_direction[26][lane] = u_direction[ev[lane * N_SHAPE + 26] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
        }

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26]};
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());
        const scalar_t *const block_adjugate[9] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

        laplace_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_direction_streams, kappa, block_output_streams);

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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 10] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 11] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 12] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 13] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 14] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 15] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 16] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 17] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 18] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 19] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 20] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 21] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 22] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 23] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 24] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 25] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 26] * out_stride] += block_output[26][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex27_jacobian_action_affine_mesh_soa(
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
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_proteus_hex27_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_proteus_hex27_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * N_SHAPE + 9] = elements[9][evbegin + lane];
            ev[lane * N_SHAPE + 10] = elements[10][evbegin + lane];
            ev[lane * N_SHAPE + 11] = elements[11][evbegin + lane];
            ev[lane * N_SHAPE + 12] = elements[12][evbegin + lane];
            ev[lane * N_SHAPE + 13] = elements[13][evbegin + lane];
            ev[lane * N_SHAPE + 14] = elements[14][evbegin + lane];
            ev[lane * N_SHAPE + 15] = elements[15][evbegin + lane];
            ev[lane * N_SHAPE + 16] = elements[16][evbegin + lane];
            ev[lane * N_SHAPE + 17] = elements[17][evbegin + lane];
            ev[lane * N_SHAPE + 18] = elements[18][evbegin + lane];
            ev[lane * N_SHAPE + 19] = elements[19][evbegin + lane];
            ev[lane * N_SHAPE + 20] = elements[20][evbegin + lane];
            ev[lane * N_SHAPE + 21] = elements[21][evbegin + lane];
            ev[lane * N_SHAPE + 22] = elements[22][evbegin + lane];
            ev[lane * N_SHAPE + 23] = elements[23][evbegin + lane];
            ev[lane * N_SHAPE + 24] = elements[24][evbegin + lane];
            ev[lane * N_SHAPE + 25] = elements[25][evbegin + lane];
            ev[lane * N_SHAPE + 26] = elements[26][evbegin + lane];
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
            block_coordinates[12][lane] = points[0][ev[lane * N_SHAPE + 4]];
            block_coordinates[13][lane] = points[1][ev[lane * N_SHAPE + 4]];
            block_coordinates[14][lane] = points[2][ev[lane * N_SHAPE + 4]];
            block_direction[4][lane] = u_direction[ev[lane * N_SHAPE + 4] * direction_stride];
            block_coordinates[15][lane] = points[0][ev[lane * N_SHAPE + 5]];
            block_coordinates[16][lane] = points[1][ev[lane * N_SHAPE + 5]];
            block_coordinates[17][lane] = points[2][ev[lane * N_SHAPE + 5]];
            block_direction[5][lane] = u_direction[ev[lane * N_SHAPE + 5] * direction_stride];
            block_coordinates[18][lane] = points[0][ev[lane * N_SHAPE + 6]];
            block_coordinates[19][lane] = points[1][ev[lane * N_SHAPE + 6]];
            block_coordinates[20][lane] = points[2][ev[lane * N_SHAPE + 6]];
            block_direction[6][lane] = u_direction[ev[lane * N_SHAPE + 6] * direction_stride];
            block_coordinates[21][lane] = points[0][ev[lane * N_SHAPE + 7]];
            block_coordinates[22][lane] = points[1][ev[lane * N_SHAPE + 7]];
            block_coordinates[23][lane] = points[2][ev[lane * N_SHAPE + 7]];
            block_direction[7][lane] = u_direction[ev[lane * N_SHAPE + 7] * direction_stride];
            block_coordinates[24][lane] = points[0][ev[lane * N_SHAPE + 8]];
            block_coordinates[25][lane] = points[1][ev[lane * N_SHAPE + 8]];
            block_coordinates[26][lane] = points[2][ev[lane * N_SHAPE + 8]];
            block_direction[8][lane] = u_direction[ev[lane * N_SHAPE + 8] * direction_stride];
            block_coordinates[27][lane] = points[0][ev[lane * N_SHAPE + 9]];
            block_coordinates[28][lane] = points[1][ev[lane * N_SHAPE + 9]];
            block_coordinates[29][lane] = points[2][ev[lane * N_SHAPE + 9]];
            block_direction[9][lane] = u_direction[ev[lane * N_SHAPE + 9] * direction_stride];
            block_coordinates[30][lane] = points[0][ev[lane * N_SHAPE + 10]];
            block_coordinates[31][lane] = points[1][ev[lane * N_SHAPE + 10]];
            block_coordinates[32][lane] = points[2][ev[lane * N_SHAPE + 10]];
            block_direction[10][lane] = u_direction[ev[lane * N_SHAPE + 10] * direction_stride];
            block_coordinates[33][lane] = points[0][ev[lane * N_SHAPE + 11]];
            block_coordinates[34][lane] = points[1][ev[lane * N_SHAPE + 11]];
            block_coordinates[35][lane] = points[2][ev[lane * N_SHAPE + 11]];
            block_direction[11][lane] = u_direction[ev[lane * N_SHAPE + 11] * direction_stride];
            block_coordinates[36][lane] = points[0][ev[lane * N_SHAPE + 12]];
            block_coordinates[37][lane] = points[1][ev[lane * N_SHAPE + 12]];
            block_coordinates[38][lane] = points[2][ev[lane * N_SHAPE + 12]];
            block_direction[12][lane] = u_direction[ev[lane * N_SHAPE + 12] * direction_stride];
            block_coordinates[39][lane] = points[0][ev[lane * N_SHAPE + 13]];
            block_coordinates[40][lane] = points[1][ev[lane * N_SHAPE + 13]];
            block_coordinates[41][lane] = points[2][ev[lane * N_SHAPE + 13]];
            block_direction[13][lane] = u_direction[ev[lane * N_SHAPE + 13] * direction_stride];
            block_coordinates[42][lane] = points[0][ev[lane * N_SHAPE + 14]];
            block_coordinates[43][lane] = points[1][ev[lane * N_SHAPE + 14]];
            block_coordinates[44][lane] = points[2][ev[lane * N_SHAPE + 14]];
            block_direction[14][lane] = u_direction[ev[lane * N_SHAPE + 14] * direction_stride];
            block_coordinates[45][lane] = points[0][ev[lane * N_SHAPE + 15]];
            block_coordinates[46][lane] = points[1][ev[lane * N_SHAPE + 15]];
            block_coordinates[47][lane] = points[2][ev[lane * N_SHAPE + 15]];
            block_direction[15][lane] = u_direction[ev[lane * N_SHAPE + 15] * direction_stride];
            block_coordinates[48][lane] = points[0][ev[lane * N_SHAPE + 16]];
            block_coordinates[49][lane] = points[1][ev[lane * N_SHAPE + 16]];
            block_coordinates[50][lane] = points[2][ev[lane * N_SHAPE + 16]];
            block_direction[16][lane] = u_direction[ev[lane * N_SHAPE + 16] * direction_stride];
            block_coordinates[51][lane] = points[0][ev[lane * N_SHAPE + 17]];
            block_coordinates[52][lane] = points[1][ev[lane * N_SHAPE + 17]];
            block_coordinates[53][lane] = points[2][ev[lane * N_SHAPE + 17]];
            block_direction[17][lane] = u_direction[ev[lane * N_SHAPE + 17] * direction_stride];
            block_coordinates[54][lane] = points[0][ev[lane * N_SHAPE + 18]];
            block_coordinates[55][lane] = points[1][ev[lane * N_SHAPE + 18]];
            block_coordinates[56][lane] = points[2][ev[lane * N_SHAPE + 18]];
            block_direction[18][lane] = u_direction[ev[lane * N_SHAPE + 18] * direction_stride];
            block_coordinates[57][lane] = points[0][ev[lane * N_SHAPE + 19]];
            block_coordinates[58][lane] = points[1][ev[lane * N_SHAPE + 19]];
            block_coordinates[59][lane] = points[2][ev[lane * N_SHAPE + 19]];
            block_direction[19][lane] = u_direction[ev[lane * N_SHAPE + 19] * direction_stride];
            block_coordinates[60][lane] = points[0][ev[lane * N_SHAPE + 20]];
            block_coordinates[61][lane] = points[1][ev[lane * N_SHAPE + 20]];
            block_coordinates[62][lane] = points[2][ev[lane * N_SHAPE + 20]];
            block_direction[20][lane] = u_direction[ev[lane * N_SHAPE + 20] * direction_stride];
            block_coordinates[63][lane] = points[0][ev[lane * N_SHAPE + 21]];
            block_coordinates[64][lane] = points[1][ev[lane * N_SHAPE + 21]];
            block_coordinates[65][lane] = points[2][ev[lane * N_SHAPE + 21]];
            block_direction[21][lane] = u_direction[ev[lane * N_SHAPE + 21] * direction_stride];
            block_coordinates[66][lane] = points[0][ev[lane * N_SHAPE + 22]];
            block_coordinates[67][lane] = points[1][ev[lane * N_SHAPE + 22]];
            block_coordinates[68][lane] = points[2][ev[lane * N_SHAPE + 22]];
            block_direction[22][lane] = u_direction[ev[lane * N_SHAPE + 22] * direction_stride];
            block_coordinates[69][lane] = points[0][ev[lane * N_SHAPE + 23]];
            block_coordinates[70][lane] = points[1][ev[lane * N_SHAPE + 23]];
            block_coordinates[71][lane] = points[2][ev[lane * N_SHAPE + 23]];
            block_direction[23][lane] = u_direction[ev[lane * N_SHAPE + 23] * direction_stride];
            block_coordinates[72][lane] = points[0][ev[lane * N_SHAPE + 24]];
            block_coordinates[73][lane] = points[1][ev[lane * N_SHAPE + 24]];
            block_coordinates[74][lane] = points[2][ev[lane * N_SHAPE + 24]];
            block_direction[24][lane] = u_direction[ev[lane * N_SHAPE + 24] * direction_stride];
            block_coordinates[75][lane] = points[0][ev[lane * N_SHAPE + 25]];
            block_coordinates[76][lane] = points[1][ev[lane * N_SHAPE + 25]];
            block_coordinates[77][lane] = points[2][ev[lane * N_SHAPE + 25]];
            block_direction[25][lane] = u_direction[ev[lane * N_SHAPE + 25] * direction_stride];
            block_coordinates[78][lane] = points[0][ev[lane * N_SHAPE + 26]];
            block_coordinates[79][lane] = points[1][ev[lane * N_SHAPE + 26]];
            block_coordinates[80][lane] = points[2][ev[lane * N_SHAPE + 26]];
            block_direction[26][lane] = u_direction[ev[lane * N_SHAPE + 26] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26]};
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_direction_streams, kappa, block_output_streams);

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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 10] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 11] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 12] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 13] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 14] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 15] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 16] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 17] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 18] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 19] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 20] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 21] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 22] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 23] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 24] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 25] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 26] * out_stride] += block_output[26][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}
