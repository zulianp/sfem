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
struct laplace_proteus_hex125_affine_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[25] = {scalar_t(0.65772788257758863), scalar_t(0.6076926946610145), scalar_t(-0.40858201526174148), scalar_t(0.17553410810741277), scalar_t(-0.032372670084274538), scalar_t(0.022063103295100275), scalar_t(1.0587971821717581), scalar_t(-0.11346384011744702), scalar_t(0.039222340750583846), scalar_t(-0.006618786099995529), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-0.006618786099995509), scalar_t(0.039222340750583728), scalar_t(-0.11346384011744673), scalar_t(1.0587971821717581), scalar_t(0.022063103295100202), scalar_t(-0.032372670084274538), scalar_t(0.17553410810741277), scalar_t(-0.40858201526174143), scalar_t(0.60769269466101461), scalar_t(0.65772788257758863)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[25] = {scalar_t(-6.3158364273482457), scalar_t(10.111278303066971), scalar_t(-5.6882551126207481), scalar_t(2.306021025433568), scalar_t(-0.41320778853154499), scalar_t(-1.3001705560202661), scalar_t(-2.759999173503255), scalar_t(5.7732663757858971), scalar_t(-2.0658530069809662), scalar_t(0.35275636071858923), scalar_t(0.33333333333333331), scalar_t(-2.6666666666666665), scalar_t(0), scalar_t(2.6666666666666665), scalar_t(-0.33333333333333331), scalar_t(-0.35275636071858935), scalar_t(2.0658530069809657), scalar_t(-5.7732663757858997), scalar_t(2.7599991735032567), scalar_t(1.3001705560202657), scalar_t(0.41320778853154499), scalar_t(-2.3060210254335685), scalar_t(5.6882551126207463), scalar_t(-10.111278303066971), scalar_t(6.3158364273482457)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[5] = {scalar_t(0.11846344252809454), scalar_t(0.23931433524968312), scalar_t(0.2844444444444445), scalar_t(0.23931433524968312), scalar_t(0.11846344252809454)};
        return data;
    }
};

template <typename scalar_t>
struct laplace_proteus_hex125_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[25] = {scalar_t(0.65772788257758863), scalar_t(0.6076926946610145), scalar_t(-0.40858201526174148), scalar_t(0.17553410810741277), scalar_t(-0.032372670084274538), scalar_t(0.022063103295100275), scalar_t(1.0587971821717581), scalar_t(-0.11346384011744702), scalar_t(0.039222340750583846), scalar_t(-0.006618786099995529), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-0.006618786099995509), scalar_t(0.039222340750583728), scalar_t(-0.11346384011744673), scalar_t(1.0587971821717581), scalar_t(0.022063103295100202), scalar_t(-0.032372670084274538), scalar_t(0.17553410810741277), scalar_t(-0.40858201526174143), scalar_t(0.60769269466101461), scalar_t(0.65772788257758863)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[25] = {scalar_t(-6.3158364273482457), scalar_t(10.111278303066971), scalar_t(-5.6882551126207481), scalar_t(2.306021025433568), scalar_t(-0.41320778853154499), scalar_t(-1.3001705560202661), scalar_t(-2.759999173503255), scalar_t(5.7732663757858971), scalar_t(-2.0658530069809662), scalar_t(0.35275636071858923), scalar_t(0.33333333333333331), scalar_t(-2.6666666666666665), scalar_t(0), scalar_t(2.6666666666666665), scalar_t(-0.33333333333333331), scalar_t(-0.35275636071858935), scalar_t(2.0658530069809657), scalar_t(-5.7732663757858997), scalar_t(2.7599991735032567), scalar_t(1.3001705560202657), scalar_t(0.41320778853154499), scalar_t(-2.3060210254335685), scalar_t(5.6882551126207463), scalar_t(-10.111278303066971), scalar_t(6.3158364273482457)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[5] = {scalar_t(0.11846344252809454), scalar_t(0.23931433524968312), scalar_t(0.2844444444444445), scalar_t(0.23931433524968312), scalar_t(0.11846344252809454)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex125_residual_element_soa_diagnostics_data = {
    "laplace_proteus_hex125_residual_element_soa",
    "PROTEUS_HEX125",
    3,
    125,
    125,
    16,
    5,
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
    50,
    5,
    1,
    125,
    0,
    125,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex125_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex125_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex125_residual_element_soa",
            &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex125_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex125_residual_element_soa_float",
            &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex125_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex125_residual_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex125_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex125_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex125_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex125_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex125_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex125_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex125_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex125_jacobian_u_u_diagnostics_data = {
    "laplace_proteus_hex125_jacobian_u_u",
    "PROTEUS_HEX125",
    3,
    125,
    125,
    16,
    5,
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
    50,
    5,
    1,
    0,
    125,
    125,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex125_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_proteus_hex125_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex125_jacobian_u_u_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex125_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex125_jacobian_u_u",
            &sfem::codegen::laplace_proteus_hex125_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex125_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex125_jacobian_u_u_float",
            &sfem::codegen::laplace_proteus_hex125_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data = {
    "laplace_proteus_hex125_jacobian_action_element_soa",
    "PROTEUS_HEX125",
    3,
    125,
    125,
    16,
    5,
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
    50,
    5,
    1,
    0,
    125,
    125,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex125_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex125_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex125_jacobian_action_element_soa",
            &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex125_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex125_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex125_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex125_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex125_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex125_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex125_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int laplace_proteus_hex125_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[125],
        const double kappa,
        double *const SFEM_RESTRICT output[125]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<double, 125, 125, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<double>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex125_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[125],
        const float kappa,
        float *const SFEM_RESTRICT output[125]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<float, 125, 125, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<float>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex125_residual_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 125;
    static constexpr int N_SHAPE = 125;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex125_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex125_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex125_affine_reference_data<scalar_t>::q_weight_1d();

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
            ev[lane * N_SHAPE + 27] = elements[27][evbegin + lane];
            ev[lane * N_SHAPE + 28] = elements[28][evbegin + lane];
            ev[lane * N_SHAPE + 29] = elements[29][evbegin + lane];
            ev[lane * N_SHAPE + 30] = elements[30][evbegin + lane];
            ev[lane * N_SHAPE + 31] = elements[31][evbegin + lane];
            ev[lane * N_SHAPE + 32] = elements[32][evbegin + lane];
            ev[lane * N_SHAPE + 33] = elements[33][evbegin + lane];
            ev[lane * N_SHAPE + 34] = elements[34][evbegin + lane];
            ev[lane * N_SHAPE + 35] = elements[35][evbegin + lane];
            ev[lane * N_SHAPE + 36] = elements[36][evbegin + lane];
            ev[lane * N_SHAPE + 37] = elements[37][evbegin + lane];
            ev[lane * N_SHAPE + 38] = elements[38][evbegin + lane];
            ev[lane * N_SHAPE + 39] = elements[39][evbegin + lane];
            ev[lane * N_SHAPE + 40] = elements[40][evbegin + lane];
            ev[lane * N_SHAPE + 41] = elements[41][evbegin + lane];
            ev[lane * N_SHAPE + 42] = elements[42][evbegin + lane];
            ev[lane * N_SHAPE + 43] = elements[43][evbegin + lane];
            ev[lane * N_SHAPE + 44] = elements[44][evbegin + lane];
            ev[lane * N_SHAPE + 45] = elements[45][evbegin + lane];
            ev[lane * N_SHAPE + 46] = elements[46][evbegin + lane];
            ev[lane * N_SHAPE + 47] = elements[47][evbegin + lane];
            ev[lane * N_SHAPE + 48] = elements[48][evbegin + lane];
            ev[lane * N_SHAPE + 49] = elements[49][evbegin + lane];
            ev[lane * N_SHAPE + 50] = elements[50][evbegin + lane];
            ev[lane * N_SHAPE + 51] = elements[51][evbegin + lane];
            ev[lane * N_SHAPE + 52] = elements[52][evbegin + lane];
            ev[lane * N_SHAPE + 53] = elements[53][evbegin + lane];
            ev[lane * N_SHAPE + 54] = elements[54][evbegin + lane];
            ev[lane * N_SHAPE + 55] = elements[55][evbegin + lane];
            ev[lane * N_SHAPE + 56] = elements[56][evbegin + lane];
            ev[lane * N_SHAPE + 57] = elements[57][evbegin + lane];
            ev[lane * N_SHAPE + 58] = elements[58][evbegin + lane];
            ev[lane * N_SHAPE + 59] = elements[59][evbegin + lane];
            ev[lane * N_SHAPE + 60] = elements[60][evbegin + lane];
            ev[lane * N_SHAPE + 61] = elements[61][evbegin + lane];
            ev[lane * N_SHAPE + 62] = elements[62][evbegin + lane];
            ev[lane * N_SHAPE + 63] = elements[63][evbegin + lane];
            ev[lane * N_SHAPE + 64] = elements[64][evbegin + lane];
            ev[lane * N_SHAPE + 65] = elements[65][evbegin + lane];
            ev[lane * N_SHAPE + 66] = elements[66][evbegin + lane];
            ev[lane * N_SHAPE + 67] = elements[67][evbegin + lane];
            ev[lane * N_SHAPE + 68] = elements[68][evbegin + lane];
            ev[lane * N_SHAPE + 69] = elements[69][evbegin + lane];
            ev[lane * N_SHAPE + 70] = elements[70][evbegin + lane];
            ev[lane * N_SHAPE + 71] = elements[71][evbegin + lane];
            ev[lane * N_SHAPE + 72] = elements[72][evbegin + lane];
            ev[lane * N_SHAPE + 73] = elements[73][evbegin + lane];
            ev[lane * N_SHAPE + 74] = elements[74][evbegin + lane];
            ev[lane * N_SHAPE + 75] = elements[75][evbegin + lane];
            ev[lane * N_SHAPE + 76] = elements[76][evbegin + lane];
            ev[lane * N_SHAPE + 77] = elements[77][evbegin + lane];
            ev[lane * N_SHAPE + 78] = elements[78][evbegin + lane];
            ev[lane * N_SHAPE + 79] = elements[79][evbegin + lane];
            ev[lane * N_SHAPE + 80] = elements[80][evbegin + lane];
            ev[lane * N_SHAPE + 81] = elements[81][evbegin + lane];
            ev[lane * N_SHAPE + 82] = elements[82][evbegin + lane];
            ev[lane * N_SHAPE + 83] = elements[83][evbegin + lane];
            ev[lane * N_SHAPE + 84] = elements[84][evbegin + lane];
            ev[lane * N_SHAPE + 85] = elements[85][evbegin + lane];
            ev[lane * N_SHAPE + 86] = elements[86][evbegin + lane];
            ev[lane * N_SHAPE + 87] = elements[87][evbegin + lane];
            ev[lane * N_SHAPE + 88] = elements[88][evbegin + lane];
            ev[lane * N_SHAPE + 89] = elements[89][evbegin + lane];
            ev[lane * N_SHAPE + 90] = elements[90][evbegin + lane];
            ev[lane * N_SHAPE + 91] = elements[91][evbegin + lane];
            ev[lane * N_SHAPE + 92] = elements[92][evbegin + lane];
            ev[lane * N_SHAPE + 93] = elements[93][evbegin + lane];
            ev[lane * N_SHAPE + 94] = elements[94][evbegin + lane];
            ev[lane * N_SHAPE + 95] = elements[95][evbegin + lane];
            ev[lane * N_SHAPE + 96] = elements[96][evbegin + lane];
            ev[lane * N_SHAPE + 97] = elements[97][evbegin + lane];
            ev[lane * N_SHAPE + 98] = elements[98][evbegin + lane];
            ev[lane * N_SHAPE + 99] = elements[99][evbegin + lane];
            ev[lane * N_SHAPE + 100] = elements[100][evbegin + lane];
            ev[lane * N_SHAPE + 101] = elements[101][evbegin + lane];
            ev[lane * N_SHAPE + 102] = elements[102][evbegin + lane];
            ev[lane * N_SHAPE + 103] = elements[103][evbegin + lane];
            ev[lane * N_SHAPE + 104] = elements[104][evbegin + lane];
            ev[lane * N_SHAPE + 105] = elements[105][evbegin + lane];
            ev[lane * N_SHAPE + 106] = elements[106][evbegin + lane];
            ev[lane * N_SHAPE + 107] = elements[107][evbegin + lane];
            ev[lane * N_SHAPE + 108] = elements[108][evbegin + lane];
            ev[lane * N_SHAPE + 109] = elements[109][evbegin + lane];
            ev[lane * N_SHAPE + 110] = elements[110][evbegin + lane];
            ev[lane * N_SHAPE + 111] = elements[111][evbegin + lane];
            ev[lane * N_SHAPE + 112] = elements[112][evbegin + lane];
            ev[lane * N_SHAPE + 113] = elements[113][evbegin + lane];
            ev[lane * N_SHAPE + 114] = elements[114][evbegin + lane];
            ev[lane * N_SHAPE + 115] = elements[115][evbegin + lane];
            ev[lane * N_SHAPE + 116] = elements[116][evbegin + lane];
            ev[lane * N_SHAPE + 117] = elements[117][evbegin + lane];
            ev[lane * N_SHAPE + 118] = elements[118][evbegin + lane];
            ev[lane * N_SHAPE + 119] = elements[119][evbegin + lane];
            ev[lane * N_SHAPE + 120] = elements[120][evbegin + lane];
            ev[lane * N_SHAPE + 121] = elements[121][evbegin + lane];
            ev[lane * N_SHAPE + 122] = elements[122][evbegin + lane];
            ev[lane * N_SHAPE + 123] = elements[123][evbegin + lane];
            ev[lane * N_SHAPE + 124] = elements[124][evbegin + lane];
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
            block_current[27][lane] = u[ev[lane * N_SHAPE + 27] * current_stride];
            block_current[28][lane] = u[ev[lane * N_SHAPE + 28] * current_stride];
            block_current[29][lane] = u[ev[lane * N_SHAPE + 29] * current_stride];
            block_current[30][lane] = u[ev[lane * N_SHAPE + 30] * current_stride];
            block_current[31][lane] = u[ev[lane * N_SHAPE + 31] * current_stride];
            block_current[32][lane] = u[ev[lane * N_SHAPE + 32] * current_stride];
            block_current[33][lane] = u[ev[lane * N_SHAPE + 33] * current_stride];
            block_current[34][lane] = u[ev[lane * N_SHAPE + 34] * current_stride];
            block_current[35][lane] = u[ev[lane * N_SHAPE + 35] * current_stride];
            block_current[36][lane] = u[ev[lane * N_SHAPE + 36] * current_stride];
            block_current[37][lane] = u[ev[lane * N_SHAPE + 37] * current_stride];
            block_current[38][lane] = u[ev[lane * N_SHAPE + 38] * current_stride];
            block_current[39][lane] = u[ev[lane * N_SHAPE + 39] * current_stride];
            block_current[40][lane] = u[ev[lane * N_SHAPE + 40] * current_stride];
            block_current[41][lane] = u[ev[lane * N_SHAPE + 41] * current_stride];
            block_current[42][lane] = u[ev[lane * N_SHAPE + 42] * current_stride];
            block_current[43][lane] = u[ev[lane * N_SHAPE + 43] * current_stride];
            block_current[44][lane] = u[ev[lane * N_SHAPE + 44] * current_stride];
            block_current[45][lane] = u[ev[lane * N_SHAPE + 45] * current_stride];
            block_current[46][lane] = u[ev[lane * N_SHAPE + 46] * current_stride];
            block_current[47][lane] = u[ev[lane * N_SHAPE + 47] * current_stride];
            block_current[48][lane] = u[ev[lane * N_SHAPE + 48] * current_stride];
            block_current[49][lane] = u[ev[lane * N_SHAPE + 49] * current_stride];
            block_current[50][lane] = u[ev[lane * N_SHAPE + 50] * current_stride];
            block_current[51][lane] = u[ev[lane * N_SHAPE + 51] * current_stride];
            block_current[52][lane] = u[ev[lane * N_SHAPE + 52] * current_stride];
            block_current[53][lane] = u[ev[lane * N_SHAPE + 53] * current_stride];
            block_current[54][lane] = u[ev[lane * N_SHAPE + 54] * current_stride];
            block_current[55][lane] = u[ev[lane * N_SHAPE + 55] * current_stride];
            block_current[56][lane] = u[ev[lane * N_SHAPE + 56] * current_stride];
            block_current[57][lane] = u[ev[lane * N_SHAPE + 57] * current_stride];
            block_current[58][lane] = u[ev[lane * N_SHAPE + 58] * current_stride];
            block_current[59][lane] = u[ev[lane * N_SHAPE + 59] * current_stride];
            block_current[60][lane] = u[ev[lane * N_SHAPE + 60] * current_stride];
            block_current[61][lane] = u[ev[lane * N_SHAPE + 61] * current_stride];
            block_current[62][lane] = u[ev[lane * N_SHAPE + 62] * current_stride];
            block_current[63][lane] = u[ev[lane * N_SHAPE + 63] * current_stride];
            block_current[64][lane] = u[ev[lane * N_SHAPE + 64] * current_stride];
            block_current[65][lane] = u[ev[lane * N_SHAPE + 65] * current_stride];
            block_current[66][lane] = u[ev[lane * N_SHAPE + 66] * current_stride];
            block_current[67][lane] = u[ev[lane * N_SHAPE + 67] * current_stride];
            block_current[68][lane] = u[ev[lane * N_SHAPE + 68] * current_stride];
            block_current[69][lane] = u[ev[lane * N_SHAPE + 69] * current_stride];
            block_current[70][lane] = u[ev[lane * N_SHAPE + 70] * current_stride];
            block_current[71][lane] = u[ev[lane * N_SHAPE + 71] * current_stride];
            block_current[72][lane] = u[ev[lane * N_SHAPE + 72] * current_stride];
            block_current[73][lane] = u[ev[lane * N_SHAPE + 73] * current_stride];
            block_current[74][lane] = u[ev[lane * N_SHAPE + 74] * current_stride];
            block_current[75][lane] = u[ev[lane * N_SHAPE + 75] * current_stride];
            block_current[76][lane] = u[ev[lane * N_SHAPE + 76] * current_stride];
            block_current[77][lane] = u[ev[lane * N_SHAPE + 77] * current_stride];
            block_current[78][lane] = u[ev[lane * N_SHAPE + 78] * current_stride];
            block_current[79][lane] = u[ev[lane * N_SHAPE + 79] * current_stride];
            block_current[80][lane] = u[ev[lane * N_SHAPE + 80] * current_stride];
            block_current[81][lane] = u[ev[lane * N_SHAPE + 81] * current_stride];
            block_current[82][lane] = u[ev[lane * N_SHAPE + 82] * current_stride];
            block_current[83][lane] = u[ev[lane * N_SHAPE + 83] * current_stride];
            block_current[84][lane] = u[ev[lane * N_SHAPE + 84] * current_stride];
            block_current[85][lane] = u[ev[lane * N_SHAPE + 85] * current_stride];
            block_current[86][lane] = u[ev[lane * N_SHAPE + 86] * current_stride];
            block_current[87][lane] = u[ev[lane * N_SHAPE + 87] * current_stride];
            block_current[88][lane] = u[ev[lane * N_SHAPE + 88] * current_stride];
            block_current[89][lane] = u[ev[lane * N_SHAPE + 89] * current_stride];
            block_current[90][lane] = u[ev[lane * N_SHAPE + 90] * current_stride];
            block_current[91][lane] = u[ev[lane * N_SHAPE + 91] * current_stride];
            block_current[92][lane] = u[ev[lane * N_SHAPE + 92] * current_stride];
            block_current[93][lane] = u[ev[lane * N_SHAPE + 93] * current_stride];
            block_current[94][lane] = u[ev[lane * N_SHAPE + 94] * current_stride];
            block_current[95][lane] = u[ev[lane * N_SHAPE + 95] * current_stride];
            block_current[96][lane] = u[ev[lane * N_SHAPE + 96] * current_stride];
            block_current[97][lane] = u[ev[lane * N_SHAPE + 97] * current_stride];
            block_current[98][lane] = u[ev[lane * N_SHAPE + 98] * current_stride];
            block_current[99][lane] = u[ev[lane * N_SHAPE + 99] * current_stride];
            block_current[100][lane] = u[ev[lane * N_SHAPE + 100] * current_stride];
            block_current[101][lane] = u[ev[lane * N_SHAPE + 101] * current_stride];
            block_current[102][lane] = u[ev[lane * N_SHAPE + 102] * current_stride];
            block_current[103][lane] = u[ev[lane * N_SHAPE + 103] * current_stride];
            block_current[104][lane] = u[ev[lane * N_SHAPE + 104] * current_stride];
            block_current[105][lane] = u[ev[lane * N_SHAPE + 105] * current_stride];
            block_current[106][lane] = u[ev[lane * N_SHAPE + 106] * current_stride];
            block_current[107][lane] = u[ev[lane * N_SHAPE + 107] * current_stride];
            block_current[108][lane] = u[ev[lane * N_SHAPE + 108] * current_stride];
            block_current[109][lane] = u[ev[lane * N_SHAPE + 109] * current_stride];
            block_current[110][lane] = u[ev[lane * N_SHAPE + 110] * current_stride];
            block_current[111][lane] = u[ev[lane * N_SHAPE + 111] * current_stride];
            block_current[112][lane] = u[ev[lane * N_SHAPE + 112] * current_stride];
            block_current[113][lane] = u[ev[lane * N_SHAPE + 113] * current_stride];
            block_current[114][lane] = u[ev[lane * N_SHAPE + 114] * current_stride];
            block_current[115][lane] = u[ev[lane * N_SHAPE + 115] * current_stride];
            block_current[116][lane] = u[ev[lane * N_SHAPE + 116] * current_stride];
            block_current[117][lane] = u[ev[lane * N_SHAPE + 117] * current_stride];
            block_current[118][lane] = u[ev[lane * N_SHAPE + 118] * current_stride];
            block_current[119][lane] = u[ev[lane * N_SHAPE + 119] * current_stride];
            block_current[120][lane] = u[ev[lane * N_SHAPE + 120] * current_stride];
            block_current[121][lane] = u[ev[lane * N_SHAPE + 121] * current_stride];
            block_current[122][lane] = u[ev[lane * N_SHAPE + 122] * current_stride];
            block_current[123][lane] = u[ev[lane * N_SHAPE + 123] * current_stride];
            block_current[124][lane] = u[ev[lane * N_SHAPE + 124] * current_stride];
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
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
            block_output[34][lane] = scalar_t(0);
            block_output[35][lane] = scalar_t(0);
            block_output[36][lane] = scalar_t(0);
            block_output[37][lane] = scalar_t(0);
            block_output[38][lane] = scalar_t(0);
            block_output[39][lane] = scalar_t(0);
            block_output[40][lane] = scalar_t(0);
            block_output[41][lane] = scalar_t(0);
            block_output[42][lane] = scalar_t(0);
            block_output[43][lane] = scalar_t(0);
            block_output[44][lane] = scalar_t(0);
            block_output[45][lane] = scalar_t(0);
            block_output[46][lane] = scalar_t(0);
            block_output[47][lane] = scalar_t(0);
            block_output[48][lane] = scalar_t(0);
            block_output[49][lane] = scalar_t(0);
            block_output[50][lane] = scalar_t(0);
            block_output[51][lane] = scalar_t(0);
            block_output[52][lane] = scalar_t(0);
            block_output[53][lane] = scalar_t(0);
            block_output[54][lane] = scalar_t(0);
            block_output[55][lane] = scalar_t(0);
            block_output[56][lane] = scalar_t(0);
            block_output[57][lane] = scalar_t(0);
            block_output[58][lane] = scalar_t(0);
            block_output[59][lane] = scalar_t(0);
            block_output[60][lane] = scalar_t(0);
            block_output[61][lane] = scalar_t(0);
            block_output[62][lane] = scalar_t(0);
            block_output[63][lane] = scalar_t(0);
            block_output[64][lane] = scalar_t(0);
            block_output[65][lane] = scalar_t(0);
            block_output[66][lane] = scalar_t(0);
            block_output[67][lane] = scalar_t(0);
            block_output[68][lane] = scalar_t(0);
            block_output[69][lane] = scalar_t(0);
            block_output[70][lane] = scalar_t(0);
            block_output[71][lane] = scalar_t(0);
            block_output[72][lane] = scalar_t(0);
            block_output[73][lane] = scalar_t(0);
            block_output[74][lane] = scalar_t(0);
            block_output[75][lane] = scalar_t(0);
            block_output[76][lane] = scalar_t(0);
            block_output[77][lane] = scalar_t(0);
            block_output[78][lane] = scalar_t(0);
            block_output[79][lane] = scalar_t(0);
            block_output[80][lane] = scalar_t(0);
            block_output[81][lane] = scalar_t(0);
            block_output[82][lane] = scalar_t(0);
            block_output[83][lane] = scalar_t(0);
            block_output[84][lane] = scalar_t(0);
            block_output[85][lane] = scalar_t(0);
            block_output[86][lane] = scalar_t(0);
            block_output[87][lane] = scalar_t(0);
            block_output[88][lane] = scalar_t(0);
            block_output[89][lane] = scalar_t(0);
            block_output[90][lane] = scalar_t(0);
            block_output[91][lane] = scalar_t(0);
            block_output[92][lane] = scalar_t(0);
            block_output[93][lane] = scalar_t(0);
            block_output[94][lane] = scalar_t(0);
            block_output[95][lane] = scalar_t(0);
            block_output[96][lane] = scalar_t(0);
            block_output[97][lane] = scalar_t(0);
            block_output[98][lane] = scalar_t(0);
            block_output[99][lane] = scalar_t(0);
            block_output[100][lane] = scalar_t(0);
            block_output[101][lane] = scalar_t(0);
            block_output[102][lane] = scalar_t(0);
            block_output[103][lane] = scalar_t(0);
            block_output[104][lane] = scalar_t(0);
            block_output[105][lane] = scalar_t(0);
            block_output[106][lane] = scalar_t(0);
            block_output[107][lane] = scalar_t(0);
            block_output[108][lane] = scalar_t(0);
            block_output[109][lane] = scalar_t(0);
            block_output[110][lane] = scalar_t(0);
            block_output[111][lane] = scalar_t(0);
            block_output[112][lane] = scalar_t(0);
            block_output[113][lane] = scalar_t(0);
            block_output[114][lane] = scalar_t(0);
            block_output[115][lane] = scalar_t(0);
            block_output[116][lane] = scalar_t(0);
            block_output[117][lane] = scalar_t(0);
            block_output[118][lane] = scalar_t(0);
            block_output[119][lane] = scalar_t(0);
            block_output[120][lane] = scalar_t(0);
            block_output[121][lane] = scalar_t(0);
            block_output[122][lane] = scalar_t(0);
            block_output[123][lane] = scalar_t(0);
            block_output[124][lane] = scalar_t(0);
        }

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33], block_current[34], block_current[35], block_current[36], block_current[37], block_current[38], block_current[39], block_current[40], block_current[41], block_current[42], block_current[43], block_current[44], block_current[45], block_current[46], block_current[47], block_current[48], block_current[49], block_current[50], block_current[51], block_current[52], block_current[53], block_current[54], block_current[55], block_current[56], block_current[57], block_current[58], block_current[59], block_current[60], block_current[61], block_current[62], block_current[63], block_current[64], block_current[65], block_current[66], block_current[67], block_current[68], block_current[69], block_current[70], block_current[71], block_current[72], block_current[73], block_current[74], block_current[75], block_current[76], block_current[77], block_current[78], block_current[79], block_current[80], block_current[81], block_current[82], block_current[83], block_current[84], block_current[85], block_current[86], block_current[87], block_current[88], block_current[89], block_current[90], block_current[91], block_current[92], block_current[93], block_current[94], block_current[95], block_current[96], block_current[97], block_current[98], block_current[99], block_current[100], block_current[101], block_current[102], block_current[103], block_current[104], block_current[105], block_current[106], block_current[107], block_current[108], block_current[109], block_current[110], block_current[111], block_current[112], block_current[113], block_current[114], block_current[115], block_current[116], block_current[117], block_current[118], block_current[119], block_current[120], block_current[121], block_current[122], block_current[123], block_current[124]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63], block_output[64], block_output[65], block_output[66], block_output[67], block_output[68], block_output[69], block_output[70], block_output[71], block_output[72], block_output[73], block_output[74], block_output[75], block_output[76], block_output[77], block_output[78], block_output[79], block_output[80], block_output[81], block_output[82], block_output[83], block_output[84], block_output[85], block_output[86], block_output[87], block_output[88], block_output[89], block_output[90], block_output[91], block_output[92], block_output[93], block_output[94], block_output[95], block_output[96], block_output[97], block_output[98], block_output[99], block_output[100], block_output[101], block_output[102], block_output[103], block_output[104], block_output[105], block_output[106], block_output[107], block_output[108], block_output[109], block_output[110], block_output[111], block_output[112], block_output[113], block_output[114], block_output[115], block_output[116], block_output[117], block_output[118], block_output[119], block_output[120], block_output[121], block_output[122], block_output[123], block_output[124]};
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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 27] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 28] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 29] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 30] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 31] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 32] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 33] * out_stride] += block_output[33][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 34] * out_stride] += block_output[34][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 35] * out_stride] += block_output[35][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 36] * out_stride] += block_output[36][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 37] * out_stride] += block_output[37][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 38] * out_stride] += block_output[38][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 39] * out_stride] += block_output[39][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 40] * out_stride] += block_output[40][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 41] * out_stride] += block_output[41][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 42] * out_stride] += block_output[42][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 43] * out_stride] += block_output[43][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 44] * out_stride] += block_output[44][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 45] * out_stride] += block_output[45][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 46] * out_stride] += block_output[46][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 47] * out_stride] += block_output[47][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 48] * out_stride] += block_output[48][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 49] * out_stride] += block_output[49][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 50] * out_stride] += block_output[50][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 51] * out_stride] += block_output[51][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 52] * out_stride] += block_output[52][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 53] * out_stride] += block_output[53][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 54] * out_stride] += block_output[54][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 55] * out_stride] += block_output[55][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 56] * out_stride] += block_output[56][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 57] * out_stride] += block_output[57][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 58] * out_stride] += block_output[58][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 59] * out_stride] += block_output[59][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 60] * out_stride] += block_output[60][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 61] * out_stride] += block_output[61][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 62] * out_stride] += block_output[62][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 63] * out_stride] += block_output[63][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 64] * out_stride] += block_output[64][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 65] * out_stride] += block_output[65][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 66] * out_stride] += block_output[66][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 67] * out_stride] += block_output[67][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 68] * out_stride] += block_output[68][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 69] * out_stride] += block_output[69][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 70] * out_stride] += block_output[70][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 71] * out_stride] += block_output[71][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 72] * out_stride] += block_output[72][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 73] * out_stride] += block_output[73][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 74] * out_stride] += block_output[74][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 75] * out_stride] += block_output[75][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 76] * out_stride] += block_output[76][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 77] * out_stride] += block_output[77][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 78] * out_stride] += block_output[78][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 79] * out_stride] += block_output[79][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 80] * out_stride] += block_output[80][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 81] * out_stride] += block_output[81][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 82] * out_stride] += block_output[82][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 83] * out_stride] += block_output[83][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 84] * out_stride] += block_output[84][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 85] * out_stride] += block_output[85][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 86] * out_stride] += block_output[86][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 87] * out_stride] += block_output[87][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 88] * out_stride] += block_output[88][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 89] * out_stride] += block_output[89][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 90] * out_stride] += block_output[90][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 91] * out_stride] += block_output[91][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 92] * out_stride] += block_output[92][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 93] * out_stride] += block_output[93][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 94] * out_stride] += block_output[94][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 95] * out_stride] += block_output[95][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 96] * out_stride] += block_output[96][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 97] * out_stride] += block_output[97][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 98] * out_stride] += block_output[98][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 99] * out_stride] += block_output[99][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 100] * out_stride] += block_output[100][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 101] * out_stride] += block_output[101][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 102] * out_stride] += block_output[102][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 103] * out_stride] += block_output[103][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 104] * out_stride] += block_output[104][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 105] * out_stride] += block_output[105][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 106] * out_stride] += block_output[106][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 107] * out_stride] += block_output[107][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 108] * out_stride] += block_output[108][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 109] * out_stride] += block_output[109][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 110] * out_stride] += block_output[110][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 111] * out_stride] += block_output[111][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 112] * out_stride] += block_output[112][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 113] * out_stride] += block_output[113][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 114] * out_stride] += block_output[114][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 115] * out_stride] += block_output[115][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 116] * out_stride] += block_output[116][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 117] * out_stride] += block_output[117][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 118] * out_stride] += block_output[118][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 119] * out_stride] += block_output[119][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 120] * out_stride] += block_output[120][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 121] * out_stride] += block_output[121][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 122] * out_stride] += block_output[122][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 123] * out_stride] += block_output[123][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 124] * out_stride] += block_output[124][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex125_residual_affine_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex125_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex125_residual_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex125_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex125_residual_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 125;
    static constexpr int N_SHAPE = 125;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<scalar_t>::q_weight_1d();

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
            ev[lane * N_SHAPE + 27] = elements[27][evbegin + lane];
            ev[lane * N_SHAPE + 28] = elements[28][evbegin + lane];
            ev[lane * N_SHAPE + 29] = elements[29][evbegin + lane];
            ev[lane * N_SHAPE + 30] = elements[30][evbegin + lane];
            ev[lane * N_SHAPE + 31] = elements[31][evbegin + lane];
            ev[lane * N_SHAPE + 32] = elements[32][evbegin + lane];
            ev[lane * N_SHAPE + 33] = elements[33][evbegin + lane];
            ev[lane * N_SHAPE + 34] = elements[34][evbegin + lane];
            ev[lane * N_SHAPE + 35] = elements[35][evbegin + lane];
            ev[lane * N_SHAPE + 36] = elements[36][evbegin + lane];
            ev[lane * N_SHAPE + 37] = elements[37][evbegin + lane];
            ev[lane * N_SHAPE + 38] = elements[38][evbegin + lane];
            ev[lane * N_SHAPE + 39] = elements[39][evbegin + lane];
            ev[lane * N_SHAPE + 40] = elements[40][evbegin + lane];
            ev[lane * N_SHAPE + 41] = elements[41][evbegin + lane];
            ev[lane * N_SHAPE + 42] = elements[42][evbegin + lane];
            ev[lane * N_SHAPE + 43] = elements[43][evbegin + lane];
            ev[lane * N_SHAPE + 44] = elements[44][evbegin + lane];
            ev[lane * N_SHAPE + 45] = elements[45][evbegin + lane];
            ev[lane * N_SHAPE + 46] = elements[46][evbegin + lane];
            ev[lane * N_SHAPE + 47] = elements[47][evbegin + lane];
            ev[lane * N_SHAPE + 48] = elements[48][evbegin + lane];
            ev[lane * N_SHAPE + 49] = elements[49][evbegin + lane];
            ev[lane * N_SHAPE + 50] = elements[50][evbegin + lane];
            ev[lane * N_SHAPE + 51] = elements[51][evbegin + lane];
            ev[lane * N_SHAPE + 52] = elements[52][evbegin + lane];
            ev[lane * N_SHAPE + 53] = elements[53][evbegin + lane];
            ev[lane * N_SHAPE + 54] = elements[54][evbegin + lane];
            ev[lane * N_SHAPE + 55] = elements[55][evbegin + lane];
            ev[lane * N_SHAPE + 56] = elements[56][evbegin + lane];
            ev[lane * N_SHAPE + 57] = elements[57][evbegin + lane];
            ev[lane * N_SHAPE + 58] = elements[58][evbegin + lane];
            ev[lane * N_SHAPE + 59] = elements[59][evbegin + lane];
            ev[lane * N_SHAPE + 60] = elements[60][evbegin + lane];
            ev[lane * N_SHAPE + 61] = elements[61][evbegin + lane];
            ev[lane * N_SHAPE + 62] = elements[62][evbegin + lane];
            ev[lane * N_SHAPE + 63] = elements[63][evbegin + lane];
            ev[lane * N_SHAPE + 64] = elements[64][evbegin + lane];
            ev[lane * N_SHAPE + 65] = elements[65][evbegin + lane];
            ev[lane * N_SHAPE + 66] = elements[66][evbegin + lane];
            ev[lane * N_SHAPE + 67] = elements[67][evbegin + lane];
            ev[lane * N_SHAPE + 68] = elements[68][evbegin + lane];
            ev[lane * N_SHAPE + 69] = elements[69][evbegin + lane];
            ev[lane * N_SHAPE + 70] = elements[70][evbegin + lane];
            ev[lane * N_SHAPE + 71] = elements[71][evbegin + lane];
            ev[lane * N_SHAPE + 72] = elements[72][evbegin + lane];
            ev[lane * N_SHAPE + 73] = elements[73][evbegin + lane];
            ev[lane * N_SHAPE + 74] = elements[74][evbegin + lane];
            ev[lane * N_SHAPE + 75] = elements[75][evbegin + lane];
            ev[lane * N_SHAPE + 76] = elements[76][evbegin + lane];
            ev[lane * N_SHAPE + 77] = elements[77][evbegin + lane];
            ev[lane * N_SHAPE + 78] = elements[78][evbegin + lane];
            ev[lane * N_SHAPE + 79] = elements[79][evbegin + lane];
            ev[lane * N_SHAPE + 80] = elements[80][evbegin + lane];
            ev[lane * N_SHAPE + 81] = elements[81][evbegin + lane];
            ev[lane * N_SHAPE + 82] = elements[82][evbegin + lane];
            ev[lane * N_SHAPE + 83] = elements[83][evbegin + lane];
            ev[lane * N_SHAPE + 84] = elements[84][evbegin + lane];
            ev[lane * N_SHAPE + 85] = elements[85][evbegin + lane];
            ev[lane * N_SHAPE + 86] = elements[86][evbegin + lane];
            ev[lane * N_SHAPE + 87] = elements[87][evbegin + lane];
            ev[lane * N_SHAPE + 88] = elements[88][evbegin + lane];
            ev[lane * N_SHAPE + 89] = elements[89][evbegin + lane];
            ev[lane * N_SHAPE + 90] = elements[90][evbegin + lane];
            ev[lane * N_SHAPE + 91] = elements[91][evbegin + lane];
            ev[lane * N_SHAPE + 92] = elements[92][evbegin + lane];
            ev[lane * N_SHAPE + 93] = elements[93][evbegin + lane];
            ev[lane * N_SHAPE + 94] = elements[94][evbegin + lane];
            ev[lane * N_SHAPE + 95] = elements[95][evbegin + lane];
            ev[lane * N_SHAPE + 96] = elements[96][evbegin + lane];
            ev[lane * N_SHAPE + 97] = elements[97][evbegin + lane];
            ev[lane * N_SHAPE + 98] = elements[98][evbegin + lane];
            ev[lane * N_SHAPE + 99] = elements[99][evbegin + lane];
            ev[lane * N_SHAPE + 100] = elements[100][evbegin + lane];
            ev[lane * N_SHAPE + 101] = elements[101][evbegin + lane];
            ev[lane * N_SHAPE + 102] = elements[102][evbegin + lane];
            ev[lane * N_SHAPE + 103] = elements[103][evbegin + lane];
            ev[lane * N_SHAPE + 104] = elements[104][evbegin + lane];
            ev[lane * N_SHAPE + 105] = elements[105][evbegin + lane];
            ev[lane * N_SHAPE + 106] = elements[106][evbegin + lane];
            ev[lane * N_SHAPE + 107] = elements[107][evbegin + lane];
            ev[lane * N_SHAPE + 108] = elements[108][evbegin + lane];
            ev[lane * N_SHAPE + 109] = elements[109][evbegin + lane];
            ev[lane * N_SHAPE + 110] = elements[110][evbegin + lane];
            ev[lane * N_SHAPE + 111] = elements[111][evbegin + lane];
            ev[lane * N_SHAPE + 112] = elements[112][evbegin + lane];
            ev[lane * N_SHAPE + 113] = elements[113][evbegin + lane];
            ev[lane * N_SHAPE + 114] = elements[114][evbegin + lane];
            ev[lane * N_SHAPE + 115] = elements[115][evbegin + lane];
            ev[lane * N_SHAPE + 116] = elements[116][evbegin + lane];
            ev[lane * N_SHAPE + 117] = elements[117][evbegin + lane];
            ev[lane * N_SHAPE + 118] = elements[118][evbegin + lane];
            ev[lane * N_SHAPE + 119] = elements[119][evbegin + lane];
            ev[lane * N_SHAPE + 120] = elements[120][evbegin + lane];
            ev[lane * N_SHAPE + 121] = elements[121][evbegin + lane];
            ev[lane * N_SHAPE + 122] = elements[122][evbegin + lane];
            ev[lane * N_SHAPE + 123] = elements[123][evbegin + lane];
            ev[lane * N_SHAPE + 124] = elements[124][evbegin + lane];
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
            block_coordinates[81][lane] = points[0][ev[lane * N_SHAPE + 27]];
            block_coordinates[82][lane] = points[1][ev[lane * N_SHAPE + 27]];
            block_coordinates[83][lane] = points[2][ev[lane * N_SHAPE + 27]];
            block_current[27][lane] = u[ev[lane * N_SHAPE + 27] * current_stride];
            block_coordinates[84][lane] = points[0][ev[lane * N_SHAPE + 28]];
            block_coordinates[85][lane] = points[1][ev[lane * N_SHAPE + 28]];
            block_coordinates[86][lane] = points[2][ev[lane * N_SHAPE + 28]];
            block_current[28][lane] = u[ev[lane * N_SHAPE + 28] * current_stride];
            block_coordinates[87][lane] = points[0][ev[lane * N_SHAPE + 29]];
            block_coordinates[88][lane] = points[1][ev[lane * N_SHAPE + 29]];
            block_coordinates[89][lane] = points[2][ev[lane * N_SHAPE + 29]];
            block_current[29][lane] = u[ev[lane * N_SHAPE + 29] * current_stride];
            block_coordinates[90][lane] = points[0][ev[lane * N_SHAPE + 30]];
            block_coordinates[91][lane] = points[1][ev[lane * N_SHAPE + 30]];
            block_coordinates[92][lane] = points[2][ev[lane * N_SHAPE + 30]];
            block_current[30][lane] = u[ev[lane * N_SHAPE + 30] * current_stride];
            block_coordinates[93][lane] = points[0][ev[lane * N_SHAPE + 31]];
            block_coordinates[94][lane] = points[1][ev[lane * N_SHAPE + 31]];
            block_coordinates[95][lane] = points[2][ev[lane * N_SHAPE + 31]];
            block_current[31][lane] = u[ev[lane * N_SHAPE + 31] * current_stride];
            block_coordinates[96][lane] = points[0][ev[lane * N_SHAPE + 32]];
            block_coordinates[97][lane] = points[1][ev[lane * N_SHAPE + 32]];
            block_coordinates[98][lane] = points[2][ev[lane * N_SHAPE + 32]];
            block_current[32][lane] = u[ev[lane * N_SHAPE + 32] * current_stride];
            block_coordinates[99][lane] = points[0][ev[lane * N_SHAPE + 33]];
            block_coordinates[100][lane] = points[1][ev[lane * N_SHAPE + 33]];
            block_coordinates[101][lane] = points[2][ev[lane * N_SHAPE + 33]];
            block_current[33][lane] = u[ev[lane * N_SHAPE + 33] * current_stride];
            block_coordinates[102][lane] = points[0][ev[lane * N_SHAPE + 34]];
            block_coordinates[103][lane] = points[1][ev[lane * N_SHAPE + 34]];
            block_coordinates[104][lane] = points[2][ev[lane * N_SHAPE + 34]];
            block_current[34][lane] = u[ev[lane * N_SHAPE + 34] * current_stride];
            block_coordinates[105][lane] = points[0][ev[lane * N_SHAPE + 35]];
            block_coordinates[106][lane] = points[1][ev[lane * N_SHAPE + 35]];
            block_coordinates[107][lane] = points[2][ev[lane * N_SHAPE + 35]];
            block_current[35][lane] = u[ev[lane * N_SHAPE + 35] * current_stride];
            block_coordinates[108][lane] = points[0][ev[lane * N_SHAPE + 36]];
            block_coordinates[109][lane] = points[1][ev[lane * N_SHAPE + 36]];
            block_coordinates[110][lane] = points[2][ev[lane * N_SHAPE + 36]];
            block_current[36][lane] = u[ev[lane * N_SHAPE + 36] * current_stride];
            block_coordinates[111][lane] = points[0][ev[lane * N_SHAPE + 37]];
            block_coordinates[112][lane] = points[1][ev[lane * N_SHAPE + 37]];
            block_coordinates[113][lane] = points[2][ev[lane * N_SHAPE + 37]];
            block_current[37][lane] = u[ev[lane * N_SHAPE + 37] * current_stride];
            block_coordinates[114][lane] = points[0][ev[lane * N_SHAPE + 38]];
            block_coordinates[115][lane] = points[1][ev[lane * N_SHAPE + 38]];
            block_coordinates[116][lane] = points[2][ev[lane * N_SHAPE + 38]];
            block_current[38][lane] = u[ev[lane * N_SHAPE + 38] * current_stride];
            block_coordinates[117][lane] = points[0][ev[lane * N_SHAPE + 39]];
            block_coordinates[118][lane] = points[1][ev[lane * N_SHAPE + 39]];
            block_coordinates[119][lane] = points[2][ev[lane * N_SHAPE + 39]];
            block_current[39][lane] = u[ev[lane * N_SHAPE + 39] * current_stride];
            block_coordinates[120][lane] = points[0][ev[lane * N_SHAPE + 40]];
            block_coordinates[121][lane] = points[1][ev[lane * N_SHAPE + 40]];
            block_coordinates[122][lane] = points[2][ev[lane * N_SHAPE + 40]];
            block_current[40][lane] = u[ev[lane * N_SHAPE + 40] * current_stride];
            block_coordinates[123][lane] = points[0][ev[lane * N_SHAPE + 41]];
            block_coordinates[124][lane] = points[1][ev[lane * N_SHAPE + 41]];
            block_coordinates[125][lane] = points[2][ev[lane * N_SHAPE + 41]];
            block_current[41][lane] = u[ev[lane * N_SHAPE + 41] * current_stride];
            block_coordinates[126][lane] = points[0][ev[lane * N_SHAPE + 42]];
            block_coordinates[127][lane] = points[1][ev[lane * N_SHAPE + 42]];
            block_coordinates[128][lane] = points[2][ev[lane * N_SHAPE + 42]];
            block_current[42][lane] = u[ev[lane * N_SHAPE + 42] * current_stride];
            block_coordinates[129][lane] = points[0][ev[lane * N_SHAPE + 43]];
            block_coordinates[130][lane] = points[1][ev[lane * N_SHAPE + 43]];
            block_coordinates[131][lane] = points[2][ev[lane * N_SHAPE + 43]];
            block_current[43][lane] = u[ev[lane * N_SHAPE + 43] * current_stride];
            block_coordinates[132][lane] = points[0][ev[lane * N_SHAPE + 44]];
            block_coordinates[133][lane] = points[1][ev[lane * N_SHAPE + 44]];
            block_coordinates[134][lane] = points[2][ev[lane * N_SHAPE + 44]];
            block_current[44][lane] = u[ev[lane * N_SHAPE + 44] * current_stride];
            block_coordinates[135][lane] = points[0][ev[lane * N_SHAPE + 45]];
            block_coordinates[136][lane] = points[1][ev[lane * N_SHAPE + 45]];
            block_coordinates[137][lane] = points[2][ev[lane * N_SHAPE + 45]];
            block_current[45][lane] = u[ev[lane * N_SHAPE + 45] * current_stride];
            block_coordinates[138][lane] = points[0][ev[lane * N_SHAPE + 46]];
            block_coordinates[139][lane] = points[1][ev[lane * N_SHAPE + 46]];
            block_coordinates[140][lane] = points[2][ev[lane * N_SHAPE + 46]];
            block_current[46][lane] = u[ev[lane * N_SHAPE + 46] * current_stride];
            block_coordinates[141][lane] = points[0][ev[lane * N_SHAPE + 47]];
            block_coordinates[142][lane] = points[1][ev[lane * N_SHAPE + 47]];
            block_coordinates[143][lane] = points[2][ev[lane * N_SHAPE + 47]];
            block_current[47][lane] = u[ev[lane * N_SHAPE + 47] * current_stride];
            block_coordinates[144][lane] = points[0][ev[lane * N_SHAPE + 48]];
            block_coordinates[145][lane] = points[1][ev[lane * N_SHAPE + 48]];
            block_coordinates[146][lane] = points[2][ev[lane * N_SHAPE + 48]];
            block_current[48][lane] = u[ev[lane * N_SHAPE + 48] * current_stride];
            block_coordinates[147][lane] = points[0][ev[lane * N_SHAPE + 49]];
            block_coordinates[148][lane] = points[1][ev[lane * N_SHAPE + 49]];
            block_coordinates[149][lane] = points[2][ev[lane * N_SHAPE + 49]];
            block_current[49][lane] = u[ev[lane * N_SHAPE + 49] * current_stride];
            block_coordinates[150][lane] = points[0][ev[lane * N_SHAPE + 50]];
            block_coordinates[151][lane] = points[1][ev[lane * N_SHAPE + 50]];
            block_coordinates[152][lane] = points[2][ev[lane * N_SHAPE + 50]];
            block_current[50][lane] = u[ev[lane * N_SHAPE + 50] * current_stride];
            block_coordinates[153][lane] = points[0][ev[lane * N_SHAPE + 51]];
            block_coordinates[154][lane] = points[1][ev[lane * N_SHAPE + 51]];
            block_coordinates[155][lane] = points[2][ev[lane * N_SHAPE + 51]];
            block_current[51][lane] = u[ev[lane * N_SHAPE + 51] * current_stride];
            block_coordinates[156][lane] = points[0][ev[lane * N_SHAPE + 52]];
            block_coordinates[157][lane] = points[1][ev[lane * N_SHAPE + 52]];
            block_coordinates[158][lane] = points[2][ev[lane * N_SHAPE + 52]];
            block_current[52][lane] = u[ev[lane * N_SHAPE + 52] * current_stride];
            block_coordinates[159][lane] = points[0][ev[lane * N_SHAPE + 53]];
            block_coordinates[160][lane] = points[1][ev[lane * N_SHAPE + 53]];
            block_coordinates[161][lane] = points[2][ev[lane * N_SHAPE + 53]];
            block_current[53][lane] = u[ev[lane * N_SHAPE + 53] * current_stride];
            block_coordinates[162][lane] = points[0][ev[lane * N_SHAPE + 54]];
            block_coordinates[163][lane] = points[1][ev[lane * N_SHAPE + 54]];
            block_coordinates[164][lane] = points[2][ev[lane * N_SHAPE + 54]];
            block_current[54][lane] = u[ev[lane * N_SHAPE + 54] * current_stride];
            block_coordinates[165][lane] = points[0][ev[lane * N_SHAPE + 55]];
            block_coordinates[166][lane] = points[1][ev[lane * N_SHAPE + 55]];
            block_coordinates[167][lane] = points[2][ev[lane * N_SHAPE + 55]];
            block_current[55][lane] = u[ev[lane * N_SHAPE + 55] * current_stride];
            block_coordinates[168][lane] = points[0][ev[lane * N_SHAPE + 56]];
            block_coordinates[169][lane] = points[1][ev[lane * N_SHAPE + 56]];
            block_coordinates[170][lane] = points[2][ev[lane * N_SHAPE + 56]];
            block_current[56][lane] = u[ev[lane * N_SHAPE + 56] * current_stride];
            block_coordinates[171][lane] = points[0][ev[lane * N_SHAPE + 57]];
            block_coordinates[172][lane] = points[1][ev[lane * N_SHAPE + 57]];
            block_coordinates[173][lane] = points[2][ev[lane * N_SHAPE + 57]];
            block_current[57][lane] = u[ev[lane * N_SHAPE + 57] * current_stride];
            block_coordinates[174][lane] = points[0][ev[lane * N_SHAPE + 58]];
            block_coordinates[175][lane] = points[1][ev[lane * N_SHAPE + 58]];
            block_coordinates[176][lane] = points[2][ev[lane * N_SHAPE + 58]];
            block_current[58][lane] = u[ev[lane * N_SHAPE + 58] * current_stride];
            block_coordinates[177][lane] = points[0][ev[lane * N_SHAPE + 59]];
            block_coordinates[178][lane] = points[1][ev[lane * N_SHAPE + 59]];
            block_coordinates[179][lane] = points[2][ev[lane * N_SHAPE + 59]];
            block_current[59][lane] = u[ev[lane * N_SHAPE + 59] * current_stride];
            block_coordinates[180][lane] = points[0][ev[lane * N_SHAPE + 60]];
            block_coordinates[181][lane] = points[1][ev[lane * N_SHAPE + 60]];
            block_coordinates[182][lane] = points[2][ev[lane * N_SHAPE + 60]];
            block_current[60][lane] = u[ev[lane * N_SHAPE + 60] * current_stride];
            block_coordinates[183][lane] = points[0][ev[lane * N_SHAPE + 61]];
            block_coordinates[184][lane] = points[1][ev[lane * N_SHAPE + 61]];
            block_coordinates[185][lane] = points[2][ev[lane * N_SHAPE + 61]];
            block_current[61][lane] = u[ev[lane * N_SHAPE + 61] * current_stride];
            block_coordinates[186][lane] = points[0][ev[lane * N_SHAPE + 62]];
            block_coordinates[187][lane] = points[1][ev[lane * N_SHAPE + 62]];
            block_coordinates[188][lane] = points[2][ev[lane * N_SHAPE + 62]];
            block_current[62][lane] = u[ev[lane * N_SHAPE + 62] * current_stride];
            block_coordinates[189][lane] = points[0][ev[lane * N_SHAPE + 63]];
            block_coordinates[190][lane] = points[1][ev[lane * N_SHAPE + 63]];
            block_coordinates[191][lane] = points[2][ev[lane * N_SHAPE + 63]];
            block_current[63][lane] = u[ev[lane * N_SHAPE + 63] * current_stride];
            block_coordinates[192][lane] = points[0][ev[lane * N_SHAPE + 64]];
            block_coordinates[193][lane] = points[1][ev[lane * N_SHAPE + 64]];
            block_coordinates[194][lane] = points[2][ev[lane * N_SHAPE + 64]];
            block_current[64][lane] = u[ev[lane * N_SHAPE + 64] * current_stride];
            block_coordinates[195][lane] = points[0][ev[lane * N_SHAPE + 65]];
            block_coordinates[196][lane] = points[1][ev[lane * N_SHAPE + 65]];
            block_coordinates[197][lane] = points[2][ev[lane * N_SHAPE + 65]];
            block_current[65][lane] = u[ev[lane * N_SHAPE + 65] * current_stride];
            block_coordinates[198][lane] = points[0][ev[lane * N_SHAPE + 66]];
            block_coordinates[199][lane] = points[1][ev[lane * N_SHAPE + 66]];
            block_coordinates[200][lane] = points[2][ev[lane * N_SHAPE + 66]];
            block_current[66][lane] = u[ev[lane * N_SHAPE + 66] * current_stride];
            block_coordinates[201][lane] = points[0][ev[lane * N_SHAPE + 67]];
            block_coordinates[202][lane] = points[1][ev[lane * N_SHAPE + 67]];
            block_coordinates[203][lane] = points[2][ev[lane * N_SHAPE + 67]];
            block_current[67][lane] = u[ev[lane * N_SHAPE + 67] * current_stride];
            block_coordinates[204][lane] = points[0][ev[lane * N_SHAPE + 68]];
            block_coordinates[205][lane] = points[1][ev[lane * N_SHAPE + 68]];
            block_coordinates[206][lane] = points[2][ev[lane * N_SHAPE + 68]];
            block_current[68][lane] = u[ev[lane * N_SHAPE + 68] * current_stride];
            block_coordinates[207][lane] = points[0][ev[lane * N_SHAPE + 69]];
            block_coordinates[208][lane] = points[1][ev[lane * N_SHAPE + 69]];
            block_coordinates[209][lane] = points[2][ev[lane * N_SHAPE + 69]];
            block_current[69][lane] = u[ev[lane * N_SHAPE + 69] * current_stride];
            block_coordinates[210][lane] = points[0][ev[lane * N_SHAPE + 70]];
            block_coordinates[211][lane] = points[1][ev[lane * N_SHAPE + 70]];
            block_coordinates[212][lane] = points[2][ev[lane * N_SHAPE + 70]];
            block_current[70][lane] = u[ev[lane * N_SHAPE + 70] * current_stride];
            block_coordinates[213][lane] = points[0][ev[lane * N_SHAPE + 71]];
            block_coordinates[214][lane] = points[1][ev[lane * N_SHAPE + 71]];
            block_coordinates[215][lane] = points[2][ev[lane * N_SHAPE + 71]];
            block_current[71][lane] = u[ev[lane * N_SHAPE + 71] * current_stride];
            block_coordinates[216][lane] = points[0][ev[lane * N_SHAPE + 72]];
            block_coordinates[217][lane] = points[1][ev[lane * N_SHAPE + 72]];
            block_coordinates[218][lane] = points[2][ev[lane * N_SHAPE + 72]];
            block_current[72][lane] = u[ev[lane * N_SHAPE + 72] * current_stride];
            block_coordinates[219][lane] = points[0][ev[lane * N_SHAPE + 73]];
            block_coordinates[220][lane] = points[1][ev[lane * N_SHAPE + 73]];
            block_coordinates[221][lane] = points[2][ev[lane * N_SHAPE + 73]];
            block_current[73][lane] = u[ev[lane * N_SHAPE + 73] * current_stride];
            block_coordinates[222][lane] = points[0][ev[lane * N_SHAPE + 74]];
            block_coordinates[223][lane] = points[1][ev[lane * N_SHAPE + 74]];
            block_coordinates[224][lane] = points[2][ev[lane * N_SHAPE + 74]];
            block_current[74][lane] = u[ev[lane * N_SHAPE + 74] * current_stride];
            block_coordinates[225][lane] = points[0][ev[lane * N_SHAPE + 75]];
            block_coordinates[226][lane] = points[1][ev[lane * N_SHAPE + 75]];
            block_coordinates[227][lane] = points[2][ev[lane * N_SHAPE + 75]];
            block_current[75][lane] = u[ev[lane * N_SHAPE + 75] * current_stride];
            block_coordinates[228][lane] = points[0][ev[lane * N_SHAPE + 76]];
            block_coordinates[229][lane] = points[1][ev[lane * N_SHAPE + 76]];
            block_coordinates[230][lane] = points[2][ev[lane * N_SHAPE + 76]];
            block_current[76][lane] = u[ev[lane * N_SHAPE + 76] * current_stride];
            block_coordinates[231][lane] = points[0][ev[lane * N_SHAPE + 77]];
            block_coordinates[232][lane] = points[1][ev[lane * N_SHAPE + 77]];
            block_coordinates[233][lane] = points[2][ev[lane * N_SHAPE + 77]];
            block_current[77][lane] = u[ev[lane * N_SHAPE + 77] * current_stride];
            block_coordinates[234][lane] = points[0][ev[lane * N_SHAPE + 78]];
            block_coordinates[235][lane] = points[1][ev[lane * N_SHAPE + 78]];
            block_coordinates[236][lane] = points[2][ev[lane * N_SHAPE + 78]];
            block_current[78][lane] = u[ev[lane * N_SHAPE + 78] * current_stride];
            block_coordinates[237][lane] = points[0][ev[lane * N_SHAPE + 79]];
            block_coordinates[238][lane] = points[1][ev[lane * N_SHAPE + 79]];
            block_coordinates[239][lane] = points[2][ev[lane * N_SHAPE + 79]];
            block_current[79][lane] = u[ev[lane * N_SHAPE + 79] * current_stride];
            block_coordinates[240][lane] = points[0][ev[lane * N_SHAPE + 80]];
            block_coordinates[241][lane] = points[1][ev[lane * N_SHAPE + 80]];
            block_coordinates[242][lane] = points[2][ev[lane * N_SHAPE + 80]];
            block_current[80][lane] = u[ev[lane * N_SHAPE + 80] * current_stride];
            block_coordinates[243][lane] = points[0][ev[lane * N_SHAPE + 81]];
            block_coordinates[244][lane] = points[1][ev[lane * N_SHAPE + 81]];
            block_coordinates[245][lane] = points[2][ev[lane * N_SHAPE + 81]];
            block_current[81][lane] = u[ev[lane * N_SHAPE + 81] * current_stride];
            block_coordinates[246][lane] = points[0][ev[lane * N_SHAPE + 82]];
            block_coordinates[247][lane] = points[1][ev[lane * N_SHAPE + 82]];
            block_coordinates[248][lane] = points[2][ev[lane * N_SHAPE + 82]];
            block_current[82][lane] = u[ev[lane * N_SHAPE + 82] * current_stride];
            block_coordinates[249][lane] = points[0][ev[lane * N_SHAPE + 83]];
            block_coordinates[250][lane] = points[1][ev[lane * N_SHAPE + 83]];
            block_coordinates[251][lane] = points[2][ev[lane * N_SHAPE + 83]];
            block_current[83][lane] = u[ev[lane * N_SHAPE + 83] * current_stride];
            block_coordinates[252][lane] = points[0][ev[lane * N_SHAPE + 84]];
            block_coordinates[253][lane] = points[1][ev[lane * N_SHAPE + 84]];
            block_coordinates[254][lane] = points[2][ev[lane * N_SHAPE + 84]];
            block_current[84][lane] = u[ev[lane * N_SHAPE + 84] * current_stride];
            block_coordinates[255][lane] = points[0][ev[lane * N_SHAPE + 85]];
            block_coordinates[256][lane] = points[1][ev[lane * N_SHAPE + 85]];
            block_coordinates[257][lane] = points[2][ev[lane * N_SHAPE + 85]];
            block_current[85][lane] = u[ev[lane * N_SHAPE + 85] * current_stride];
            block_coordinates[258][lane] = points[0][ev[lane * N_SHAPE + 86]];
            block_coordinates[259][lane] = points[1][ev[lane * N_SHAPE + 86]];
            block_coordinates[260][lane] = points[2][ev[lane * N_SHAPE + 86]];
            block_current[86][lane] = u[ev[lane * N_SHAPE + 86] * current_stride];
            block_coordinates[261][lane] = points[0][ev[lane * N_SHAPE + 87]];
            block_coordinates[262][lane] = points[1][ev[lane * N_SHAPE + 87]];
            block_coordinates[263][lane] = points[2][ev[lane * N_SHAPE + 87]];
            block_current[87][lane] = u[ev[lane * N_SHAPE + 87] * current_stride];
            block_coordinates[264][lane] = points[0][ev[lane * N_SHAPE + 88]];
            block_coordinates[265][lane] = points[1][ev[lane * N_SHAPE + 88]];
            block_coordinates[266][lane] = points[2][ev[lane * N_SHAPE + 88]];
            block_current[88][lane] = u[ev[lane * N_SHAPE + 88] * current_stride];
            block_coordinates[267][lane] = points[0][ev[lane * N_SHAPE + 89]];
            block_coordinates[268][lane] = points[1][ev[lane * N_SHAPE + 89]];
            block_coordinates[269][lane] = points[2][ev[lane * N_SHAPE + 89]];
            block_current[89][lane] = u[ev[lane * N_SHAPE + 89] * current_stride];
            block_coordinates[270][lane] = points[0][ev[lane * N_SHAPE + 90]];
            block_coordinates[271][lane] = points[1][ev[lane * N_SHAPE + 90]];
            block_coordinates[272][lane] = points[2][ev[lane * N_SHAPE + 90]];
            block_current[90][lane] = u[ev[lane * N_SHAPE + 90] * current_stride];
            block_coordinates[273][lane] = points[0][ev[lane * N_SHAPE + 91]];
            block_coordinates[274][lane] = points[1][ev[lane * N_SHAPE + 91]];
            block_coordinates[275][lane] = points[2][ev[lane * N_SHAPE + 91]];
            block_current[91][lane] = u[ev[lane * N_SHAPE + 91] * current_stride];
            block_coordinates[276][lane] = points[0][ev[lane * N_SHAPE + 92]];
            block_coordinates[277][lane] = points[1][ev[lane * N_SHAPE + 92]];
            block_coordinates[278][lane] = points[2][ev[lane * N_SHAPE + 92]];
            block_current[92][lane] = u[ev[lane * N_SHAPE + 92] * current_stride];
            block_coordinates[279][lane] = points[0][ev[lane * N_SHAPE + 93]];
            block_coordinates[280][lane] = points[1][ev[lane * N_SHAPE + 93]];
            block_coordinates[281][lane] = points[2][ev[lane * N_SHAPE + 93]];
            block_current[93][lane] = u[ev[lane * N_SHAPE + 93] * current_stride];
            block_coordinates[282][lane] = points[0][ev[lane * N_SHAPE + 94]];
            block_coordinates[283][lane] = points[1][ev[lane * N_SHAPE + 94]];
            block_coordinates[284][lane] = points[2][ev[lane * N_SHAPE + 94]];
            block_current[94][lane] = u[ev[lane * N_SHAPE + 94] * current_stride];
            block_coordinates[285][lane] = points[0][ev[lane * N_SHAPE + 95]];
            block_coordinates[286][lane] = points[1][ev[lane * N_SHAPE + 95]];
            block_coordinates[287][lane] = points[2][ev[lane * N_SHAPE + 95]];
            block_current[95][lane] = u[ev[lane * N_SHAPE + 95] * current_stride];
            block_coordinates[288][lane] = points[0][ev[lane * N_SHAPE + 96]];
            block_coordinates[289][lane] = points[1][ev[lane * N_SHAPE + 96]];
            block_coordinates[290][lane] = points[2][ev[lane * N_SHAPE + 96]];
            block_current[96][lane] = u[ev[lane * N_SHAPE + 96] * current_stride];
            block_coordinates[291][lane] = points[0][ev[lane * N_SHAPE + 97]];
            block_coordinates[292][lane] = points[1][ev[lane * N_SHAPE + 97]];
            block_coordinates[293][lane] = points[2][ev[lane * N_SHAPE + 97]];
            block_current[97][lane] = u[ev[lane * N_SHAPE + 97] * current_stride];
            block_coordinates[294][lane] = points[0][ev[lane * N_SHAPE + 98]];
            block_coordinates[295][lane] = points[1][ev[lane * N_SHAPE + 98]];
            block_coordinates[296][lane] = points[2][ev[lane * N_SHAPE + 98]];
            block_current[98][lane] = u[ev[lane * N_SHAPE + 98] * current_stride];
            block_coordinates[297][lane] = points[0][ev[lane * N_SHAPE + 99]];
            block_coordinates[298][lane] = points[1][ev[lane * N_SHAPE + 99]];
            block_coordinates[299][lane] = points[2][ev[lane * N_SHAPE + 99]];
            block_current[99][lane] = u[ev[lane * N_SHAPE + 99] * current_stride];
            block_coordinates[300][lane] = points[0][ev[lane * N_SHAPE + 100]];
            block_coordinates[301][lane] = points[1][ev[lane * N_SHAPE + 100]];
            block_coordinates[302][lane] = points[2][ev[lane * N_SHAPE + 100]];
            block_current[100][lane] = u[ev[lane * N_SHAPE + 100] * current_stride];
            block_coordinates[303][lane] = points[0][ev[lane * N_SHAPE + 101]];
            block_coordinates[304][lane] = points[1][ev[lane * N_SHAPE + 101]];
            block_coordinates[305][lane] = points[2][ev[lane * N_SHAPE + 101]];
            block_current[101][lane] = u[ev[lane * N_SHAPE + 101] * current_stride];
            block_coordinates[306][lane] = points[0][ev[lane * N_SHAPE + 102]];
            block_coordinates[307][lane] = points[1][ev[lane * N_SHAPE + 102]];
            block_coordinates[308][lane] = points[2][ev[lane * N_SHAPE + 102]];
            block_current[102][lane] = u[ev[lane * N_SHAPE + 102] * current_stride];
            block_coordinates[309][lane] = points[0][ev[lane * N_SHAPE + 103]];
            block_coordinates[310][lane] = points[1][ev[lane * N_SHAPE + 103]];
            block_coordinates[311][lane] = points[2][ev[lane * N_SHAPE + 103]];
            block_current[103][lane] = u[ev[lane * N_SHAPE + 103] * current_stride];
            block_coordinates[312][lane] = points[0][ev[lane * N_SHAPE + 104]];
            block_coordinates[313][lane] = points[1][ev[lane * N_SHAPE + 104]];
            block_coordinates[314][lane] = points[2][ev[lane * N_SHAPE + 104]];
            block_current[104][lane] = u[ev[lane * N_SHAPE + 104] * current_stride];
            block_coordinates[315][lane] = points[0][ev[lane * N_SHAPE + 105]];
            block_coordinates[316][lane] = points[1][ev[lane * N_SHAPE + 105]];
            block_coordinates[317][lane] = points[2][ev[lane * N_SHAPE + 105]];
            block_current[105][lane] = u[ev[lane * N_SHAPE + 105] * current_stride];
            block_coordinates[318][lane] = points[0][ev[lane * N_SHAPE + 106]];
            block_coordinates[319][lane] = points[1][ev[lane * N_SHAPE + 106]];
            block_coordinates[320][lane] = points[2][ev[lane * N_SHAPE + 106]];
            block_current[106][lane] = u[ev[lane * N_SHAPE + 106] * current_stride];
            block_coordinates[321][lane] = points[0][ev[lane * N_SHAPE + 107]];
            block_coordinates[322][lane] = points[1][ev[lane * N_SHAPE + 107]];
            block_coordinates[323][lane] = points[2][ev[lane * N_SHAPE + 107]];
            block_current[107][lane] = u[ev[lane * N_SHAPE + 107] * current_stride];
            block_coordinates[324][lane] = points[0][ev[lane * N_SHAPE + 108]];
            block_coordinates[325][lane] = points[1][ev[lane * N_SHAPE + 108]];
            block_coordinates[326][lane] = points[2][ev[lane * N_SHAPE + 108]];
            block_current[108][lane] = u[ev[lane * N_SHAPE + 108] * current_stride];
            block_coordinates[327][lane] = points[0][ev[lane * N_SHAPE + 109]];
            block_coordinates[328][lane] = points[1][ev[lane * N_SHAPE + 109]];
            block_coordinates[329][lane] = points[2][ev[lane * N_SHAPE + 109]];
            block_current[109][lane] = u[ev[lane * N_SHAPE + 109] * current_stride];
            block_coordinates[330][lane] = points[0][ev[lane * N_SHAPE + 110]];
            block_coordinates[331][lane] = points[1][ev[lane * N_SHAPE + 110]];
            block_coordinates[332][lane] = points[2][ev[lane * N_SHAPE + 110]];
            block_current[110][lane] = u[ev[lane * N_SHAPE + 110] * current_stride];
            block_coordinates[333][lane] = points[0][ev[lane * N_SHAPE + 111]];
            block_coordinates[334][lane] = points[1][ev[lane * N_SHAPE + 111]];
            block_coordinates[335][lane] = points[2][ev[lane * N_SHAPE + 111]];
            block_current[111][lane] = u[ev[lane * N_SHAPE + 111] * current_stride];
            block_coordinates[336][lane] = points[0][ev[lane * N_SHAPE + 112]];
            block_coordinates[337][lane] = points[1][ev[lane * N_SHAPE + 112]];
            block_coordinates[338][lane] = points[2][ev[lane * N_SHAPE + 112]];
            block_current[112][lane] = u[ev[lane * N_SHAPE + 112] * current_stride];
            block_coordinates[339][lane] = points[0][ev[lane * N_SHAPE + 113]];
            block_coordinates[340][lane] = points[1][ev[lane * N_SHAPE + 113]];
            block_coordinates[341][lane] = points[2][ev[lane * N_SHAPE + 113]];
            block_current[113][lane] = u[ev[lane * N_SHAPE + 113] * current_stride];
            block_coordinates[342][lane] = points[0][ev[lane * N_SHAPE + 114]];
            block_coordinates[343][lane] = points[1][ev[lane * N_SHAPE + 114]];
            block_coordinates[344][lane] = points[2][ev[lane * N_SHAPE + 114]];
            block_current[114][lane] = u[ev[lane * N_SHAPE + 114] * current_stride];
            block_coordinates[345][lane] = points[0][ev[lane * N_SHAPE + 115]];
            block_coordinates[346][lane] = points[1][ev[lane * N_SHAPE + 115]];
            block_coordinates[347][lane] = points[2][ev[lane * N_SHAPE + 115]];
            block_current[115][lane] = u[ev[lane * N_SHAPE + 115] * current_stride];
            block_coordinates[348][lane] = points[0][ev[lane * N_SHAPE + 116]];
            block_coordinates[349][lane] = points[1][ev[lane * N_SHAPE + 116]];
            block_coordinates[350][lane] = points[2][ev[lane * N_SHAPE + 116]];
            block_current[116][lane] = u[ev[lane * N_SHAPE + 116] * current_stride];
            block_coordinates[351][lane] = points[0][ev[lane * N_SHAPE + 117]];
            block_coordinates[352][lane] = points[1][ev[lane * N_SHAPE + 117]];
            block_coordinates[353][lane] = points[2][ev[lane * N_SHAPE + 117]];
            block_current[117][lane] = u[ev[lane * N_SHAPE + 117] * current_stride];
            block_coordinates[354][lane] = points[0][ev[lane * N_SHAPE + 118]];
            block_coordinates[355][lane] = points[1][ev[lane * N_SHAPE + 118]];
            block_coordinates[356][lane] = points[2][ev[lane * N_SHAPE + 118]];
            block_current[118][lane] = u[ev[lane * N_SHAPE + 118] * current_stride];
            block_coordinates[357][lane] = points[0][ev[lane * N_SHAPE + 119]];
            block_coordinates[358][lane] = points[1][ev[lane * N_SHAPE + 119]];
            block_coordinates[359][lane] = points[2][ev[lane * N_SHAPE + 119]];
            block_current[119][lane] = u[ev[lane * N_SHAPE + 119] * current_stride];
            block_coordinates[360][lane] = points[0][ev[lane * N_SHAPE + 120]];
            block_coordinates[361][lane] = points[1][ev[lane * N_SHAPE + 120]];
            block_coordinates[362][lane] = points[2][ev[lane * N_SHAPE + 120]];
            block_current[120][lane] = u[ev[lane * N_SHAPE + 120] * current_stride];
            block_coordinates[363][lane] = points[0][ev[lane * N_SHAPE + 121]];
            block_coordinates[364][lane] = points[1][ev[lane * N_SHAPE + 121]];
            block_coordinates[365][lane] = points[2][ev[lane * N_SHAPE + 121]];
            block_current[121][lane] = u[ev[lane * N_SHAPE + 121] * current_stride];
            block_coordinates[366][lane] = points[0][ev[lane * N_SHAPE + 122]];
            block_coordinates[367][lane] = points[1][ev[lane * N_SHAPE + 122]];
            block_coordinates[368][lane] = points[2][ev[lane * N_SHAPE + 122]];
            block_current[122][lane] = u[ev[lane * N_SHAPE + 122] * current_stride];
            block_coordinates[369][lane] = points[0][ev[lane * N_SHAPE + 123]];
            block_coordinates[370][lane] = points[1][ev[lane * N_SHAPE + 123]];
            block_coordinates[371][lane] = points[2][ev[lane * N_SHAPE + 123]];
            block_current[123][lane] = u[ev[lane * N_SHAPE + 123] * current_stride];
            block_coordinates[372][lane] = points[0][ev[lane * N_SHAPE + 124]];
            block_coordinates[373][lane] = points[1][ev[lane * N_SHAPE + 124]];
            block_coordinates[374][lane] = points[2][ev[lane * N_SHAPE + 124]];
            block_current[124][lane] = u[ev[lane * N_SHAPE + 124] * current_stride];
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
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
            block_output[34][lane] = scalar_t(0);
            block_output[35][lane] = scalar_t(0);
            block_output[36][lane] = scalar_t(0);
            block_output[37][lane] = scalar_t(0);
            block_output[38][lane] = scalar_t(0);
            block_output[39][lane] = scalar_t(0);
            block_output[40][lane] = scalar_t(0);
            block_output[41][lane] = scalar_t(0);
            block_output[42][lane] = scalar_t(0);
            block_output[43][lane] = scalar_t(0);
            block_output[44][lane] = scalar_t(0);
            block_output[45][lane] = scalar_t(0);
            block_output[46][lane] = scalar_t(0);
            block_output[47][lane] = scalar_t(0);
            block_output[48][lane] = scalar_t(0);
            block_output[49][lane] = scalar_t(0);
            block_output[50][lane] = scalar_t(0);
            block_output[51][lane] = scalar_t(0);
            block_output[52][lane] = scalar_t(0);
            block_output[53][lane] = scalar_t(0);
            block_output[54][lane] = scalar_t(0);
            block_output[55][lane] = scalar_t(0);
            block_output[56][lane] = scalar_t(0);
            block_output[57][lane] = scalar_t(0);
            block_output[58][lane] = scalar_t(0);
            block_output[59][lane] = scalar_t(0);
            block_output[60][lane] = scalar_t(0);
            block_output[61][lane] = scalar_t(0);
            block_output[62][lane] = scalar_t(0);
            block_output[63][lane] = scalar_t(0);
            block_output[64][lane] = scalar_t(0);
            block_output[65][lane] = scalar_t(0);
            block_output[66][lane] = scalar_t(0);
            block_output[67][lane] = scalar_t(0);
            block_output[68][lane] = scalar_t(0);
            block_output[69][lane] = scalar_t(0);
            block_output[70][lane] = scalar_t(0);
            block_output[71][lane] = scalar_t(0);
            block_output[72][lane] = scalar_t(0);
            block_output[73][lane] = scalar_t(0);
            block_output[74][lane] = scalar_t(0);
            block_output[75][lane] = scalar_t(0);
            block_output[76][lane] = scalar_t(0);
            block_output[77][lane] = scalar_t(0);
            block_output[78][lane] = scalar_t(0);
            block_output[79][lane] = scalar_t(0);
            block_output[80][lane] = scalar_t(0);
            block_output[81][lane] = scalar_t(0);
            block_output[82][lane] = scalar_t(0);
            block_output[83][lane] = scalar_t(0);
            block_output[84][lane] = scalar_t(0);
            block_output[85][lane] = scalar_t(0);
            block_output[86][lane] = scalar_t(0);
            block_output[87][lane] = scalar_t(0);
            block_output[88][lane] = scalar_t(0);
            block_output[89][lane] = scalar_t(0);
            block_output[90][lane] = scalar_t(0);
            block_output[91][lane] = scalar_t(0);
            block_output[92][lane] = scalar_t(0);
            block_output[93][lane] = scalar_t(0);
            block_output[94][lane] = scalar_t(0);
            block_output[95][lane] = scalar_t(0);
            block_output[96][lane] = scalar_t(0);
            block_output[97][lane] = scalar_t(0);
            block_output[98][lane] = scalar_t(0);
            block_output[99][lane] = scalar_t(0);
            block_output[100][lane] = scalar_t(0);
            block_output[101][lane] = scalar_t(0);
            block_output[102][lane] = scalar_t(0);
            block_output[103][lane] = scalar_t(0);
            block_output[104][lane] = scalar_t(0);
            block_output[105][lane] = scalar_t(0);
            block_output[106][lane] = scalar_t(0);
            block_output[107][lane] = scalar_t(0);
            block_output[108][lane] = scalar_t(0);
            block_output[109][lane] = scalar_t(0);
            block_output[110][lane] = scalar_t(0);
            block_output[111][lane] = scalar_t(0);
            block_output[112][lane] = scalar_t(0);
            block_output[113][lane] = scalar_t(0);
            block_output[114][lane] = scalar_t(0);
            block_output[115][lane] = scalar_t(0);
            block_output[116][lane] = scalar_t(0);
            block_output[117][lane] = scalar_t(0);
            block_output[118][lane] = scalar_t(0);
            block_output[119][lane] = scalar_t(0);
            block_output[120][lane] = scalar_t(0);
            block_output[121][lane] = scalar_t(0);
            block_output[122][lane] = scalar_t(0);
            block_output[123][lane] = scalar_t(0);
            block_output[124][lane] = scalar_t(0);
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[81], block_coordinates[82], block_coordinates[83], block_coordinates[84], block_coordinates[85], block_coordinates[86], block_coordinates[87], block_coordinates[88], block_coordinates[89], block_coordinates[90], block_coordinates[91], block_coordinates[92], block_coordinates[93], block_coordinates[94], block_coordinates[95], block_coordinates[96], block_coordinates[97], block_coordinates[98], block_coordinates[99], block_coordinates[100], block_coordinates[101], block_coordinates[102], block_coordinates[103], block_coordinates[104], block_coordinates[105], block_coordinates[106], block_coordinates[107], block_coordinates[108], block_coordinates[109], block_coordinates[110], block_coordinates[111], block_coordinates[112], block_coordinates[113], block_coordinates[114], block_coordinates[115], block_coordinates[116], block_coordinates[117], block_coordinates[118], block_coordinates[119], block_coordinates[120], block_coordinates[121], block_coordinates[122], block_coordinates[123], block_coordinates[124], block_coordinates[125], block_coordinates[126], block_coordinates[127], block_coordinates[128], block_coordinates[129], block_coordinates[130], block_coordinates[131], block_coordinates[132], block_coordinates[133], block_coordinates[134], block_coordinates[135], block_coordinates[136], block_coordinates[137], block_coordinates[138], block_coordinates[139], block_coordinates[140], block_coordinates[141], block_coordinates[142], block_coordinates[143], block_coordinates[144], block_coordinates[145], block_coordinates[146], block_coordinates[147], block_coordinates[148], block_coordinates[149], block_coordinates[150], block_coordinates[151], block_coordinates[152], block_coordinates[153], block_coordinates[154], block_coordinates[155], block_coordinates[156], block_coordinates[157], block_coordinates[158], block_coordinates[159], block_coordinates[160], block_coordinates[161], block_coordinates[162], block_coordinates[163], block_coordinates[164], block_coordinates[165], block_coordinates[166], block_coordinates[167], block_coordinates[168], block_coordinates[169], block_coordinates[170], block_coordinates[171], block_coordinates[172], block_coordinates[173], block_coordinates[174], block_coordinates[175], block_coordinates[176], block_coordinates[177], block_coordinates[178], block_coordinates[179], block_coordinates[180], block_coordinates[181], block_coordinates[182], block_coordinates[183], block_coordinates[184], block_coordinates[185], block_coordinates[186], block_coordinates[187], block_coordinates[188], block_coordinates[189], block_coordinates[190], block_coordinates[191], block_coordinates[192], block_coordinates[193], block_coordinates[194], block_coordinates[195], block_coordinates[196], block_coordinates[197], block_coordinates[198], block_coordinates[199], block_coordinates[200], block_coordinates[201], block_coordinates[202], block_coordinates[203], block_coordinates[204], block_coordinates[205], block_coordinates[206], block_coordinates[207], block_coordinates[208], block_coordinates[209], block_coordinates[210], block_coordinates[211], block_coordinates[212], block_coordinates[213], block_coordinates[214], block_coordinates[215], block_coordinates[216], block_coordinates[217], block_coordinates[218], block_coordinates[219], block_coordinates[220], block_coordinates[221], block_coordinates[222], block_coordinates[223], block_coordinates[224], block_coordinates[225], block_coordinates[226], block_coordinates[227], block_coordinates[228], block_coordinates[229], block_coordinates[230], block_coordinates[231], block_coordinates[232], block_coordinates[233], block_coordinates[234], block_coordinates[235], block_coordinates[236], block_coordinates[237], block_coordinates[238], block_coordinates[239], block_coordinates[240], block_coordinates[241], block_coordinates[242], block_coordinates[243], block_coordinates[244], block_coordinates[245], block_coordinates[246], block_coordinates[247], block_coordinates[248], block_coordinates[249], block_coordinates[250], block_coordinates[251], block_coordinates[252], block_coordinates[253], block_coordinates[254], block_coordinates[255], block_coordinates[256], block_coordinates[257], block_coordinates[258], block_coordinates[259], block_coordinates[260], block_coordinates[261], block_coordinates[262], block_coordinates[263], block_coordinates[264], block_coordinates[265], block_coordinates[266], block_coordinates[267], block_coordinates[268], block_coordinates[269], block_coordinates[270], block_coordinates[271], block_coordinates[272], block_coordinates[273], block_coordinates[274], block_coordinates[275], block_coordinates[276], block_coordinates[277], block_coordinates[278], block_coordinates[279], block_coordinates[280], block_coordinates[281], block_coordinates[282], block_coordinates[283], block_coordinates[284], block_coordinates[285], block_coordinates[286], block_coordinates[287], block_coordinates[288], block_coordinates[289], block_coordinates[290], block_coordinates[291], block_coordinates[292], block_coordinates[293], block_coordinates[294], block_coordinates[295], block_coordinates[296], block_coordinates[297], block_coordinates[298], block_coordinates[299], block_coordinates[300], block_coordinates[301], block_coordinates[302], block_coordinates[303], block_coordinates[304], block_coordinates[305], block_coordinates[306], block_coordinates[307], block_coordinates[308], block_coordinates[309], block_coordinates[310], block_coordinates[311], block_coordinates[312], block_coordinates[313], block_coordinates[314], block_coordinates[315], block_coordinates[316], block_coordinates[317], block_coordinates[318], block_coordinates[319], block_coordinates[320], block_coordinates[321], block_coordinates[322], block_coordinates[323], block_coordinates[324], block_coordinates[325], block_coordinates[326], block_coordinates[327], block_coordinates[328], block_coordinates[329], block_coordinates[330], block_coordinates[331], block_coordinates[332], block_coordinates[333], block_coordinates[334], block_coordinates[335], block_coordinates[336], block_coordinates[337], block_coordinates[338], block_coordinates[339], block_coordinates[340], block_coordinates[341], block_coordinates[342], block_coordinates[343], block_coordinates[344], block_coordinates[345], block_coordinates[346], block_coordinates[347], block_coordinates[348], block_coordinates[349], block_coordinates[350], block_coordinates[351], block_coordinates[352], block_coordinates[353], block_coordinates[354], block_coordinates[355], block_coordinates[356], block_coordinates[357], block_coordinates[358], block_coordinates[359], block_coordinates[360], block_coordinates[361], block_coordinates[362], block_coordinates[363], block_coordinates[364], block_coordinates[365], block_coordinates[366], block_coordinates[367], block_coordinates[368], block_coordinates[369], block_coordinates[370], block_coordinates[371], block_coordinates[372], block_coordinates[373], block_coordinates[374]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33], block_current[34], block_current[35], block_current[36], block_current[37], block_current[38], block_current[39], block_current[40], block_current[41], block_current[42], block_current[43], block_current[44], block_current[45], block_current[46], block_current[47], block_current[48], block_current[49], block_current[50], block_current[51], block_current[52], block_current[53], block_current[54], block_current[55], block_current[56], block_current[57], block_current[58], block_current[59], block_current[60], block_current[61], block_current[62], block_current[63], block_current[64], block_current[65], block_current[66], block_current[67], block_current[68], block_current[69], block_current[70], block_current[71], block_current[72], block_current[73], block_current[74], block_current[75], block_current[76], block_current[77], block_current[78], block_current[79], block_current[80], block_current[81], block_current[82], block_current[83], block_current[84], block_current[85], block_current[86], block_current[87], block_current[88], block_current[89], block_current[90], block_current[91], block_current[92], block_current[93], block_current[94], block_current[95], block_current[96], block_current[97], block_current[98], block_current[99], block_current[100], block_current[101], block_current[102], block_current[103], block_current[104], block_current[105], block_current[106], block_current[107], block_current[108], block_current[109], block_current[110], block_current[111], block_current[112], block_current[113], block_current[114], block_current[115], block_current[116], block_current[117], block_current[118], block_current[119], block_current[120], block_current[121], block_current[122], block_current[123], block_current[124]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63], block_output[64], block_output[65], block_output[66], block_output[67], block_output[68], block_output[69], block_output[70], block_output[71], block_output[72], block_output[73], block_output[74], block_output[75], block_output[76], block_output[77], block_output[78], block_output[79], block_output[80], block_output[81], block_output[82], block_output[83], block_output[84], block_output[85], block_output[86], block_output[87], block_output[88], block_output[89], block_output[90], block_output[91], block_output[92], block_output[93], block_output[94], block_output[95], block_output[96], block_output[97], block_output[98], block_output[99], block_output[100], block_output[101], block_output[102], block_output[103], block_output[104], block_output[105], block_output[106], block_output[107], block_output[108], block_output[109], block_output[110], block_output[111], block_output[112], block_output[113], block_output[114], block_output[115], block_output[116], block_output[117], block_output[118], block_output[119], block_output[120], block_output[121], block_output[122], block_output[123], block_output[124]};
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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 27] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 28] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 29] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 30] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 31] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 32] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 33] * out_stride] += block_output[33][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 34] * out_stride] += block_output[34][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 35] * out_stride] += block_output[35][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 36] * out_stride] += block_output[36][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 37] * out_stride] += block_output[37][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 38] * out_stride] += block_output[38][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 39] * out_stride] += block_output[39][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 40] * out_stride] += block_output[40][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 41] * out_stride] += block_output[41][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 42] * out_stride] += block_output[42][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 43] * out_stride] += block_output[43][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 44] * out_stride] += block_output[44][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 45] * out_stride] += block_output[45][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 46] * out_stride] += block_output[46][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 47] * out_stride] += block_output[47][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 48] * out_stride] += block_output[48][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 49] * out_stride] += block_output[49][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 50] * out_stride] += block_output[50][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 51] * out_stride] += block_output[51][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 52] * out_stride] += block_output[52][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 53] * out_stride] += block_output[53][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 54] * out_stride] += block_output[54][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 55] * out_stride] += block_output[55][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 56] * out_stride] += block_output[56][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 57] * out_stride] += block_output[57][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 58] * out_stride] += block_output[58][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 59] * out_stride] += block_output[59][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 60] * out_stride] += block_output[60][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 61] * out_stride] += block_output[61][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 62] * out_stride] += block_output[62][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 63] * out_stride] += block_output[63][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 64] * out_stride] += block_output[64][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 65] * out_stride] += block_output[65][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 66] * out_stride] += block_output[66][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 67] * out_stride] += block_output[67][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 68] * out_stride] += block_output[68][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 69] * out_stride] += block_output[69][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 70] * out_stride] += block_output[70][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 71] * out_stride] += block_output[71][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 72] * out_stride] += block_output[72][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 73] * out_stride] += block_output[73][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 74] * out_stride] += block_output[74][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 75] * out_stride] += block_output[75][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 76] * out_stride] += block_output[76][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 77] * out_stride] += block_output[77][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 78] * out_stride] += block_output[78][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 79] * out_stride] += block_output[79][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 80] * out_stride] += block_output[80][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 81] * out_stride] += block_output[81][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 82] * out_stride] += block_output[82][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 83] * out_stride] += block_output[83][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 84] * out_stride] += block_output[84][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 85] * out_stride] += block_output[85][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 86] * out_stride] += block_output[86][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 87] * out_stride] += block_output[87][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 88] * out_stride] += block_output[88][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 89] * out_stride] += block_output[89][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 90] * out_stride] += block_output[90][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 91] * out_stride] += block_output[91][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 92] * out_stride] += block_output[92][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 93] * out_stride] += block_output[93][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 94] * out_stride] += block_output[94][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 95] * out_stride] += block_output[95][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 96] * out_stride] += block_output[96][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 97] * out_stride] += block_output[97][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 98] * out_stride] += block_output[98][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 99] * out_stride] += block_output[99][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 100] * out_stride] += block_output[100][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 101] * out_stride] += block_output[101][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 102] * out_stride] += block_output[102][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 103] * out_stride] += block_output[103][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 104] * out_stride] += block_output[104][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 105] * out_stride] += block_output[105][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 106] * out_stride] += block_output[106][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 107] * out_stride] += block_output[107][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 108] * out_stride] += block_output[108][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 109] * out_stride] += block_output[109][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 110] * out_stride] += block_output[110][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 111] * out_stride] += block_output[111][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 112] * out_stride] += block_output[112][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 113] * out_stride] += block_output[113][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 114] * out_stride] += block_output[114][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 115] * out_stride] += block_output[115][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 116] * out_stride] += block_output[116][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 117] * out_stride] += block_output[117][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 118] * out_stride] += block_output[118][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 119] * out_stride] += block_output[119][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 120] * out_stride] += block_output[120][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 121] * out_stride] += block_output[121][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 122] * out_stride] += block_output[122][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 123] * out_stride] += block_output[123][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 124] * out_stride] += block_output[124][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex125_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex125_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex125_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex125_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex125_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex125_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT direction[125],
        const double kappa,
        double *const SFEM_RESTRICT output[125]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<double, 125, 125, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<double>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex125_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT direction[125],
        const float kappa,
        float *const SFEM_RESTRICT output[125]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<float, 125, 125, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<float>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex125_jacobian_action_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 125;
    static constexpr int N_SHAPE = 125;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex125_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex125_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex125_affine_reference_data<scalar_t>::q_weight_1d();

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
            ev[lane * N_SHAPE + 27] = elements[27][evbegin + lane];
            ev[lane * N_SHAPE + 28] = elements[28][evbegin + lane];
            ev[lane * N_SHAPE + 29] = elements[29][evbegin + lane];
            ev[lane * N_SHAPE + 30] = elements[30][evbegin + lane];
            ev[lane * N_SHAPE + 31] = elements[31][evbegin + lane];
            ev[lane * N_SHAPE + 32] = elements[32][evbegin + lane];
            ev[lane * N_SHAPE + 33] = elements[33][evbegin + lane];
            ev[lane * N_SHAPE + 34] = elements[34][evbegin + lane];
            ev[lane * N_SHAPE + 35] = elements[35][evbegin + lane];
            ev[lane * N_SHAPE + 36] = elements[36][evbegin + lane];
            ev[lane * N_SHAPE + 37] = elements[37][evbegin + lane];
            ev[lane * N_SHAPE + 38] = elements[38][evbegin + lane];
            ev[lane * N_SHAPE + 39] = elements[39][evbegin + lane];
            ev[lane * N_SHAPE + 40] = elements[40][evbegin + lane];
            ev[lane * N_SHAPE + 41] = elements[41][evbegin + lane];
            ev[lane * N_SHAPE + 42] = elements[42][evbegin + lane];
            ev[lane * N_SHAPE + 43] = elements[43][evbegin + lane];
            ev[lane * N_SHAPE + 44] = elements[44][evbegin + lane];
            ev[lane * N_SHAPE + 45] = elements[45][evbegin + lane];
            ev[lane * N_SHAPE + 46] = elements[46][evbegin + lane];
            ev[lane * N_SHAPE + 47] = elements[47][evbegin + lane];
            ev[lane * N_SHAPE + 48] = elements[48][evbegin + lane];
            ev[lane * N_SHAPE + 49] = elements[49][evbegin + lane];
            ev[lane * N_SHAPE + 50] = elements[50][evbegin + lane];
            ev[lane * N_SHAPE + 51] = elements[51][evbegin + lane];
            ev[lane * N_SHAPE + 52] = elements[52][evbegin + lane];
            ev[lane * N_SHAPE + 53] = elements[53][evbegin + lane];
            ev[lane * N_SHAPE + 54] = elements[54][evbegin + lane];
            ev[lane * N_SHAPE + 55] = elements[55][evbegin + lane];
            ev[lane * N_SHAPE + 56] = elements[56][evbegin + lane];
            ev[lane * N_SHAPE + 57] = elements[57][evbegin + lane];
            ev[lane * N_SHAPE + 58] = elements[58][evbegin + lane];
            ev[lane * N_SHAPE + 59] = elements[59][evbegin + lane];
            ev[lane * N_SHAPE + 60] = elements[60][evbegin + lane];
            ev[lane * N_SHAPE + 61] = elements[61][evbegin + lane];
            ev[lane * N_SHAPE + 62] = elements[62][evbegin + lane];
            ev[lane * N_SHAPE + 63] = elements[63][evbegin + lane];
            ev[lane * N_SHAPE + 64] = elements[64][evbegin + lane];
            ev[lane * N_SHAPE + 65] = elements[65][evbegin + lane];
            ev[lane * N_SHAPE + 66] = elements[66][evbegin + lane];
            ev[lane * N_SHAPE + 67] = elements[67][evbegin + lane];
            ev[lane * N_SHAPE + 68] = elements[68][evbegin + lane];
            ev[lane * N_SHAPE + 69] = elements[69][evbegin + lane];
            ev[lane * N_SHAPE + 70] = elements[70][evbegin + lane];
            ev[lane * N_SHAPE + 71] = elements[71][evbegin + lane];
            ev[lane * N_SHAPE + 72] = elements[72][evbegin + lane];
            ev[lane * N_SHAPE + 73] = elements[73][evbegin + lane];
            ev[lane * N_SHAPE + 74] = elements[74][evbegin + lane];
            ev[lane * N_SHAPE + 75] = elements[75][evbegin + lane];
            ev[lane * N_SHAPE + 76] = elements[76][evbegin + lane];
            ev[lane * N_SHAPE + 77] = elements[77][evbegin + lane];
            ev[lane * N_SHAPE + 78] = elements[78][evbegin + lane];
            ev[lane * N_SHAPE + 79] = elements[79][evbegin + lane];
            ev[lane * N_SHAPE + 80] = elements[80][evbegin + lane];
            ev[lane * N_SHAPE + 81] = elements[81][evbegin + lane];
            ev[lane * N_SHAPE + 82] = elements[82][evbegin + lane];
            ev[lane * N_SHAPE + 83] = elements[83][evbegin + lane];
            ev[lane * N_SHAPE + 84] = elements[84][evbegin + lane];
            ev[lane * N_SHAPE + 85] = elements[85][evbegin + lane];
            ev[lane * N_SHAPE + 86] = elements[86][evbegin + lane];
            ev[lane * N_SHAPE + 87] = elements[87][evbegin + lane];
            ev[lane * N_SHAPE + 88] = elements[88][evbegin + lane];
            ev[lane * N_SHAPE + 89] = elements[89][evbegin + lane];
            ev[lane * N_SHAPE + 90] = elements[90][evbegin + lane];
            ev[lane * N_SHAPE + 91] = elements[91][evbegin + lane];
            ev[lane * N_SHAPE + 92] = elements[92][evbegin + lane];
            ev[lane * N_SHAPE + 93] = elements[93][evbegin + lane];
            ev[lane * N_SHAPE + 94] = elements[94][evbegin + lane];
            ev[lane * N_SHAPE + 95] = elements[95][evbegin + lane];
            ev[lane * N_SHAPE + 96] = elements[96][evbegin + lane];
            ev[lane * N_SHAPE + 97] = elements[97][evbegin + lane];
            ev[lane * N_SHAPE + 98] = elements[98][evbegin + lane];
            ev[lane * N_SHAPE + 99] = elements[99][evbegin + lane];
            ev[lane * N_SHAPE + 100] = elements[100][evbegin + lane];
            ev[lane * N_SHAPE + 101] = elements[101][evbegin + lane];
            ev[lane * N_SHAPE + 102] = elements[102][evbegin + lane];
            ev[lane * N_SHAPE + 103] = elements[103][evbegin + lane];
            ev[lane * N_SHAPE + 104] = elements[104][evbegin + lane];
            ev[lane * N_SHAPE + 105] = elements[105][evbegin + lane];
            ev[lane * N_SHAPE + 106] = elements[106][evbegin + lane];
            ev[lane * N_SHAPE + 107] = elements[107][evbegin + lane];
            ev[lane * N_SHAPE + 108] = elements[108][evbegin + lane];
            ev[lane * N_SHAPE + 109] = elements[109][evbegin + lane];
            ev[lane * N_SHAPE + 110] = elements[110][evbegin + lane];
            ev[lane * N_SHAPE + 111] = elements[111][evbegin + lane];
            ev[lane * N_SHAPE + 112] = elements[112][evbegin + lane];
            ev[lane * N_SHAPE + 113] = elements[113][evbegin + lane];
            ev[lane * N_SHAPE + 114] = elements[114][evbegin + lane];
            ev[lane * N_SHAPE + 115] = elements[115][evbegin + lane];
            ev[lane * N_SHAPE + 116] = elements[116][evbegin + lane];
            ev[lane * N_SHAPE + 117] = elements[117][evbegin + lane];
            ev[lane * N_SHAPE + 118] = elements[118][evbegin + lane];
            ev[lane * N_SHAPE + 119] = elements[119][evbegin + lane];
            ev[lane * N_SHAPE + 120] = elements[120][evbegin + lane];
            ev[lane * N_SHAPE + 121] = elements[121][evbegin + lane];
            ev[lane * N_SHAPE + 122] = elements[122][evbegin + lane];
            ev[lane * N_SHAPE + 123] = elements[123][evbegin + lane];
            ev[lane * N_SHAPE + 124] = elements[124][evbegin + lane];
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
            block_direction[27][lane] = u_direction[ev[lane * N_SHAPE + 27] * direction_stride];
            block_direction[28][lane] = u_direction[ev[lane * N_SHAPE + 28] * direction_stride];
            block_direction[29][lane] = u_direction[ev[lane * N_SHAPE + 29] * direction_stride];
            block_direction[30][lane] = u_direction[ev[lane * N_SHAPE + 30] * direction_stride];
            block_direction[31][lane] = u_direction[ev[lane * N_SHAPE + 31] * direction_stride];
            block_direction[32][lane] = u_direction[ev[lane * N_SHAPE + 32] * direction_stride];
            block_direction[33][lane] = u_direction[ev[lane * N_SHAPE + 33] * direction_stride];
            block_direction[34][lane] = u_direction[ev[lane * N_SHAPE + 34] * direction_stride];
            block_direction[35][lane] = u_direction[ev[lane * N_SHAPE + 35] * direction_stride];
            block_direction[36][lane] = u_direction[ev[lane * N_SHAPE + 36] * direction_stride];
            block_direction[37][lane] = u_direction[ev[lane * N_SHAPE + 37] * direction_stride];
            block_direction[38][lane] = u_direction[ev[lane * N_SHAPE + 38] * direction_stride];
            block_direction[39][lane] = u_direction[ev[lane * N_SHAPE + 39] * direction_stride];
            block_direction[40][lane] = u_direction[ev[lane * N_SHAPE + 40] * direction_stride];
            block_direction[41][lane] = u_direction[ev[lane * N_SHAPE + 41] * direction_stride];
            block_direction[42][lane] = u_direction[ev[lane * N_SHAPE + 42] * direction_stride];
            block_direction[43][lane] = u_direction[ev[lane * N_SHAPE + 43] * direction_stride];
            block_direction[44][lane] = u_direction[ev[lane * N_SHAPE + 44] * direction_stride];
            block_direction[45][lane] = u_direction[ev[lane * N_SHAPE + 45] * direction_stride];
            block_direction[46][lane] = u_direction[ev[lane * N_SHAPE + 46] * direction_stride];
            block_direction[47][lane] = u_direction[ev[lane * N_SHAPE + 47] * direction_stride];
            block_direction[48][lane] = u_direction[ev[lane * N_SHAPE + 48] * direction_stride];
            block_direction[49][lane] = u_direction[ev[lane * N_SHAPE + 49] * direction_stride];
            block_direction[50][lane] = u_direction[ev[lane * N_SHAPE + 50] * direction_stride];
            block_direction[51][lane] = u_direction[ev[lane * N_SHAPE + 51] * direction_stride];
            block_direction[52][lane] = u_direction[ev[lane * N_SHAPE + 52] * direction_stride];
            block_direction[53][lane] = u_direction[ev[lane * N_SHAPE + 53] * direction_stride];
            block_direction[54][lane] = u_direction[ev[lane * N_SHAPE + 54] * direction_stride];
            block_direction[55][lane] = u_direction[ev[lane * N_SHAPE + 55] * direction_stride];
            block_direction[56][lane] = u_direction[ev[lane * N_SHAPE + 56] * direction_stride];
            block_direction[57][lane] = u_direction[ev[lane * N_SHAPE + 57] * direction_stride];
            block_direction[58][lane] = u_direction[ev[lane * N_SHAPE + 58] * direction_stride];
            block_direction[59][lane] = u_direction[ev[lane * N_SHAPE + 59] * direction_stride];
            block_direction[60][lane] = u_direction[ev[lane * N_SHAPE + 60] * direction_stride];
            block_direction[61][lane] = u_direction[ev[lane * N_SHAPE + 61] * direction_stride];
            block_direction[62][lane] = u_direction[ev[lane * N_SHAPE + 62] * direction_stride];
            block_direction[63][lane] = u_direction[ev[lane * N_SHAPE + 63] * direction_stride];
            block_direction[64][lane] = u_direction[ev[lane * N_SHAPE + 64] * direction_stride];
            block_direction[65][lane] = u_direction[ev[lane * N_SHAPE + 65] * direction_stride];
            block_direction[66][lane] = u_direction[ev[lane * N_SHAPE + 66] * direction_stride];
            block_direction[67][lane] = u_direction[ev[lane * N_SHAPE + 67] * direction_stride];
            block_direction[68][lane] = u_direction[ev[lane * N_SHAPE + 68] * direction_stride];
            block_direction[69][lane] = u_direction[ev[lane * N_SHAPE + 69] * direction_stride];
            block_direction[70][lane] = u_direction[ev[lane * N_SHAPE + 70] * direction_stride];
            block_direction[71][lane] = u_direction[ev[lane * N_SHAPE + 71] * direction_stride];
            block_direction[72][lane] = u_direction[ev[lane * N_SHAPE + 72] * direction_stride];
            block_direction[73][lane] = u_direction[ev[lane * N_SHAPE + 73] * direction_stride];
            block_direction[74][lane] = u_direction[ev[lane * N_SHAPE + 74] * direction_stride];
            block_direction[75][lane] = u_direction[ev[lane * N_SHAPE + 75] * direction_stride];
            block_direction[76][lane] = u_direction[ev[lane * N_SHAPE + 76] * direction_stride];
            block_direction[77][lane] = u_direction[ev[lane * N_SHAPE + 77] * direction_stride];
            block_direction[78][lane] = u_direction[ev[lane * N_SHAPE + 78] * direction_stride];
            block_direction[79][lane] = u_direction[ev[lane * N_SHAPE + 79] * direction_stride];
            block_direction[80][lane] = u_direction[ev[lane * N_SHAPE + 80] * direction_stride];
            block_direction[81][lane] = u_direction[ev[lane * N_SHAPE + 81] * direction_stride];
            block_direction[82][lane] = u_direction[ev[lane * N_SHAPE + 82] * direction_stride];
            block_direction[83][lane] = u_direction[ev[lane * N_SHAPE + 83] * direction_stride];
            block_direction[84][lane] = u_direction[ev[lane * N_SHAPE + 84] * direction_stride];
            block_direction[85][lane] = u_direction[ev[lane * N_SHAPE + 85] * direction_stride];
            block_direction[86][lane] = u_direction[ev[lane * N_SHAPE + 86] * direction_stride];
            block_direction[87][lane] = u_direction[ev[lane * N_SHAPE + 87] * direction_stride];
            block_direction[88][lane] = u_direction[ev[lane * N_SHAPE + 88] * direction_stride];
            block_direction[89][lane] = u_direction[ev[lane * N_SHAPE + 89] * direction_stride];
            block_direction[90][lane] = u_direction[ev[lane * N_SHAPE + 90] * direction_stride];
            block_direction[91][lane] = u_direction[ev[lane * N_SHAPE + 91] * direction_stride];
            block_direction[92][lane] = u_direction[ev[lane * N_SHAPE + 92] * direction_stride];
            block_direction[93][lane] = u_direction[ev[lane * N_SHAPE + 93] * direction_stride];
            block_direction[94][lane] = u_direction[ev[lane * N_SHAPE + 94] * direction_stride];
            block_direction[95][lane] = u_direction[ev[lane * N_SHAPE + 95] * direction_stride];
            block_direction[96][lane] = u_direction[ev[lane * N_SHAPE + 96] * direction_stride];
            block_direction[97][lane] = u_direction[ev[lane * N_SHAPE + 97] * direction_stride];
            block_direction[98][lane] = u_direction[ev[lane * N_SHAPE + 98] * direction_stride];
            block_direction[99][lane] = u_direction[ev[lane * N_SHAPE + 99] * direction_stride];
            block_direction[100][lane] = u_direction[ev[lane * N_SHAPE + 100] * direction_stride];
            block_direction[101][lane] = u_direction[ev[lane * N_SHAPE + 101] * direction_stride];
            block_direction[102][lane] = u_direction[ev[lane * N_SHAPE + 102] * direction_stride];
            block_direction[103][lane] = u_direction[ev[lane * N_SHAPE + 103] * direction_stride];
            block_direction[104][lane] = u_direction[ev[lane * N_SHAPE + 104] * direction_stride];
            block_direction[105][lane] = u_direction[ev[lane * N_SHAPE + 105] * direction_stride];
            block_direction[106][lane] = u_direction[ev[lane * N_SHAPE + 106] * direction_stride];
            block_direction[107][lane] = u_direction[ev[lane * N_SHAPE + 107] * direction_stride];
            block_direction[108][lane] = u_direction[ev[lane * N_SHAPE + 108] * direction_stride];
            block_direction[109][lane] = u_direction[ev[lane * N_SHAPE + 109] * direction_stride];
            block_direction[110][lane] = u_direction[ev[lane * N_SHAPE + 110] * direction_stride];
            block_direction[111][lane] = u_direction[ev[lane * N_SHAPE + 111] * direction_stride];
            block_direction[112][lane] = u_direction[ev[lane * N_SHAPE + 112] * direction_stride];
            block_direction[113][lane] = u_direction[ev[lane * N_SHAPE + 113] * direction_stride];
            block_direction[114][lane] = u_direction[ev[lane * N_SHAPE + 114] * direction_stride];
            block_direction[115][lane] = u_direction[ev[lane * N_SHAPE + 115] * direction_stride];
            block_direction[116][lane] = u_direction[ev[lane * N_SHAPE + 116] * direction_stride];
            block_direction[117][lane] = u_direction[ev[lane * N_SHAPE + 117] * direction_stride];
            block_direction[118][lane] = u_direction[ev[lane * N_SHAPE + 118] * direction_stride];
            block_direction[119][lane] = u_direction[ev[lane * N_SHAPE + 119] * direction_stride];
            block_direction[120][lane] = u_direction[ev[lane * N_SHAPE + 120] * direction_stride];
            block_direction[121][lane] = u_direction[ev[lane * N_SHAPE + 121] * direction_stride];
            block_direction[122][lane] = u_direction[ev[lane * N_SHAPE + 122] * direction_stride];
            block_direction[123][lane] = u_direction[ev[lane * N_SHAPE + 123] * direction_stride];
            block_direction[124][lane] = u_direction[ev[lane * N_SHAPE + 124] * direction_stride];
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
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
            block_output[34][lane] = scalar_t(0);
            block_output[35][lane] = scalar_t(0);
            block_output[36][lane] = scalar_t(0);
            block_output[37][lane] = scalar_t(0);
            block_output[38][lane] = scalar_t(0);
            block_output[39][lane] = scalar_t(0);
            block_output[40][lane] = scalar_t(0);
            block_output[41][lane] = scalar_t(0);
            block_output[42][lane] = scalar_t(0);
            block_output[43][lane] = scalar_t(0);
            block_output[44][lane] = scalar_t(0);
            block_output[45][lane] = scalar_t(0);
            block_output[46][lane] = scalar_t(0);
            block_output[47][lane] = scalar_t(0);
            block_output[48][lane] = scalar_t(0);
            block_output[49][lane] = scalar_t(0);
            block_output[50][lane] = scalar_t(0);
            block_output[51][lane] = scalar_t(0);
            block_output[52][lane] = scalar_t(0);
            block_output[53][lane] = scalar_t(0);
            block_output[54][lane] = scalar_t(0);
            block_output[55][lane] = scalar_t(0);
            block_output[56][lane] = scalar_t(0);
            block_output[57][lane] = scalar_t(0);
            block_output[58][lane] = scalar_t(0);
            block_output[59][lane] = scalar_t(0);
            block_output[60][lane] = scalar_t(0);
            block_output[61][lane] = scalar_t(0);
            block_output[62][lane] = scalar_t(0);
            block_output[63][lane] = scalar_t(0);
            block_output[64][lane] = scalar_t(0);
            block_output[65][lane] = scalar_t(0);
            block_output[66][lane] = scalar_t(0);
            block_output[67][lane] = scalar_t(0);
            block_output[68][lane] = scalar_t(0);
            block_output[69][lane] = scalar_t(0);
            block_output[70][lane] = scalar_t(0);
            block_output[71][lane] = scalar_t(0);
            block_output[72][lane] = scalar_t(0);
            block_output[73][lane] = scalar_t(0);
            block_output[74][lane] = scalar_t(0);
            block_output[75][lane] = scalar_t(0);
            block_output[76][lane] = scalar_t(0);
            block_output[77][lane] = scalar_t(0);
            block_output[78][lane] = scalar_t(0);
            block_output[79][lane] = scalar_t(0);
            block_output[80][lane] = scalar_t(0);
            block_output[81][lane] = scalar_t(0);
            block_output[82][lane] = scalar_t(0);
            block_output[83][lane] = scalar_t(0);
            block_output[84][lane] = scalar_t(0);
            block_output[85][lane] = scalar_t(0);
            block_output[86][lane] = scalar_t(0);
            block_output[87][lane] = scalar_t(0);
            block_output[88][lane] = scalar_t(0);
            block_output[89][lane] = scalar_t(0);
            block_output[90][lane] = scalar_t(0);
            block_output[91][lane] = scalar_t(0);
            block_output[92][lane] = scalar_t(0);
            block_output[93][lane] = scalar_t(0);
            block_output[94][lane] = scalar_t(0);
            block_output[95][lane] = scalar_t(0);
            block_output[96][lane] = scalar_t(0);
            block_output[97][lane] = scalar_t(0);
            block_output[98][lane] = scalar_t(0);
            block_output[99][lane] = scalar_t(0);
            block_output[100][lane] = scalar_t(0);
            block_output[101][lane] = scalar_t(0);
            block_output[102][lane] = scalar_t(0);
            block_output[103][lane] = scalar_t(0);
            block_output[104][lane] = scalar_t(0);
            block_output[105][lane] = scalar_t(0);
            block_output[106][lane] = scalar_t(0);
            block_output[107][lane] = scalar_t(0);
            block_output[108][lane] = scalar_t(0);
            block_output[109][lane] = scalar_t(0);
            block_output[110][lane] = scalar_t(0);
            block_output[111][lane] = scalar_t(0);
            block_output[112][lane] = scalar_t(0);
            block_output[113][lane] = scalar_t(0);
            block_output[114][lane] = scalar_t(0);
            block_output[115][lane] = scalar_t(0);
            block_output[116][lane] = scalar_t(0);
            block_output[117][lane] = scalar_t(0);
            block_output[118][lane] = scalar_t(0);
            block_output[119][lane] = scalar_t(0);
            block_output[120][lane] = scalar_t(0);
            block_output[121][lane] = scalar_t(0);
            block_output[122][lane] = scalar_t(0);
            block_output[123][lane] = scalar_t(0);
            block_output[124][lane] = scalar_t(0);
        }

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26], block_direction[27], block_direction[28], block_direction[29], block_direction[30], block_direction[31], block_direction[32], block_direction[33], block_direction[34], block_direction[35], block_direction[36], block_direction[37], block_direction[38], block_direction[39], block_direction[40], block_direction[41], block_direction[42], block_direction[43], block_direction[44], block_direction[45], block_direction[46], block_direction[47], block_direction[48], block_direction[49], block_direction[50], block_direction[51], block_direction[52], block_direction[53], block_direction[54], block_direction[55], block_direction[56], block_direction[57], block_direction[58], block_direction[59], block_direction[60], block_direction[61], block_direction[62], block_direction[63], block_direction[64], block_direction[65], block_direction[66], block_direction[67], block_direction[68], block_direction[69], block_direction[70], block_direction[71], block_direction[72], block_direction[73], block_direction[74], block_direction[75], block_direction[76], block_direction[77], block_direction[78], block_direction[79], block_direction[80], block_direction[81], block_direction[82], block_direction[83], block_direction[84], block_direction[85], block_direction[86], block_direction[87], block_direction[88], block_direction[89], block_direction[90], block_direction[91], block_direction[92], block_direction[93], block_direction[94], block_direction[95], block_direction[96], block_direction[97], block_direction[98], block_direction[99], block_direction[100], block_direction[101], block_direction[102], block_direction[103], block_direction[104], block_direction[105], block_direction[106], block_direction[107], block_direction[108], block_direction[109], block_direction[110], block_direction[111], block_direction[112], block_direction[113], block_direction[114], block_direction[115], block_direction[116], block_direction[117], block_direction[118], block_direction[119], block_direction[120], block_direction[121], block_direction[122], block_direction[123], block_direction[124]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63], block_output[64], block_output[65], block_output[66], block_output[67], block_output[68], block_output[69], block_output[70], block_output[71], block_output[72], block_output[73], block_output[74], block_output[75], block_output[76], block_output[77], block_output[78], block_output[79], block_output[80], block_output[81], block_output[82], block_output[83], block_output[84], block_output[85], block_output[86], block_output[87], block_output[88], block_output[89], block_output[90], block_output[91], block_output[92], block_output[93], block_output[94], block_output[95], block_output[96], block_output[97], block_output[98], block_output[99], block_output[100], block_output[101], block_output[102], block_output[103], block_output[104], block_output[105], block_output[106], block_output[107], block_output[108], block_output[109], block_output[110], block_output[111], block_output[112], block_output[113], block_output[114], block_output[115], block_output[116], block_output[117], block_output[118], block_output[119], block_output[120], block_output[121], block_output[122], block_output[123], block_output[124]};
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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 27] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 28] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 29] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 30] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 31] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 32] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 33] * out_stride] += block_output[33][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 34] * out_stride] += block_output[34][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 35] * out_stride] += block_output[35][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 36] * out_stride] += block_output[36][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 37] * out_stride] += block_output[37][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 38] * out_stride] += block_output[38][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 39] * out_stride] += block_output[39][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 40] * out_stride] += block_output[40][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 41] * out_stride] += block_output[41][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 42] * out_stride] += block_output[42][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 43] * out_stride] += block_output[43][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 44] * out_stride] += block_output[44][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 45] * out_stride] += block_output[45][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 46] * out_stride] += block_output[46][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 47] * out_stride] += block_output[47][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 48] * out_stride] += block_output[48][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 49] * out_stride] += block_output[49][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 50] * out_stride] += block_output[50][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 51] * out_stride] += block_output[51][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 52] * out_stride] += block_output[52][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 53] * out_stride] += block_output[53][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 54] * out_stride] += block_output[54][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 55] * out_stride] += block_output[55][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 56] * out_stride] += block_output[56][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 57] * out_stride] += block_output[57][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 58] * out_stride] += block_output[58][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 59] * out_stride] += block_output[59][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 60] * out_stride] += block_output[60][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 61] * out_stride] += block_output[61][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 62] * out_stride] += block_output[62][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 63] * out_stride] += block_output[63][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 64] * out_stride] += block_output[64][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 65] * out_stride] += block_output[65][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 66] * out_stride] += block_output[66][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 67] * out_stride] += block_output[67][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 68] * out_stride] += block_output[68][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 69] * out_stride] += block_output[69][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 70] * out_stride] += block_output[70][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 71] * out_stride] += block_output[71][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 72] * out_stride] += block_output[72][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 73] * out_stride] += block_output[73][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 74] * out_stride] += block_output[74][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 75] * out_stride] += block_output[75][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 76] * out_stride] += block_output[76][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 77] * out_stride] += block_output[77][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 78] * out_stride] += block_output[78][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 79] * out_stride] += block_output[79][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 80] * out_stride] += block_output[80][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 81] * out_stride] += block_output[81][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 82] * out_stride] += block_output[82][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 83] * out_stride] += block_output[83][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 84] * out_stride] += block_output[84][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 85] * out_stride] += block_output[85][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 86] * out_stride] += block_output[86][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 87] * out_stride] += block_output[87][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 88] * out_stride] += block_output[88][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 89] * out_stride] += block_output[89][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 90] * out_stride] += block_output[90][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 91] * out_stride] += block_output[91][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 92] * out_stride] += block_output[92][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 93] * out_stride] += block_output[93][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 94] * out_stride] += block_output[94][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 95] * out_stride] += block_output[95][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 96] * out_stride] += block_output[96][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 97] * out_stride] += block_output[97][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 98] * out_stride] += block_output[98][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 99] * out_stride] += block_output[99][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 100] * out_stride] += block_output[100][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 101] * out_stride] += block_output[101][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 102] * out_stride] += block_output[102][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 103] * out_stride] += block_output[103][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 104] * out_stride] += block_output[104][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 105] * out_stride] += block_output[105][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 106] * out_stride] += block_output[106][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 107] * out_stride] += block_output[107][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 108] * out_stride] += block_output[108][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 109] * out_stride] += block_output[109][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 110] * out_stride] += block_output[110][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 111] * out_stride] += block_output[111][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 112] * out_stride] += block_output[112][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 113] * out_stride] += block_output[113][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 114] * out_stride] += block_output[114][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 115] * out_stride] += block_output[115][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 116] * out_stride] += block_output[116][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 117] * out_stride] += block_output[117][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 118] * out_stride] += block_output[118][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 119] * out_stride] += block_output[119][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 120] * out_stride] += block_output[120][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 121] * out_stride] += block_output[121][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 122] * out_stride] += block_output[122][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 123] * out_stride] += block_output[123][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 124] * out_stride] += block_output[124][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex125_jacobian_action_affine_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex125_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex125_jacobian_action_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex125_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 125;
    static constexpr int N_SHAPE = 125;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex125_isoparametric_reference_data<scalar_t>::q_weight_1d();

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
            ev[lane * N_SHAPE + 27] = elements[27][evbegin + lane];
            ev[lane * N_SHAPE + 28] = elements[28][evbegin + lane];
            ev[lane * N_SHAPE + 29] = elements[29][evbegin + lane];
            ev[lane * N_SHAPE + 30] = elements[30][evbegin + lane];
            ev[lane * N_SHAPE + 31] = elements[31][evbegin + lane];
            ev[lane * N_SHAPE + 32] = elements[32][evbegin + lane];
            ev[lane * N_SHAPE + 33] = elements[33][evbegin + lane];
            ev[lane * N_SHAPE + 34] = elements[34][evbegin + lane];
            ev[lane * N_SHAPE + 35] = elements[35][evbegin + lane];
            ev[lane * N_SHAPE + 36] = elements[36][evbegin + lane];
            ev[lane * N_SHAPE + 37] = elements[37][evbegin + lane];
            ev[lane * N_SHAPE + 38] = elements[38][evbegin + lane];
            ev[lane * N_SHAPE + 39] = elements[39][evbegin + lane];
            ev[lane * N_SHAPE + 40] = elements[40][evbegin + lane];
            ev[lane * N_SHAPE + 41] = elements[41][evbegin + lane];
            ev[lane * N_SHAPE + 42] = elements[42][evbegin + lane];
            ev[lane * N_SHAPE + 43] = elements[43][evbegin + lane];
            ev[lane * N_SHAPE + 44] = elements[44][evbegin + lane];
            ev[lane * N_SHAPE + 45] = elements[45][evbegin + lane];
            ev[lane * N_SHAPE + 46] = elements[46][evbegin + lane];
            ev[lane * N_SHAPE + 47] = elements[47][evbegin + lane];
            ev[lane * N_SHAPE + 48] = elements[48][evbegin + lane];
            ev[lane * N_SHAPE + 49] = elements[49][evbegin + lane];
            ev[lane * N_SHAPE + 50] = elements[50][evbegin + lane];
            ev[lane * N_SHAPE + 51] = elements[51][evbegin + lane];
            ev[lane * N_SHAPE + 52] = elements[52][evbegin + lane];
            ev[lane * N_SHAPE + 53] = elements[53][evbegin + lane];
            ev[lane * N_SHAPE + 54] = elements[54][evbegin + lane];
            ev[lane * N_SHAPE + 55] = elements[55][evbegin + lane];
            ev[lane * N_SHAPE + 56] = elements[56][evbegin + lane];
            ev[lane * N_SHAPE + 57] = elements[57][evbegin + lane];
            ev[lane * N_SHAPE + 58] = elements[58][evbegin + lane];
            ev[lane * N_SHAPE + 59] = elements[59][evbegin + lane];
            ev[lane * N_SHAPE + 60] = elements[60][evbegin + lane];
            ev[lane * N_SHAPE + 61] = elements[61][evbegin + lane];
            ev[lane * N_SHAPE + 62] = elements[62][evbegin + lane];
            ev[lane * N_SHAPE + 63] = elements[63][evbegin + lane];
            ev[lane * N_SHAPE + 64] = elements[64][evbegin + lane];
            ev[lane * N_SHAPE + 65] = elements[65][evbegin + lane];
            ev[lane * N_SHAPE + 66] = elements[66][evbegin + lane];
            ev[lane * N_SHAPE + 67] = elements[67][evbegin + lane];
            ev[lane * N_SHAPE + 68] = elements[68][evbegin + lane];
            ev[lane * N_SHAPE + 69] = elements[69][evbegin + lane];
            ev[lane * N_SHAPE + 70] = elements[70][evbegin + lane];
            ev[lane * N_SHAPE + 71] = elements[71][evbegin + lane];
            ev[lane * N_SHAPE + 72] = elements[72][evbegin + lane];
            ev[lane * N_SHAPE + 73] = elements[73][evbegin + lane];
            ev[lane * N_SHAPE + 74] = elements[74][evbegin + lane];
            ev[lane * N_SHAPE + 75] = elements[75][evbegin + lane];
            ev[lane * N_SHAPE + 76] = elements[76][evbegin + lane];
            ev[lane * N_SHAPE + 77] = elements[77][evbegin + lane];
            ev[lane * N_SHAPE + 78] = elements[78][evbegin + lane];
            ev[lane * N_SHAPE + 79] = elements[79][evbegin + lane];
            ev[lane * N_SHAPE + 80] = elements[80][evbegin + lane];
            ev[lane * N_SHAPE + 81] = elements[81][evbegin + lane];
            ev[lane * N_SHAPE + 82] = elements[82][evbegin + lane];
            ev[lane * N_SHAPE + 83] = elements[83][evbegin + lane];
            ev[lane * N_SHAPE + 84] = elements[84][evbegin + lane];
            ev[lane * N_SHAPE + 85] = elements[85][evbegin + lane];
            ev[lane * N_SHAPE + 86] = elements[86][evbegin + lane];
            ev[lane * N_SHAPE + 87] = elements[87][evbegin + lane];
            ev[lane * N_SHAPE + 88] = elements[88][evbegin + lane];
            ev[lane * N_SHAPE + 89] = elements[89][evbegin + lane];
            ev[lane * N_SHAPE + 90] = elements[90][evbegin + lane];
            ev[lane * N_SHAPE + 91] = elements[91][evbegin + lane];
            ev[lane * N_SHAPE + 92] = elements[92][evbegin + lane];
            ev[lane * N_SHAPE + 93] = elements[93][evbegin + lane];
            ev[lane * N_SHAPE + 94] = elements[94][evbegin + lane];
            ev[lane * N_SHAPE + 95] = elements[95][evbegin + lane];
            ev[lane * N_SHAPE + 96] = elements[96][evbegin + lane];
            ev[lane * N_SHAPE + 97] = elements[97][evbegin + lane];
            ev[lane * N_SHAPE + 98] = elements[98][evbegin + lane];
            ev[lane * N_SHAPE + 99] = elements[99][evbegin + lane];
            ev[lane * N_SHAPE + 100] = elements[100][evbegin + lane];
            ev[lane * N_SHAPE + 101] = elements[101][evbegin + lane];
            ev[lane * N_SHAPE + 102] = elements[102][evbegin + lane];
            ev[lane * N_SHAPE + 103] = elements[103][evbegin + lane];
            ev[lane * N_SHAPE + 104] = elements[104][evbegin + lane];
            ev[lane * N_SHAPE + 105] = elements[105][evbegin + lane];
            ev[lane * N_SHAPE + 106] = elements[106][evbegin + lane];
            ev[lane * N_SHAPE + 107] = elements[107][evbegin + lane];
            ev[lane * N_SHAPE + 108] = elements[108][evbegin + lane];
            ev[lane * N_SHAPE + 109] = elements[109][evbegin + lane];
            ev[lane * N_SHAPE + 110] = elements[110][evbegin + lane];
            ev[lane * N_SHAPE + 111] = elements[111][evbegin + lane];
            ev[lane * N_SHAPE + 112] = elements[112][evbegin + lane];
            ev[lane * N_SHAPE + 113] = elements[113][evbegin + lane];
            ev[lane * N_SHAPE + 114] = elements[114][evbegin + lane];
            ev[lane * N_SHAPE + 115] = elements[115][evbegin + lane];
            ev[lane * N_SHAPE + 116] = elements[116][evbegin + lane];
            ev[lane * N_SHAPE + 117] = elements[117][evbegin + lane];
            ev[lane * N_SHAPE + 118] = elements[118][evbegin + lane];
            ev[lane * N_SHAPE + 119] = elements[119][evbegin + lane];
            ev[lane * N_SHAPE + 120] = elements[120][evbegin + lane];
            ev[lane * N_SHAPE + 121] = elements[121][evbegin + lane];
            ev[lane * N_SHAPE + 122] = elements[122][evbegin + lane];
            ev[lane * N_SHAPE + 123] = elements[123][evbegin + lane];
            ev[lane * N_SHAPE + 124] = elements[124][evbegin + lane];
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
            block_coordinates[81][lane] = points[0][ev[lane * N_SHAPE + 27]];
            block_coordinates[82][lane] = points[1][ev[lane * N_SHAPE + 27]];
            block_coordinates[83][lane] = points[2][ev[lane * N_SHAPE + 27]];
            block_direction[27][lane] = u_direction[ev[lane * N_SHAPE + 27] * direction_stride];
            block_coordinates[84][lane] = points[0][ev[lane * N_SHAPE + 28]];
            block_coordinates[85][lane] = points[1][ev[lane * N_SHAPE + 28]];
            block_coordinates[86][lane] = points[2][ev[lane * N_SHAPE + 28]];
            block_direction[28][lane] = u_direction[ev[lane * N_SHAPE + 28] * direction_stride];
            block_coordinates[87][lane] = points[0][ev[lane * N_SHAPE + 29]];
            block_coordinates[88][lane] = points[1][ev[lane * N_SHAPE + 29]];
            block_coordinates[89][lane] = points[2][ev[lane * N_SHAPE + 29]];
            block_direction[29][lane] = u_direction[ev[lane * N_SHAPE + 29] * direction_stride];
            block_coordinates[90][lane] = points[0][ev[lane * N_SHAPE + 30]];
            block_coordinates[91][lane] = points[1][ev[lane * N_SHAPE + 30]];
            block_coordinates[92][lane] = points[2][ev[lane * N_SHAPE + 30]];
            block_direction[30][lane] = u_direction[ev[lane * N_SHAPE + 30] * direction_stride];
            block_coordinates[93][lane] = points[0][ev[lane * N_SHAPE + 31]];
            block_coordinates[94][lane] = points[1][ev[lane * N_SHAPE + 31]];
            block_coordinates[95][lane] = points[2][ev[lane * N_SHAPE + 31]];
            block_direction[31][lane] = u_direction[ev[lane * N_SHAPE + 31] * direction_stride];
            block_coordinates[96][lane] = points[0][ev[lane * N_SHAPE + 32]];
            block_coordinates[97][lane] = points[1][ev[lane * N_SHAPE + 32]];
            block_coordinates[98][lane] = points[2][ev[lane * N_SHAPE + 32]];
            block_direction[32][lane] = u_direction[ev[lane * N_SHAPE + 32] * direction_stride];
            block_coordinates[99][lane] = points[0][ev[lane * N_SHAPE + 33]];
            block_coordinates[100][lane] = points[1][ev[lane * N_SHAPE + 33]];
            block_coordinates[101][lane] = points[2][ev[lane * N_SHAPE + 33]];
            block_direction[33][lane] = u_direction[ev[lane * N_SHAPE + 33] * direction_stride];
            block_coordinates[102][lane] = points[0][ev[lane * N_SHAPE + 34]];
            block_coordinates[103][lane] = points[1][ev[lane * N_SHAPE + 34]];
            block_coordinates[104][lane] = points[2][ev[lane * N_SHAPE + 34]];
            block_direction[34][lane] = u_direction[ev[lane * N_SHAPE + 34] * direction_stride];
            block_coordinates[105][lane] = points[0][ev[lane * N_SHAPE + 35]];
            block_coordinates[106][lane] = points[1][ev[lane * N_SHAPE + 35]];
            block_coordinates[107][lane] = points[2][ev[lane * N_SHAPE + 35]];
            block_direction[35][lane] = u_direction[ev[lane * N_SHAPE + 35] * direction_stride];
            block_coordinates[108][lane] = points[0][ev[lane * N_SHAPE + 36]];
            block_coordinates[109][lane] = points[1][ev[lane * N_SHAPE + 36]];
            block_coordinates[110][lane] = points[2][ev[lane * N_SHAPE + 36]];
            block_direction[36][lane] = u_direction[ev[lane * N_SHAPE + 36] * direction_stride];
            block_coordinates[111][lane] = points[0][ev[lane * N_SHAPE + 37]];
            block_coordinates[112][lane] = points[1][ev[lane * N_SHAPE + 37]];
            block_coordinates[113][lane] = points[2][ev[lane * N_SHAPE + 37]];
            block_direction[37][lane] = u_direction[ev[lane * N_SHAPE + 37] * direction_stride];
            block_coordinates[114][lane] = points[0][ev[lane * N_SHAPE + 38]];
            block_coordinates[115][lane] = points[1][ev[lane * N_SHAPE + 38]];
            block_coordinates[116][lane] = points[2][ev[lane * N_SHAPE + 38]];
            block_direction[38][lane] = u_direction[ev[lane * N_SHAPE + 38] * direction_stride];
            block_coordinates[117][lane] = points[0][ev[lane * N_SHAPE + 39]];
            block_coordinates[118][lane] = points[1][ev[lane * N_SHAPE + 39]];
            block_coordinates[119][lane] = points[2][ev[lane * N_SHAPE + 39]];
            block_direction[39][lane] = u_direction[ev[lane * N_SHAPE + 39] * direction_stride];
            block_coordinates[120][lane] = points[0][ev[lane * N_SHAPE + 40]];
            block_coordinates[121][lane] = points[1][ev[lane * N_SHAPE + 40]];
            block_coordinates[122][lane] = points[2][ev[lane * N_SHAPE + 40]];
            block_direction[40][lane] = u_direction[ev[lane * N_SHAPE + 40] * direction_stride];
            block_coordinates[123][lane] = points[0][ev[lane * N_SHAPE + 41]];
            block_coordinates[124][lane] = points[1][ev[lane * N_SHAPE + 41]];
            block_coordinates[125][lane] = points[2][ev[lane * N_SHAPE + 41]];
            block_direction[41][lane] = u_direction[ev[lane * N_SHAPE + 41] * direction_stride];
            block_coordinates[126][lane] = points[0][ev[lane * N_SHAPE + 42]];
            block_coordinates[127][lane] = points[1][ev[lane * N_SHAPE + 42]];
            block_coordinates[128][lane] = points[2][ev[lane * N_SHAPE + 42]];
            block_direction[42][lane] = u_direction[ev[lane * N_SHAPE + 42] * direction_stride];
            block_coordinates[129][lane] = points[0][ev[lane * N_SHAPE + 43]];
            block_coordinates[130][lane] = points[1][ev[lane * N_SHAPE + 43]];
            block_coordinates[131][lane] = points[2][ev[lane * N_SHAPE + 43]];
            block_direction[43][lane] = u_direction[ev[lane * N_SHAPE + 43] * direction_stride];
            block_coordinates[132][lane] = points[0][ev[lane * N_SHAPE + 44]];
            block_coordinates[133][lane] = points[1][ev[lane * N_SHAPE + 44]];
            block_coordinates[134][lane] = points[2][ev[lane * N_SHAPE + 44]];
            block_direction[44][lane] = u_direction[ev[lane * N_SHAPE + 44] * direction_stride];
            block_coordinates[135][lane] = points[0][ev[lane * N_SHAPE + 45]];
            block_coordinates[136][lane] = points[1][ev[lane * N_SHAPE + 45]];
            block_coordinates[137][lane] = points[2][ev[lane * N_SHAPE + 45]];
            block_direction[45][lane] = u_direction[ev[lane * N_SHAPE + 45] * direction_stride];
            block_coordinates[138][lane] = points[0][ev[lane * N_SHAPE + 46]];
            block_coordinates[139][lane] = points[1][ev[lane * N_SHAPE + 46]];
            block_coordinates[140][lane] = points[2][ev[lane * N_SHAPE + 46]];
            block_direction[46][lane] = u_direction[ev[lane * N_SHAPE + 46] * direction_stride];
            block_coordinates[141][lane] = points[0][ev[lane * N_SHAPE + 47]];
            block_coordinates[142][lane] = points[1][ev[lane * N_SHAPE + 47]];
            block_coordinates[143][lane] = points[2][ev[lane * N_SHAPE + 47]];
            block_direction[47][lane] = u_direction[ev[lane * N_SHAPE + 47] * direction_stride];
            block_coordinates[144][lane] = points[0][ev[lane * N_SHAPE + 48]];
            block_coordinates[145][lane] = points[1][ev[lane * N_SHAPE + 48]];
            block_coordinates[146][lane] = points[2][ev[lane * N_SHAPE + 48]];
            block_direction[48][lane] = u_direction[ev[lane * N_SHAPE + 48] * direction_stride];
            block_coordinates[147][lane] = points[0][ev[lane * N_SHAPE + 49]];
            block_coordinates[148][lane] = points[1][ev[lane * N_SHAPE + 49]];
            block_coordinates[149][lane] = points[2][ev[lane * N_SHAPE + 49]];
            block_direction[49][lane] = u_direction[ev[lane * N_SHAPE + 49] * direction_stride];
            block_coordinates[150][lane] = points[0][ev[lane * N_SHAPE + 50]];
            block_coordinates[151][lane] = points[1][ev[lane * N_SHAPE + 50]];
            block_coordinates[152][lane] = points[2][ev[lane * N_SHAPE + 50]];
            block_direction[50][lane] = u_direction[ev[lane * N_SHAPE + 50] * direction_stride];
            block_coordinates[153][lane] = points[0][ev[lane * N_SHAPE + 51]];
            block_coordinates[154][lane] = points[1][ev[lane * N_SHAPE + 51]];
            block_coordinates[155][lane] = points[2][ev[lane * N_SHAPE + 51]];
            block_direction[51][lane] = u_direction[ev[lane * N_SHAPE + 51] * direction_stride];
            block_coordinates[156][lane] = points[0][ev[lane * N_SHAPE + 52]];
            block_coordinates[157][lane] = points[1][ev[lane * N_SHAPE + 52]];
            block_coordinates[158][lane] = points[2][ev[lane * N_SHAPE + 52]];
            block_direction[52][lane] = u_direction[ev[lane * N_SHAPE + 52] * direction_stride];
            block_coordinates[159][lane] = points[0][ev[lane * N_SHAPE + 53]];
            block_coordinates[160][lane] = points[1][ev[lane * N_SHAPE + 53]];
            block_coordinates[161][lane] = points[2][ev[lane * N_SHAPE + 53]];
            block_direction[53][lane] = u_direction[ev[lane * N_SHAPE + 53] * direction_stride];
            block_coordinates[162][lane] = points[0][ev[lane * N_SHAPE + 54]];
            block_coordinates[163][lane] = points[1][ev[lane * N_SHAPE + 54]];
            block_coordinates[164][lane] = points[2][ev[lane * N_SHAPE + 54]];
            block_direction[54][lane] = u_direction[ev[lane * N_SHAPE + 54] * direction_stride];
            block_coordinates[165][lane] = points[0][ev[lane * N_SHAPE + 55]];
            block_coordinates[166][lane] = points[1][ev[lane * N_SHAPE + 55]];
            block_coordinates[167][lane] = points[2][ev[lane * N_SHAPE + 55]];
            block_direction[55][lane] = u_direction[ev[lane * N_SHAPE + 55] * direction_stride];
            block_coordinates[168][lane] = points[0][ev[lane * N_SHAPE + 56]];
            block_coordinates[169][lane] = points[1][ev[lane * N_SHAPE + 56]];
            block_coordinates[170][lane] = points[2][ev[lane * N_SHAPE + 56]];
            block_direction[56][lane] = u_direction[ev[lane * N_SHAPE + 56] * direction_stride];
            block_coordinates[171][lane] = points[0][ev[lane * N_SHAPE + 57]];
            block_coordinates[172][lane] = points[1][ev[lane * N_SHAPE + 57]];
            block_coordinates[173][lane] = points[2][ev[lane * N_SHAPE + 57]];
            block_direction[57][lane] = u_direction[ev[lane * N_SHAPE + 57] * direction_stride];
            block_coordinates[174][lane] = points[0][ev[lane * N_SHAPE + 58]];
            block_coordinates[175][lane] = points[1][ev[lane * N_SHAPE + 58]];
            block_coordinates[176][lane] = points[2][ev[lane * N_SHAPE + 58]];
            block_direction[58][lane] = u_direction[ev[lane * N_SHAPE + 58] * direction_stride];
            block_coordinates[177][lane] = points[0][ev[lane * N_SHAPE + 59]];
            block_coordinates[178][lane] = points[1][ev[lane * N_SHAPE + 59]];
            block_coordinates[179][lane] = points[2][ev[lane * N_SHAPE + 59]];
            block_direction[59][lane] = u_direction[ev[lane * N_SHAPE + 59] * direction_stride];
            block_coordinates[180][lane] = points[0][ev[lane * N_SHAPE + 60]];
            block_coordinates[181][lane] = points[1][ev[lane * N_SHAPE + 60]];
            block_coordinates[182][lane] = points[2][ev[lane * N_SHAPE + 60]];
            block_direction[60][lane] = u_direction[ev[lane * N_SHAPE + 60] * direction_stride];
            block_coordinates[183][lane] = points[0][ev[lane * N_SHAPE + 61]];
            block_coordinates[184][lane] = points[1][ev[lane * N_SHAPE + 61]];
            block_coordinates[185][lane] = points[2][ev[lane * N_SHAPE + 61]];
            block_direction[61][lane] = u_direction[ev[lane * N_SHAPE + 61] * direction_stride];
            block_coordinates[186][lane] = points[0][ev[lane * N_SHAPE + 62]];
            block_coordinates[187][lane] = points[1][ev[lane * N_SHAPE + 62]];
            block_coordinates[188][lane] = points[2][ev[lane * N_SHAPE + 62]];
            block_direction[62][lane] = u_direction[ev[lane * N_SHAPE + 62] * direction_stride];
            block_coordinates[189][lane] = points[0][ev[lane * N_SHAPE + 63]];
            block_coordinates[190][lane] = points[1][ev[lane * N_SHAPE + 63]];
            block_coordinates[191][lane] = points[2][ev[lane * N_SHAPE + 63]];
            block_direction[63][lane] = u_direction[ev[lane * N_SHAPE + 63] * direction_stride];
            block_coordinates[192][lane] = points[0][ev[lane * N_SHAPE + 64]];
            block_coordinates[193][lane] = points[1][ev[lane * N_SHAPE + 64]];
            block_coordinates[194][lane] = points[2][ev[lane * N_SHAPE + 64]];
            block_direction[64][lane] = u_direction[ev[lane * N_SHAPE + 64] * direction_stride];
            block_coordinates[195][lane] = points[0][ev[lane * N_SHAPE + 65]];
            block_coordinates[196][lane] = points[1][ev[lane * N_SHAPE + 65]];
            block_coordinates[197][lane] = points[2][ev[lane * N_SHAPE + 65]];
            block_direction[65][lane] = u_direction[ev[lane * N_SHAPE + 65] * direction_stride];
            block_coordinates[198][lane] = points[0][ev[lane * N_SHAPE + 66]];
            block_coordinates[199][lane] = points[1][ev[lane * N_SHAPE + 66]];
            block_coordinates[200][lane] = points[2][ev[lane * N_SHAPE + 66]];
            block_direction[66][lane] = u_direction[ev[lane * N_SHAPE + 66] * direction_stride];
            block_coordinates[201][lane] = points[0][ev[lane * N_SHAPE + 67]];
            block_coordinates[202][lane] = points[1][ev[lane * N_SHAPE + 67]];
            block_coordinates[203][lane] = points[2][ev[lane * N_SHAPE + 67]];
            block_direction[67][lane] = u_direction[ev[lane * N_SHAPE + 67] * direction_stride];
            block_coordinates[204][lane] = points[0][ev[lane * N_SHAPE + 68]];
            block_coordinates[205][lane] = points[1][ev[lane * N_SHAPE + 68]];
            block_coordinates[206][lane] = points[2][ev[lane * N_SHAPE + 68]];
            block_direction[68][lane] = u_direction[ev[lane * N_SHAPE + 68] * direction_stride];
            block_coordinates[207][lane] = points[0][ev[lane * N_SHAPE + 69]];
            block_coordinates[208][lane] = points[1][ev[lane * N_SHAPE + 69]];
            block_coordinates[209][lane] = points[2][ev[lane * N_SHAPE + 69]];
            block_direction[69][lane] = u_direction[ev[lane * N_SHAPE + 69] * direction_stride];
            block_coordinates[210][lane] = points[0][ev[lane * N_SHAPE + 70]];
            block_coordinates[211][lane] = points[1][ev[lane * N_SHAPE + 70]];
            block_coordinates[212][lane] = points[2][ev[lane * N_SHAPE + 70]];
            block_direction[70][lane] = u_direction[ev[lane * N_SHAPE + 70] * direction_stride];
            block_coordinates[213][lane] = points[0][ev[lane * N_SHAPE + 71]];
            block_coordinates[214][lane] = points[1][ev[lane * N_SHAPE + 71]];
            block_coordinates[215][lane] = points[2][ev[lane * N_SHAPE + 71]];
            block_direction[71][lane] = u_direction[ev[lane * N_SHAPE + 71] * direction_stride];
            block_coordinates[216][lane] = points[0][ev[lane * N_SHAPE + 72]];
            block_coordinates[217][lane] = points[1][ev[lane * N_SHAPE + 72]];
            block_coordinates[218][lane] = points[2][ev[lane * N_SHAPE + 72]];
            block_direction[72][lane] = u_direction[ev[lane * N_SHAPE + 72] * direction_stride];
            block_coordinates[219][lane] = points[0][ev[lane * N_SHAPE + 73]];
            block_coordinates[220][lane] = points[1][ev[lane * N_SHAPE + 73]];
            block_coordinates[221][lane] = points[2][ev[lane * N_SHAPE + 73]];
            block_direction[73][lane] = u_direction[ev[lane * N_SHAPE + 73] * direction_stride];
            block_coordinates[222][lane] = points[0][ev[lane * N_SHAPE + 74]];
            block_coordinates[223][lane] = points[1][ev[lane * N_SHAPE + 74]];
            block_coordinates[224][lane] = points[2][ev[lane * N_SHAPE + 74]];
            block_direction[74][lane] = u_direction[ev[lane * N_SHAPE + 74] * direction_stride];
            block_coordinates[225][lane] = points[0][ev[lane * N_SHAPE + 75]];
            block_coordinates[226][lane] = points[1][ev[lane * N_SHAPE + 75]];
            block_coordinates[227][lane] = points[2][ev[lane * N_SHAPE + 75]];
            block_direction[75][lane] = u_direction[ev[lane * N_SHAPE + 75] * direction_stride];
            block_coordinates[228][lane] = points[0][ev[lane * N_SHAPE + 76]];
            block_coordinates[229][lane] = points[1][ev[lane * N_SHAPE + 76]];
            block_coordinates[230][lane] = points[2][ev[lane * N_SHAPE + 76]];
            block_direction[76][lane] = u_direction[ev[lane * N_SHAPE + 76] * direction_stride];
            block_coordinates[231][lane] = points[0][ev[lane * N_SHAPE + 77]];
            block_coordinates[232][lane] = points[1][ev[lane * N_SHAPE + 77]];
            block_coordinates[233][lane] = points[2][ev[lane * N_SHAPE + 77]];
            block_direction[77][lane] = u_direction[ev[lane * N_SHAPE + 77] * direction_stride];
            block_coordinates[234][lane] = points[0][ev[lane * N_SHAPE + 78]];
            block_coordinates[235][lane] = points[1][ev[lane * N_SHAPE + 78]];
            block_coordinates[236][lane] = points[2][ev[lane * N_SHAPE + 78]];
            block_direction[78][lane] = u_direction[ev[lane * N_SHAPE + 78] * direction_stride];
            block_coordinates[237][lane] = points[0][ev[lane * N_SHAPE + 79]];
            block_coordinates[238][lane] = points[1][ev[lane * N_SHAPE + 79]];
            block_coordinates[239][lane] = points[2][ev[lane * N_SHAPE + 79]];
            block_direction[79][lane] = u_direction[ev[lane * N_SHAPE + 79] * direction_stride];
            block_coordinates[240][lane] = points[0][ev[lane * N_SHAPE + 80]];
            block_coordinates[241][lane] = points[1][ev[lane * N_SHAPE + 80]];
            block_coordinates[242][lane] = points[2][ev[lane * N_SHAPE + 80]];
            block_direction[80][lane] = u_direction[ev[lane * N_SHAPE + 80] * direction_stride];
            block_coordinates[243][lane] = points[0][ev[lane * N_SHAPE + 81]];
            block_coordinates[244][lane] = points[1][ev[lane * N_SHAPE + 81]];
            block_coordinates[245][lane] = points[2][ev[lane * N_SHAPE + 81]];
            block_direction[81][lane] = u_direction[ev[lane * N_SHAPE + 81] * direction_stride];
            block_coordinates[246][lane] = points[0][ev[lane * N_SHAPE + 82]];
            block_coordinates[247][lane] = points[1][ev[lane * N_SHAPE + 82]];
            block_coordinates[248][lane] = points[2][ev[lane * N_SHAPE + 82]];
            block_direction[82][lane] = u_direction[ev[lane * N_SHAPE + 82] * direction_stride];
            block_coordinates[249][lane] = points[0][ev[lane * N_SHAPE + 83]];
            block_coordinates[250][lane] = points[1][ev[lane * N_SHAPE + 83]];
            block_coordinates[251][lane] = points[2][ev[lane * N_SHAPE + 83]];
            block_direction[83][lane] = u_direction[ev[lane * N_SHAPE + 83] * direction_stride];
            block_coordinates[252][lane] = points[0][ev[lane * N_SHAPE + 84]];
            block_coordinates[253][lane] = points[1][ev[lane * N_SHAPE + 84]];
            block_coordinates[254][lane] = points[2][ev[lane * N_SHAPE + 84]];
            block_direction[84][lane] = u_direction[ev[lane * N_SHAPE + 84] * direction_stride];
            block_coordinates[255][lane] = points[0][ev[lane * N_SHAPE + 85]];
            block_coordinates[256][lane] = points[1][ev[lane * N_SHAPE + 85]];
            block_coordinates[257][lane] = points[2][ev[lane * N_SHAPE + 85]];
            block_direction[85][lane] = u_direction[ev[lane * N_SHAPE + 85] * direction_stride];
            block_coordinates[258][lane] = points[0][ev[lane * N_SHAPE + 86]];
            block_coordinates[259][lane] = points[1][ev[lane * N_SHAPE + 86]];
            block_coordinates[260][lane] = points[2][ev[lane * N_SHAPE + 86]];
            block_direction[86][lane] = u_direction[ev[lane * N_SHAPE + 86] * direction_stride];
            block_coordinates[261][lane] = points[0][ev[lane * N_SHAPE + 87]];
            block_coordinates[262][lane] = points[1][ev[lane * N_SHAPE + 87]];
            block_coordinates[263][lane] = points[2][ev[lane * N_SHAPE + 87]];
            block_direction[87][lane] = u_direction[ev[lane * N_SHAPE + 87] * direction_stride];
            block_coordinates[264][lane] = points[0][ev[lane * N_SHAPE + 88]];
            block_coordinates[265][lane] = points[1][ev[lane * N_SHAPE + 88]];
            block_coordinates[266][lane] = points[2][ev[lane * N_SHAPE + 88]];
            block_direction[88][lane] = u_direction[ev[lane * N_SHAPE + 88] * direction_stride];
            block_coordinates[267][lane] = points[0][ev[lane * N_SHAPE + 89]];
            block_coordinates[268][lane] = points[1][ev[lane * N_SHAPE + 89]];
            block_coordinates[269][lane] = points[2][ev[lane * N_SHAPE + 89]];
            block_direction[89][lane] = u_direction[ev[lane * N_SHAPE + 89] * direction_stride];
            block_coordinates[270][lane] = points[0][ev[lane * N_SHAPE + 90]];
            block_coordinates[271][lane] = points[1][ev[lane * N_SHAPE + 90]];
            block_coordinates[272][lane] = points[2][ev[lane * N_SHAPE + 90]];
            block_direction[90][lane] = u_direction[ev[lane * N_SHAPE + 90] * direction_stride];
            block_coordinates[273][lane] = points[0][ev[lane * N_SHAPE + 91]];
            block_coordinates[274][lane] = points[1][ev[lane * N_SHAPE + 91]];
            block_coordinates[275][lane] = points[2][ev[lane * N_SHAPE + 91]];
            block_direction[91][lane] = u_direction[ev[lane * N_SHAPE + 91] * direction_stride];
            block_coordinates[276][lane] = points[0][ev[lane * N_SHAPE + 92]];
            block_coordinates[277][lane] = points[1][ev[lane * N_SHAPE + 92]];
            block_coordinates[278][lane] = points[2][ev[lane * N_SHAPE + 92]];
            block_direction[92][lane] = u_direction[ev[lane * N_SHAPE + 92] * direction_stride];
            block_coordinates[279][lane] = points[0][ev[lane * N_SHAPE + 93]];
            block_coordinates[280][lane] = points[1][ev[lane * N_SHAPE + 93]];
            block_coordinates[281][lane] = points[2][ev[lane * N_SHAPE + 93]];
            block_direction[93][lane] = u_direction[ev[lane * N_SHAPE + 93] * direction_stride];
            block_coordinates[282][lane] = points[0][ev[lane * N_SHAPE + 94]];
            block_coordinates[283][lane] = points[1][ev[lane * N_SHAPE + 94]];
            block_coordinates[284][lane] = points[2][ev[lane * N_SHAPE + 94]];
            block_direction[94][lane] = u_direction[ev[lane * N_SHAPE + 94] * direction_stride];
            block_coordinates[285][lane] = points[0][ev[lane * N_SHAPE + 95]];
            block_coordinates[286][lane] = points[1][ev[lane * N_SHAPE + 95]];
            block_coordinates[287][lane] = points[2][ev[lane * N_SHAPE + 95]];
            block_direction[95][lane] = u_direction[ev[lane * N_SHAPE + 95] * direction_stride];
            block_coordinates[288][lane] = points[0][ev[lane * N_SHAPE + 96]];
            block_coordinates[289][lane] = points[1][ev[lane * N_SHAPE + 96]];
            block_coordinates[290][lane] = points[2][ev[lane * N_SHAPE + 96]];
            block_direction[96][lane] = u_direction[ev[lane * N_SHAPE + 96] * direction_stride];
            block_coordinates[291][lane] = points[0][ev[lane * N_SHAPE + 97]];
            block_coordinates[292][lane] = points[1][ev[lane * N_SHAPE + 97]];
            block_coordinates[293][lane] = points[2][ev[lane * N_SHAPE + 97]];
            block_direction[97][lane] = u_direction[ev[lane * N_SHAPE + 97] * direction_stride];
            block_coordinates[294][lane] = points[0][ev[lane * N_SHAPE + 98]];
            block_coordinates[295][lane] = points[1][ev[lane * N_SHAPE + 98]];
            block_coordinates[296][lane] = points[2][ev[lane * N_SHAPE + 98]];
            block_direction[98][lane] = u_direction[ev[lane * N_SHAPE + 98] * direction_stride];
            block_coordinates[297][lane] = points[0][ev[lane * N_SHAPE + 99]];
            block_coordinates[298][lane] = points[1][ev[lane * N_SHAPE + 99]];
            block_coordinates[299][lane] = points[2][ev[lane * N_SHAPE + 99]];
            block_direction[99][lane] = u_direction[ev[lane * N_SHAPE + 99] * direction_stride];
            block_coordinates[300][lane] = points[0][ev[lane * N_SHAPE + 100]];
            block_coordinates[301][lane] = points[1][ev[lane * N_SHAPE + 100]];
            block_coordinates[302][lane] = points[2][ev[lane * N_SHAPE + 100]];
            block_direction[100][lane] = u_direction[ev[lane * N_SHAPE + 100] * direction_stride];
            block_coordinates[303][lane] = points[0][ev[lane * N_SHAPE + 101]];
            block_coordinates[304][lane] = points[1][ev[lane * N_SHAPE + 101]];
            block_coordinates[305][lane] = points[2][ev[lane * N_SHAPE + 101]];
            block_direction[101][lane] = u_direction[ev[lane * N_SHAPE + 101] * direction_stride];
            block_coordinates[306][lane] = points[0][ev[lane * N_SHAPE + 102]];
            block_coordinates[307][lane] = points[1][ev[lane * N_SHAPE + 102]];
            block_coordinates[308][lane] = points[2][ev[lane * N_SHAPE + 102]];
            block_direction[102][lane] = u_direction[ev[lane * N_SHAPE + 102] * direction_stride];
            block_coordinates[309][lane] = points[0][ev[lane * N_SHAPE + 103]];
            block_coordinates[310][lane] = points[1][ev[lane * N_SHAPE + 103]];
            block_coordinates[311][lane] = points[2][ev[lane * N_SHAPE + 103]];
            block_direction[103][lane] = u_direction[ev[lane * N_SHAPE + 103] * direction_stride];
            block_coordinates[312][lane] = points[0][ev[lane * N_SHAPE + 104]];
            block_coordinates[313][lane] = points[1][ev[lane * N_SHAPE + 104]];
            block_coordinates[314][lane] = points[2][ev[lane * N_SHAPE + 104]];
            block_direction[104][lane] = u_direction[ev[lane * N_SHAPE + 104] * direction_stride];
            block_coordinates[315][lane] = points[0][ev[lane * N_SHAPE + 105]];
            block_coordinates[316][lane] = points[1][ev[lane * N_SHAPE + 105]];
            block_coordinates[317][lane] = points[2][ev[lane * N_SHAPE + 105]];
            block_direction[105][lane] = u_direction[ev[lane * N_SHAPE + 105] * direction_stride];
            block_coordinates[318][lane] = points[0][ev[lane * N_SHAPE + 106]];
            block_coordinates[319][lane] = points[1][ev[lane * N_SHAPE + 106]];
            block_coordinates[320][lane] = points[2][ev[lane * N_SHAPE + 106]];
            block_direction[106][lane] = u_direction[ev[lane * N_SHAPE + 106] * direction_stride];
            block_coordinates[321][lane] = points[0][ev[lane * N_SHAPE + 107]];
            block_coordinates[322][lane] = points[1][ev[lane * N_SHAPE + 107]];
            block_coordinates[323][lane] = points[2][ev[lane * N_SHAPE + 107]];
            block_direction[107][lane] = u_direction[ev[lane * N_SHAPE + 107] * direction_stride];
            block_coordinates[324][lane] = points[0][ev[lane * N_SHAPE + 108]];
            block_coordinates[325][lane] = points[1][ev[lane * N_SHAPE + 108]];
            block_coordinates[326][lane] = points[2][ev[lane * N_SHAPE + 108]];
            block_direction[108][lane] = u_direction[ev[lane * N_SHAPE + 108] * direction_stride];
            block_coordinates[327][lane] = points[0][ev[lane * N_SHAPE + 109]];
            block_coordinates[328][lane] = points[1][ev[lane * N_SHAPE + 109]];
            block_coordinates[329][lane] = points[2][ev[lane * N_SHAPE + 109]];
            block_direction[109][lane] = u_direction[ev[lane * N_SHAPE + 109] * direction_stride];
            block_coordinates[330][lane] = points[0][ev[lane * N_SHAPE + 110]];
            block_coordinates[331][lane] = points[1][ev[lane * N_SHAPE + 110]];
            block_coordinates[332][lane] = points[2][ev[lane * N_SHAPE + 110]];
            block_direction[110][lane] = u_direction[ev[lane * N_SHAPE + 110] * direction_stride];
            block_coordinates[333][lane] = points[0][ev[lane * N_SHAPE + 111]];
            block_coordinates[334][lane] = points[1][ev[lane * N_SHAPE + 111]];
            block_coordinates[335][lane] = points[2][ev[lane * N_SHAPE + 111]];
            block_direction[111][lane] = u_direction[ev[lane * N_SHAPE + 111] * direction_stride];
            block_coordinates[336][lane] = points[0][ev[lane * N_SHAPE + 112]];
            block_coordinates[337][lane] = points[1][ev[lane * N_SHAPE + 112]];
            block_coordinates[338][lane] = points[2][ev[lane * N_SHAPE + 112]];
            block_direction[112][lane] = u_direction[ev[lane * N_SHAPE + 112] * direction_stride];
            block_coordinates[339][lane] = points[0][ev[lane * N_SHAPE + 113]];
            block_coordinates[340][lane] = points[1][ev[lane * N_SHAPE + 113]];
            block_coordinates[341][lane] = points[2][ev[lane * N_SHAPE + 113]];
            block_direction[113][lane] = u_direction[ev[lane * N_SHAPE + 113] * direction_stride];
            block_coordinates[342][lane] = points[0][ev[lane * N_SHAPE + 114]];
            block_coordinates[343][lane] = points[1][ev[lane * N_SHAPE + 114]];
            block_coordinates[344][lane] = points[2][ev[lane * N_SHAPE + 114]];
            block_direction[114][lane] = u_direction[ev[lane * N_SHAPE + 114] * direction_stride];
            block_coordinates[345][lane] = points[0][ev[lane * N_SHAPE + 115]];
            block_coordinates[346][lane] = points[1][ev[lane * N_SHAPE + 115]];
            block_coordinates[347][lane] = points[2][ev[lane * N_SHAPE + 115]];
            block_direction[115][lane] = u_direction[ev[lane * N_SHAPE + 115] * direction_stride];
            block_coordinates[348][lane] = points[0][ev[lane * N_SHAPE + 116]];
            block_coordinates[349][lane] = points[1][ev[lane * N_SHAPE + 116]];
            block_coordinates[350][lane] = points[2][ev[lane * N_SHAPE + 116]];
            block_direction[116][lane] = u_direction[ev[lane * N_SHAPE + 116] * direction_stride];
            block_coordinates[351][lane] = points[0][ev[lane * N_SHAPE + 117]];
            block_coordinates[352][lane] = points[1][ev[lane * N_SHAPE + 117]];
            block_coordinates[353][lane] = points[2][ev[lane * N_SHAPE + 117]];
            block_direction[117][lane] = u_direction[ev[lane * N_SHAPE + 117] * direction_stride];
            block_coordinates[354][lane] = points[0][ev[lane * N_SHAPE + 118]];
            block_coordinates[355][lane] = points[1][ev[lane * N_SHAPE + 118]];
            block_coordinates[356][lane] = points[2][ev[lane * N_SHAPE + 118]];
            block_direction[118][lane] = u_direction[ev[lane * N_SHAPE + 118] * direction_stride];
            block_coordinates[357][lane] = points[0][ev[lane * N_SHAPE + 119]];
            block_coordinates[358][lane] = points[1][ev[lane * N_SHAPE + 119]];
            block_coordinates[359][lane] = points[2][ev[lane * N_SHAPE + 119]];
            block_direction[119][lane] = u_direction[ev[lane * N_SHAPE + 119] * direction_stride];
            block_coordinates[360][lane] = points[0][ev[lane * N_SHAPE + 120]];
            block_coordinates[361][lane] = points[1][ev[lane * N_SHAPE + 120]];
            block_coordinates[362][lane] = points[2][ev[lane * N_SHAPE + 120]];
            block_direction[120][lane] = u_direction[ev[lane * N_SHAPE + 120] * direction_stride];
            block_coordinates[363][lane] = points[0][ev[lane * N_SHAPE + 121]];
            block_coordinates[364][lane] = points[1][ev[lane * N_SHAPE + 121]];
            block_coordinates[365][lane] = points[2][ev[lane * N_SHAPE + 121]];
            block_direction[121][lane] = u_direction[ev[lane * N_SHAPE + 121] * direction_stride];
            block_coordinates[366][lane] = points[0][ev[lane * N_SHAPE + 122]];
            block_coordinates[367][lane] = points[1][ev[lane * N_SHAPE + 122]];
            block_coordinates[368][lane] = points[2][ev[lane * N_SHAPE + 122]];
            block_direction[122][lane] = u_direction[ev[lane * N_SHAPE + 122] * direction_stride];
            block_coordinates[369][lane] = points[0][ev[lane * N_SHAPE + 123]];
            block_coordinates[370][lane] = points[1][ev[lane * N_SHAPE + 123]];
            block_coordinates[371][lane] = points[2][ev[lane * N_SHAPE + 123]];
            block_direction[123][lane] = u_direction[ev[lane * N_SHAPE + 123] * direction_stride];
            block_coordinates[372][lane] = points[0][ev[lane * N_SHAPE + 124]];
            block_coordinates[373][lane] = points[1][ev[lane * N_SHAPE + 124]];
            block_coordinates[374][lane] = points[2][ev[lane * N_SHAPE + 124]];
            block_direction[124][lane] = u_direction[ev[lane * N_SHAPE + 124] * direction_stride];
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
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
            block_output[34][lane] = scalar_t(0);
            block_output[35][lane] = scalar_t(0);
            block_output[36][lane] = scalar_t(0);
            block_output[37][lane] = scalar_t(0);
            block_output[38][lane] = scalar_t(0);
            block_output[39][lane] = scalar_t(0);
            block_output[40][lane] = scalar_t(0);
            block_output[41][lane] = scalar_t(0);
            block_output[42][lane] = scalar_t(0);
            block_output[43][lane] = scalar_t(0);
            block_output[44][lane] = scalar_t(0);
            block_output[45][lane] = scalar_t(0);
            block_output[46][lane] = scalar_t(0);
            block_output[47][lane] = scalar_t(0);
            block_output[48][lane] = scalar_t(0);
            block_output[49][lane] = scalar_t(0);
            block_output[50][lane] = scalar_t(0);
            block_output[51][lane] = scalar_t(0);
            block_output[52][lane] = scalar_t(0);
            block_output[53][lane] = scalar_t(0);
            block_output[54][lane] = scalar_t(0);
            block_output[55][lane] = scalar_t(0);
            block_output[56][lane] = scalar_t(0);
            block_output[57][lane] = scalar_t(0);
            block_output[58][lane] = scalar_t(0);
            block_output[59][lane] = scalar_t(0);
            block_output[60][lane] = scalar_t(0);
            block_output[61][lane] = scalar_t(0);
            block_output[62][lane] = scalar_t(0);
            block_output[63][lane] = scalar_t(0);
            block_output[64][lane] = scalar_t(0);
            block_output[65][lane] = scalar_t(0);
            block_output[66][lane] = scalar_t(0);
            block_output[67][lane] = scalar_t(0);
            block_output[68][lane] = scalar_t(0);
            block_output[69][lane] = scalar_t(0);
            block_output[70][lane] = scalar_t(0);
            block_output[71][lane] = scalar_t(0);
            block_output[72][lane] = scalar_t(0);
            block_output[73][lane] = scalar_t(0);
            block_output[74][lane] = scalar_t(0);
            block_output[75][lane] = scalar_t(0);
            block_output[76][lane] = scalar_t(0);
            block_output[77][lane] = scalar_t(0);
            block_output[78][lane] = scalar_t(0);
            block_output[79][lane] = scalar_t(0);
            block_output[80][lane] = scalar_t(0);
            block_output[81][lane] = scalar_t(0);
            block_output[82][lane] = scalar_t(0);
            block_output[83][lane] = scalar_t(0);
            block_output[84][lane] = scalar_t(0);
            block_output[85][lane] = scalar_t(0);
            block_output[86][lane] = scalar_t(0);
            block_output[87][lane] = scalar_t(0);
            block_output[88][lane] = scalar_t(0);
            block_output[89][lane] = scalar_t(0);
            block_output[90][lane] = scalar_t(0);
            block_output[91][lane] = scalar_t(0);
            block_output[92][lane] = scalar_t(0);
            block_output[93][lane] = scalar_t(0);
            block_output[94][lane] = scalar_t(0);
            block_output[95][lane] = scalar_t(0);
            block_output[96][lane] = scalar_t(0);
            block_output[97][lane] = scalar_t(0);
            block_output[98][lane] = scalar_t(0);
            block_output[99][lane] = scalar_t(0);
            block_output[100][lane] = scalar_t(0);
            block_output[101][lane] = scalar_t(0);
            block_output[102][lane] = scalar_t(0);
            block_output[103][lane] = scalar_t(0);
            block_output[104][lane] = scalar_t(0);
            block_output[105][lane] = scalar_t(0);
            block_output[106][lane] = scalar_t(0);
            block_output[107][lane] = scalar_t(0);
            block_output[108][lane] = scalar_t(0);
            block_output[109][lane] = scalar_t(0);
            block_output[110][lane] = scalar_t(0);
            block_output[111][lane] = scalar_t(0);
            block_output[112][lane] = scalar_t(0);
            block_output[113][lane] = scalar_t(0);
            block_output[114][lane] = scalar_t(0);
            block_output[115][lane] = scalar_t(0);
            block_output[116][lane] = scalar_t(0);
            block_output[117][lane] = scalar_t(0);
            block_output[118][lane] = scalar_t(0);
            block_output[119][lane] = scalar_t(0);
            block_output[120][lane] = scalar_t(0);
            block_output[121][lane] = scalar_t(0);
            block_output[122][lane] = scalar_t(0);
            block_output[123][lane] = scalar_t(0);
            block_output[124][lane] = scalar_t(0);
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[81], block_coordinates[82], block_coordinates[83], block_coordinates[84], block_coordinates[85], block_coordinates[86], block_coordinates[87], block_coordinates[88], block_coordinates[89], block_coordinates[90], block_coordinates[91], block_coordinates[92], block_coordinates[93], block_coordinates[94], block_coordinates[95], block_coordinates[96], block_coordinates[97], block_coordinates[98], block_coordinates[99], block_coordinates[100], block_coordinates[101], block_coordinates[102], block_coordinates[103], block_coordinates[104], block_coordinates[105], block_coordinates[106], block_coordinates[107], block_coordinates[108], block_coordinates[109], block_coordinates[110], block_coordinates[111], block_coordinates[112], block_coordinates[113], block_coordinates[114], block_coordinates[115], block_coordinates[116], block_coordinates[117], block_coordinates[118], block_coordinates[119], block_coordinates[120], block_coordinates[121], block_coordinates[122], block_coordinates[123], block_coordinates[124], block_coordinates[125], block_coordinates[126], block_coordinates[127], block_coordinates[128], block_coordinates[129], block_coordinates[130], block_coordinates[131], block_coordinates[132], block_coordinates[133], block_coordinates[134], block_coordinates[135], block_coordinates[136], block_coordinates[137], block_coordinates[138], block_coordinates[139], block_coordinates[140], block_coordinates[141], block_coordinates[142], block_coordinates[143], block_coordinates[144], block_coordinates[145], block_coordinates[146], block_coordinates[147], block_coordinates[148], block_coordinates[149], block_coordinates[150], block_coordinates[151], block_coordinates[152], block_coordinates[153], block_coordinates[154], block_coordinates[155], block_coordinates[156], block_coordinates[157], block_coordinates[158], block_coordinates[159], block_coordinates[160], block_coordinates[161], block_coordinates[162], block_coordinates[163], block_coordinates[164], block_coordinates[165], block_coordinates[166], block_coordinates[167], block_coordinates[168], block_coordinates[169], block_coordinates[170], block_coordinates[171], block_coordinates[172], block_coordinates[173], block_coordinates[174], block_coordinates[175], block_coordinates[176], block_coordinates[177], block_coordinates[178], block_coordinates[179], block_coordinates[180], block_coordinates[181], block_coordinates[182], block_coordinates[183], block_coordinates[184], block_coordinates[185], block_coordinates[186], block_coordinates[187], block_coordinates[188], block_coordinates[189], block_coordinates[190], block_coordinates[191], block_coordinates[192], block_coordinates[193], block_coordinates[194], block_coordinates[195], block_coordinates[196], block_coordinates[197], block_coordinates[198], block_coordinates[199], block_coordinates[200], block_coordinates[201], block_coordinates[202], block_coordinates[203], block_coordinates[204], block_coordinates[205], block_coordinates[206], block_coordinates[207], block_coordinates[208], block_coordinates[209], block_coordinates[210], block_coordinates[211], block_coordinates[212], block_coordinates[213], block_coordinates[214], block_coordinates[215], block_coordinates[216], block_coordinates[217], block_coordinates[218], block_coordinates[219], block_coordinates[220], block_coordinates[221], block_coordinates[222], block_coordinates[223], block_coordinates[224], block_coordinates[225], block_coordinates[226], block_coordinates[227], block_coordinates[228], block_coordinates[229], block_coordinates[230], block_coordinates[231], block_coordinates[232], block_coordinates[233], block_coordinates[234], block_coordinates[235], block_coordinates[236], block_coordinates[237], block_coordinates[238], block_coordinates[239], block_coordinates[240], block_coordinates[241], block_coordinates[242], block_coordinates[243], block_coordinates[244], block_coordinates[245], block_coordinates[246], block_coordinates[247], block_coordinates[248], block_coordinates[249], block_coordinates[250], block_coordinates[251], block_coordinates[252], block_coordinates[253], block_coordinates[254], block_coordinates[255], block_coordinates[256], block_coordinates[257], block_coordinates[258], block_coordinates[259], block_coordinates[260], block_coordinates[261], block_coordinates[262], block_coordinates[263], block_coordinates[264], block_coordinates[265], block_coordinates[266], block_coordinates[267], block_coordinates[268], block_coordinates[269], block_coordinates[270], block_coordinates[271], block_coordinates[272], block_coordinates[273], block_coordinates[274], block_coordinates[275], block_coordinates[276], block_coordinates[277], block_coordinates[278], block_coordinates[279], block_coordinates[280], block_coordinates[281], block_coordinates[282], block_coordinates[283], block_coordinates[284], block_coordinates[285], block_coordinates[286], block_coordinates[287], block_coordinates[288], block_coordinates[289], block_coordinates[290], block_coordinates[291], block_coordinates[292], block_coordinates[293], block_coordinates[294], block_coordinates[295], block_coordinates[296], block_coordinates[297], block_coordinates[298], block_coordinates[299], block_coordinates[300], block_coordinates[301], block_coordinates[302], block_coordinates[303], block_coordinates[304], block_coordinates[305], block_coordinates[306], block_coordinates[307], block_coordinates[308], block_coordinates[309], block_coordinates[310], block_coordinates[311], block_coordinates[312], block_coordinates[313], block_coordinates[314], block_coordinates[315], block_coordinates[316], block_coordinates[317], block_coordinates[318], block_coordinates[319], block_coordinates[320], block_coordinates[321], block_coordinates[322], block_coordinates[323], block_coordinates[324], block_coordinates[325], block_coordinates[326], block_coordinates[327], block_coordinates[328], block_coordinates[329], block_coordinates[330], block_coordinates[331], block_coordinates[332], block_coordinates[333], block_coordinates[334], block_coordinates[335], block_coordinates[336], block_coordinates[337], block_coordinates[338], block_coordinates[339], block_coordinates[340], block_coordinates[341], block_coordinates[342], block_coordinates[343], block_coordinates[344], block_coordinates[345], block_coordinates[346], block_coordinates[347], block_coordinates[348], block_coordinates[349], block_coordinates[350], block_coordinates[351], block_coordinates[352], block_coordinates[353], block_coordinates[354], block_coordinates[355], block_coordinates[356], block_coordinates[357], block_coordinates[358], block_coordinates[359], block_coordinates[360], block_coordinates[361], block_coordinates[362], block_coordinates[363], block_coordinates[364], block_coordinates[365], block_coordinates[366], block_coordinates[367], block_coordinates[368], block_coordinates[369], block_coordinates[370], block_coordinates[371], block_coordinates[372], block_coordinates[373], block_coordinates[374]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26], block_direction[27], block_direction[28], block_direction[29], block_direction[30], block_direction[31], block_direction[32], block_direction[33], block_direction[34], block_direction[35], block_direction[36], block_direction[37], block_direction[38], block_direction[39], block_direction[40], block_direction[41], block_direction[42], block_direction[43], block_direction[44], block_direction[45], block_direction[46], block_direction[47], block_direction[48], block_direction[49], block_direction[50], block_direction[51], block_direction[52], block_direction[53], block_direction[54], block_direction[55], block_direction[56], block_direction[57], block_direction[58], block_direction[59], block_direction[60], block_direction[61], block_direction[62], block_direction[63], block_direction[64], block_direction[65], block_direction[66], block_direction[67], block_direction[68], block_direction[69], block_direction[70], block_direction[71], block_direction[72], block_direction[73], block_direction[74], block_direction[75], block_direction[76], block_direction[77], block_direction[78], block_direction[79], block_direction[80], block_direction[81], block_direction[82], block_direction[83], block_direction[84], block_direction[85], block_direction[86], block_direction[87], block_direction[88], block_direction[89], block_direction[90], block_direction[91], block_direction[92], block_direction[93], block_direction[94], block_direction[95], block_direction[96], block_direction[97], block_direction[98], block_direction[99], block_direction[100], block_direction[101], block_direction[102], block_direction[103], block_direction[104], block_direction[105], block_direction[106], block_direction[107], block_direction[108], block_direction[109], block_direction[110], block_direction[111], block_direction[112], block_direction[113], block_direction[114], block_direction[115], block_direction[116], block_direction[117], block_direction[118], block_direction[119], block_direction[120], block_direction[121], block_direction[122], block_direction[123], block_direction[124]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63], block_output[64], block_output[65], block_output[66], block_output[67], block_output[68], block_output[69], block_output[70], block_output[71], block_output[72], block_output[73], block_output[74], block_output[75], block_output[76], block_output[77], block_output[78], block_output[79], block_output[80], block_output[81], block_output[82], block_output[83], block_output[84], block_output[85], block_output[86], block_output[87], block_output[88], block_output[89], block_output[90], block_output[91], block_output[92], block_output[93], block_output[94], block_output[95], block_output[96], block_output[97], block_output[98], block_output[99], block_output[100], block_output[101], block_output[102], block_output[103], block_output[104], block_output[105], block_output[106], block_output[107], block_output[108], block_output[109], block_output[110], block_output[111], block_output[112], block_output[113], block_output[114], block_output[115], block_output[116], block_output[117], block_output[118], block_output[119], block_output[120], block_output[121], block_output[122], block_output[123], block_output[124]};
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
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 27] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 28] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 29] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 30] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 31] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 32] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 33] * out_stride] += block_output[33][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 34] * out_stride] += block_output[34][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 35] * out_stride] += block_output[35][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 36] * out_stride] += block_output[36][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 37] * out_stride] += block_output[37][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 38] * out_stride] += block_output[38][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 39] * out_stride] += block_output[39][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 40] * out_stride] += block_output[40][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 41] * out_stride] += block_output[41][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 42] * out_stride] += block_output[42][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 43] * out_stride] += block_output[43][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 44] * out_stride] += block_output[44][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 45] * out_stride] += block_output[45][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 46] * out_stride] += block_output[46][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 47] * out_stride] += block_output[47][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 48] * out_stride] += block_output[48][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 49] * out_stride] += block_output[49][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 50] * out_stride] += block_output[50][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 51] * out_stride] += block_output[51][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 52] * out_stride] += block_output[52][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 53] * out_stride] += block_output[53][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 54] * out_stride] += block_output[54][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 55] * out_stride] += block_output[55][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 56] * out_stride] += block_output[56][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 57] * out_stride] += block_output[57][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 58] * out_stride] += block_output[58][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 59] * out_stride] += block_output[59][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 60] * out_stride] += block_output[60][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 61] * out_stride] += block_output[61][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 62] * out_stride] += block_output[62][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 63] * out_stride] += block_output[63][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 64] * out_stride] += block_output[64][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 65] * out_stride] += block_output[65][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 66] * out_stride] += block_output[66][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 67] * out_stride] += block_output[67][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 68] * out_stride] += block_output[68][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 69] * out_stride] += block_output[69][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 70] * out_stride] += block_output[70][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 71] * out_stride] += block_output[71][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 72] * out_stride] += block_output[72][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 73] * out_stride] += block_output[73][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 74] * out_stride] += block_output[74][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 75] * out_stride] += block_output[75][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 76] * out_stride] += block_output[76][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 77] * out_stride] += block_output[77][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 78] * out_stride] += block_output[78][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 79] * out_stride] += block_output[79][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 80] * out_stride] += block_output[80][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 81] * out_stride] += block_output[81][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 82] * out_stride] += block_output[82][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 83] * out_stride] += block_output[83][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 84] * out_stride] += block_output[84][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 85] * out_stride] += block_output[85][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 86] * out_stride] += block_output[86][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 87] * out_stride] += block_output[87][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 88] * out_stride] += block_output[88][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 89] * out_stride] += block_output[89][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 90] * out_stride] += block_output[90][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 91] * out_stride] += block_output[91][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 92] * out_stride] += block_output[92][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 93] * out_stride] += block_output[93][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 94] * out_stride] += block_output[94][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 95] * out_stride] += block_output[95][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 96] * out_stride] += block_output[96][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 97] * out_stride] += block_output[97][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 98] * out_stride] += block_output[98][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 99] * out_stride] += block_output[99][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 100] * out_stride] += block_output[100][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 101] * out_stride] += block_output[101][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 102] * out_stride] += block_output[102][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 103] * out_stride] += block_output[103][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 104] * out_stride] += block_output[104][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 105] * out_stride] += block_output[105][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 106] * out_stride] += block_output[106][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 107] * out_stride] += block_output[107][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 108] * out_stride] += block_output[108][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 109] * out_stride] += block_output[109][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 110] * out_stride] += block_output[110][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 111] * out_stride] += block_output[111][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 112] * out_stride] += block_output[112][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 113] * out_stride] += block_output[113][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 114] * out_stride] += block_output[114][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 115] * out_stride] += block_output[115][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 116] * out_stride] += block_output[116][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 117] * out_stride] += block_output[117][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 118] * out_stride] += block_output[118][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 119] * out_stride] += block_output[119][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 120] * out_stride] += block_output[120][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 121] * out_stride] += block_output[121][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 122] * out_stride] += block_output[122][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 123] * out_stride] += block_output[123][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[ev[scatter * N_SHAPE + 124] * out_stride] += block_output[124][scatter];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex125_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}
