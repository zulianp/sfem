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
struct laplace_proteus_hex64_affine_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[16] = {scalar_t(0.66000566507280345), scalar_t(0.52093768771170357), scalar_t(-0.23018790325073893), scalar_t(0.049244550466231808), scalar_t(0.0033737364327725092), scalar_t(1.0048858548256456), scalar_t(-0.0099213535723245878), scalar_t(0.0016617623139063829), scalar_t(0.0016617623139064115), scalar_t(-0.0099213535723247544), scalar_t(1.0048858548256461), scalar_t(0.0033737364327725652), scalar_t(0.049244550466231864), scalar_t(-0.23018790325073912), scalar_t(0.52093768771170401), scalar_t(0.66000566507280323)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[16] = {scalar_t(-4.3153073477037243), scalar_t(6.070808640937936), scalar_t(-2.1956952387646993), scalar_t(0.44019394553048763), scalar_t(-1.0300638443059633), scalar_t(-1.4397231632139631), scalar_t(2.9696378593458164), scalar_t(-0.49985085182588956), scalar_t(0.49985085182588984), scalar_t(-2.9696378593458164), scalar_t(1.4397231632139631), scalar_t(1.0300638443059635), scalar_t(-0.44019394553048752), scalar_t(2.1956952387646984), scalar_t(-6.0708086409379352), scalar_t(4.3153073477037225)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
};

template <typename scalar_t>
struct laplace_proteus_hex64_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[16] = {scalar_t(0.66000566507280345), scalar_t(0.52093768771170357), scalar_t(-0.23018790325073893), scalar_t(0.049244550466231808), scalar_t(0.0033737364327725092), scalar_t(1.0048858548256456), scalar_t(-0.0099213535723245878), scalar_t(0.0016617623139063829), scalar_t(0.0016617623139064115), scalar_t(-0.0099213535723247544), scalar_t(1.0048858548256461), scalar_t(0.0033737364327725652), scalar_t(0.049244550466231864), scalar_t(-0.23018790325073912), scalar_t(0.52093768771170401), scalar_t(0.66000566507280323)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[16] = {scalar_t(-4.3153073477037243), scalar_t(6.070808640937936), scalar_t(-2.1956952387646993), scalar_t(0.44019394553048763), scalar_t(-1.0300638443059633), scalar_t(-1.4397231632139631), scalar_t(2.9696378593458164), scalar_t(-0.49985085182588956), scalar_t(0.49985085182588984), scalar_t(-2.9696378593458164), scalar_t(1.4397231632139631), scalar_t(1.0300638443059635), scalar_t(-0.44019394553048752), scalar_t(2.1956952387646984), scalar_t(-6.0708086409379352), scalar_t(4.3153073477037225)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex64_residual_element_soa_diagnostics_data = {
    "laplace_proteus_hex64_residual_element_soa",
    "PROTEUS_HEX64",
    3,
    64,
    64,
    16,
    4,
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
    32,
    4,
    1,
    64,
    0,
    64,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex64_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex64_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex64_residual_element_soa",
            &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex64_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex64_residual_element_soa_float",
            &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex64_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex64_residual_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex64_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex64_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex64_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex64_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex64_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex64_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex64_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex64_jacobian_u_u_diagnostics_data = {
    "laplace_proteus_hex64_jacobian_u_u",
    "PROTEUS_HEX64",
    3,
    64,
    64,
    16,
    4,
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
    32,
    4,
    1,
    0,
    64,
    64,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex64_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_proteus_hex64_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex64_jacobian_u_u_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex64_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex64_jacobian_u_u",
            &sfem::codegen::laplace_proteus_hex64_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex64_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex64_jacobian_u_u_float",
            &sfem::codegen::laplace_proteus_hex64_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data = {
    "laplace_proteus_hex64_jacobian_action_element_soa",
    "PROTEUS_HEX64",
    3,
    64,
    64,
    16,
    4,
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
    32,
    4,
    1,
    0,
    64,
    64,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_proteus_hex64_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_proteus_hex64_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex64_jacobian_action_element_soa",
            &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex64_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_proteus_hex64_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex64_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex64_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex64_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_proteus_hex64_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_proteus_hex64_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int laplace_proteus_hex64_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[64],
        const double kappa,
        double *const SFEM_RESTRICT output[64]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<double, 64, 64, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<double>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex64_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[64],
        const float kappa,
        float *const SFEM_RESTRICT output[64]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<float, 64, 64, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<float>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex64_residual_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 64;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex64_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex64_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex64_affine_reference_data<scalar_t>::q_weight_1d();

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
        }

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33], block_current[34], block_current[35], block_current[36], block_current[37], block_current[38], block_current[39], block_current[40], block_current[41], block_current[42], block_current[43], block_current[44], block_current[45], block_current[46], block_current[47], block_current[48], block_current[49], block_current[50], block_current[51], block_current[52], block_current[53], block_current[54], block_current[55], block_current[56], block_current[57], block_current[58], block_current[59], block_current[60], block_current[61], block_current[62], block_current[63]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63]};
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
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex64_residual_affine_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex64_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex64_residual_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex64_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex64_residual_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 64;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<scalar_t>::q_weight_1d();

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
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[81], block_coordinates[82], block_coordinates[83], block_coordinates[84], block_coordinates[85], block_coordinates[86], block_coordinates[87], block_coordinates[88], block_coordinates[89], block_coordinates[90], block_coordinates[91], block_coordinates[92], block_coordinates[93], block_coordinates[94], block_coordinates[95], block_coordinates[96], block_coordinates[97], block_coordinates[98], block_coordinates[99], block_coordinates[100], block_coordinates[101], block_coordinates[102], block_coordinates[103], block_coordinates[104], block_coordinates[105], block_coordinates[106], block_coordinates[107], block_coordinates[108], block_coordinates[109], block_coordinates[110], block_coordinates[111], block_coordinates[112], block_coordinates[113], block_coordinates[114], block_coordinates[115], block_coordinates[116], block_coordinates[117], block_coordinates[118], block_coordinates[119], block_coordinates[120], block_coordinates[121], block_coordinates[122], block_coordinates[123], block_coordinates[124], block_coordinates[125], block_coordinates[126], block_coordinates[127], block_coordinates[128], block_coordinates[129], block_coordinates[130], block_coordinates[131], block_coordinates[132], block_coordinates[133], block_coordinates[134], block_coordinates[135], block_coordinates[136], block_coordinates[137], block_coordinates[138], block_coordinates[139], block_coordinates[140], block_coordinates[141], block_coordinates[142], block_coordinates[143], block_coordinates[144], block_coordinates[145], block_coordinates[146], block_coordinates[147], block_coordinates[148], block_coordinates[149], block_coordinates[150], block_coordinates[151], block_coordinates[152], block_coordinates[153], block_coordinates[154], block_coordinates[155], block_coordinates[156], block_coordinates[157], block_coordinates[158], block_coordinates[159], block_coordinates[160], block_coordinates[161], block_coordinates[162], block_coordinates[163], block_coordinates[164], block_coordinates[165], block_coordinates[166], block_coordinates[167], block_coordinates[168], block_coordinates[169], block_coordinates[170], block_coordinates[171], block_coordinates[172], block_coordinates[173], block_coordinates[174], block_coordinates[175], block_coordinates[176], block_coordinates[177], block_coordinates[178], block_coordinates[179], block_coordinates[180], block_coordinates[181], block_coordinates[182], block_coordinates[183], block_coordinates[184], block_coordinates[185], block_coordinates[186], block_coordinates[187], block_coordinates[188], block_coordinates[189], block_coordinates[190], block_coordinates[191]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33], block_current[34], block_current[35], block_current[36], block_current[37], block_current[38], block_current[39], block_current[40], block_current[41], block_current[42], block_current[43], block_current[44], block_current[45], block_current[46], block_current[47], block_current[48], block_current[49], block_current[50], block_current[51], block_current[52], block_current[53], block_current[54], block_current[55], block_current[56], block_current[57], block_current[58], block_current[59], block_current[60], block_current[61], block_current[62], block_current[63]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63]};
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
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex64_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex64_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex64_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex64_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex64_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex64_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT direction[64],
        const double kappa,
        double *const SFEM_RESTRICT output[64]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<double, 64, 64, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<double>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_proteus_hex64_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT direction[64],
        const float kappa,
        float *const SFEM_RESTRICT output[64]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<float, 64, 64, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<float>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_proteus_hex64_jacobian_action_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 64;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_proteus_hex64_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_proteus_hex64_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_proteus_hex64_affine_reference_data<scalar_t>::q_weight_1d();

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
        }

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26], block_direction[27], block_direction[28], block_direction[29], block_direction[30], block_direction[31], block_direction[32], block_direction[33], block_direction[34], block_direction[35], block_direction[36], block_direction[37], block_direction[38], block_direction[39], block_direction[40], block_direction[41], block_direction[42], block_direction[43], block_direction[44], block_direction[45], block_direction[46], block_direction[47], block_direction[48], block_direction[49], block_direction[50], block_direction[51], block_direction[52], block_direction[53], block_direction[54], block_direction[55], block_direction[56], block_direction[57], block_direction[58], block_direction[59], block_direction[60], block_direction[61], block_direction[62], block_direction[63]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63]};
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
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex64_jacobian_action_affine_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex64_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex64_jacobian_action_affine_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex64_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 64;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_proteus_hex64_isoparametric_reference_data<scalar_t>::q_weight_1d();

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
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[18], block_coordinates[19], block_coordinates[20], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[81], block_coordinates[82], block_coordinates[83], block_coordinates[84], block_coordinates[85], block_coordinates[86], block_coordinates[87], block_coordinates[88], block_coordinates[89], block_coordinates[90], block_coordinates[91], block_coordinates[92], block_coordinates[93], block_coordinates[94], block_coordinates[95], block_coordinates[96], block_coordinates[97], block_coordinates[98], block_coordinates[99], block_coordinates[100], block_coordinates[101], block_coordinates[102], block_coordinates[103], block_coordinates[104], block_coordinates[105], block_coordinates[106], block_coordinates[107], block_coordinates[108], block_coordinates[109], block_coordinates[110], block_coordinates[111], block_coordinates[112], block_coordinates[113], block_coordinates[114], block_coordinates[115], block_coordinates[116], block_coordinates[117], block_coordinates[118], block_coordinates[119], block_coordinates[120], block_coordinates[121], block_coordinates[122], block_coordinates[123], block_coordinates[124], block_coordinates[125], block_coordinates[126], block_coordinates[127], block_coordinates[128], block_coordinates[129], block_coordinates[130], block_coordinates[131], block_coordinates[132], block_coordinates[133], block_coordinates[134], block_coordinates[135], block_coordinates[136], block_coordinates[137], block_coordinates[138], block_coordinates[139], block_coordinates[140], block_coordinates[141], block_coordinates[142], block_coordinates[143], block_coordinates[144], block_coordinates[145], block_coordinates[146], block_coordinates[147], block_coordinates[148], block_coordinates[149], block_coordinates[150], block_coordinates[151], block_coordinates[152], block_coordinates[153], block_coordinates[154], block_coordinates[155], block_coordinates[156], block_coordinates[157], block_coordinates[158], block_coordinates[159], block_coordinates[160], block_coordinates[161], block_coordinates[162], block_coordinates[163], block_coordinates[164], block_coordinates[165], block_coordinates[166], block_coordinates[167], block_coordinates[168], block_coordinates[169], block_coordinates[170], block_coordinates[171], block_coordinates[172], block_coordinates[173], block_coordinates[174], block_coordinates[175], block_coordinates[176], block_coordinates[177], block_coordinates[178], block_coordinates[179], block_coordinates[180], block_coordinates[181], block_coordinates[182], block_coordinates[183], block_coordinates[184], block_coordinates[185], block_coordinates[186], block_coordinates[187], block_coordinates[188], block_coordinates[189], block_coordinates[190], block_coordinates[191]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26], block_direction[27], block_direction[28], block_direction[29], block_direction[30], block_direction[31], block_direction[32], block_direction[33], block_direction[34], block_direction[35], block_direction[36], block_direction[37], block_direction[38], block_direction[39], block_direction[40], block_direction[41], block_direction[42], block_direction[43], block_direction[44], block_direction[45], block_direction[46], block_direction[47], block_direction[48], block_direction[49], block_direction[50], block_direction[51], block_direction[52], block_direction[53], block_direction[54], block_direction[55], block_direction[56], block_direction[57], block_direction[58], block_direction[59], block_direction[60], block_direction[61], block_direction[62], block_direction[63]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33], block_output[34], block_output[35], block_output[36], block_output[37], block_output[38], block_output[39], block_output[40], block_output[41], block_output[42], block_output[43], block_output[44], block_output[45], block_output[46], block_output[47], block_output[48], block_output[49], block_output[50], block_output[51], block_output[52], block_output[53], block_output[54], block_output[55], block_output[56], block_output[57], block_output[58], block_output[59], block_output[60], block_output[61], block_output[62], block_output[63]};
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
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa(
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
    return sfem::codegen::laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float(
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
    return sfem::codegen::laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_proteus_hex64_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}
