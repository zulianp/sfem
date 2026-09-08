#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../two_phase_flow_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#if defined(__has_include)
#if __has_include("smesh_types.hpp")
#include "smesh_types.hpp"
#endif
#endif

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
struct two_phase_flow_proteus_hex8_affine_reference_data {
    static const s_t *shape_1d() {
        static const s_t data[6] = {s_t(0.8872983346207417), s_t(0.1127016653792583), s_t(0.5), s_t(0.5), s_t(0.1127016653792583), s_t(0.8872983346207417)};
        return data;
    }
    static const s_t *grad_1d() {
        static const s_t data[6] = {s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1)};
        return data;
    }
    static const s_t *q_weight_1d() {
        static const s_t data[3] = {s_t(0.27777777777777779), s_t(0.44444444444444442), s_t(0.27777777777777779)};
        return data;
    }
};

template <typename s_t>
struct two_phase_flow_proteus_hex8_isoparametric_reference_data {
    static const s_t *shape_1d() {
        static const s_t data[6] = {s_t(0.8872983346207417), s_t(0.1127016653792583), s_t(0.5), s_t(0.5), s_t(0.1127016653792583), s_t(0.8872983346207417)};
        return data;
    }
    static const s_t *grad_1d() {
        static const s_t data[6] = {s_t(-1), s_t(1), s_t(-1), s_t(1), s_t(-1), s_t(1)};
        return data;
    }
    static const s_t *q_weight_1d() {
        static const s_t data[3] = {s_t(0.27777777777777779), s_t(0.44444444444444442), s_t(0.27777777777777779)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data = {
    "two_phase_flow_proteus_hex8_residual_element_soa",
    "PROTEUS_HEX8",
    3,
    27,
    8,
    16,
    3,
    37,
    76,
    9,
    1,
    11,
    2,
    0,
    0,
    44,
    17,
    248,
    0,
    0,
    15,
    35,
    10,
    12,
    3,
    26,
    32,
    0,
    16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_residual_element_soa",
            &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_residual_element_soa_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_proteus_hex8_residual_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_proteus_hex8_residual_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_proteus_hex8_jacobian_p_w_p_w_diagnostics_data = {
    "two_phase_flow_proteus_hex8_jacobian_p_w_p_w",
    "PROTEUS_HEX8",
    3,
    27,
    8,
    16,
    3,
    29,
    66,
    9,
    1,
    5,
    1,
    0,
    0,
    32,
    17,
    204,
    0,
    0,
    16,
    34,
    10,
    12,
    3,
    19,
    16,
    16,
    16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_w_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_w_p_w",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_w_p_w_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_proteus_hex8_jacobian_p_w_p_c_diagnostics_data = {
    "two_phase_flow_proteus_hex8_jacobian_p_w_p_c",
    "PROTEUS_HEX8",
    3,
    27,
    8,
    16,
    3,
    18,
    42,
    8,
    0,
    6,
    1,
    0,
    0,
    29,
    12,
    150,
    0,
    0,
    11,
    27,
    10,
    12,
    3,
    19,
    16,
    16,
    16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_w_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_w_p_c",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_w_p_c_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_proteus_hex8_jacobian_p_c_p_w_diagnostics_data = {
    "two_phase_flow_proteus_hex8_jacobian_p_c_p_w",
    "PROTEUS_HEX8",
    3,
    27,
    8,
    16,
    3,
    16,
    45,
    10,
    0,
    5,
    0,
    0,
    0,
    31,
    10,
    146,
    0,
    0,
    9,
    25,
    10,
    12,
    3,
    21,
    16,
    16,
    16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_c_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_c_p_w",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_c_p_w_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_proteus_hex8_jacobian_p_c_p_c_diagnostics_data = {
    "two_phase_flow_proteus_hex8_jacobian_p_c_p_c",
    "PROTEUS_HEX8",
    3,
    27,
    8,
    16,
    3,
    30,
    66,
    10,
    0,
    6,
    0,
    0,
    0,
    34,
    16,
    182,
    0,
    0,
    15,
    35,
    10,
    12,
    3,
    21,
    16,
    16,
    16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_c_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_c_p_c",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_p_c_p_c_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data = {
    "two_phase_flow_proteus_hex8_jacobian_action_element_soa",
    "PROTEUS_HEX8",
    3,
    27,
    8,
    16,
    3,
    61,
    135,
    14,
    1,
    9,
    1,
    0,
    0,
    50,
    41,
    349,
    0,
    0,
    39,
    36,
    10,
    12,
    3,
    26,
    16,
    16,
    16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_action_element_soa",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_proteus_hex8_jacobian_action_element_soa_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_proteus_hex8_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[16],
        const double *const SFEM_RESTRICT previous[16],
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_d3_tensor_product_residual_block<double, 27, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<double>::q_weight_1d(), current, previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_proteus_hex8_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[16],
        const float *const SFEM_RESTRICT previous[16],
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_d3_tensor_product_residual_block<float, 27, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<float>::q_weight_1d(), current, previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_proteus_hex8_residual_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t K_4,
        const s_t K_5,
        const s_t K_6,
        const s_t K_7,
        const s_t K_8,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
        const s_t mu_w,
        const s_t p_wr,
        const s_t porosity,
        const s_t rho_w0,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT p_w,
        const s_t *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT p_w_old,
        const s_t *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT p_w_out,
        s_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 27;
    static constexpr int NS = 8;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_shape_1d = sfem::codegen::two_phase_flow_proteus_hex8_affine_reference_data<s_t>::shape_1d();
    const s_t *const affine_grad_1d = sfem::codegen::two_phase_flow_proteus_hex8_affine_reference_data<s_t>::grad_1d();
    const s_t *const affine_q_weight_1d = sfem::codegen::two_phase_flow_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_current[NC * NS][VS];
        s_t block_previous[NC * NS][VS];
        s_t block_output[NC * NS][VS];
        const s_t *const current_components[NC] = {p_w, p_c};
        const s_t *const previous_components[NC] = {p_w_old, p_c_old};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        const g_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evb, g_jacobian_adjugate1 + evb, g_jacobian_adjugate2 + evb, g_jacobian_adjugate3 + evb, g_jacobian_adjugate4 + evb, g_jacobian_adjugate5 + evb, g_jacobian_adjugate6 + evb, g_jacobian_adjugate7 + evb, g_jacobian_adjugate8 + evb, g_jacobian_determinant0 + evb};
        s_t block_affine_geometry_data[10][VS];
        const s_t *bageom_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = bageom_streams[component];
        }

        two_phase_flow_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(nelems, 0, bageom_streams[9], block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current, block_previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, block_output);

        s_t *const output_components[NC] = {p_w_out, p_c_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_proteus_hex8_residual_affine_mesh_soa(
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
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_proteus_hex8_residual_affine_mesh_soa_float(
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
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t K_4,
        const s_t K_5,
        const s_t K_6,
        const s_t K_7,
        const s_t K_8,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
        const s_t mu_w,
        const s_t p_wr,
        const s_t porosity,
        const s_t rho_w0,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT p_w,
        const s_t *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT p_w_old,
        const s_t *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT p_w_out,
        s_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 27;
    static constexpr int NS = 8;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_coordinates[3 * NS][VS];
        s_t block_adjugate_data[9][NQ * VS];
        s_t block_determinant[NQ * VS];
        s_t block_current[NC * NS][VS];
        s_t block_previous[NC * NS][VS];
        s_t block_output[NC * NS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_coordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const s_t *const current_components[NC] = {p_w, p_c};
        const s_t *const previous_components[NC] = {p_w_old, p_c_old};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const s_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        two_phase_flow_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(nelems, VS, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current, block_previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, block_output);

        s_t *const output_components[NC] = {p_w_out, p_c_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16], parameters[17], parameters[18], parameters[19], parameters[20], parameters[21], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16], parameters[17], parameters[18], parameters[19], parameters[20], parameters[21], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[16],
        const double *const SFEM_RESTRICT direction[16],
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_d3_tensor_product_jacobian_action_block<double, 27, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<double>::q_weight_1d(), current, direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[16],
        const float *const SFEM_RESTRICT direction[16],
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_d3_tensor_product_jacobian_action_block<float, 27, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<float>::q_weight_1d(), current, direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t K_4,
        const s_t K_5,
        const s_t K_6,
        const s_t K_7,
        const s_t K_8,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
        const s_t mu_w,
        const s_t p_wr,
        const s_t porosity,
        const s_t rho_w0,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT p_w,
        const s_t *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT p_w_direction,
        const s_t *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT p_w_out,
        s_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 27;
    static constexpr int NS = 8;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const affine_shape_1d = sfem::codegen::two_phase_flow_proteus_hex8_affine_reference_data<s_t>::shape_1d();
    const s_t *const affine_grad_1d = sfem::codegen::two_phase_flow_proteus_hex8_affine_reference_data<s_t>::grad_1d();
    const s_t *const affine_q_weight_1d = sfem::codegen::two_phase_flow_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_current[NC * NS][VS];
        s_t block_direction[NC * NS][VS];
        s_t block_output[NC * NS][VS];
        const s_t *const current_components[NC] = {p_w, p_c};
        const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        const g_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evb, g_jacobian_adjugate1 + evb, g_jacobian_adjugate2 + evb, g_jacobian_adjugate3 + evb, g_jacobian_adjugate4 + evb, g_jacobian_adjugate5 + evb, g_jacobian_adjugate6 + evb, g_jacobian_adjugate7 + evb, g_jacobian_adjugate8 + evb, g_jacobian_determinant0 + evb};
        s_t block_affine_geometry_data[10][VS];
        const s_t *bageom_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = bageom_streams[component];
        }

        two_phase_flow_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(nelems, 0, bageom_streams[9], block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current, block_direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, block_output);

        s_t *const output_components[NC] = {p_w_out, p_c_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa(
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
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t K_4,
        const s_t K_5,
        const s_t K_6,
        const s_t K_7,
        const s_t K_8,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
        const s_t mu_w,
        const s_t p_wr,
        const s_t porosity,
        const s_t rho_w0,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT p_w,
        const s_t *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT p_w_direction,
        const s_t *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT p_w_out,
        s_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 27;
    static constexpr int NS = 8;
    static constexpr int NC = 2;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_coordinates[3 * NS][VS];
        s_t block_adjugate_data[9][NQ * VS];
        s_t block_determinant[NQ * VS];
        s_t block_current[NC * NS][VS];
        s_t block_direction[NC * NS][VS];
        s_t block_output[NC * NS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_coordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const s_t *const current_components[NC] = {p_w, p_c};
        const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const s_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        two_phase_flow_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(nelems, VS, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current, block_direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, block_output);

        s_t *const output_components[NC] = {p_w_out, p_c_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16], parameters[17], parameters[18], parameters[19], parameters[20], parameters[21], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16], parameters[17], parameters[18], parameters[19], parameters[20], parameters[21], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}
