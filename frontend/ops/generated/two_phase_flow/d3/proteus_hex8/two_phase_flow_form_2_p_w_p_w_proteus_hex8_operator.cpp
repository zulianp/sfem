#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include "../two_phase_flow_form_2_p_w_p_w_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"

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
struct two_phase_flow_form_2_p_w_p_w_proteus_hex8_affine_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[6] = {scalar_t(0.8872983346207417), scalar_t(0.1127016653792583), scalar_t(0.5), scalar_t(0.5), scalar_t(0.1127016653792583), scalar_t(0.8872983346207417)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[6] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
        return data;
    }
};

template <typename scalar_t>
struct two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[6] = {scalar_t(0.8872983346207417), scalar_t(0.1127016653792583), scalar_t(0.5), scalar_t(0.5), scalar_t(0.1127016653792583), scalar_t(0.8872983346207417)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[6] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
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

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_tensor_product_residual_block<double, 27, 8, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<double>::q_weight_1d(), output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_tensor_product_residual_block<float, 27, 8, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<float>::q_weight_1d(), output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const jacobian_t *const affine_geometry_sources[1] = {g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[1][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[1];
        for (int geometry_stream = 0; geometry_stream < 1; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }

        two_phase_flow_form_2_p_w_p_w_d3_tensor_product_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[0], affine_shape_1d, affine_q_weight_1d, block_output);

        scalar_t *const output_components[N_FIELDS] = {p_w_out, p_c_out};
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

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
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

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);


        two_phase_flow_form_2_p_w_p_w_d3_tensor_product_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, isoparametric_shape_1d, isoparametric_q_weight_1d, block_output);

        scalar_t *const output_components[N_FIELDS] = {p_w_out, p_c_out};
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

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[16],
        const double *const SFEM_RESTRICT direction[16],
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
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        double *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_tensor_product_jacobian_action_block<double, 27, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<double>::q_weight_1d(), current, direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[16],
        const float *const SFEM_RESTRICT direction[16],
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
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        float *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_tensor_product_jacobian_action_block<float, 27, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<float>::q_weight_1d(), current, direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_impl(
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
        const scalar_t C_kw1,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        const scalar_t P_r,
        const scalar_t S_res,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT p_w,
        const scalar_t *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT p_w_direction,
        const scalar_t *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        const scalar_t *const current_components[N_FIELDS] = {p_w, p_c};
        const scalar_t *const direction_components[N_FIELDS] = {p_w_direction, p_c_direction};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const jacobian_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[10][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        two_phase_flow_form_2_p_w_p_w_d3_tensor_product_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current, block_direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output);

        scalar_t *const output_components[N_FIELDS] = {p_w_out, p_c_out};
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

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa(
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
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t C_kw1,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        const scalar_t P_r,
        const scalar_t S_res,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT p_w,
        const scalar_t *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT p_w_direction,
        const scalar_t *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
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
        const scalar_t *const current_components[N_FIELDS] = {p_w, p_c};
        const scalar_t *const direction_components[N_FIELDS] = {p_w_direction, p_c_direction};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        two_phase_flow_form_2_p_w_p_w_d3_tensor_product_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current, block_direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output);

        scalar_t *const output_components[N_FIELDS] = {p_w_out, p_c_out};
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

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[13], parameters[15], parameters[18], parameters[19], parameters[20], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[13], parameters[15], parameters[18], parameters[19], parameters[20], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

namespace sfem {
namespace codegen {

static SFEM_INLINE void two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_find_cols(
        const idx_t *const SFEM_RESTRICT targets,
        const idx_t *const SFEM_RESTRICT row,
        const int lenrow,
        idx_t *const SFEM_RESTRICT ks) {
#pragma unroll(8)
    for (int d = 0; d < 8; ++d) {
        ks[d] = 0;
    }
    for (int k = 0; k < lenrow; ++k) {
#pragma unroll(8)
        for (int d = 0; d < 8; ++d) {
            ks[d] += row[k] < targets[d];
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_scatter_crs(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_ROW_STREAMS = 8;
    static constexpr int N_COL_STREAMS = 8;
    static constexpr int ROW_COMPONENT[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr int ROW_SHAPE[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    static constexpr int COL_COMPONENT[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr int COL_SHAPE[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    count_t entries[N_SHAPE * N_SHAPE];
    idx_t ks[N_SHAPE];
    bool valid_graph = true;
    for (int i = 0; i < N_SHAPE; ++i) {
        const count_t row_begin = rowptr[ev[i]];
        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < N_SHAPE; ++j) {
            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {
                if (valid_graph) {
                    std::fprintf(stderr, "two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_scatter_crs missing graph entry (%ld, %ld)\n", (long)ev[i], (long)ev[j]);
                }
                entries[i * N_SHAPE + j] = row_begin;
                valid_graph = false;
            } else {
                entries[i * N_SHAPE + j] = row_begin + ks[j];
            }
        }
    }
    if (!valid_graph) return SFEM_FAILURE;
    for (int row_stream = 0; row_stream < N_ROW_STREAMS; ++row_stream) {
        const int row_shape = ROW_SHAPE[row_stream];
        const int bi = ROW_COMPONENT[row_stream];
        for (int col_stream = 0; col_stream < N_COL_STREAMS; ++col_stream) {
            const int col_shape = COL_SHAPE[col_stream];
            const int bj = COL_COMPONENT[col_stream];
            scalar_t *const block = &values[entries[row_shape * N_SHAPE + col_shape] * N_FIELDS * N_FIELDS];
#pragma omp atomic update
            block[bi * N_FIELDS + bj] += element_matrix[row_stream * N_COL_STREAMS + col_stream];
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t C_kw1,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        const scalar_t P_r,
        const scalar_t S_res,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT p_w,
        const scalar_t *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;
    static constexpr int VECTOR_SIZE = 1;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d();

    int invalid_matrix_graph = 0;
#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const ptrdiff_t evbegin = element;
        const int nelems = 1;
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[64];
        scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_STREAMS][VECTOR_SIZE];
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
            block_current[0 * N_SHAPE + shape][0] = p_w[node * current_stride];
            block_current[1 * N_SHAPE + shape][0] = p_c[node * current_stride];
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        for (int entry = 0; entry < 64; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }
        for (int trial_local = 0; trial_local < 8; ++trial_local) {
            const int trial = trial_local;
            for (int stream = 0; stream < N_STREAMS; ++stream) {
                block_direction[stream][0] = scalar_t(0);
                block_output[stream][0] = scalar_t(0);
            }
            block_direction[trial][0] = scalar_t(1);
            two_phase_flow_form_2_p_w_p_w_d3_tensor_product_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(1, 1, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current, block_direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output);
            for (int test_local = 0; test_local < 8; ++test_local) {
                const int test = test_local;
                element_matrix[test_local * 8 + trial_local] = block_output[test][0];
            }
        }

        invalid_matrix_graph |= (two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_scatter_crs(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);
    }

    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_crs_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, rowptr, colidx, values);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE void two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(
        const idx_t *const SFEM_RESTRICT ev,
        const ptrdiff_t out_stride,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        const ptrdiff_t element,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;
    const ptrdiff_t element_offset = element * N_STREAMS * N_STREAMS;
    for (int row_field = 0; row_field < N_FIELDS; ++row_field) {
        for (int row_shape = 0; row_shape < N_SHAPE; ++row_shape) {
            const int row_stream = row_field * N_SHAPE + row_shape;
            const idx_t global_row = ev[row_shape] * out_stride + row_field;
            for (int col_field = 0; col_field < N_FIELDS; ++col_field) {
                for (int col_shape = 0; col_shape < N_SHAPE; ++col_shape) {
                    const int col_stream = col_field * N_SHAPE + col_shape;
                    const ptrdiff_t entry = element_offset + row_stream * N_STREAMS + col_stream;
                    rows[entry] = global_row;
                    cols[entry] = ev[col_shape] * out_stride + col_field;
                    values[entry] = element_matrix[row_stream * N_STREAMS + col_stream];
                }
            }
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t C_kw1,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        const scalar_t P_r,
        const scalar_t S_res,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT p_w,
        const scalar_t *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;
    static constexpr int VECTOR_SIZE = 1;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const ptrdiff_t evbegin = element;
        const int nelems = 1;
        idx_t ev[N_SHAPE];
        scalar_t element_matrix[N_STREAMS * N_STREAMS];
        scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_STREAMS][VECTOR_SIZE];
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
            block_current[0 * N_SHAPE + shape][0] = p_w[node * current_stride];
            block_current[1 * N_SHAPE + shape][0] = p_c[node * current_stride];
        }

        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 0,
                coordinate_grad_ref + 0 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 1,
                coordinate_grad_ref + 1 * N_QP * DIM * VECTOR_SIZE);
        tensor_gradient_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinates, 2,
                coordinate_grad_ref + 2 * N_QP * DIM * VECTOR_SIZE);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        for (int entry = 0; entry < N_STREAMS * N_STREAMS; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }
        static constexpr int TENSOR_STREAMS[N_STREAMS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        for (int trial = 0; trial < N_STREAMS; ++trial) {
            const int tensor_trial = TENSOR_STREAMS[trial];
            for (int stream = 0; stream < N_STREAMS; ++stream) {
                block_direction[stream][0] = scalar_t(0);
                block_output[stream][0] = scalar_t(0);
            }
            block_direction[tensor_trial][0] = scalar_t(1);
            two_phase_flow_form_2_p_w_p_w_d3_tensor_product_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(1, 1, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current, block_direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output);
            for (int test = 0; test < N_STREAMS; ++test) {
                const int tensor_test = TENSOR_STREAMS[test];
                element_matrix[test * N_STREAMS + trial] = block_output[tensor_test][0];
            }
        }

        two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(ev, out_stride, element_matrix, element, rows, cols, values);
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, out_stride, rows, cols, values);
}
