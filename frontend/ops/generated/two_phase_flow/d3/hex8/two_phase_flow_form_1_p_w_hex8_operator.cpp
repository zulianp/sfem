#include <type_traits>
#include "../two_phase_flow_form_1_p_w_d3_tensor_product_local.hpp"
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
struct two_phase_flow_form_1_p_w_hex8_affine_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[8] = {scalar_t(0.93056815579702623), scalar_t(0.069431844202973714), scalar_t(0.66999052179242813), scalar_t(0.33000947820757187), scalar_t(0.33000947820757187), scalar_t(0.66999052179242813), scalar_t(0.069431844202973769), scalar_t(0.93056815579702623)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[8] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
};

template <typename scalar_t>
struct two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[8] = {scalar_t(0.93056815579702623), scalar_t(0.069431844202973714), scalar_t(0.66999052179242813), scalar_t(0.33000947820757187), scalar_t(0.33000947820757187), scalar_t(0.66999052179242813), scalar_t(0.069431844202973769), scalar_t(0.93056815579702623)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[8] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
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

static const KernelDiagnostics two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data = {
    "two_phase_flow_form_1_p_w_hex8_residual_element_soa",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
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
    16,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_1_p_w_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_residual_element_soa",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_residual_element_soa_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_diagnostics_data = {
    "two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
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
    16,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_diagnostics_data = {
    "two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
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
    16,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_diagnostics_data = {
    "two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
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
    16,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_diagnostics_data = {
    "two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
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
    16,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data = {
    "two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
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
    16,
    4,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[16],
        const double *const SFEM_RESTRICT previous[16],
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
    sfem::codegen::two_phase_flow_form_1_p_w_d3_tensor_product_residual_block<double, 64, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<double>::q_weight_1d(), current, previous, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[16],
        const float *const SFEM_RESTRICT previous[16],
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
    sfem::codegen::two_phase_flow_form_1_p_w_d3_tensor_product_residual_block<float, 64, 8, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<float>::q_weight_1d(), current, previous, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_impl(
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
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT p_w_old,
        const scalar_t *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        const scalar_t *const current_components[N_FIELDS] = {p_w, p_c};
        const scalar_t *const previous_components[N_FIELDS] = {p_w_old, p_c_old};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[6], block_current[7], block_current[4], block_current[5], block_current[8], block_current[9], block_current[10], block_current[11], block_current[14], block_current[15], block_current[12], block_current[13]};
        const scalar_t *const block_previous_streams[N_FIELDS * N_SHAPE] = {block_previous[0], block_previous[1], block_previous[2], block_previous[3], block_previous[6], block_previous[7], block_previous[4], block_previous[5], block_previous[8], block_previous[9], block_previous[10], block_previous[11], block_previous[14], block_previous[15], block_previous[12], block_previous[13]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[6], block_output[7], block_output[4], block_output[5], block_output[8], block_output[9], block_output[10], block_output[11], block_output[14], block_output[15], block_output[12], block_output[13]};
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

        two_phase_flow_form_1_p_w_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current_streams, block_previous_streams, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output_streams);

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

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa(
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
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_float(
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
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_impl(
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
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT p_w_old,
        const scalar_t *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];
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
        const scalar_t *const previous_components[N_FIELDS] = {p_w_old, p_c_old};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 16; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[18], block_coordinates[19], block_coordinates[20]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[6], block_current[7], block_current[4], block_current[5], block_current[8], block_current[9], block_current[10], block_current[11], block_current[14], block_current[15], block_current[12], block_current[13]};
        const scalar_t *const block_previous_streams[N_FIELDS * N_SHAPE] = {block_previous[0], block_previous[1], block_previous[2], block_previous[3], block_previous[6], block_previous[7], block_previous[4], block_previous[5], block_previous[8], block_previous[9], block_previous[10], block_previous[11], block_previous[14], block_previous[15], block_previous[12], block_previous[13]};
        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[6], block_output[7], block_output[4], block_output[5], block_output[8], block_output[9], block_output[10], block_output[11], block_output[14], block_output[15], block_output[12], block_output[13]};
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        two_phase_flow_form_1_p_w_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current_streams, block_previous_streams, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output_streams);

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

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa(
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
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float(
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
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[13], parameters[15], parameters[18], parameters[19], parameters[20], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[13], parameters[15], parameters[18], parameters[19], parameters[20], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, previous + 0, previous + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_form_1_p_w_d3_tensor_product_jacobian_action_block<double, 64, 8, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<double>::q_weight_1d(), output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[16]
) {
    sfem::codegen::two_phase_flow_form_1_p_w_d3_tensor_product_jacobian_action_block<float, 64, 8, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<float>::q_weight_1d(), output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_affine_reference_data<scalar_t>::q_weight_1d();

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

        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[6], block_output[7], block_output[4], block_output[5], block_output[8], block_output[9], block_output[10], block_output[11], block_output[14], block_output[15], block_output[12], block_output[13]};
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        two_phase_flow_form_1_p_w_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, affine_shape_1d, affine_q_weight_1d, block_output_streams);

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

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int N_SHAPE = 8;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::two_phase_flow_form_1_p_w_hex8_isoparametric_reference_data<scalar_t>::q_weight_1d();

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

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[18], block_coordinates[19], block_coordinates[20]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[6], block_output[7], block_output[4], block_output[5], block_output[8], block_output[9], block_output[10], block_output[11], block_output[14], block_output[15], block_output[12], block_output[13]};

        two_phase_flow_form_1_p_w_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, isoparametric_shape_1d, isoparametric_q_weight_1d, block_output_streams);

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

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}
