#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../two_phase_flow_form_2_p_c_p_c_d2_simplex_local.hpp"
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
struct two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[18] = {scalar_t(0.10810301816807022), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};
        return data;
    }
};

template <typename scalar_t>
struct two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[18] = {scalar_t(0.10810301816807022), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data = {
    "two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa",
    "TRI3",
    2,
    6,
    3,
    16,
    4,
    27,
    64,
    9,
    1,
    11,
    2,
    0,
    0,
    35,
    17,
    226,
    0,
    0,
    15,
    28,
    5,
    54,
    6,
    21,
    12,
    0,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_diagnostics_data = {
    "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w",
    "TRI3",
    2,
    6,
    3,
    16,
    4,
    19,
    53,
    9,
    1,
    5,
    1,
    0,
    0,
    24,
    17,
    181,
    0,
    0,
    16,
    27,
    5,
    54,
    6,
    14,
    6,
    6,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_diagnostics_data = {
    "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c",
    "TRI3",
    2,
    6,
    3,
    16,
    4,
    13,
    36,
    8,
    0,
    6,
    1,
    0,
    0,
    22,
    12,
    139,
    0,
    0,
    11,
    20,
    5,
    54,
    6,
    14,
    6,
    6,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_diagnostics_data = {
    "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w",
    "TRI3",
    2,
    6,
    3,
    16,
    4,
    11,
    39,
    10,
    0,
    5,
    0,
    0,
    0,
    24,
    10,
    135,
    0,
    0,
    9,
    21,
    5,
    54,
    6,
    16,
    6,
    6,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_diagnostics_data = {
    "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c",
    "TRI3",
    2,
    6,
    3,
    16,
    4,
    20,
    53,
    10,
    0,
    6,
    0,
    0,
    0,
    26,
    16,
    159,
    0,
    0,
    15,
    28,
    5,
    54,
    6,
    16,
    6,
    6,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data = {
    "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa",
    "TRI3",
    2,
    6,
    3,
    16,
    4,
    41,
    109,
    14,
    1,
    9,
    1,
    0,
    0,
    39,
    41,
    303,
    0,
    0,
    39,
    29,
    5,
    54,
    6,
    21,
    6,
    6,
    6,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[6]
) {
    sfem::codegen::two_phase_flow_form_2_p_c_p_c_d2_simplex_residual_block<double, 6, 3, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<double>::q_weight(), output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[6]
) {
    sfem::codegen::two_phase_flow_form_2_p_c_p_c_d2_simplex_residual_block<float, 6, 3, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<float>::q_weight(), output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_q_weight = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        for (int stream = 0; stream < 6; ++stream) {
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

        two_phase_flow_form_2_p_c_p_c_d2_simplex_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[0], affine_shape, affine_q_weight, block_output);

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

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[2 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[4][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1]};
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

        for (int stream = 0; stream < 6; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }


        two_phase_flow_form_2_p_c_p_c_d2_simplex_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, isoparametric_shape, isoparametric_q_weight, block_output);

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

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[4],
        const double *const SFEM_RESTRICT current[6],
        const double *const SFEM_RESTRICT direction[6],
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        double *const SFEM_RESTRICT output[6]
) {
    sfem::codegen::two_phase_flow_form_2_p_c_p_c_d2_simplex_jacobian_action_block<double, 6, 3, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<double>::q_weight(), current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[4],
        const float *const SFEM_RESTRICT current[6],
        const float *const SFEM_RESTRICT direction[6],
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        float *const SFEM_RESTRICT output[6]
) {
    sfem::codegen::two_phase_flow_form_2_p_c_p_c_d2_simplex_jacobian_action_block<float, 6, 3, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<float>::q_weight(), current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t C_ka1,
        const scalar_t C_ka2,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t M_c,
        const scalar_t P_r,
        const scalar_t R,
        const scalar_t S_res,
        const scalar_t T,
        const scalar_t Z,
        const scalar_t dt,
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t porosity,
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
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_q_weight = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_affine_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 6; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const jacobian_t *const affine_geometry_sources[5] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[5][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[5];
        for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[4];
        for (int component = 0; component < 4; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        two_phase_flow_form_2_p_c_p_c_d2_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[4], block_adjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, block_current, block_direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, block_output);

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

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
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
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
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
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t C_ka1,
        const scalar_t C_ka2,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t M_c,
        const scalar_t P_r,
        const scalar_t R,
        const scalar_t S_res,
        const scalar_t T,
        const scalar_t Z,
        const scalar_t dt,
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t porosity,
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
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int N_SHAPE = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[2 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[4][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1]};
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

        for (int stream = 0; stream < 6; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * N_SHAPE + 2];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * N_SHAPE + 2];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const block_adjugate[4] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};

        two_phase_flow_form_2_p_c_p_c_d2_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, block_current, block_direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, block_output);

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

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
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
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
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
    return sfem::codegen::two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[15], parameters[16], parameters[19], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[15], parameters[16], parameters[19], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}
