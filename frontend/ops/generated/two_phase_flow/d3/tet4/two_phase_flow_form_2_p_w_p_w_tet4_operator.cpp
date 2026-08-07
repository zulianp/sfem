#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../two_phase_flow_form_2_p_w_p_w_d3_simplex_local.hpp"
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
struct two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[44] = {scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.78571428571428581), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571452), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.07142857142857148), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571508), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.10059642383320075), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.10059642383320078), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.10059642383320078), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[11] = {scalar_t(-0.013155555555555556), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887)};
        return data;
    }
};

template <typename scalar_t>
struct two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[44] = {scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.78571428571428581), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571452), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.07142857142857148), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571508), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.10059642383320075), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.10059642383320078), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.10059642383320078), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[11] = {scalar_t(-0.013155555555555556), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa",
    "TET4",
    3,
    11,
    4,
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
    176,
    11,
    26,
    16,
    0,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w",
    "TET4",
    3,
    11,
    4,
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
    176,
    11,
    19,
    8,
    8,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c",
    "TET4",
    3,
    11,
    4,
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
    176,
    11,
    19,
    8,
    8,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w",
    "TET4",
    3,
    11,
    4,
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
    176,
    11,
    21,
    8,
    8,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c",
    "TET4",
    3,
    11,
    4,
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
    176,
    11,
    21,
    8,
    8,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data = {
    "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa",
    "TET4",
    3,
    11,
    4,
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
    176,
    11,
    26,
    8,
    8,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        double *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_simplex_residual_block<double, 11, 4, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<double>::q_weight(), output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        float *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_simplex_residual_block<float, 11, 4, 16>(nelems, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<float>::q_weight(), output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        for (int stream = 0; stream < 8; ++stream) {
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

        two_phase_flow_form_2_p_w_p_w_d3_simplex_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[0], affine_shape, affine_q_weight, block_output);

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

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_determinant0, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_w_out,
        scalar_t *const SFEM_RESTRICT p_c_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 8; ++stream) {
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


        two_phase_flow_form_2_p_w_p_w_d3_simplex_residual_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, isoparametric_shape, isoparametric_q_weight, block_output);

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

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
) {
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[8],
        const double *const SFEM_RESTRICT direction[8],
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
        double *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_simplex_jacobian_action_block<double, 11, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<double>::q_weight(), current, direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[8],
        const float *const SFEM_RESTRICT direction[8],
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
        float *const SFEM_RESTRICT output[8]
) {
    sfem::codegen::two_phase_flow_form_2_p_w_p_w_d3_simplex_jacobian_action_block<float, 11, 4, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<float>::q_weight(), current, direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_impl(
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
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::shape();
    const scalar_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_affine_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 8; ++stream) {
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

        two_phase_flow_form_2_p_w_p_w_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, block_current, block_direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output);

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

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa(
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_float(
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_impl(
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
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_FIELDS = 2;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::shape();
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_isoparametric_reference_data<scalar_t>::q_weight();

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

        for (int stream = 0; stream < 8; ++stream) {
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

        two_phase_flow_form_2_p_w_p_w_d3_simplex_jacobian_action_block_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_current, block_direction, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, block_output);

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

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa(
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_float(
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
    return sfem::codegen::two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[13], parameters[15], parameters[18], parameters[19], parameters[20], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[13], parameters[15], parameters[18], parameters[19], parameters[20], parameters[22], parameters[23], parameters[24], parameters[25], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}
