#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../two_phase_flow_form_2_p_w_p_c_d2_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/line_p1_q4.hpp"
#include "../../../reference/quad_line_q4.hpp"
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
    const g_t *const RSTR source,
    s_t *const RSTR,
    std::true_type) {
  return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
    const int ne,
    const g_t *const RSTR source,
    s_t *const RSTR converted,
    std::false_type) {
  #pragma omp simd
  for (int lane = 0; lane < ne; ++lane) {
    converted[lane] = s_t(source[lane]);
  }
  return converted;
}

} // namespace codegen
} // namespace sfem
namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  27,
  44,
  9,
  1,
  11,
  2,
  0,
  0,
  35,
  17,
  206,
  1584,
  2144,
  15,
  28,
  5,
  16,
  4,
  21,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  19,
  38,
  9,
  1,
  5,
  1,
  0,
  0,
  24,
  17,
  166,
  1584,
  2144,
  16,
  27,
  5,
  16,
  4,
  14,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  13,
  24,
  8,
  0,
  6,
  1,
  0,
  0,
  22,
  12,
  127,
  1584,
  2144,
  11,
  20,
  5,
  16,
  4,
  14,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  11,
  24,
  10,
  0,
  5,
  0,
  0,
  0,
  24,
  10,
  120,
  1584,
  2144,
  9,
  21,
  5,
  16,
  4,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  20,
  36,
  10,
  0,
  6,
  0,
  0,
  0,
  26,
  16,
  142,
  1584,
  2144,
  15,
  28,
  5,
  16,
  4,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  41,
  83,
  14,
  1,
  9,
  1,
  0,
  0,
  39,
  41,
  277,
  1584,
  2144,
  39,
  29,
  5,
  16,
  4,
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    double *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_form_2_p_w_p_c_d2_tensor_product_residual_block<double, 16, 4, 16>(ne, geometry_stride, determinant, sfem::codegen::ref_line_p1_q4<double>::shape_1d(), sfem::codegen::quad_line_q4<double>::q_weight_1d(), output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    float *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_form_2_p_w_p_c_d2_tensor_product_residual_block<float, 16, 4, 16>(ne, geometry_stride, determinant, sfem::codegen::ref_line_p1_q4<float>::shape_1d(), sfem::codegen::quad_line_q4<float>::q_weight_1d(), output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 16;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t boutput[NC * NS][VS];

    for (int stream = 0; stream < 8; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1][VS];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t, VS>(
        ne, affine_geometry_sources[0], baffine_geometry_data[0], std::is_same<g_t, s_t>());

    two_phase_flow_form_2_p_w_p_c_d2_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape_1d, affine_q_weight_1d, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 16;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[2 * NS][VS];
    s_t badjugate_data[4][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
        }
      }
    }

    for (int stream = 0; stream < 8; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);


    two_phase_flow_form_2_p_w_p_c_d2_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_maos(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double *const RSTR parameters,
    double *const RSTR output
) {
  return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_maos_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float *const RSTR parameters,
    float *const RSTR output
) {
  return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa_float(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[4],
    const double *const RSTR current[8],
    const double *const RSTR direction[8],
    const double C_kw1,
    const double K_0,
    const double K_1,
    const double K_2,
    const double K_3,
    const double P_r,
    const double S_res,
    const double dt,
    const double kappa_T,
    const double m,
    const double mu_w,
    const double p_wr,
    const double porosity,
    const double rho_w0,
    double *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_form_2_p_w_p_c_d2_tensor_product_jacobian_action_block<double, 16, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::ref_line_p1_q4<double>::shape_1d(), sfem::codegen::ref_line_p1_q4<double>::grad_1d(), sfem::codegen::quad_line_q4<double>::q_weight_1d(), current, direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[4],
    const float *const RSTR current[8],
    const float *const RSTR direction[8],
    const float C_kw1,
    const float K_0,
    const float K_1,
    const float K_2,
    const float K_3,
    const float P_r,
    const float S_res,
    const float dt,
    const float kappa_T,
    const float m,
    const float mu_w,
    const float p_wr,
    const float porosity,
    const float rho_w0,
    float *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_form_2_p_w_p_c_d2_tensor_product_jacobian_action_block<float, 16, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::ref_line_p1_q4<float>::shape_1d(), sfem::codegen::ref_line_p1_q4<float>::grad_1d(), sfem::codegen::quad_line_q4<float>::q_weight_1d(), current, direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    const ptrdiff_t current_stride,
    const s_t *const RSTR p_w,
    const s_t *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR p_w_direction,
    const s_t *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 16;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bdirection[stream][lane] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 8; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[5] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_det0 + evb};
    s_t baffine_geometry_data[5][VS];
    const s_t *bageom_streams[5];
    for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[4];
    for (int component = 0; component < 4; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    two_phase_flow_form_2_p_w_p_c_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[4], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bdirection, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const double C_kw1,
    const double K_0,
    const double K_1,
    const double K_2,
    const double K_3,
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
    const double *const RSTR p_w,
    const double *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const double *const RSTR p_w_direction,
    const double *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const float C_kw1,
    const float K_0,
    const float K_1,
    const float K_2,
    const float K_3,
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
    const float *const RSTR p_w,
    const float *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const float *const RSTR p_w_direction,
    const float *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    const ptrdiff_t current_stride,
    const s_t *const RSTR p_w,
    const s_t *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR p_w_direction,
    const s_t *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 16;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[4] = {elements[0], elements[1], elements[3], elements[2]};

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[2 * NS][VS];
    s_t badjugate_data[4][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bdirection[stream][lane] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 8; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

    const s_t *const badjugate[4] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};

    two_phase_flow_form_2_p_w_p_c_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bdirection, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double C_kw1,
    const double K_0,
    const double K_1,
    const double K_2,
    const double K_3,
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
    const double *const RSTR p_w,
    const double *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const double *const RSTR p_w_direction,
    const double *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_impl<double>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float C_kw1,
    const float K_0,
    const float K_1,
    const float K_2,
    const float K_3,
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
    const float *const RSTR p_w,
    const float *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const float *const RSTR p_w_direction,
    const float *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_impl<float>(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_maos(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double *const RSTR parameters,
    const double *const RSTR current,
    const double *const RSTR direction,
    double *const RSTR output
) {
  return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[8], parameters[10], parameters[13], parameters[14], parameters[15], parameters[17], parameters[18], parameters[19], parameters[20], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_maos_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float *const RSTR parameters,
    const float *const RSTR current,
    const float *const RSTR direction,
    float *const RSTR output
) {
  return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa_float(nelements, nnodes, elements, points, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[8], parameters[10], parameters[13], parameters[14], parameters[15], parameters[17], parameters[18], parameters[19], parameters[20], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}
