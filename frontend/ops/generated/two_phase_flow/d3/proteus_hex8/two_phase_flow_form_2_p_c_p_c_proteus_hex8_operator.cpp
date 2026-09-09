#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../two_phase_flow_form_2_p_c_p_c_d3_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/line_p1_q3.hpp"
#include "../../../reference/quad_line_q3.hpp"
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

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    double *const RSTR output[16]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_tensor_product_residual_block<double, 27, 8, 16>(ne, geometry_stride, determinant, sfem::codegen::ref_line_p1_q3<double>::shape_1d(), sfem::codegen::quad_line_q3<double>::q_weight_1d(), output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    float *const RSTR output[16]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_tensor_product_residual_block<float, 27, 8, 16>(ne, geometry_stride, determinant, sfem::codegen::ref_line_p1_q3<float>::shape_1d(), sfem::codegen::quad_line_q3<float>::q_weight_1d(), output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 27;
  static constexpr int NS = 8;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q3<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q3<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q3<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t boutput[NC * NS][VS];

    for (int stream = 0; stream < 16; ++stream) {
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

    two_phase_flow_form_2_p_c_p_c_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape_1d, affine_q_weight_1d, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 27;
  static constexpr int NS = 8;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q3<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q3<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q3<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
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

    for (int stream = 0; stream < 16; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);


    two_phase_flow_form_2_p_c_p_c_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_maos(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double *const RSTR parameters,
    double *const RSTR output
) {
  return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_maos_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float *const RSTR parameters,
    float *const RSTR output
) {
  return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa_float(nelements, nnodes, elements, points, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[9],
    const double *const RSTR current[16],
    const double *const RSTR direction[16],
    const double C_ka1,
    const double C_ka2,
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
    const double m,
    const double mu_c,
    const double porosity,
    double *const RSTR output[16]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block<double, 27, 8, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::ref_line_p1_q3<double>::shape_1d(), sfem::codegen::ref_line_p1_q3<double>::grad_1d(), sfem::codegen::quad_line_q3<double>::q_weight_1d(), current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[9],
    const float *const RSTR current[16],
    const float *const RSTR direction[16],
    const float C_ka1,
    const float C_ka2,
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
    const float m,
    const float mu_c,
    const float porosity,
    float *const RSTR output[16]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block<float, 27, 8, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::ref_line_p1_q3<float>::shape_1d(), sfem::codegen::ref_line_p1_q3<float>::grad_1d(), sfem::codegen::quad_line_q3<float>::q_weight_1d(), current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_adj4,
    const g_t *const RSTR g_adj5,
    const g_t *const RSTR g_adj6,
    const g_t *const RSTR g_adj7,
    const g_t *const RSTR g_adj8,
    const g_t *const RSTR g_det0,
    const s_t C_ka1,
    const s_t C_ka2,
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
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
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
  static constexpr int ND = 3;
  static constexpr int NQ = 27;
  static constexpr int NS = 8;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q3<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q3<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q3<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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

    for (int stream = 0; stream < 16; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[10] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_adj4 + evb, g_adj5 + evb, g_adj6 + evb, g_adj7 + evb, g_adj8 + evb, g_det0 + evb};
    s_t baffine_geometry_data[10][VS];
    const s_t *bageom_streams[10];
    for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double C_ka1,
    const double C_ka2,
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
    const double m,
    const double mu_c,
    const double porosity,
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
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float C_ka1,
    const float C_ka2,
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
    const float m,
    const float mu_c,
    const float porosity,
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
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t C_ka1,
    const s_t C_ka2,
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
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
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
  static constexpr int ND = 3;
  static constexpr int NQ = 27;
  static constexpr int NS = 8;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q3<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q3<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q3<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
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
      const idx_t *const RSTR element_shape = elements[shape];
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

    for (int stream = 0; stream < 16; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double C_ka1,
    const double C_ka2,
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
    const double m,
    const double mu_c,
    const double porosity,
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
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_impl<double>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float C_ka1,
    const float C_ka2,
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
    const float m,
    const float mu_c,
    const float porosity,
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
  return sfem::codegen::two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_impl<float>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_maos(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double *const RSTR parameters,
    const double *const RSTR current,
    const double *const RSTR direction,
    double *const RSTR output
) {
  return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16], parameters[17], parameters[18], parameters[20], parameters[21], parameters[24], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_maos_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float *const RSTR parameters,
    const float *const RSTR current,
    const float *const RSTR direction,
    float *const RSTR output
) {
  return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], parameters[16], parameters[17], parameters[18], parameters[20], parameters[21], parameters[24], 2, current + 0, current + 1, 2, direction + 0, direction + 1, 2, output + 0, output + 1);
}
