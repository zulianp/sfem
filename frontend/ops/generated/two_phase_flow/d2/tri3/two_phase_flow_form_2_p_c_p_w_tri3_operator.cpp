#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../two_phase_flow_form_2_p_c_p_w_d2_simplex_local.hpp"
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


template <typename s_t>
struct two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data {
  static const s_t *shape() {
    static const s_t data[18] = {s_t(0.10810301816807022), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[18] = {s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[18] = {s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[6] = {s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.054975871827660998), s_t(0.054975871827660998), s_t(0.054975871827660998)};
    return data;
  }
};

template <typename s_t>
struct two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data {
  static const s_t *shape() {
    static const s_t data[18] = {s_t(0.10810301816807022), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[18] = {s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[18] = {s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[6] = {s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.054975871827660998), s_t(0.054975871827660998), s_t(0.054975871827660998)};
    return data;
  }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_isoparametric_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_residual_isoparametric_mesh_soa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_isoparametric_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_residual_isoparametric_mesh_soa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_isoparametric_mesh_soa",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_isoparametric_mesh_soa_float",
      &sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    double *const RSTR output[6]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_w_d2_simplex_residual_block<double, 6, 3, 16>(ne, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<double>::q_weight(), output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    float *const RSTR output[6]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_w_d2_simplex_residual_block<float, 6, 3, 16>(ne, geometry_stride, determinant, sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<float>::q_weight(), output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_q_weight = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t boutput[NC * NS][VS];

    for (int stream = 0; stream < 6; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1][VS];
    const s_t *bageom_streams[1];
    for (int geometry_stream = 0; geometry_stream < 1; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }

    two_phase_flow_form_2_p_c_p_w_d2_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape, affine_q_weight, boutput);

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

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[4],
    const double *const RSTR current[6],
    const double *const RSTR direction[6],
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
    double *const RSTR output[6]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_w_d2_simplex_jacobian_action_block<double, 6, 3, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<double>::q_weight(), current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[4],
    const float *const RSTR current[6],
    const float *const RSTR direction[6],
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
    float *const RSTR output[6]
) {
  sfem::codegen::two_phase_flow_form_2_p_c_p_w_d2_simplex_jacobian_action_block<float, 6, 3, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_isoparametric_reference_data<float>::q_weight(), current, direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
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
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_q_weight = sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_affine_reference_data<s_t>::q_weight();

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

    for (int stream = 0; stream < 6; ++stream) {
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

    two_phase_flow_form_2_p_c_p_w_d2_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[4], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

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

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
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
    const double *const RSTR p_w,
    const double *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const double *const RSTR p_w_direction,
    const double *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
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
    const float *const RSTR p_w,
    const float *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const float *const RSTR p_w_direction,
    const float *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}
