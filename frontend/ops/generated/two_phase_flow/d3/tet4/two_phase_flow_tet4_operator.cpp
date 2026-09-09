#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../two_phase_flow_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/quad_tet_q11.hpp"
#include "../../../reference/tet4_q11.hpp"
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
struct two_phase_flow_tet4_affine_reference_data {
  static const s_t *shape() { return ref_tet4_q11<s_t>::shape(); }
  static const s_t *grad_ref_x() { return ref_tet4_q11<s_t>::grad_ref_x(); }
  static const s_t *grad_ref_y() { return ref_tet4_q11<s_t>::grad_ref_y(); }
  static const s_t *grad_ref_z() { return ref_tet4_q11<s_t>::grad_ref_z(); }
  static const s_t *q_weight() { return quad_tet_q11<s_t>::q_weight(); }
};

template <typename s_t>
struct two_phase_flow_tet4_isoparametric_reference_data {
  static const s_t *shape() { return ref_tet4_q11<s_t>::shape(); }
  static const s_t *grad_ref_x() { return ref_tet4_q11<s_t>::grad_ref_x(); }
  static const s_t *grad_ref_y() { return ref_tet4_q11<s_t>::grad_ref_y(); }
  static const s_t *grad_ref_z() { return ref_tet4_q11<s_t>::grad_ref_z(); }
  static const s_t *q_weight() { return quad_tet_q11<s_t>::q_weight(); }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_tet4_residual_esoa_diagnostics_data = {
  "two_phase_flow_tet4_residual_esoa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data;
}

extern "C" double two_phase_flow_tet4_residual_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_tet4_residual_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_residual_esoa",
      &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_residual_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_residual_esoa_float",
      &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_tet4_residual_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_tet4_residual_a_msoa",
      &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_residual_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_tet4_residual_a_msoa_float",
      &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_tet4_residual_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_tet4_residual_i_msoa",
      &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_residual_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_tet4_residual_i_msoa_float",
      &sfem::codegen::two_phase_flow_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_tet4_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_tet4_jacobian_p_w_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_tet4_jacobian_p_w_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_w_p_w",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_w_p_w_float",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_tet4_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_tet4_jacobian_p_w_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_tet4_jacobian_p_w_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_w_p_c",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_w_p_c_float",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_w_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_tet4_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_tet4_jacobian_p_c_p_w",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_w_diagnostics_data;
}

extern "C" double two_phase_flow_tet4_jacobian_p_c_p_w_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_w_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_w_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_c_p_w",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_w_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_c_p_w_float",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_w_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_tet4_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_tet4_jacobian_p_c_p_c",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_c_diagnostics_data;
}

extern "C" double two_phase_flow_tet4_jacobian_p_c_p_c_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_c_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_c_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_c_p_c",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_c_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_p_c_p_c_float",
      &sfem::codegen::two_phase_flow_tet4_jacobian_p_c_p_c_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data = {
  "two_phase_flow_tet4_jacobian_action_esoa",
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

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data;
}

extern "C" double two_phase_flow_tet4_jacobian_action_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void two_phase_flow_tet4_jacobian_action_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_action_esoa",
      &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_jacobian_action_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "two_phase_flow_tet4_jacobian_action_esoa_float",
      &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_tet4_jacobian_action_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_tet4_jacobian_action_a_msoa",
      &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_jacobian_action_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "two_phase_flow_tet4_jacobian_action_a_msoa_float",
      &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void two_phase_flow_tet4_jacobian_action_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_tet4_jacobian_action_i_msoa",
      &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void two_phase_flow_tet4_jacobian_action_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "two_phase_flow_tet4_jacobian_action_i_msoa_float",
      &sfem::codegen::two_phase_flow_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int two_phase_flow_tet4_residual_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[9],
    const double *const RSTR current[8],
    const double *const RSTR previous[8],
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
    double *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_d3_simplex_residual_block<double, 11, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::q_weight(), current, previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_tet4_residual_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[9],
    const float *const RSTR current[8],
    const float *const RSTR previous[8],
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
    float *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_d3_simplex_residual_block<float, 11, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::q_weight(), current, previous, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_tet4_residual_a_msoa_impl(
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
    const s_t *const RSTR p_w,
    const s_t *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR p_w_old,
    const s_t *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const previous_components[NC] = {p_w_old, p_c_old};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bprevious[stream][lane] = previous_components[field][node * previous_stride];
        }
      }
    }

    for (int stream = 0; stream < 8; ++stream) {
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

    two_phase_flow_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, boutput);

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

extern "C" int two_phase_flow_tet4_residual_a_msoa(
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
    const double *const RSTR p_w,
    const double *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const double *const RSTR p_w_old,
    const double *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_tet4_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_tet4_residual_a_msoa_float(
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
    const float *const RSTR p_w,
    const float *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const float *const RSTR p_w_old,
    const float *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_tet4_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_tet4_jacobian_action_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[9],
    const double *const RSTR current[8],
    const double *const RSTR direction[8],
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
    double *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_d3_simplex_jacobian_action_block<double, 11, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<double>::q_weight(), current, direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
  return SFEM_SUCCESS;
}

extern "C" int two_phase_flow_tet4_jacobian_action_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[9],
    const float *const RSTR current[8],
    const float *const RSTR direction[8],
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
    float *const RSTR output[8]
) {
  sfem::codegen::two_phase_flow_d3_simplex_jacobian_action_block<float, 11, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::two_phase_flow_tet4_isoparametric_reference_data<float>::q_weight(), current, direction, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int two_phase_flow_tet4_jacobian_action_a_msoa_impl(
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
  static constexpr int NQ = 11;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::two_phase_flow_tet4_affine_reference_data<s_t>::q_weight();

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

    for (int stream = 0; stream < 8; ++stream) {
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

    two_phase_flow_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bdirection, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, boutput);

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

extern "C" int two_phase_flow_tet4_jacobian_action_a_msoa(
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
    const double *const RSTR p_w,
    const double *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const double *const RSTR p_w_direction,
    const double *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    double *const RSTR p_w_out,
    double *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_tet4_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_tet4_jacobian_action_a_msoa_float(
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
    const float *const RSTR p_w,
    const float *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const float *const RSTR p_w_direction,
    const float *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    float *const RSTR p_w_out,
    float *const RSTR p_c_out
) {
  return sfem::codegen::two_phase_flow_tet4_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}
