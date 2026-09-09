#include <type_traits>
#include "../navier_stokes_form_1_u_d3_simplex_mixed_local.hpp"
#include "../../../reference/quad_tet_q11.hpp"
#include "../../../reference/tet10_q11.hpp"
#include "../../../reference/tet4_q11.hpp"
#include "../../../kernel_math.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif
#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

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

static const KernelDiagnostics navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data = {
  "navier_stokes_form_1_u_tet10_tet4_residual_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  32,
  45,
  1,
  0,
  0,
  0,
  0,
  0,
  36,
  11,
  85,
  6479,
  8910,
  7,
  21,
  10,
  616,
  11,
  7,
  68,
  0,
  34,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data;
}

extern "C" double navier_stokes_form_1_u_tet10_tet4_residual_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_1_u_tet10_tet4_residual_esoa",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_1_u_tet10_tet4_residual_esoa_float",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_1_u_tet10_tet4_residual_a_msoa",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_1_u_tet10_tet4_residual_a_msoa_float",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_1_u_tet10_tet4_residual_i_msoa",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_1_u_tet10_tet4_residual_i_msoa_float",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data = {
  "navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  29,
  56,
  1,
  0,
  0,
  0,
  0,
  0,
  33,
  21,
  93,
  6479,
  8910,
  17,
  26,
  10,
  616,
  11,
  0,
  0,
  0,
  34,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data;
}

extern "C" double navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_float",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_1_u_tet10_tet4_jacobian_action_a_msoa",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_1_u_tet10_tet4_jacobian_action_a_msoa_float",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_1_u_tet10_tet4_jacobian_action_i_msoa",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_1_u_tet10_tet4_jacobian_action_i_msoa_float",
      &sfem::codegen::navier_stokes_form_1_u_tet10_tet4_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_1_u_tet10_tet4_residual_affine_mesh_mixed_impl(
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
    const s_t convection_scale,
    const s_t dt,
    const s_t f0,
    const s_t f1,
    const s_t f2,
    const s_t nu,
    const s_t rho,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u_data[3],
    const s_t *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u_old_data[3],
    const s_t *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int CELL_NS = 10;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const field_shape[NC] = {sfem::codegen::ref_tet10_q11<s_t>::shape(), sfem::codegen::ref_tet4_q11<s_t>::shape()};
  const s_t *const fgref[NC * ND] = {sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::ref_tet4_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q11<s_t>::grad_ref_z()};

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[N_FIELD_STREAMS][VS];
    s_t bprevious[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

    for (int local_shape = 0; local_shape < 10; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 0 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = u_data[0][node * current_stride];
        bprevious[stream][lane] = u_old_data[0][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 10; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 10 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = u_data[1][node * current_stride];
        bprevious[stream][lane] = u_old_data[1][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 10; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 20 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = u_data[2][node * current_stride];
        bprevious[stream][lane] = u_old_data[2][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 4; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 30 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = p_data[node * current_stride];
        bprevious[stream][lane] = p_old_data[node * previous_stride];
      }
    }

    for (int stream = 0; stream < 34; ++stream) {
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
    const s_t *badjugate[ND * ND];
    for (int component = 0; component < ND * ND; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    navier_stokes_form_1_u_d3_simplex_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, 0, bageom_streams[9], badjugate, field_shape, fgref, sfem::codegen::quad_tet_q11<s_t>::q_weight(), bcurrent, bprevious, convection_scale, dt, f0, f1, f2, nu, rho, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 10; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 10; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 10 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[2];
      for (int local_shape = 0; local_shape < 10; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 20 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 4; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 30 + local_shape;
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

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_a_msoa(
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
    const double convection_scale,
    const double dt,
    const double f0,
    const double f1,
    const double f2,
    const double nu,
    const double rho,
    const ptrdiff_t current_stride,
    const double *const RSTR u_data[3],
    const double *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const double *const RSTR u_old_data[3],
    const double *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    double *const RSTR u_out[3],
    double *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_a_msoa_float(
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
    const float convection_scale,
    const float dt,
    const float f0,
    const float f1,
    const float f2,
    const float nu,
    const float rho,
    const ptrdiff_t current_stride,
    const float *const RSTR u_data[3],
    const float *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const float *const RSTR u_old_data[3],
    const float *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    float *const RSTR u_out[3],
    float *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t convection_scale,
    const s_t dt,
    const s_t f0,
    const s_t f1,
    const s_t f2,
    const s_t nu,
    const s_t rho,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u_data[3],
    const s_t *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u_old_data[3],
    const s_t *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int CELL_NS = 10;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_cell_grad_ref_2 = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[ND * CELL_NS][VS];
    s_t badjugate_data[ND * ND][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[N_FIELD_STREAMS][VS];
    s_t bprevious[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

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

    for (int local_shape = 0; local_shape < 10; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 0 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = u_data[0][node * current_stride];
        bprevious[stream][lane] = u_old_data[0][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 10; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 10 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = u_data[1][node * current_stride];
        bprevious[stream][lane] = u_old_data[1][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 10; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 20 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = u_data[2][node * current_stride];
        bprevious[stream][lane] = u_old_data[2][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 4; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 30 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bcurrent[stream][lane] = p_data[node * current_stride];
        bprevious[stream][lane] = p_old_data[node * previous_stride];
      }
    }

    for (int stream = 0; stream < 34; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = bcoordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS] + bcoordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + bcoordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + bcoordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + bcoordinates[12][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + bcoordinates[15][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5] + bcoordinates[18][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 6] + bcoordinates[21][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 7] + bcoordinates[24][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 8] + bcoordinates[27][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 9];
        const s_t J01 = bcoordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS] + bcoordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + bcoordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + bcoordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + bcoordinates[12][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + bcoordinates[15][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5] + bcoordinates[18][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 6] + bcoordinates[21][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 7] + bcoordinates[24][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 8] + bcoordinates[27][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 9];
        const s_t J02 = bcoordinates[0][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS] + bcoordinates[3][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 1] + bcoordinates[6][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 2] + bcoordinates[9][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 3] + bcoordinates[12][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 4] + bcoordinates[15][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 5] + bcoordinates[18][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 6] + bcoordinates[21][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 7] + bcoordinates[24][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 8] + bcoordinates[27][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 9];
        const s_t J10 = bcoordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS] + bcoordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + bcoordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + bcoordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + bcoordinates[13][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + bcoordinates[16][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5] + bcoordinates[19][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 6] + bcoordinates[22][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 7] + bcoordinates[25][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 8] + bcoordinates[28][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 9];
        const s_t J11 = bcoordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS] + bcoordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + bcoordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + bcoordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + bcoordinates[13][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + bcoordinates[16][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5] + bcoordinates[19][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 6] + bcoordinates[22][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 7] + bcoordinates[25][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 8] + bcoordinates[28][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 9];
        const s_t J12 = bcoordinates[1][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS] + bcoordinates[4][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 1] + bcoordinates[7][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 2] + bcoordinates[10][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 3] + bcoordinates[13][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 4] + bcoordinates[16][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 5] + bcoordinates[19][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 6] + bcoordinates[22][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 7] + bcoordinates[25][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 8] + bcoordinates[28][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 9];
        const s_t J20 = bcoordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS] + bcoordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + bcoordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + bcoordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + bcoordinates[14][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + bcoordinates[17][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5] + bcoordinates[20][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 6] + bcoordinates[23][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 7] + bcoordinates[26][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 8] + bcoordinates[29][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 9];
        const s_t J21 = bcoordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS] + bcoordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + bcoordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + bcoordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + bcoordinates[14][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + bcoordinates[17][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5] + bcoordinates[20][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 6] + bcoordinates[23][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 7] + bcoordinates[26][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 8] + bcoordinates[29][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 9];
        const s_t J22 = bcoordinates[2][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS] + bcoordinates[5][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 1] + bcoordinates[8][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 2] + bcoordinates[11][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 3] + bcoordinates[14][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 4] + bcoordinates[17][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 5] + bcoordinates[20][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 6] + bcoordinates[23][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 7] + bcoordinates[26][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 8] + bcoordinates[29][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 9];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS + lane);
      }
    }

    const s_t *const field_shape[NC] = {sfem::codegen::ref_tet10_q11<s_t>::shape(), sfem::codegen::ref_tet4_q11<s_t>::shape()};
    const s_t *const fgref[NC * ND] = {sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::ref_tet4_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q11<s_t>::grad_ref_z()};
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    navier_stokes_form_1_u_d3_simplex_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, VS, bdeterminant, badjugate, field_shape, fgref, sfem::codegen::quad_tet_q11<s_t>::q_weight(), bcurrent, bprevious, convection_scale, dt, f0, f1, f2, nu, rho, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 10; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 10; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 10 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[2];
      for (int local_shape = 0; local_shape < 10; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 20 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 4; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 30 + local_shape;
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

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double convection_scale,
    const double dt,
    const double f0,
    const double f1,
    const double f2,
    const double nu,
    const double rho,
    const ptrdiff_t current_stride,
    const double *const RSTR u_data[3],
    const double *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const double *const RSTR u_old_data[3],
    const double *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    double *const RSTR u_out[3],
    double *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float convection_scale,
    const float dt,
    const float f0,
    const float f1,
    const float f2,
    const float nu,
    const float rho,
    const ptrdiff_t current_stride,
    const float *const RSTR u_data[3],
    const float *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const float *const RSTR u_old_data[3],
    const float *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    float *const RSTR u_out[3],
    float *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_1_u_tet10_tet4_jacobian_action_affine_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int CELL_NS = 10;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  static constexpr int VS = 16;
  (void)nnodes;
  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_1_u_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int CELL_NS = 10;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  static constexpr int VS = 16;
  (void)nnodes;
  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
