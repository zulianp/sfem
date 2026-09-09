#include <type_traits>
#include "../navier_stokes_form_2_p_u_d3_tensor_product_mixed_local.hpp"
#include "../../../reference/line_p1_q4.hpp"
#include "../../../reference/line_p2_q4.hpp"
#include "../../../reference/quad_line_q4.hpp"
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


template <typename s_t>
struct navier_stokes_form_2_p_u_affine_reference_data {
  static const s_t *q_weight_1d() { return quad_line_q4<s_t>::q_weight_1d(); }
  static const s_t *hex27_shape_1d() { return ref_line_p2_q4<s_t>::shape_1d(); }
  static const s_t *hex27_grad_1d() { return ref_line_p2_q4<s_t>::grad_1d(); }
  static const s_t *hex8_shape_1d() { return ref_line_p1_q4<s_t>::shape_1d(); }
  static const s_t *hex8_grad_1d() { return ref_line_p1_q4<s_t>::grad_1d(); }
};

template <typename s_t>
struct navier_stokes_form_2_p_u_isoparametric_reference_data {
  static const s_t *q_weight_1d() { return quad_line_q4<s_t>::q_weight_1d(); }
  static const s_t *hex27_shape_1d() { return ref_line_p2_q4<s_t>::shape_1d(); }
  static const s_t *hex27_grad_1d() { return ref_line_p2_q4<s_t>::grad_1d(); }
  static const s_t *hex8_shape_1d() { return ref_line_p1_q4<s_t>::shape_1d(); }
  static const s_t *hex8_grad_1d() { return ref_line_p1_q4<s_t>::grad_1d(); }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data = {
  "navier_stokes_form_2_p_u_hex27_hex8_residual_esoa",
  "HEX27",
  3,
  64,
  27,
  16,
  4,
  32,
  52,
  1,
  0,
  0,
  0,
  0,
  0,
  36,
  11,
  92,
  0,
  0,
  7,
  21,
  10,
  40,
  4,
  0,
  0,
  0,
  89,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_2_p_u_hex27_hex8_residual_esoa",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_float",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_residual_a_msoa",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_residual_a_msoa_float",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_residual_i_msoa",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_residual_i_msoa_float",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data = {
  "navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa",
  "HEX27",
  3,
  64,
  27,
  16,
  4,
  29,
  60,
  1,
  0,
  0,
  0,
  0,
  0,
  33,
  21,
  97,
  0,
  0,
  17,
  26,
  10,
  40,
  4,
  0,
  0,
  89,
  89,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data,
      nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_float",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa_float",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa_float",
      &sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_2_p_u_hex27_hex8_residual_affine_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 64;
  static constexpr int CELL_NS = 27;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 89;
  static constexpr int VS = 16;
  (void)nnodes;
  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_2_p_u_hex27_hex8_residual_isoparametric_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 64;
  static constexpr int CELL_NS = 27;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 89;
  static constexpr int VS = 16;
  (void)nnodes;
  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_affine_mesh_mixed_impl(
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
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u_direction_data[3],
    const s_t *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 64;
  static constexpr int CELL_NS = 27;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 89;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const field_shape_1d[NC] = {sfem::codegen::navier_stokes_form_2_p_u_affine_reference_data<s_t>::hex27_shape_1d(), sfem::codegen::navier_stokes_form_2_p_u_affine_reference_data<s_t>::hex8_shape_1d()};
  const s_t *const field_grad_1d[NC] = {sfem::codegen::navier_stokes_form_2_p_u_affine_reference_data<s_t>::hex27_grad_1d(), sfem::codegen::navier_stokes_form_2_p_u_affine_reference_data<s_t>::hex8_grad_1d()};
  const idx_t *const RSTR field_0_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const idx_t *const RSTR field_1_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const idx_t *const RSTR field_2_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const idx_t *const RSTR field_3_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bdirection[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = field_0_elements[local_shape];
      const int stream = 0 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = u_direction_data[0][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = field_1_elements[local_shape];
      const int stream = 27 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = u_direction_data[1][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = field_2_elements[local_shape];
      const int stream = 54 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = u_direction_data[2][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 8; ++local_shape) {
      const idx_t *const RSTR element_shape = field_3_elements[local_shape];
      const int stream = 81 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = p_direction_data[node * direction_stride];
      }
    }

    for (int stream = 0; stream < 89; ++stream) {
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

    navier_stokes_form_2_p_u_d3_tensor_product_mixed_jacobian_action_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, 0, bageom_streams[9], badjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_p_u_affine_reference_data<s_t>::q_weight_1d(), bdirection, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = field_0_elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = field_1_elements[local_shape];
        const int stream = 27 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[2];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = field_2_elements[local_shape];
        const int stream = 54 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 8; ++local_shape) {
        const idx_t *const RSTR element_shape = field_3_elements[local_shape];
        const int stream = 81 + local_shape;
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

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa(
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
    const ptrdiff_t direction_stride,
    const double *const RSTR u_direction_data[3],
    const double *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    double *const RSTR u_out[3],
    double *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa_float(
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
    const ptrdiff_t direction_stride,
    const float *const RSTR u_direction_data[3],
    const float *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    float *const RSTR u_out[3],
    float *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u_direction_data[3],
    const s_t *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 64;
  static constexpr int CELL_NS = 27;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 89;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_shape_1d = sfem::codegen::navier_stokes_form_2_p_u_isoparametric_reference_data<s_t>::hex27_shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::navier_stokes_form_2_p_u_isoparametric_reference_data<s_t>::hex27_grad_1d();
  const idx_t *const RSTR field_0_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const idx_t *const RSTR field_1_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const idx_t *const RSTR field_2_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const idx_t *const RSTR field_3_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};
  const idx_t *const RSTR coordinate_elements[27] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[ND * CELL_NS][VS];
    s_t badjugate_data[ND * ND][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bdirection[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = coordinate_elements[shape];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
        }
      }
    }

    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = field_0_elements[local_shape];
      const int stream = 0 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = u_direction_data[0][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = field_1_elements[local_shape];
      const int stream = 27 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = u_direction_data[1][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = field_2_elements[local_shape];
      const int stream = 54 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = u_direction_data[2][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 8; ++local_shape) {
      const idx_t *const RSTR element_shape = field_3_elements[local_shape];
      const int stream = 81 + local_shape;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const idx_t node = element_shape[evb + lane];
        bdirection[stream][lane] = p_direction_data[node * direction_stride];
      }
    }

    for (int stream = 0; stream < 89; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

    const s_t *const field_shape_1d[NC] = {sfem::codegen::navier_stokes_form_2_p_u_isoparametric_reference_data<s_t>::hex27_shape_1d(), sfem::codegen::navier_stokes_form_2_p_u_isoparametric_reference_data<s_t>::hex8_shape_1d()};
    const s_t *const field_grad_1d[NC] = {sfem::codegen::navier_stokes_form_2_p_u_isoparametric_reference_data<s_t>::hex27_grad_1d(), sfem::codegen::navier_stokes_form_2_p_u_isoparametric_reference_data<s_t>::hex8_grad_1d()};
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    navier_stokes_form_2_p_u_d3_tensor_product_mixed_jacobian_action_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, VS, bdeterminant, badjugate, field_shape_1d, field_grad_1d, sfem::codegen::navier_stokes_form_2_p_u_isoparametric_reference_data<s_t>::q_weight_1d(), bdirection, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = field_0_elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = field_1_elements[local_shape];
        const int stream = 27 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = u_out[2];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = field_2_elements[local_shape];
        const int stream = 54 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 8; ++local_shape) {
        const idx_t *const RSTR element_shape = field_3_elements[local_shape];
        const int stream = 81 + local_shape;
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

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const double *const RSTR u_direction_data[3],
    const double *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    double *const RSTR u_out[3],
    double *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const float *const RSTR u_direction_data[3],
    const float *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    float *const RSTR u_out[3],
    float *const RSTR p_out
) {
  return sfem::codegen::navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}
