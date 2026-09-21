#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../body_force_d2_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/quad_tri_q1.hpp"
#include "../../../reference/tri3_q1.hpp"
#if defined(__has_include)
#if __has_include("smesh_types.hpp")
#include "smesh_types.hpp"
#endif
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

static const KernelDiagnostics body_force_tri3_residual_esoa_diagnostics_data = {
  "body_force_tri3_residual_esoa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  5,
  2,
  4,
  81,
  108,
  0,
  3,
  5,
  9,
  1,
  3,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *body_force_tri3_residual_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tri3_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics body_force_tri3_jacobian_action_esoa_diagnostics_data = {
  "body_force_tri3_jacobian_action_esoa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  2,
  0,
  81,
  108,
  0,
  0,
  5,
  9,
  1,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *body_force_tri3_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tri3_jacobian_action_esoa_diagnostics_data;
}

extern "C" int body_force_tri3_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const real_t density,
    const real_t g0,
    const real_t g1,
    void *const RSTR output[6]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::body_force_d2_simplex_residual_block<double, 1, 3, 16>(ne, geometry_stride, (const double *)determinant, sfem::codegen::ref_tri3_q1<double>::shape(), sfem::codegen::quad_tri_q1<double>::q_weight(), density, g0, g1, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::body_force_d2_simplex_residual_block<float, 1, 3, 16>(ne, geometry_stride, (const float *)determinant, sfem::codegen::ref_tri3_q1<float>::shape(), sfem::codegen::quad_tri_q1<float>::q_weight(), density, g0, g1, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tri3_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int body_force_tri3_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const affine_shape = sfem::codegen::ref_tri3_q1<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
  const s_t *const affine_q_weight = sfem::codegen::quad_tri_q1<s_t>::q_weight();

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
    bageom_streams[0] = ageom_stream<s_t, g_t, VS>(
        ne, affine_geometry_sources[0], baffine_geometry_data[0], std::is_same<g_t, s_t>());

    body_force_d2_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape, affine_q_weight, density, g0, g1, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out};
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

extern "C" int body_force_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::body_force_tri3_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, density, g0, g1, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::body_force_tri3_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, density, g0, g1, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tri3_residual_a_msoa", -1, (int)scalar_bytes);
}
