#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_total_d2_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
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
    const int ne,
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

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  57,
  66,
  1,
  0,
  4,
  0,
  0,
  0,
  17,
  24,
  135,
  81,
  108,
  22,
  18,
  5,
  9,
  1,
  5,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u0",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  50,
  65,
  1,
  0,
  3,
  0,
  0,
  0,
  17,
  30,
  126,
  81,
  108,
  29,
  26,
  5,
  9,
  1,
  5,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u1",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  56,
  69,
  1,
  0,
  1,
  0,
  0,
  0,
  17,
  33,
  134,
  81,
  108,
  32,
  26,
  5,
  9,
  1,
  5,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u0",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  55,
  70,
  1,
  0,
  1,
  0,
  0,
  0,
  17,
  32,
  134,
  81,
  108,
  31,
  27,
  5,
  9,
  1,
  5,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u1",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  48,
  62,
  1,
  0,
  3,
  0,
  0,
  0,
  17,
  30,
  121,
  81,
  108,
  29,
  23,
  5,
  9,
  1,
  5,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  131,
  165,
  1,
  0,
  5,
  0,
  0,
  0,
  21,
  70,
  309,
  81,
  108,
  68,
  51,
  5,
  9,
  1,
  5,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa_diagnostics_data;
}

extern "C" int mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[4],
    const void *const RSTR current[6],
    const void *const RSTR previous[6],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[6]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_residual_block<double, 1, 3, 16>(ne, (const double *)determinant, (const double *const *)adjugate, (const double *const *)current, (const double *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_residual_block<float, 1, 3, 16>(ne, (const float *)determinant, (const float *const *)adjugate, (const float *const *)current, (const float *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tri3_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 16;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1};
    const s_t *const previous_components[NC] = {u0_old, u1_old};

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

    mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_residual_block_contiguous<s_t, NQ, NS, VS>(ne, bageom_streams[4], badjugate, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_total_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tri3_residual_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[4],
    const void *const RSTR current[6],
    const void *const RSTR previous[6],
    const void *const RSTR direction[6],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[6]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_jacobian_action_block<double, 1, 3, 16>(ne, (const double *)determinant, (const double *const *)adjugate, (const double *const *)current, (const double *const *)previous, (const double *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_jacobian_action_block<float, 1, 3, 16>(ne, (const float *)determinant, (const float *const *)adjugate, (const float *const *)current, (const float *const *)previous, (const float *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 16;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1};
    const s_t *const previous_components[NC] = {u0_old, u1_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bprevious[stream][lane] = previous_components[field][node * previous_stride];
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

    mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, bageom_streams[4], badjugate, bcurrent, bprevious, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_tri3_hessian_crs_a_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(3)
  for (int d = 0; d < 3; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(3)
    for (int d = 0; d < 3; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_tri3_hessian_crs_a_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int N_COL_STREAMS = 6;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    mooney_rivlin_kelvin_voigt_total_tri3_hessian_crs_a_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int bi = 0; bi < NC; ++bi) {
    for (int row_shape = 0; row_shape < NS; ++row_shape) {
      const s_t *const RSTR row = &element_matrix[(bi * NS + row_shape) * N_COL_STREAMS];
      for (int bj = 0; bj < NC; ++bj) {
        for (int col_shape = 0; col_shape < NS; ++col_shape) {
          s_t *const block = &values[entries[row_shape * NS + col_shape] * NC * NC];
#pragma omp atomic update
          block[bi * NC + bj] += row[bj * NS + col_shape];
        }
      }
    }
  }
}

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tri3_hessian_crs_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int N_STREAMS = NC * NS;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    idx_t ev[NS];
    s_t element_matrix[36];
    s_t badjugate_data[ND * ND][NQ];
    s_t bdeterminant[NQ];
    s_t bcurrent[N_STREAMS];
    s_t bprevious[N_STREAMS];

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
      bcurrent[shape * NC + 0] = u0[node * current_stride];
      bcurrent[shape * NC + 1] = u1[node * current_stride];
      bprevious[shape * NC + 0] = u0_old[node * previous_stride];
      bprevious[shape * NC + 1] = u1_old[node * previous_stride];
    }


    badjugate_data[0][0] = s_t(g_adj0[element]);
    badjugate_data[1][0] = s_t(g_adj1[element]);
    badjugate_data[2][0] = s_t(g_adj2[element]);
    badjugate_data[3][0] = s_t(g_adj3[element]);
    bdeterminant[0] = s_t(g_det0[element]);
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};

    mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_hessian_block<s_t, NQ, NS>(1, bdeterminant, badjugate, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, element_matrix);

    mooney_rivlin_kelvin_voigt_total_tri3_hessian_crs_a_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_total_tri3_hessian_bsr_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_hessian_crs_a_msoa_impl<double>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, rowptr, colidx, (double *)values);
    }
    case (int)sizeof(float): {
      return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_hessian_crs_a_msoa_impl<float>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, rowptr, colidx, (float *)values);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tri3_hessian_bsr_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static constexpr int pm_orientation[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};

template <typename s_t, typename g_t, int NQ, int NS, int VS>
static int mooney_rivlin_kelvin_voigt_total_tri3_merit_patch(
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const g_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const int nsteps,
    const s_t *const RSTR steps,
    const s_t *const RSTR x,
    const s_t *const RSTR h,
    const s_t *const RSTR p,
    const s_t *const RSTR accumulator,
    s_t *const RSTR merit
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;

#pragma omp parallel
  {
    // Per thread, and never larger than the vector width: the
    // caller rounds the step count to it, which is the whole
    // reason the steps carry the lanes.
    s_t merit_local[VS];
    for (int lane = 0; lane < VS; ++lane) merit_local[lane] = s_t(0);
    s_t rho[NC * VS];
    s_t pm_test_grad[2 * VS];
    s_t pm_weight[1 * VS];
    s_t pm_state_grad[4 * VS];
    s_t pm_direction_grad[4 * VS];
    s_t pm_previous_grad[4 * VS];
    element_idx_t pm_incident[VS];
    uint8_t pm_local_node[VS];

#pragma omp for schedule(static)
    for (ptrdiff_t node = 0; node < n_owned_nodes; ++node) {
      // Seed with everything that does not move with the state, so
      // the square below is over the whole residual.
      for (int c = 0; c < NC; ++c) {
        for (int lane = 0; lane < VS; ++lane) {
          rho[c * VS + lane] = accumulator[node * NC + c];
        }
      }
      const count_t begin = n2e_ptr[node];
      const count_t end = n2e_ptr[node + 1];
      for (count_t block = begin; block < end; block += VS) {
        const int ne = (int)MIN((count_t)VS, end - block);
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          pm_incident[lane] = n2e_idx[block + lane];
          pm_local_node[lane] = n2e_local[block + lane];
        }
        // loop 1 -- lanes are the elements incident on this node.
        {  // TRI3 evaluates in closed form
          #pragma omp simd
          for (int lane = 0; lane < ne; ++lane) {
            const idx_t element = pm_incident[lane];
            const int *const RSTR perm = pm_orientation[pm_local_node[lane]];
            s_t state[6];
            s_t direction[6];
            s_t previous[6];
            for (int j = 0; j < NS; ++j) {
              const idx_t node = elements[perm[j]][element];
              for (int c = 0; c < NC; ++c) {
                state[j * NC + c] = x[node * NC + c];
                direction[j * NC + c] = h[node * NC + c];
                previous[j * NC + c] = p[node * NC + c];
              }
            }
            // The Jacobian of the *permuted* element: its columns are the
            // edges from the visited node, which the permutation put at
            // slot 0.  Constant over the cell, because the orientation
            // gate admits only affine simplices.
            s_t jac[ND * ND];
            for (int d = 0; d < ND; ++d) {
              const s_t origin = (s_t)points[d][elements[perm[0]][element]];
              for (int k = 0; k < ND; ++k) {
                jac[d * ND + k] =
                    (s_t)points[d][elements[perm[k + 1]][element]] - origin;
              }
            }
            const s_t det = jac[0] * jac[3] - jac[1] * jac[2];
            s_t adj[4];
            adj[0] =  jac[3];
            adj[1] = -jac[1];
            adj[2] = -jac[2];
            adj[3] =  jac[0];
            // physical gradient of the state: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                const s_t mapped = (-state[0 * NC + c] + state[1 * NC + c]) * adj[0 * ND + d] + (-state[0 * NC + c] + state[2 * NC + c]) * adj[1 * ND + d];
                pm_state_grad[(c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // physical gradient of the direction: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                const s_t mapped = (-direction[0 * NC + c] + direction[1 * NC + c]) * adj[0 * ND + d] + (-direction[0 * NC + c] + direction[2 * NC + c]) * adj[1 * ND + d];
                pm_direction_grad[(c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // physical gradient of the previous: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                const s_t mapped = (-previous[0 * NC + c] + previous[1 * NC + c]) * adj[0 * ND + d] + (-previous[0 * NC + c] + previous[2 * NC + c]) * adj[1 * ND + d];
                pm_previous_grad[(c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // the fixed basis function's quantities, and the
            // integration weight.  Both are what the orientation buys:
            // `phi_0` is the same function in every element and at
            // every step, so this leaves the step loop entirely.
            for (int d = 0; d < ND; ++d) {
              const s_t mapped = s_t(-1) * adj[0 * ND + d] + s_t(-1) * adj[1 * ND + d];
              pm_test_grad[(d) * VS + lane] = mapped / det;
            }
            pm_weight[lane] = (s_t(1) / s_t(2)) * det;
          }
        }
        // loop 2 -- lanes are the sampled step lengths.
        for (int lane_e = 0; lane_e < ne; ++lane_e) {
          {  // TRI3 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nsteps; ++lane) {
              const s_t alpha = steps[lane];
              const s_t u0_grad_0 = pm_state_grad[(0) * VS + lane_e] + alpha * pm_direction_grad[(0) * VS + lane_e];
              const s_t u0_grad_1 = pm_state_grad[(1) * VS + lane_e] + alpha * pm_direction_grad[(1) * VS + lane_e];
              const s_t u1_grad_0 = pm_state_grad[(2) * VS + lane_e] + alpha * pm_direction_grad[(2) * VS + lane_e];
              const s_t u1_grad_1 = pm_state_grad[(3) * VS + lane_e] + alpha * pm_direction_grad[(3) * VS + lane_e];
              const s_t u0_old_grad_0 = pm_previous_grad[(0) * VS + lane_e];
              const s_t u0_old_grad_1 = pm_previous_grad[(1) * VS + lane_e];
              const s_t u1_old_grad_0 = pm_previous_grad[(2) * VS + lane_e];
              const s_t u1_old_grad_1 = pm_previous_grad[(3) * VS + lane_e];
              const s_t residual_tmp0 = u1_grad_1 + s_t(1);
              const s_t residual_tmp1 = u0_grad_1*u1_grad_0;
              const s_t residual_tmp2 = u0_grad_0 + s_t(1);
              const s_t residual_tmp3 = lmbda*(residual_tmp0*residual_tmp2 - residual_tmp1 + s_t(-1));
              const s_t residual_tmp4 = residual_tmp0*u1_grad_0 + residual_tmp2*u0_grad_1;
              const s_t residual_tmp5 = s_t(2)*u0_grad_1;
              const s_t residual_tmp6 = pow_2(residual_tmp2) + pow_2(u1_grad_0);
              const s_t residual_tmp7 = s_t(2)*residual_tmp2;
              const s_t residual_tmp8 = pow_2(residual_tmp0) + pow_2(u0_grad_1);
              const s_t residual_tmp9 = residual_tmp6 + residual_tmp8;
              const s_t residual_tmp10 = pow_m1(-residual_tmp1 + residual_tmp2 + u0_grad_0*u1_grad_1 + u1_grad_1);
              const s_t residual_tmp11 = u0_grad_0*u_dt_shift + u0_old_grad_0;
              const s_t residual_tmp12 = u1_grad_1*u_dt_shift + u1_old_grad_1;
              const s_t residual_tmp13 = u0_grad_1*u_dt_shift + u0_old_grad_1;
              const s_t residual_tmp14 = u1_grad_0*u_dt_shift + u1_old_grad_0;
              const s_t residual_tmp15 = eta_s*(-residual_tmp0*residual_tmp14 + residual_tmp11*u0_grad_1 + residual_tmp12*u1_grad_0 - residual_tmp13*residual_tmp2);
              const s_t residual_tmp16 = residual_tmp13*u1_grad_0;
              const s_t residual_tmp17 = residual_tmp0*residual_tmp11;
              const s_t residual_tmp18 = -residual_tmp12*residual_tmp2 + residual_tmp14*u0_grad_1;
              const s_t residual_tmp19 = eta_b*(residual_tmp16 - residual_tmp17 + residual_tmp18);
              const s_t residual_tmp20 = -residual_tmp16 + residual_tmp17 + residual_tmp18;
              const s_t residual_tmp21 = -eta_s*residual_tmp20 + residual_tmp19;
              const s_t residual_tmp22 = s_t(2)*u1_grad_0;
              const s_t residual_tmp23 = s_t(2)*residual_tmp0;
              const s_t residual_tmp24 = eta_s*residual_tmp20 + residual_tmp19;
              const s_t grad_coeff0_0 = mu*(s_t(2)*residual_tmp2*residual_tmp9 - residual_tmp4*residual_tmp5 - residual_tmp6*residual_tmp7 + s_t(4)*u0_grad_0 - s_t(6)*u1_grad_1 + s_t(-2)) + residual_tmp0*residual_tmp3 + residual_tmp10*(-residual_tmp0*residual_tmp21 + residual_tmp15*u0_grad_1);
              const s_t grad_coeff0_1 = mu*(-residual_tmp4*residual_tmp7 - residual_tmp5*residual_tmp8 + residual_tmp5*residual_tmp9 + s_t(4)*u0_grad_1 + s_t(6)*u1_grad_0) + residual_tmp10*(-residual_tmp15*residual_tmp2 + residual_tmp21*u1_grad_0) - residual_tmp3*u1_grad_0;
              const s_t grad_coeff1_0 = mu*(-residual_tmp22*residual_tmp6 + residual_tmp22*residual_tmp9 - residual_tmp23*residual_tmp4 + s_t(6)*u0_grad_1 + s_t(4)*u1_grad_0) + residual_tmp10*(-residual_tmp0*residual_tmp15 + residual_tmp24*u0_grad_1) - residual_tmp3*u0_grad_1;
              const s_t grad_coeff1_1 = mu*(s_t(2)*residual_tmp0*residual_tmp9 - residual_tmp22*residual_tmp4 - residual_tmp23*residual_tmp8 - s_t(6)*u0_grad_0 + s_t(4)*u1_grad_1 + s_t(-2)) + residual_tmp10*(residual_tmp15*u1_grad_0 - residual_tmp2*residual_tmp24) + residual_tmp2*residual_tmp3;
              const s_t weight = pm_weight[lane_e];
              rho[0 * VS + lane] += weight * (grad_coeff0_0 * pm_test_grad[(0) * VS + lane_e] + grad_coeff0_1 * pm_test_grad[(1) * VS + lane_e]);
              rho[1 * VS + lane] += weight * (grad_coeff1_0 * pm_test_grad[(0) * VS + lane_e] + grad_coeff1_1 * pm_test_grad[(1) * VS + lane_e]);
            }
          }
        }
      }

      // The node is finished, so it may be squared.
      #pragma omp simd
      for (int lane = 0; lane < nsteps; ++lane) {
        s_t squared = s_t(0);
        squared += rho[0 * VS + lane] * rho[0 * VS + lane];
        squared += rho[1 * VS + lane] * rho[1 * VS + lane];
        merit_local[lane] += s_t(0.5) * squared;
      }
    }

    // One reduction per thread, not one per node.
    for (int lane = 0; lane < nsteps; ++lane) {
#pragma omp atomic update
      merit[lane] += merit_local[lane];
    }
  }
  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


extern "C" int mooney_rivlin_kelvin_voigt_total_tri3_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const int nsteps,
    const void *const RSTR steps,
    const void *const RSTR x,
    const void *const RSTR h,
    const void *const RSTR p,
    const void *const RSTR accumulator,
    void *const RSTR merit
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_merit_patch<double, geom_t, 1, 3, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, nsteps, (const double *)steps, (const double *)x, (const double *)h, (const double *)p, (const double *)accumulator, (double *)merit);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_merit_patch<float, geom_t, 1, 3, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, nsteps, (const float *)steps, (const float *)x, (const float *)h, (const float *)p, (const float *)accumulator, (float *)merit);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tri3_merit_patch_a_msoa", -1, (int)scalar_bytes);
}
