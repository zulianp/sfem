#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_total_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/quad_tet_q4.hpp"
#include "../../../reference/tet10_q4.hpp"
#include "../../../reference/quad_tet_q11.hpp"
#include "../../../reference/tet10_q11.hpp"
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

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  206,
  239,
  1,
  0,
  9,
  0,
  0,
  0,
  32,
  84,
  462,
  5159,
  7590,
  81,
  62,
  10,
  440,
  11,
  5,
  60,
  0,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  186,
  230,
  1,
  0,
  10,
  0,
  0,
  0,
  29,
  99,
  434,
  5159,
  7590,
  98,
  59,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  211,
  268,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  115,
  488,
  5159,
  7590,
  114,
  77,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u2_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  113,
  486,
  5159,
  7590,
  112,
  77,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u2_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u0_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  211,
  268,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  115,
  488,
  5159,
  7590,
  114,
  77,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  186,
  230,
  1,
  0,
  10,
  0,
  0,
  0,
  29,
  98,
  434,
  5159,
  7590,
  97,
  58,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u2_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  114,
  486,
  5159,
  7590,
  113,
  77,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u2_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u1_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  114,
  486,
  5159,
  7590,
  113,
  76,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  112,
  486,
  5159,
  7590,
  111,
  76,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u2_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  185,
  227,
  1,
  0,
  10,
  0,
  0,
  0,
  29,
  97,
  430,
  5159,
  7590,
  96,
  57,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u2_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_u2_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  973,
  1170,
  1,
  0,
  19,
  0,
  0,
  0,
  41,
  464,
  2170,
  5159,
  7590,
  461,
  267,
  10,
  440,
  11,
  5,
  60,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa_diagnostics_data;
}

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[30],
    const void *const RSTR previous[30],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[30]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_residual_block<double, 11, 10, 16>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tet10_q11<double>::grad_ref_x(), sfem::codegen::ref_tet10_q11<double>::grad_ref_y(), sfem::codegen::ref_tet10_q11<double>::grad_ref_z(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)current, (const double *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_residual_block<float, 11, 10, 16>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tet10_q11<float>::grad_ref_x(), sfem::codegen::ref_tet10_q11<float>::grad_ref_y(), sfem::codegen::ref_tet10_q11<float>::grad_ref_z(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)current, (const float *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tet10_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
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
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q4<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

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

    for (int stream = 0; stream < 30; ++stream) {
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

    mooney_rivlin_kelvin_voigt_total_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_residual_a_msoa(
    const int scalar_bytes,
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
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const void *const RSTR u2_old,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tet10_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        s_t *const RSTR bcoordinate_row = bcoordinates[shape * ND + d];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinate_row[lane] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

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

    for (int stream = 0; stream < 30; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const s_t *const RSTR coordinate_row0 = bcoordinates[0];
      const s_t *const RSTR coordinate_row1 = bcoordinates[1];
      const s_t *const RSTR coordinate_row2 = bcoordinates[2];
      const s_t *const RSTR coordinate_row3 = bcoordinates[3];
      const s_t *const RSTR coordinate_row4 = bcoordinates[4];
      const s_t *const RSTR coordinate_row5 = bcoordinates[5];
      const s_t *const RSTR coordinate_row6 = bcoordinates[6];
      const s_t *const RSTR coordinate_row7 = bcoordinates[7];
      const s_t *const RSTR coordinate_row8 = bcoordinates[8];
      const s_t *const RSTR coordinate_row9 = bcoordinates[9];
      const s_t *const RSTR coordinate_row10 = bcoordinates[10];
      const s_t *const RSTR coordinate_row11 = bcoordinates[11];
      const s_t *const RSTR coordinate_row12 = bcoordinates[12];
      const s_t *const RSTR coordinate_row13 = bcoordinates[13];
      const s_t *const RSTR coordinate_row14 = bcoordinates[14];
      const s_t *const RSTR coordinate_row15 = bcoordinates[15];
      const s_t *const RSTR coordinate_row16 = bcoordinates[16];
      const s_t *const RSTR coordinate_row17 = bcoordinates[17];
      const s_t *const RSTR coordinate_row18 = bcoordinates[18];
      const s_t *const RSTR coordinate_row19 = bcoordinates[19];
      const s_t *const RSTR coordinate_row20 = bcoordinates[20];
      const s_t *const RSTR coordinate_row21 = bcoordinates[21];
      const s_t *const RSTR coordinate_row22 = bcoordinates[22];
      const s_t *const RSTR coordinate_row23 = bcoordinates[23];
      const s_t *const RSTR coordinate_row24 = bcoordinates[24];
      const s_t *const RSTR coordinate_row25 = bcoordinates[25];
      const s_t *const RSTR coordinate_row26 = bcoordinates[26];
      const s_t *const RSTR coordinate_row27 = bcoordinates[27];
      const s_t *const RSTR coordinate_row28 = bcoordinates[28];
      const s_t *const RSTR coordinate_row29 = bcoordinates[29];
      const s_t cell_grad_ref0_0 = isoparametric_grad_ref_x[q * NS + 0];
      const s_t cell_grad_ref0_1 = isoparametric_grad_ref_x[q * NS + 1];
      const s_t cell_grad_ref0_2 = isoparametric_grad_ref_x[q * NS + 2];
      const s_t cell_grad_ref0_3 = isoparametric_grad_ref_x[q * NS + 3];
      const s_t cell_grad_ref0_4 = isoparametric_grad_ref_x[q * NS + 4];
      const s_t cell_grad_ref0_5 = isoparametric_grad_ref_x[q * NS + 5];
      const s_t cell_grad_ref0_6 = isoparametric_grad_ref_x[q * NS + 6];
      const s_t cell_grad_ref0_7 = isoparametric_grad_ref_x[q * NS + 7];
      const s_t cell_grad_ref0_8 = isoparametric_grad_ref_x[q * NS + 8];
      const s_t cell_grad_ref0_9 = isoparametric_grad_ref_x[q * NS + 9];
      const s_t cell_grad_ref1_0 = isoparametric_grad_ref_y[q * NS + 0];
      const s_t cell_grad_ref1_1 = isoparametric_grad_ref_y[q * NS + 1];
      const s_t cell_grad_ref1_2 = isoparametric_grad_ref_y[q * NS + 2];
      const s_t cell_grad_ref1_3 = isoparametric_grad_ref_y[q * NS + 3];
      const s_t cell_grad_ref1_4 = isoparametric_grad_ref_y[q * NS + 4];
      const s_t cell_grad_ref1_5 = isoparametric_grad_ref_y[q * NS + 5];
      const s_t cell_grad_ref1_6 = isoparametric_grad_ref_y[q * NS + 6];
      const s_t cell_grad_ref1_7 = isoparametric_grad_ref_y[q * NS + 7];
      const s_t cell_grad_ref1_8 = isoparametric_grad_ref_y[q * NS + 8];
      const s_t cell_grad_ref1_9 = isoparametric_grad_ref_y[q * NS + 9];
      const s_t cell_grad_ref2_0 = isoparametric_grad_ref_z[q * NS + 0];
      const s_t cell_grad_ref2_1 = isoparametric_grad_ref_z[q * NS + 1];
      const s_t cell_grad_ref2_2 = isoparametric_grad_ref_z[q * NS + 2];
      const s_t cell_grad_ref2_3 = isoparametric_grad_ref_z[q * NS + 3];
      const s_t cell_grad_ref2_4 = isoparametric_grad_ref_z[q * NS + 4];
      const s_t cell_grad_ref2_5 = isoparametric_grad_ref_z[q * NS + 5];
      const s_t cell_grad_ref2_6 = isoparametric_grad_ref_z[q * NS + 6];
      const s_t cell_grad_ref2_7 = isoparametric_grad_ref_z[q * NS + 7];
      const s_t cell_grad_ref2_8 = isoparametric_grad_ref_z[q * NS + 8];
      const s_t cell_grad_ref2_9 = isoparametric_grad_ref_z[q * NS + 9];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = coordinate_row0[lane] * cell_grad_ref0_0 + coordinate_row3[lane] * cell_grad_ref0_1 + coordinate_row6[lane] * cell_grad_ref0_2 + coordinate_row9[lane] * cell_grad_ref0_3 + coordinate_row12[lane] * cell_grad_ref0_4 + coordinate_row15[lane] * cell_grad_ref0_5 + coordinate_row18[lane] * cell_grad_ref0_6 + coordinate_row21[lane] * cell_grad_ref0_7 + coordinate_row24[lane] * cell_grad_ref0_8 + coordinate_row27[lane] * cell_grad_ref0_9;
        const s_t J01 = coordinate_row0[lane] * cell_grad_ref1_0 + coordinate_row3[lane] * cell_grad_ref1_1 + coordinate_row6[lane] * cell_grad_ref1_2 + coordinate_row9[lane] * cell_grad_ref1_3 + coordinate_row12[lane] * cell_grad_ref1_4 + coordinate_row15[lane] * cell_grad_ref1_5 + coordinate_row18[lane] * cell_grad_ref1_6 + coordinate_row21[lane] * cell_grad_ref1_7 + coordinate_row24[lane] * cell_grad_ref1_8 + coordinate_row27[lane] * cell_grad_ref1_9;
        const s_t J02 = coordinate_row0[lane] * cell_grad_ref2_0 + coordinate_row3[lane] * cell_grad_ref2_1 + coordinate_row6[lane] * cell_grad_ref2_2 + coordinate_row9[lane] * cell_grad_ref2_3 + coordinate_row12[lane] * cell_grad_ref2_4 + coordinate_row15[lane] * cell_grad_ref2_5 + coordinate_row18[lane] * cell_grad_ref2_6 + coordinate_row21[lane] * cell_grad_ref2_7 + coordinate_row24[lane] * cell_grad_ref2_8 + coordinate_row27[lane] * cell_grad_ref2_9;
        const s_t J10 = coordinate_row1[lane] * cell_grad_ref0_0 + coordinate_row4[lane] * cell_grad_ref0_1 + coordinate_row7[lane] * cell_grad_ref0_2 + coordinate_row10[lane] * cell_grad_ref0_3 + coordinate_row13[lane] * cell_grad_ref0_4 + coordinate_row16[lane] * cell_grad_ref0_5 + coordinate_row19[lane] * cell_grad_ref0_6 + coordinate_row22[lane] * cell_grad_ref0_7 + coordinate_row25[lane] * cell_grad_ref0_8 + coordinate_row28[lane] * cell_grad_ref0_9;
        const s_t J11 = coordinate_row1[lane] * cell_grad_ref1_0 + coordinate_row4[lane] * cell_grad_ref1_1 + coordinate_row7[lane] * cell_grad_ref1_2 + coordinate_row10[lane] * cell_grad_ref1_3 + coordinate_row13[lane] * cell_grad_ref1_4 + coordinate_row16[lane] * cell_grad_ref1_5 + coordinate_row19[lane] * cell_grad_ref1_6 + coordinate_row22[lane] * cell_grad_ref1_7 + coordinate_row25[lane] * cell_grad_ref1_8 + coordinate_row28[lane] * cell_grad_ref1_9;
        const s_t J12 = coordinate_row1[lane] * cell_grad_ref2_0 + coordinate_row4[lane] * cell_grad_ref2_1 + coordinate_row7[lane] * cell_grad_ref2_2 + coordinate_row10[lane] * cell_grad_ref2_3 + coordinate_row13[lane] * cell_grad_ref2_4 + coordinate_row16[lane] * cell_grad_ref2_5 + coordinate_row19[lane] * cell_grad_ref2_6 + coordinate_row22[lane] * cell_grad_ref2_7 + coordinate_row25[lane] * cell_grad_ref2_8 + coordinate_row28[lane] * cell_grad_ref2_9;
        const s_t J20 = coordinate_row2[lane] * cell_grad_ref0_0 + coordinate_row5[lane] * cell_grad_ref0_1 + coordinate_row8[lane] * cell_grad_ref0_2 + coordinate_row11[lane] * cell_grad_ref0_3 + coordinate_row14[lane] * cell_grad_ref0_4 + coordinate_row17[lane] * cell_grad_ref0_5 + coordinate_row20[lane] * cell_grad_ref0_6 + coordinate_row23[lane] * cell_grad_ref0_7 + coordinate_row26[lane] * cell_grad_ref0_8 + coordinate_row29[lane] * cell_grad_ref0_9;
        const s_t J21 = coordinate_row2[lane] * cell_grad_ref1_0 + coordinate_row5[lane] * cell_grad_ref1_1 + coordinate_row8[lane] * cell_grad_ref1_2 + coordinate_row11[lane] * cell_grad_ref1_3 + coordinate_row14[lane] * cell_grad_ref1_4 + coordinate_row17[lane] * cell_grad_ref1_5 + coordinate_row20[lane] * cell_grad_ref1_6 + coordinate_row23[lane] * cell_grad_ref1_7 + coordinate_row26[lane] * cell_grad_ref1_8 + coordinate_row29[lane] * cell_grad_ref1_9;
        const s_t J22 = coordinate_row2[lane] * cell_grad_ref2_0 + coordinate_row5[lane] * cell_grad_ref2_1 + coordinate_row8[lane] * cell_grad_ref2_2 + coordinate_row11[lane] * cell_grad_ref2_3 + coordinate_row14[lane] * cell_grad_ref2_4 + coordinate_row17[lane] * cell_grad_ref2_5 + coordinate_row20[lane] * cell_grad_ref2_6 + coordinate_row23[lane] * cell_grad_ref2_7 + coordinate_row26[lane] * cell_grad_ref2_8 + coordinate_row29[lane] * cell_grad_ref2_9;
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS + lane);
      }
    }

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_total_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const void *const RSTR u2_old,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return mooney_rivlin_kelvin_voigt_total_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], ((const double *)parameters)[3], ((const double *)parameters)[4], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (const double *)previous + 0, (const double *)previous + 1, (const double *)previous + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2);
    }
    case (int)sizeof(float): {
        return mooney_rivlin_kelvin_voigt_total_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], ((const float *)parameters)[3], ((const float *)parameters)[4], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (const float *)previous + 0, (const float *)previous + 1, (const float *)previous + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_residual_i_maos", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[30],
    const void *const RSTR previous[30],
    const void *const RSTR direction[30],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[30]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_jacobian_action_block<double, 11, 10, 16>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tet10_q11<double>::grad_ref_x(), sfem::codegen::ref_tet10_q11<double>::grad_ref_y(), sfem::codegen::ref_tet10_q11<double>::grad_ref_z(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)current, (const double *const *)previous, (const double *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_jacobian_action_block<float, 11, 10, 16>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tet10_q11<float>::grad_ref_x(), sfem::codegen::ref_tet10_q11<float>::grad_ref_y(), sfem::codegen::ref_tet10_q11<float>::grad_ref_z(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)current, (const float *const *)previous, (const float *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
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
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q4<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

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

    for (int stream = 0; stream < 30; ++stream) {
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

    mooney_rivlin_kelvin_voigt_total_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const void *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        s_t *const RSTR bcoordinate_row = bcoordinates[shape * ND + d];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinate_row[lane] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

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

    for (int stream = 0; stream < 30; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const s_t *const RSTR coordinate_row0 = bcoordinates[0];
      const s_t *const RSTR coordinate_row1 = bcoordinates[1];
      const s_t *const RSTR coordinate_row2 = bcoordinates[2];
      const s_t *const RSTR coordinate_row3 = bcoordinates[3];
      const s_t *const RSTR coordinate_row4 = bcoordinates[4];
      const s_t *const RSTR coordinate_row5 = bcoordinates[5];
      const s_t *const RSTR coordinate_row6 = bcoordinates[6];
      const s_t *const RSTR coordinate_row7 = bcoordinates[7];
      const s_t *const RSTR coordinate_row8 = bcoordinates[8];
      const s_t *const RSTR coordinate_row9 = bcoordinates[9];
      const s_t *const RSTR coordinate_row10 = bcoordinates[10];
      const s_t *const RSTR coordinate_row11 = bcoordinates[11];
      const s_t *const RSTR coordinate_row12 = bcoordinates[12];
      const s_t *const RSTR coordinate_row13 = bcoordinates[13];
      const s_t *const RSTR coordinate_row14 = bcoordinates[14];
      const s_t *const RSTR coordinate_row15 = bcoordinates[15];
      const s_t *const RSTR coordinate_row16 = bcoordinates[16];
      const s_t *const RSTR coordinate_row17 = bcoordinates[17];
      const s_t *const RSTR coordinate_row18 = bcoordinates[18];
      const s_t *const RSTR coordinate_row19 = bcoordinates[19];
      const s_t *const RSTR coordinate_row20 = bcoordinates[20];
      const s_t *const RSTR coordinate_row21 = bcoordinates[21];
      const s_t *const RSTR coordinate_row22 = bcoordinates[22];
      const s_t *const RSTR coordinate_row23 = bcoordinates[23];
      const s_t *const RSTR coordinate_row24 = bcoordinates[24];
      const s_t *const RSTR coordinate_row25 = bcoordinates[25];
      const s_t *const RSTR coordinate_row26 = bcoordinates[26];
      const s_t *const RSTR coordinate_row27 = bcoordinates[27];
      const s_t *const RSTR coordinate_row28 = bcoordinates[28];
      const s_t *const RSTR coordinate_row29 = bcoordinates[29];
      const s_t cell_grad_ref0_0 = isoparametric_grad_ref_x[q * NS + 0];
      const s_t cell_grad_ref0_1 = isoparametric_grad_ref_x[q * NS + 1];
      const s_t cell_grad_ref0_2 = isoparametric_grad_ref_x[q * NS + 2];
      const s_t cell_grad_ref0_3 = isoparametric_grad_ref_x[q * NS + 3];
      const s_t cell_grad_ref0_4 = isoparametric_grad_ref_x[q * NS + 4];
      const s_t cell_grad_ref0_5 = isoparametric_grad_ref_x[q * NS + 5];
      const s_t cell_grad_ref0_6 = isoparametric_grad_ref_x[q * NS + 6];
      const s_t cell_grad_ref0_7 = isoparametric_grad_ref_x[q * NS + 7];
      const s_t cell_grad_ref0_8 = isoparametric_grad_ref_x[q * NS + 8];
      const s_t cell_grad_ref0_9 = isoparametric_grad_ref_x[q * NS + 9];
      const s_t cell_grad_ref1_0 = isoparametric_grad_ref_y[q * NS + 0];
      const s_t cell_grad_ref1_1 = isoparametric_grad_ref_y[q * NS + 1];
      const s_t cell_grad_ref1_2 = isoparametric_grad_ref_y[q * NS + 2];
      const s_t cell_grad_ref1_3 = isoparametric_grad_ref_y[q * NS + 3];
      const s_t cell_grad_ref1_4 = isoparametric_grad_ref_y[q * NS + 4];
      const s_t cell_grad_ref1_5 = isoparametric_grad_ref_y[q * NS + 5];
      const s_t cell_grad_ref1_6 = isoparametric_grad_ref_y[q * NS + 6];
      const s_t cell_grad_ref1_7 = isoparametric_grad_ref_y[q * NS + 7];
      const s_t cell_grad_ref1_8 = isoparametric_grad_ref_y[q * NS + 8];
      const s_t cell_grad_ref1_9 = isoparametric_grad_ref_y[q * NS + 9];
      const s_t cell_grad_ref2_0 = isoparametric_grad_ref_z[q * NS + 0];
      const s_t cell_grad_ref2_1 = isoparametric_grad_ref_z[q * NS + 1];
      const s_t cell_grad_ref2_2 = isoparametric_grad_ref_z[q * NS + 2];
      const s_t cell_grad_ref2_3 = isoparametric_grad_ref_z[q * NS + 3];
      const s_t cell_grad_ref2_4 = isoparametric_grad_ref_z[q * NS + 4];
      const s_t cell_grad_ref2_5 = isoparametric_grad_ref_z[q * NS + 5];
      const s_t cell_grad_ref2_6 = isoparametric_grad_ref_z[q * NS + 6];
      const s_t cell_grad_ref2_7 = isoparametric_grad_ref_z[q * NS + 7];
      const s_t cell_grad_ref2_8 = isoparametric_grad_ref_z[q * NS + 8];
      const s_t cell_grad_ref2_9 = isoparametric_grad_ref_z[q * NS + 9];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = coordinate_row0[lane] * cell_grad_ref0_0 + coordinate_row3[lane] * cell_grad_ref0_1 + coordinate_row6[lane] * cell_grad_ref0_2 + coordinate_row9[lane] * cell_grad_ref0_3 + coordinate_row12[lane] * cell_grad_ref0_4 + coordinate_row15[lane] * cell_grad_ref0_5 + coordinate_row18[lane] * cell_grad_ref0_6 + coordinate_row21[lane] * cell_grad_ref0_7 + coordinate_row24[lane] * cell_grad_ref0_8 + coordinate_row27[lane] * cell_grad_ref0_9;
        const s_t J01 = coordinate_row0[lane] * cell_grad_ref1_0 + coordinate_row3[lane] * cell_grad_ref1_1 + coordinate_row6[lane] * cell_grad_ref1_2 + coordinate_row9[lane] * cell_grad_ref1_3 + coordinate_row12[lane] * cell_grad_ref1_4 + coordinate_row15[lane] * cell_grad_ref1_5 + coordinate_row18[lane] * cell_grad_ref1_6 + coordinate_row21[lane] * cell_grad_ref1_7 + coordinate_row24[lane] * cell_grad_ref1_8 + coordinate_row27[lane] * cell_grad_ref1_9;
        const s_t J02 = coordinate_row0[lane] * cell_grad_ref2_0 + coordinate_row3[lane] * cell_grad_ref2_1 + coordinate_row6[lane] * cell_grad_ref2_2 + coordinate_row9[lane] * cell_grad_ref2_3 + coordinate_row12[lane] * cell_grad_ref2_4 + coordinate_row15[lane] * cell_grad_ref2_5 + coordinate_row18[lane] * cell_grad_ref2_6 + coordinate_row21[lane] * cell_grad_ref2_7 + coordinate_row24[lane] * cell_grad_ref2_8 + coordinate_row27[lane] * cell_grad_ref2_9;
        const s_t J10 = coordinate_row1[lane] * cell_grad_ref0_0 + coordinate_row4[lane] * cell_grad_ref0_1 + coordinate_row7[lane] * cell_grad_ref0_2 + coordinate_row10[lane] * cell_grad_ref0_3 + coordinate_row13[lane] * cell_grad_ref0_4 + coordinate_row16[lane] * cell_grad_ref0_5 + coordinate_row19[lane] * cell_grad_ref0_6 + coordinate_row22[lane] * cell_grad_ref0_7 + coordinate_row25[lane] * cell_grad_ref0_8 + coordinate_row28[lane] * cell_grad_ref0_9;
        const s_t J11 = coordinate_row1[lane] * cell_grad_ref1_0 + coordinate_row4[lane] * cell_grad_ref1_1 + coordinate_row7[lane] * cell_grad_ref1_2 + coordinate_row10[lane] * cell_grad_ref1_3 + coordinate_row13[lane] * cell_grad_ref1_4 + coordinate_row16[lane] * cell_grad_ref1_5 + coordinate_row19[lane] * cell_grad_ref1_6 + coordinate_row22[lane] * cell_grad_ref1_7 + coordinate_row25[lane] * cell_grad_ref1_8 + coordinate_row28[lane] * cell_grad_ref1_9;
        const s_t J12 = coordinate_row1[lane] * cell_grad_ref2_0 + coordinate_row4[lane] * cell_grad_ref2_1 + coordinate_row7[lane] * cell_grad_ref2_2 + coordinate_row10[lane] * cell_grad_ref2_3 + coordinate_row13[lane] * cell_grad_ref2_4 + coordinate_row16[lane] * cell_grad_ref2_5 + coordinate_row19[lane] * cell_grad_ref2_6 + coordinate_row22[lane] * cell_grad_ref2_7 + coordinate_row25[lane] * cell_grad_ref2_8 + coordinate_row28[lane] * cell_grad_ref2_9;
        const s_t J20 = coordinate_row2[lane] * cell_grad_ref0_0 + coordinate_row5[lane] * cell_grad_ref0_1 + coordinate_row8[lane] * cell_grad_ref0_2 + coordinate_row11[lane] * cell_grad_ref0_3 + coordinate_row14[lane] * cell_grad_ref0_4 + coordinate_row17[lane] * cell_grad_ref0_5 + coordinate_row20[lane] * cell_grad_ref0_6 + coordinate_row23[lane] * cell_grad_ref0_7 + coordinate_row26[lane] * cell_grad_ref0_8 + coordinate_row29[lane] * cell_grad_ref0_9;
        const s_t J21 = coordinate_row2[lane] * cell_grad_ref1_0 + coordinate_row5[lane] * cell_grad_ref1_1 + coordinate_row8[lane] * cell_grad_ref1_2 + coordinate_row11[lane] * cell_grad_ref1_3 + coordinate_row14[lane] * cell_grad_ref1_4 + coordinate_row17[lane] * cell_grad_ref1_5 + coordinate_row20[lane] * cell_grad_ref1_6 + coordinate_row23[lane] * cell_grad_ref1_7 + coordinate_row26[lane] * cell_grad_ref1_8 + coordinate_row29[lane] * cell_grad_ref1_9;
        const s_t J22 = coordinate_row2[lane] * cell_grad_ref2_0 + coordinate_row5[lane] * cell_grad_ref2_1 + coordinate_row8[lane] * cell_grad_ref2_2 + coordinate_row11[lane] * cell_grad_ref2_3 + coordinate_row14[lane] * cell_grad_ref2_4 + coordinate_row17[lane] * cell_grad_ref2_5 + coordinate_row20[lane] * cell_grad_ref2_6 + coordinate_row23[lane] * cell_grad_ref2_7 + coordinate_row26[lane] * cell_grad_ref2_8 + coordinate_row29[lane] * cell_grad_ref2_9;
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS + lane);
      }
    }

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_total_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const void *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    const void *const RSTR direction,
    void *const RSTR output
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], ((const double *)parameters)[3], ((const double *)parameters)[4], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (const double *)previous + 0, (const double *)previous + 1, (const double *)previous + 2, 3, (const double *)direction + 0, (const double *)direction + 1, (const double *)direction + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2);
    }
    case (int)sizeof(float): {
        return mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], ((const float *)parameters)[3], ((const float *)parameters)[4], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (const float *)previous + 0, (const float *)previous + 1, (const float *)previous + 2, 3, (const float *)direction + 0, (const float *)direction + 1, (const float *)direction + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_i_maos", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_tet10_hessian_crs_i_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(10)
  for (int d = 0; d < 10; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(10)
    for (int d = 0; d < 10; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_tet10_hessian_crs_i_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int N_COL_STREAMS = 30;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    mooney_rivlin_kelvin_voigt_total_tet10_hessian_crs_i_msoa_find_cols(ev, cols, lenrow, ks);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_total_tet10_hessian_crs_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int N_STREAMS = NC * NS;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    idx_t ev[NS];
    s_t element_matrix[900];
    s_t bcoordinates[ND * NS];
    s_t badjugate_data[ND * ND][NQ];
    s_t bdeterminant[NQ];
    s_t bcurrent[N_STREAMS];
    s_t bprevious[N_STREAMS];
    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      const idx_t coordinate_node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinates[shape * ND + d] = s_t(coordinate_components[d][coordinate_node]);
      }
      bcurrent[shape * NC + 0] = u0[node * current_stride];
      bcurrent[shape * NC + 1] = u1[node * current_stride];
      bcurrent[shape * NC + 2] = u2[node * current_stride];
      bprevious[shape * NC + 0] = u0_old[node * previous_stride];
      bprevious[shape * NC + 1] = u1_old[node * previous_stride];
      bprevious[shape * NC + 2] = u2_old[node * previous_stride];
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const s_t cell_grad_ref0_0 = isoparametric_grad_ref_x[q * NS + 0];
      const s_t cell_grad_ref0_1 = isoparametric_grad_ref_x[q * NS + 1];
      const s_t cell_grad_ref0_2 = isoparametric_grad_ref_x[q * NS + 2];
      const s_t cell_grad_ref0_3 = isoparametric_grad_ref_x[q * NS + 3];
      const s_t cell_grad_ref0_4 = isoparametric_grad_ref_x[q * NS + 4];
      const s_t cell_grad_ref0_5 = isoparametric_grad_ref_x[q * NS + 5];
      const s_t cell_grad_ref0_6 = isoparametric_grad_ref_x[q * NS + 6];
      const s_t cell_grad_ref0_7 = isoparametric_grad_ref_x[q * NS + 7];
      const s_t cell_grad_ref0_8 = isoparametric_grad_ref_x[q * NS + 8];
      const s_t cell_grad_ref0_9 = isoparametric_grad_ref_x[q * NS + 9];
      const s_t cell_grad_ref1_0 = isoparametric_grad_ref_y[q * NS + 0];
      const s_t cell_grad_ref1_1 = isoparametric_grad_ref_y[q * NS + 1];
      const s_t cell_grad_ref1_2 = isoparametric_grad_ref_y[q * NS + 2];
      const s_t cell_grad_ref1_3 = isoparametric_grad_ref_y[q * NS + 3];
      const s_t cell_grad_ref1_4 = isoparametric_grad_ref_y[q * NS + 4];
      const s_t cell_grad_ref1_5 = isoparametric_grad_ref_y[q * NS + 5];
      const s_t cell_grad_ref1_6 = isoparametric_grad_ref_y[q * NS + 6];
      const s_t cell_grad_ref1_7 = isoparametric_grad_ref_y[q * NS + 7];
      const s_t cell_grad_ref1_8 = isoparametric_grad_ref_y[q * NS + 8];
      const s_t cell_grad_ref1_9 = isoparametric_grad_ref_y[q * NS + 9];
      const s_t cell_grad_ref2_0 = isoparametric_grad_ref_z[q * NS + 0];
      const s_t cell_grad_ref2_1 = isoparametric_grad_ref_z[q * NS + 1];
      const s_t cell_grad_ref2_2 = isoparametric_grad_ref_z[q * NS + 2];
      const s_t cell_grad_ref2_3 = isoparametric_grad_ref_z[q * NS + 3];
      const s_t cell_grad_ref2_4 = isoparametric_grad_ref_z[q * NS + 4];
      const s_t cell_grad_ref2_5 = isoparametric_grad_ref_z[q * NS + 5];
      const s_t cell_grad_ref2_6 = isoparametric_grad_ref_z[q * NS + 6];
      const s_t cell_grad_ref2_7 = isoparametric_grad_ref_z[q * NS + 7];
      const s_t cell_grad_ref2_8 = isoparametric_grad_ref_z[q * NS + 8];
      const s_t cell_grad_ref2_9 = isoparametric_grad_ref_z[q * NS + 9];
      const s_t J00 = bcoordinates[0] * cell_grad_ref0_0 + bcoordinates[3] * cell_grad_ref0_1 + bcoordinates[6] * cell_grad_ref0_2 + bcoordinates[9] * cell_grad_ref0_3 + bcoordinates[12] * cell_grad_ref0_4 + bcoordinates[15] * cell_grad_ref0_5 + bcoordinates[18] * cell_grad_ref0_6 + bcoordinates[21] * cell_grad_ref0_7 + bcoordinates[24] * cell_grad_ref0_8 + bcoordinates[27] * cell_grad_ref0_9;
      const s_t J01 = bcoordinates[0] * cell_grad_ref1_0 + bcoordinates[3] * cell_grad_ref1_1 + bcoordinates[6] * cell_grad_ref1_2 + bcoordinates[9] * cell_grad_ref1_3 + bcoordinates[12] * cell_grad_ref1_4 + bcoordinates[15] * cell_grad_ref1_5 + bcoordinates[18] * cell_grad_ref1_6 + bcoordinates[21] * cell_grad_ref1_7 + bcoordinates[24] * cell_grad_ref1_8 + bcoordinates[27] * cell_grad_ref1_9;
      const s_t J02 = bcoordinates[0] * cell_grad_ref2_0 + bcoordinates[3] * cell_grad_ref2_1 + bcoordinates[6] * cell_grad_ref2_2 + bcoordinates[9] * cell_grad_ref2_3 + bcoordinates[12] * cell_grad_ref2_4 + bcoordinates[15] * cell_grad_ref2_5 + bcoordinates[18] * cell_grad_ref2_6 + bcoordinates[21] * cell_grad_ref2_7 + bcoordinates[24] * cell_grad_ref2_8 + bcoordinates[27] * cell_grad_ref2_9;
      const s_t J10 = bcoordinates[1] * cell_grad_ref0_0 + bcoordinates[4] * cell_grad_ref0_1 + bcoordinates[7] * cell_grad_ref0_2 + bcoordinates[10] * cell_grad_ref0_3 + bcoordinates[13] * cell_grad_ref0_4 + bcoordinates[16] * cell_grad_ref0_5 + bcoordinates[19] * cell_grad_ref0_6 + bcoordinates[22] * cell_grad_ref0_7 + bcoordinates[25] * cell_grad_ref0_8 + bcoordinates[28] * cell_grad_ref0_9;
      const s_t J11 = bcoordinates[1] * cell_grad_ref1_0 + bcoordinates[4] * cell_grad_ref1_1 + bcoordinates[7] * cell_grad_ref1_2 + bcoordinates[10] * cell_grad_ref1_3 + bcoordinates[13] * cell_grad_ref1_4 + bcoordinates[16] * cell_grad_ref1_5 + bcoordinates[19] * cell_grad_ref1_6 + bcoordinates[22] * cell_grad_ref1_7 + bcoordinates[25] * cell_grad_ref1_8 + bcoordinates[28] * cell_grad_ref1_9;
      const s_t J12 = bcoordinates[1] * cell_grad_ref2_0 + bcoordinates[4] * cell_grad_ref2_1 + bcoordinates[7] * cell_grad_ref2_2 + bcoordinates[10] * cell_grad_ref2_3 + bcoordinates[13] * cell_grad_ref2_4 + bcoordinates[16] * cell_grad_ref2_5 + bcoordinates[19] * cell_grad_ref2_6 + bcoordinates[22] * cell_grad_ref2_7 + bcoordinates[25] * cell_grad_ref2_8 + bcoordinates[28] * cell_grad_ref2_9;
      const s_t J20 = bcoordinates[2] * cell_grad_ref0_0 + bcoordinates[5] * cell_grad_ref0_1 + bcoordinates[8] * cell_grad_ref0_2 + bcoordinates[11] * cell_grad_ref0_3 + bcoordinates[14] * cell_grad_ref0_4 + bcoordinates[17] * cell_grad_ref0_5 + bcoordinates[20] * cell_grad_ref0_6 + bcoordinates[23] * cell_grad_ref0_7 + bcoordinates[26] * cell_grad_ref0_8 + bcoordinates[29] * cell_grad_ref0_9;
      const s_t J21 = bcoordinates[2] * cell_grad_ref1_0 + bcoordinates[5] * cell_grad_ref1_1 + bcoordinates[8] * cell_grad_ref1_2 + bcoordinates[11] * cell_grad_ref1_3 + bcoordinates[14] * cell_grad_ref1_4 + bcoordinates[17] * cell_grad_ref1_5 + bcoordinates[20] * cell_grad_ref1_6 + bcoordinates[23] * cell_grad_ref1_7 + bcoordinates[26] * cell_grad_ref1_8 + bcoordinates[29] * cell_grad_ref1_9;
      const s_t J22 = bcoordinates[2] * cell_grad_ref2_0 + bcoordinates[5] * cell_grad_ref2_1 + bcoordinates[8] * cell_grad_ref2_2 + bcoordinates[11] * cell_grad_ref2_3 + bcoordinates[14] * cell_grad_ref2_4 + bcoordinates[17] * cell_grad_ref2_5 + bcoordinates[20] * cell_grad_ref2_6 + bcoordinates[23] * cell_grad_ref2_7 + bcoordinates[26] * cell_grad_ref2_8 + bcoordinates[29] * cell_grad_ref2_9;
      geometry_jacobian_adjugate_and_determinant_3<s_t>(
          J00, J01, J02, J10, J11, J12, J20, J21, J22,
          badjugate_streams, bdeterminant, q);
    }
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_total_d3_simplex_hessian_block<s_t, NQ, NS>(1, bdeterminant, badjugate, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, element_matrix);

    mooney_rivlin_kelvin_voigt_total_tet10_hessian_crs_i_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_total_tet10_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const void *const RSTR u2_old,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_hessian_crs_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, rowptr, colidx, (double *)values);
    }
    case (int)sizeof(float): {
      return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet10_hessian_crs_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, rowptr, colidx, (float *)values);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet10_hessian_bsr_i_msoa", -1, (int)scalar_bytes);
}
