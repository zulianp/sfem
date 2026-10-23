#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "vector_mass_d3_simplex_local.hpp"
#include "geometry_kernels.hpp"
#include "kernel_diagnostics.hpp"
#include "packed_thread_scratch.hpp"
#include "quad_tet_q11.hpp"
#include "tet10_q11.hpp"
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

static const KernelDiagnostics vector_mass_tet10_residual_esoa_diagnostics_data = {
  "vector_mass_tet10_residual_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  6,
  0,
  0,
  0,
  0,
  0,
  0,
  7,
  3,
  6,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_residual_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u0_u0_diagnostics_data = {
  "vector_mass_tet10_jacobian_u0_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  1,
  2,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u0_u1_diagnostics_data = {
  "vector_mass_tet10_jacobian_u0_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u0_u2_diagnostics_data = {
  "vector_mass_tet10_jacobian_u0_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u0_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u0_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u1_u0_diagnostics_data = {
  "vector_mass_tet10_jacobian_u1_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u1_u1_diagnostics_data = {
  "vector_mass_tet10_jacobian_u1_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  1,
  2,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u1_u2_diagnostics_data = {
  "vector_mass_tet10_jacobian_u1_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u1_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u1_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u2_u0_diagnostics_data = {
  "vector_mass_tet10_jacobian_u2_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u2_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u2_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u2_u1_diagnostics_data = {
  "vector_mass_tet10_jacobian_u2_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u2_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u2_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u2_u2_diagnostics_data = {
  "vector_mass_tet10_jacobian_u2_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  1,
  2,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_u2_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u2_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_action_esoa_diagnostics_data = {
  "vector_mass_tet10_jacobian_action_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  6,
  0,
  0,
  0,
  0,
  0,
  0,
  7,
  3,
  6,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_action_esoa_diagnostics_data;
}

extern "C" int vector_mass_tet10_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR current[30],
    const real_t density,
    void *const RSTR output[30]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d3_simplex_residual_block<double, 11, 10, 16>(ne, geometry_stride, (const double *)determinant, sfem::codegen::ref_tet10_q11<double>::shape(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)current, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d3_simplex_residual_block<float, 11, 10, 16>(ne, geometry_stride, (const float *)determinant, sfem::codegen::ref_tet10_q11<float>::shape(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)current, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int vector_mass_tet10_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1, u2};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
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

    vector_mass_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape, affine_q_weight, bcurrent, density, boutput);

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

extern "C" int vector_mass_tet10_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::vector_mass_tet10_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, density, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_tet10_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, density, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int vector_mass_tet10_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
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

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
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


    vector_mass_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape, isoparametric_q_weight, bcurrent, density, boutput);

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

extern "C" int vector_mass_tet10_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::vector_mass_tet10_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, density, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_tet10_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, density, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int vector_mass_tet10_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return vector_mass_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2);
    }
    case (int)sizeof(float): {
        return vector_mass_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_i_maos", -1, (int)scalar_bytes);
}

extern "C" int vector_mass_tet10_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR direction[30],
    const real_t density,
    void *const RSTR output[30]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d3_simplex_jacobian_action_block<double, 11, 10, 16>(ne, geometry_stride, (const double *)determinant, sfem::codegen::ref_tet10_q11<double>::shape(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)direction, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d3_simplex_jacobian_action_block<float, 11, 10, 16>(ne, geometry_stride, (const float *)determinant, sfem::codegen::ref_tet10_q11<float>::shape(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)direction, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int vector_mass_tet10_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
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

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1][VS];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t, VS>(
        ne, affine_geometry_sources[0], baffine_geometry_data[0], std::is_same<g_t, s_t>());

    vector_mass_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape, affine_q_weight, bdirection, density, boutput);

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

extern "C" int vector_mass_tet10_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
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
        return sfem::codegen::vector_mass_tet10_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_tet10_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int vector_mass_tet10_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
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
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
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


    vector_mass_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape, isoparametric_q_weight, bdirection, density, boutput);

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

extern "C" int vector_mass_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
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
        return sfem::codegen::vector_mass_tet10_jacobian_action_i_msoa_impl<double>(nelements, nnodes, elements, points, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_tet10_jacobian_action_i_msoa_impl<float>(nelements, nnodes, elements, points, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int vector_mass_tet10_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return vector_mass_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 3, (const double *)direction + 0, (const double *)direction + 1, (const double *)direction + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2);
    }
    case (int)sizeof(float): {
        return vector_mass_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 3, (const float *)direction + 0, (const float *)direction + 1, (const float *)direction + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_i_maos", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

static SFEM_INLINE void vector_mass_tet10_hessian_crs_i_msoa_find_cols(
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
static SFEM_INLINE void vector_mass_tet10_hessian_crs_i_msoa_scatter_crs(
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
    vector_mass_tet10_hessian_crs_i_msoa_find_cols(ev, cols, lenrow, ks);
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
static SFEM_INLINE int vector_mass_tet10_hessian_crs_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
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
    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      const idx_t coordinate_node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinates[shape * ND + d] = s_t(coordinate_components[d][coordinate_node]);
      }
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

    vector_mass_d3_simplex_hessian_block<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape, isoparametric_q_weight, density, element_matrix);

    vector_mass_tet10_hessian_crs_i_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int vector_mass_tet10_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::vector_mass_tet10_hessian_crs_i_msoa_impl<double>(nelements, nnodes, elements, points, density, rowptr, colidx, (double *)values);
    }
    case (int)sizeof(float): {
      return sfem::codegen::vector_mass_tet10_hessian_crs_i_msoa_impl<float>(nelements, nnodes, elements, points, density, rowptr, colidx, (float *)values);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_hessian_bsr_i_msoa", -1, (int)scalar_bytes);
}
