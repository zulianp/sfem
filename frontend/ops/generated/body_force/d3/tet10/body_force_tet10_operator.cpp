#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../body_force_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/quad_tet_q4.hpp"
#include "../../../reference/tet10_q4.hpp"
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

static const KernelDiagnostics body_force_tet10_residual_esoa_diagnostics_data = {
  "body_force_tet10_residual_esoa",
  "TET10",
  3,
  4,
  10,
  16,
  2,
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
  1876,
  2760,
  0,
  3,
  10,
  160,
  4,
  4,
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

extern "C" const sfem::codegen::KernelDiagnostics *body_force_tet10_residual_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tet10_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics body_force_tet10_jacobian_action_esoa_diagnostics_data = {
  "body_force_tet10_jacobian_action_esoa",
  "TET10",
  3,
  4,
  10,
  16,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  0,
  1876,
  2760,
  0,
  0,
  10,
  160,
  4,
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

extern "C" const sfem::codegen::KernelDiagnostics *body_force_tet10_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tet10_jacobian_action_esoa_diagnostics_data;
}

extern "C" int body_force_tet10_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    void *const RSTR output[30]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::body_force_d3_simplex_residual_block<double, 4, 10, 16>(ne, geometry_stride, (const double *)determinant, sfem::codegen::ref_tet10_q4<double>::shape(), sfem::codegen::quad_tet_q4<double>::q_weight(), density, g0, g1, g2, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::body_force_d3_simplex_residual_block<float, 4, 10, 16>(ne, geometry_stride, (const float *)determinant, sfem::codegen::ref_tet10_q4<float>::shape(), sfem::codegen::quad_tet_q4<float>::q_weight(), density, g0, g1, g2, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int body_force_tet10_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
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
    s_t boutput[NC * NS][VS];

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

    body_force_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape, affine_q_weight, density, g0, g1, g2, boutput);

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

extern "C" int body_force_tet10_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::body_force_tet10_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::body_force_tet10_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int body_force_tet10_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q4<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

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
        s_t *const RSTR bcoordinate_row = bcoordinates[shape * ND + d];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinate_row[lane] = coordinate_components[d][node];
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


    body_force_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape, isoparametric_q_weight, density, g0, g1, g2, boutput);

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

extern "C" int body_force_tet10_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::body_force_tet10_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, density, g0, g1, g2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::body_force_tet10_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, density, g0, g1, g2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int body_force_tet10_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return body_force_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], ((const double *)parameters)[3], 3, (double *)output + 0, (double *)output + 1, (double *)output + 2);
    }
    case (int)sizeof(float): {
        return body_force_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], ((const float *)parameters)[3], 3, (float *)output + 0, (float *)output + 1, (float *)output + 2);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_i_maos", -1, (int)scalar_bytes);
}
