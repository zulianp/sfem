#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_local.hpp"
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[30],
    const void *const RSTR previous[30],
    const real_t eta_b,
    const real_t eta_s,
    const real_t u_dt_shift,
    void *const RSTR output[30]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<double, 11, 10, 16>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tet10_q11<double>::shape(), sfem::codegen::ref_tet10_q11<double>::grad_ref_x(), sfem::codegen::ref_tet10_q11<double>::grad_ref_y(), sfem::codegen::ref_tet10_q11<double>::grad_ref_z(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)current, (const double *const *)previous, eta_b, eta_s, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<float, 11, 10, 16>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tet10_q11<float>::shape(), sfem::codegen::ref_tet10_q11<float>::grad_ref_x(), sfem::codegen::ref_tet10_q11<float>::grad_ref_y(), sfem::codegen::ref_tet10_q11<float>::grad_ref_z(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)current, (const float *const *)previous, eta_b, eta_s, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_impl(
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

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, eta_b, eta_s, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa(
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
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = bcoordinates[0][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J01 = bcoordinates[0][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J02 = bcoordinates[0][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J10 = bcoordinates[1][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J11 = bcoordinates[1][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J12 = bcoordinates[1][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J20 = bcoordinates[2][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J21 = bcoordinates[2][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J22 = bcoordinates[2][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_z[q * NS + 9];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS + lane);
      }
    }

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, eta_b, eta_s, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t eta_b,
    const real_t eta_s,
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
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos(
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
        return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (const double *)previous + 0, (const double *)previous + 1, (const double *)previous + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2);
    }
    case (int)sizeof(float): {
        return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (const float *)previous + 0, (const float *)previous + 1, (const float *)previous + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_esoa(
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
    const real_t u_dt_shift,
    void *const RSTR output[30]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<double, 11, 10, 16>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tet10_q11<double>::shape(), sfem::codegen::ref_tet10_q11<double>::grad_ref_x(), sfem::codegen::ref_tet10_q11<double>::grad_ref_y(), sfem::codegen::ref_tet10_q11<double>::grad_ref_z(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)current, (const double *const *)previous, (const double *const *)direction, eta_b, eta_s, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<float, 11, 10, 16>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tet10_q11<float>::shape(), sfem::codegen::ref_tet10_q11<float>::grad_ref_x(), sfem::codegen::ref_tet10_q11<float>::grad_ref_y(), sfem::codegen::ref_tet10_q11<float>::grad_ref_z(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)current, (const float *const *)previous, (const float *const *)direction, eta_b, eta_s, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_impl(
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

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa(
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
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = bcoordinates[0][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J01 = bcoordinates[0][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J02 = bcoordinates[0][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J10 = bcoordinates[1][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J11 = bcoordinates[1][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J12 = bcoordinates[1][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J20 = bcoordinates[2][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J21 = bcoordinates[2][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J22 = bcoordinates[2][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_z[q * NS + 9];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS + lane);
      }
    }

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t eta_b,
    const real_t eta_s,
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
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos(
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
        return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (const double *)previous + 0, (const double *)previous + 1, (const double *)previous + 2, 3, (const double *)direction + 0, (const double *)direction + 1, (const double *)direction + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2);
    }
    case (int)sizeof(float): {
        return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (const float *)previous + 0, (const float *)previous + 1, (const float *)previous + 2, 3, (const float *)direction + 0, (const float *)direction + 1, (const float *)direction + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_crs_i_msoa_find_cols(
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_crs_i_msoa_scatter_crs(
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
    mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_crs_i_msoa_find_cols(ev, cols, lenrow, ks);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_crs_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
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
  static constexpr int VS = 1;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    const ptrdiff_t evb = element;
    const int ne = 1;
    idx_t ev[NS];
    s_t element_matrix[900];
    s_t bcoordinates[ND * NS][VS];
    s_t badjugate_data[ND * ND][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[N_STREAMS][VS];
    s_t bprevious[N_STREAMS][VS];
    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      const idx_t coordinate_node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinates[shape * ND + d][0] = s_t(coordinate_components[d][coordinate_node]);
      }
      bcurrent[shape * NC + 0][0] = u0[node * current_stride];
      bcurrent[shape * NC + 1][0] = u1[node * current_stride];
      bcurrent[shape * NC + 2][0] = u2[node * current_stride];
      bprevious[shape * NC + 0][0] = u0_old[node * previous_stride];
      bprevious[shape * NC + 1][0] = u1_old[node * previous_stride];
      bprevious[shape * NC + 2][0] = u2_old[node * previous_stride];
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const int lane = 0;
      const s_t J00 = bcoordinates[0][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_x[q * NS + 9];
      const s_t J01 = bcoordinates[0][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_y[q * NS + 9];
      const s_t J02 = bcoordinates[0][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_z[q * NS + 9];
      const s_t J10 = bcoordinates[1][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_x[q * NS + 9];
      const s_t J11 = bcoordinates[1][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_y[q * NS + 9];
      const s_t J12 = bcoordinates[1][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_z[q * NS + 9];
      const s_t J20 = bcoordinates[2][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_x[q * NS + 9];
      const s_t J21 = bcoordinates[2][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_y[q * NS + 9];
      const s_t J22 = bcoordinates[2][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_z[q * NS + 9];
      geometry_jacobian_adjugate_and_determinant_3<s_t>(
          J00, J01, J02, J10, J11, J12, J20, J21, J22,
          badjugate_streams, bdeterminant, q * VS + lane);
    }
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_hessian_block<s_t, NQ, NS, VS>(1, 1, bdeterminant, badjugate, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, eta_b, eta_s, u_dt_shift, element_matrix);

    mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_crs_i_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t eta_b,
    const real_t eta_s,
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
      return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_crs_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, rowptr, colidx, (double *)values);
    }
    case (int)sizeof(float): {
      return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_crs_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, rowptr, colidx, (float *)values);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa", -1, (int)scalar_bytes);
}
