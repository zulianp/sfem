#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_local.hpp"
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
struct mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data {
  static const s_t *shape() {
    static const s_t data[4] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[4] = {s_t(-1), s_t(1), s_t(0), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[4] = {s_t(-1), s_t(0), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_z() {
    static const s_t data[4] = {s_t(-1), s_t(0), s_t(0), s_t(1)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[1] = {s_t(0.16666666666666666)};
    return data;
  }
};

template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data {
  static const s_t *shape() {
    static const s_t data[4] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[4] = {s_t(-1), s_t(1), s_t(0), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[4] = {s_t(-1), s_t(0), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_z() {
    static const s_t data[4] = {s_t(-1), s_t(0), s_t(0), s_t(1)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[1] = {s_t(0.16666666666666666)};
    return data;
  }
};

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[9],
    const double *const RSTR current[12],
    const double *const RSTR previous[12],
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    double *const RSTR output[12]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<double, 1, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::q_weight(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[9],
    const float *const RSTR current[12],
    const float *const RSTR previous[12],
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    float *const RSTR output[12]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<float, 1, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::q_weight(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa_impl(
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
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::q_weight();

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

    for (int stream = 0; stream < 12; ++stream) {
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

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, eta_b, eta_s, newmark_velocity_alpha, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa(
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
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa_float(
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
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[9],
    const double *const RSTR current[12],
    const double *const RSTR previous[12],
    const double *const RSTR direction[12],
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    double *const RSTR output[12]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<double, 1, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<double>::q_weight(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[9],
    const float *const RSTR current[12],
    const float *const RSTR previous[12],
    const float *const RSTR direction[12],
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    float *const RSTR output[12]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<float, 1, 4, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_isoparametric_reference_data<float>::q_weight(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa_impl(
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
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_affine_reference_data<s_t>::q_weight();

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

    for (int stream = 0; stream < 12; ++stream) {
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

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, newmark_velocity_alpha, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa(
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
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const double *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa_float(
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
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const float *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}
