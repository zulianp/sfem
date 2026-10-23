#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "vector_mass_d2_tensor_product_local.hpp"
#include "geometry_kernels.hpp"
#include "kernel_diagnostics.hpp"
#include "packed_thread_scratch.hpp"
#include "line_p1_q2.hpp"
#include "quad_line_q2.hpp"
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

static const KernelDiagnostics vector_mass_proteus_quad4_residual_esoa_diagnostics_data = {
  "vector_mass_proteus_quad4_residual_esoa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
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
  468,
  640,
  0,
  3,
  5,
  8,
  2,
  1,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_quad4_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_quad4_jacobian_u0_u0_diagnostics_data = {
  "vector_mass_proteus_quad4_jacobian_u0_u0",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
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
  468,
  640,
  0,
  3,
  5,
  8,
  2,
  1,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_quad4_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_quad4_jacobian_u0_u1_diagnostics_data = {
  "vector_mass_proteus_quad4_jacobian_u0_u1",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
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
  1,
  0,
  468,
  640,
  0,
  0,
  5,
  8,
  2,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_quad4_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_quad4_jacobian_u1_u0_diagnostics_data = {
  "vector_mass_proteus_quad4_jacobian_u1_u0",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
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
  1,
  0,
  468,
  640,
  0,
  0,
  5,
  8,
  2,
  0,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_quad4_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_quad4_jacobian_u1_u1_diagnostics_data = {
  "vector_mass_proteus_quad4_jacobian_u1_u1",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
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
  468,
  640,
  0,
  3,
  5,
  8,
  2,
  1,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_quad4_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_quad4_jacobian_action_esoa_diagnostics_data = {
  "vector_mass_proteus_quad4_jacobian_action_esoa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
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
  468,
  640,
  0,
  3,
  5,
  8,
  2,
  1,
  0,
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

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_quad4_jacobian_action_esoa_diagnostics_data;
}

extern "C" int vector_mass_proteus_quad4_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR current[8],
    const real_t density,
    void *const RSTR output[8]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d2_tensor_product_residual_block<double, 4, 4, 16>(ne, geometry_stride, (const double *)determinant, sfem::codegen::ref_line_p1_q2<double>::shape_1d(), sfem::codegen::quad_line_q2<double>::q_weight_1d(), (const double *const *)current, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d2_tensor_product_residual_block<float, 4, 4, 16>(ne, geometry_stride, (const float *)determinant, sfem::codegen::ref_line_p1_q2<float>::shape_1d(), sfem::codegen::quad_line_q2<float>::q_weight_1d(), (const float *const *)current, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int vector_mass_proteus_quad4_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1};

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

    for (int stream = 0; stream < 8; ++stream) {
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

    vector_mass_d2_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape_1d, affine_q_weight_1d, bcurrent, density, boutput);

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

extern "C" int vector_mass_proteus_quad4_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::vector_mass_proteus_quad4_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, density, current_stride, (const double *)u0, (const double *)u1, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_proteus_quad4_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, density, current_stride, (const float *)u0, (const float *)u1, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int vector_mass_proteus_quad4_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[2 * NS][VS];
    s_t badjugate_data[4][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1]};
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
    const s_t *const current_components[NC] = {u0, u1};

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

    for (int stream = 0; stream < 8; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);


    vector_mass_d2_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, bcurrent, density, boutput);

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

extern "C" int vector_mass_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::vector_mass_proteus_quad4_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, density, current_stride, (const double *)u0, (const double *)u1, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_proteus_quad4_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, density, current_stride, (const float *)u0, (const float *)u1, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int vector_mass_proteus_quad4_residual_i_maos(
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
        return vector_mass_proteus_quad4_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 2, (const double *)current + 0, (const double *)current + 1, 2, (double *)output + 0, (double *)output + 1);
    }
    case (int)sizeof(float): {
        return vector_mass_proteus_quad4_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 2, (const float *)current + 0, (const float *)current + 1, 2, (float *)output + 0, (float *)output + 1);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_residual_i_maos", -1, (int)scalar_bytes);
}

extern "C" int vector_mass_proteus_quad4_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR direction[8],
    const real_t density,
    void *const RSTR output[8]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d2_tensor_product_jacobian_action_block<double, 4, 4, 16>(ne, geometry_stride, (const double *)determinant, sfem::codegen::ref_line_p1_q2<double>::shape_1d(), sfem::codegen::quad_line_q2<double>::q_weight_1d(), (const double *const *)direction, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d2_tensor_product_jacobian_action_block<float, 4, 4, 16>(ne, geometry_stride, (const float *)determinant, sfem::codegen::ref_line_p1_q2<float>::shape_1d(), sfem::codegen::quad_line_q2<float>::q_weight_1d(), (const float *const *)direction, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int vector_mass_proteus_quad4_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const direction_components[NC] = {u0_direction, u1_direction};

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

    for (int stream = 0; stream < 8; ++stream) {
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

    vector_mass_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape_1d, affine_q_weight_1d, bdirection, density, boutput);

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

extern "C" int vector_mass_proteus_quad4_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::vector_mass_proteus_quad4_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_det0, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_proteus_quad4_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_det0, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int vector_mass_proteus_quad4_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[2 * NS][VS];
    s_t badjugate_data[4][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1]};
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
    const s_t *const direction_components[NC] = {u0_direction, u1_direction};

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

    for (int stream = 0; stream < 8; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);


    vector_mass_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, bdirection, density, boutput);

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

extern "C" int vector_mass_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::vector_mass_proteus_quad4_jacobian_action_i_msoa_impl<double>(nelements, nnodes, elements, points, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::vector_mass_proteus_quad4_jacobian_action_i_msoa_impl<float>(nelements, nnodes, elements, points, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int vector_mass_proteus_quad4_jacobian_action_i_maos(
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
        return vector_mass_proteus_quad4_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 2, (const double *)direction + 0, (const double *)direction + 1, 2, (double *)output + 0, (double *)output + 1);
    }
    case (int)sizeof(float): {
        return vector_mass_proteus_quad4_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 2, (const float *)direction + 0, (const float *)direction + 1, 2, (float *)output + 0, (float *)output + 1);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_quad4_jacobian_action_i_maos", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

static SFEM_INLINE void vector_mass_proteus_quad4_hessian_crs_i_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(4)
  for (int d = 0; d < 4; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(4)
    for (int d = 0; d < 4; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t, typename v_t>
static SFEM_INLINE void vector_mass_proteus_quad4_hessian_crs_i_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    v_t *const RSTR values) {
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int N_COL_STREAMS = 8;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    vector_mass_proteus_quad4_hessian_crs_i_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int bi = 0; bi < NC; ++bi) {
    for (int row_shape = 0; row_shape < NS; ++row_shape) {
      const s_t *const RSTR row = &element_matrix[(bi * NS + row_shape) * N_COL_STREAMS];
      for (int bj = 0; bj < NC; ++bj) {
        for (int col_shape = 0; col_shape < NS; ++col_shape) {
          v_t *const block = &values[entries[row_shape * NS + col_shape] * NC * NC];
#pragma omp atomic update
          block[bi * NC + bj] += v_t(row[bj * NS + col_shape]);
        }
      }
    }
  }
}

template <typename s_t, typename v_t>
static SFEM_INLINE int vector_mass_proteus_quad4_hessian_crs_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    v_t *const RSTR values
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    idx_t ev[NS];
    s_t element_matrix[64];
    s_t bcoordinates[ND * NS];
    s_t badjugate_data[ND * ND][NQ];
    s_t bdeterminant[NQ];
    const geom_t *const coordinate_components[ND] = {points[0], points[1]};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      const idx_t coordinate_node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinates[shape * ND + d] = s_t(coordinate_components[d][coordinate_node]);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND];
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 2>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 2>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
    geometry_jacobian_adjugate_and_determinant_scalar<s_t, ND, NQ>(
        coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

    vector_mass_d2_tensor_product_hessian_block<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, density, element_matrix);

    vector_mass_proteus_quad4_hessian_crs_i_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int vector_mass_proteus_quad4_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const int values_bytes,
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
      switch (values_bytes) {
        case (int)sizeof(double): {
          return sfem::codegen::vector_mass_proteus_quad4_hessian_crs_i_msoa_impl<double, double>(nelements, nnodes, elements, points, density, rowptr, colidx, (double *)values);
        }
        case (int)sizeof(float): {
          return sfem::codegen::vector_mass_proteus_quad4_hessian_crs_i_msoa_impl<double, float>(nelements, nnodes, elements, points, density, rowptr, colidx, (float *)values);
        }
        default:
          break;
      }
      break;
    }
    case (int)sizeof(float): {
      switch (values_bytes) {
        case (int)sizeof(float): {
          return sfem::codegen::vector_mass_proteus_quad4_hessian_crs_i_msoa_impl<float, float>(nelements, nnodes, elements, points, density, rowptr, colidx, (float *)values);
        }
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_values_dispatch("vector_mass_proteus_quad4_hessian_bsr_i_msoa", (int)scalar_bytes, (int)values_bytes);
}
