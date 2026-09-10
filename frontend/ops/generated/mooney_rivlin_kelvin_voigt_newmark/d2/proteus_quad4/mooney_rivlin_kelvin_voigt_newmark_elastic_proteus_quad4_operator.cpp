#include <cstdio>
#include <type_traits>
#include "../mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_local.hpp"
#include "../../../reference/line_p1_q2.hpp"
#include "../../../reference/quad_line_q2.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <cstdlib>
#include "../../../packed_thread_scratch.hpp"

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

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
  19,
  15,
  0,
  0,
  9,
  0,
  0,
  0,
  2,
  11,
  43,
  240,
  412,
  10,
  14,
  5,
  8,
  2,
  2,
  8,
  0,
  1,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bu_base_data[NS * NC][VS];
    s_t bh_data[NS * NC][VS];
    s_t bvalue[VS];
    s_t bcoordinate_data[NS * ND][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
        }
      }
    }

    const s_t *const u_components[NC] = {ux, uy};
    const s_t *const h_components[NC] = {hx, hy};
    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < NC; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = ev[shape * VS + lane];
          bu_base_data[shape * NC + d][lane] = u_components[d][node * u_stride];
          bh_data[shape * NC + d][lane] = h_components[d][node * h_stride];
        }
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      for (int shape = 0; shape < NS; ++shape) {
        for (int d = 0; d < NC; ++d) {
          #pragma omp simd
          for (int lane = 0; lane < ne; ++lane) {
            bu_data[shape * NC + d][lane] = bu_base_data[shape * NC + d][lane] + alpha * bh_data[shape * NC + d][lane];
          }
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bvalue[lane] = s_t(0);
      }

      mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bvalue);

      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa_impl<double, geom_t, 16>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa_impl<float, geom_t, 16>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
  30,
  34,
  0,
  0,
  4,
  0,
  0,
  0,
  2,
  12,
  68,
  468,
  640,
  8,
  11,
  5,
  8,
  2,
  2,
  8,
  0,
  8,
  8,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < NC; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = ev[shape * VS + lane];
          bu_data[shape * NC + d][lane] = u_components[d][node * u_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bout_data[stream][lane] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = bout_data[stream];
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < NC; ++d) {
        {
          for (int scatter = 0; scatter < ne; ++scatter) {
            #pragma omp atomic update
            out_components[d][ev[shape * VS + scatter] * out_stride] += bout_data[shape * NC + d][scatter];
          }
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_impl<double, geom_t, 16>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_impl<float, geom_t, 16>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
  33,
  41,
  0,
  0,
  4,
  0,
  0,
  0,
  2,
  23,
  78,
  468,
  640,
  19,
  16,
  5,
  8,
  2,
  2,
  8,
  8,
  8,
  8,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bh_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy};
    const s_t *const h_components[NC] = {hx, hy};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < NC; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = ev[shape * VS + lane];
          bu_data[shape * NC + d][lane] = u_components[d][node * u_stride];
          bh_data[shape * NC + d][lane] = h_components[d][node * h_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bout_data[stream][lane] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = bh_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = bout_data[stream];
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < NC; ++d) {
        {
          for (int scatter = 0; scatter < ne; ++scatter) {
            #pragma omp atomic update
            out_components[d][ev[shape * VS + scatter] * out_stride] += bout_data[shape * NC + d][scatter];
          }
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_impl<double, geom_t, 16>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_impl<float, geom_t, 16>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa", -1, (int)scalar_bytes);
}
