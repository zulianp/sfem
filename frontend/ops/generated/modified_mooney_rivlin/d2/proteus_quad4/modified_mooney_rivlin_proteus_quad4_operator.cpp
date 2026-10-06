#include <cstdio>
#include <type_traits>
#include "../modified_mooney_rivlin_d2_tensor_product_local.hpp"
#include "../modified_mooney_rivlin_d2_tensor_product_hessian.hpp"
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

static const KernelDiagnostics modified_mooney_rivlin_proteus_quad4_objective_soa_diagnostics_data = {
  "modified_mooney_rivlin_proteus_quad4_objective_soa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
  19,
  16,
  0,
  0,
  11,
  0,
  1,
  0,
  2,
  11,
  66,
  240,
  412,
  10,
  15,
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

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_quad4_objective_soa_diagnostics(void) {
  return &sfem::codegen::modified_mooney_rivlin_proteus_quad4_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_proteus_quad4_objective_steps_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
    s_t bh_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev_node[lane] = element_shape[lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev_shape[lane]];
        }
      }
    }

    const s_t *const u_components[NC] = {ux, uy};
    const s_t *const h_components[NC] = {hx, hy};
    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = bh_data[stream];
    }

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = ev_shape[lane];
          bu_data[shape * NC + d][lane] = u_components[d][node * u_stride];
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        value[(ptrdiff_t)step * nelements + evb + lane] = s_t(0);
      }
    }

    modified_mooney_rivlin_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, c1, c2, kappa, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int modified_mooney_rivlin_proteus_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_objective_steps_i_msoa_impl<double, geom_t, 16>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
    }
    case (int)sizeof(float): {
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_objective_steps_i_msoa_impl<float, geom_t, 16>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_proteus_quad4_objective_steps_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics modified_mooney_rivlin_proteus_quad4_gradient_soa_diagnostics_data = {
  "modified_mooney_rivlin_proteus_quad4_gradient_soa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
  46,
  62,
  2,
  0,
  10,
  0,
  1,
  0,
  2,
  31,
  154,
  468,
  640,
  27,
  23,
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

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_quad4_gradient_soa_diagnostics(void) {
  return &sfem::codegen::modified_mooney_rivlin_proteus_quad4_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_proteus_quad4_gradient_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev_node[lane] = element_shape[lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev_shape[lane]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = ev_shape[lane];
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

    modified_mooney_rivlin_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, c1, c2, kappa, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          for (int scatter = 0; scatter < ne; ++scatter) {
            #pragma omp atomic update
            out_components[d][ev_shape[scatter] * out_stride] += bout_data[shape * NC + d][scatter];
          }
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int modified_mooney_rivlin_proteus_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_gradient_i_msoa_impl<double, geom_t, 16>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_gradient_i_msoa_impl<float, geom_t, 16>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_proteus_quad4_gradient_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics modified_mooney_rivlin_proteus_quad4_apply_soa_diagnostics_data = {
  "modified_mooney_rivlin_proteus_quad4_apply_soa",
  "PROTEUS_QUAD4",
  2,
  4,
  4,
  16,
  2,
  129,
  183,
  2,
  0,
  13,
  0,
  1,
  0,
  2,
  86,
  361,
  468,
  640,
  82,
  51,
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

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_quad4_apply_soa_diagnostics(void) {
  return &sfem::codegen::modified_mooney_rivlin_proteus_quad4_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
static SFEM_INLINE int modified_mooney_rivlin_proteus_quad4_apply_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev_node[lane] = element_shape[lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev_shape[lane]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy};
    const s_t *const h_components[NC] = {hx, hy};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = ev_shape[lane];
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

    modified_mooney_rivlin_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, c1, c2, kappa, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          for (int scatter = 0; scatter < ne; ++scatter) {
            #pragma omp atomic update
            out_components[d][ev_shape[scatter] * out_stride] += bout_data[shape * NC + d][scatter];
          }
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int modified_mooney_rivlin_proteus_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_apply_i_msoa_impl<double, geom_t, 16>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_apply_i_msoa_impl<float, geom_t, 16>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_proteus_quad4_apply_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static SFEM_INLINE void modified_mooney_rivlin_proteus_quad4_hessian_i_msoa_find_cols(
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

template <typename s_t>
static SFEM_INLINE void modified_mooney_rivlin_proteus_quad4_hessian_i_msoa_scatter_bsr(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NC = 2;
  static constexpr int NS = 4;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const idx_t dof_i = ev[i];
    const count_t row_begin = rowptr[dof_i];
    const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    modified_mooney_rivlin_proteus_quad4_hessian_i_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int i = 0; i < NS; ++i) {
    for (int j = 0; j < NS; ++j) {
      s_t *const block = &values[entries[i * NS + j] * NC * NC];
      for (int bi = 0; bi < NC; ++bi) {
        const int row = bi * NS + i;
        for (int bj = 0; bj < NC; ++bj) {
          const int col = bj * NS + j;
#pragma omp atomic update
          block[bi * NC + bj] += element_matrix[row * (NC * NS) + col];
        }
      }
    }
  }
}

template <typename s_t, typename g_t, int FORMAT>
static int modified_mooney_rivlin_proteus_quad4_hessian_i_msoa_assemble_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const *const RSTR points,
    const s_t c1,
    const s_t c2,
    const s_t kappa,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values,
    const int *const RSTR,
    const ptrdiff_t,
    const ptrdiff_t,
    const idx_t *const RSTR,
    const idx_t *const RSTR,
    idx_t *const RSTR,
    idx_t *const RSTR) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  static constexpr int VS = 1;
  static constexpr int NDOFS = NC * NS;
  const s_t *const u_components[NC] = {ux, uy};
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

  static_assert(FORMAT == 1,
                "this kernel has no scatter for the requested matrix format");
#pragma omp parallel for schedule(static)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    idx_t ev[NS];
    s_t element_matrix[NDOFS * NDOFS];
    s_t bcoordinate_data[NS * ND][VS];
    static constexpr int ne = VS;
    s_t bu_data[NS * NC][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinate_data[shape * ND + d][0] = s_t(points[d][node]);
        bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
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

    modified_mooney_rivlin_d2_tensor_product_direct_hessian_tensor_product_element_matrix<s_t, NQ, NS, VS>(badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, c1, c2, kappa, bu_data, element_matrix);

    if constexpr (FORMAT == 1) {
      modified_mooney_rivlin_proteus_quad4_hessian_i_msoa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int modified_mooney_rivlin_proteus_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_hessian_i_msoa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, rowptr, colidx, (double *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
    }
    case (int)sizeof(float): {
        return sfem::codegen::modified_mooney_rivlin_proteus_quad4_hessian_i_msoa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, rowptr, colidx, (float *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_proteus_quad4_hessian_bsr_i_msoa", -1, (int)scalar_bytes);
}
