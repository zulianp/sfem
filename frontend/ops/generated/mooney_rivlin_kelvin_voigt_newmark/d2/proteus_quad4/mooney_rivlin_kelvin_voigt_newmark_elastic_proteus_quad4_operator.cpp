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
#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

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
  18,
  0,
  0,
  9,
  0,
  0,
  0,
  2,
  11,
  46,
  240,
  444,
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

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_a_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_a_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_i_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_i_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
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
  static constexpr int VS = 16;
  (void)nnodes;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

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
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);

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
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
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
  45,
  0,
  0,
  4,
  0,
  0,
  0,
  2,
  12,
  79,
  468,
  672,
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

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_a_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_a_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
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
  static constexpr int VS = 16;
  (void)nnodes;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

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
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);

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
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
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
  53,
  0,
  0,
  4,
  0,
  0,
  0,
  2,
  23,
  90,
  468,
  672,
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

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_a_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_a_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
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
  static constexpr int VS = 16;
  (void)nnodes;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

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
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);

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
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}
