#include <cstdio>
#include <type_traits>
#include "../mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_local.hpp"
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


template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data {
  static const s_t *shape_1d() { return ref_line_p1_q2<s_t>::shape_1d(); }
  static const s_t *grad_1d() { return ref_line_p1_q2<s_t>::grad_1d(); }
  static const s_t *q_weight_1d() { return quad_line_q2<s_t>::q_weight_1d(); }
};

template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data {
  static const s_t *shape_1d() { return ref_line_p1_q2<s_t>::shape_1d(); }
  static const s_t *grad_1d() { return ref_line_p1_q2<s_t>::grad_1d(); }
  static const s_t *q_weight_1d() { return quad_line_q2<s_t>::q_weight_1d(); }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa",
  "PROTEUS_HEX8",
  3,
  8,
  8,
  16,
  2,
  45,
  49,
  0,
  0,
  17,
  0,
  0,
  0,
  6,
  22,
  111,
  1232,
  2328,
  21,
  29,
  10,
  8,
  2,
  2,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_a_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_a_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_i_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_i_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa_impl(
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
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
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

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }

    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};
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
    s_t badj0_data[VS];
    const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj0 + evb, badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data[VS];
    const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj1 + evb, badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data[VS];
    const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj2 + evb, badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data[VS];
    const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj3 + evb, badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data[VS];
    const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj4 + evb, badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data[VS];
    const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj5 + evb, badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data[VS];
    const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj6 + evb, badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data[VS];
    const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj7 + evb, badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data[VS];
    const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj8 + evb, badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data[VS];
    const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
        ne, g_det0 + evb, bdet0_data, std::is_same<g_t, s_t>());

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

      mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bvalue);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_a_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const int nsteps,
    const s_t *const RSTR steps,
    s_t *const RSTR value
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  (void)n_shared_nodes;

  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_u_base = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const s_t *const u_components[NC] = {ux, uy, uz};
      const s_t *const h_components[NC] = {hx, hy, hz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_u_base_component = pk_u_base + d * max_nodes_per_pack;
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        const s_t *const RSTR u_component = u_components[d];
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_u_base_component[k] = u_component[node * u_stride];
          pk_h_component[k] = h_component[node * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_u_base_component[n_contiguous + k] = u_component[node * u_stride];
          pk_h_component[n_contiguous + k] = h_component[node * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bu_base_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bvalue[VS];

        const s_t *bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[3], bu_data[4], bu_data[5], bu_data[6], bu_data[7], bu_data[8], bu_data[9], bu_data[10], bu_data[11], bu_data[12], bu_data[13], bu_data[14], bu_data[15], bu_data[16], bu_data[17], bu_data[18], bu_data[19], bu_data[20], bu_data[21], bu_data[22], bu_data[23]};

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_base_data[shape * NC + d][lane] = pk_u_base[d * max_nodes_per_pack + packed_node];
              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
            }
          }
        }

        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj0 + evb, badj0_data, std::is_same<geom_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj1 + evb, badj1_data, std::is_same<geom_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj2 + evb, badj2_data, std::is_same<geom_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj3 + evb, badj3_data, std::is_same<geom_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj4 + evb, badj4_data, std::is_same<geom_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj5 + evb, badj5_data, std::is_same<geom_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj6 + evb, badj6_data, std::is_same<geom_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj7 + evb, badj7_data, std::is_same<geom_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj8 + evb, badj8_data, std::is_same<geom_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_det0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

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

          mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bvalue);

#pragma omp simd
          for (int lane = 0; lane < ne; ++lane) {
            value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];
          }
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_a_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const int nsteps,
    const double *const RSTR steps,
    double *const RSTR value
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_a_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const int nsteps,
    const float *const RSTR steps,
    float *const RSTR value
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
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
    s_t badj4[NQ * VS];
    s_t badj5[NQ * VS];
    s_t badj6[NQ * VS];
    s_t badj7[NQ * VS];
    s_t badj8[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y, z};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
        }
      }
    }

    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};
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
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
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

      mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bvalue);

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_i_msoa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_i_msoa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_i_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const int nsteps,
    const s_t *const RSTR steps,
    s_t *const RSTR value
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  (void)n_shared_nodes;

  const geom_t *const RSTR x = points[0];
  const geom_t *const RSTR y = points[1];
  const geom_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_u_base = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const geom_t *const coordinate_components[ND] = {x, y, z};
      for (int d = 0; d < ND; ++d) {
        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;
        const geom_t *const RSTR coordinate_component = coordinate_components[d];
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_coordinate[k] = s_t(coordinate_component[node]);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);
        }
      }
      const s_t *const u_components[NC] = {ux, uy, uz};
      const s_t *const h_components[NC] = {hx, hy, hz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_u_base_component = pk_u_base + d * max_nodes_per_pack;
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        const s_t *const RSTR u_component = u_components[d];
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_u_base_component[k] = u_component[node * u_stride];
          pk_h_component[k] = h_component[node * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_u_base_component[n_contiguous + k] = u_component[node * u_stride];
          pk_h_component[n_contiguous + k] = h_component[node * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bu_base_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bvalue[VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t badj4[NQ * VS];
        s_t badj5[NQ * VS];
        s_t badj6[NQ * VS];
        s_t badj7[NQ * VS];
        s_t badj8[NQ * VS];
        s_t bdet0[NQ * VS];
        s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};

        const s_t *bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[3], bu_data[4], bu_data[5], bu_data[6], bu_data[7], bu_data[8], bu_data[9], bu_data[10], bu_data[11], bu_data[12], bu_data[13], bu_data[14], bu_data[15], bu_data[16], bu_data[17], bu_data[18], bu_data[19], bu_data[20], bu_data[21], bu_data[22], bu_data[23]};

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < ND; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bcoordinate_data[shape * ND + d][lane] = pk_coordinates[d * max_nodes_per_pack + packed_node];
            }
          }
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_base_data[shape * NC + d][lane] = pk_u_base[d * max_nodes_per_pack + packed_node];
              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
            }
          }
        }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
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

          mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bvalue);

#pragma omp simd
          for (int lane = 0; lane < ne; ++lane) {
            value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];
          }
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_i_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const int nsteps,
    const double *const RSTR steps,
    double *const RSTR value
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_i_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_i_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const int nsteps,
    const float *const RSTR steps,
    float *const RSTR value
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_packed_i_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa",
  "PROTEUS_HEX8",
  3,
  8,
  8,
  16,
  2,
  95,
  150,
  0,
  0,
  9,
  0,
  0,
  0,
  6,
  39,
  254,
  2456,
  3552,
  30,
  32,
  10,
  8,
  2,
  2,
  24,
  0,
  24,
  24,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_impl(
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
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};

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
    s_t badj0_data[VS];
    const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj0 + evb, badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data[VS];
    const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj1 + evb, badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data[VS];
    const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj2 + evb, badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data[VS];
    const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj3 + evb, badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data[VS];
    const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj4 + evb, badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data[VS];
    const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj5 + evb, badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data[VS];
    const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj6 + evb, badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data[VS];
    const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj7 + evb, badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data[VS];
    const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj8 + evb, badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data[VS];
    const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
        ne, g_det0 + evb, bdet0_data, std::is_same<g_t, s_t>());

    mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_a_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_shared = n_shared_nodes[pack];
      const ptrdiff_t n_not_shared = n_contiguous - n_shared;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const s_t *const u_components[NC] = {ux, uy, uz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        const s_t *const RSTR u_component = u_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_u_component[k] = u_component[node * u_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bu_streams[stream] = bu_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bout_streams[stream] = bout_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj0 + evb, badj0_data, std::is_same<geom_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj1 + evb, badj1_data, std::is_same<geom_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj2 + evb, badj2_data, std::is_same<geom_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj3 + evb, badj3_data, std::is_same<geom_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj4 + evb, badj4_data, std::is_same<geom_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj5 + evb, badj5_data, std::is_same<geom_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj6 + evb, badj6_data, std::is_same<geom_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj7 + evb, badj7_data, std::is_same<geom_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj8 + evb, badj8_data, std::is_same<geom_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_det0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
          global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_a_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_a_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_a_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    s_t *const RSTR ghost_buf,
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
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      (void)n_shared_nodes;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const ptrdiff_t ghost_off = ghost_ptr[pack];
      const s_t *const u_components[NC] = {ux, uy, uz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        const s_t *const RSTR u_component = u_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_u_component[k] = u_component[node * u_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bu_streams[stream] = bu_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bout_streams[stream] = bout_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj0 + evb, badj0_data, std::is_same<geom_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj1 + evb, badj1_data, std::is_same<geom_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj2 + evb, badj2_data, std::is_same<geom_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj3 + evb, badj3_data, std::is_same<geom_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj4 + evb, badj4_data, std::is_same<geom_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj5 + evb, badj5_data, std::is_same<geom_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj6 + evb, badj6_data, std::is_same<geom_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj7 + evb, badj7_data, std::is_same<geom_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj8 + evb, badj8_data, std::is_same<geom_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_det0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }

  s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
    const idx_t dest = ghost_reduce_dest[row];
    const ptrdiff_t begin = ghost_reduce_ptr[row];
    const ptrdiff_t end = ghost_reduce_ptr[row + 1];
    for (int d = 0; d < NC; ++d) {
      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
      s_t sum = s_t(0);
      for (ptrdiff_t j = begin; j < end; ++j) {
        sum += ghost_component[ghost_reduce_idx[j]];
      }
      out_components[d][dest * out_stride] += sum;
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_a_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    double *const RSTR ghost_buf,
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
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_a_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    float *const RSTR ghost_buf,
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
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
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
    s_t badj4[NQ * VS];
    s_t badj5[NQ * VS];
    s_t badj6[NQ * VS];
    s_t badj7[NQ * VS];
    s_t badj8[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y, z};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};

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
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_i_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const geom_t *const RSTR x = points[0];
  const geom_t *const RSTR y = points[1];
  const geom_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_shared = n_shared_nodes[pack];
      const ptrdiff_t n_not_shared = n_contiguous - n_shared;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const geom_t *const coordinate_components[ND] = {x, y, z};
      const s_t *const u_components[NC] = {ux, uy, uz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        const geom_t *const RSTR coordinate_component = coordinate_components[d];
        const s_t *const RSTR u_component = u_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_coordinate[k] = s_t(coordinate_component[node]);
          pk_u_component[k] = u_component[node * u_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t badj4[NQ * VS];
        s_t badj5[NQ * VS];
        s_t badj6[NQ * VS];
        s_t badj7[NQ * VS];
        s_t badj8[NQ * VS];
        s_t bdet0[NQ * VS];
        s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bu_streams[stream] = bu_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bout_streams[stream] = bout_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < ND; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bcoordinate_data[shape * ND + d][lane] = pk_coordinates[d * max_nodes_per_pack + packed_node];
            }
          }
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
          global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_i_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_i_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_i_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_i_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_i_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    s_t *const RSTR ghost_buf,
    const geom_t *const *const RSTR points,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const geom_t *const RSTR x = points[0];
  const geom_t *const RSTR y = points[1];
  const geom_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      (void)n_shared_nodes;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const ptrdiff_t ghost_off = ghost_ptr[pack];
      const geom_t *const coordinate_components[ND] = {x, y, z};
      const s_t *const u_components[NC] = {ux, uy, uz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        const geom_t *const RSTR coordinate_component = coordinate_components[d];
        const s_t *const RSTR u_component = u_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_coordinate[k] = s_t(coordinate_component[node]);
          pk_u_component[k] = u_component[node * u_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t badj4[NQ * VS];
        s_t badj5[NQ * VS];
        s_t badj6[NQ * VS];
        s_t badj7[NQ * VS];
        s_t badj8[NQ * VS];
        s_t bdet0[NQ * VS];
        s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bu_streams[stream] = bu_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
          bout_streams[stream] = bout_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < ND; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bcoordinate_data[shape * ND + d][lane] = pk_coordinates[d * max_nodes_per_pack + packed_node];
            }
          }
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }

  s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
    const idx_t dest = ghost_reduce_dest[row];
    const ptrdiff_t begin = ghost_reduce_ptr[row];
    const ptrdiff_t end = ghost_reduce_ptr[row + 1];
    for (int d = 0; d < NC; ++d) {
      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
      s_t sum = s_t(0);
      for (ptrdiff_t j = begin; j < end; ++j) {
        sum += ghost_component[ghost_reduce_idx[j]];
      }
      out_components[d][dest * out_stride] += sum;
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_i_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    double *const RSTR ghost_buf,
    const geom_t *const *const RSTR points,
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_i_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_i_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    float *const RSTR ghost_buf,
    const geom_t *const *const RSTR points,
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_i_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa",
  "PROTEUS_HEX8",
  3,
  8,
  8,
  16,
  2,
  233,
  332,
  0,
  0,
  18,
  0,
  0,
  0,
  6,
  143,
  583,
  2456,
  3552,
  134,
  78,
  10,
  8,
  2,
  2,
  24,
  24,
  24,
  24,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data;
}

extern "C" double mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa_float",
      &sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_impl(
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
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bh_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};

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
    s_t badj0_data[VS];
    const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj0 + evb, badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data[VS];
    const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj1 + evb, badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data[VS];
    const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj2 + evb, badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data[VS];
    const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj3 + evb, badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data[VS];
    const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj4 + evb, badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data[VS];
    const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj5 + evb, badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data[VS];
    const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj6 + evb, badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data[VS];
    const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj7 + evb, badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data[VS];
    const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj8 + evb, badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data[VS];
    const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
        ne, g_det0 + evb, bdet0_data, std::is_same<g_t, s_t>());

    mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_a_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_shared = n_shared_nodes[pack];
      const ptrdiff_t n_not_shared = n_contiguous - n_shared;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const s_t *const u_components[NC] = {ux, uy, uz};
      const s_t *const h_components[NC] = {hx, hy, hz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        const s_t *const RSTR u_component = u_components[d];
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_u_component[k] = u_component[node * u_stride];
          pk_h_component[k] = h_component[node * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
          pk_h_component[n_contiguous + k] = h_component[node * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
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

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj0 + evb, badj0_data, std::is_same<geom_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj1 + evb, badj1_data, std::is_same<geom_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj2 + evb, badj2_data, std::is_same<geom_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj3 + evb, badj3_data, std::is_same<geom_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj4 + evb, badj4_data, std::is_same<geom_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj5 + evb, badj5_data, std::is_same<geom_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj6 + evb, badj6_data, std::is_same<geom_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj7 + evb, badj7_data, std::is_same<geom_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj8 + evb, badj8_data, std::is_same<geom_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_det0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bh_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
          global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_a_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_a_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
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
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_a_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    s_t *const RSTR ghost_buf,
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
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_affine_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      (void)n_shared_nodes;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const ptrdiff_t ghost_off = ghost_ptr[pack];
      const s_t *const u_components[NC] = {ux, uy, uz};
      const s_t *const h_components[NC] = {hx, hy, hz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        const s_t *const RSTR u_component = u_components[d];
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_u_component[k] = u_component[node * u_stride];
          pk_h_component[k] = h_component[node * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
          pk_h_component[n_contiguous + k] = h_component[node * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
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

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

        s_t badj0_data[VS];
        const s_t *const badj0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj0 + evb, badj0_data, std::is_same<geom_t, s_t>());
        s_t badj1_data[VS];
        const s_t *const badj1 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj1 + evb, badj1_data, std::is_same<geom_t, s_t>());
        s_t badj2_data[VS];
        const s_t *const badj2 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj2 + evb, badj2_data, std::is_same<geom_t, s_t>());
        s_t badj3_data[VS];
        const s_t *const badj3 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj3 + evb, badj3_data, std::is_same<geom_t, s_t>());
        s_t badj4_data[VS];
        const s_t *const badj4 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj4 + evb, badj4_data, std::is_same<geom_t, s_t>());
        s_t badj5_data[VS];
        const s_t *const badj5 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj5 + evb, badj5_data, std::is_same<geom_t, s_t>());
        s_t badj6_data[VS];
        const s_t *const badj6 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj6 + evb, badj6_data, std::is_same<geom_t, s_t>());
        s_t badj7_data[VS];
        const s_t *const badj7 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj7 + evb, badj7_data, std::is_same<geom_t, s_t>());
        s_t badj8_data[VS];
        const s_t *const badj8 = ageom_stream<s_t, geom_t, VS>(
            ne, g_adj8 + evb, badj8_data, std::is_same<geom_t, s_t>());
        s_t bdet0_data[VS];
        const s_t *const bdet0 = ageom_stream<s_t, geom_t, VS>(
            ne, g_det0 + evb, bdet0_data, std::is_same<geom_t, s_t>());

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, lmbda, mu, bu_streams, bh_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }

  s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
    const idx_t dest = ghost_reduce_dest[row];
    const ptrdiff_t begin = ghost_reduce_ptr[row];
    const ptrdiff_t end = ghost_reduce_ptr[row + 1];
    for (int d = 0; d < NC; ++d) {
      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
      s_t sum = s_t(0);
      for (ptrdiff_t j = begin; j < end; ++j) {
        sum += ghost_component[ghost_reduce_idx[j]];
      }
      out_components[d][dest * out_stride] += sum;
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_a_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    double *const RSTR ghost_buf,
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
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_a_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    float *const RSTR ghost_buf,
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
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
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
    s_t badj4[NQ * VS];
    s_t badj5[NQ * VS];
    s_t badj6[NQ * VS];
    s_t badj7[NQ * VS];
    s_t badj8[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        ev[element_node * VS + lane] = element_shape[evb + lane];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y, z};

    for (int shape = 0; shape < NS; ++shape) {
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};

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
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_i_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const geom_t *const RSTR x = points[0];
  const geom_t *const RSTR y = points[1];
  const geom_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      const ptrdiff_t n_shared = n_shared_nodes[pack];
      const ptrdiff_t n_not_shared = n_contiguous - n_shared;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const geom_t *const coordinate_components[ND] = {x, y, z};
      const s_t *const u_components[NC] = {ux, uy, uz};
      const s_t *const h_components[NC] = {hx, hy, hz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        const geom_t *const RSTR coordinate_component = coordinate_components[d];
        const s_t *const RSTR u_component = u_components[d];
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_coordinate[k] = s_t(coordinate_component[node]);
          pk_u_component[k] = u_component[node * u_stride];
          pk_h_component[k] = h_component[node * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
          pk_h_component[n_contiguous + k] = h_component[node * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t badj4[NQ * VS];
        s_t badj5[NQ * VS];
        s_t badj6[NQ * VS];
        s_t badj7[NQ * VS];
        s_t badj8[NQ * VS];
        s_t bdet0[NQ * VS];
        s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
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

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < ND; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bcoordinate_data[shape * ND + d][lane] = pk_coordinates[d * max_nodes_per_pack + packed_node];
            }
          }
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bh_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
          global_out[ghosts[k] * out_stride] += pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_i_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_i_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_i_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const geom_t *const *const RSTR points,
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_i_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_i_msoa_impl(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    s_t *const RSTR ghost_buf,
    const geom_t *const *const RSTR points,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
    const s_t *const RSTR uz,
    const ptrdiff_t h_stride,
    const s_t *const RSTR hx,
    const s_t *const RSTR hy,
    const s_t *const RSTR hz,
    const ptrdiff_t out_stride,
    s_t *const RSTR outx,
    s_t *const RSTR outy,
    s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int VS = 16;
  (void)nnodes;

  const geom_t *const RSTR x = points[0];
  const geom_t *const RSTR y = points[1];
  const geom_t *const RSTR z = points[2];
  const s_t *const isoparametric_shape_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_isoparametric_reference_data<s_t>::q_weight_1d();
  static constexpr int NQ1 = 2;
  static constexpr int NS1 = 2;

#pragma omp parallel
  {
    s_t *const RSTR pk_coordinates = sfem::codegen::thread_scratch<s_t>(0, (size_t)ND * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_u = sfem::codegen::thread_scratch<s_t>(1, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_h = sfem::codegen::thread_scratch<s_t>(2, (size_t)NC * (size_t)max_nodes_per_pack);
    s_t *const RSTR pk_out = sfem::codegen::thread_scratch<s_t>(3, (size_t)NC * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
      const ptrdiff_t e_start = pack * n_elements_per_pack;
      const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
      const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
      (void)n_shared_nodes;
      const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
      const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
      const idx_t *const RSTR ghosts = &ghost_idx[ghost_ptr[pack]];
      const ptrdiff_t ghost_off = ghost_ptr[pack];
      const geom_t *const coordinate_components[ND] = {x, y, z};
      const s_t *const u_components[NC] = {ux, uy, uz};
      const s_t *const h_components[NC] = {hx, hy, hz};
      s_t *const out_components[NC] = {outx, outy, outz};
      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR pk_coordinate = pk_coordinates + d * max_nodes_per_pack;
        s_t *const RSTR pk_u_component = pk_u + d * max_nodes_per_pack;
        s_t *const RSTR pk_h_component = pk_h + d * max_nodes_per_pack;
        const geom_t *const RSTR coordinate_component = coordinate_components[d];
        const s_t *const RSTR u_component = u_components[d];
        const s_t *const RSTR h_component = h_components[d];
        for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          const idx_t node = owned_nodes_ptr[pack] + k;
          pk_coordinate[k] = s_t(coordinate_component[node]);
          pk_u_component[k] = u_component[node * u_stride];
          pk_h_component[k] = h_component[node * h_stride];
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          const idx_t node = ghosts[k];
          pk_coordinate[n_contiguous + k] = s_t(coordinate_component[node]);
          pk_u_component[n_contiguous + k] = u_component[node * u_stride];
          pk_h_component[n_contiguous + k] = h_component[node * h_stride];
        }
      }

      for (ptrdiff_t evb = e_start; evb < e_end; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, e_end - evb);
        s_t bu_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t badj4[NQ * VS];
        s_t badj5[NQ * VS];
        s_t badj6[NQ * VS];
        s_t badj7[NQ * VS];
        s_t badj8[NQ * VS];
        s_t bdet0[NQ * VS];
        s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
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

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < ND; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bcoordinate_data[shape * ND + d][lane] = pk_coordinates[d * max_nodes_per_pack + packed_node];
            }
          }
          for (int d = 0; d < NC; ++d) {
#pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
              const uint16_t packed_node = element_shape[evb + lane];
              bu_data[shape * NC + d][lane] = pk_u[d * max_nodes_per_pack + packed_node];
              bh_data[shape * NC + d][lane] = pk_h[d * max_nodes_per_pack + packed_node];
              bout_data[shape * NC + d][lane] = s_t(0);
            }
          }
        }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + 1 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bh_streams, bout_streams);

        for (int shape = 0; shape < NS; ++shape) {
          const uint16_t *const RSTR element_shape = elements[shape];
          for (int d = 0; d < NC; ++d) {
            s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
            for (int lane = 0; lane < ne; ++lane) {
              pk_component_out[element_shape[evb + lane]] += bout_data[shape * NC + d][lane];
            }
          }
        }
      }

      for (int d = 0; d < NC; ++d) {
        s_t *const RSTR pk_component_out = pk_out + d * max_nodes_per_pack;
        s_t *const RSTR global_out = out_components[d];
        s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
        for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
          global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pk_component_out[k];
          pk_component_out[k] = s_t(0);
        }
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
          ghost_component[ghost_off + k] = pk_component_out[n_contiguous + k];
          pk_component_out[n_contiguous + k] = s_t(0);
        }
      }
    }
  }

  s_t *const out_components[NC] = {outx, outy, outz};
#pragma omp parallel for schedule(static)
  for (ptrdiff_t row = 0; row < n_ghost_reduce_rows; ++row) {
    const idx_t dest = ghost_reduce_dest[row];
    const ptrdiff_t begin = ghost_reduce_ptr[row];
    const ptrdiff_t end = ghost_reduce_ptr[row + 1];
    for (int d = 0; d < NC; ++d) {
      const s_t *const RSTR ghost_component = ghost_buf + d * n_ghost_entries;
      s_t sum = s_t(0);
      for (ptrdiff_t j = begin; j < end; ++j) {
        sum += ghost_component[ghost_reduce_idx[j]];
      }
      out_components[d][dest * out_stride] += sum;
    }
  }
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_i_msoa(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    double *const RSTR ghost_buf,
    const geom_t *const *const RSTR points,
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_i_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_i_msoa_float(
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t *const RSTR n_shared_nodes,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    float *const RSTR ghost_buf,
    const geom_t *const *const RSTR points,
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_packed_two_pass_i_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

} // namespace codegen
} // namespace sfem
