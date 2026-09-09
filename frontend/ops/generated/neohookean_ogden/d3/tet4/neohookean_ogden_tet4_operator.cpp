#include <cstdio>
#include <type_traits>
#include "../neohookean_ogden_d3_simplex_local.hpp"
#include "../../../reference/quad_tet_q1.hpp"
#include "../../../reference/tet4_q1.hpp"
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
struct neohookean_ogden_tet4_affine_reference_data {
  static const s_t *shape() { return ref_tet4_q1<s_t>::shape(); }
  static const s_t *grad_ref_x() { return ref_tet4_q1<s_t>::grad_ref_x(); }
  static const s_t *grad_ref_y() { return ref_tet4_q1<s_t>::grad_ref_y(); }
  static const s_t *grad_ref_z() { return ref_tet4_q1<s_t>::grad_ref_z(); }
  static const s_t *q_weight() { return quad_tet_q1<s_t>::q_weight(); }
};

template <typename s_t>
struct neohookean_ogden_tet4_isoparametric_reference_data {
  static const s_t *shape() { return ref_tet4_q1<s_t>::shape(); }
  static const s_t *grad_ref_x() { return ref_tet4_q1<s_t>::grad_ref_x(); }
  static const s_t *grad_ref_y() { return ref_tet4_q1<s_t>::grad_ref_y(); }
  static const s_t *grad_ref_z() { return ref_tet4_q1<s_t>::grad_ref_z(); }
  static const s_t *q_weight() { return quad_tet_q1<s_t>::q_weight(); }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tet4_objective_soa_diagnostics_data = {
  "neohookean_ogden_tet4_objective_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  19,
  21,
  0,
  0,
  10,
  0,
  1,
  0,
  6,
  5,
  70,
  0,
  0,
  4,
  12,
  10,
  12,
  1,
  2,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet4_objective_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_tet4_objective_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_tet4_objective_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "neohookean_ogden_tet4_objective_soa",
      &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_objective_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "neohookean_ogden_tet4_objective_soa_float",
      &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tet4_objective_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "neohookean_ogden_tet4_objective_a_msoa",
      &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_objective_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "neohookean_ogden_tet4_objective_a_msoa_float",
      &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tet4_objective_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "neohookean_ogden_tet4_objective_i_msoa",
      &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_objective_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "neohookean_ogden_tet4_objective_i_msoa_float",
      &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int neohookean_ogden_tet4_objective_steps_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

      neohookean_ogden_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bvalue);

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

extern "C" int neohookean_ogden_tet4_objective_steps_a_msoa(
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
  return sfem::codegen::neohookean_ogden_tet4_objective_steps_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_tet4_objective_steps_a_msoa_float(
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
  return sfem::codegen::neohookean_ogden_tet4_objective_steps_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int neohookean_ogden_tet4_objective_steps_packed_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;
  (void)n_shared_nodes;

  const s_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

        const s_t *bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[3], bu_data[4], bu_data[5], bu_data[6], bu_data[7], bu_data[8], bu_data[9], bu_data[10], bu_data[11]};

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

          neohookean_ogden_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bvalue);

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

extern "C" int neohookean_ogden_tet4_objective_steps_packed_a_msoa(
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
  return neohookean_ogden_tet4_objective_steps_packed_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int neohookean_ogden_tet4_objective_steps_packed_a_msoa_float(
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
  return neohookean_ogden_tet4_objective_steps_packed_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tet4_gradient_soa_diagnostics_data = {
  "neohookean_ogden_tet4_gradient_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  35,
  76,
  1,
  0,
  0,
  0,
  1,
  0,
  6,
  31,
  139,
  0,
  0,
  22,
  16,
  10,
  12,
  1,
  2,
  12,
  0,
  12,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet4_gradient_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_tet4_gradient_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_tet4_gradient_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "neohookean_ogden_tet4_gradient_soa",
      &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_gradient_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "neohookean_ogden_tet4_gradient_soa_float",
      &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tet4_gradient_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "neohookean_ogden_tet4_gradient_a_msoa",
      &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_gradient_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "neohookean_ogden_tet4_gradient_a_msoa_float",
      &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tet4_gradient_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "neohookean_ogden_tet4_gradient_i_msoa",
      &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_gradient_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "neohookean_ogden_tet4_gradient_i_msoa_float",
      &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int neohookean_ogden_tet4_gradient_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

    neohookean_ogden_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

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

extern "C" int neohookean_ogden_tet4_gradient_a_msoa(
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
  return sfem::codegen::neohookean_ogden_tet4_gradient_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_tet4_gradient_a_msoa_float(
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
  return sfem::codegen::neohookean_ogden_tet4_gradient_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int neohookean_ogden_tet4_gradient_packed_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

        neohookean_ogden_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

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

extern "C" int neohookean_ogden_tet4_gradient_packed_a_msoa(
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
  return neohookean_ogden_tet4_gradient_packed_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_tet4_gradient_packed_a_msoa_float(
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
  return neohookean_ogden_tet4_gradient_packed_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

template <typename s_t>
static SFEM_INLINE int neohookean_ogden_tet4_gradient_packed_two_pass_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

        neohookean_ogden_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

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

extern "C" int neohookean_ogden_tet4_gradient_packed_two_pass_a_msoa(
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
  return neohookean_ogden_tet4_gradient_packed_two_pass_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_tet4_gradient_packed_two_pass_a_msoa_float(
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
  return neohookean_ogden_tet4_gradient_packed_two_pass_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tet4_apply_soa_diagnostics_data = {
  "neohookean_ogden_tet4_apply_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  190,
  387,
  1,
  0,
  10,
  0,
  1,
  0,
  6,
  129,
  615,
  0,
  0,
  120,
  69,
  10,
  12,
  1,
  2,
  12,
  12,
  12,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet4_apply_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_tet4_apply_soa_arithmetic_intensity(
    const ptrdiff_t nelements,
    const size_t scalar_bytes,
    const size_t real_bytes,
    const size_t accumulator_bytes) {
  return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_tet4_apply_soa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "neohookean_ogden_tet4_apply_soa",
      &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_apply_soa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate(
      "neohookean_ogden_tet4_apply_soa_float",
      &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tet4_apply_a_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "neohookean_ogden_tet4_apply_a_msoa",
      &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_apply_a_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
      "neohookean_ogden_tet4_apply_a_msoa_float",
      &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_tet4_apply_i_msoa_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "neohookean_ogden_tet4_apply_i_msoa",
      &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_tet4_apply_i_msoa_float_print_rate(
    const double elapsed,
    const ptrdiff_t nelements,
    const ptrdiff_t ndofs) {
  sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
      "neohookean_ogden_tet4_apply_i_msoa_float",
      &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data,
      elapsed, nelements, ndofs,
      sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int neohookean_ogden_tet4_apply_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

    neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

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

extern "C" int neohookean_ogden_tet4_apply_a_msoa(
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
  return sfem::codegen::neohookean_ogden_tet4_apply_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_tet4_apply_a_msoa_float(
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
  return sfem::codegen::neohookean_ogden_tet4_apply_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int neohookean_ogden_tet4_apply_packed_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

        neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

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

extern "C" int neohookean_ogden_tet4_apply_packed_a_msoa(
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
  return neohookean_ogden_tet4_apply_packed_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_tet4_apply_packed_a_msoa_float(
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
  return neohookean_ogden_tet4_apply_packed_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

template <typename s_t>
static SFEM_INLINE int neohookean_ogden_tet4_apply_packed_two_pass_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 16;
  (void)nnodes;

  const s_t *const affine_grad_ref_x = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::neohookean_ogden_tet4_affine_reference_data<s_t>::q_weight();

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

        neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

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

extern "C" int neohookean_ogden_tet4_apply_packed_two_pass_a_msoa(
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
  return neohookean_ogden_tet4_apply_packed_two_pass_a_msoa_impl<double>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_tet4_apply_packed_two_pass_a_msoa_float(
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
  return neohookean_ogden_tet4_apply_packed_two_pass_a_msoa_impl<float>(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static SFEM_INLINE void neohookean_ogden_tet4_hessian_i_msoa_find_cols(
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
static SFEM_INLINE void neohookean_ogden_tet4_hessian_i_msoa_scatter_bsr(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NC = 3;
  static constexpr int NS = 4;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const idx_t dof_i = ev[i];
    const count_t row_begin = rowptr[dof_i];
    const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    neohookean_ogden_tet4_hessian_i_msoa_find_cols(ev, cols, lenrow, ks);
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
static int neohookean_ogden_tet4_hessian_i_msoa_assemble_impl(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values,
    const int *const RSTR diag_offsets,
    const ptrdiff_t ndiag,
    const ptrdiff_t coo_nnz,
    const idx_t *const RSTR coo_rows,
    const idx_t *const RSTR coo_cols,
    idx_t *const RSTR coo_triplet_rows,
    idx_t *const RSTR coo_triplet_cols) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int VS = 1;
  static constexpr int NDOFS = NC * NS;
  (void)nnodes;
  const s_t *const u_components[NC] = {ux, uy, uz};
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::neohookean_ogden_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::neohookean_ogden_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::neohookean_ogden_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::neohookean_ogden_tet4_isoparametric_reference_data<s_t>::q_weight();

  int unsupported_matrix_format = 0;
#pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format)
  for (ptrdiff_t element = 0; element < nelements; ++element) {
    idx_t ev[NS];
    s_t element_matrix[NDOFS * NDOFS];
    s_t bh_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    static constexpr int ne = VS;
    s_t bu_data[NS * NC][VS];
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
      const idx_t node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinate_data[shape * ND + d][0] = s_t(points[d][node]);
        bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
      }
    }


    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J02_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      s_t J12_values[VS];
      s_t J20_values[VS];
      s_t J21_values[VS];
      s_t J22_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J01_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J02_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J10_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J11_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J12_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J20_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J21_values[lane] = s_t(0);
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J22_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        const s_t g2 = isoparametric_grad_ref_z[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J02_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J12_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J20_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J21_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J22_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g2;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J02 = J02_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        const s_t J12 = J12_values[lane];
        const s_t J20 = J20_values[lane];
        const s_t J21 = J21_values[lane];
        const s_t J22 = J22_values[lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + lane);
      }
    }

    for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
      element_matrix[entry] = s_t(0);
    }

    for (int trial_component = 0; trial_component < NC; ++trial_component) {
      for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
        for (int stream = 0; stream < NS * NC; ++stream) {
          bh_data[stream][0] = s_t(0);
          bout_data[stream][0] = s_t(0);
        }
        bh_data[trial_shape * NC + trial_component][0] = s_t(1);
        neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(1, 1, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);
        const int col = trial_component * NS + trial_shape;
        for (int test_component = 0; test_component < NC; ++test_component) {
          for (int test_shape = 0; test_shape < NS; ++test_shape) {
            const int row = test_component * NS + test_shape;
            element_matrix[row * NDOFS + col] = bout_data[test_shape * NC + test_component][0];
          }
        }
      }
    }

    if constexpr (FORMAT == 1) {
      neohookean_ogden_tet4_hessian_i_msoa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
    } else {
      unsupported_matrix_format |= 1;
    }
  }

  return unsupported_matrix_format ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_tet4_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
  return sfem::codegen::neohookean_ogden_tet4_hessian_i_msoa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_tet4_hessian_bsr_i_msoa_float(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
  return sfem::codegen::neohookean_ogden_tet4_hessian_i_msoa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
