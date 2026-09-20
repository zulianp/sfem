#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_residual_merit_d2_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#include "../../../reference/quad_tri_q1.hpp"
#include "../../../reference/tri3_q1.hpp"
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
extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_tri3_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[4],
    const void *const RSTR current[6],
    const void *const RSTR previous[6],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[6]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d2_simplex_residual_block<double, 1, 3, 16>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tri3_q1<double>::grad_ref_x(), sfem::codegen::ref_tri3_q1<double>::grad_ref_y(), sfem::codegen::quad_tri_q1<double>::q_weight(), (const double *const *)current, (const double *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d2_simplex_residual_block<float, 1, 3, 16>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tri3_q1<float>::grad_ref_x(), sfem::codegen::ref_tri3_q1<float>::grad_ref_y(), sfem::codegen::quad_tri_q1<float>::q_weight(), (const float *const *)current, (const float *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_tri3_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_residual_merit_tri3_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const affine_shape = sfem::codegen::ref_tri3_q1<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
  const s_t *const affine_q_weight = sfem::codegen::quad_tri_q1<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1};
    const s_t *const previous_components[NC] = {u0_old, u1_old};

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

    for (int stream = 0; stream < 6; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[5] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_det0 + evb};
    s_t baffine_geometry_data[5][VS];
    const s_t *bageom_streams[5];
    for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[4];
    for (int component = 0; component < 4; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    mooney_rivlin_kelvin_voigt_residual_merit_d2_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[4], badjugate, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_tri3_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_tri3_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_tri3_residual_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_tri3_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[4],
    const void *const RSTR current[6],
    const void *const RSTR previous[6],
    const void *const RSTR direction[6],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[6]
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d2_simplex_jacobian_action_block<double, 1, 3, 16>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tri3_q1<double>::grad_ref_x(), sfem::codegen::ref_tri3_q1<double>::grad_ref_y(), sfem::codegen::quad_tri_q1<double>::q_weight(), (const double *const *)current, (const double *const *)previous, (const double *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d2_simplex_jacobian_action_block<float, 1, 3, 16>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tri3_q1<float>::grad_ref_x(), sfem::codegen::ref_tri3_q1<float>::grad_ref_y(), sfem::codegen::quad_tri_q1<float>::q_weight(), (const float *const *)current, (const float *const *)previous, (const float *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_tri3_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_residual_merit_tri3_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const affine_shape = sfem::codegen::ref_tri3_q1<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
  const s_t *const affine_q_weight = sfem::codegen::quad_tri_q1<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1};
    const s_t *const previous_components[NC] = {u0_old, u1_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction};

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

    for (int stream = 0; stream < 6; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[5] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_det0 + evb};
    s_t baffine_geometry_data[5][VS];
    const s_t *bageom_streams[5];
    for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[4];
    for (int component = 0; component < 4; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    mooney_rivlin_kelvin_voigt_residual_merit_d2_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[4], badjugate, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

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

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_tri3_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_tri3_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_tri3_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_tri3_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}


static constexpr int pm_orientation[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};

template <typename s_t, int NQ, int NS, int VS>
static int mooney_rivlin_kelvin_voigt_residual_merit_tri3_merit_patch(
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref[2],
    const s_t *const RSTR q_weight,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const int nsteps,
    const s_t *const RSTR steps,
    const s_t *const RSTR x,
    const s_t *const RSTR h,
    const s_t *const RSTR p,
    const s_t *const RSTR accumulator,
    s_t *const RSTR merit
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;

#pragma omp parallel
  {
    // Per thread, and never larger than the vector width: the
    // caller rounds the step count to it, which is the whole
    // reason the steps carry the lanes.
    s_t merit_local[VS];
    for (int lane = 0; lane < VS; ++lane) merit_local[lane] = s_t(0);
    s_t rho[NC * VS];
    s_t pm_test_grad[NQ * 2 * VS];
    s_t pm_weight[NQ * 1 * VS];
    s_t pm_state_grad[NQ * 4 * VS];
    s_t pm_direction_grad[NQ * 4 * VS];
    s_t pm_previous_grad[NQ * 4 * VS];
    element_idx_t pm_incident[VS];
    uint8_t pm_local_node[VS];

#pragma omp for schedule(static)
    for (ptrdiff_t node = 0; node < n_owned_nodes; ++node) {
      // Seed with everything that does not move with the state, so
      // the square below is over the whole residual.
      for (int c = 0; c < NC; ++c) {
        for (int lane = 0; lane < VS; ++lane) {
          rho[c * VS + lane] = accumulator[node * NC + c];
        }
      }
      const count_t begin = n2e_ptr[node];
      const count_t end = n2e_ptr[node + 1];
      for (count_t block = begin; block < end; block += VS) {
        const int ne = (int)MIN((count_t)VS, end - block);
        for (int lane = 0; lane < ne; ++lane) {
          pm_incident[lane] = n2e_idx[block + lane];
          pm_local_node[lane] = n2e_local[block + lane];
        }
        // loop 1 -- lanes are the elements incident on this node.
        for (int q = 0; q < NQ; ++q) {
          #pragma omp simd
          for (int lane = 0; lane < ne; ++lane) {
            const idx_t element = pm_incident[lane];
            const int *const RSTR perm = pm_orientation[pm_local_node[lane]];
            s_t state[6];
            s_t direction[6];
            s_t previous[6];
            for (int j = 0; j < NS; ++j) {
              const idx_t node = elements[perm[j]][element];
              for (int c = 0; c < NC; ++c) {
                state[j * NC + c] = x[node * NC + c];
                direction[j * NC + c] = h[node * NC + c];
                previous[j * NC + c] = p[node * NC + c];
              }
            }
            // The geometry is indexed by the element, not by a block
            // offset: a node's incident elements are scattered through
            // the mesh rather than contiguous.  It carries no quadrature
            // index either, because the orientation gate has already
            // restricted this kernel to affine simplices, whose Jacobian
            // is constant over the cell.
            const s_t det = determinant[element];
            // physical gradient of the state: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                s_t mapped = s_t(0);
                for (int k = 0; k < ND; ++k) {
                  s_t acc = s_t(0);
                  for (int j = 0; j < NS; ++j) {
                    acc += state[j * NC + c] * grad_ref[k][q * NS + j];
                  }
                  mapped += acc * adjugate[k * ND + d][element];
                }
                pm_state_grad[(q * 4 + c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // physical gradient of the direction: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                s_t mapped = s_t(0);
                for (int k = 0; k < ND; ++k) {
                  s_t acc = s_t(0);
                  for (int j = 0; j < NS; ++j) {
                    acc += direction[j * NC + c] * grad_ref[k][q * NS + j];
                  }
                  mapped += acc * adjugate[k * ND + d][element];
                }
                pm_direction_grad[(q * 4 + c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // physical gradient of the previous: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                s_t mapped = s_t(0);
                for (int k = 0; k < ND; ++k) {
                  s_t acc = s_t(0);
                  for (int j = 0; j < NS; ++j) {
                    acc += previous[j * NC + c] * grad_ref[k][q * NS + j];
                  }
                  mapped += acc * adjugate[k * ND + d][element];
                }
                pm_previous_grad[(q * 4 + c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // the fixed basis function's physical gradient, and the
            // integration weight.  Both are what the orientation buys:
            // `phi_0` is the same function in every lane and at every
            // step, so this leaves the step loop entirely.
            for (int d = 0; d < ND; ++d) {
              s_t mapped = s_t(0);
              for (int k = 0; k < ND; ++k) {
                mapped += grad_ref[k][q * NS + 0] * adjugate[k * ND + d][element];
              }
              pm_test_grad[(q * 2 + d) * VS + lane] = mapped / det;
            }
            pm_weight[q * VS + lane] = q_weight[q] * det;
          }
        }
        // loop 2 -- lanes are the sampled step lengths.
        for (int lane_e = 0; lane_e < ne; ++lane_e) {
          for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nsteps; ++lane) {
              const s_t alpha = steps[lane];
              const s_t u0_grad_0 = pm_state_grad[(q * 4 + 0) * VS + lane_e] + alpha * pm_direction_grad[(q * 4 + 0) * VS + lane_e];
              const s_t u0_grad_1 = pm_state_grad[(q * 4 + 1) * VS + lane_e] + alpha * pm_direction_grad[(q * 4 + 1) * VS + lane_e];
              const s_t u1_grad_0 = pm_state_grad[(q * 4 + 2) * VS + lane_e] + alpha * pm_direction_grad[(q * 4 + 2) * VS + lane_e];
              const s_t u1_grad_1 = pm_state_grad[(q * 4 + 3) * VS + lane_e] + alpha * pm_direction_grad[(q * 4 + 3) * VS + lane_e];
              const s_t u0_old_grad_0 = pm_previous_grad[(q * 4 + 0) * VS + lane_e];
              const s_t u0_old_grad_1 = pm_previous_grad[(q * 4 + 1) * VS + lane_e];
              const s_t u1_old_grad_0 = pm_previous_grad[(q * 4 + 2) * VS + lane_e];
              const s_t u1_old_grad_1 = pm_previous_grad[(q * 4 + 3) * VS + lane_e];
              const s_t residual_tmp0 = u1_grad_1 + s_t(1);
              const s_t residual_tmp1 = u0_grad_1*u1_grad_0;
              const s_t residual_tmp2 = u0_grad_0 + s_t(1);
              const s_t residual_tmp3 = lmbda*(residual_tmp0*residual_tmp2 - residual_tmp1 + s_t(-1));
              const s_t residual_tmp4 = residual_tmp0*u1_grad_0 + residual_tmp2*u0_grad_1;
              const s_t residual_tmp5 = s_t(2)*u0_grad_1;
              const s_t residual_tmp6 = pow_2(residual_tmp2) + pow_2(u1_grad_0);
              const s_t residual_tmp7 = s_t(2)*residual_tmp2;
              const s_t residual_tmp8 = pow_2(residual_tmp0) + pow_2(u0_grad_1);
              const s_t residual_tmp9 = residual_tmp6 + residual_tmp8;
              const s_t residual_tmp10 = pow_m1(-residual_tmp1 + residual_tmp2 + u0_grad_0*u1_grad_1 + u1_grad_1);
              const s_t residual_tmp11 = u0_grad_0*u_dt_shift + u0_old_grad_0;
              const s_t residual_tmp12 = u1_grad_1*u_dt_shift + u1_old_grad_1;
              const s_t residual_tmp13 = u0_grad_1*u_dt_shift + u0_old_grad_1;
              const s_t residual_tmp14 = u1_grad_0*u_dt_shift + u1_old_grad_0;
              const s_t residual_tmp15 = eta_s*(-residual_tmp0*residual_tmp14 + residual_tmp11*u0_grad_1 + residual_tmp12*u1_grad_0 - residual_tmp13*residual_tmp2);
              const s_t residual_tmp16 = residual_tmp13*u1_grad_0;
              const s_t residual_tmp17 = residual_tmp0*residual_tmp11;
              const s_t residual_tmp18 = -residual_tmp12*residual_tmp2 + residual_tmp14*u0_grad_1;
              const s_t residual_tmp19 = eta_b*(residual_tmp16 - residual_tmp17 + residual_tmp18);
              const s_t residual_tmp20 = -residual_tmp16 + residual_tmp17 + residual_tmp18;
              const s_t residual_tmp21 = -eta_s*residual_tmp20 + residual_tmp19;
              const s_t residual_tmp22 = s_t(2)*u1_grad_0;
              const s_t residual_tmp23 = s_t(2)*residual_tmp0;
              const s_t residual_tmp24 = eta_s*residual_tmp20 + residual_tmp19;
              const s_t grad_coeff0_0 = mu*(s_t(2)*residual_tmp2*residual_tmp9 - residual_tmp4*residual_tmp5 - residual_tmp6*residual_tmp7 + s_t(4)*u0_grad_0 - s_t(6)*u1_grad_1 + s_t(-2)) + residual_tmp0*residual_tmp3 + residual_tmp10*(-residual_tmp0*residual_tmp21 + residual_tmp15*u0_grad_1);
              const s_t grad_coeff0_1 = mu*(-residual_tmp4*residual_tmp7 - residual_tmp5*residual_tmp8 + residual_tmp5*residual_tmp9 + s_t(4)*u0_grad_1 + s_t(6)*u1_grad_0) + residual_tmp10*(-residual_tmp15*residual_tmp2 + residual_tmp21*u1_grad_0) - residual_tmp3*u1_grad_0;
              const s_t grad_coeff1_0 = mu*(-residual_tmp22*residual_tmp6 + residual_tmp22*residual_tmp9 - residual_tmp23*residual_tmp4 + s_t(6)*u0_grad_1 + s_t(4)*u1_grad_0) + residual_tmp10*(-residual_tmp0*residual_tmp15 + residual_tmp24*u0_grad_1) - residual_tmp3*u0_grad_1;
              const s_t grad_coeff1_1 = mu*(s_t(2)*residual_tmp0*residual_tmp9 - residual_tmp22*residual_tmp4 - residual_tmp23*residual_tmp8 - s_t(6)*u0_grad_0 + s_t(4)*u1_grad_1 + s_t(-2)) + residual_tmp10*(residual_tmp15*u1_grad_0 - residual_tmp2*residual_tmp24) + residual_tmp2*residual_tmp3;
              const s_t weight = pm_weight[q * VS + lane_e];
              rho[0 * VS + lane] += weight * (grad_coeff0_0 * pm_test_grad[(q * 2 + 0) * VS + lane_e] + grad_coeff0_1 * pm_test_grad[(q * 2 + 1) * VS + lane_e]);
              rho[1 * VS + lane] += weight * (grad_coeff1_0 * pm_test_grad[(q * 2 + 0) * VS + lane_e] + grad_coeff1_1 * pm_test_grad[(q * 2 + 1) * VS + lane_e]);
            }
          }
        }
      }

      // The node is finished, so it may be squared.
      #pragma omp simd
      for (int lane = 0; lane < nsteps; ++lane) {
        s_t squared = s_t(0);
        squared += rho[0 * VS + lane] * rho[0 * VS + lane];
        squared += rho[1 * VS + lane] * rho[1 * VS + lane];
        merit[lane] += s_t(0.5) * squared;
      }
    }

    // One reduction per thread, not one per node.
    for (int lane = 0; lane < nsteps; ++lane) {
#pragma omp atomic update
      merit[lane] += merit_local[lane];
    }
  }
  return SFEM_SUCCESS;
}
