#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#endif
#endif
#include "../../../kernel_math.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
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

static constexpr int pm_orientation[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};

template <typename s_t, typename g_t, int NQ, int NS, int VS>
static int neohookean_ogden_total_tet4_merit_patch(
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const g_t *const *const RSTR points,
    const s_t lmbda,
    const s_t mu,
    const int nsteps,
    const s_t *const RSTR steps,
    const s_t *const RSTR x,
    const s_t *const RSTR h,
    const s_t *const RSTR accumulator,
    s_t *const RSTR merit
) {
  static constexpr int ND = 3;
  static constexpr int NC = 3;

#pragma omp parallel
  {
    // Per thread, and never larger than the vector width: the
    // caller rounds the step count to it, which is the whole
    // reason the steps carry the lanes.
    s_t merit_local[VS];
    for (int lane = 0; lane < VS; ++lane) merit_local[lane] = s_t(0);
    s_t rho[NC * VS];
    s_t pm_test_grad[3 * VS];
    s_t pm_weight[1 * VS];
    s_t pm_state_grad[9 * VS];
    s_t pm_direction_grad[9 * VS];
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          pm_incident[lane] = n2e_idx[block + lane];
          pm_local_node[lane] = n2e_local[block + lane];
        }
        // loop 1 -- lanes are the elements incident on this node.
        {  // TET4 evaluates in closed form
          #pragma omp simd
          for (int lane = 0; lane < ne; ++lane) {
            const idx_t element = pm_incident[lane];
            const int *const RSTR perm = pm_orientation[pm_local_node[lane]];
            s_t state[12];
            s_t direction[12];
            for (int j = 0; j < NS; ++j) {
              const idx_t node = elements[perm[j]][element];
              for (int c = 0; c < NC; ++c) {
                state[j * NC + c] = x[node * NC + c];
                direction[j * NC + c] = h[node * NC + c];
              }
            }
            // The Jacobian of the *permuted* element: its columns are the
            // edges from the visited node, which the permutation put at
            // slot 0.  Constant over the cell, because the orientation
            // gate admits only affine simplices.
            s_t jac[ND * ND];
            for (int d = 0; d < ND; ++d) {
              const s_t origin = (s_t)points[d][elements[perm[0]][element]];
              for (int k = 0; k < ND; ++k) {
                jac[d * ND + k] =
                    (s_t)points[d][elements[perm[k + 1]][element]] - origin;
              }
            }
            s_t adj[9];
            adj[0] = jac[4] * jac[8] - jac[5] * jac[7];
            adj[1] = jac[2] * jac[7] - jac[1] * jac[8];
            adj[2] = jac[1] * jac[5] - jac[2] * jac[4];
            adj[3] = jac[5] * jac[6] - jac[3] * jac[8];
            adj[4] = jac[0] * jac[8] - jac[2] * jac[6];
            adj[5] = jac[2] * jac[3] - jac[0] * jac[5];
            adj[6] = jac[3] * jac[7] - jac[4] * jac[6];
            adj[7] = jac[1] * jac[6] - jac[0] * jac[7];
            adj[8] = jac[0] * jac[4] - jac[1] * jac[3];
            const s_t det = jac[0] * adj[0] + jac[1] * adj[3] + jac[2] * adj[6];
            // physical gradient of the state: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                const s_t mapped = (-state[0 * NC + c] + state[1 * NC + c]) * adj[0 * ND + d] + (-state[0 * NC + c] + state[2 * NC + c]) * adj[1 * ND + d] + (-state[0 * NC + c] + state[3 * NC + c]) * adj[2 * ND + d];
                pm_state_grad[(c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // physical gradient of the direction: summed over shape functions,
            // then mapped.  Mapped here and not in loop 2 because the map
            // is linear and does not depend on the step length.
            for (int c = 0; c < NC; ++c) {
              for (int d = 0; d < ND; ++d) {
                const s_t mapped = (-direction[0 * NC + c] + direction[1 * NC + c]) * adj[0 * ND + d] + (-direction[0 * NC + c] + direction[2 * NC + c]) * adj[1 * ND + d] + (-direction[0 * NC + c] + direction[3 * NC + c]) * adj[2 * ND + d];
                pm_direction_grad[(c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // the fixed basis function's quantities, and the
            // integration weight.  Both are what the orientation buys:
            // `phi_0` is the same function in every element and at
            // every step, so this leaves the step loop entirely.
            for (int d = 0; d < ND; ++d) {
              const s_t mapped = s_t(-1) * adj[0 * ND + d] + s_t(-1) * adj[1 * ND + d] + s_t(-1) * adj[2 * ND + d];
              pm_test_grad[(d) * VS + lane] = mapped / det;
            }
            pm_weight[lane] = (s_t(1) / s_t(6)) * det;
          }
        }
        // loop 2 -- lanes are the sampled step lengths.
        for (int lane_e = 0; lane_e < ne; ++lane_e) {
          {  // TET4 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nsteps; ++lane) {
              const s_t alpha = steps[lane];
              const s_t u0_grad_0 = pm_state_grad[(0) * VS + lane_e] + alpha * pm_direction_grad[(0) * VS + lane_e];
              const s_t u0_grad_1 = pm_state_grad[(1) * VS + lane_e] + alpha * pm_direction_grad[(1) * VS + lane_e];
              const s_t u0_grad_2 = pm_state_grad[(2) * VS + lane_e] + alpha * pm_direction_grad[(2) * VS + lane_e];
              const s_t u1_grad_0 = pm_state_grad[(3) * VS + lane_e] + alpha * pm_direction_grad[(3) * VS + lane_e];
              const s_t u1_grad_1 = pm_state_grad[(4) * VS + lane_e] + alpha * pm_direction_grad[(4) * VS + lane_e];
              const s_t u1_grad_2 = pm_state_grad[(5) * VS + lane_e] + alpha * pm_direction_grad[(5) * VS + lane_e];
              const s_t u2_grad_0 = pm_state_grad[(6) * VS + lane_e] + alpha * pm_direction_grad[(6) * VS + lane_e];
              const s_t u2_grad_1 = pm_state_grad[(7) * VS + lane_e] + alpha * pm_direction_grad[(7) * VS + lane_e];
              const s_t u2_grad_2 = pm_state_grad[(8) * VS + lane_e] + alpha * pm_direction_grad[(8) * VS + lane_e];
              const s_t residual_tmp0 = u0_grad_0 + s_t(1);
              const s_t residual_tmp1 = u1_grad_2*u2_grad_1;
              const s_t residual_tmp2 = u1_grad_1 + s_t(1);
              const s_t residual_tmp3 = u2_grad_2 + s_t(1);
              const s_t residual_tmp4 = -residual_tmp1 + residual_tmp2*residual_tmp3;
              const s_t residual_tmp5 = residual_tmp3*u1_grad_0;
              const s_t residual_tmp6 = residual_tmp2*u2_grad_0;
              const s_t residual_tmp7 = -residual_tmp0*residual_tmp1 + residual_tmp0*residual_tmp2*residual_tmp3 - residual_tmp5*u0_grad_1 - residual_tmp6*u0_grad_2 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1;
              const s_t residual_tmp8 = pow_m1(residual_tmp7);
              const s_t residual_tmp9 = mu*residual_tmp8;
              const s_t residual_tmp10 = lmbda*residual_tmp8*log(residual_tmp7);
              const s_t residual_tmp11 = -residual_tmp5 + u1_grad_2*u2_grad_0;
              const s_t residual_tmp12 = -residual_tmp6 + u1_grad_0*u2_grad_1;
              const s_t residual_tmp13 = -residual_tmp3*u0_grad_1 + u0_grad_2*u2_grad_1;
              const s_t residual_tmp14 = residual_tmp0*residual_tmp3 - u0_grad_2*u2_grad_0;
              const s_t residual_tmp15 = -residual_tmp0*u2_grad_1 + u0_grad_1*u2_grad_0;
              const s_t residual_tmp16 = -residual_tmp2*u0_grad_2 + u0_grad_1*u1_grad_2;
              const s_t residual_tmp17 = -residual_tmp0*u1_grad_2 + u0_grad_2*u1_grad_0;
              const s_t residual_tmp18 = residual_tmp0*residual_tmp2 - u0_grad_1*u1_grad_0;
              const s_t grad_coeff0_0 = mu*residual_tmp0 + residual_tmp10*residual_tmp4 - residual_tmp4*residual_tmp9;
              const s_t grad_coeff0_1 = mu*u0_grad_1 + residual_tmp10*residual_tmp11 - residual_tmp11*residual_tmp9;
              const s_t grad_coeff0_2 = mu*u0_grad_2 + residual_tmp10*residual_tmp12 - residual_tmp12*residual_tmp9;
              const s_t grad_coeff1_0 = mu*u1_grad_0 + residual_tmp10*residual_tmp13 - residual_tmp13*residual_tmp9;
              const s_t grad_coeff1_1 = mu*residual_tmp2 + residual_tmp10*residual_tmp14 - residual_tmp14*residual_tmp9;
              const s_t grad_coeff1_2 = mu*u1_grad_2 + residual_tmp10*residual_tmp15 - residual_tmp15*residual_tmp9;
              const s_t grad_coeff2_0 = mu*u2_grad_0 + residual_tmp10*residual_tmp16 - residual_tmp16*residual_tmp9;
              const s_t grad_coeff2_1 = mu*u2_grad_1 + residual_tmp10*residual_tmp17 - residual_tmp17*residual_tmp9;
              const s_t grad_coeff2_2 = mu*residual_tmp3 + residual_tmp10*residual_tmp18 - residual_tmp18*residual_tmp9;
              const s_t weight = pm_weight[lane_e];
              rho[0 * VS + lane] += weight * (grad_coeff0_0 * pm_test_grad[(0) * VS + lane_e] + grad_coeff0_1 * pm_test_grad[(1) * VS + lane_e] + grad_coeff0_2 * pm_test_grad[(2) * VS + lane_e]);
              rho[1 * VS + lane] += weight * (grad_coeff1_0 * pm_test_grad[(0) * VS + lane_e] + grad_coeff1_1 * pm_test_grad[(1) * VS + lane_e] + grad_coeff1_2 * pm_test_grad[(2) * VS + lane_e]);
              rho[2 * VS + lane] += weight * (grad_coeff2_0 * pm_test_grad[(0) * VS + lane_e] + grad_coeff2_1 * pm_test_grad[(1) * VS + lane_e] + grad_coeff2_2 * pm_test_grad[(2) * VS + lane_e]);
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
        squared += rho[2 * VS + lane] * rho[2 * VS + lane];
        merit_local[lane] += s_t(0.5) * squared;
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

} // namespace codegen
} // namespace sfem


extern "C" int neohookean_ogden_total_tet4_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t lmbda,
    const real_t mu,
    const int nsteps,
    const void *const RSTR steps,
    const void *const RSTR x,
    const void *const RSTR h,
    const void *const RSTR accumulator,
    void *const RSTR merit
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::neohookean_ogden_total_tet4_merit_patch<double, geom_t, 1, 4, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, lmbda, mu, nsteps, (const double *)steps, (const double *)x, (const double *)h, (const double *)accumulator, (double *)merit);
    }
    case (int)sizeof(float): {
        return sfem::codegen::neohookean_ogden_total_tet4_merit_patch<float, geom_t, 1, 4, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, lmbda, mu, nsteps, (const float *)steps, (const float *)x, (const float *)h, (const float *)accumulator, (float *)merit);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_total_tet4_merit_patch_a_msoa", -1, (int)scalar_bytes);
}
