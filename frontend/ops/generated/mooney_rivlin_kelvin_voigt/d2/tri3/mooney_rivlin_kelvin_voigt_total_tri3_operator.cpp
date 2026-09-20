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

namespace sfem {
namespace codegen {

static constexpr int pm_orientation[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};

template <typename s_t, typename g_t, int NQ, int NS, int VS>
static int mooney_rivlin_kelvin_voigt_total_tri3_merit_patch(
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const g_t *const *const RSTR points,
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
        {
            const int q = 0;  // TRI3 evaluates in closed form
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
            const s_t det = jac[0] * jac[3] - jac[1] * jac[2];
            s_t adj[4];
            adj[0] =  jac[3];
            adj[1] = -jac[1];
            adj[2] = -jac[2];
            adj[3] =  jac[0];
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
                  mapped += acc * adj[k * ND + d];
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
                  mapped += acc * adj[k * ND + d];
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
                  mapped += acc * adj[k * ND + d];
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
                mapped += grad_ref[k][q * NS + 0] * adj[k * ND + d];
              }
              pm_test_grad[(q * 2 + d) * VS + lane] = mapped / det;
            }
            pm_weight[q * VS + lane] = q_weight[q] * det;
          }
        }
        // loop 2 -- lanes are the sampled step lengths.
        for (int lane_e = 0; lane_e < ne; ++lane_e) {
          {
              const int q = 0;  // TRI3 evaluates in closed form
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


extern "C" int mooney_rivlin_kelvin_voigt_total_tri3_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR shape,
    const void *const RSTR grad_ref[2],
    const void *const RSTR q_weight,
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const int nsteps,
    const void *const RSTR steps,
    const void *const RSTR x,
    const void *const RSTR h,
    const void *const RSTR p,
    const void *const RSTR accumulator,
    void *const RSTR merit
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_merit_patch<double, geom_t, 1, 3, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, (const double *)shape, (const double *const *)grad_ref, (const double *)q_weight, eta_b, eta_s, lmbda, mu, u_dt_shift, nsteps, (const double *)steps, (const double *)x, (const double *)h, (const double *)p, (const double *)accumulator, (double *)merit);
    }
    case (int)sizeof(float): {
        return sfem::codegen::mooney_rivlin_kelvin_voigt_total_tri3_merit_patch<float, geom_t, 1, 3, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, (const float *)shape, (const float *const *)grad_ref, (const float *)q_weight, eta_b, eta_s, lmbda, mu, u_dt_shift, nsteps, (const float *)steps, (const float *)x, (const float *)h, (const float *)p, (const float *)accumulator, (float *)merit);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tri3_merit_patch_a_msoa", -1, (int)scalar_bytes);
}
