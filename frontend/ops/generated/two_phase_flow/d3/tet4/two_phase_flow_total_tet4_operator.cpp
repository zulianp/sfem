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
#include "../../../reference/quad_tet_q11.hpp"
#include "../../../reference/tet4_q11.hpp"
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

static constexpr int pm_orientation[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};

template <typename s_t, typename g_t, int NQ, int NS, int VS>
static int two_phase_flow_total_tet4_merit_patch(
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const g_t *const *const RSTR points,
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref[3],
    const s_t *const RSTR q_weight,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t mu_w,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t R,
    const s_t T,
    const s_t Z,
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t mu_c,
    const int nsteps,
    const s_t *const RSTR steps,
    const s_t *const RSTR x,
    const s_t *const RSTR h,
    const s_t *const RSTR p,
    const s_t *const RSTR accumulator,
    s_t *const RSTR merit
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;

#pragma omp parallel
  {
    // Per thread, and never larger than the vector width: the
    // caller rounds the step count to it, which is the whole
    // reason the steps carry the lanes.
    s_t merit_local[VS];
    for (int lane = 0; lane < VS; ++lane) merit_local[lane] = s_t(0);
    s_t rho[NC * VS];
    s_t pm_test[NQ * 1 * VS];
    s_t pm_test_grad[NQ * 3 * VS];
    s_t pm_weight[NQ * 1 * VS];
    s_t pm_state_value[NQ * 2 * VS];
    s_t pm_state_grad[NQ * 6 * VS];
    s_t pm_direction_value[NQ * 2 * VS];
    s_t pm_direction_grad[NQ * 6 * VS];
    s_t pm_previous_value[NQ * 2 * VS];
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
        {
            const int q = 0;  // TET4 evaluates in closed form
          #pragma omp simd
          for (int lane = 0; lane < ne; ++lane) {
            const idx_t element = pm_incident[lane];
            const int *const RSTR perm = pm_orientation[pm_local_node[lane]];
            s_t state[8];
            s_t direction[8];
            s_t previous[8];
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
            // interpolated state value
            for (int c = 0; c < NC; ++c) {
              s_t acc = s_t(0);
              for (int j = 0; j < NS; ++j) {
                acc += state[j * NC + c] * shape[q * NS + j];
              }
              pm_state_value[(q * 2 + c) * VS + lane] = acc;
            }
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
                pm_state_grad[(q * 6 + c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // interpolated direction value
            for (int c = 0; c < NC; ++c) {
              s_t acc = s_t(0);
              for (int j = 0; j < NS; ++j) {
                acc += direction[j * NC + c] * shape[q * NS + j];
              }
              pm_direction_value[(q * 2 + c) * VS + lane] = acc;
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
                pm_direction_grad[(q * 6 + c * ND + d) * VS + lane] = mapped / det;
              }
            }
            // interpolated previous value
            for (int c = 0; c < NC; ++c) {
              s_t acc = s_t(0);
              for (int j = 0; j < NS; ++j) {
                acc += previous[j * NC + c] * shape[q * NS + j];
              }
              pm_previous_value[(q * 2 + c) * VS + lane] = acc;
            }
            // the fixed basis function's quantities, and the
            // integration weight.  Both are what the orientation buys:
            // `phi_0` is the same function in every element and at
            // every step, so this leaves the step loop entirely.
            pm_test[q * VS + lane] = shape[q * NS + 0];
            for (int d = 0; d < ND; ++d) {
              s_t mapped = s_t(0);
              for (int k = 0; k < ND; ++k) {
                mapped += grad_ref[k][q * NS + 0] * adj[k * ND + d];
              }
              pm_test_grad[(q * 3 + d) * VS + lane] = mapped / det;
            }
            pm_weight[q * VS + lane] = q_weight[q] * det;
          }
        }
        // loop 2 -- lanes are the sampled step lengths.
        for (int lane_e = 0; lane_e < ne; ++lane_e) {
          {
              const int q = 0;  // TET4 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nsteps; ++lane) {
              const s_t alpha = steps[lane];
              const s_t p_w = pm_state_value[(q * 2 + 0) * VS + lane_e] + alpha * pm_direction_value[(q * 2 + 0) * VS + lane_e];
              const s_t p_c = pm_state_value[(q * 2 + 1) * VS + lane_e] + alpha * pm_direction_value[(q * 2 + 1) * VS + lane_e];
              const s_t p_w_grad_0 = pm_state_grad[(q * 6 + 0) * VS + lane_e] + alpha * pm_direction_grad[(q * 6 + 0) * VS + lane_e];
              const s_t p_w_grad_1 = pm_state_grad[(q * 6 + 1) * VS + lane_e] + alpha * pm_direction_grad[(q * 6 + 1) * VS + lane_e];
              const s_t p_w_grad_2 = pm_state_grad[(q * 6 + 2) * VS + lane_e] + alpha * pm_direction_grad[(q * 6 + 2) * VS + lane_e];
              const s_t p_c_grad_0 = pm_state_grad[(q * 6 + 3) * VS + lane_e] + alpha * pm_direction_grad[(q * 6 + 3) * VS + lane_e];
              const s_t p_c_grad_1 = pm_state_grad[(q * 6 + 4) * VS + lane_e] + alpha * pm_direction_grad[(q * 6 + 4) * VS + lane_e];
              const s_t p_c_grad_2 = pm_state_grad[(q * 6 + 5) * VS + lane_e] + alpha * pm_direction_grad[(q * 6 + 5) * VS + lane_e];
              const s_t p_w_old = pm_previous_value[(q * 2 + 0) * VS + lane_e];
              const s_t p_c_old = pm_previous_value[(q * 2 + 1) * VS + lane_e];
              const s_t residual_tmp0 = -p_wr;
              const s_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
              const s_t residual_tmp2 = S_res + s_t(-1);
              const s_t residual_tmp3 = -residual_tmp2;
              const s_t residual_tmp4 = pow_m1(P_r);
              const s_t residual_tmp5 = (s_t(1) - m)/m;
              const s_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + s_t(1), residual_tmp5);
              const s_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5);
              const s_t residual_tmp8 = porosity/dt;
              const s_t residual_tmp9 = residual_tmp2*residual_tmp6;
              const s_t residual_tmp10 = S_res - residual_tmp9;
              const s_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
              const s_t residual_tmp12 = s_t(1) - S_res;
              const s_t residual_tmp13 = M_c/(R*T*Z);
              const s_t residual_tmp14 = p_c*residual_tmp13*pow(s_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + s_t(-1))/mu_c;
              const s_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
              const s_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
              const s_t grad_coeff0_1 = residual_tmp11*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
              const s_t grad_coeff0_2 = residual_tmp11*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
              const s_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
              const s_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
              const s_t grad_coeff1_1 = residual_tmp14*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
              const s_t grad_coeff1_2 = residual_tmp14*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
              const s_t weight = pm_weight[q * VS + lane_e];
              rho[0 * VS + lane] += weight * (value_coeff0 * pm_test[q * VS + lane_e] + grad_coeff0_0 * pm_test_grad[(q * 3 + 0) * VS + lane_e] + grad_coeff0_1 * pm_test_grad[(q * 3 + 1) * VS + lane_e] + grad_coeff0_2 * pm_test_grad[(q * 3 + 2) * VS + lane_e]);
              rho[1 * VS + lane] += weight * (value_coeff1 * pm_test[q * VS + lane_e] + grad_coeff1_0 * pm_test_grad[(q * 3 + 0) * VS + lane_e] + grad_coeff1_1 * pm_test_grad[(q * 3 + 1) * VS + lane_e] + grad_coeff1_2 * pm_test_grad[(q * 3 + 2) * VS + lane_e]);
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


extern "C" int two_phase_flow_total_tet4_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR shape,
    const void *const RSTR grad_ref[3],
    const void *const RSTR q_weight,
    const real_t P_r,
    const real_t S_res,
    const real_t dt,
    const real_t kappa_T,
    const real_t m,
    const real_t p_wr,
    const real_t porosity,
    const real_t rho_w0,
    const real_t C_kw1,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t mu_w,
    const real_t K_3,
    const real_t K_4,
    const real_t K_5,
    const real_t K_6,
    const real_t K_7,
    const real_t K_8,
    const real_t M_c,
    const real_t R,
    const real_t T,
    const real_t Z,
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t mu_c,
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
        return sfem::codegen::two_phase_flow_total_tet4_merit_patch<double, geom_t, 11, 4, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, (const double *)shape, (const double *const *)grad_ref, (const double *)q_weight, P_r, S_res, dt, kappa_T, m, p_wr, porosity, rho_w0, C_kw1, K_0, K_1, K_2, mu_w, K_3, K_4, K_5, K_6, K_7, K_8, M_c, R, T, Z, C_ka1, C_ka2, mu_c, nsteps, (const double *)steps, (const double *)x, (const double *)h, (const double *)p, (const double *)accumulator, (double *)merit);
    }
    case (int)sizeof(float): {
        return sfem::codegen::two_phase_flow_total_tet4_merit_patch<float, geom_t, 11, 4, 16>(n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, (const float *)shape, (const float *const *)grad_ref, (const float *)q_weight, P_r, S_res, dt, kappa_T, m, p_wr, porosity, rho_w0, C_kw1, K_0, K_1, K_2, mu_w, K_3, K_4, K_5, K_6, K_7, K_8, M_c, R, T, Z, C_ka1, C_ka2, mu_c, nsteps, (const float *)steps, (const float *)x, (const float *)h, (const float *)p, (const float *)accumulator, (float *)merit);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_total_tet4_merit_patch_a_msoa", -1, (int)scalar_bytes);
}
