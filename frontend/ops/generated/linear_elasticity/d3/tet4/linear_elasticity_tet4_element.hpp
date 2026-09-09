#ifndef LINEAR_ELASTICITY_TET4_ELEMENT_API_HPP
#define LINEAR_ELASTICITY_TET4_ELEMENT_API_HPP

#include <stddef.h>
#include "../linear_elasticity_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../reference/quad_tet_q1.hpp"
#include "../../../reference/tet4_q1.hpp"

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


template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *const bvalue = values + evb;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bvalue[lane] = s_t(0);
    }
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
    {
        const int q = 0;  // TET4 evaluates in closed form
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0[q * VS + lane] = adj[0][q * nelements + evb + lane];
        badj1[q * VS + lane] = adj[1][q * nelements + evb + lane];
        badj2[q * VS + lane] = adj[2][q * nelements + evb + lane];
        badj3[q * VS + lane] = adj[3][q * nelements + evb + lane];
        badj4[q * VS + lane] = adj[4][q * nelements + evb + lane];
        badj5[q * VS + lane] = adj[5][q * nelements + evb + lane];
        badj6[q * VS + lane] = adj[6][q * nelements + evb + lane];
        badj7[q * VS + lane] = adj[7][q * nelements + evb + lane];
        badj8[q * VS + lane] = adj[8][q * nelements + evb + lane];
        bdet0[q * VS + lane] = det[q * nelements + evb + lane];
      }
    }
    linear_elasticity_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_energy_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *const bvalue = values + evb;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bvalue[lane] = s_t(0);
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z();
    {
        const int q = 0;  // TET4 evaluates in closed form
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
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[3 * shape][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[3 * shape][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J02_values[lane] += bcoordinate_data[3 * shape][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J12_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J20_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J21_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J22_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g2;
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
    linear_elasticity_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_energy_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *const bvalue = values + evb;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bvalue[lane] = s_t(0);
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z();
    {
        const int q = 0;  // TET4 evaluates in closed form
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
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[3 * shape][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[3 * shape][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J02_values[lane] += bcoordinate_data[3 * shape][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J12_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J20_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J21_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J22_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g2;
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
    linear_elasticity_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *bout_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bout_streams[stream] = out_streams[stream] + evb;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bout_streams[stream][lane] = s_t(0);
      }
    }
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
    {
        const int q = 0;  // TET4 evaluates in closed form
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0[q * VS + lane] = adj[0][q * nelements + evb + lane];
        badj1[q * VS + lane] = adj[1][q * nelements + evb + lane];
        badj2[q * VS + lane] = adj[2][q * nelements + evb + lane];
        badj3[q * VS + lane] = adj[3][q * nelements + evb + lane];
        badj4[q * VS + lane] = adj[4][q * nelements + evb + lane];
        badj5[q * VS + lane] = adj[5][q * nelements + evb + lane];
        badj6[q * VS + lane] = adj[6][q * nelements + evb + lane];
        badj7[q * VS + lane] = adj[7][q * nelements + evb + lane];
        badj8[q * VS + lane] = adj[8][q * nelements + evb + lane];
        bdet0[q * VS + lane] = det[q * nelements + evb + lane];
      }
    }
    linear_elasticity_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_gradient_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *bout_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bout_streams[stream] = out_streams[stream] + evb;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bout_streams[stream][lane] = s_t(0);
      }
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z();
    {
        const int q = 0;  // TET4 evaluates in closed form
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
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[3 * shape][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[3 * shape][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J02_values[lane] += bcoordinate_data[3 * shape][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J12_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J20_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J21_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J22_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g2;
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
    linear_elasticity_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_gradient_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *bout_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bout_streams[stream] = out_streams[stream] + evb;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bout_streams[stream][lane] = s_t(0);
      }
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z();
    {
        const int q = 0;  // TET4 evaluates in closed form
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
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[3 * shape][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[3 * shape][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J02_values[lane] += bcoordinate_data[3 * shape][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J12_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J20_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J21_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J22_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g2;
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
    linear_elasticity_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
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
    {
        const int q = 0;  // TET4 evaluates in closed form
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0[q * VS + lane] = adj[0][q * nelements + evb + lane];
        badj1[q * VS + lane] = adj[1][q * nelements + evb + lane];
        badj2[q * VS + lane] = adj[2][q * nelements + evb + lane];
        badj3[q * VS + lane] = adj[3][q * nelements + evb + lane];
        badj4[q * VS + lane] = adj[4][q * nelements + evb + lane];
        badj5[q * VS + lane] = adj[5][q * nelements + evb + lane];
        badj6[q * VS + lane] = adj[6][q * nelements + evb + lane];
        badj7[q * VS + lane] = adj[7][q * nelements + evb + lane];
        badj8[q * VS + lane] = adj[8][q * nelements + evb + lane];
        bdet0[q * VS + lane] = det[q * nelements + evb + lane];
      }
    }
    s_t bh_data[NDOFS][VS];
    s_t bout_data[NDOFS][VS];
    const s_t *bh_streams[NDOFS];
    s_t *bout_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bh_streams[stream] = bh_data[stream];
      bout_streams[stream] = bout_data[stream];
    }
    for (int col = 0; col < NDOFS; ++col) {
      for (int stream = 0; stream < NDOFS; ++stream) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][lane] = s_t(0);
        }
      }
      linear_elasticity_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bh_streams, bout_streams);
      for (int row = 0; row < NDOFS; ++row) {
        s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          matrix_stream[lane] = bout_data[row][lane];
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z();
    {
        const int q = 0;  // TET4 evaluates in closed form
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
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[3 * shape][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[3 * shape][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J02_values[lane] += bcoordinate_data[3 * shape][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J12_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J20_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J21_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J22_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g2;
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
    s_t bh_data[NDOFS][VS];
    s_t bout_data[NDOFS][VS];
    const s_t *bh_streams[NDOFS];
    s_t *bout_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bh_streams[stream] = bh_data[stream];
      bout_streams[stream] = bout_data[stream];
    }
    for (int col = 0; col < NDOFS; ++col) {
      for (int stream = 0; stream < NDOFS; ++stream) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][lane] = s_t(0);
        }
      }
      linear_elasticity_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bh_streams, bout_streams);
      for (int row = 0; row < NDOFS; ++row) {
        s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          matrix_stream[lane] = bout_data[row][lane];
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int linear_elasticity_tet4_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NQ = 1;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z();
    {
        const int q = 0;  // TET4 evaluates in closed form
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
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[3 * shape][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J01_values[lane] += bcoordinate_data[3 * shape][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J02_values[lane] += bcoordinate_data[3 * shape][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J10_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J11_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J12_values[lane] += bcoordinate_data[3 * shape + 1][lane] * g2;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J20_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g0;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J21_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g1;
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J22_values[lane] += bcoordinate_data[3 * shape + 2][lane] * g2;
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
    s_t bh_data[NDOFS][VS];
    s_t bout_data[NDOFS][VS];
    const s_t *bh_streams[NDOFS];
    s_t *bout_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bh_streams[stream] = bh_data[stream];
      bout_streams[stream] = bout_data[stream];
    }
    for (int col = 0; col < NDOFS; ++col) {
      for (int stream = 0; stream < NDOFS; ++stream) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][lane] = s_t(0);
        }
      }
      linear_elasticity_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet4_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_y(), sfem::codegen::ref_tet4_q1<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q1<s_t>::q_weight(), lmbda, mu, bh_streams, bout_streams);
      for (int row = 0; row < NDOFS; ++row) {
        s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          matrix_stream[lane] = bout_data[row][lane];
        }
      }
    }
  }
  return SFEM_SUCCESS;
}


} // namespace codegen
} // namespace sfem

#endif
