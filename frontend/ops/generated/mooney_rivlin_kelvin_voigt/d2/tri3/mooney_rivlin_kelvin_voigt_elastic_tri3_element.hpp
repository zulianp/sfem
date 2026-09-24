#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_ELASTIC_TRI3_ELEMENT_API_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_ELASTIC_TRI3_ELEMENT_API_HPP

#include <stddef.h>
#include "../mooney_rivlin_kelvin_voigt_elastic_d2_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"

namespace sfem {
namespace codegen {


template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_elastic_tri3_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 2;
  static constexpr int NS = 3;
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
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t bdet0[VS];
    {  // TRI3 evaluates in closed form
      s_t *const RSTR badj0_q = badj0;
      const s_t *const RSTR adj0_q = adj[0] + evb;
      s_t *const RSTR badj1_q = badj1;
      const s_t *const RSTR adj1_q = adj[1] + evb;
      s_t *const RSTR badj2_q = badj2;
      const s_t *const RSTR adj2_q = adj[2] + evb;
      s_t *const RSTR badj3_q = badj3;
      const s_t *const RSTR adj3_q = adj[3] + evb;
      s_t *const RSTR bdet0_q = bdet0;
      const s_t *const RSTR det_q = det + evb;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0_q[lane] = adj0_q[lane];
        badj1_q[lane] = adj1_q[lane];
        badj2_q[lane] = adj2_q[lane];
        badj3_q[lane] = adj3_q[lane];
        bdet0_q[lane] = det_q[lane];
      }
    }
    mooney_rivlin_kelvin_voigt_elastic_d2_simplex_tri3_gradient_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_elastic_tri3_gradient_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 3;
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
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t bdet0[VS];
    {  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[2][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[4][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[3][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[5][lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, lane);
      }
    }
    mooney_rivlin_kelvin_voigt_elastic_d2_simplex_tri3_gradient_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_elastic_tri3_gradient_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 3;
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
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t bdet0[VS];
    {  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[2][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[4][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[3][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[5][lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, lane);
      }
    }
    mooney_rivlin_kelvin_voigt_elastic_d2_simplex_tri3_gradient_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_elastic_tri3_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 2;
  static constexpr int NS = 3;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t bdet0[VS];
    {  // TRI3 evaluates in closed form
      s_t *const RSTR badj0_q = badj0;
      const s_t *const RSTR adj0_q = adj[0] + evb;
      s_t *const RSTR badj1_q = badj1;
      const s_t *const RSTR adj1_q = adj[1] + evb;
      s_t *const RSTR badj2_q = badj2;
      const s_t *const RSTR adj2_q = adj[2] + evb;
      s_t *const RSTR badj3_q = badj3;
      const s_t *const RSTR adj3_q = adj[3] + evb;
      s_t *const RSTR bdet0_q = bdet0;
      const s_t *const RSTR det_q = det + evb;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0_q[lane] = adj0_q[lane];
        badj1_q[lane] = adj1_q[lane];
        badj2_q[lane] = adj2_q[lane];
        badj3_q[lane] = adj3_q[lane];
        bdet0_q[lane] = det_q[lane];
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
      mooney_rivlin_kelvin_voigt_elastic_d2_simplex_tri3_apply_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_elastic_tri3_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 3;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t bdet0[VS];
    {  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[2][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[4][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[3][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[5][lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, lane);
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
      mooney_rivlin_kelvin_voigt_elastic_d2_simplex_tri3_apply_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_elastic_tri3_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 3;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t bdet0[VS];
    {  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[2][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[4][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[3][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[5][lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, lane);
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
      mooney_rivlin_kelvin_voigt_elastic_d2_simplex_tri3_apply_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
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
