#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_ELASTIC_TET4_ELEMENT_API_CUH
#define MOONEY_RIVLIN_KELVIN_VOIGT_ELASTIC_TET4_ELEMENT_API_CUH

#include <stddef.h>
#include "../../cuda/mooney_rivlin_kelvin_voigt_elastic_d3_simplex_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"

namespace sfem {
namespace codegen {


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_elastic_tet4_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 4;
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
      {
        bout_streams[stream][0] = s_t(0);
      }
    }
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *const RSTR badj0_q = badj0;
      const s_t *const RSTR adj0_q = adj[0] + evb;
      s_t *const RSTR badj1_q = badj1;
      const s_t *const RSTR adj1_q = adj[1] + evb;
      s_t *const RSTR badj2_q = badj2;
      const s_t *const RSTR adj2_q = adj[2] + evb;
      s_t *const RSTR badj3_q = badj3;
      const s_t *const RSTR adj3_q = adj[3] + evb;
      s_t *const RSTR badj4_q = badj4;
      const s_t *const RSTR adj4_q = adj[4] + evb;
      s_t *const RSTR badj5_q = badj5;
      const s_t *const RSTR adj5_q = adj[5] + evb;
      s_t *const RSTR badj6_q = badj6;
      const s_t *const RSTR adj6_q = adj[6] + evb;
      s_t *const RSTR badj7_q = badj7;
      const s_t *const RSTR adj7_q = adj[7] + evb;
      s_t *const RSTR badj8_q = badj8;
      const s_t *const RSTR adj8_q = adj[8] + evb;
      s_t *const RSTR bdet0_q = bdet0;
      const s_t *const RSTR det_q = det + evb;
      {
        badj0_q[0] = adj0_q[0];
        badj1_q[0] = adj1_q[0];
        badj2_q[0] = adj2_q[0];
        badj3_q[0] = adj3_q[0];
        badj4_q[0] = adj4_q[0];
        badj5_q[0] = adj5_q[0];
        badj6_q[0] = adj6_q[0];
        badj7_q[0] = adj7_q[0];
        badj8_q[0] = adj8_q[0];
        bdet0_q[0] = det_q[0];
      }
    }
    mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_gradient_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_elastic_tet4_gradient_ecoords_soa(
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
      {
        bout_streams[stream][0] = s_t(0);
      }
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
      }
    }
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      {
        const s_t J00 = -bcoordinate_data[0][0] + bcoordinate_data[3][0];
        const s_t J01 = -bcoordinate_data[0][0] + bcoordinate_data[6][0];
        const s_t J02 = -bcoordinate_data[0][0] + bcoordinate_data[9][0];
        const s_t J10 = -bcoordinate_data[1][0] + bcoordinate_data[4][0];
        const s_t J11 = -bcoordinate_data[1][0] + bcoordinate_data[7][0];
        const s_t J12 = bcoordinate_data[10][0] - bcoordinate_data[1][0];
        const s_t J20 = -bcoordinate_data[2][0] + bcoordinate_data[5][0];
        const s_t J21 = -bcoordinate_data[2][0] + bcoordinate_data[8][0];
        const s_t J22 = bcoordinate_data[11][0] - bcoordinate_data[2][0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, 0);
      }
    }
    mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_gradient_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_elastic_tet4_gradient_esoa(
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
      {
        bout_streams[stream][0] = s_t(0);
      }
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
      }
    }
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      {
        const s_t J00 = -bcoordinate_data[0][0] + bcoordinate_data[3][0];
        const s_t J01 = -bcoordinate_data[0][0] + bcoordinate_data[6][0];
        const s_t J02 = -bcoordinate_data[0][0] + bcoordinate_data[9][0];
        const s_t J10 = -bcoordinate_data[1][0] + bcoordinate_data[4][0];
        const s_t J11 = -bcoordinate_data[1][0] + bcoordinate_data[7][0];
        const s_t J12 = bcoordinate_data[10][0] - bcoordinate_data[1][0];
        const s_t J20 = -bcoordinate_data[2][0] + bcoordinate_data[5][0];
        const s_t J21 = -bcoordinate_data[2][0] + bcoordinate_data[8][0];
        const s_t J22 = bcoordinate_data[11][0] - bcoordinate_data[2][0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, 0);
      }
    }
    mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_gradient_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_elastic_tet4_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 4;
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
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *const RSTR badj0_q = badj0;
      const s_t *const RSTR adj0_q = adj[0] + evb;
      s_t *const RSTR badj1_q = badj1;
      const s_t *const RSTR adj1_q = adj[1] + evb;
      s_t *const RSTR badj2_q = badj2;
      const s_t *const RSTR adj2_q = adj[2] + evb;
      s_t *const RSTR badj3_q = badj3;
      const s_t *const RSTR adj3_q = adj[3] + evb;
      s_t *const RSTR badj4_q = badj4;
      const s_t *const RSTR adj4_q = adj[4] + evb;
      s_t *const RSTR badj5_q = badj5;
      const s_t *const RSTR adj5_q = adj[5] + evb;
      s_t *const RSTR badj6_q = badj6;
      const s_t *const RSTR adj6_q = adj[6] + evb;
      s_t *const RSTR badj7_q = badj7;
      const s_t *const RSTR adj7_q = adj[7] + evb;
      s_t *const RSTR badj8_q = badj8;
      const s_t *const RSTR adj8_q = adj[8] + evb;
      s_t *const RSTR bdet0_q = bdet0;
      const s_t *const RSTR det_q = det + evb;
      {
        badj0_q[0] = adj0_q[0];
        badj1_q[0] = adj1_q[0];
        badj2_q[0] = adj2_q[0];
        badj3_q[0] = adj3_q[0];
        badj4_q[0] = adj4_q[0];
        badj5_q[0] = adj5_q[0];
        badj6_q[0] = adj6_q[0];
        badj7_q[0] = adj7_q[0];
        badj8_q[0] = adj8_q[0];
        bdet0_q[0] = det_q[0];
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
        {
          bh_data[stream][0] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][0] = s_t(0);
        }
      }
      mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_apply_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
      for (int row = 0; row < NDOFS; ++row) {
        s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
        {
          matrix_stream[0] = bout_data[row][0];
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_elastic_tet4_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
      }
    }
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      {
        const s_t J00 = -bcoordinate_data[0][0] + bcoordinate_data[3][0];
        const s_t J01 = -bcoordinate_data[0][0] + bcoordinate_data[6][0];
        const s_t J02 = -bcoordinate_data[0][0] + bcoordinate_data[9][0];
        const s_t J10 = -bcoordinate_data[1][0] + bcoordinate_data[4][0];
        const s_t J11 = -bcoordinate_data[1][0] + bcoordinate_data[7][0];
        const s_t J12 = bcoordinate_data[10][0] - bcoordinate_data[1][0];
        const s_t J20 = -bcoordinate_data[2][0] + bcoordinate_data[5][0];
        const s_t J21 = -bcoordinate_data[2][0] + bcoordinate_data[8][0];
        const s_t J22 = bcoordinate_data[11][0] - bcoordinate_data[2][0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, 0);
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
        {
          bh_data[stream][0] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][0] = s_t(0);
        }
      }
      mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_apply_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
      for (int row = 0; row < NDOFS; ++row) {
        s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
        {
          matrix_stream[0] = bout_data[row][0];
        }
      }
    }
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_elastic_tet4_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NS = 4;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
      }
    }
    s_t badj0[VS];
    s_t badj1[VS];
    s_t badj2[VS];
    s_t badj3[VS];
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      {
        const s_t J00 = -bcoordinate_data[0][0] + bcoordinate_data[3][0];
        const s_t J01 = -bcoordinate_data[0][0] + bcoordinate_data[6][0];
        const s_t J02 = -bcoordinate_data[0][0] + bcoordinate_data[9][0];
        const s_t J10 = -bcoordinate_data[1][0] + bcoordinate_data[4][0];
        const s_t J11 = -bcoordinate_data[1][0] + bcoordinate_data[7][0];
        const s_t J12 = bcoordinate_data[10][0] - bcoordinate_data[1][0];
        const s_t J20 = -bcoordinate_data[2][0] + bcoordinate_data[5][0];
        const s_t J21 = -bcoordinate_data[2][0] + bcoordinate_data[8][0];
        const s_t J22 = bcoordinate_data[11][0] - bcoordinate_data[2][0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, 0);
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
        {
          bh_data[stream][0] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][0] = s_t(0);
        }
      }
      mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_apply_block<s_t, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
      for (int row = 0; row < NDOFS; ++row) {
        s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
        {
          matrix_stream[0] = bout_data[row][0];
        }
      }
    }
  }
  return SFEM_SUCCESS;
}


} // namespace codegen
} // namespace sfem

#endif
