#ifndef NEOHOOKEAN_OGDEN_TET4_ELEMENT_API_HPP
#define NEOHOOKEAN_OGDEN_TET4_ELEMENT_API_HPP

#include <stddef.h>
#include "../neohookean_ogden_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../reference/quad_tet_q1.hpp"
#include "../../../reference/tet4_q1.hpp"

namespace sfem {
namespace codegen {


template <typename s_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet4_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
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
    const s_t objective_step = s_t(0);
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0_q[lane] = adj0_q[lane];
        badj1_q[lane] = adj1_q[lane];
        badj2_q[lane] = adj2_q[lane];
        badj3_q[lane] = adj3_q[lane];
        badj4_q[lane] = adj4_q[lane];
        badj5_q[lane] = adj5_q[lane];
        badj6_q[lane] = adj6_q[lane];
        badj7_q[lane] = adj7_q[lane];
        badj8_q[lane] = adj8_q[lane];
        bdet0_q[lane] = det_q[lane];
      }
    }
    neohookean_ogden_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bu_streams, 1, &objective_step, 0, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet4_energy_ecoords_soa(
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
    const s_t objective_step = s_t(0);
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
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[3][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[6][lane];
        const s_t J02 = -bcoordinate_data[0][lane] + bcoordinate_data[9][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[4][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[7][lane];
        const s_t J12 = bcoordinate_data[10][lane] - bcoordinate_data[1][lane];
        const s_t J20 = -bcoordinate_data[2][lane] + bcoordinate_data[5][lane];
        const s_t J21 = -bcoordinate_data[2][lane] + bcoordinate_data[8][lane];
        const s_t J22 = bcoordinate_data[11][lane] - bcoordinate_data[2][lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, lane);
      }
    }
    neohookean_ogden_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bu_streams, 1, &objective_step, 0, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet4_energy_esoa(
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
    const s_t objective_step = s_t(0);
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
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[3][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[6][lane];
        const s_t J02 = -bcoordinate_data[0][lane] + bcoordinate_data[9][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[4][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[7][lane];
        const s_t J12 = bcoordinate_data[10][lane] - bcoordinate_data[1][lane];
        const s_t J20 = -bcoordinate_data[2][lane] + bcoordinate_data[5][lane];
        const s_t J21 = -bcoordinate_data[2][lane] + bcoordinate_data[8][lane];
        const s_t J22 = bcoordinate_data[11][lane] - bcoordinate_data[2][lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, lane);
      }
    }
    neohookean_ogden_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bu_streams, 1, &objective_step, 0, bvalue);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet4_gradient_egeometry_soa(
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0_q[lane] = adj0_q[lane];
        badj1_q[lane] = adj1_q[lane];
        badj2_q[lane] = adj2_q[lane];
        badj3_q[lane] = adj3_q[lane];
        badj4_q[lane] = adj4_q[lane];
        badj5_q[lane] = adj5_q[lane];
        badj6_q[lane] = adj6_q[lane];
        badj7_q[lane] = adj7_q[lane];
        badj8_q[lane] = adj8_q[lane];
        bdet0_q[lane] = det_q[lane];
      }
    }
    neohookean_ogden_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet4_gradient_ecoords_soa(
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[3][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[6][lane];
        const s_t J02 = -bcoordinate_data[0][lane] + bcoordinate_data[9][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[4][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[7][lane];
        const s_t J12 = bcoordinate_data[10][lane] - bcoordinate_data[1][lane];
        const s_t J20 = -bcoordinate_data[2][lane] + bcoordinate_data[5][lane];
        const s_t J21 = -bcoordinate_data[2][lane] + bcoordinate_data[8][lane];
        const s_t J22 = bcoordinate_data[11][lane] - bcoordinate_data[2][lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, lane);
      }
    }
    neohookean_ogden_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet4_gradient_esoa(
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[3][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[6][lane];
        const s_t J02 = -bcoordinate_data[0][lane] + bcoordinate_data[9][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[4][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[7][lane];
        const s_t J12 = bcoordinate_data[10][lane] - bcoordinate_data[1][lane];
        const s_t J20 = -bcoordinate_data[2][lane] + bcoordinate_data[5][lane];
        const s_t J21 = -bcoordinate_data[2][lane] + bcoordinate_data[8][lane];
        const s_t J22 = bcoordinate_data[11][lane] - bcoordinate_data[2][lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, lane);
      }
    }
    neohookean_ogden_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static SFEM_INLINE int neohookean_ogden_tet4_hessian_egeometry_soa(
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
  static constexpr int NQ = 1;
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0_q[lane] = adj0_q[lane];
        badj1_q[lane] = adj1_q[lane];
        badj2_q[lane] = adj2_q[lane];
        badj3_q[lane] = adj3_q[lane];
        badj4_q[lane] = adj4_q[lane];
        badj5_q[lane] = adj5_q[lane];
        badj6_q[lane] = adj6_q[lane];
        badj7_q[lane] = adj7_q[lane];
        badj8_q[lane] = adj8_q[lane];
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
      neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static SFEM_INLINE int neohookean_ogden_tet4_hessian_ecoords_soa(
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
  static constexpr int NQ = 1;
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
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[3][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[6][lane];
        const s_t J02 = -bcoordinate_data[0][lane] + bcoordinate_data[9][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[4][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[7][lane];
        const s_t J12 = bcoordinate_data[10][lane] - bcoordinate_data[1][lane];
        const s_t J20 = -bcoordinate_data[2][lane] + bcoordinate_data[5][lane];
        const s_t J21 = -bcoordinate_data[2][lane] + bcoordinate_data[8][lane];
        const s_t J22 = bcoordinate_data[11][lane] - bcoordinate_data[2][lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, lane);
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
      neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static SFEM_INLINE int neohookean_ogden_tet4_hessian_esoa(
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
  static constexpr int NQ = 1;
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
    s_t badj4[VS];
    s_t badj5[VS];
    s_t badj6[VS];
    s_t badj7[VS];
    s_t badj8[VS];
    s_t bdet0[VS];
    {  // TET4 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = -bcoordinate_data[0][lane] + bcoordinate_data[3][lane];
        const s_t J01 = -bcoordinate_data[0][lane] + bcoordinate_data[6][lane];
        const s_t J02 = -bcoordinate_data[0][lane] + bcoordinate_data[9][lane];
        const s_t J10 = -bcoordinate_data[1][lane] + bcoordinate_data[4][lane];
        const s_t J11 = -bcoordinate_data[1][lane] + bcoordinate_data[7][lane];
        const s_t J12 = bcoordinate_data[10][lane] - bcoordinate_data[1][lane];
        const s_t J20 = -bcoordinate_data[2][lane] + bcoordinate_data[5][lane];
        const s_t J21 = -bcoordinate_data[2][lane] + bcoordinate_data[8][lane];
        const s_t J22 = bcoordinate_data[11][lane] - bcoordinate_data[2][lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, lane);
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
      neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(ne, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);
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
