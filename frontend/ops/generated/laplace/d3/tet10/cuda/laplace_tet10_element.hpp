#ifndef LAPLACE_TET10_ELEMENT_API_CUH
#define LAPLACE_TET10_ELEMENT_API_CUH

#include <stddef.h>
#include "../../cuda/laplace_d3_simplex_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../reference/cuda/quad_tet_q11.hpp"
#include "../../../../reference/cuda/tet10_q11.hpp"

namespace sfem {
namespace codegen {


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int laplace_tet10_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 1;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *const bvalue = values + evb;
    {
      bvalue[0] = s_t(0);
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
    for (int q = 0; q < NQ; ++q) {
      s_t *const RSTR badj0_q = &badj0[q * VS];
      const s_t *const RSTR adj0_q = adj[0] + q * nelements + evb;
      s_t *const RSTR badj1_q = &badj1[q * VS];
      const s_t *const RSTR adj1_q = adj[1] + q * nelements + evb;
      s_t *const RSTR badj2_q = &badj2[q * VS];
      const s_t *const RSTR adj2_q = adj[2] + q * nelements + evb;
      s_t *const RSTR badj3_q = &badj3[q * VS];
      const s_t *const RSTR adj3_q = adj[3] + q * nelements + evb;
      s_t *const RSTR badj4_q = &badj4[q * VS];
      const s_t *const RSTR adj4_q = adj[4] + q * nelements + evb;
      s_t *const RSTR badj5_q = &badj5[q * VS];
      const s_t *const RSTR adj5_q = adj[5] + q * nelements + evb;
      s_t *const RSTR badj6_q = &badj6[q * VS];
      const s_t *const RSTR adj6_q = adj[6] + q * nelements + evb;
      s_t *const RSTR badj7_q = &badj7[q * VS];
      const s_t *const RSTR adj7_q = adj[7] + q * nelements + evb;
      s_t *const RSTR badj8_q = &badj8[q * VS];
      const s_t *const RSTR adj8_q = adj[8] + q * nelements + evb;
      s_t *const RSTR bdet0_q = &bdet0[q * VS];
      const s_t *const RSTR det_q = det + q * nelements + evb;
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
    laplace_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int laplace_tet10_energy_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 1;
  static constexpr int ND = 3;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *const bvalue = values + evb;
    {
      bvalue[0] = s_t(0);
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
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
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
      }
    }
    laplace_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int laplace_tet10_energy_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 1;
  static constexpr int ND = 3;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      bu_streams[stream] = u_streams[stream] + evb;
    }
    s_t *const bvalue = values + evb;
    {
      bvalue[0] = s_t(0);
    }
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
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
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
      }
    }
    laplace_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int laplace_tet10_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 1;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
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
    for (int q = 0; q < NQ; ++q) {
      s_t *const RSTR badj0_q = &badj0[q * VS];
      const s_t *const RSTR adj0_q = adj[0] + q * nelements + evb;
      s_t *const RSTR badj1_q = &badj1[q * VS];
      const s_t *const RSTR adj1_q = adj[1] + q * nelements + evb;
      s_t *const RSTR badj2_q = &badj2[q * VS];
      const s_t *const RSTR adj2_q = adj[2] + q * nelements + evb;
      s_t *const RSTR badj3_q = &badj3[q * VS];
      const s_t *const RSTR adj3_q = adj[3] + q * nelements + evb;
      s_t *const RSTR badj4_q = &badj4[q * VS];
      const s_t *const RSTR adj4_q = adj[4] + q * nelements + evb;
      s_t *const RSTR badj5_q = &badj5[q * VS];
      const s_t *const RSTR adj5_q = adj[5] + q * nelements + evb;
      s_t *const RSTR badj6_q = &badj6[q * VS];
      const s_t *const RSTR adj6_q = adj[6] + q * nelements + evb;
      s_t *const RSTR badj7_q = &badj7[q * VS];
      const s_t *const RSTR adj7_q = adj[7] + q * nelements + evb;
      s_t *const RSTR badj8_q = &badj8[q * VS];
      const s_t *const RSTR adj8_q = adj[8] + q * nelements + evb;
      s_t *const RSTR bdet0_q = &bdet0[q * VS];
      const s_t *const RSTR det_q = det + q * nelements + evb;
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
    laplace_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int laplace_tet10_gradient_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 1;
  static constexpr int ND = 3;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
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
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
      }
    }
    laplace_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int laplace_tet10_gradient_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 1;
  static constexpr int ND = 3;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
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
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
      }
    }
    laplace_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int laplace_tet10_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t kappa,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 1;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
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
    for (int q = 0; q < NQ; ++q) {
      s_t *const RSTR badj0_q = &badj0[q * VS];
      const s_t *const RSTR adj0_q = adj[0] + q * nelements + evb;
      s_t *const RSTR badj1_q = &badj1[q * VS];
      const s_t *const RSTR adj1_q = adj[1] + q * nelements + evb;
      s_t *const RSTR badj2_q = &badj2[q * VS];
      const s_t *const RSTR adj2_q = adj[2] + q * nelements + evb;
      s_t *const RSTR badj3_q = &badj3[q * VS];
      const s_t *const RSTR adj3_q = adj[3] + q * nelements + evb;
      s_t *const RSTR badj4_q = &badj4[q * VS];
      const s_t *const RSTR adj4_q = adj[4] + q * nelements + evb;
      s_t *const RSTR badj5_q = &badj5[q * VS];
      const s_t *const RSTR adj5_q = adj[5] + q * nelements + evb;
      s_t *const RSTR badj6_q = &badj6[q * VS];
      const s_t *const RSTR adj6_q = adj[6] + q * nelements + evb;
      s_t *const RSTR badj7_q = &badj7[q * VS];
      const s_t *const RSTR adj7_q = adj[7] + q * nelements + evb;
      s_t *const RSTR badj8_q = &badj8[q * VS];
      const s_t *const RSTR adj8_q = adj[8] + q * nelements + evb;
      s_t *const RSTR bdet0_q = &bdet0[q * VS];
      const s_t *const RSTR det_q = det + q * nelements + evb;
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
      laplace_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bh_streams, bout_streams);
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
static __host__ __device__ __forceinline__ int laplace_tet10_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 1;
  static constexpr int ND = 3;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
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
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
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
      laplace_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bh_streams, bout_streams);
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
static __host__ __device__ __forceinline__ int laplace_tet10_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 1;
  static constexpr int ND = 3;
  static constexpr int NS = 10;
  static constexpr int NQ = 11;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinate_data[NDOFS][VS];
    for (int stream = 0; stream < NDOFS; ++stream) {
      {
        bcoordinate_data[stream][0] = coords[stream][evb + 0];
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
    const s_t *const grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
    const s_t *const grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
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
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        const s_t g2 = grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
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
      laplace_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y(), sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z(), sfem::codegen::quad_tet_q11<s_t>::q_weight(), kappa, bh_streams, bout_streams);
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
