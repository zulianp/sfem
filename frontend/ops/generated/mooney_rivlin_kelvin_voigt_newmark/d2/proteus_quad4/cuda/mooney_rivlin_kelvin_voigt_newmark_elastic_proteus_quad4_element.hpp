#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_PROTEUS_QUAD4_ELEMENT_API_CUH
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_PROTEUS_QUAD4_ELEMENT_API_CUH

#include <stddef.h>
#include "../../cuda/mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../reference/cuda/line_p1_q2.hpp"
#include "../../../../reference/cuda/quad_line_q2.hpp"

namespace sfem {
namespace codegen {


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
      s_t *const RSTR bdet0_q = &bdet0[q * VS];
      const s_t *const RSTR det_q = det + q * nelements + evb;
      {
        badj0_q[0] = adj0_q[0];
        badj1_q[0] = adj1_q[0];
        badj2_q[0] = adj2_q[0];
        badj3_q[0] = adj3_q[0];
        bdet0_q[0] = det_q[0];
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_energy_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_energy_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
      s_t *const RSTR bdet0_q = &bdet0[q * VS];
      const s_t *const RSTR det_q = det + q * nelements + evb;
      {
        badj0_q[0] = adj0_q[0];
        badj1_q[0] = adj1_q[0];
        badj2_q[0] = adj2_q[0];
        badj3_q[0] = adj3_q[0];
        bdet0_q[0] = det_q[0];
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
  static constexpr int NDOFS = NC * NS;
  if (nelements <= 0) return SFEM_SUCCESS;
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    const s_t *bu_streams[NDOFS];
    for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
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
      s_t *const RSTR bdet0_q = &bdet0[q * VS];
      const s_t *const RSTR det_q = det + q * nelements + evb;
      {
        badj0_q[0] = adj0_q[0];
        badj1_q[0] = adj1_q[0];
        badj2_q[0] = adj2_q[0];
        badj3_q[0] = adj3_q[0];
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
      mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
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
      mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static __host__ __device__ __forceinline__ int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 4;
  static constexpr int NQ = 4;
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
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
      mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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
