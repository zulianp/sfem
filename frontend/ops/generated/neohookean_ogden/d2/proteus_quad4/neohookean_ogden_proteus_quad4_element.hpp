#ifndef NEOHOOKEAN_OGDEN_PROTEUS_QUAD4_ELEMENT_API_HPP
#define NEOHOOKEAN_OGDEN_PROTEUS_QUAD4_ELEMENT_API_HPP

#include <stddef.h>
#include "../neohookean_ogden_d2_tensor_product_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../reference/line_p1_q2.hpp"
#include "../../../reference/quad_line_q2.hpp"

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


template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      bvalue[lane] = s_t(0);
    }
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0[q * VS + lane] = adj[0][q * nelements + evb + lane];
        badj1[q * VS + lane] = adj[1][q * nelements + evb + lane];
        badj2[q * VS + lane] = adj[2][q * nelements + evb + lane];
        badj3[q * VS + lane] = adj[3][q * nelements + evb + lane];
        bdet0[q * VS + lane] = det[q * nelements + evb + lane];
      }
    }
    neohookean_ogden_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_energy_ecoords_soa(
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    neohookean_ogden_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_energy_esoa(
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    neohookean_ogden_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bvalue);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bout_streams[stream][lane] = s_t(0);
      }
    }
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0[q * VS + lane] = adj[0][q * nelements + evb + lane];
        badj1[q * VS + lane] = adj[1][q * nelements + evb + lane];
        badj2[q * VS + lane] = adj[2][q * nelements + evb + lane];
        badj3[q * VS + lane] = adj[3][q * nelements + evb + lane];
        bdet0[q * VS + lane] = det[q * nelements + evb + lane];
      }
    }
    neohookean_ogden_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_gradient_ecoords_soa(
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    neohookean_ogden_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_gradient_esoa(
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
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);
    neohookean_ogden_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0[q * VS + lane] = adj[0][q * nelements + evb + lane];
        badj1[q * VS + lane] = adj[1][q * nelements + evb + lane];
        badj2[q * VS + lane] = adj[2][q * nelements + evb + lane];
        badj3[q * VS + lane] = adj[3][q * nelements + evb + lane];
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
      neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_hessian_ecoords_soa(
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][lane] = s_t(0);
        }
      }
      neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_hessian_esoa(
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        bcoordinate_data[stream][lane] = coords[stream][evb + lane];
      }
    }
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(ne, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), bcoordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
          bout_data[stream][lane] = s_t(0);
        }
      }
      neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_line_p1_q2<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q2<s_t>::grad_1d(), sfem::codegen::quad_line_q2<s_t>::q_weight_1d(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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
