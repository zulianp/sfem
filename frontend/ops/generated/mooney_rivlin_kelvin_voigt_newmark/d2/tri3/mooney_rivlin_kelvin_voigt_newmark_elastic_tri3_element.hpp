#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_TRI3_ELEMENT_API_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_TRI3_ELEMENT_API_HPP

#include <stddef.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../reference/quad_tri_q1.hpp"
#include "../../../reference/tri3_q1.hpp"

namespace sfem {
namespace codegen {


template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 2;
  static constexpr int NS = 3;
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    {
        const int q = 0;  // TRI3 evaluates in closed form
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0_q[lane] = adj0_q[lane];
        badj1_q[lane] = adj1_q[lane];
        badj2_q[lane] = adj2_q[lane];
        badj3_q[lane] = adj3_q[lane];
        bdet0_q[lane] = det_q[lane];
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bu_streams, 1, &objective_step, 0, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_energy_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 3;
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    const s_t *const grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
    {
        const int q = 0;  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
        J01_values[lane] = s_t(0);
        J10_values[lane] = s_t(0);
        J11_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[2 * shape][lane] * g0;
          J01_values[lane] += bcoordinate_data[2 * shape][lane] * g1;
          J10_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g0;
          J11_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q * VS + lane);
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bu_streams, 1, &objective_step, 0, bvalue);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_energy_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NS = 3;
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    const s_t *const grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
    {
        const int q = 0;  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
        J01_values[lane] = s_t(0);
        J10_values[lane] = s_t(0);
        J11_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[2 * shape][lane] * g0;
          J01_values[lane] += bcoordinate_data[2 * shape][lane] * g1;
          J10_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g0;
          J11_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q * VS + lane);
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bu_streams, 1, &objective_step, 0, bvalue);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_egeometry_soa(
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
    s_t bdet0[NQ * VS];
    {
        const int q = 0;  // TRI3 evaluates in closed form
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
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        badj0_q[lane] = adj0_q[lane];
        badj1_q[lane] = adj1_q[lane];
        badj2_q[lane] = adj2_q[lane];
        badj3_q[lane] = adj3_q[lane];
        bdet0_q[lane] = det_q[lane];
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_ecoords_soa(
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
    s_t bdet0[NQ * VS];
    const s_t *const grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
    {
        const int q = 0;  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
        J01_values[lane] = s_t(0);
        J10_values[lane] = s_t(0);
        J11_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[2 * shape][lane] * g0;
          J01_values[lane] += bcoordinate_data[2 * shape][lane] * g1;
          J10_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g0;
          J11_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q * VS + lane);
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}

template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_esoa(
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
    s_t bdet0[NQ * VS];
    const s_t *const grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
    {
        const int q = 0;  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
        J01_values[lane] = s_t(0);
        J10_values[lane] = s_t(0);
        J11_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[2 * shape][lane] * g0;
          J01_values[lane] += bcoordinate_data[2 * shape][lane] * g1;
          J10_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g0;
          J11_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q * VS + lane);
      }
    }
    mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
  }
  return SFEM_SUCCESS;
}


template <typename s_t, int VS>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_egeometry_soa(
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
  static constexpr int NQ = 1;
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
    {
        const int q = 0;  // TRI3 evaluates in closed form
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
      mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_ecoords_soa(
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    const s_t *const grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
    {
        const int q = 0;  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
        J01_values[lane] = s_t(0);
        J10_values[lane] = s_t(0);
        J11_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[2 * shape][lane] * g0;
          J01_values[lane] += bcoordinate_data[2 * shape][lane] * g1;
          J10_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g0;
          J11_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q * VS + lane);
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
      mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_esoa(
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
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t bdet0[NQ * VS];
    const s_t *const grad_ref_x = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x();
    const s_t *const grad_ref_y = sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y();
    {
        const int q = 0;  // TRI3 evaluates in closed form
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        J00_values[lane] = s_t(0);
        J01_values[lane] = s_t(0);
        J10_values[lane] = s_t(0);
        J11_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = grad_ref_x[q * NS + shape];
        const s_t g1 = grad_ref_y[q * NS + shape];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          J00_values[lane] += bcoordinate_data[2 * shape][lane] * g0;
          J01_values[lane] += bcoordinate_data[2 * shape][lane] * g1;
          J10_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g0;
          J11_values[lane] += bcoordinate_data[2 * shape + 1][lane] * g1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = J00_values[lane];
        const s_t J01 = J01_values[lane];
        const s_t J10 = J10_values[lane];
        const s_t J11 = J11_values[lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q * VS + lane);
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
      mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, sfem::codegen::ref_tri3_q1<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q1<s_t>::grad_ref_y(), sfem::codegen::quad_tri_q1<s_t>::q_weight(), lmbda, mu, bu_streams, bh_streams, bout_streams);
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
