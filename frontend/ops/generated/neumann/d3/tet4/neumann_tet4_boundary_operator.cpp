#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#if defined(__has_include)
#if __has_include("sfem_macros.hpp")
#include "sfem_macros.hpp"
#endif
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#include <math.h>
#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct neumann_tet4_trishell3_boundary_residual_soa_reference_data {
  static constexpr int NS = 3;

};

template <typename s_t>
static SFEM_INLINE s_t neumann_tet4_trishell3_boundary_residual_soa_measure(
    const idx_t *const RSTR ev,
    const geom_t *const *const RSTR points) {
  const s_t dxdr0 = s_t(points[0][ev[1]]) - s_t(points[0][ev[0]]);
  const s_t dxdr1 = s_t(points[1][ev[1]]) - s_t(points[1][ev[0]]);
  const s_t dxdr2 = s_t(points[2][ev[1]]) - s_t(points[2][ev[0]]);
  const s_t dxds0 = s_t(points[0][ev[2]]) - s_t(points[0][ev[0]]);
  const s_t dxds1 = s_t(points[1][ev[2]]) - s_t(points[1][ev[0]]);
  const s_t dxds2 = s_t(points[2][ev[2]]) - s_t(points[2][ev[0]]);
  const s_t c0 = dxdr1 * dxds2 - dxdr2 * dxds1;
  const s_t c1 = dxdr2 * dxds0 - dxdr0 * dxds2;
  const s_t c2 = dxdr0 * dxds1 - dxdr1 * dxds0;
  return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}

static SFEM_INLINE const int *neumann_tet4_trishell3_boundary_residual_soa_side_nodes() {
  static const int data[12] = {
    0,
    1,
    3,
    1,
    2,
    3,
    0,
    3,
    2,
    0,
    2,
    1
  };
  return data;
}

static SFEM_INLINE void neumann_tet4_trishell3_boundary_residual_soa_gather_sideset_element(
    const element_idx_t parent_element,
    const int side,
    idx_t **const RSTR elements,
    idx_t *const RSTR ev) {
  const int *const RSTR side_nodes = neumann_tet4_trishell3_boundary_residual_soa_side_nodes();
  constexpr int n_shape = 3;
  for (int i = 0; i < n_shape; ++i) {
    ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
  }
}

template <typename s_t>
static SFEM_INLINE void neumann_tet4_trishell3_boundary_residual_soa_element(
    const idx_t *const RSTR ev,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t1, const s_t t2,
    s_t element_vector[3][3]) {
  const int n_shape = neumann_tet4_trishell3_boundary_residual_soa_reference_data<s_t>::NS;

    const s_t coeff0 = -t0;
    const s_t coeff1 = -t1;
    const s_t coeff2 = -t2;

  {
    const s_t test = (s_t(1) / s_t(6)) * neumann_tet4_trishell3_boundary_residual_soa_measure<s_t>(ev, points);
#pragma omp simd
    for (int i = 0; i < n_shape; ++i) {
        element_vector[0][i] += coeff0 * test;
        element_vector[1][i] += coeff1 * test;
        element_vector[2][i] += coeff2 * test;
    }
  }
}

template <typename s_t>
static SFEM_INLINE void neumann_tet4_trishell3_boundary_residual_soa_scatter_element(
    const idx_t *const RSTR ev,
    const s_t element_vector[3][3],
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1,
    s_t *const RSTR out2) {
  constexpr int n_shape = 3;
  for (int i = 0; i < n_shape; ++i) {
    const idx_t node = ev[i];
#pragma omp atomic update
            out0[node * out_stride] += element_vector[0][i];
#pragma omp atomic update
            out1[node * out_stride] += element_vector[1][i];
#pragma omp atomic update
            out2[node * out_stride] += element_vector[2][i];
  }
}

template <typename s_t>
static SFEM_INLINE int neumann_tet4_trishell3_boundary_residual_soa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t1, const s_t t2,
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1,
    s_t *const RSTR out2) {
#pragma omp parallel for
  for (ptrdiff_t e = 0; e < nelements; ++e) {
    idx_t ev[3];
    s_t element_vector[3][3];
    for (int i = 0; i < 3; ++i) {
      ev[i] = elements[i][e];
    }
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < 3; ++i) {
        element_vector[c][i] = s_t(0);
      }
    }
    neumann_tet4_trishell3_boundary_residual_soa_element<s_t>(ev, points, t0, t1, t2, element_vector);
    neumann_tet4_trishell3_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1, out2);
  }

  return SFEM_SUCCESS;
}

template <typename s_t>
static SFEM_INLINE int neumann_tet4_trishell3_boundary_residual_ss_soa_impl(
    const ptrdiff_t nsides,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t1, const s_t t2,
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1,
    s_t *const RSTR out2) {
#pragma omp parallel for
  for (ptrdiff_t s = 0; s < nsides; ++s) {
    idx_t ev[3];
    s_t element_vector[3][3];
    neumann_tet4_trishell3_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < 3; ++i) {
        element_vector[c][i] = s_t(0);
      }
    }
    neumann_tet4_trishell3_boundary_residual_soa_element<s_t>(ev, points, t0, t1, t2, element_vector);
    neumann_tet4_trishell3_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1, out2);
  }

  return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_tet4_trishell3_boundary_residual_soa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1, const real_t t2,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2) {
  return sfem::codegen::neumann_tet4_trishell3_boundary_residual_soa_impl<real_t>(
      nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_tet4_trishell3_boundary_residual_soa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const float t0, const float t1, const float t2,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2) {
  return sfem::codegen::neumann_tet4_trishell3_boundary_residual_soa_impl<float>(
      nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_tet4_trishell3_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1, const real_t t2,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2) {
  return sfem::codegen::neumann_tet4_trishell3_boundary_residual_ss_soa_impl<real_t>(
      nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_tet4_trishell3_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1, const float t2,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2) {
  return sfem::codegen::neumann_tet4_trishell3_boundary_residual_ss_soa_impl<float>(
      nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}
