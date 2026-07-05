#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t>
struct neumann_quad4_edgeshell2_boundary_residual_soa_reference_data {
    static constexpr int N_SHAPE = 2;
    static constexpr int N_QP = 2;
    static constexpr int REF_DIM = 1;
    static constexpr int PHYSICAL_DIM = 2;

    static const scalar_t *shape() {
        static const scalar_t data[4] = {
            scalar_t(0.78867513459481287),
            scalar_t(0.21132486540518708),
            scalar_t(0.21132486540518713),
            scalar_t(0.78867513459481287)
        };
        return data;
    }

    static const scalar_t *grad() {
        static const scalar_t data[4] = {
            scalar_t(-1),
            scalar_t(1),
            scalar_t(-1),
            scalar_t(1)
        };
        return data;
    }

    static const scalar_t *weight() {
        static const scalar_t data[2] = {
            scalar_t(0.5),
            scalar_t(0.5)
        };
        return data;
    }
};

template <typename scalar_t>
static SFEM_INLINE scalar_t neumann_quad4_edgeshell2_boundary_residual_soa_measure(
        const int q,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {
    const scalar_t *const grad = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<scalar_t>::grad();
    const int n_shape = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<scalar_t>::N_SHAPE;
    scalar_t dx0 = scalar_t(0);
    scalar_t dx1 = scalar_t(0);
    for (int i = 0; i < n_shape; ++i) {
        const scalar_t gi = grad[q * n_shape + i];
        const idx_t node = ev[i];
        dx0 += scalar_t(points[0][node]) * gi;
        dx1 += scalar_t(points[1][node]) * gi;
    }
    return sqrt(dx0 * dx0 + dx1 * dx1);
}

static SFEM_INLINE const int *neumann_quad4_edgeshell2_boundary_residual_soa_side_nodes() {
    static const int data[8] = {
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0
    };
    return data;
}

static SFEM_INLINE void neumann_quad4_edgeshell2_boundary_residual_soa_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {
    const int *const SFEM_RESTRICT side_nodes = neumann_quad4_edgeshell2_boundary_residual_soa_side_nodes();
    constexpr int n_shape = 2;
    for (int i = 0; i < n_shape; ++i) {
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_quad4_edgeshell2_boundary_residual_soa_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1,
        scalar_t element_vector[2][2]) {
    const scalar_t *const shape = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<scalar_t>::shape();
    const scalar_t *const weight = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<scalar_t>::weight();
    const int n_shape = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<scalar_t>::N_SHAPE;
    const int n_qp = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<scalar_t>::N_QP;

        const scalar_t coeff0 = -t0;
        const scalar_t coeff1 = -t1;

    for (int q = 0; q < n_qp; ++q) {
        const scalar_t dS = neumann_quad4_edgeshell2_boundary_residual_soa_measure<scalar_t>(q, ev, points);
        const scalar_t qw = weight[q] * dS;

#pragma omp simd
        for (int i = 0; i < n_shape; ++i) {
            const scalar_t test = shape[q * n_shape + i] * qw;
                element_vector[0][i] += coeff0 * test;
                element_vector[1][i] += coeff1 * test;
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_quad4_edgeshell2_boundary_residual_soa_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t element_vector[2][2],
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1) {
    constexpr int n_shape = 2;
    for (int i = 0; i < n_shape; ++i) {
        const idx_t node = ev[i];
#pragma omp atomic update
            out0[node * out_stride] += element_vector[0][i];
#pragma omp atomic update
            out1[node * out_stride] += element_vector[1][i];
    }
}

template <typename scalar_t>
static SFEM_INLINE int neumann_quad4_edgeshell2_boundary_residual_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1,
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        idx_t ev[2];
        scalar_t element_vector[2][2];
        for (int i = 0; i < 2; ++i) {
            ev[i] = elements[i][e];
        }
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < 2; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_quad4_edgeshell2_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, element_vector);
        neumann_quad4_edgeshell2_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1);
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neumann_quad4_edgeshell2_boundary_residual_sideset_soa_impl(
        const ptrdiff_t nsides,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1,
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1) {
#pragma omp parallel for
    for (ptrdiff_t s = 0; s < nsides; ++s) {
        idx_t ev[2];
        scalar_t element_vector[2][2];
        neumann_quad4_edgeshell2_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < 2; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_quad4_edgeshell2_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, element_vector);
        neumann_quad4_edgeshell2_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1);
    }

    return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_quad4_edgeshell2_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_soa_impl<real_t>(
            nelements, nnodes, elements, points, t0, t1, out_stride, out0, out1);
}

extern "C" int neumann_quad4_edgeshell2_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_soa_impl<float>(
            nelements, nnodes, elements, points, t0, t1, out_stride, out0, out1);
}

extern "C" int neumann_quad4_edgeshell2_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_sideset_soa_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
}

extern "C" int neumann_quad4_edgeshell2_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_sideset_soa_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
}
