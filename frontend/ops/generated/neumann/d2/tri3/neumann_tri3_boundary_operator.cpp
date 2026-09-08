#include "sfem_base.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct neumann_tri3_edgeshell2_boundary_residual_soa_reference_data {
    static constexpr int NS = 2;
    static constexpr int NQ = 2;
    static constexpr int REF_DIM = 1;
    static constexpr int PHYSICAL_DIM = 2;

    static const s_t *shape() {
        static const s_t data[4] = {
            s_t(0.78867513459481287),
            s_t(0.21132486540518708),
            s_t(0.21132486540518713),
            s_t(0.78867513459481287)
        };
        return data;
    }

    static const s_t *grad() {
        static const s_t data[4] = {
            s_t(-1),
            s_t(1),
            s_t(-1),
            s_t(1)
        };
        return data;
    }

    static const s_t *weight() {
        static const s_t data[2] = {
            s_t(0.5),
            s_t(0.5)
        };
        return data;
    }
};

template <typename s_t>
static SFEM_INLINE s_t neumann_tri3_edgeshell2_boundary_residual_soa_measure(
        const int q,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {
    const s_t *const grad = neumann_tri3_edgeshell2_boundary_residual_soa_reference_data<s_t>::grad();
    const int n_shape = neumann_tri3_edgeshell2_boundary_residual_soa_reference_data<s_t>::NS;
    s_t dx0 = s_t(0);
    s_t dx1 = s_t(0);
    for (int i = 0; i < n_shape; ++i) {
        const s_t gi = grad[q * n_shape + i];
        const idx_t node = ev[i];
        dx0 += s_t(points[0][node]) * gi;
        dx1 += s_t(points[1][node]) * gi;
    }
    return sqrt(dx0 * dx0 + dx1 * dx1);
}

static SFEM_INLINE const int *neumann_tri3_edgeshell2_boundary_residual_soa_side_nodes() {
    static const int data[6] = {
        0,
        1,
        1,
        2,
        2,
        0
    };
    return data;
}

static SFEM_INLINE void neumann_tri3_edgeshell2_boundary_residual_soa_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {
    const int *const SFEM_RESTRICT side_nodes = neumann_tri3_edgeshell2_boundary_residual_soa_side_nodes();
    constexpr int n_shape = 2;
    for (int i = 0; i < n_shape; ++i) {
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }
}

template <typename s_t>
static SFEM_INLINE void neumann_tri3_edgeshell2_boundary_residual_soa_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points, const s_t t0, const s_t t1,
        s_t element_vector[2][2]) {
    const s_t *const shape = neumann_tri3_edgeshell2_boundary_residual_soa_reference_data<s_t>::shape();
    const s_t *const weight = neumann_tri3_edgeshell2_boundary_residual_soa_reference_data<s_t>::weight();
    const int n_shape = neumann_tri3_edgeshell2_boundary_residual_soa_reference_data<s_t>::NS;
    const int n_qp = neumann_tri3_edgeshell2_boundary_residual_soa_reference_data<s_t>::NQ;

        const s_t coeff0 = -t0;
        const s_t coeff1 = -t1;

    for (int q = 0; q < n_qp; ++q) {
        const s_t dS = neumann_tri3_edgeshell2_boundary_residual_soa_measure<s_t>(q, ev, points);
        const s_t qw = weight[q] * dS;

#pragma omp simd
        for (int i = 0; i < n_shape; ++i) {
            const s_t test = shape[q * n_shape + i] * qw;
                element_vector[0][i] += coeff0 * test;
                element_vector[1][i] += coeff1 * test;
        }
    }
}

template <typename s_t>
static SFEM_INLINE void neumann_tri3_edgeshell2_boundary_residual_soa_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const s_t element_vector[2][2],
        const int out_stride,
        s_t *const SFEM_RESTRICT out0,
        s_t *const SFEM_RESTRICT out1) {
    constexpr int n_shape = 2;
    for (int i = 0; i < n_shape; ++i) {
        const idx_t node = ev[i];
#pragma omp atomic update
            out0[node * out_stride] += element_vector[0][i];
#pragma omp atomic update
            out1[node * out_stride] += element_vector[1][i];
    }
}

template <typename s_t>
static SFEM_INLINE int neumann_tri3_edgeshell2_boundary_residual_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const s_t t0, const s_t t1,
        const int out_stride,
        s_t *const SFEM_RESTRICT out0,
        s_t *const SFEM_RESTRICT out1) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        idx_t ev[2];
        s_t element_vector[2][2];
        for (int i = 0; i < 2; ++i) {
            ev[i] = elements[i][e];
        }
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < 2; ++i) {
                element_vector[c][i] = s_t(0);
            }
        }
        neumann_tri3_edgeshell2_boundary_residual_soa_element<s_t>(ev, points, t0, t1, element_vector);
        neumann_tri3_edgeshell2_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1);
    }

    return SFEM_SUCCESS;
}

template <typename s_t>
static SFEM_INLINE int neumann_tri3_edgeshell2_boundary_residual_sideset_soa_impl(
        const ptrdiff_t nsides,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const s_t t0, const s_t t1,
        const int out_stride,
        s_t *const SFEM_RESTRICT out0,
        s_t *const SFEM_RESTRICT out1) {
#pragma omp parallel for
    for (ptrdiff_t s = 0; s < nsides; ++s) {
        idx_t ev[2];
        s_t element_vector[2][2];
        neumann_tri3_edgeshell2_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < 2; ++i) {
                element_vector[c][i] = s_t(0);
            }
        }
        neumann_tri3_edgeshell2_boundary_residual_soa_element<s_t>(ev, points, t0, t1, element_vector);
        neumann_tri3_edgeshell2_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1);
    }

    return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_tri3_edgeshell2_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_tri3_edgeshell2_boundary_residual_soa_impl<real_t>(
            nelements, nnodes, elements, points, t0, t1, out_stride, out0, out1);
}

extern "C" int neumann_tri3_edgeshell2_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_tri3_edgeshell2_boundary_residual_soa_impl<float>(
            nelements, nnodes, elements, points, t0, t1, out_stride, out0, out1);
}

extern "C" int neumann_tri3_edgeshell2_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_tri3_edgeshell2_boundary_residual_sideset_soa_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
}

extern "C" int neumann_tri3_edgeshell2_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1) {
    return sfem::codegen::neumann_tri3_edgeshell2_boundary_residual_sideset_soa_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
}
