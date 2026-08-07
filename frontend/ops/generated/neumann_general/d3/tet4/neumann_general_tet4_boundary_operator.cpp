#include "sfem_base.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t>
struct neumann_general_tet4_trishell3_boundary_residual_soa_reference_data {
    static constexpr int N_SHAPE = 3;
    static constexpr int N_QP = 3;
    static constexpr int REF_DIM = 2;
    static constexpr int PHYSICAL_DIM = 3;

    static const scalar_t *shape() {
        static const scalar_t data[9] = {
            scalar_t(0.66666666666666674),
            scalar_t(0.16666666666666666),
            scalar_t(0.16666666666666666),
            scalar_t(0.16666666666666671),
            scalar_t(0.66666666666666663),
            scalar_t(0.16666666666666666),
            scalar_t(0.16666666666666674),
            scalar_t(0.16666666666666666),
            scalar_t(0.66666666666666663)
        };
        return data;
    }

    static const scalar_t *grad() {
        static const scalar_t data[18] = {
            scalar_t(-1),
            scalar_t(-1),
            scalar_t(1),
            scalar_t(0),
            scalar_t(0),
            scalar_t(1),
            scalar_t(-1),
            scalar_t(-1),
            scalar_t(1),
            scalar_t(0),
            scalar_t(0),
            scalar_t(1),
            scalar_t(-1),
            scalar_t(-1),
            scalar_t(1),
            scalar_t(0),
            scalar_t(0),
            scalar_t(1)
        };
        return data;
    }

    static const scalar_t *weight() {
        static const scalar_t data[3] = {
            scalar_t(0.16666666666666666),
            scalar_t(0.16666666666666666),
            scalar_t(0.16666666666666666)
        };
        return data;
    }
};

template <typename scalar_t>
static SFEM_INLINE scalar_t neumann_general_tet4_trishell3_boundary_residual_soa_measure(
        const int q,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {
    const scalar_t *const grad = neumann_general_tet4_trishell3_boundary_residual_soa_reference_data<scalar_t>::grad();
    const int n_shape = neumann_general_tet4_trishell3_boundary_residual_soa_reference_data<scalar_t>::N_SHAPE;
    scalar_t dxdr0 = scalar_t(0);
    scalar_t dxdr1 = scalar_t(0);
    scalar_t dxdr2 = scalar_t(0);
    scalar_t dxds0 = scalar_t(0);
    scalar_t dxds1 = scalar_t(0);
    scalar_t dxds2 = scalar_t(0);
    for (int i = 0; i < n_shape; ++i) {
        const scalar_t gr = grad[(q * n_shape + i) * 2 + 0];
        const scalar_t gs = grad[(q * n_shape + i) * 2 + 1];
        const idx_t node = ev[i];
        const scalar_t x = scalar_t(points[0][node]);
        const scalar_t y = scalar_t(points[1][node]);
        const scalar_t z = scalar_t(points[2][node]);
        dxdr0 += x * gr;
        dxdr1 += y * gr;
        dxdr2 += z * gr;
        dxds0 += x * gs;
        dxds1 += y * gs;
        dxds2 += z * gs;
    }
    const scalar_t c0 = dxdr1 * dxds2 - dxdr2 * dxds1;
    const scalar_t c1 = dxdr2 * dxds0 - dxdr0 * dxds2;
    const scalar_t c2 = dxdr0 * dxds1 - dxdr1 * dxds0;
    return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}

static SFEM_INLINE const int *neumann_general_tet4_trishell3_boundary_residual_soa_side_nodes() {
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

static SFEM_INLINE void neumann_general_tet4_trishell3_boundary_residual_soa_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {
    const int *const SFEM_RESTRICT side_nodes = neumann_general_tet4_trishell3_boundary_residual_soa_side_nodes();
    constexpr int n_shape = 3;
    for (int i = 0; i < n_shape; ++i) {
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_general_tet4_trishell3_boundary_residual_soa_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t0_001, const scalar_t t0_010, const scalar_t t0_100, const scalar_t t1, const scalar_t t1_001, const scalar_t t1_010, const scalar_t t1_100, const scalar_t t2, const scalar_t t2_001, const scalar_t t2_010, const scalar_t t2_100,
        scalar_t element_vector[3][3]) {
    const scalar_t *const shape = neumann_general_tet4_trishell3_boundary_residual_soa_reference_data<scalar_t>::shape();
    const scalar_t *const weight = neumann_general_tet4_trishell3_boundary_residual_soa_reference_data<scalar_t>::weight();
    const int n_shape = neumann_general_tet4_trishell3_boundary_residual_soa_reference_data<scalar_t>::N_SHAPE;
    const int n_qp = neumann_general_tet4_trishell3_boundary_residual_soa_reference_data<scalar_t>::N_QP;



    for (int q = 0; q < n_qp; ++q) {
        const scalar_t dS = neumann_general_tet4_trishell3_boundary_residual_soa_measure<scalar_t>(q, ev, points);
        const scalar_t qw = weight[q] * dS;
        scalar_t x0 = scalar_t(0);
        scalar_t x1 = scalar_t(0);
        scalar_t x2 = scalar_t(0);
        for (int j = 0; j < n_shape; ++j) {
            const scalar_t phi = shape[q * n_shape + j];
            const idx_t node = ev[j];
            x0 += scalar_t(points[0][node]) * phi;
            x1 += scalar_t(points[1][node]) * phi;
            x2 += scalar_t(points[2][node]) * phi;
        }
        const scalar_t coeff0 = t0 + t0_001*x2 + t0_010*x1 + t0_100*x0;
        const scalar_t coeff1 = t1 + t1_001*x2 + t1_010*x1 + t1_100*x0;
        const scalar_t coeff2 = t2 + t2_001*x2 + t2_010*x1 + t2_100*x0;
#pragma omp simd
        for (int i = 0; i < n_shape; ++i) {
            const scalar_t test = shape[q * n_shape + i] * qw;
                element_vector[0][i] += coeff0 * test;
                element_vector[1][i] += coeff1 * test;
                element_vector[2][i] += coeff2 * test;
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_general_tet4_trishell3_boundary_residual_soa_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t element_vector[3][3],
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
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

template <typename scalar_t>
static SFEM_INLINE int neumann_general_tet4_trishell3_boundary_residual_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t0_001, const scalar_t t0_010, const scalar_t t0_100, const scalar_t t1, const scalar_t t1_001, const scalar_t t1_010, const scalar_t t1_100, const scalar_t t2, const scalar_t t2_001, const scalar_t t2_010, const scalar_t t2_100,
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        idx_t ev[3];
        scalar_t element_vector[3][3];
        for (int i = 0; i < 3; ++i) {
            ev[i] = elements[i][e];
        }
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 3; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_general_tet4_trishell3_boundary_residual_soa_element<scalar_t>(ev, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, element_vector);
        neumann_general_tet4_trishell3_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neumann_general_tet4_trishell3_boundary_residual_sideset_soa_impl(
        const ptrdiff_t nsides,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t0_001, const scalar_t t0_010, const scalar_t t0_100, const scalar_t t1, const scalar_t t1_001, const scalar_t t1_010, const scalar_t t1_100, const scalar_t t2, const scalar_t t2_001, const scalar_t t2_010, const scalar_t t2_100,
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
#pragma omp parallel for
    for (ptrdiff_t s = 0; s < nsides; ++s) {
        idx_t ev[3];
        scalar_t element_vector[3][3];
        neumann_general_tet4_trishell3_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 3; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_general_tet4_trishell3_boundary_residual_soa_element<scalar_t>(ev, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, element_vector);
        neumann_general_tet4_trishell3_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_general_tet4_trishell3_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t0_001, const real_t t0_010, const real_t t0_100, const real_t t1, const real_t t1_001, const real_t t1_010, const real_t t1_100, const real_t t2, const real_t t2_001, const real_t t2_010, const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_general_tet4_trishell3_boundary_residual_soa_impl<real_t>(
            nelements, nnodes, elements, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}

extern "C" int neumann_general_tet4_trishell3_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t0_001, const float t0_010, const float t0_100, const float t1, const float t1_001, const float t1_010, const float t1_100, const float t2, const float t2_001, const float t2_010, const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_general_tet4_trishell3_boundary_residual_soa_impl<float>(
            nelements, nnodes, elements, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}

extern "C" int neumann_general_tet4_trishell3_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t0_001, const real_t t0_010, const real_t t0_100, const real_t t1, const real_t t1_001, const real_t t1_010, const real_t t1_100, const real_t t2, const real_t t2_001, const real_t t2_010, const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_general_tet4_trishell3_boundary_residual_sideset_soa_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}

extern "C" int neumann_general_tet4_trishell3_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t0_001, const float t0_010, const float t0_100, const float t1, const float t1_001, const float t1_010, const float t1_100, const float t2, const float t2_001, const float t2_010, const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_general_tet4_trishell3_boundary_residual_sideset_soa_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}
