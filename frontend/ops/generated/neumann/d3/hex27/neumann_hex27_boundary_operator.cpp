#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t>
struct neumann_hex27_quadshell9_boundary_residual_soa_reference_data {
    static constexpr int N_SHAPE_1D = 3;
    static constexpr int N_QP_1D = 3;
    static constexpr int N_SHAPE = 9;
    static constexpr int N_QP = 9;
    static constexpr int REF_DIM = 2;
    static constexpr int PHYSICAL_DIM = 3;

    static const scalar_t *shape_1d() {
        static const scalar_t data[9] = {
            scalar_t(0.68729833462074175),
            scalar_t(0.39999999999999997),
            scalar_t(-0.087298334620741685),
            scalar_t(0),
            scalar_t(1),
            scalar_t(0),
            scalar_t(-0.087298334620741658),
            scalar_t(0.39999999999999991),
            scalar_t(0.68729833462074175)
        };
        return data;
    }

    static const scalar_t *grad_1d() {
        static const scalar_t data[9] = {
            scalar_t(-2.5491933384829668),
            scalar_t(3.0983866769659336),
            scalar_t(-0.54919333848296681),
            scalar_t(-1),
            scalar_t(0),
            scalar_t(1),
            scalar_t(0.54919333848296681),
            scalar_t(-3.0983866769659336),
            scalar_t(2.5491933384829668)
        };
        return data;
    }

    static const scalar_t *weight_1d() {
        static const scalar_t data[3] = {
            scalar_t(0.27777777777777779),
            scalar_t(0.44444444444444442),
            scalar_t(0.27777777777777779)
        };
        return data;
    }

    static const int *shape_index() {
        static const int data[9] = {
        0,
        4,
        1,
        7,
        8,
        5,
        3,
        6,
        2
        };
        return data;
    }
};

template <typename scalar_t>
static SFEM_INLINE scalar_t neumann_hex27_quadshell9_boundary_residual_soa_measure(
        const int qx,
        const int qy,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {
    const scalar_t *const SFEM_RESTRICT shape_1d = neumann_hex27_quadshell9_boundary_residual_soa_reference_data<scalar_t>::shape_1d();
    const scalar_t *const SFEM_RESTRICT grad_1d = neumann_hex27_quadshell9_boundary_residual_soa_reference_data<scalar_t>::grad_1d();
    const int *const SFEM_RESTRICT shape_index = neumann_hex27_quadshell9_boundary_residual_soa_reference_data<scalar_t>::shape_index();
    constexpr int S = 3;
    scalar_t dxdr0 = scalar_t(0);
    scalar_t dxdr1 = scalar_t(0);
    scalar_t dxdr2 = scalar_t(0);
    scalar_t dxds0 = scalar_t(0);
    scalar_t dxds1 = scalar_t(0);
    scalar_t dxds2 = scalar_t(0);
    for (int sy = 0; sy < S; ++sy) {
        const scalar_t vy = shape_1d[qy * S + sy];
        const scalar_t gy = grad_1d[qy * S + sy];
        for (int sx = 0; sx < S; ++sx) {
            const int i = shape_index[sy * S + sx];
            const idx_t node = ev[i];
            const scalar_t vx = shape_1d[qx * S + sx];
            const scalar_t gx = grad_1d[qx * S + sx];
            const scalar_t gr = gx * vy;
            const scalar_t gs = vx * gy;
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
    }
    const scalar_t c0 = dxdr1 * dxds2 - dxdr2 * dxds1;
    const scalar_t c1 = dxdr2 * dxds0 - dxdr0 * dxds2;
    const scalar_t c2 = dxdr0 * dxds1 - dxdr1 * dxds0;
    return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}

static SFEM_INLINE const int *neumann_hex27_quadshell9_boundary_residual_soa_side_nodes() {
    static const int data[54] = {
        0,
        1,
        5,
        4,
        8,
        17,
        12,
        16,
        20,
        1,
        2,
        6,
        5,
        9,
        18,
        13,
        17,
        21,
        2,
        3,
        7,
        6,
        10,
        19,
        14,
        18,
        22,
        3,
        0,
        4,
        7,
        11,
        16,
        15,
        19,
        23,
        3,
        2,
        1,
        0,
        10,
        9,
        8,
        11,
        24,
        4,
        5,
        6,
        7,
        12,
        13,
        14,
        15,
        25
    };
    return data;
}

static SFEM_INLINE void neumann_hex27_quadshell9_boundary_residual_soa_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {
    const int *const SFEM_RESTRICT side_nodes = neumann_hex27_quadshell9_boundary_residual_soa_side_nodes();
    constexpr int n_shape = 9;
    for (int i = 0; i < n_shape; ++i) {
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_hex27_quadshell9_boundary_residual_soa_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1, const scalar_t t2,
        scalar_t element_vector[3][9]) {
    const scalar_t *const SFEM_RESTRICT shape_1d = neumann_hex27_quadshell9_boundary_residual_soa_reference_data<scalar_t>::shape_1d();
    const scalar_t *const SFEM_RESTRICT weight_1d = neumann_hex27_quadshell9_boundary_residual_soa_reference_data<scalar_t>::weight_1d();
    const int *const SFEM_RESTRICT shape_index = neumann_hex27_quadshell9_boundary_residual_soa_reference_data<scalar_t>::shape_index();
    constexpr int S = 3;
    constexpr int Q = 3;

    const scalar_t coeff0 = -t0;
    const scalar_t coeff1 = -t1;
    const scalar_t coeff2 = -t2;

    for (int qy = 0; qy < Q; ++qy) {
        for (int qx = 0; qx < Q; ++qx) {
            const scalar_t dS = neumann_hex27_quadshell9_boundary_residual_soa_measure<scalar_t>(qx, qy, ev, points);
            const scalar_t qw = weight_1d[qx] * weight_1d[qy] * dS;

            for (int sy = 0; sy < S; ++sy) {
                const scalar_t vy = shape_1d[qy * S + sy];
                for (int sx = 0; sx < S; ++sx) {
                    const int i = shape_index[sy * S + sx];
                    const scalar_t test = shape_1d[qx * S + sx] * vy * qw;
                    element_vector[0][i] += coeff0 * test;
                    element_vector[1][i] += coeff1 * test;
                    element_vector[2][i] += coeff2 * test;
                }
            }
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_hex27_quadshell9_boundary_residual_soa_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t element_vector[3][9],
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
    constexpr int n_shape = 9;
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
static SFEM_INLINE int neumann_hex27_quadshell9_boundary_residual_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1, const scalar_t t2,
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        idx_t ev[9];
        scalar_t element_vector[3][9];
        for (int i = 0; i < 9; ++i) {
            ev[i] = elements[i][e];
        }
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 9; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_hex27_quadshell9_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, t2, element_vector);
        neumann_hex27_quadshell9_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neumann_hex27_quadshell9_boundary_residual_sideset_soa_impl(
        const ptrdiff_t nsides,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1, const scalar_t t2,
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
#pragma omp parallel for
    for (ptrdiff_t s = 0; s < nsides; ++s) {
        idx_t ev[9];
        scalar_t element_vector[3][9];
        neumann_hex27_quadshell9_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 9; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_hex27_quadshell9_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, t2, element_vector);
        neumann_hex27_quadshell9_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_hex27_quadshell9_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_hex27_quadshell9_boundary_residual_soa_impl<real_t>(
            nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_hex27_quadshell9_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_hex27_quadshell9_boundary_residual_soa_impl<float>(
            nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_hex27_quadshell9_boundary_residual_sideset_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_hex27_quadshell9_boundary_residual_sideset_soa_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_hex27_quadshell9_boundary_residual_sideset_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_hex27_quadshell9_boundary_residual_sideset_soa_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}
