#include "sfem_base.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_reference_data {
    static constexpr int NS1 = 2;
    static constexpr int NQ1 = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int REF_DIM = 2;
    static constexpr int PHYSICAL_DIM = 3;

    static const s_t *shape_1d() {
        static const s_t data[4] = {
            s_t(0.78867513459481287),
            s_t(0.21132486540518708),
            s_t(0.21132486540518713),
            s_t(0.78867513459481287)
        };
        return data;
    }

    static const s_t *grad_1d() {
        static const s_t data[4] = {
            s_t(-1),
            s_t(1),
            s_t(-1),
            s_t(1)
        };
        return data;
    }

    static const s_t *weight_1d() {
        static const s_t data[2] = {
            s_t(0.5),
            s_t(0.5)
        };
        return data;
    }

    static const int *shape_index() {
        static const int data[4] = {
        0,
        1,
        2,
        3
        };
        return data;
    }
};

template <typename s_t>
static SFEM_INLINE s_t neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_measure(
        const int qx,
        const int qy,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {
    const s_t *const SFEM_RESTRICT shape_1d = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_reference_data<s_t>::shape_1d();
    const s_t *const SFEM_RESTRICT grad_1d = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_reference_data<s_t>::grad_1d();
    const int *const SFEM_RESTRICT shape_index = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_reference_data<s_t>::shape_index();
    constexpr int NS1 = 2;
    s_t dxdr0 = s_t(0);
    s_t dxdr1 = s_t(0);
    s_t dxdr2 = s_t(0);
    s_t dxds0 = s_t(0);
    s_t dxds1 = s_t(0);
    s_t dxds2 = s_t(0);
    for (int sy = 0; sy < NS1; ++sy) {
        const s_t vy = shape_1d[qy * NS1 + sy];
        const s_t gy = grad_1d[qy * NS1 + sy];
        for (int sx = 0; sx < NS1; ++sx) {
            const int i = shape_index[sy * NS1 + sx];
            const idx_t node = ev[i];
            const s_t vx = shape_1d[qx * NS1 + sx];
            const s_t gx = grad_1d[qx * NS1 + sx];
            const s_t gr = gx * vy;
            const s_t gs = vx * gy;
            const s_t x = s_t(points[0][node]);
            const s_t y = s_t(points[1][node]);
            const s_t z = s_t(points[2][node]);
            dxdr0 += x * gr;
            dxdr1 += y * gr;
            dxdr2 += z * gr;
            dxds0 += x * gs;
            dxds1 += y * gs;
            dxds2 += z * gs;
        }
    }
    const s_t c0 = dxdr1 * dxds2 - dxdr2 * dxds1;
    const s_t c1 = dxdr2 * dxds0 - dxdr0 * dxds2;
    const s_t c2 = dxdr0 * dxds1 - dxdr1 * dxds0;
    return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}

static SFEM_INLINE const int *neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_side_nodes() {
    static const int data[24] = {
        0,
        1,
        4,
        5,
        1,
        3,
        5,
        7,
        3,
        2,
        7,
        6,
        2,
        0,
        6,
        4,
        2,
        3,
        0,
        1,
        4,
        5,
        6,
        7
    };
    return data;
}

static SFEM_INLINE void neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {
    const int *const SFEM_RESTRICT side_nodes = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_side_nodes();
    constexpr int n_shape = 4;
    for (int i = 0; i < n_shape; ++i) {
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }
}

template <typename s_t>
static SFEM_INLINE void neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points, const s_t t0, const s_t t0_001, const s_t t0_010, const s_t t0_100, const s_t t1, const s_t t1_001, const s_t t1_010, const s_t t1_100, const s_t t2, const s_t t2_001, const s_t t2_010, const s_t t2_100,
        s_t element_vector[3][4]) {
    const s_t *const SFEM_RESTRICT shape_1d = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_reference_data<s_t>::shape_1d();
    const s_t *const SFEM_RESTRICT weight_1d = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_reference_data<s_t>::weight_1d();
    const int *const SFEM_RESTRICT shape_index = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_reference_data<s_t>::shape_index();
    constexpr int NS1 = 2;
    constexpr int NQ1 = 2;



    for (int qy = 0; qy < NQ1; ++qy) {
        for (int qx = 0; qx < NQ1; ++qx) {
            const s_t dS = neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_measure<s_t>(qx, qy, ev, points);
            const s_t qw = weight_1d[qx] * weight_1d[qy] * dS;
            s_t x0 = s_t(0);
            s_t x1 = s_t(0);
            s_t x2 = s_t(0);
            for (int cy = 0; cy < NS1; ++cy) {
                const s_t vy_coord = shape_1d[qy * NS1 + cy];
                for (int cx = 0; cx < NS1; ++cx) {
                    const int j = shape_index[cy * NS1 + cx];
                    const idx_t node = ev[j];
                    const s_t phi = shape_1d[qx * NS1 + cx] * vy_coord;
                    x0 += s_t(points[0][node]) * phi;
                    x1 += s_t(points[1][node]) * phi;
                    x2 += s_t(points[2][node]) * phi;
                }
            }
            const s_t coeff0 = t0 + t0_001*x2 + t0_010*x1 + t0_100*x0;
            const s_t coeff1 = t1 + t1_001*x2 + t1_010*x1 + t1_100*x0;
            const s_t coeff2 = t2 + t2_001*x2 + t2_010*x1 + t2_100*x0;
            for (int sy = 0; sy < NS1; ++sy) {
                const s_t vy = shape_1d[qy * NS1 + sy];
#pragma omp simd
                for (int sx = 0; sx < NS1; ++sx) {
                    const int i = shape_index[sy * NS1 + sx];
                    const s_t test = shape_1d[qx * NS1 + sx] * vy * qw;
                    element_vector[0][i] += coeff0 * test;
                    element_vector[1][i] += coeff1 * test;
                    element_vector[2][i] += coeff2 * test;
                }
            }
        }
    }
}

template <typename s_t>
static SFEM_INLINE void neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const s_t element_vector[3][4],
        const int out_stride,
        s_t *const SFEM_RESTRICT out0,
        s_t *const SFEM_RESTRICT out1,
        s_t *const SFEM_RESTRICT out2) {
    constexpr int n_shape = 4;
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
static SFEM_INLINE int neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const s_t t0, const s_t t0_001, const s_t t0_010, const s_t t0_100, const s_t t1, const s_t t1_001, const s_t t1_010, const s_t t1_100, const s_t t2, const s_t t2_001, const s_t t2_010, const s_t t2_100,
        const int out_stride,
        s_t *const SFEM_RESTRICT out0,
        s_t *const SFEM_RESTRICT out1,
        s_t *const SFEM_RESTRICT out2) {
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        idx_t ev[4];
        s_t element_vector[3][4];
        for (int i = 0; i < 4; ++i) {
            ev[i] = elements[i][e];
        }
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 4; ++i) {
                element_vector[c][i] = s_t(0);
            }
        }
        neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_element<s_t>(ev, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, element_vector);
        neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

template <typename s_t>
static SFEM_INLINE int neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa_impl(
        const ptrdiff_t nsides,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points, const s_t t0, const s_t t0_001, const s_t t0_010, const s_t t0_100, const s_t t1, const s_t t1_001, const s_t t1_010, const s_t t1_100, const s_t t2, const s_t t2_001, const s_t t2_010, const s_t t2_100,
        const int out_stride,
        s_t *const SFEM_RESTRICT out0,
        s_t *const SFEM_RESTRICT out1,
        s_t *const SFEM_RESTRICT out2) {
#pragma omp parallel for
    for (ptrdiff_t s = 0; s < nsides; ++s) {
        idx_t ev[4];
        s_t element_vector[3][4];
        neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 4; ++i) {
                element_vector[c][i] = s_t(0);
            }
        }
        neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_element<s_t>(ev, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, element_vector);
        neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t0_001, const real_t t0_010, const real_t t0_100, const real_t t1, const real_t t1_001, const real_t t1_010, const real_t t1_100, const real_t t2, const real_t t2_001, const real_t t2_010, const real_t t2_100,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_impl<real_t>(
            nelements, nnodes, elements, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}

extern "C" int neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t0_001, const float t0_010, const float t0_100, const float t1, const float t1_001, const float t1_010, const float t1_100, const float t2, const float t2_001, const float t2_010, const float t2_100,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_soa_impl<float>(
            nelements, nnodes, elements, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}

extern "C" int neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa(
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
    return sfem::codegen::neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}

extern "C" int neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa_float(
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
    return sfem::codegen::neumann_general_proteus_hex8_proteus_quadshell4_boundary_residual_sideset_soa_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
}
