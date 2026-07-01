#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t>
struct neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_reference_data {
    static constexpr int N_SHAPE_1D = 5;
    static constexpr int N_QP_1D = 5;
    static constexpr int N_SHAPE = 25;
    static constexpr int N_QP = 25;
    static constexpr int REF_DIM = 2;
    static constexpr int PHYSICAL_DIM = 3;

    static const scalar_t *shape_1d() {
        static const scalar_t data[25] = {
            scalar_t(0.65772788257758819),
            scalar_t(0.60769269466101494),
            scalar_t(-0.40858201526174176),
            scalar_t(0.17553410810741293),
            scalar_t(-0.032372670084274559),
            scalar_t(0.022063103295100275),
            scalar_t(1.0587971821717581),
            scalar_t(-0.11346384011744702),
            scalar_t(0.039222340750583846),
            scalar_t(-0.006618786099995529),
            scalar_t(0),
            scalar_t(-0),
            scalar_t(1),
            scalar_t(0),
            scalar_t(-0),
            scalar_t(-0.006618786099995509),
            scalar_t(0.039222340750583728),
            scalar_t(-0.11346384011744673),
            scalar_t(1.0587971821717581),
            scalar_t(0.022063103295100202),
            scalar_t(-0.03237267008427458),
            scalar_t(0.17553410810741304),
            scalar_t(-0.40858201526174215),
            scalar_t(0.60769269466101572),
            scalar_t(0.65772788257758796)
        };
        return data;
    }

    static const scalar_t *grad_1d() {
        static const scalar_t data[25] = {
            scalar_t(-6.315836427348243),
            scalar_t(10.11127830306696),
            scalar_t(-5.6882551126207392),
            scalar_t(2.3060210254335645),
            scalar_t(-0.41320778853154438),
            scalar_t(-1.3001705560202661),
            scalar_t(-2.759999173503255),
            scalar_t(5.7732663757858971),
            scalar_t(-2.0658530069809662),
            scalar_t(0.35275636071858923),
            scalar_t(0.33333333333333331),
            scalar_t(-2.6666666666666665),
            scalar_t(0),
            scalar_t(2.6666666666666665),
            scalar_t(-0.33333333333333331),
            scalar_t(-0.35275636071858935),
            scalar_t(2.0658530069809657),
            scalar_t(-5.7732663757858997),
            scalar_t(2.7599991735032567),
            scalar_t(1.3001705560202657),
            scalar_t(0.41320778853154383),
            scalar_t(-2.3060210254335618),
            scalar_t(5.6882551126207348),
            scalar_t(-10.111278303066957),
            scalar_t(6.3158364273482412)
        };
        return data;
    }

    static const scalar_t *weight_1d() {
        static const scalar_t data[5] = {
            scalar_t(0.1184634425280947),
            scalar_t(0.23931433524968312),
            scalar_t(0.28444444444444428),
            scalar_t(0.23931433524968312),
            scalar_t(0.1184634425280947)
        };
        return data;
    }

    static const int *shape_index() {
        static const int data[25] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24
        };
        return data;
    }
};

template <typename scalar_t>
static SFEM_INLINE scalar_t neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_measure(
        const int qx,
        const int qy,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {
    const scalar_t *const SFEM_RESTRICT shape_1d = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_reference_data<scalar_t>::shape_1d();
    const scalar_t *const SFEM_RESTRICT grad_1d = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_reference_data<scalar_t>::grad_1d();
    const int *const SFEM_RESTRICT shape_index = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_reference_data<scalar_t>::shape_index();
    constexpr int S = 5;
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

static SFEM_INLINE const int *neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_side_nodes() {
    static const int data[150] = {
        0,
        1,
        2,
        3,
        4,
        25,
        26,
        27,
        28,
        29,
        50,
        51,
        52,
        53,
        54,
        75,
        76,
        77,
        78,
        79,
        100,
        101,
        102,
        103,
        104,
        4,
        9,
        14,
        19,
        24,
        29,
        34,
        39,
        44,
        49,
        54,
        59,
        64,
        69,
        74,
        79,
        84,
        89,
        94,
        99,
        104,
        109,
        114,
        119,
        124,
        24,
        23,
        22,
        21,
        20,
        49,
        48,
        47,
        46,
        45,
        74,
        73,
        72,
        71,
        70,
        99,
        98,
        97,
        96,
        95,
        124,
        123,
        122,
        121,
        120,
        20,
        15,
        10,
        5,
        0,
        45,
        40,
        35,
        30,
        25,
        70,
        65,
        60,
        55,
        50,
        95,
        90,
        85,
        80,
        75,
        120,
        115,
        110,
        105,
        100,
        20,
        21,
        22,
        23,
        24,
        15,
        16,
        17,
        18,
        19,
        10,
        11,
        12,
        13,
        14,
        5,
        6,
        7,
        8,
        9,
        0,
        1,
        2,
        3,
        4,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124
    };
    return data;
}

static SFEM_INLINE void neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {
    const int *const SFEM_RESTRICT side_nodes = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_side_nodes();
    constexpr int n_shape = 25;
    for (int i = 0; i < n_shape; ++i) {
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1, const scalar_t t2,
        scalar_t element_vector[3][25]) {
    const scalar_t *const SFEM_RESTRICT shape_1d = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_reference_data<scalar_t>::shape_1d();
    const scalar_t *const SFEM_RESTRICT weight_1d = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_reference_data<scalar_t>::weight_1d();
    const int *const SFEM_RESTRICT shape_index = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_reference_data<scalar_t>::shape_index();
    constexpr int S = 5;
    constexpr int Q = 5;

    const scalar_t coeff0 = -t0;
    const scalar_t coeff1 = -t1;
    const scalar_t coeff2 = -t2;

    for (int qy = 0; qy < Q; ++qy) {
        for (int qx = 0; qx < Q; ++qx) {
            const scalar_t dS = neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_measure<scalar_t>(qx, qy, ev, points);
            const scalar_t qw = weight_1d[qx] * weight_1d[qy] * dS;

            for (int sy = 0; sy < S; ++sy) {
                const scalar_t vy = shape_1d[qy * S + sy];
#pragma omp simd
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
static SFEM_INLINE void neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t element_vector[3][25],
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
    constexpr int n_shape = 25;
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
static SFEM_INLINE int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_impl(
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
        idx_t ev[25];
        scalar_t element_vector[3][25];
        for (int i = 0; i < 25; ++i) {
            ev[i] = elements[i][e];
        }
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 25; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, t2, element_vector);
        neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa_impl(
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
        idx_t ev[25];
        scalar_t element_vector[3][25];
        neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 25; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, t2, element_vector);
        neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_impl<real_t>(
            nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_proteus_hex125_proteus_quadshell25_boundary_residual_soa_impl<float>(
            nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa(
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
    return sfem::codegen::neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa_float(
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
    return sfem::codegen::neumann_proteus_hex125_proteus_quadshell25_boundary_residual_sideset_soa_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}
