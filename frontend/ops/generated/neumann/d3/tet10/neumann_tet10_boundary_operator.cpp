#include "sfem_base.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "../../../kernel_math.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t>
struct neumann_tet10_trishell6_boundary_residual_soa_reference_data {
    static constexpr int N_SHAPE = 6;
    static constexpr int N_QP = 6;
    static constexpr int REF_DIM = 2;
    static constexpr int PHYSICAL_DIM = 3;

    static const scalar_t *shape() {
        static const scalar_t data[36] = {
            scalar_t(-0.084730493093977982),
            scalar_t(-0.04820837781551205),
            scalar_t(-0.04820837781551205),
            scalar_t(0.1928335112620482),
            scalar_t(0.79548022620090564),
            scalar_t(0.1928335112620482),
            scalar_t(-0.04820837781551205),
            scalar_t(-0.084730493093977968),
            scalar_t(-0.04820837781551205),
            scalar_t(0.19283351126204817),
            scalar_t(0.19283351126204817),
            scalar_t(0.79548022620090564),
            scalar_t(-0.04820837781551205),
            scalar_t(-0.04820837781551205),
            scalar_t(-0.084730493093977968),
            scalar_t(0.79548022620090564),
            scalar_t(0.19283351126204817),
            scalar_t(0.19283351126204817),
            scalar_t(0.5176323419876725),
            scalar_t(-0.074803807748196505),
            scalar_t(-0.074803807748196505),
            scalar_t(0.29921523099278602),
            scalar_t(0.03354481152314847),
            scalar_t(0.29921523099278602),
            scalar_t(-0.074803807748196505),
            scalar_t(0.5176323419876725),
            scalar_t(-0.074803807748196505),
            scalar_t(0.29921523099278602),
            scalar_t(0.29921523099278602),
            scalar_t(0.03354481152314847),
            scalar_t(-0.074803807748196505),
            scalar_t(-0.074803807748196505),
            scalar_t(0.5176323419876725),
            scalar_t(0.03354481152314847),
            scalar_t(0.29921523099278602),
            scalar_t(0.29921523099278602)
        };
        return data;
    }

    static const scalar_t *grad() {
        static const scalar_t data[72] = {
            scalar_t(0.56758792732771912),
            scalar_t(0.56758792732771912),
            scalar_t(0.78379396366385956),
            scalar_t(0),
            scalar_t(0),
            scalar_t(0.78379396366385956),
            scalar_t(-1.3513818909915787),
            scalar_t(-1.7837939636638596),
            scalar_t(1.7837939636638596),
            scalar_t(1.7837939636638596),
            scalar_t(-1.7837939636638596),
            scalar_t(-1.3513818909915787),
            scalar_t(-0.78379396366385956),
            scalar_t(-0.78379396366385956),
            scalar_t(-0.56758792732771912),
            scalar_t(0),
            scalar_t(0),
            scalar_t(0.78379396366385956),
            scalar_t(1.3513818909915787),
            scalar_t(-0.43241207267228082),
            scalar_t(1.7837939636638596),
            scalar_t(0.43241207267228082),
            scalar_t(-1.7837939636638596),
            scalar_t(0),
            scalar_t(-0.78379396366385956),
            scalar_t(-0.78379396366385956),
            scalar_t(0.78379396366385956),
            scalar_t(0),
            scalar_t(0),
            scalar_t(-0.56758792732771912),
            scalar_t(5.5511151231257827e-17),
            scalar_t(-1.7837939636638596),
            scalar_t(0.43241207267228082),
            scalar_t(1.7837939636638596),
            scalar_t(-0.43241207267228082),
            scalar_t(1.3513818909915787),
            scalar_t(-2.2673902919218341),
            scalar_t(-2.2673902919218341),
            scalar_t(-0.63369514596091703),
            scalar_t(0),
            scalar_t(0),
            scalar_t(-0.63369514596091703),
            scalar_t(2.9010854378827511),
            scalar_t(-0.36630485403908297),
            scalar_t(0.36630485403908297),
            scalar_t(0.36630485403908297),
            scalar_t(-0.36630485403908297),
            scalar_t(2.9010854378827511),
            scalar_t(0.63369514596091703),
            scalar_t(0.63369514596091703),
            scalar_t(2.2673902919218341),
            scalar_t(0),
            scalar_t(0),
            scalar_t(-0.63369514596091703),
            scalar_t(-2.9010854378827511),
            scalar_t(-3.2673902919218341),
            scalar_t(0.36630485403908297),
            scalar_t(3.2673902919218341),
            scalar_t(-0.36630485403908297),
            scalar_t(0),
            scalar_t(0.63369514596091703),
            scalar_t(0.63369514596091703),
            scalar_t(-0.63369514596091703),
            scalar_t(0),
            scalar_t(0),
            scalar_t(2.2673902919218341),
            scalar_t(0),
            scalar_t(-0.36630485403908297),
            scalar_t(3.2673902919218341),
            scalar_t(0.36630485403908297),
            scalar_t(-3.2673902919218341),
            scalar_t(-2.9010854378827511)
        };
        return data;
    }

    static const scalar_t *weight() {
        static const scalar_t data[6] = {
            scalar_t(0.11169079483900569),
            scalar_t(0.11169079483900569),
            scalar_t(0.11169079483900569),
            scalar_t(0.054975871827660998),
            scalar_t(0.054975871827660998),
            scalar_t(0.054975871827660998)
        };
        return data;
    }
};

template <typename scalar_t>
static SFEM_INLINE scalar_t neumann_tet10_trishell6_boundary_residual_soa_measure(
        const int q,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {
    const scalar_t *const grad = neumann_tet10_trishell6_boundary_residual_soa_reference_data<scalar_t>::grad();
    const int n_shape = neumann_tet10_trishell6_boundary_residual_soa_reference_data<scalar_t>::N_SHAPE;
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

static SFEM_INLINE const int *neumann_tet10_trishell6_boundary_residual_soa_side_nodes() {
    static const int data[24] = {
        0,
        1,
        3,
        4,
        8,
        7,
        1,
        2,
        3,
        5,
        9,
        8,
        0,
        3,
        2,
        7,
        9,
        6,
        0,
        2,
        1,
        6,
        5,
        4
    };
    return data;
}

static SFEM_INLINE void neumann_tet10_trishell6_boundary_residual_soa_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {
    const int *const SFEM_RESTRICT side_nodes = neumann_tet10_trishell6_boundary_residual_soa_side_nodes();
    constexpr int n_shape = 6;
    for (int i = 0; i < n_shape; ++i) {
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }
}

template <typename scalar_t>
static SFEM_INLINE void neumann_tet10_trishell6_boundary_residual_soa_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points, const scalar_t t0, const scalar_t t1, const scalar_t t2,
        scalar_t element_vector[3][6]) {
    const scalar_t *const shape = neumann_tet10_trishell6_boundary_residual_soa_reference_data<scalar_t>::shape();
    const scalar_t *const weight = neumann_tet10_trishell6_boundary_residual_soa_reference_data<scalar_t>::weight();
    const int n_shape = neumann_tet10_trishell6_boundary_residual_soa_reference_data<scalar_t>::N_SHAPE;
    const int n_qp = neumann_tet10_trishell6_boundary_residual_soa_reference_data<scalar_t>::N_QP;

        const scalar_t coeff0 = -t0;
        const scalar_t coeff1 = -t1;
        const scalar_t coeff2 = -t2;

    for (int q = 0; q < n_qp; ++q) {
        const scalar_t dS = neumann_tet10_trishell6_boundary_residual_soa_measure<scalar_t>(q, ev, points);
        const scalar_t qw = weight[q] * dS;

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
static SFEM_INLINE void neumann_tet10_trishell6_boundary_residual_soa_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t element_vector[3][6],
        const int out_stride,
        scalar_t *const SFEM_RESTRICT out0,
        scalar_t *const SFEM_RESTRICT out1,
        scalar_t *const SFEM_RESTRICT out2) {
    constexpr int n_shape = 6;
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
static SFEM_INLINE int neumann_tet10_trishell6_boundary_residual_soa_impl(
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
        idx_t ev[6];
        scalar_t element_vector[3][6];
        for (int i = 0; i < 6; ++i) {
            ev[i] = elements[i][e];
        }
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 6; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_tet10_trishell6_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, t2, element_vector);
        neumann_tet10_trishell6_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

template <typename scalar_t>
static SFEM_INLINE int neumann_tet10_trishell6_boundary_residual_sideset_soa_impl(
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
        idx_t ev[6];
        scalar_t element_vector[3][6];
        neumann_tet10_trishell6_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < 6; ++i) {
                element_vector[c][i] = scalar_t(0);
            }
        }
        neumann_tet10_trishell6_boundary_residual_soa_element<scalar_t>(ev, points, t0, t1, t2, element_vector);
        neumann_tet10_trishell6_boundary_residual_soa_scatter_element<scalar_t>(ev, element_vector, out_stride, out0, out1, out2);
    }

    return SFEM_SUCCESS;
}

}  // namespace codegen
}  // namespace sfem

extern "C" int neumann_tet10_trishell6_boundary_residual_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const real_t t0, const real_t t1, const real_t t2,
        const int out_stride,
        real_t *const SFEM_RESTRICT out0,
        real_t *const SFEM_RESTRICT out1,
        real_t *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_tet10_trishell6_boundary_residual_soa_impl<real_t>(
            nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_tet10_trishell6_boundary_residual_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points, const float t0, const float t1, const float t2,
        const int out_stride,
        float *const SFEM_RESTRICT out0,
        float *const SFEM_RESTRICT out1,
        float *const SFEM_RESTRICT out2) {
    return sfem::codegen::neumann_tet10_trishell6_boundary_residual_soa_impl<float>(
            nelements, nnodes, elements, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_tet10_trishell6_boundary_residual_sideset_soa(
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
    return sfem::codegen::neumann_tet10_trishell6_boundary_residual_sideset_soa_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}

extern "C" int neumann_tet10_trishell6_boundary_residual_sideset_soa_float(
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
    return sfem::codegen::neumann_tet10_trishell6_boundary_residual_sideset_soa_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points, t0, t1, t2, out_stride, out0, out1, out2);
}
