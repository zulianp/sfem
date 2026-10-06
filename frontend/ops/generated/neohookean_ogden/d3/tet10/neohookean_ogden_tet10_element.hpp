#ifndef NEOHOOKEAN_OGDEN_TET10_ELEMENT_API_HPP
#define NEOHOOKEAN_OGDEN_TET10_ELEMENT_API_HPP

#include <stddef.h>
#include "../neohookean_ogden_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"

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


template <typename scalar_t>
struct neohookean_ogden_tet10_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[110] = {scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.44897959183673491), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.22448979591836737), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918387), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836743), scalar_t(0.22448979591836732), scalar_t(0.020408163265306128), scalar_t(0.020408163265306128), scalar_t(0.22448979591836732), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918401), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(0.020408163265306135), scalar_t(0.22448979591836732), scalar_t(0.22448979591836751), scalar_t(0.020408163265306135), scalar_t(0.020408163265306121), scalar_t(0.22448979591836732), scalar_t(-0.061224489795918421), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(0.020408163265306145), scalar_t(0.020408163265306121), scalar_t(0.020408163265306145), scalar_t(0.2244897959183676), scalar_t(0.22448979591836732), scalar_t(0.22448979591836732), scalar_t(-0.080357142857142821), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428564), scalar_t(0.63809286661931275), scalar_t(0.16071428571428564), scalar_t(0.040478561952115862), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.040478561952115875), scalar_t(0.1607142857142857), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(0.040478561952115875), scalar_t(0.16071428571428573), scalar_t(0.1607142857142857), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-2.1428571428571428), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(-3.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571441), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719645), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(-1.1952286093343947), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(-1.1952286093343938), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(3.1428571428571428), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.40238569533280322)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[11] = {scalar_t(-0.013155555555555556), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887)};
        return data;
    }
};

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        neohookean_ogden_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
        const scalar_t *const grad_ref_z = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        neohookean_ogden_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_energy_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *const block_value = values + evbegin;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
        const scalar_t *const grad_ref_z = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        neohookean_ogden_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        neohookean_ogden_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
        const scalar_t *const grad_ref_z = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        neohookean_ogden_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_gradient_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evbegin;
        }
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evbegin;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
        const scalar_t *const grad_ref_z = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        neohookean_ogden_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}


template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) block_u_streams[stream] = u_streams[stream] + evbegin;
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VECTOR_SIZE + lane] = jacobian_adjugate[0][q * nelements + evbegin + lane];
                block_jacobian_adjugate1[q * VECTOR_SIZE + lane] = jacobian_adjugate[1][q * nelements + evbegin + lane];
                block_jacobian_adjugate2[q * VECTOR_SIZE + lane] = jacobian_adjugate[2][q * nelements + evbegin + lane];
                block_jacobian_adjugate3[q * VECTOR_SIZE + lane] = jacobian_adjugate[3][q * nelements + evbegin + lane];
                block_jacobian_adjugate4[q * VECTOR_SIZE + lane] = jacobian_adjugate[4][q * nelements + evbegin + lane];
                block_jacobian_adjugate5[q * VECTOR_SIZE + lane] = jacobian_adjugate[5][q * nelements + evbegin + lane];
                block_jacobian_adjugate6[q * VECTOR_SIZE + lane] = jacobian_adjugate[6][q * nelements + evbegin + lane];
                block_jacobian_adjugate7[q * VECTOR_SIZE + lane] = jacobian_adjugate[7][q * nelements + evbegin + lane];
                block_jacobian_adjugate8[q * VECTOR_SIZE + lane] = jacobian_adjugate[8][q * nelements + evbegin + lane];
                block_jacobian_determinant0[q * VECTOR_SIZE + lane] = jacobian_determinant[q * nelements + evbegin + lane];
            }
        }
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            neohookean_ogden_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) block_u_streams[stream] = u_streams[stream] + evbegin;
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
        const scalar_t *const grad_ref_z = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            neohookean_ogden_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int neohookean_ogden_tet10_hessian_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 3;
    static constexpr int N_SHAPE = 10;
    static constexpr int N_QP = 11;
    static constexpr int NDOFS = DIM * N_SHAPE;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        const scalar_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) block_u_streams[stream] = u_streams[stream] + evbegin;
        scalar_t block_coordinate_data[NDOFS][VECTOR_SIZE];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evbegin + lane];
            }
        }
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
        const scalar_t *const grad_ref_x = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
        const scalar_t *const grad_ref_y = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
        const scalar_t *const grad_ref_z = sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }
        scalar_t block_h_data[NDOFS][VECTOR_SIZE];
        scalar_t block_out_data[NDOFS][VECTOR_SIZE];
        const scalar_t *block_h_streams[NDOFS];
        scalar_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? scalar_t(1) : scalar_t(0);
                    block_out_data[stream][lane] = scalar_t(0);
                }
            }
            neohookean_ogden_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z(), sfem::codegen::neohookean_ogden_tet10_isoparametric_reference_data<scalar_t>::q_weight(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                scalar_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evbegin;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}


} // namespace codegen
} // namespace sfem

#endif
