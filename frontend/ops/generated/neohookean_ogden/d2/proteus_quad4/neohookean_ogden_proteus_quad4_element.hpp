#ifndef NEOHOOKEAN_OGDEN_PROTEUS_QUAD4_ELEMENT_API_HPP
#define NEOHOOKEAN_OGDEN_PROTEUS_QUAD4_ELEMENT_API_HPP

#include <stddef.h>
#include "../neohookean_ogden_d2_tensor_product_local.hpp"
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


template <typename s_t>
struct neohookean_ogden_proteus_quad4_isoparametric_reference_data {
    static const s_t *shape_1d() {
        static const s_t data[4] = {s_t(0.78867513459481287), s_t(0.21132486540518708), s_t(0.21132486540518713), s_t(0.78867513459481287)};
        return data;
    }
    static const s_t *grad_1d() {
        static const s_t data[4] = {s_t(-1), s_t(1), s_t(-1), s_t(1)};
        return data;
    }
    static const s_t *q_weight_1d() {
        static const s_t data[2] = {s_t(0.5), s_t(0.5)};
        return data;
    }
};

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evb;
        }
        s_t *const block_value = values + evb;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = s_t(0);
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                block_jacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                block_jacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                block_jacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                block_jacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        neohookean_ogden_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evb;
        }
        s_t *const block_value = values + evb;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = s_t(0);
        }
        s_t block_coordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        neohookean_ogden_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_energy_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evb;
        }
        s_t *const block_value = values + evb;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = s_t(0);
        }
        s_t block_coordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        neohookean_ogden_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evb;
        }
        s_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evb;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = s_t(0);
            }
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                block_jacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                block_jacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                block_jacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                block_jacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        neohookean_ogden_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evb;
        }
        s_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evb;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = s_t(0);
            }
        }
        s_t block_coordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        neohookean_ogden_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_gradient_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_u_streams[stream] = u_streams[stream] + evb;
        }
        s_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_out_streams[stream] = out_streams[stream] + evb;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_streams[stream][lane] = s_t(0);
            }
        }
        s_t block_coordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        neohookean_ogden_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) block_u_streams[stream] = u_streams[stream] + evb;
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                block_jacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                block_jacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                block_jacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                block_jacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        s_t block_h_data[NDOFS][VS];
        s_t block_out_data[NDOFS][VS];
        const s_t *block_h_streams[NDOFS];
        s_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
                    block_out_data[stream][lane] = s_t(0);
                }
            }
            neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) block_u_streams[stream] = u_streams[stream] + evb;
        s_t block_coordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        s_t block_h_data[NDOFS][VS];
        s_t block_out_data[NDOFS][VS];
        const s_t *block_h_streams[NDOFS];
        s_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
                    block_out_data[stream][lane] = s_t(0);
                }
            }
            neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = block_out_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_hessian_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *block_u_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) block_u_streams[stream] = u_streams[stream] + evb;
        s_t block_coordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_coordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 0, coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(nelems, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), block_coordinate_data, 1, coordinate_grad_ref + 1 * NQ * ND * VS);
        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);
        s_t block_h_data[NDOFS][VS];
        s_t block_out_data[NDOFS][VS];
        const s_t *block_h_streams[NDOFS];
        s_t *block_out_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
            block_out_streams[stream] = block_out_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_h_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
                    block_out_data[stream][lane] = s_t(0);
                }
            }
            neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d(), sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
            for (int row = 0; row < NDOFS; ++row) {
                s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
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
