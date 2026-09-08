#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_TRI3_ELEMENT_API_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_TRI3_ELEMENT_API_HPP

#include <stddef.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_local.hpp"
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
struct mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data {
    static const s_t *shape() {
        static const s_t data[3] = {s_t(0.33333333333333343), s_t(0.33333333333333331), s_t(0.33333333333333331)};
        return data;
    }
    static const s_t *grad_ref_x() {
        static const s_t data[3] = {s_t(-1), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_y() {
        static const s_t data[3] = {s_t(-1), s_t(0), s_t(1)};
        return data;
    }
    static const s_t *q_weight() {
        static const s_t data[1] = {s_t(0.5)};
        return data;
    }
};

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_energy_element_geometry_soa(
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
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        {
            const int q = 0;  // TRI3 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                block_jacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                block_jacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                block_jacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                block_jacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_objective_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y();
        {
            const int q = 0;  // TRI3 evaluates in closed form
            s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<s_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_objective_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_energy_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y();
        {
            const int q = 0;  // TRI3 evaluates in closed form
            s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<s_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_objective_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_value);
    }
    return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_element_geometry_soa(
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
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        {
            const int q = 0;  // TRI3 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_jacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                block_jacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                block_jacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                block_jacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                block_jacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_gradient_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y();
        {
            const int q = 0;  // TRI3 evaluates in closed form
            s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<s_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_gradient_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_gradient_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y();
        {
            const int q = 0;  // TRI3 evaluates in closed form
            s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<s_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_gradient_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_out_streams);
    }
    return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_element_geometry_soa(
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
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        {
            const int q = 0;  // TRI3 evaluates in closed form
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
            mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_apply_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y();
        {
            const int q = 0;  // TRI3 evaluates in closed form
            s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<s_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VS + lane);
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
            mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_apply_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
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
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 3;
    static constexpr int NQ = 1;
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
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y();
        {
            const int q = 0;  // TRI3 evaluates in closed form
            s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 2 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 2 + 1][lane] * g1;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                geometry_jacobian_adjugate_and_determinant_2<s_t>(
                        J00, J01, J10, J11, block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VS + lane);
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
            mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_apply_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
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
