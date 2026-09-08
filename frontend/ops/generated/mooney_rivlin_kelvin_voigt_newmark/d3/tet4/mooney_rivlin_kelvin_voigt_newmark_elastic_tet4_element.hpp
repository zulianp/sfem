#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_TET4_ELEMENT_API_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_TET4_ELEMENT_API_HPP

#include <stddef.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_local.hpp"
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
struct mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data {
    static const s_t *shape() {
        static const s_t data[4] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25)};
        return data;
    }
    static const s_t *grad_ref_x() {
        static const s_t data[4] = {s_t(-1), s_t(1), s_t(0), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_y() {
        static const s_t data[4] = {s_t(-1), s_t(0), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *grad_ref_z() {
        static const s_t data[4] = {s_t(-1), s_t(0), s_t(0), s_t(1)};
        return data;
    }
    static const s_t *q_weight() {
        static const s_t data[1] = {s_t(0.16666666666666666)};
        return data;
    }
};

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bu_streams[stream] = u_streams[stream] + evb;
        }
        s_t *const bvalue = values + evb;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            bvalue[lane] = s_t(0);
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        {
            const int q = 0;  // TET4 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bjacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                bjacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                bjacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                bjacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                bjacobian_adjugate4[q * VS + lane] = jacobian_adjugate[4][q * nelements + evb + lane];
                bjacobian_adjugate5[q * VS + lane] = jacobian_adjugate[5][q * nelements + evb + lane];
                bjacobian_adjugate6[q * VS + lane] = jacobian_adjugate[6][q * nelements + evb + lane];
                bjacobian_adjugate7[q * VS + lane] = jacobian_adjugate[7][q * nelements + evb + lane];
                bjacobian_adjugate8[q * VS + lane] = jacobian_adjugate[8][q * nelements + evb + lane];
                bjacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_objective_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bvalue);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bu_streams[stream] = u_streams[stream] + evb;
        }
        s_t *const bvalue = values + evb;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            bvalue[lane] = s_t(0);
        }
        s_t bcoordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bcoordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
        const s_t *const grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
        {
            const int q = 0;  // TET4 evaluates in closed form
            s_t *bjacobian_adjugate_streams[ND * ND] = {bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J02_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            s_t J12_values[VS];
            s_t J20_values[VS];
            s_t J21_values[VS];
            s_t J22_values[VS];
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
                J02_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                const s_t g2 = grad_ref_z[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J02 = J02_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                const s_t J12 = J12_values[lane];
                const s_t J20 = J20_values[lane];
                const s_t J21 = J21_values[lane];
                const s_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        bjacobian_adjugate_streams, bjacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_objective_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bvalue);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_energy_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bu_streams[stream] = u_streams[stream] + evb;
        }
        s_t *const bvalue = values + evb;
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            bvalue[lane] = s_t(0);
        }
        s_t bcoordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bcoordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
        const s_t *const grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
        {
            const int q = 0;  // TET4 evaluates in closed form
            s_t *bjacobian_adjugate_streams[ND * ND] = {bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J02_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            s_t J12_values[VS];
            s_t J20_values[VS];
            s_t J21_values[VS];
            s_t J22_values[VS];
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
                J02_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                const s_t g2 = grad_ref_z[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J02 = J02_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                const s_t J12 = J12_values[lane];
                const s_t J20 = J20_values[lane];
                const s_t J21 = J21_values[lane];
                const s_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        bjacobian_adjugate_streams, bjacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_objective_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bvalue);
    }
    return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bu_streams[stream] = u_streams[stream] + evb;
        }
        s_t *bout_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bout_streams[stream] = out_streams[stream] + evb;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bout_streams[stream][lane] = s_t(0);
            }
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        {
            const int q = 0;  // TET4 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bjacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                bjacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                bjacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                bjacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                bjacobian_adjugate4[q * VS + lane] = jacobian_adjugate[4][q * nelements + evb + lane];
                bjacobian_adjugate5[q * VS + lane] = jacobian_adjugate[5][q * nelements + evb + lane];
                bjacobian_adjugate6[q * VS + lane] = jacobian_adjugate[6][q * nelements + evb + lane];
                bjacobian_adjugate7[q * VS + lane] = jacobian_adjugate[7][q * nelements + evb + lane];
                bjacobian_adjugate8[q * VS + lane] = jacobian_adjugate[8][q * nelements + evb + lane];
                bjacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_gradient_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bu_streams[stream] = u_streams[stream] + evb;
        }
        s_t *bout_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bout_streams[stream] = out_streams[stream] + evb;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bout_streams[stream][lane] = s_t(0);
            }
        }
        s_t bcoordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bcoordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
        const s_t *const grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
        {
            const int q = 0;  // TET4 evaluates in closed form
            s_t *bjacobian_adjugate_streams[ND * ND] = {bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J02_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            s_t J12_values[VS];
            s_t J20_values[VS];
            s_t J21_values[VS];
            s_t J22_values[VS];
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
                J02_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                const s_t g2 = grad_ref_z[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J02 = J02_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                const s_t J12 = J12_values[lane];
                const s_t J20 = J20_values[lane];
                const s_t J21 = J21_values[lane];
                const s_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        bjacobian_adjugate_streams, bjacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_gradient_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_gradient_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bu_streams[stream] = u_streams[stream] + evb;
        }
        s_t *bout_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bout_streams[stream] = out_streams[stream] + evb;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bout_streams[stream][lane] = s_t(0);
            }
        }
        s_t bcoordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bcoordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
        const s_t *const grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
        {
            const int q = 0;  // TET4 evaluates in closed form
            s_t *bjacobian_adjugate_streams[ND * ND] = {bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J02_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            s_t J12_values[VS];
            s_t J20_values[VS];
            s_t J21_values[VS];
            s_t J22_values[VS];
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
                J02_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                const s_t g2 = grad_ref_z[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J02 = J02_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                const s_t J12 = J12_values[lane];
                const s_t J20 = J20_values[lane];
                const s_t J21 = J21_values[lane];
                const s_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        bjacobian_adjugate_streams, bjacobian_determinant0, q * VS + lane);
            }
        }
        mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_gradient_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bout_streams);
    }
    return SFEM_SUCCESS;
}


template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        {
            const int q = 0;  // TET4 evaluates in closed form
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bjacobian_adjugate0[q * VS + lane] = jacobian_adjugate[0][q * nelements + evb + lane];
                bjacobian_adjugate1[q * VS + lane] = jacobian_adjugate[1][q * nelements + evb + lane];
                bjacobian_adjugate2[q * VS + lane] = jacobian_adjugate[2][q * nelements + evb + lane];
                bjacobian_adjugate3[q * VS + lane] = jacobian_adjugate[3][q * nelements + evb + lane];
                bjacobian_adjugate4[q * VS + lane] = jacobian_adjugate[4][q * nelements + evb + lane];
                bjacobian_adjugate5[q * VS + lane] = jacobian_adjugate[5][q * nelements + evb + lane];
                bjacobian_adjugate6[q * VS + lane] = jacobian_adjugate[6][q * nelements + evb + lane];
                bjacobian_adjugate7[q * VS + lane] = jacobian_adjugate[7][q * nelements + evb + lane];
                bjacobian_adjugate8[q * VS + lane] = jacobian_adjugate[8][q * nelements + evb + lane];
                bjacobian_determinant0[q * VS + lane] = jacobian_determinant[q * nelements + evb + lane];
            }
        }
        s_t bh_data[NDOFS][VS];
        s_t bout_data[NDOFS][VS];
        const s_t *bh_streams[NDOFS];
        s_t *bout_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bh_streams[stream] = bh_data[stream];
            bout_streams[stream] = bout_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
                    bout_data[stream][lane] = s_t(0);
                }
            }
            mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_apply_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bh_streams, bout_streams);
            for (int row = 0; row < NDOFS; ++row) {
                s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = bout_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
        s_t bcoordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bcoordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
        const s_t *const grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
        {
            const int q = 0;  // TET4 evaluates in closed form
            s_t *bjacobian_adjugate_streams[ND * ND] = {bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J02_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            s_t J12_values[VS];
            s_t J20_values[VS];
            s_t J21_values[VS];
            s_t J22_values[VS];
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
                J02_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                const s_t g2 = grad_ref_z[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J02 = J02_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                const s_t J12 = J12_values[lane];
                const s_t J20 = J20_values[lane];
                const s_t J21 = J21_values[lane];
                const s_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        bjacobian_adjugate_streams, bjacobian_determinant0, q * VS + lane);
            }
        }
        s_t bh_data[NDOFS][VS];
        s_t bout_data[NDOFS][VS];
        const s_t *bh_streams[NDOFS];
        s_t *bout_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bh_streams[stream] = bh_data[stream];
            bout_streams[stream] = bout_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
                    bout_data[stream][lane] = s_t(0);
                }
            }
            mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_apply_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bh_streams, bout_streams);
            for (int row = 0; row < NDOFS; ++row) {
                s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = bout_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_hessian_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 3;
    static constexpr int ND = 3;
    static constexpr int NS = 4;
    static constexpr int NQ = 1;
    static constexpr int NDOFS = NC * NS;
    if (nelements <= 0) return SFEM_SUCCESS;
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        const s_t *bu_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) bu_streams[stream] = u_streams[stream] + evb;
        s_t bcoordinate_data[NDOFS][VS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                bcoordinate_data[stream][lane] = coords[stream][evb + lane];
            }
        }
        s_t bjacobian_adjugate0[NQ * VS];
        s_t bjacobian_adjugate1[NQ * VS];
        s_t bjacobian_adjugate2[NQ * VS];
        s_t bjacobian_adjugate3[NQ * VS];
        s_t bjacobian_adjugate4[NQ * VS];
        s_t bjacobian_adjugate5[NQ * VS];
        s_t bjacobian_adjugate6[NQ * VS];
        s_t bjacobian_adjugate7[NQ * VS];
        s_t bjacobian_adjugate8[NQ * VS];
        s_t bjacobian_determinant0[NQ * VS];
        const s_t *const grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x();
        const s_t *const grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y();
        const s_t *const grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z();
        {
            const int q = 0;  // TET4 evaluates in closed form
            s_t *bjacobian_adjugate_streams[ND * ND] = {bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8};
            s_t J00_values[VS];
            s_t J01_values[VS];
            s_t J02_values[VS];
            s_t J10_values[VS];
            s_t J11_values[VS];
            s_t J12_values[VS];
            s_t J20_values[VS];
            s_t J21_values[VS];
            s_t J22_values[VS];
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
                J02_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                const s_t g0 = grad_ref_x[q * NS + shape];
                const s_t g1 = grad_ref_y[q * NS + shape];
                const s_t g2 = grad_ref_z[q * NS + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += bcoordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += bcoordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += bcoordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = J00_values[lane];
                const s_t J01 = J01_values[lane];
                const s_t J02 = J02_values[lane];
                const s_t J10 = J10_values[lane];
                const s_t J11 = J11_values[lane];
                const s_t J12 = J12_values[lane];
                const s_t J20 = J20_values[lane];
                const s_t J21 = J21_values[lane];
                const s_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        bjacobian_adjugate_streams, bjacobian_determinant0, q * VS + lane);
            }
        }
        s_t bh_data[NDOFS][VS];
        s_t bout_data[NDOFS][VS];
        const s_t *bh_streams[NDOFS];
        s_t *bout_streams[NDOFS];
        for (int stream = 0; stream < NDOFS; ++stream) {
            bh_streams[stream] = bh_data[stream];
            bout_streams[stream] = bout_data[stream];
        }
        for (int col = 0; col < NDOFS; ++col) {
            for (int stream = 0; stream < NDOFS; ++stream) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    bh_data[stream][lane] = stream == col ? s_t(1) : s_t(0);
                    bout_data[stream][lane] = s_t(0);
                }
            }
            mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_apply_block<s_t, NQ, NS, VS>(nelems, VS, bjacobian_adjugate0, bjacobian_adjugate1, bjacobian_adjugate2, bjacobian_adjugate3, bjacobian_adjugate4, bjacobian_adjugate5, bjacobian_adjugate6, bjacobian_adjugate7, bjacobian_adjugate8, bjacobian_determinant0, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_isoparametric_reference_data<s_t>::q_weight(), lmbda, mu, bu_streams, bh_streams, bout_streams);
            for (int row = 0; row < NDOFS; ++row) {
                s_t *const matrix_stream = matrix_streams[row * NDOFS + col] + evb;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    matrix_stream[lane] = bout_data[row][lane];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}


} // namespace codegen
} // namespace sfem

#endif
