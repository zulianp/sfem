#include <cstdio>
#include <type_traits>
#include "../linear_elasticity_d2_tensor_product_local.hpp"
#include "../linear_elasticity_d2_tensor_product_hessian.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <cstdlib>
#include "../../../packed_thread_scratch.hpp"
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

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
        const int,
        const g_t *const RSTR source,
        s_t *const RSTR,
        std::true_type) {
    return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
        const int ne,
        const g_t *const RSTR source,
        s_t *const RSTR converted,
        std::false_type) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
        converted[lane] = s_t(source[lane]);
    }
    return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename s_t>
struct linear_elasticity_proteus_quad4_affine_reference_data {
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

template <typename s_t>
struct linear_elasticity_proteus_quad4_isoparametric_reference_data {
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

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_proteus_quad4_objective_soa_diagnostics_data = {
    "linear_elasticity_proteus_quad4_objective_soa",
    "PROTEUS_QUAD4",
    2,
    4,
    4,
    16,
    2,
    5,
    5,
    0,
    0,
    4,
    0,
    0,
    0,
    2,
    1,
    14,
    240,
    444,
    0,
    6,
    5,
    8,
    2,
    2,
    8,
    0,
    1,
    1,
    1,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_objective_soa_diagnostics(void) {
    return &sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data;
}

extern "C" double linear_elasticity_proteus_quad4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void linear_elasticity_proteus_quad4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_proteus_quad4_objective_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_proteus_quad4_objective_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_proteus_quad4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_proteus_quad4_objective_affine_mesh_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_proteus_quad4_objective_affine_mesh_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_proteus_quad4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_proteus_quad4_objective_isoparametric_mesh_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_proteus_quad4_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int linear_elasticity_proteus_quad4_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const g_t *const RSTR x = points[0];
    const g_t *const RSTR y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    static constexpr int NQ1 = 2;
    static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
        idx_t ev[VS * NS];
        s_t bu_data[NS * NC][VS];
        s_t bu_base_data[NS * NC][VS];
        s_t bh_data[NS * NC][VS];
        s_t bvalue[VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t bdet0[NQ * VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const RSTR element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                ev[element_node * VS + lane] = element_shape[evb + lane];
            }
        }
        const g_t *const coordinate_components[ND] = {x, y};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
                }
            }
        }

        const s_t *const u_components[NC] = {ux, uy};
        const s_t *const h_components[NC] = {hx, hy};
        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bu_streams[stream] = bu_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    bu_base_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                    bh_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        for (int step = 0; step < nsteps; ++step) {
            const s_t alpha = steps[step];
            for (int shape = 0; shape < NS; ++shape) {
                for (int d = 0; d < NC; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
                        bu_data[shape * NC + d][lane] = bu_base_data[shape * NC + d][lane] + alpha * bh_data[shape * NC + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                bvalue[lane] = s_t(0);
            }

            linear_elasticity_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bvalue);

            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                value[(ptrdiff_t)step * nelements + evb + lane] = bvalue[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_proteus_quad4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int linear_elasticity_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data = {
    "linear_elasticity_proteus_quad4_gradient_soa",
    "PROTEUS_QUAD4",
    2,
    4,
    4,
    16,
    2,
    4,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    7,
    9,
    468,
    672,
    3,
    6,
    5,
    8,
    2,
    2,
    8,
    0,
    8,
    8,
    8,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_gradient_soa_diagnostics(void) {
    return &sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data;
}

extern "C" double linear_elasticity_proteus_quad4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void linear_elasticity_proteus_quad4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_proteus_quad4_gradient_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_proteus_quad4_gradient_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_proteus_quad4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_proteus_quad4_gradient_affine_mesh_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_proteus_quad4_gradient_affine_mesh_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const g_t *const RSTR x = points[0];
    const g_t *const RSTR y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    static constexpr int NQ1 = 2;
    static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
        idx_t ev[VS * NS];
        s_t bu_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t bdet0[NQ * VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const RSTR element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                ev[element_node * VS + lane] = element_shape[evb + lane];
            }
        }
        const g_t *const coordinate_components[ND] = {x, y};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
                }
            }
        }
        const s_t *const u_components[NC] = {ux, uy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    bu_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                bout_data[stream][lane] = s_t(0);
            }
        }

        const s_t *bu_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bu_streams[stream] = bu_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bout_streams[stream] = bout_data[stream];
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        linear_elasticity_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bu_streams, bout_streams);

        s_t *const out_components[NC] = {outx, outy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < ne; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += bout_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_proteus_quad4_apply_soa_diagnostics_data = {
    "linear_elasticity_proteus_quad4_apply_soa",
    "PROTEUS_QUAD4",
    2,
    4,
    4,
    16,
    2,
    4,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    9,
    468,
    672,
    3,
    6,
    5,
    8,
    2,
    2,
    0,
    8,
    8,
    8,
    8,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_apply_soa_diagnostics(void) {
    return &sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data;
}

extern "C" double linear_elasticity_proteus_quad4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void linear_elasticity_proteus_quad4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_proteus_quad4_apply_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_proteus_quad4_apply_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_proteus_quad4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_proteus_quad4_apply_affine_mesh_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_proteus_quad4_apply_affine_mesh_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa",
            &sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::linear_elasticity_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const g_t *const RSTR x = points[0];
    const g_t *const RSTR y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    static constexpr int NQ1 = 2;
    static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
        idx_t ev[VS * NS];
        s_t bh_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        s_t bcoordinate_data[NS * ND][VS];
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t bdet0[NQ * VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const RSTR element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                ev[element_node * VS + lane] = element_shape[evb + lane];
            }
        }
        const g_t *const coordinate_components[ND] = {x, y};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    bcoordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
                }
            }
        }
        const s_t *const h_components[NC] = {hx, hy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    bh_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                bout_data[stream][lane] = s_t(0);
            }
        }

        const s_t *bh_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bh_streams[stream] = bh_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bout_streams[stream] = bout_data[stream];
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        linear_elasticity_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, bh_streams, bout_streams);

        s_t *const out_components[NC] = {outx, outy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < ne; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += bout_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, h_stride, hx, hy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static SFEM_INLINE void linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_find_cols(
        const idx_t *const RSTR targets,
        const idx_t *const RSTR row,
        const int lenrow,
        idx_t *const RSTR ks) {
#pragma unroll(4)
    for (int d = 0; d < 4; ++d) {
        ks[d] = 0;
    }
    for (int k = 0; k < lenrow; ++k) {
#pragma unroll(4)
        for (int d = 0; d < 4; ++d) {
            ks[d] += row[k] < targets[d];
        }
    }
}

template <typename s_t>
static SFEM_INLINE void linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_scatter_bsr(
        const idx_t *const RSTR ev,
        const s_t *const RSTR element_matrix,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        s_t *const RSTR values) {
    static constexpr int NC = 2;
    static constexpr int NS = 4;
    count_t entries[NS * NS];
    idx_t ks[NS];
    for (int i = 0; i < NS; ++i) {
        const idx_t dof_i = ev[i];
        const count_t row_begin = rowptr[dof_i];
        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
        const idx_t *const RSTR cols = &colidx[row_begin];
        linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
        for (int j = 0; j < NS; ++j) {
            entries[i * NS + j] = row_begin + ks[j];
        }
    }
    for (int i = 0; i < NS; ++i) {
        for (int j = 0; j < NS; ++j) {
            s_t *const block = &values[entries[i * NS + j] * NC * NC];
            for (int bi = 0; bi < NC; ++bi) {
                const int row = bi * NS + i;
                for (int bj = 0; bj < NC; ++bj) {
                    const int col = bj * NS + j;
#pragma omp atomic update
                    block[bi * NC + bj] += element_matrix[row * (NC * NS) + col];
                }
            }
        }
    }
}

template <typename s_t>
static SFEM_INLINE void linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_scatter_block_diag_sym(
        const idx_t *const RSTR ev,
        const s_t *const RSTR element_matrix,
        s_t *const RSTR values) {
    static constexpr int NC = 2;
    static constexpr int NS = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SYM_DIM = (NC * (NC + 1)) / 2;
    for (int i = 0; i < NS; ++i) {
        s_t *const block = &values[(ptrdiff_t)ev[i] * SYM_DIM];
        int sym = 0;
        for (int bi = 0; bi < NC; ++bi) {
            const int row = bi * NS + i;
            for (int bj = bi; bj < NC; ++bj) {
                const int col = bj * NS + i;
#pragma omp atomic update
                block[sym++] += element_matrix[row * NDOFS + col];
            }
        }
    }
}

template <typename s_t, typename g_t, int FORMAT>
static int linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        s_t *const RSTR values,
        const int *const RSTR diag_offsets,
        const ptrdiff_t ndiag,
        const ptrdiff_t coo_nnz,
        const idx_t *const RSTR coo_rows,
        const idx_t *const RSTR coo_cols,
        idx_t *const RSTR coo_triplet_rows,
        idx_t *const RSTR coo_triplet_cols) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 1;
    static constexpr int NDOFS = NC * NS;
    (void)nnodes;
    const g_t *const RSTR x = points[0];
    const g_t *const RSTR y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::linear_elasticity_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();

    int unsupported_matrix_format = 0;
#pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        idx_t ev[NS];
        s_t element_matrix[NDOFS * NDOFS];
        s_t bh_data[NS * NC][VS];
        s_t bout_data[NS * NC][VS];
        s_t bcoordinate_data[NS * ND][VS];
        static constexpr int ne = VS;
        s_t badj0[NQ * VS];
        s_t badj1[NQ * VS];
        s_t badj2[NQ * VS];
        s_t badj3[NQ * VS];
        s_t bdet0[NQ * VS];
        s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
        const s_t *bh_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bh_streams[stream] = bh_data[stream];
        }
        s_t *bout_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            bout_streams[stream] = bout_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < ND; ++d) {
                bcoordinate_data[shape * ND + d][0] = s_t(points[d][node]);
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

        linear_elasticity_d2_tensor_product_direct_hessian_tensor_product_element_matrix<s_t, NQ, NS, VS>(badj0, badj1, badj2, badj3, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, element_matrix);

        if constexpr (FORMAT == 1) {
            linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
        } else if constexpr (FORMAT == 6) {
            linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_scatter_block_diag_sym(ev, element_matrix, values);
        } else {
            unsupported_matrix_format |= 1;
        }
    }

    return unsupported_matrix_format ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int linear_elasticity_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int linear_elasticity_proteus_quad4_hessian_block_diag_sym_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        double *const RSTR values
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 6>(nelements, nnodes, elements, points, lmbda, mu, nullptr, nullptr, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int linear_elasticity_proteus_quad4_hessian_block_diag_sym_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        float *const RSTR values
) {
    return sfem::codegen::linear_elasticity_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 6>(nelements, nnodes, elements, points, lmbda, mu, nullptr, nullptr, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
