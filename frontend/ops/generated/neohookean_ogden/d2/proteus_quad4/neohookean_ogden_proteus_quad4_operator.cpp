#include <cstdio>
#include <type_traits>
#include "../neohookean_ogden_d2_tensor_product_local.hpp"
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
SFEM_INLINE const s_t *affine_geometry_stream(
        const int,
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *affine_geometry_stream(
        const int nelems,
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT converted,
        std::false_type) {
    #pragma omp simd
    for (int lane = 0; lane < nelems; ++lane) {
        converted[lane] = s_t(source[lane]);
    }
    return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename s_t>
struct neohookean_ogden_proteus_quad4_affine_reference_data {
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

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data = {
    "neohookean_ogden_proteus_quad4_objective_soa",
    "PROTEUS_QUAD4",
    2,
    4,
    4,
    16,
    2,
    9,
    9,
    0,
    0,
    5,
    0,
    1,
    0,
    2,
    4,
    43,
    240,
    444,
    3,
    7,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_quad4_objective_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_quad4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_quad4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_quad4_objective_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_quad4_objective_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_quad4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_quad4_objective_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_quad4_objective_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_quad4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_quad4_objective_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_quad4_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const *const SFEM_RESTRICT points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const s_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const s_t *const SFEM_RESTRICT hx,
        const s_t *const SFEM_RESTRICT hy,
        const int nsteps,
        const s_t *const SFEM_RESTRICT steps,
        s_t *const SFEM_RESTRICT value
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const g_t *const SFEM_RESTRICT x = points[0];
    const g_t *const SFEM_RESTRICT y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    static constexpr int NQ1 = 2;
    static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        idx_t ev[VS * NS];
        s_t block_u_data[NS * NC][VS];
        s_t block_u_base_data[NS * NC][VS];
        s_t block_h_data[NS * NC][VS];
        s_t block_value[VS];
        s_t block_coordinate_data[NS * ND][VS];
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evbegin + lane];
            }
        }
        const g_t *const coordinate_components[ND] = {x, y};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
                }
            }
        }

        const s_t *const u_components[NC] = {ux, uy};
        const s_t *const h_components[NC] = {hx, hy};
        const s_t *block_u_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    block_u_base_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        for (int step = 0; step < nsteps; ++step) {
            const s_t alpha = steps[step];
            for (int shape = 0; shape < NS; ++shape) {
                for (int d = 0; d < NC; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * NC + d][lane] = block_u_base_data[shape * NC + d][lane] + alpha * block_h_data[shape * NC + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = s_t(0);
            }

            neohookean_ogden_d2_tensor_product_objective_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_value);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_quad4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int neohookean_ogden_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data = {
    "neohookean_ogden_proteus_quad4_gradient_soa",
    "PROTEUS_QUAD4",
    2,
    4,
    4,
    16,
    2,
    11,
    22,
    1,
    0,
    0,
    0,
    1,
    0,
    2,
    13,
    61,
    468,
    672,
    9,
    10,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_quad4_gradient_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_quad4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_quad4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_quad4_gradient_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_quad4_gradient_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_quad4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_quad4_gradient_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_quad4_gradient_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const *const SFEM_RESTRICT points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const s_t *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT outx,
        s_t *const SFEM_RESTRICT outy
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const g_t *const SFEM_RESTRICT x = points[0];
    const g_t *const SFEM_RESTRICT y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    static constexpr int NQ1 = 2;
    static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        idx_t ev[VS * NS];
        s_t block_u_data[NS * NC][VS];
        s_t block_out_data[NS * NC][VS];
        s_t block_coordinate_data[NS * ND][VS];
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evbegin + lane];
            }
        }
        const g_t *const coordinate_components[ND] = {x, y};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
                }
            }
        }
        const s_t *const u_components[NC] = {ux, uy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    block_u_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = s_t(0);
            }
        }

        const s_t *block_u_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        s_t *block_out_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d2_tensor_product_gradient_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_out_streams);

        s_t *const out_components[NC] = {outx, outy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += block_out_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data = {
    "neohookean_ogden_proteus_quad4_apply_soa",
    "PROTEUS_QUAD4",
    2,
    4,
    4,
    16,
    2,
    36,
    75,
    1,
    0,
    6,
    0,
    1,
    0,
    2,
    34,
    145,
    468,
    672,
    30,
    20,
    5,
    8,
    2,
    2,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_quad4_apply_soa_diagnostics(void) {
    return &sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data;
}

extern "C" double neohookean_ogden_proteus_quad4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void neohookean_ogden_proteus_quad4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_quad4_apply_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "neohookean_ogden_proteus_quad4_apply_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_quad4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_quad4_apply_affine_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "neohookean_ogden_proteus_quad4_apply_affine_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa",
            &sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::neohookean_ogden_proteus_quad4_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const *const SFEM_RESTRICT points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const s_t *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const s_t *const SFEM_RESTRICT hx,
        const s_t *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT outx,
        s_t *const SFEM_RESTRICT outy
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 16;
    (void)nnodes;
    const g_t *const SFEM_RESTRICT x = points[0];
    const g_t *const SFEM_RESTRICT y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();
    static constexpr int NQ1 = 2;
    static constexpr int NS1 = 2;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evbegin);
        idx_t ev[VS * NS];
        s_t block_u_data[NS * NC][VS];
        s_t block_h_data[NS * NC][VS];
        s_t block_out_data[NS * NC][VS];
        s_t block_coordinate_data[NS * ND][VS];
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];

        for (int element_node = 0; element_node < NS; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VS + lane] = element_shape[evbegin + lane];
            }
        }
        const g_t *const coordinate_components[ND] = {x, y};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * ND + d][lane] = coordinate_components[d][ev[shape * VS + lane]];
                }
            }
        }
        const s_t *const u_components[NC] = {ux, uy};
        const s_t *const h_components[NC] = {hx, hy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VS + lane];
                    block_u_data[shape * NC + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * NC + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < NS * NC; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = s_t(0);
            }
        }

        const s_t *block_u_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const s_t *block_h_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        s_t *block_out_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(nelems, VS, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);

        s_t *const out_components[NC] = {outx, outy};

        for (int shape = 0; shape < NS; ++shape) {
            for (int d = 0; d < NC; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VS + scatter] * out_stride] += block_out_data[shape * NC + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}


namespace sfem {
namespace codegen {

static SFEM_INLINE void neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_find_cols(
        const idx_t *const SFEM_RESTRICT targets,
        const idx_t *const SFEM_RESTRICT row,
        const int lenrow,
        idx_t *const SFEM_RESTRICT ks) {
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
static SFEM_INLINE void neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_scatter_bsr(
        const idx_t *const SFEM_RESTRICT ev,
        const s_t *const SFEM_RESTRICT element_matrix,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        s_t *const SFEM_RESTRICT values) {
    static constexpr int NC = 2;
    static constexpr int NS = 4;
    count_t entries[NS * NS];
    idx_t ks[NS];
    for (int i = 0; i < NS; ++i) {
        const idx_t dof_i = ev[i];
        const count_t row_begin = rowptr[dof_i];
        const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];
        neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_find_cols(ev, cols, lenrow, ks);
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

template <typename s_t, typename g_t, int FORMAT>
static int neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const *const SFEM_RESTRICT points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const SFEM_RESTRICT ux,
        const s_t *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        s_t *const SFEM_RESTRICT values,
        const int *const SFEM_RESTRICT diag_offsets,
        const ptrdiff_t ndiag,
        const ptrdiff_t coo_nnz,
        const idx_t *const SFEM_RESTRICT coo_rows,
        const idx_t *const SFEM_RESTRICT coo_cols,
        idx_t *const SFEM_RESTRICT coo_triplet_rows,
        idx_t *const SFEM_RESTRICT coo_triplet_cols) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NQ = 4;
    static constexpr int NS = 4;
    static constexpr int VS = 1;
    static constexpr int NDOFS = NC * NS;
    (void)nnodes;
    const s_t *const u_components[NC] = {ux, uy};
    const g_t *const SFEM_RESTRICT x = points[0];
    const g_t *const SFEM_RESTRICT y = points[1];
    const s_t *const isoparametric_shape_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::shape_1d();
    const s_t *const isoparametric_grad_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::grad_1d();
    const s_t *const isoparametric_q_weight_1d = sfem::codegen::neohookean_ogden_proteus_quad4_isoparametric_reference_data<s_t>::q_weight_1d();

    int unsupported_matrix_format = 0;
#pragma omp parallel for schedule(static) reduction(|:unsupported_matrix_format)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        idx_t ev[NS];
        s_t element_matrix[NDOFS * NDOFS];
        s_t block_h_data[NS * NC][VS];
        s_t block_out_data[NS * NC][VS];
        s_t block_coordinate_data[NS * ND][VS];
        static constexpr int nelems = VS;
        s_t block_u_data[NS * NC][VS];
        s_t block_jacobian_adjugate0[NQ * VS];
        s_t block_jacobian_adjugate1[NQ * VS];
        s_t block_jacobian_adjugate2[NQ * VS];
        s_t block_jacobian_adjugate3[NQ * VS];
        s_t block_jacobian_determinant0[NQ * VS];
        s_t *block_jacobian_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        const s_t *block_u_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        const s_t *block_h_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        s_t *block_out_streams[NS * NC];
        for (int stream = 0; stream < NS * NC; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t node = elements[shape][element];
            ev[shape] = node;
            for (int d = 0; d < ND; ++d) {
                block_coordinate_data[shape * ND + d][0] = s_t(points[d][node]);
                block_u_data[shape * NC + d][0] = u_components[d][node * u_stride];
            }
        }

        s_t coordinate_grad_ref[ND * NQ * ND * VS];
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 0,
                coordinate_grad_ref + 0 * NQ * ND * VS);
        tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_data, 1,
                coordinate_grad_ref + 1 * NQ * ND * VS);

        s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_jacobian_determinant0);

        for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
            element_matrix[entry] = s_t(0);
        }

        for (int trial_component = 0; trial_component < NC; ++trial_component) {
            for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
                for (int stream = 0; stream < NS * NC; ++stream) {
                    block_h_data[stream][0] = s_t(0);
                    block_out_data[stream][0] = s_t(0);
                }
                block_h_data[trial_shape * NC + trial_component][0] = s_t(1);
                neohookean_ogden_d2_tensor_product_apply_block<s_t, NQ, NS, VS>(1, 1, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_determinant0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, lmbda, mu, block_u_streams, block_h_streams, block_out_streams);
                const int col = trial_component * NS + trial_shape;
                for (int test_component = 0; test_component < NC; ++test_component) {
                    for (int test_shape = 0; test_shape < NS; ++test_shape) {
                        const int row = test_component * NS + test_shape;
                        element_matrix[row * NDOFS + col] = block_out_data[test_shape * NC + test_component][0];
                    }
                }
            }
        }

        if constexpr (FORMAT == 1) {
            neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
        } else {
            unsupported_matrix_format |= 1;
        }
    }

    return unsupported_matrix_format ? SFEM_FAILURE : SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_quad4_hessian_bsr_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl<double, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int neohookean_ogden_proteus_quad4_hessian_bsr_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::neohookean_ogden_proteus_quad4_hessian_isoparametric_mesh_soa_assemble_impl<float, geom_t, 1>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, rowptr, colidx, values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
