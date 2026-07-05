#include <type_traits>
#include "../laplace_d3_tensor_product_local.hpp"
#include "../../geometry_kernels.hpp"
#include "../../kernel_diagnostics.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *affine_geometry_stream(
        const int,
        const jacobian_t *const SFEM_RESTRICT source,
        scalar_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *affine_geometry_stream(
        const int nelems,
        const jacobian_t *const SFEM_RESTRICT source,
        scalar_t *const SFEM_RESTRICT converted,
        std::false_type) {
    #pragma omp simd
    for (int lane = 0; lane < nelems; ++lane) {
        converted[lane] = scalar_t(source[lane]);
    }
    return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename scalar_t>
struct laplace_hex27_affine_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
        return data;
    }
};

template <typename scalar_t>
struct laplace_hex27_isoparametric_reference_data {
    static const scalar_t *shape_1d() {
        static const scalar_t data[9] = {scalar_t(0.68729833462074175), scalar_t(0.39999999999999997), scalar_t(-0.087298334620741685), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-0.087298334620741658), scalar_t(0.39999999999999991), scalar_t(0.68729833462074175)};
        return data;
    }
    static const scalar_t *grad_1d() {
        static const scalar_t data[9] = {scalar_t(-2.5491933384829668), scalar_t(3.0983866769659336), scalar_t(-0.54919333848296681), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0.54919333848296681), scalar_t(-3.0983866769659336), scalar_t(2.5491933384829668)};
        return data;
    }
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[3] = {scalar_t(0.27777777777777779), scalar_t(0.44444444444444442), scalar_t(0.27777777777777779)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_hex27_residual_element_soa_diagnostics_data = {
    "laplace_hex27_residual_element_soa",
    "HEX27",
    3,
    27,
    27,
    16,
    3,
    2,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    1,
    6,
    0,
    0,
    0,
    7,
    10,
    18,
    3,
    1,
    27,
    0,
    27,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data;
}

extern "C" double laplace_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_hex27_residual_element_soa",
            &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_hex27_residual_element_soa_float",
            &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_hex27_residual_affine_mesh_soa",
            &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_hex27_residual_affine_mesh_soa_float",
            &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_hex27_residual_isoparametric_mesh_soa",
            &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_hex27_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_hex27_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_hex27_jacobian_u_u_diagnostics_data = {
    "laplace_hex27_jacobian_u_u",
    "HEX27",
    3,
    27,
    27,
    16,
    3,
    2,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    1,
    6,
    0,
    0,
    0,
    7,
    10,
    18,
    3,
    1,
    0,
    27,
    27,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_jacobian_u_u_diagnostics(void) {
    return &sfem::codegen::laplace_hex27_jacobian_u_u_diagnostics_data;
}

extern "C" double laplace_hex27_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_hex27_jacobian_u_u_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_hex27_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_hex27_jacobian_u_u",
            &sfem::codegen::laplace_hex27_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_hex27_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_hex27_jacobian_u_u_float",
            &sfem::codegen::laplace_hex27_jacobian_u_u_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_hex27_jacobian_action_element_soa_diagnostics_data = {
    "laplace_hex27_jacobian_action_element_soa",
    "HEX27",
    3,
    27,
    27,
    16,
    3,
    2,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    7,
    1,
    6,
    0,
    0,
    0,
    7,
    10,
    18,
    3,
    1,
    0,
    27,
    27,
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

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double laplace_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void laplace_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_hex27_jacobian_action_element_soa",
            &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "laplace_hex27_jacobian_action_element_soa_float",
            &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_hex27_jacobian_action_affine_mesh_soa",
            &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "laplace_hex27_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void laplace_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_hex27_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void laplace_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "laplace_hex27_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::laplace_hex27_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" int laplace_hex27_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[27],
        const double kappa,
        double *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<double, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_hex27_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<double>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_hex27_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[27],
        const float kappa,
        float *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_residual_block<float, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_hex27_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<float>::q_weight_1d(), current, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_hex27_residual_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t kappa,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_hex27_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        const scalar_t *const current_components[N_FIELDS] = {u};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                }
            }
        }

        for (int stream = 0; stream < 27; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_current_streams[27] = {block_current[0], block_current[8], block_current[1], block_current[11], block_current[24], block_current[9], block_current[3], block_current[10], block_current[2], block_current[16], block_current[20], block_current[17], block_current[23], block_current[26], block_current[21], block_current[19], block_current[22], block_current[18], block_current[4], block_current[12], block_current[5], block_current[15], block_current[25], block_current[13], block_current[7], block_current[14], block_current[6]};
        scalar_t *const block_output_streams[27] = {block_output[0], block_output[8], block_output[1], block_output[11], block_output[24], block_output[9], block_output[3], block_output[10], block_output[2], block_output[16], block_output[20], block_output[17], block_output[23], block_output[26], block_output[21], block_output[19], block_output[22], block_output[18], block_output[4], block_output[12], block_output[5], block_output[15], block_output[25], block_output[13], block_output[7], block_output[14], block_output[6]};
        const jacobian_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[10][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        laplace_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_current_streams, kappa, block_output_streams);

        scalar_t *const output_components[N_FIELDS] = {u_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_hex27_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_hex27_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, current_stride, u, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_hex27_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const scalar_t *const current_components[N_FIELDS] = {u};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                }
            }
        }

        for (int stream = 0; stream < 27; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[18], block_coordinates[19], block_coordinates[20]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_current_streams[27] = {block_current[0], block_current[8], block_current[1], block_current[11], block_current[24], block_current[9], block_current[3], block_current[10], block_current[2], block_current[16], block_current[20], block_current[17], block_current[23], block_current[26], block_current[21], block_current[19], block_current[22], block_current[18], block_current[4], block_current[12], block_current[5], block_current[15], block_current[25], block_current[13], block_current[7], block_current[14], block_current[6]};
        scalar_t *const block_output_streams[27] = {block_output[0], block_output[8], block_output[1], block_output[11], block_output[24], block_output[9], block_output[3], block_output[10], block_output[2], block_output[16], block_output[20], block_output[17], block_output[23], block_output[26], block_output[21], block_output[19], block_output[22], block_output[18], block_output[4], block_output[12], block_output[5], block_output[15], block_output[25], block_output[13], block_output[7], block_output[14], block_output[6]};
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_tensor_product_residual_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_current_streams, kappa, block_output_streams);

        scalar_t *const output_components[N_FIELDS] = {u_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_hex27_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_hex27_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, current_stride, u, out_stride, u_out);
}

extern "C" int laplace_hex27_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
) {
    return laplace_hex27_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_hex27_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
) {
    return laplace_hex27_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, current + 0, 1, output + 0);
}

extern "C" int laplace_hex27_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT direction[27],
        const double kappa,
        double *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<double, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_hex27_isoparametric_reference_data<double>::shape_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<double>::grad_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<double>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

extern "C" int laplace_hex27_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT direction[27],
        const float kappa,
        float *const SFEM_RESTRICT output[27]
) {
    sfem::codegen::laplace_d3_tensor_product_jacobian_action_block<float, 27, 27, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::laplace_hex27_isoparametric_reference_data<float>::shape_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<float>::grad_1d(), sfem::codegen::laplace_hex27_isoparametric_reference_data<float>::q_weight_1d(), direction, kappa, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int laplace_hex27_jacobian_action_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_shape_1d = sfem::codegen::laplace_hex27_affine_reference_data<scalar_t>::shape_1d();
    const scalar_t *const affine_grad_1d = sfem::codegen::laplace_hex27_affine_reference_data<scalar_t>::grad_1d();
    const scalar_t *const affine_q_weight_1d = sfem::codegen::laplace_hex27_affine_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        const scalar_t *const direction_components[N_FIELDS] = {u_direction};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 27; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_direction_streams[27] = {block_direction[0], block_direction[8], block_direction[1], block_direction[11], block_direction[24], block_direction[9], block_direction[3], block_direction[10], block_direction[2], block_direction[16], block_direction[20], block_direction[17], block_direction[23], block_direction[26], block_direction[21], block_direction[19], block_direction[22], block_direction[18], block_direction[4], block_direction[12], block_direction[5], block_direction[15], block_direction[25], block_direction[13], block_direction[7], block_direction[14], block_direction[6]};
        scalar_t *const block_output_streams[27] = {block_output[0], block_output[8], block_output[1], block_output[11], block_output[24], block_output[9], block_output[3], block_output[10], block_output[2], block_output[16], block_output[20], block_output[17], block_output[23], block_output[26], block_output[21], block_output[19], block_output[22], block_output[18], block_output[4], block_output[12], block_output[5], block_output[15], block_output[25], block_output[13], block_output[7], block_output[14], block_output[6]};
        const jacobian_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[10][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        laplace_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, block_direction_streams, kappa, block_output_streams);

        scalar_t *const output_components[N_FIELDS] = {u_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_hex27_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, kappa, direction_stride, u_direction, out_stride, u_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int laplace_hex27_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t kappa,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 27;
    static constexpr int N_SHAPE = 27;
    static constexpr int N_FIELDS = 1;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::laplace_hex27_isoparametric_reference_data<scalar_t>::shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::laplace_hex27_isoparametric_reference_data<scalar_t>::grad_1d();
    const scalar_t *const isoparametric_q_weight_1d = sfem::codegen::laplace_hex27_isoparametric_reference_data<scalar_t>::q_weight_1d();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[3 * N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[9][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];
        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const scalar_t *const direction_components[N_FIELDS] = {u_direction};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 27; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[24], block_coordinates[25], block_coordinates[26], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[33], block_coordinates[34], block_coordinates[35], block_coordinates[72], block_coordinates[73], block_coordinates[74], block_coordinates[27], block_coordinates[28], block_coordinates[29], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[30], block_coordinates[31], block_coordinates[32], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[48], block_coordinates[49], block_coordinates[50], block_coordinates[60], block_coordinates[61], block_coordinates[62], block_coordinates[51], block_coordinates[52], block_coordinates[53], block_coordinates[69], block_coordinates[70], block_coordinates[71], block_coordinates[78], block_coordinates[79], block_coordinates[80], block_coordinates[63], block_coordinates[64], block_coordinates[65], block_coordinates[57], block_coordinates[58], block_coordinates[59], block_coordinates[66], block_coordinates[67], block_coordinates[68], block_coordinates[54], block_coordinates[55], block_coordinates[56], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[36], block_coordinates[37], block_coordinates[38], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[45], block_coordinates[46], block_coordinates[47], block_coordinates[75], block_coordinates[76], block_coordinates[77], block_coordinates[39], block_coordinates[40], block_coordinates[41], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[42], block_coordinates[43], block_coordinates[44], block_coordinates[18], block_coordinates[19], block_coordinates[20]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const block_direction_streams[27] = {block_direction[0], block_direction[8], block_direction[1], block_direction[11], block_direction[24], block_direction[9], block_direction[3], block_direction[10], block_direction[2], block_direction[16], block_direction[20], block_direction[17], block_direction[23], block_direction[26], block_direction[21], block_direction[19], block_direction[22], block_direction[18], block_direction[4], block_direction[12], block_direction[5], block_direction[15], block_direction[25], block_direction[13], block_direction[7], block_direction[14], block_direction[6]};
        scalar_t *const block_output_streams[27] = {block_output[0], block_output[8], block_output[1], block_output[11], block_output[24], block_output[9], block_output[3], block_output[10], block_output[2], block_output[16], block_output[20], block_output[17], block_output[23], block_output[26], block_output[21], block_output[19], block_output[22], block_output[18], block_output[4], block_output[12], block_output[5], block_output[15], block_output[25], block_output[13], block_output[7], block_output[14], block_output[6]};
        const scalar_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        laplace_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, block_direction_streams, kappa, block_output_streams);

        scalar_t *const output_components[N_FIELDS] = {u_out};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < N_FIELDS; ++field) {
                const int stream = shape * N_FIELDS + field;
                scalar_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
) {
    return sfem::codegen::laplace_hex27_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, kappa, direction_stride, u_direction, out_stride, u_out);
}

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return laplace_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}

extern "C" int laplace_hex27_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return laplace_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], 1, direction + 0, 1, output + 0);
}
