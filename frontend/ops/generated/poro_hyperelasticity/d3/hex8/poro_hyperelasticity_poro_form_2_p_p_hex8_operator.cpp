#include <type_traits>
#include "../poro_hyperelasticity_poro_form_2_p_p_d3_tensor_product_local.hpp"
#include "../../kernel_math.hpp"
#include "../../geometry_kernels.hpp"
#include "../../kernel_diagnostics.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif
#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef double geom_t;
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
struct poro_hyperelasticity_poro_form_2_p_p_affine_reference_data {
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
    static const scalar_t *hex8_shape_1d() {
        static const scalar_t data[8] = {scalar_t(0.93056815579702623), scalar_t(0.069431844202973714), scalar_t(0.66999052179242813), scalar_t(0.33000947820757187), scalar_t(0.33000947820757187), scalar_t(0.66999052179242813), scalar_t(0.069431844202973769), scalar_t(0.93056815579702623)};
        return data;
    }
    static const scalar_t *hex8_grad_1d() {
        static const scalar_t data[8] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
        return data;
    }
};

template <typename scalar_t>
struct poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data {
    static const scalar_t *q_weight_1d() {
        static const scalar_t data[4] = {scalar_t(0.17392742256872692), scalar_t(0.3260725774312731), scalar_t(0.3260725774312731), scalar_t(0.17392742256872692)};
        return data;
    }
    static const scalar_t *hex8_shape_1d() {
        static const scalar_t data[8] = {scalar_t(0.93056815579702623), scalar_t(0.069431844202973714), scalar_t(0.66999052179242813), scalar_t(0.33000947820757187), scalar_t(0.33000947820757187), scalar_t(0.66999052179242813), scalar_t(0.069431844202973769), scalar_t(0.93056815579702623)};
        return data;
    }
    static const scalar_t *hex8_grad_1d() {
        static const scalar_t data[8] = {scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1), scalar_t(-1), scalar_t(1)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    10,
    16,
    4,
    0,
    0,
    0,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_residual_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_residual_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_residual_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa",
    "HEX8",
    3,
    64,
    8,
    16,
    4,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    10,
    16,
    4,
    3,
    0,
    8,
    8,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_hex8_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 8;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_hex8_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 8;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_mixed_impl(
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
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 8;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_affine_reference_data<scalar_t>::hex8_shape_1d()};
    const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_affine_reference_data<scalar_t>::hex8_grad_1d()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        for (int local_shape = 0; local_shape < 8; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 8; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());
        const scalar_t *const block_adjugate[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[3], block_direction[2], block_direction[4], block_direction[5], block_direction[7], block_direction[6]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[3], block_output[2], block_output[4], block_output[5], block_output[7], block_output[6]};

        poro_hyperelasticity_poro_form_2_p_p_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_affine_reference_data<scalar_t>::q_weight_1d(), block_direction_streams, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            scalar_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 8; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 0 + local_shape;
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

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa(
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
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 64;
    static constexpr int CELL_N_SHAPE = 8;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_shape_1d = sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::hex8_shape_1d();
    const scalar_t *const isoparametric_grad_1d = sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::hex8_grad_1d();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

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

        for (int local_shape = 0; local_shape < 8; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 8; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *const block_coordinate_streams[DIM * N_SHAPE] = {block_coordinates[0], block_coordinates[1], block_coordinates[2], block_coordinates[3], block_coordinates[4], block_coordinates[5], block_coordinates[9], block_coordinates[10], block_coordinates[11], block_coordinates[6], block_coordinates[7], block_coordinates[8], block_coordinates[12], block_coordinates[13], block_coordinates[14], block_coordinates[15], block_coordinates[16], block_coordinates[17], block_coordinates[21], block_coordinates[22], block_coordinates[23], block_coordinates[18], block_coordinates[19], block_coordinates[20]};
        scalar_t coordinate_grad_ref[DIM * N_QP * DIM * VECTOR_SIZE];
        scalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];
        tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, DIM>(
                nelems, isoparametric_shape_1d, isoparametric_grad_1d, block_coordinate_streams,
                coordinate_value, coordinate_grad_ref);

        scalar_t *coordinate_grad_ref_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        geometry_jacobian_adjugate_and_determinant<scalar_t, DIM, N_QP, VECTOR_SIZE>(
                nelems, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, block_determinant);

        const scalar_t *const field_shape_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::hex8_shape_1d()};
        const scalar_t *const field_grad_1d[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::hex8_grad_1d()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[3], block_direction[2], block_direction[4], block_direction[5], block_direction[7], block_direction[6]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[3], block_output[2], block_output[4], block_output[5], block_output[7], block_output[6]};

        poro_hyperelasticity_poro_form_2_p_p_d3_tensor_product_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape_1d, field_grad_1d, sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::q_weight_1d(), block_direction_streams, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            scalar_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 8; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 0 + local_shape;
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

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_hex8_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}
