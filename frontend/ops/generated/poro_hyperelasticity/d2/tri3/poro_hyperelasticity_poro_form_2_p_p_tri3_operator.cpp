#include <type_traits>
#include "../poro_hyperelasticity_poro_form_2_p_p_d2_simplex_local.hpp"
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
    static const scalar_t *q_weight() {
        static const scalar_t data[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};
        return data;
    }
    static const scalar_t *tri3_shape() {
        static const scalar_t data[18] = {scalar_t(0.10810301816807022), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851)};
        return data;
    }
    static const scalar_t *tri3_grad_ref_x() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *tri3_grad_ref_y() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1)};
        return data;
    }
};

template <typename scalar_t>
struct poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data {
    static const scalar_t *q_weight() {
        static const scalar_t data[6] = {scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.11169079483900569), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998), scalar_t(0.054975871827660998)};
        return data;
    }
    static const scalar_t *tri3_shape() {
        static const scalar_t data[18] = {scalar_t(0.10810301816807022), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.44594849091596489), scalar_t(0.10810301816807021), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.091576213509770743), scalar_t(0.81684757298045851)};
        return data;
    }
    static const scalar_t *tri3_grad_ref_x() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *tri3_grad_ref_y() {
        static const scalar_t data[18] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(1)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa",
    "TRI3",
    2,
    6,
    3,
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
    5,
    54,
    6,
    0,
    0,
    0,
    3,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_residual_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_residual_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_residual_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa",
    "TRI3",
    2,
    6,
    3,
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
    5,
    54,
    6,
    3,
    0,
    3,
    3,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_tri3_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 3;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_tri3_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 3;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 3;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_affine_reference_data<scalar_t>::tri3_shape()};
    const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_affine_reference_data<scalar_t>::tri3_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_affine_reference_data<scalar_t>::tri3_grad_ref_y()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_direction[0][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[1][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[2][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 2] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
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
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());
        const scalar_t *const block_adjugate[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[2]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2]};

        poro_hyperelasticity_poro_form_2_p_p_d2_simplex_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_affine_reference_data<scalar_t>::q_weight(), block_direction_streams, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_mixed_impl(
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
    static constexpr int DIM = 2;
    static constexpr int N_QP = 6;
    static constexpr int CELL_N_SHAPE = 3;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 3;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::tri3_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::tri3_grad_ref_y();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_coordinates[0][lane] = points[0][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[1][lane] = points[1][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[2][lane] = points[0][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[3][lane] = points[1][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[4][lane] = points[0][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[5][lane] = points[1][ev[lane * CELL_N_SHAPE + 2]];
            block_direction[0][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[1][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[2][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 2] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::tri3_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::tri3_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::tri3_grad_ref_y()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3]};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[2]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2]};

        poro_hyperelasticity_poro_form_2_p_p_d2_simplex_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_isoparametric_reference_data<scalar_t>::q_weight(), block_direction_streams, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_soa(
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
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}

extern "C" int poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_soa_float(
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
    return sfem::codegen::poro_hyperelasticity_poro_form_2_p_p_tri3_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, dt, hydraulic_conductivity, storage, direction_stride, p_direction_data, out_stride, p_out);
}
